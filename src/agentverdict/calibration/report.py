"""Judge-vs-human calibration: how far can this judge actually be trusted?

An LLM judge is a measurement instrument, and an uncalibrated instrument is not
evidence. This module answers one question over rows already in the database:
when the judge scored a trajectory the humans also scored, how often — and how
badly — did it disagree with them?

The report has four parts, each answering a different follow-up:

* **agreement** — the confusion matrix, raw agreement, Cohen's kappa with a
  bootstrap interval, and an ordinal weighted kappa. This is the headline.
* **judge vs. each annotator** — the judge scored against one human at a time, on
  exactly the trajectories both of them rated. This exists because the headline is
  computed against the human *majority*, and a majority does not exist when the
  annotators tie, so those trajectories are silently dropped from the comparison.
  With two annotators *every* disagreement is a tie: the judge ends up scored only
  on the items the humans found easy, while the human figures below are computed
  over everything the judge touched. Two statistics measured on different samples
  cannot be read against each other, and this project's own data shows how far
  that misleads — a headline kappa of 0.786 against a human ceiling of 0.523 read
  as a judge beating its annotators, when on the identical 13 items the judge
  scored 0.337 against one of them and 0.764 against the other, either side of a
  human-vs-human 0.523. The per-rater rows hold every rater to the same items, so
  the numbers printed next to each other actually describe the same thing.
* **human ceiling** — chance-corrected agreement between each pair of human
  annotators on the trajectories they both labeled. This is the part that makes
  the headline readable. Verdicts on real agent trajectories are genuinely
  contestable, so two careful humans routinely land around kappa 0.7; a judge at
  0.65 measured against that ceiling is close to the limit of what the label set
  can even resolve, while the same 0.65 read against a fictional 1.0 looks like a
  failure. Without the ceiling the headline kappa is uninterpretable.
* **disagreements** — the specific trajectories where judge and humans parted
  ways, worst first by ordinal distance (a pass/fail flip before a
  pass/borderline one) and carrying the judge's own rationale, so the failure
  mode can be read rather than guessed at. This is the actionable output.

A report can also be scoped to a named set of annotators. Once a rubric ambiguity
is written down, a second annotation round measures a *different* thing than the
first, so overwriting the original labels would destroy the evidence that the
ambiguity mattered. Recording the re-labels under distinct annotator names and
scoping the report to them keeps each round independently readable, with round one
preserved as the honest pre-clarification record.

Everything here is pure analysis over stored verdicts and labels: no model is
called, no API key is needed, and running it costs nothing, which is what makes
it safe to put in a CI gate.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Container, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from agentverdict.calibration.stats import (
    VERDICT_ORDER,
    AgreementStats,
    agreement_stats,
    bootstrap_kappa,
    cohens_kappa,
    observed_agreement,
)
from agentverdict.models import HumanLabel, Judge, JudgeVerdict, Task, Trajectory, utcnow
from agentverdict.schemas import (
    AgreementRead,
    AnnotatorPairAgreement,
    CalibrationReport,
    ClassMetricsRead,
    DisagreementRow,
    HeldOutAnnotatorAgreement,
    JudgeAnnotatorAgreement,
)

#: Position of each verdict on the ordinal scale, for distance arithmetic.
_VERDICT_INDEX: dict[str, int] = {label: index for index, label in enumerate(VERDICT_ORDER)}


@dataclass(frozen=True)
class _JudgeCall:
    """The judge's surviving decision for one trajectory."""

    verdict: str
    rationale: str | None
    task_key: str


@dataclass(frozen=True)
class _ComparedItem:
    """A trajectory carrying both a human majority and a judge verdict."""

    trajectory_id: str
    task_key: str
    human_verdict: str
    judge_verdict: str
    annotators: tuple[str, ...]
    judge_rationale: str | None

    @property
    def distance(self) -> int:
        """Steps apart on ``fail < borderline < pass``; 2 is a pass/fail flip."""
        return abs(_VERDICT_INDEX[self.human_verdict] - _VERDICT_INDEX[self.judge_verdict])


def human_majority(labels: Sequence[HumanLabel]) -> str | None:
    """The majority human verdict for one trajectory.

    Returns ``None`` when nobody labeled the trajectory and when the top two
    verdicts are level: a split panel has no majority to hold the judge to, and
    inventing one (by annotator order, say) would quietly bias the comparison.
    Such trajectories are excluded from the statistics and counted as ties.
    """
    ranked = Counter(label.verdict for label in labels).most_common()
    if not ranked:
        return None
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0]


def _to_agreement_read(stats: AgreementStats) -> AgreementRead:
    """Flatten the stats dataclass into the wire/template schema."""
    low, high = stats.kappa_ci if stats.kappa_ci is not None else (None, None)
    return AgreementRead(
        n=stats.n,
        labels=list(stats.labels),
        matrix=stats.matrix,
        observed_agreement=stats.observed_agreement,
        cohens_kappa=stats.cohens_kappa,
        weighted_kappa=stats.weighted_kappa,
        kappa_ci_low=low,
        kappa_ci_high=high,
        interpretation=stats.interpretation,
        bootstrap_usable=stats.bootstrap_usable,
        bootstrap_iterations=stats.bootstrap_iterations,
        per_class=[
            ClassMetricsRead(
                label=metrics.label,
                support=metrics.support,
                predicted=metrics.predicted,
                precision=metrics.precision,
                recall=metrics.recall,
                f1=metrics.f1,
            )
            for metrics in stats.per_class
        ],
    )


def _latest_judge_calls(
    session: Session,
    judge: Judge,
    *,
    eval_run_id: str | None,
    task_key: str | None,
) -> tuple[dict[str, _JudgeCall], int]:
    """This judge's most recent decision per trajectory, plus its error count.

    A judge may score the same trajectory in several runs; the newest decision is
    the one that reflects the current prompt, so it is the one that gets compared.
    Ordering by ``(created_at, id)`` keeps the choice deterministic when a batch
    lands on the same timestamp. Rows with a null verdict are recorded API
    failures rather than decisions: they never win the comparison, and they are
    counted so a low sample size can be traced back to a flaky run.
    """
    stmt = (
        select(
            JudgeVerdict.trajectory_id,
            JudgeVerdict.verdict,
            JudgeVerdict.rationale,
            Task.key,
        )
        .join(Trajectory, JudgeVerdict.trajectory_id == Trajectory.id)
        .join(Task, Trajectory.task_id == Task.id)
        .where(JudgeVerdict.judge_id == judge.id)
        .order_by(JudgeVerdict.created_at, JudgeVerdict.id)
    )
    if eval_run_id is not None:
        stmt = stmt.where(JudgeVerdict.eval_run_id == eval_run_id)
    if task_key is not None:
        stmt = stmt.where(Task.key == task_key)

    calls: dict[str, _JudgeCall] = {}
    errors = 0
    for trajectory_id, verdict, rationale, key in session.execute(stmt):
        if verdict is None:
            errors += 1
            continue
        # Rows arrive oldest first, so the last write per trajectory wins.
        calls[trajectory_id] = _JudgeCall(verdict=verdict, rationale=rationale, task_key=key)
    return calls, errors


def _labels_by_trajectory(
    session: Session, *, task_key: str | None, annotators: Sequence[str] | None = None
) -> dict[str, list[HumanLabel]]:
    """Every human label in scope, grouped by trajectory in one query.

    ``annotators``, when given, restricts the whole report to one named panel — the
    filter is applied here, at the single point every downstream statistic reads
    from, so the majority, the ties, the ceiling, the baseline and the per-rater
    rows can never disagree about who was in scope. A name nobody labeled under
    simply contributes nothing; it is not an error, because asking for a round that
    has not been annotated yet should report an empty comparison rather than raise.
    Trajectories left with no in-scope label drop out of the mapping entirely, which
    is what keeps ``labeled_trajectories`` honest under scoping.
    """
    stmt = (
        select(HumanLabel)
        .join(Trajectory, HumanLabel.trajectory_id == Trajectory.id)
        .join(Task, Trajectory.task_id == Task.id)
        .order_by(HumanLabel.trajectory_id, HumanLabel.annotator)
    )
    if task_key is not None:
        stmt = stmt.where(Task.key == task_key)
    if annotators is not None:
        stmt = stmt.where(HumanLabel.annotator.in_(annotators))

    grouped: dict[str, list[HumanLabel]] = defaultdict(list)
    for label in session.scalars(stmt):
        grouped[label.trajectory_id].append(label)
    return dict(grouped)


def _judge_vs_annotator(
    labels_by_trajectory: Mapping[str, list[HumanLabel]],
    judge_calls: Mapping[str, _JudgeCall],
    *,
    iterations: int,
    seed: int,
) -> list[JudgeAnnotatorAgreement]:
    """Score the judge against each annotator individually, on their shared items.

    Deliberately *not* against the majority. A majority is undefined when the
    annotators tie, so the headline comparison quietly drops exactly those
    trajectories — and with two annotators a tie is what every disagreement looks
    like. The headline is therefore computed on the subset the humans agreed
    about, while the ceiling and baseline below it are computed on everything
    judged: the judge is being graded on an easier sample than the humans, which
    is how a merely-average judge comes to look superhuman.

    Pairing one annotator's verdict directly with the judge's on the trajectories
    both of them rated removes that asymmetry. The judge no longer benefits from
    the hard cases being dropped, and its number lands on the same items — and so
    the same difficulty — as the human-vs-human number it is read against.

    Reads only the two in-memory mappings the caller already fetched, so it adds
    no queries. Annotators sharing no trajectory with the judge are skipped
    (nothing to compare); widest overlap first, as the sturdiest estimate.
    """
    pairs_by_annotator: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for trajectory_id in sorted(labels_by_trajectory):
        call = judge_calls.get(trajectory_id)
        if call is None or call.verdict not in _VERDICT_INDEX:
            continue
        for label in labels_by_trajectory[trajectory_id]:
            if label.verdict in _VERDICT_INDEX:
                # Human first: the annotator is the reference rater, as everywhere else.
                pairs_by_annotator[label.annotator].append((label.verdict, call.verdict))

    rows = [
        JudgeAnnotatorAgreement(
            annotator=annotator,
            **_pair_fields(pairs, iterations=iterations, seed=seed),
        )
        for annotator, pairs in sorted(pairs_by_annotator.items())
        if pairs
    ]
    rows.sort(key=lambda row: (-row.n, row.annotator))
    return rows


def _pair_fields(
    pairs: Sequence[tuple[str, str]], *, iterations: int, seed: int
) -> dict[str, Any]:
    """The statistics every secondary agreement row carries, uncertainty included.

    These rows are frequently built from a handful of trajectories, and the report
    directs readers to prefer them over the headline — so they get the same
    bootstrap treatment. Without it a row reading "n=3 kappa 1.000" would be
    rendered as "almost perfect" with nothing to signal how little is behind it.
    """
    result = bootstrap_kappa(pairs, VERDICT_ORDER, iterations=iterations, seed=seed)
    low, high = result.interval if result.interval is not None else (None, None)
    return {
        "n": len(pairs),
        "observed_agreement": observed_agreement(pairs, VERDICT_ORDER),
        "cohens_kappa": cohens_kappa(pairs, VERDICT_ORDER),
        "kappa_ci_low": low,
        "kappa_ci_high": high,
        "bootstrap_usable": result.usable,
        "bootstrap_iterations": result.iterations,
    }


def _human_ceiling(
    labels_by_trajectory: Mapping[str, list[HumanLabel]],
    scope: Container[str],
    *,
    iterations: int,
    seed: int,
) -> list[AnnotatorPairAgreement]:
    """Pairwise inter-annotator agreement, for spotting a divergent grader.

    Every unordered pair of annotators is scored on exactly the trajectories both
    of them labeled; pairs with no overlap are skipped because there is nothing
    to compare. Widest overlap first, since that is the most trustworthy estimate.

    Restricted to ``scope`` — the trajectories the judge produced a verdict for —
    for the same reason ``_human_baseline`` is: the report states that the human
    rows and the per-rater judge rows describe one item set, and that claim is
    false the moment the judge has not scored every labeled trajectory.
    """
    by_annotator: dict[str, dict[str, str]] = defaultdict(dict)
    for trajectory_id, labels in labels_by_trajectory.items():
        if trajectory_id not in scope:
            continue
        for label in labels:
            if label.verdict in _VERDICT_INDEX:
                by_annotator[label.annotator][trajectory_id] = label.verdict

    rows: list[AnnotatorPairAgreement] = []
    for annotator_a, annotator_b in combinations(sorted(by_annotator), 2):
        first, second = by_annotator[annotator_a], by_annotator[annotator_b]
        shared = sorted(first.keys() & second.keys())
        if not shared:
            continue
        pairs = [(first[trajectory_id], second[trajectory_id]) for trajectory_id in shared]
        rows.append(
            AnnotatorPairAgreement(
                annotator_a=annotator_a,
                annotator_b=annotator_b,
                **_pair_fields(pairs, iterations=iterations, seed=seed),
            )
        )
    rows.sort(key=lambda row: (-row.n, row.annotator_a, row.annotator_b))
    return rows


def _human_baseline(
    labels_by_trajectory: Mapping[str, list[HumanLabel]],
    scope: Container[str],
    *,
    iterations: int,
    seed: int,
) -> list[HeldOutAnnotatorAgreement]:
    """Score each annotator the way the judge is scored: against a majority of others.

    Pairwise annotator agreement is not the bar the judge has to clear. The judge
    is compared against the aggregated human majority, and aggregating cancels out
    individual noise — so a judge exactly as accurate as an average annotator will
    outscore raw annotator-vs-annotator agreement. Holding one annotator out and
    scoring them against the majority of the rest puts human and judge on the same
    footing, which is the only comparison worth printing next to each other.

    Restricted to ``scope`` — the trajectories the judge produced a verdict for —
    so both numbers describe the same population. Deliberately *not* restricted to
    the compared set: with exactly two annotators every disagreement becomes a tie
    and is dropped from ``compared``, so scoring humans there would measure them
    only on the items they already agreed about and always report a perfect 1.000.
    Needs at least two annotators.
    """
    by_annotator: dict[str, dict[str, str]] = defaultdict(dict)
    for trajectory_id, labels in labels_by_trajectory.items():
        if trajectory_id not in scope:
            continue
        for label in labels:
            if label.verdict in _VERDICT_INDEX:
                by_annotator[label.annotator][trajectory_id] = label.verdict

    rows: list[HeldOutAnnotatorAgreement] = []
    for annotator in sorted(by_annotator):
        pairs: list[tuple[str, str]] = []
        for trajectory_id, verdict in sorted(by_annotator[annotator].items()):
            others = [
                label
                for label in labels_by_trajectory[trajectory_id]
                if label.annotator != annotator and label.verdict in _VERDICT_INDEX
            ]
            reference = human_majority(others)
            if reference is not None:
                pairs.append((reference, verdict))
        if not pairs:
            continue
        rows.append(
            HeldOutAnnotatorAgreement(
                annotator=annotator,
                **_pair_fields(pairs, iterations=iterations, seed=seed),
            )
        )
    rows.sort(key=lambda row: (-row.n, row.annotator))
    return rows


def build_calibration_report(
    session: Session,
    judge: Judge,
    *,
    eval_run_id: str | None = None,
    task_key: str | None = None,
    annotators: Sequence[str] | None = None,
    max_disagreements: int = 25,
    bootstrap_iterations: int = 1000,
    seed: int = 0,
) -> CalibrationReport:
    """Score ``judge`` against the human golden set and return the full report.

    Args:
        session: Open session; nothing is written, so a read-only one is fine.
        judge: The judge whose verdicts are being calibrated.
        eval_run_id: Restrict the judge's verdicts to a single eval run. Human
            labels are unaffected — they belong to the dataset, not to a run.
        task_key: Restrict the whole report to one task's trajectories.
        annotators: Consider only labels from these annotators, everywhere — the
            majority, the tie count, the ceiling, the baseline, the per-rater
            rows, and ``labeled_trajectories``. This is how a second annotation
            round, recorded under its own names, gets scored on its own. Names
            nobody labeled under yield an empty comparison rather than an error.
            ``None`` — or an empty sequence, which no caller can have meant as
            "score nobody" — means every annotator.
        max_disagreements: Cap on the drill-down list, worst cases kept.
        bootstrap_iterations: Resamples behind the kappa confidence interval.
        seed: Bootstrap seed, so the same data always yields the same interval.
    """
    scoped_annotators = sorted(set(annotators)) if annotators else []
    judge_calls, judge_errors = _latest_judge_calls(
        session, judge, eval_run_id=eval_run_id, task_key=task_key
    )
    labels_by_trajectory = _labels_by_trajectory(
        session, task_key=task_key, annotators=scoped_annotators or None
    )

    ties_excluded = 0
    majorities: dict[str, str] = {}
    for trajectory_id, labels in labels_by_trajectory.items():
        majority = human_majority(labels)
        if majority is None:
            ties_excluded += 1
        else:
            majorities[trajectory_id] = majority

    compared_items: list[_ComparedItem] = []
    for trajectory_id, call in judge_calls.items():
        human_verdict = majorities.get(trajectory_id)
        # An unknown verdict string would be dropped by the statistics anyway;
        # skipping it here keeps `compared` equal to the n kappa was computed on.
        if human_verdict is None or human_verdict not in _VERDICT_INDEX:
            continue
        if call.verdict not in _VERDICT_INDEX:
            continue
        compared_items.append(
            _ComparedItem(
                trajectory_id=trajectory_id,
                task_key=call.task_key,
                human_verdict=human_verdict,
                judge_verdict=call.verdict,
                annotators=tuple(
                    sorted(label.annotator for label in labels_by_trajectory[trajectory_id])
                ),
                judge_rationale=call.rationale,
            )
        )
    # A stable order keeps the seeded bootstrap reproducible across runs.
    compared_items.sort(key=lambda item: (item.task_key, item.trajectory_id))

    # Reference first: the humans are the ground truth the judge is scored against.
    stats = agreement_stats(
        [(item.human_verdict, item.judge_verdict) for item in compared_items],
        VERDICT_ORDER,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )

    disagreements = [
        DisagreementRow(
            trajectory_id=item.trajectory_id,
            task_key=item.task_key,
            human_verdict=item.human_verdict,
            judge_verdict=item.judge_verdict,
            distance=item.distance,
            annotators=list(item.annotators),
            judge_rationale=item.judge_rationale,
        )
        for item in compared_items
        if item.human_verdict != item.judge_verdict
    ]
    disagreements.sort(key=lambda row: (-row.distance, row.task_key, row.trajectory_id))

    return CalibrationReport(
        judge_id=judge.id,
        judge_name=judge.name,
        judge_model=judge.model,
        eval_run_id=eval_run_id,
        task_key=task_key,
        annotators=scoped_annotators,
        # Who actually contributed a label, which is what decides whether the
        # "majority" this report speaks of is a panel or one person's opinion.
        annotators_present=sorted(
            {
                label.annotator
                for labels in labels_by_trajectory.values()
                for label in labels
                if label.verdict in _VERDICT_INDEX
            }
        ),
        generated_at=utcnow(),
        judged_trajectories=len(judge_calls),
        labeled_trajectories=len(labels_by_trajectory),
        compared=len(compared_items),
        judge_errors=judge_errors,
        ties_excluded=ties_excluded,
        agreement=_to_agreement_read(stats),
        judge_vs_annotator=_judge_vs_annotator(
            labels_by_trajectory, judge_calls, iterations=bootstrap_iterations, seed=seed
        ),
        human_ceiling=_human_ceiling(
            labels_by_trajectory,
            judge_calls.keys(),
            iterations=bootstrap_iterations,
            seed=seed,
        ),
        human_baseline=_human_baseline(
            labels_by_trajectory,
            judge_calls.keys(),
            iterations=bootstrap_iterations,
            seed=seed,
        ),
        disagreements=disagreements[: max(max_disagreements, 0)],
        disagreements_total=len(disagreements),
    )
