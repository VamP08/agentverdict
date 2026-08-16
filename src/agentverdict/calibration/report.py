"""Judge-vs-human calibration: how far can this judge actually be trusted?

An LLM judge is a measurement instrument, and an uncalibrated instrument is not
evidence. This module answers one question over rows already in the database:
when the judge scored a trajectory the humans also scored, how often — and how
badly — did it disagree with them?

The report has five parts, each answering a different follow-up:

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
* **per-criterion agreement** — the same statistics computed separately for each
  rubric criterion, over the trajectories where judge and humans both answered
  it, and with them ``rule_disagreements``: the trajectories where the two sides
  answered every criterion either of them reached identically and still returned
  different verdicts. A verdict token is a summary of a reading, and two raters
  can read a run the same way and summarise it differently — this section
  separates those cases from genuine misreadings, which is the difference
  between fixing the judge and fixing the rubric. A verdict split whose two
  sides were barely compared — some criterion answered on one side only, or
  none answered at all — goes to ``rule_undetermined`` instead: attributed to
  neither artifact, because the honest fix for an unmeasured split is more
  labeling.

Every report also names the rulebooks behind it: which rubric version each
in-scope label was made under, and which judge prompt graded each surviving
verdict. A rubric edit re-grades the corpus, so a kappa averaged across one
describes neither rulebook — the report counts both sides' versions and carries
a single warning sentence when the figures span more than one rulebook, rather
than leaving the split to be inferred from annotator names.

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
from collections.abc import Container, Iterable, Mapping, Sequence
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
from agentverdict.judging.prompts import PROMPT_VERSION
from agentverdict.models import (
    EvalRun,
    HumanLabel,
    Judge,
    JudgeVerdict,
    Rubric,
    Task,
    Trajectory,
    utcnow,
)
from agentverdict.rubric import (
    CRITERIA,
    CRITERIA_BY_KEY,
    RUBRIC_NAME,
    RUBRIC_VERSION,
    SCORE_LABELS,
    score_label,
)
from agentverdict.schemas import (
    AgreementRead,
    AnnotatorPairAgreement,
    CalibrationReport,
    ClassMetricsRead,
    CriterionAgreement,
    DisagreementRow,
    HeldOutAnnotatorAgreement,
    JudgeAnnotatorAgreement,
)

#: Position of each verdict on the ordinal scale, for distance arithmetic.
_VERDICT_INDEX: dict[str, int] = {label: index for index, label in enumerate(VERDICT_ORDER)}

#: The "yes" answer to a criterion, as an agreement label. Counting it on each side is
#: what distinguishes a criterion two raters genuinely agree on from one they both
#: answered the same way every time — where kappa is undefined and prints nothing.
_POSITIVE_ANSWER = score_label(1.0)

#: How a rulebook nothing recorded is named, spelled as ``compare`` and the bias report
#: already spell it. Round-one labels point at no rubric row because none existed when
#: they were made, and runs from before ``eval_runs.judge_prompt_version`` was added
#: carry no fingerprint. Both are rulebooks in their own right rather than missing
#: fields, neither can be recovered after the fact, and the name cannot collide with a
#: real version: rubrics render as ``name vN`` and prompt versions are hex digests.
_UNRECORDED = "unrecorded"

#: The rulebook this process grades by, as label provenance spells it. Its two halves
#: cannot drift apart — ``judging/prompts.py`` renders :data:`CRITERIA` into the text
#: ``PROMPT_VERSION`` hashes — so a stored version that differs from either half was in
#: force at a different time.
_IN_FORCE_RUBRIC = f"{RUBRIC_NAME} v{RUBRIC_VERSION}"


@dataclass(frozen=True)
class _JudgeCall:
    """The judge's surviving decision for one trajectory."""

    verdict: str
    rationale: str | None
    task_key: str
    rubric_scores: Mapping[str, Any]
    # The prompt that graded it, taken from the run rather than the module constant:
    # the constant names today's rules and these rows are history.
    prompt_version: str | None


@dataclass(frozen=True)
class _ComparedItem:
    """A trajectory carrying both a human majority and a judge verdict."""

    trajectory_id: str
    task_key: str
    human_verdict: str
    judge_verdict: str
    annotators: tuple[str, ...]
    judge_rationale: str | None
    # ``key -> (human, judge)``, holding only the criteria both sides answered.
    criterion_answers: Mapping[str, tuple[str, str]]
    # Criteria exactly one side answered, in rubric order. Not a comparison and not an
    # absence either side can be held to: it is the part of the rubric this trajectory
    # was never measured on, which is what stops a thin overlap from passing as a
    # reading the two sides shared.
    criterion_gaps: tuple[str, ...]

    @property
    def distance(self) -> int:
        """Steps apart on ``fail < borderline < pass``; 2 is a pass/fail flip."""
        return abs(_VERDICT_INDEX[self.human_verdict] - _VERDICT_INDEX[self.judge_verdict])


def _majority(answers: Iterable[str]) -> str | None:
    """The single most common answer, or ``None`` when there is none or the top two tie.

    One implementation, because verdicts and criterion scores must break ties the same
    way: a report that dropped a split panel from the headline while letting one
    annotator's answer stand in for the panel on a criterion would be aggregating two
    different ideas of what the humans said.
    """
    ranked = Counter(answers).most_common()
    if not ranked:
        return None
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0]


def human_majority(labels: Sequence[HumanLabel]) -> str | None:
    """The majority human verdict for one trajectory.

    Returns ``None`` when nobody labeled the trajectory and when the top two
    verdicts are level: a split panel has no majority to hold the judge to, and
    inventing one (by annotator order, say) would quietly bias the comparison.
    Such trajectories are excluded from the statistics and counted as ties.
    """
    return _majority(label.verdict for label in labels)


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
    counted so a low sample size can be traced back to a flaky run. The run's
    prompt version rides along because which rulebook graded a verdict is a
    property of the run it belongs to, not of the judge.
    """
    stmt = (
        select(
            JudgeVerdict.trajectory_id,
            JudgeVerdict.verdict,
            JudgeVerdict.rationale,
            JudgeVerdict.rubric_scores,
            Task.key,
            EvalRun.judge_prompt_version,
        )
        # Inner join, safe because every verdict row carries a non-null eval_run_id;
        # were that ever relaxed this must become an outer join or the orphaned
        # verdicts would silently vanish from every calibration figure.
        .join(EvalRun, JudgeVerdict.eval_run_id == EvalRun.id)
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
    for trajectory_id, verdict, rationale, scores, key, prompt_version in session.execute(stmt):
        if verdict is None:
            errors += 1
            continue
        # Rows arrive oldest first, so the last write per trajectory wins.
        calls[trajectory_id] = _JudgeCall(
            verdict=verdict,
            rationale=rationale,
            task_key=key,
            # Verdicts recorded before the rubric was structured carry no scores at all.
            rubric_scores=scores or {},
            prompt_version=prompt_version,
        )
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


def _answers(scores: Mapping[str, Any] | None) -> dict[str, str]:
    """One stored ``rubric_scores`` blob as agreement labels, keeping only real answers.

    A key outside :data:`~agentverdict.rubric.CRITERIA`, or a value outside the
    criterion's permitted set, is dropped rather than repaired. Both are answers to a
    question this rubric did not ask, and snapping 0.7 to the nearest permitted score
    would invent an opinion nobody held; dropped, the criterion reads as unanswered,
    which is what it is.
    """
    answers: dict[str, str] = {}
    for key, value in (scores or {}).items():
        criterion = CRITERIA_BY_KEY.get(key)
        if criterion is not None and value in criterion.values:
            answers[key] = score_label(value)
    return answers


def _human_criterion_answers(labels: Sequence[HumanLabel]) -> dict[str, str]:
    """The panel's answer per criterion: a majority of the annotators who answered it.

    Computed per key rather than per label, because a criterion two annotators answered
    and a third skipped still has an answer behind it. Ties drop out under the same rule
    that drops a tied verdict — a split panel holds no position for the judge to be
    scored against, and picking a side by annotator order would settle the comparison
    alphabetically. With two annotators every disagreement is a tie, so a criterion the
    panel splits on leaves the judge unmeasured on precisely the item that was hardest
    to call; the drop is counted in ``criteria_unanswered`` rather than hidden, because
    a criterion's kappa is only readable next to how much of the sample reached it.
    """
    given: dict[str, list[str]] = defaultdict(list)
    for label in labels:
        for key, answer in _answers(label.rubric_scores).items():
            given[key].append(answer)
    majorities = {key: _majority(answers) for key, answers in given.items()}
    return {key: answer for key, answer in majorities.items() if answer is not None}


def _compare_criteria(
    labels: Sequence[HumanLabel], judge_scores: Mapping[str, Any] | None
) -> tuple[dict[str, tuple[str, str]], tuple[str, ...]]:
    """The criteria both sides answered, and the ones exactly one side did.

    A key one side left out is excluded, never filled in with 0.0. 0.0 is an answer —
    *the agent failed this* — and spending it on *the question did not arise* would book
    a consent violation against every run that only looked something up, then report the
    two raters as agreeing about it. Human first, matching the ``(reference,
    comparison)`` order every statistic in this module takes; rubric order, so the rows
    downstream read in the order the rulebook and the labeling form ask the questions.

    The excluded keys are returned rather than discarded. They are what separates *the
    two sides answered the same* from *the two sides were barely compared*, and only
    ``_rule_disagreements`` can tell those apart.
    """
    human = _human_criterion_answers(labels)
    judge = _answers(judge_scores)
    shared = {
        key: (human[key], judge[key])
        for key in CRITERIA_BY_KEY
        if key in human and key in judge
    }
    gaps = tuple(key for key in CRITERIA_BY_KEY if (key in human) != (key in judge))
    return shared, gaps


def _criterion_agreement(
    items: Sequence[_ComparedItem], *, iterations: int, seed: int
) -> tuple[list[CriterionAgreement], dict[str, int]]:
    """Judge-vs-human agreement per criterion, and what each criterion could not reach.

    One row per criterion, including the ones nobody has answered yet: a row reading
    ``n=0`` says the column is empty, whereas omitting it says the rubric has four
    criteria. The statistics come from the same ``agreement_stats`` the verdicts use,
    over the score labels instead of the verdict vocabulary — a kappa written separately
    for scores would be a second set of decisions about degenerate raters, empty samples
    and discarded resamples, and the two would drift apart on the first edit.

    Every criterion is measured over the trajectories the headline is measured over, so
    ``n + criteria_unanswered[key] == len(items)`` holds for each key and the difference
    between the two numbers is always visible. Scoring a criterion on a wider set than
    the verdicts would print two figures side by side that describe different item sets,
    which is the error the per-rater rows further up exist to undo.
    """
    rows: list[CriterionAgreement] = []
    unanswered: dict[str, int] = {}
    for criterion in CRITERIA:
        pairs = [
            item.criterion_answers[criterion.key]
            for item in items
            if criterion.key in item.criterion_answers
        ]
        unanswered[criterion.key] = len(items) - len(pairs)
        stats = agreement_stats(pairs, SCORE_LABELS, bootstrap_iterations=iterations, seed=seed)
        low, high = stats.kappa_ci if stats.kappa_ci is not None else (None, None)
        rows.append(
            CriterionAgreement(
                key=criterion.key,
                description=criterion.description,
                n=stats.n,
                observed_agreement=stats.observed_agreement,
                cohens_kappa=stats.cohens_kappa,
                kappa_ci_low=low,
                kappa_ci_high=high,
                bootstrap_usable=stats.bootstrap_usable,
                bootstrap_iterations=stats.bootstrap_iterations,
                human_positive=sum(1 for human, _ in pairs if human == _POSITIVE_ANSWER),
                judge_positive=sum(1 for _, judge in pairs if judge == _POSITIVE_ANSWER),
            )
        )
    return rows, unanswered


#: How many one-sided criteria — answered by the judge or the panel but not both — a
#: verdict split may carry and still be attributed to the rubric: none. The rubric-fault
#: claim is "the two sides read the run identically and the rule split them", and it has
#: to be earned on everything either side answered, not on whatever the two happened to
#: overlap on. Every criterion is optional to an annotator (``LABELING.md`` says so in
#: as many words), so an overlap of one is the ordinary submission rather than a
#: degenerate one — and with five criteria, a single unmeasured one is a fifth of the
#: rubric that can carry the whole verdict by itself (``ended_waiting`` alone is what
#: rule 1 turns on). Any allowance above zero therefore re-admits the case this count
#: exists to exclude: a judge that misread four criteria in five, reported as proof the
#: rulebook is at fault.
_RULE_MAX_ONE_SIDED_CRITERIA = 0


def _rule_disagreements(items: Sequence[_ComparedItem]) -> tuple[int, int]:
    """Split the verdict disagreements into rubric faults and unmeasured ones.

    These are not judge errors. The judge read the run exactly as the annotators did —
    same goal, same tools, same ending — and the two still landed on different verdicts,
    which leaves one place for the difference to live: the rule that turns answers into
    a verdict. ``order-status-01`` is the case that put this number on the report. The
    judge's rationale agreed with both annotators about every fact of the run and it
    still said ``borderline``, because the unfinished-conversation rule as then written
    fired on an agent that had done what was asked.

    So when this is a large share of ``disagreements_total``, the artifact to change is
    the rubric — the rules in ``DESIGN.md`` and the prompt paragraph that mirrors them —
    and not the model, the temperature or the examples. Re-prompting a judge that
    already sees what you see moves nothing. Rewrite the rule, bump ``RUBRIC_VERSION``
    and ``PROMPT_VERSION``, re-judge, and treat the earlier numbers as describing a
    different rulebook. When this is near zero and disagreements are still high, the
    opposite reading holds: the two sides are reading the runs differently, and the
    judge is what needs the work.

    Which is why the claim has to be earned on the whole rubric and not on whatever the
    two sides happened to overlap on. Every criterion is optional to an annotator —
    ``LABELING.md`` says so in as many words — so a trajectory where one annotator ticked
    ``tools_correct`` and left the rest alone shares exactly one answer with a judge that
    filled in all five. "They agreed on everything they both answered" is then a fact
    about how little was compared, not about the rule, and counting it would report a
    judge that misread four criteria out of five as evidence that the rulebook is at
    fault. So a disagreement with more than :data:`_RULE_MAX_ONE_SIDED_CRITERIA`
    one-sided criteria goes to the second count, and so does one with no shared criteria
    at all — round one, labeled before the criteria existed, is 29 of those. Neither is
    a rubric fault or a judge error: it is unmeasured, and it is reported as its own
    number because the fix for it is more labeling rather than a rewrite of anything.
    """
    rubric = 0
    undetermined = 0
    for item in items:
        if item.human_verdict == item.judge_verdict:
            continue
        # `not item.criterion_answers` is load-bearing: without it, a trajectory where
        # NEITHER side scored anything has no gaps either, and "they agreed on every
        # criterion" would hold vacuously for the whole pre-criteria corpus.
        if len(item.criterion_gaps) > _RULE_MAX_ONE_SIDED_CRITERIA or not item.criterion_answers:
            undetermined += 1
        elif all(human == judge for human, judge in item.criterion_answers.values()):
            rubric += 1
    return rubric, undetermined


def _label_rubrics(
    session: Session,
    labels_by_trajectory: Mapping[str, list[HumanLabel]],
    judge_calls: Mapping[str, _JudgeCall],
) -> dict[str, int]:
    """Which rulebook each in-scope label was made under, as ``"name vN" -> labels``.

    Scoped to the trajectories this judge scored, because that is the population every
    agreement figure is computed over: a label on a trajectory nobody judged moves no
    statistic here, and letting it name a second rubric would raise a warning about an
    average that was never taken. Counted over labels rather than trajectories, because
    a trajectory labeled in both rounds sits on both sides of a rubric change and a
    per-trajectory count would have to pick one.

    Labels pointing at no rubric row are counted under :data:`_UNRECORDED` rather than
    dropped — they are round one, collected under rules that were never written down,
    and they are the reason this function exists. A rubric id resolves to ``name vN``,
    which a reader can place; a row deleted out from under a finished round falls back
    to the raw id, still separating one round from another.
    """
    in_scope = [
        label
        for trajectory_id in judge_calls
        for label in labels_by_trajectory.get(trajectory_id, ())
    ]
    ids = {label.rubric_id for label in in_scope if label.rubric_id is not None}
    named: dict[str, str] = {}
    if ids:
        rows = session.execute(
            select(Rubric.id, Rubric.name, Rubric.version).where(Rubric.id.in_(ids))
        )
        named = {row.id: f"{row.name} v{row.version}" for row in rows}
    counts = Counter(
        _UNRECORDED if label.rubric_id is None else named.get(label.rubric_id, label.rubric_id)
        for label in in_scope
    )
    return dict(sorted(counts.items()))


def _judge_prompt_versions(judge_calls: Mapping[str, _JudgeCall]) -> dict[str, int]:
    """The judge's side of the same question: prompt fingerprint -> verdicts it graded.

    Counted over the surviving verdict per trajectory — the rows every figure in this
    report is computed from — never over every row the judge ever wrote.
    """
    counts = Counter(call.prompt_version or _UNRECORDED for call in judge_calls.values())
    return dict(sorted(counts.items()))


def _rubric_warning(
    label_rubrics: Mapping[str, int], judge_prompt_versions: Mapping[str, int]
) -> str | None:
    """What has to be said before a single number is quoted off this report, or None.

    A rubric edit re-grades the corpus: figures made under different rulebooks are
    different measurements, and an average of them describes no rulebook that ever
    existed. That is the failure recorded in ``docs/rubric-change-2026-08-11.md`` and
    the reason a label now carries the rubric it was made under. The split is stated as
    one sentence every surface prints verbatim — the CLI, the web page and the raw JSON
    must not each re-derive their own reading of it. The sentence names the split and
    the ways out and stops there: scoping the report to one round, re-labeling and
    re-judging are all valid answers, and only the caller knows which is affordable.

    :data:`_UNRECORDED` is a rulebook of its own for the two *span* checks — a report
    mixing round-one labels with stamped ones really does average across a rubric
    change — but it is never called a mismatch against the version in force. An
    unstamped run may well have been graded by the prompt in force now, and saying
    otherwise would invent the provenance the nullable columns exist to refuse.
    Returns None when either side is empty: with nothing compared there is no average
    to warn about, and the empty state is already reported as such.
    """
    if not label_rubrics or not judge_prompt_versions:
        return None
    faults: list[str] = []
    if len(label_rubrics) > 1:
        faults.append(
            f"labels were made under {len(label_rubrics)} rubric versions"
            f" ({', '.join(label_rubrics)})"
        )
    if len(judge_prompt_versions) > 1:
        faults.append(
            f"verdicts were graded under {len(judge_prompt_versions)} judge prompt versions"
            f" ({', '.join(judge_prompt_versions)})"
        )
    stale_rubrics = [
        version for version in label_rubrics if version not in (_UNRECORDED, _IN_FORCE_RUBRIC)
    ]
    if stale_rubrics:
        faults.append(
            f"the rubric in force is {_IN_FORCE_RUBRIC} and these labels answer"
            f" {', '.join(stale_rubrics)}"
        )
    stale_prompts = [
        version
        for version in judge_prompt_versions
        if version not in (_UNRECORDED, PROMPT_VERSION)
    ]
    if stale_prompts:
        faults.append(
            f"the judge prompt in force is {PROMPT_VERSION} and these verdicts were"
            f" graded under {', '.join(stale_prompts)}"
        )
    if not faults:
        return None
    return (
        f"This report spans more than one rulebook: {'; '.join(faults)}. A rubric edit"
        " re-grades the corpus, so figures made under different rulebooks are different"
        " measurements and an average across them describes no rulebook that ever"
        " existed. Scope the report to one annotation round or eval run, or re-label"
        " and re-judge under the current rubric, before quoting a single number off it."
    )


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
        labels = labels_by_trajectory[trajectory_id]
        shared, gaps = _compare_criteria(labels, call.rubric_scores)
        compared_items.append(
            _ComparedItem(
                trajectory_id=trajectory_id,
                task_key=call.task_key,
                human_verdict=human_verdict,
                judge_verdict=call.verdict,
                annotators=tuple(sorted(label.annotator for label in labels)),
                judge_rationale=call.rationale,
                criterion_answers=shared,
                criterion_gaps=gaps,
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

    criteria, criteria_unanswered = _criterion_agreement(
        compared_items, iterations=bootstrap_iterations, seed=seed
    )
    rule_disagreements, rule_undetermined = _rule_disagreements(compared_items)
    label_rubrics = _label_rubrics(session, labels_by_trajectory, judge_calls)
    judge_prompt_versions = _judge_prompt_versions(judge_calls)

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
        criteria=criteria,
        criteria_unanswered=criteria_unanswered,
        rule_disagreements=rule_disagreements,
        rule_undetermined=rule_undetermined,
        label_rubrics=label_rubrics,
        judge_prompt_versions=judge_prompt_versions,
        rubric_warning=_rubric_warning(label_rubrics, judge_prompt_versions),
    )
