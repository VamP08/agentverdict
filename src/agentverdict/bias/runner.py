"""Probe orchestration and the report read back off the stored rows.

Two halves that are deliberately not one function. ``run_probe`` spends tokens: it
re-asks the judge about every stored trajectory under each arm of a probe and records
one row per call. ``build_bias_probe_report`` spends nothing: it reads those rows and
computes the intervals. The API and the CLI both need the second without the first —
a page refresh must never re-run a probe — and keeping the statistics on the far side
of the database also means the numbers a reader sees can be recomputed from what was
stored rather than from what a run happened to hold in memory.

Four hazards shape the writing half, each of which would be silent:

* ``run_eval`` is not called. It creates an ``EvalRun`` and writes ``judge_verdicts``,
  and those rows are what calibration and the merge gate read — a verdict on a
  deliberately reordered or padded transcript would quietly become this judge's
  current opinion of the run.
* Nothing writable is opened on ``trajectories`` or ``trajectory_steps``. The
  perturbation happens on the rendered string, downstream of the ORM, so filler has
  no route into the golden dataset.
* The run records the *unperturbed* ``PROMPT_VERSION``, because the artifact under
  test is the judge that grades in production; each arm's departure from it is named
  per row by ``variant`` and fingerprinted by ``prompt_sha``.
* An arm that renders identity's bytes for a trajectory measured nothing there, and
  an arm that does so for every trajectory is reported ``not_applied`` rather than
  ``inconclusive``. Reporting a broken harness as "no bias detected" is the single
  easiest way for this whole milestone to be worthless. Failing every call is the
  *other* way to measure nothing and keeps its own name: the perturbation did reach
  the model there, so ``not_applied`` would send a reader to debug a renderer that
  worked. Such an arm carries ``measured_on == 0``, and it is that count rather than
  the delivery count that every figure beside it is denominated in.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from math import ceil
from statistics import fmean
from typing import Literal

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, selectinload

from agentverdict.bias.perturb import (
    LENGTH_VARIANTS,
    ORDER_VARIANTS,
    doses_for,
    pad_transcript,
    plan_for,
)
from agentverdict.bias.prompts import build_probe_messages, render_transcript
from agentverdict.bias.stats import (
    ArmSamples,
    adjusted_confidence,
    bootstrap_clustered_mean_ci,
    disagreement_across,
    disagreement_within,
    min_movers,
    signed_shift,
)
from agentverdict.calibration.stats import VERDICT_SCORES
from agentverdict.judging.client import GroqChatClient, JudgeClientError
from agentverdict.judging.prompts import PROMPT_VERSION
from agentverdict.judging.runner import judge_messages
from agentverdict.models import (
    BiasProbeResult,
    BiasProbeRun,
    HumanLabel,
    Judge,
    Task,
    Trajectory,
    utcnow,
)
from agentverdict.schemas import ArmOutcome, BiasProbeReport

#: The control arm. Present in every probe, always judged, and the arm every other one
#: is measured against.
IDENTITY = "identity"

#: Probe name -> its arms, control first. The registry is the run's plan and the
#: report's reading order, so both halves agree on what a probe consists of without
#: either inferring it from rows that happen to exist.
_PROBE_VARIANTS: dict[str, dict[str, str]] = {"order": ORDER_VARIANTS, "length": LENGTH_VARIANTS}

#: Probe names the surfaces accept.
PROBES: tuple[str, ...] = tuple(_PROBE_VARIANTS)

#: Draws per arm below which a probe is not an experiment. At one draw the identity
#: arm has no within-arm pair, so its disagreement is undefined and the control
#: measures nothing -- every arm's shift would then be a number with no floor under it.
MIN_REPEATS = 2

ProgressCallback = Callable[[int, int, Trajectory, BiasProbeResult], None]

Outcome = Literal["biased", "inverse", "inconclusive", "not_applied", "no_data"]


def variants_for(probe: str) -> dict[str, str]:
    """The arms of ``probe``, control first, mapped to their descriptions."""
    try:
        return _PROBE_VARIANTS[probe]
    except KeyError:
        raise ValueError(f"unknown probe {probe!r}; known probes: {', '.join(PROBES)}") from None


# --- Rendering ---------------------------------------------------------------


@dataclass(frozen=True)
class _RenderedArm:
    """One arm's prompt for one trajectory, with the delivery check already applied."""

    variant: str
    messages: list[dict[str, str]]
    prompt_sha: str
    #: False when this arm rendered identity's bytes, so its perturbation never
    #: reached the model for this trajectory. Such a pair is not judged and not
    #: counted: a control compared against itself is a zero by construction, and
    #: averaging those zeros in would flatter the judge.
    delivered: bool


def _prompt_sha(messages: list[dict[str, str]]) -> str:
    """Fingerprint the rendered user turn: sixteen hex characters of its sha256.

    Deliberately unrelated to ``PROMPT_VERSION``, which fingerprints the rubric the
    judge was reading. This identifies what one trajectory rendered to under one arm,
    which is what makes the delivery check possible at all — two arms that hash the
    same sent the same prompt, whatever the variant column claims.
    """
    user = next(message["content"] for message in messages if message["role"] == "user")
    return hashlib.sha256(user.encode("utf-8")).hexdigest()[:16]


def _render(
    probe: str, task: Task, transcript: str, variant: str
) -> tuple[list[dict[str, str]], str]:
    """Render one arm of one trajectory, returning its messages and their fingerprint."""
    # The order probe permutes the layout and leaves the transcript alone; the length
    # probe does the opposite. Both go through the same two calls, with the arm that
    # does not apply contributing its no-op (zero doses, canonical plan).
    doses = doses_for(variant) if probe == "length" else 0
    messages = build_probe_messages(task, pad_transcript(transcript, doses), plan_for(variant))
    return messages, _prompt_sha(messages)


def _render_arms(probe: str, task: Task, transcript: str) -> list[_RenderedArm]:
    """Render every arm of one trajectory, control first.

    Identity is rendered first because every other arm's delivery check is a
    comparison against its bytes.
    """
    messages, identity_sha = _render(probe, task, transcript, IDENTITY)
    arms = [_RenderedArm(IDENTITY, messages, identity_sha, delivered=True)]
    for variant in variants_for(probe):
        if variant == IDENTITY:
            continue
        messages, prompt_sha = _render(probe, task, transcript, variant)
        arms.append(
            _RenderedArm(variant, messages, prompt_sha, delivered=prompt_sha != identity_sha)
        )
    return arms


# --- The probe run -----------------------------------------------------------


def run_probe(
    session: Session,
    judge: Judge,
    probe: str,
    *,
    repeats: int = 3,
    limit: int | None = None,
    seed: int = 0,
    client: GroqChatClient | None = None,
    on_progress: ProgressCallback | None = None,
) -> BiasProbeRun:
    """Re-judge stored trajectories under every arm of ``probe`` and record each call.

    A probe compares the judge against itself, so it needs no human labels and reads
    every stored trajectory rather than the labeled slice. One failed call never
    aborts a run — it is recorded as an error row and the sweep continues — but a
    failure that is not a judge failure marks the run ``failed`` rather than leaving
    it at ``running`` forever.

    The call schedule is fixed as ``(trajectory, variant, repeat)`` and nothing about
    it is random, so a run that dies partway is a prefix of the run it would have
    been. ``seed`` reaches the bootstrap in the report, never the schedule.

    Args:
        session: Open session. Probe rows are written; trajectories are only read.
        judge: The judge under test, called at its own configured temperature —
            the artifact being measured is the judge as production runs it.
        probe: ``order`` or ``length``.
        repeats: Draws per arm, at least ``MIN_REPEATS``.
        limit: Cap on trajectories, applied over ``(created_at, id)`` so the same cap
            always selects the same trajectories.
        seed: Recorded on the run and used by the report's bootstrap, so a report is
            reproducible from the run that produced it.
        client: Chat client to use. Tests inject one and never reach the network;
            when omitted, one is created here and closed on the way out.
        on_progress: Called after each row is written, as
            ``(position, planned, trajectory, result)``.

    Raises:
        ValueError: unknown probe name, or ``repeats`` below two.
    """
    variants = variants_for(probe)  # validates the name before anything is written
    if repeats < MIN_REPEATS:
        raise ValueError(
            f"repeats must be at least {MIN_REPEATS}, got {repeats}: at one draw the "
            "identity arm has no within-arm pair, so the control measures nothing"
        )

    # Every trajectory, not the labeled slice. Steps and task are loaded up front
    # because each is rendered once per arm and a lazy load per arm would issue a
    # query inside the judging loop.
    stmt = (
        select(Trajectory)
        .options(selectinload(Trajectory.steps), selectinload(Trajectory.task))
        .order_by(Trajectory.created_at, Trajectory.id)
    )
    if limit is not None:
        stmt = stmt.limit(limit)
    trajectories = list(session.scalars(stmt))

    # Rendering is pure and cheap, so the whole plan is built before the first call.
    # That is what lets the delivery check skip an arm rather than pay for a call
    # whose result would have to be discarded, and it makes the progress total exact.
    plans = [
        (trajectory, _render_arms(probe, trajectory.task, render_transcript(trajectory)))
        for trajectory in trajectories
    ]
    planned = sum(repeats if arm.delivered else 1 for _, arms in plans for arm in arms)

    run = BiasProbeRun(
        judge=judge,
        probe=probe,
        status="running",
        repeats=repeats,
        seed=seed,
        # The unperturbed fingerprint, stamped at creation like EvalRun's: a run that
        # dies partway still has to be attributable to a grader.
        judge_prompt_version=PROMPT_VERSION,
        # What was attempted, which is what "13 trajectories" in a summary means.
        # Per-arm denominators differ and belong to the arms, not here.
        trajectory_count=len(trajectories),
        # `arms` is not derivable from the rows once the registry moves on, and `limit`
        # is not derivable at all: five trajectories out of twenty-nine and five out of
        # five leave identical row sets.
        meta={"limit": limit, "arms": list(variants)},
    )
    session.add(run)
    session.flush()

    owns_client = client is None
    if owns_client:
        client = GroqChatClient()
    position = 0
    try:
        for trajectory, arms in plans:
            for arm in arms:
                # An undelivered arm gets one row, not `repeats` of them: the fact on
                # record is that the perturbation did not exist for this trajectory,
                # not that some calls failed. Verdict and error both stay null, so it
                # inflates no error count anywhere, and the report reads it back off
                # the fingerprint -- which is what the delivery check is.
                for repeat in range(repeats if arm.delivered else 1):
                    row = BiasProbeResult(
                        probe_run_id=run.id,
                        # Ids, never relationships. Attaching the loaded Trajectory
                        # would cascade it and its steps into the session that writes
                        # probe rows, which is the only route by which a perturbation
                        # could ever reach the golden dataset; and a row that failed
                        # before it was added would otherwise sit in `run.results`
                        # unsaved, which SQLAlchemy can only warn about at flush.
                        trajectory_id=trajectory.id,
                        variant=arm.variant,
                        repeat=repeat,
                        prompt_sha=arm.prompt_sha,
                        # Explicit zeros: column defaults only apply at flush, and the
                        # totals below read these before that happens.
                        input_tokens=0,
                        output_tokens=0,
                    )
                    if arm.delivered:
                        try:
                            # The production entry point, one level below the prompt:
                            # same model, same temperature, same corrective retry. A
                            # probe that called the judge its own way would report
                            # intervals for a judge nobody grades with.
                            call = judge_messages(client, judge, arm.messages)
                        except (JudgeClientError, ValueError) as exc:
                            row.error = str(exc)[:1000]
                            run.error_count += 1
                        else:
                            row.verdict = call.decision.verdict
                            row.rationale = call.decision.rationale
                            row.latency_ms = call.latency_ms
                            row.input_tokens = call.input_tokens
                            row.output_tokens = call.output_tokens
                        run.input_tokens += row.input_tokens
                        run.output_tokens += row.output_tokens
                    # Committed per row, not flushed, so a run killed at call ninety
                    # keeps ninety rows. A flush is only durable if the caller commits,
                    # and no caller does on this path: the CLI's session context manager
                    # rolls back on the way out, and Ctrl-C is not an `Exception`, so it
                    # reaches no `except` clause at all and `session.close()` discards
                    # the sweep. Committing here also releases SQLite's write lock
                    # between calls, which one transaction spanning every model call
                    # would hold against the labeling UI for the length of the run.
                    session.add(row)
                    session.commit()
                    position += 1
                    if on_progress is not None:
                        on_progress(position, planned, trajectory, row)
    except BaseException:
        # Anything that got past the per-call handler is not a judge failure, and a
        # run left at "running" is indistinguishable from one still in flight.
        run.status = "failed"
        run.completed_at = utcnow()
        with suppress(SQLAlchemyError):
            # Committed for the same reason the rows above are: the status set two
            # lines up is the last write of the run and the only one no row commit
            # carries, so leaving it to the caller loses exactly it -- the sweep's
            # rows would survive under a run still claiming to be in flight. Best
            # effort: when the session is what broke, the original exception is the
            # one worth propagating, and the caller's rollback is still correct after
            # this.
            session.commit()
        raise
    finally:
        if owns_client:
            client.close()

    run.status = "completed"
    run.completed_at = utcnow()
    # Committed for the same reason the rows above are: every result is already durable
    # by this point, so a flush here would leave a fully measured sweep recorded as
    # `running` if anything between this return and the caller's commit went wrong --
    # a finished run wearing the one status that means "still in flight".
    session.commit()
    return run


# --- The report --------------------------------------------------------------


@dataclass(frozen=True)
class _ProbeRows:
    """One probe run's stored rows, indexed the way the statistics read them."""

    task_keys: dict[str, str]  # trajectory id -> task key, the resampling unit
    draws: dict[tuple[str, str], tuple[str | None, ...]]  # (trajectory, variant) -> verdicts
    shas: dict[tuple[str, str], str]  # (trajectory, variant) -> rendered prompt fingerprint
    errors: int
    #: trajectory id -> the human majority verdict, for the trajectories that have one.
    #: Ties are absent rather than stored as None: with two annotators every
    #: disagreement is a tie, and a tie is not a direction to move toward. Defaulted
    #: because the direction check is a diagnostic beside the measurement, not part of
    #: it -- a probe on an unlabeled corpus is still a probe.
    majorities: dict[str, str] = field(default_factory=dict)

    @property
    def trajectory_ids(self) -> list[str]:
        """Every trajectory the run touched, in a stable order for the seeded bootstrap."""
        return sorted(self.task_keys)

    def arm(self, trajectory_id: str, variant: str) -> ArmSamples:
        """One trajectory's draws in one arm, carrying the task key they cluster under."""
        return ArmSamples(
            trajectory_id=trajectory_id,
            task_key=self.task_keys[trajectory_id],
            verdicts=self.draws.get((trajectory_id, variant), ()),
        )


def _latest_probe_run(
    session: Session, judge: Judge, *, probe: str, probe_run_id: str | None
) -> BiasProbeRun | None:
    """The run a report should describe: a named one, else the newest completed, else the newest.

    Completed wins over newest so that a run which crashed on its third call cannot
    hide last week's finished measurement. When nothing completed, the newest failed
    run is still reported rather than an empty page — it measured something, and its
    error count says how much.

    A named run belonging to another judge or another probe yields ``None``, and with
    it an empty report: answering with it would mislabel every number below.
    """
    if probe_run_id is not None:
        run = session.get(BiasProbeRun, probe_run_id)
        if run is None or run.judge_id != judge.id or run.probe != probe:
            return None
        return run
    stmt = (
        select(BiasProbeRun)
        .where(BiasProbeRun.judge_id == judge.id, BiasProbeRun.probe == probe)
        .order_by(BiasProbeRun.started_at.desc(), BiasProbeRun.id.desc())
    )
    completed = session.scalars(stmt.where(BiasProbeRun.status == "completed").limit(1)).first()
    return completed if completed is not None else session.scalars(stmt.limit(1)).first()


def _load_rows(session: Session, run: BiasProbeRun) -> _ProbeRows:
    """Read one run's result rows in a single query, grouped for the statistics."""
    stmt = (
        select(
            BiasProbeResult.trajectory_id,
            Task.key,
            BiasProbeResult.variant,
            BiasProbeResult.verdict,
            BiasProbeResult.error,
            BiasProbeResult.prompt_sha,
        )
        .join(Trajectory, BiasProbeResult.trajectory_id == Trajectory.id)
        .join(Task, Trajectory.task_id == Task.id)
        .where(BiasProbeResult.probe_run_id == run.id)
        .order_by(BiasProbeResult.trajectory_id, BiasProbeResult.variant, BiasProbeResult.repeat)
    )
    task_keys: dict[str, str] = {}
    draws: dict[tuple[str, str], list[str | None]] = defaultdict(list)
    shas: dict[tuple[str, str], str] = {}
    errors = 0
    for trajectory_id, task_key, variant, verdict, error, prompt_sha in session.execute(stmt):
        task_keys[trajectory_id] = task_key
        draws[(trajectory_id, variant)].append(verdict)
        # Every repeat of an arm renders the same bytes, so the first row's
        # fingerprint is the arm's.
        shas.setdefault((trajectory_id, variant), prompt_sha)
        # Counted from `error` and never from a null verdict: an arm that never
        # applied stores a null verdict too, and that is not a failed call.
        if error is not None:
            errors += 1
    return _ProbeRows(
        task_keys=task_keys,
        draws={key: tuple(values) for key, values in draws.items()},
        shas=shas,
        errors=errors,
        majorities=_human_majorities(session, list(task_keys)),
    )


def _human_majorities(session: Session, trajectory_ids: list[str]) -> dict[str, str]:
    """The human majority verdict per trajectory, for those that have one.

    Read for the direction check alone. A probe measures the judge against itself and
    needs no labels, but where labels exist they answer a question the shift cannot: a
    verdict that moves *toward* the humans under perturbation is the judge being
    corrected by a nudge, which is a different fact about it than being knocked off a
    verdict the humans agreed with. Both raise the shift identically.
    """
    if not trajectory_ids:
        return {}
    stmt = select(HumanLabel.trajectory_id, HumanLabel.verdict).where(
        HumanLabel.trajectory_id.in_(trajectory_ids)
    )
    votes: dict[str, Counter[str]] = defaultdict(Counter)
    for trajectory_id, verdict in session.execute(stmt):
        votes[trajectory_id][verdict] += 1
    majorities: dict[str, str] = {}
    for trajectory_id, counts in votes.items():
        ranked = counts.most_common()
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            continue  # a tie names no majority, so there is nothing to move toward
        majorities[trajectory_id] = ranked[0][0]
    return majorities


#: Below this, an interval endpoint is arithmetic residue rather than a measurement.
#: A per-trajectory shift is a difference of means of ``VERDICT_SCORES`` over ``repeats``
#: draws, so it lands on the lattice of 1/(2*repeats) — sixths at three repeats — and a
#: resample mean divides that lattice by the pooled trajectory count. The smallest
#: non-zero value any interval here can hold is therefore about 3e-4 even at five
#: hundred trajectories, and coarser for every corpus this project will see. Binary
#: floating point cannot represent sixths, so a resample whose *exact* mean is zero
#: does not sum to 0.0; it sums to ±1.4e-17. This threshold sits seven orders above
#: that residue and five below the coarsest real signal, so it can only ever discard
#: the residue — no attainable effect size is small enough to be swallowed by it.
ZERO_RESIDUE = 1e-9

#: Resamples that must land beyond each endpoint before that endpoint is a property of
#: the data rather than of the seed. The merge gate already fixed this density for an
#: interval that is thresholded rather than read — 10,000 resamples at a 2.5% tail —
#: and an arm outcome is thresholded the same way, so the probe inherits the number
#: instead of inventing a second one.
TAIL_RESAMPLES = 250


def _tail_sized_iterations(requested: int, confidence: float) -> int:
    """Resamples enough that ``TAIL_RESAMPLES`` of them land beyond each endpoint.

    The 1000 that reporting statistics use elsewhere is sized for a 2.5% tail, where it
    puts 25 resamples past the endpoint. These intervals are read at the Sidak-adjusted
    level instead — 0.98726 over four arms, so a tail of 0.637% — and 1000 resamples
    then place the bound at ``means[6]``, the seventh smallest. The correction moved the
    tail by a factor of four and the resample count has to follow it, or the endpoint is
    an order statistic estimated from six draws.

    It is the *verdict* that moves, not just the last digit. The mover guard admits an
    arm at ``movers == min_movers``, which is by construction the first mover count
    whose atom at zero falls below the tail — so a published arm sits next to that
    boundary by design, and at 1000 resamples which side it lands on is decided by the
    seed. Over a scan of random shift vectors on six to ten tasks, a tenth of them
    changed between ``inconclusive`` and ``biased`` across seeds 0-19; seeded
    reproducibility makes that stable per run without making it right, so two runs over
    identical data publish contradictory findings and each one reproduces.

    Sizing from the tail rather than pinning a constant also keeps the two in step if
    the arm count or the nominal level ever changes.

    It narrows the problem rather than closing it, and the remainder is not a resample
    count away. Where the atom's mass falls just under the tail — ``(6/10)**10`` is
    0.00605 against a tail of 0.00637, so four movers on ten tasks — separating the two
    at three sigma needs half a million resamples, and the gap can be made arbitrarily
    small by choice of n. Those arms are the ones reporting ``movers_observed ==
    min_movers``, which the report prints beside every outcome.
    """
    tail = (1.0 - confidence) / 2.0
    if tail <= 0.0:  # no interval is reported at that level; leave the caller's number
        return requested
    return max(requested, ceil(TAIL_RESAMPLES / tail))


def _clear_residue(interval: tuple[float, float] | None) -> tuple[float, float] | None:
    """Snap interval endpoints that are a rounding error rather than a distance from zero.

    Applied before the interval is either classified or reported, so the two cannot
    disagree. Without it an arm whose lower bound accumulated to +1.4e-17 is called
    ``biased`` beside an interval that renders as ``[+0.000, +0.479]`` and a sentence
    claiming the whole interval lies above zero — and a caller of the JSON API applying
    its own ``low > 0`` test would reach that same wrong answer independently. The
    mover guard does not cover this: the residue appears exactly when many trajectories
    moved in both directions and cancelled, which is when the mover count is highest.
    """
    if interval is None:
        return None
    low, high = (0.0 if abs(bound) < ZERO_RESIDUE else bound for bound in interval)
    return low, high


def _classify(
    interval: tuple[float, float] | None,
    *,
    applied: int,
    measured: int,
    movers: int,
    movers_down: int,
    floor: int | None,
) -> tuple[Outcome, bool]:
    """Read one arm's interval, with the power check the interval cannot make itself.

    An interval touching zero includes "no change at all" among the things it cannot
    rule out, so only one lying wholly on one side says anything — the same strict
    boundary the merge gate reads its own interval by. Strict is only safe because
    the interval arrives from ``_clear_residue``: read off the raw bootstrap, this
    comparison decides the sign of a shift from the last bit of a float.

    The mover count is the second half of the rule. A percentile bootstrap over a
    vector of mostly exact zeros carries an atom at zero of size ``((n-k)/n)**n``, and
    until ``floor`` trajectories move, the tail cannot clear zero however large the
    movement is; an interval that excludes zero with fewer movers than that is an
    artifact of the resample, not a finding. The negative side is counted separately
    because the difference vector is left-truncated — with an interval below zero, the
    trajectories supporting it are the ones that moved down, and a handful of those
    beneath a crowd of upward movers must not pass a power check on the strength of
    movement in the wrong direction.
    """
    if applied == 0:
        return "not_applied", False
    if measured == 0:
        # Delivered to every trajectory and measured on none: the perturbed prompts went
        # out and every judge call answering them failed. `inconclusive` is the wrong
        # word for that -- in this report it means "measured, nothing found", and it is
        # what a reader acts on. Reporting a dead arm in the vocabulary of a clean
        # result is the same error as reporting an undelivered one that way, which is
        # why `not_applied` exists; this is its mirror. It matters most exactly when it
        # is least visible: the calls that fail are the long transcripts, so the arm
        # that dies first is the most heavily padded one.
        return "no_data", True
    if interval is not None and floor is not None:
        low, high = interval
        if low > 0.0 and movers >= floor:
            return "biased", False
        if high < 0.0 and movers_down >= floor:
            return "inverse", False
    # `floor` is None below two trajectories, where no interval can exclude zero at
    # all -- the strongest form of the same statement, so the flag is set, not cleared.
    return "inconclusive", floor is None or movers < floor



def _scores(verdicts: tuple[str | None, ...]) -> list[float]:
    """Usable draws as ordinal scores. Errored draws carry no opinion and are dropped."""
    return [VERDICT_SCORES[verdict] for verdict in verdicts if verdict in VERDICT_SCORES]

def _score_arm(
    rows: _ProbeRows,
    variant: str,
    description: str,
    *,
    floor: int | None,
    confidence: float,
    iterations: int,
    seed: int,
) -> ArmOutcome:
    """Score one perturbed arm against the identity arm, trajectory by trajectory."""
    shifts: dict[str, list[float]] = defaultdict(list)  # task key -> per-trajectory shifts
    crossings: list[float] = []
    applied = measured = movers = movers_down = 0
    movers_labeled = toward = away = 0
    for trajectory_id in rows.trajectory_ids:
        identity_sha = rows.shas.get((trajectory_id, IDENTITY))
        variant_sha = rows.shas.get((trajectory_id, variant))
        # The delivery check. Equal fingerprints mean this arm sent the control's
        # prompt, so the pair measures the judge's noise and not the perturbation;
        # it is dropped whole, because subtracting a control draw that has no variant
        # observation beside it is not a paired comparison.
        if identity_sha is None or variant_sha is None or variant_sha == identity_sha:
            continue
        applied += 1
        control = rows.arm(trajectory_id, IDENTITY)
        perturbed = rows.arm(trajectory_id, variant)
        shift = signed_shift(control.verdicts, perturbed.verdicts)
        if shift is None:  # every call on one side of this pair errored
            continue
        # Counted apart from `applied`, which is the delivery check and says only that
        # the bytes differed. The padded arms carry the longest prompts and lose the
        # most calls, so an arm can be delivered on every trajectory and measured on
        # none; without this the report would print "0 of 13 moved" -- a claim about a
        # judge that held steady -- for a sweep in which the judge never answered.
        measured += 1
        shifts[control.task_key].append(shift)
        crossing = disagreement_across(control.verdicts, perturbed.verdicts)
        if crossing is not None:  # defined under exactly the same condition as the shift
            crossings.append(crossing)
        if shift != 0.0:
            movers += 1
            if shift < 0.0:
                movers_down += 1
            # Direction against the humans, where they said anything. A shift is a
            # magnitude and a sign; it cannot tell being nudged off a verdict the
            # annotators agreed with from being nudged onto one. Counted, never folded
            # into the outcome and never given an interval of its own -- the arm already
            # spends one, and a second would cost family-wise error for a diagnostic.
            majority = rows.majorities.get(trajectory_id)
            if majority is not None:
                movers_labeled += 1
                target = VERDICT_SCORES[majority]
                before = abs(fmean(_scores(control.verdicts)) - target)
                after = abs(fmean(_scores(perturbed.verdicts)) - target)
                if after < before:
                    toward += 1
                elif after > before:
                    away += 1

    values = [value for group in shifts.values() for value in group]
    # Residue cleared once, here, so the bounds this arm reports and the bounds it is
    # classified by are the same numbers.
    interval = _clear_residue(
        bootstrap_clustered_mean_ci(
            shifts, iterations=iterations, confidence=confidence, seed=seed
        ).interval
    )
    low, high = interval if interval is not None else (None, None)
    outcome, power_limited = _classify(
        interval,
        applied=applied,
        measured=measured,
        movers=movers,
        movers_down=movers_down,
        floor=floor,
    )
    return ArmOutcome(
        variant=variant,
        description=description,
        applied_to=applied,
        measured_on=measured,
        movers_labeled=movers_labeled,
        moved_toward_humans=toward,
        moved_away_from_humans=away,
        # The flat mean over trajectories, which is the quantity the clustered
        # bootstrap resamples; a macro-average over tasks would print an estimator the
        # interval beside it does not describe.
        signed_shift=fmean(values) if values else None,
        shift_ci_low=low,
        shift_ci_high=high,
        disagreement=fmean(crossings) if crossings else None,
        movers_observed=movers,
        outcome=outcome,
        power_limited=power_limited,
    )


def _mean_control_disagreement(
    rows: _ProbeRows,
) -> tuple[float | None, Mapping[str, list[float]]]:
    """The identity arm's disagreement per trajectory, pooled and clustered by task.

    This is the judge's own test-retest noise: how often two byte-identical calls come
    back different. It is a headline in its own right and the floor every shift above
    has to be read against, so it is computed over every trajectory that has one —
    never restricted to the subset some arm happened to apply to.
    """
    grouped: dict[str, list[float]] = defaultdict(list)
    for trajectory_id in rows.trajectory_ids:
        control = rows.arm(trajectory_id, IDENTITY)
        within = disagreement_within(control.verdicts)
        if within is not None:  # None below two usable draws, which measure no noise
            grouped[control.task_key].append(within)
    values = [value for group in grouped.values() for value in group]
    return (fmean(values) if values else None), grouped


def build_bias_probe_report(
    session: Session,
    judge: Judge,
    *,
    probe: str = "order",
    probe_run_id: str | None = None,
    confidence: float = 0.95,
    bootstrap_iterations: int = 1000,
) -> BiasProbeReport:
    """Read a stored probe run back as a report. Calls no model and writes nothing.

    Every interval is read at the Sidak-adjusted level and both levels are printed.
    The number of intervals is fixed before the run starts — one signed shift per
    perturbed arm plus the identity arm's own noise floor, so ``len(variants)`` — and
    is taken from the registry rather than from the rows, because a correction sized
    by what came back is not a correction.

    Args:
        session: Open session. Nothing is written, so a read-only one is fine.
        judge: The judge whose probe is being reported.
        probe: ``order`` or ``length``.
        probe_run_id: Report exactly this run instead of the latest one. The CLI uses
            it to report the run it just executed, which on a shared database is not
            reliably "the newest run for this judge".
        confidence: Nominal per-interval level, before the correction.
        bootstrap_iterations: Floor on the resamples behind every interval. Raised to
            whatever the adjusted tail needs (see ``_tail_sized_iterations``), because a
            count that leaves six draws past the endpoint reports the seed.

    Raises:
        ValueError: unknown probe name.
    """
    variants = variants_for(probe)
    level = adjusted_confidence(confidence, len(variants))
    # Sized from the level the intervals are actually read at, not from the nominal one.
    iterations = _tail_sized_iterations(bootstrap_iterations, level)
    run = _latest_probe_run(session, judge, probe=probe, probe_run_id=probe_run_id)
    if run is None:
        # Not an error: a judge nobody has probed yet has no findings, and rendering
        # zeros here would read as a clean bill of health for a probe that never ran.
        return BiasProbeReport(
            judge_id=judge.id,
            judge_name=judge.name,
            probe=probe,
            confidence=confidence,
            adjusted_confidence=level,
            generated_at=utcnow(),
        )

    rows = _load_rows(session, run)
    trajectory_ids = rows.trajectory_ids
    # The seed travels with the run rather than with the caller, so the CLI, the API
    # and the web page all quote the same interval for the same run.
    seed = run.seed
    control_disagreement, control_groups = _mean_control_disagreement(rows)
    control_interval = bootstrap_clustered_mean_ci(
        control_groups, iterations=iterations, confidence=level, seed=seed
    ).interval
    control_low, control_high = control_interval if control_interval is not None else (None, None)

    # One floor for the whole run, computed from the trajectories the run touched and
    # at the level the intervals are read at. Arms that applied to fewer trajectories
    # are held to it unchanged: it barely moves with n (four movers at 13, still four
    # at 60), and a single printed threshold is the one a reader can check.
    floor = min_movers(len(trajectory_ids), confidence=level)

    return BiasProbeReport(
        judge_id=judge.id,
        judge_name=judge.name,
        judge_prompt_version=run.judge_prompt_version,
        probe=probe,
        trajectories=len(trajectory_ids),
        tasks=len(set(rows.task_keys.values())),
        repeats=run.repeats,
        control_disagreement=control_disagreement,
        control_ci_low=control_low,
        control_ci_high=control_high,
        min_movers=floor,
        confidence=confidence,
        adjusted_confidence=level,
        arms=[
            _score_arm(
                rows,
                variant,
                description,
                floor=floor,
                confidence=level,
                iterations=iterations,
                seed=seed,
            )
            for variant, description in variants.items()
            if variant != IDENTITY
        ],
        errors=rows.errors,
        generated_at=utcnow(),
    )
