"""Comparing two eval runs: did this change actually make the agent worse?

Someone opens a pull request that edits a prompt; CI replays the eval suite, judges
it, and this module decides whether the result is worse than a stored baseline run.

Deciding that by putting two suite averages side by side does not work. Two runs of
the *same* suite against the *same* agent already differ: the agent is sampled, the
judge is sampled, and the suite itself is a sample of the work the agent will meet
in production. A couple of tasks drifting between ``borderline`` and ``pass`` moves
the headline average further than most genuine regressions do, so "the mean went
down" is not evidence of anything.

The gate therefore asks a different question. It scores every task in both runs,
takes the per-task differences, and resamples *tasks* with replacement
(``stats.bootstrap_mean_ci``) to see how far the average difference travels when the
suite membership is shuffled. An interval lying entirely below zero means the drop
outlives that shuffling, and the merge is blocked. An interval straddling zero means
nothing was measured, and the merge is not blocked — which is the whole point. A
gate that fires on noise gets switched off within a week, and a switched-off gate
catches nothing at all; being deliberately silent on small samples is what buys the
right to be believed on the large ones.

Two details do most of the remaining work:

* **Pairing is by task key, never by trajectory id.** Replay mints fresh
  trajectories on every run, so a baseline and a candidate share no trajectory at
  all. Tasks scored in only one of the two runs are reported separately instead of
  being quietly dropped: a suite that changed shape must not be readable as a change
  in quality.
* **Both runs must have used the same judge.** Two judges disagree with each other
  roughly as much as two humans do, so a judge swap moves the suite score on its
  own. Comparing across judges would attribute that move to the pull request, with
  full statistical confidence, so it raises instead.

Nothing here calls a model: it is arithmetic over verdicts already stored.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from agentverdict.calibration.stats import VERDICT_SCORES, bootstrap_mean_ci
from agentverdict.models import EvalRun, Judge, JudgeVerdict, Task, Trajectory, utcnow
from agentverdict.schemas import RunComparison, TaskScoreDelta

Outcome = Literal["regression", "improvement", "inconclusive"]


@dataclass(frozen=True)
class _TaskScore:
    """One task's score within one run, and how many trajectories back it."""

    score: float
    trajectories: int


def _plural(count: int) -> str:
    return "" if count == 1 else "s"


def _judge_names(session: Session, judge_ids: Sequence[str]) -> dict[str, str]:
    """Names for the given judge ids, in one query.

    A missing row is not an error here: the caller falls back to the raw id so that
    a mismatch can still be reported by *something* a human recognises.
    """
    rows = session.execute(select(Judge.id, Judge.name).where(Judge.id.in_(set(judge_ids))))
    return {judge_id: name for judge_id, name in rows}


def _scores_by_run(
    session: Session, run_ids: Sequence[str]
) -> tuple[dict[str, dict[str, _TaskScore]], dict[str, int]]:
    """Per-task scores for each run, keyed ``[run_id][task_key]``, plus error counts.

    A verdict is worth ``fail 0.0 / borderline 0.5 / pass 1.0``. Trajectory scores
    are averaged into a task score, so running a task with ``--repeats 3`` averages
    three samples of it rather than giving that task three times the say; task
    scores are averaged into a run score for the same reason one flaky task should
    not dominate a suite.

    Rows with a null verdict record a failed API call rather than a decision, and an
    opinion the judge never formed must not be counted as a ``fail`` — that would
    turn a rate limit into a regression. They are skipped, so a task whose every
    verdict in a run errored has no score in that run at all and drops out of the
    comparison rather than being scored zero.

    Skipping them is right but not harmless: the calls that fail are the long, hard
    transcripts, so a run that errored on its worst tasks scores better than it
    deserves and can hide a real regression. The second return value is the count of
    skipped rows per run, so every surface can say how much was thrown away.
    """
    stmt = (
        select(
            JudgeVerdict.eval_run_id,
            Task.key,
            JudgeVerdict.trajectory_id,
            JudgeVerdict.verdict,
        )
        .join(Trajectory, JudgeVerdict.trajectory_id == Trajectory.id)
        .join(Task, Trajectory.task_id == Task.id)
        .where(JudgeVerdict.eval_run_id.in_(set(run_ids)))
    )

    # run_id -> task_key -> trajectory_id -> verdict scores
    grouped: dict[str, dict[str, dict[str, list[float]]]] = {}
    errors: dict[str, int] = dict.fromkeys(set(run_ids), 0)
    for run_id, task_key, trajectory_id, verdict in session.execute(stmt):
        if verdict not in VERDICT_SCORES:
            errors[run_id] = errors.get(run_id, 0) + 1
            continue
        tasks = grouped.setdefault(run_id, {})
        trajectories = tasks.setdefault(task_key, {})
        trajectories.setdefault(trajectory_id, []).append(VERDICT_SCORES[verdict])

    by_run = {
        run_id: {
            task_key: _TaskScore(
                score=fmean([fmean(scores) for scores in trajectories.values()]),
                trajectories=len(trajectories),
            )
            for task_key, trajectories in tasks.items()
        }
        for run_id, tasks in grouped.items()
    }
    return by_run, errors


def _classify(interval: tuple[float, float] | None) -> Outcome:
    """Read the interval as a three-valued verdict.

    Only an interval lying wholly on one side of zero says anything. One that
    touches zero includes "no change at all" among the outcomes it cannot rule out,
    so it is inconclusive — the boundary is strict on purpose, because the whole
    value of the gate is that it does not block on a coin flip.
    """
    if interval is None:
        return "inconclusive"
    low, high = interval
    if high < 0.0:
        return "regression"
    if low > 0.0:
        return "improvement"
    return "inconclusive"


def _summarize(
    outcome: Outcome,
    *,
    tasks_compared: int,
    base_score: float | None,
    candidate_score: float | None,
    delta: float | None,
    interval: tuple[float, float] | None,
    confidence: float,
    excluded: int,
) -> str:
    """One sentence a reviewer can read in a pull-request comment without training.

    It has to carry the outcome, the size of the change, the range the change could
    plausibly be, and how many tasks that rests on — because the number a reader
    will otherwise act on is the raw delta, which is exactly the number this module
    exists to say is not enough on its own.
    """
    aside = ""
    if excluded:
        was = "was" if excluded == 1 else "were"
        aside = (
            f" ({excluded} task{_plural(excluded)} scored in only one of the two runs "
            f"{was} left out)"
        )

    if base_score is None or candidate_score is None or delta is None or tasks_compared == 0:
        return (
            "Inconclusive: the two runs share no task with a usable verdict, so there is "
            f"nothing to compare and the merge is not blocked{aside}."
        )

    scale = (
        f"{candidate_score:.3f} against the baseline's {base_score:.3f} on the "
        f"{tasks_compared} task{_plural(tasks_compared)} both runs scored, a change of "
        f"{delta:+.3f}"
    )

    if interval is None:
        if tasks_compared < 2:
            reason = (
                f"{tasks_compared} task{_plural(tasks_compared)} "
                f"{'is' if tasks_compared == 1 else 'are'} too few to resample"
            )
        else:
            reason = "the resampling test could not be run at the requested confidence level"
        return (
            f"Inconclusive: the candidate scored {scale}, but {reason}, so a real change "
            f"cannot be told apart from noise and the merge is not blocked{aside}."
        )

    low, high = interval
    band = (
        f"re-drawing those tasks at random kept the change between {low:+.3f} and "
        f"{high:+.3f} {confidence:.0%} of the time"
    )
    if outcome == "regression":
        return (
            f"Regression: the candidate scored {scale}, and {band} — a range entirely below "
            "zero, so the drop is larger than the noise of which tasks happen to be in the "
            f"suite, and the merge is blocked{aside}."
        )
    if outcome == "improvement":
        return (
            f"Improvement: the candidate scored {scale}, and {band} — a range entirely above "
            "zero, so the gain is larger than the noise of which tasks happen to be in the "
            f"suite; the merge is not blocked{aside}."
        )
    return (
        f"Inconclusive: the candidate scored {scale}, but {band} — a range that still "
        "includes zero, so this change cannot be told apart from the noise of which tasks "
        f"happen to be in the suite, and the merge is not blocked{aside}."
    )


def compare_runs(
    session: Session,
    base_run: EvalRun,
    candidate_run: EvalRun,
    *,
    iterations: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> RunComparison:
    """Compare a candidate eval run against its baseline and decide whether to block.

    Both runs are scored per task, paired on task key, and the per-task differences
    are bootstrapped over tasks. ``blocks_merge`` is set only when the resulting
    interval lies entirely below zero; every other result — including a suite too
    small to resample — leaves the merge open.

    Args:
        session: Open session; nothing is written, so a read-only one is fine.
        base_run: The stored baseline, i.e. the run the candidate must not be worse
            than. Ordering matters: deltas are candidate minus base.
        candidate_run: The run produced by the change under review.
        iterations: Resamples behind the interval. Higher here than the 1000 used for
            reporting statistics elsewhere, because this interval is not read, it is
            *thresholded*: the merge blocks on whether its upper bound sits below
            zero. Near that boundary the resampling noise in the bound is the entire
            decision, and a fixed seed makes that noise reproducible without making it
            smaller — the same borderline change would block or pass depending on
            which resamples the seed happened to draw. More iterations shrink it.
        confidence: Coverage of the interval, e.g. ``0.95``. Wider is stricter about
            calling a change real, and therefore less likely to block.
        seed: Resampling seed, so the same two runs always yield the same verdict —
            a gate whose answer changed between re-runs of the same CI job would be
            argued with rather than obeyed.

    Raises:
        ValueError: If the two runs were judged by different judges, or by different
            versions of the judge prompt. Judges disagree with each other about as
            much as two humans do, so their scores are not on a common scale;
            comparing them would report the judge swap as a change in agent quality,
            and report it confidently. An edited rubric does the same thing while
            still naming the same judge.
    """
    names = _judge_names(session, (base_run.judge_id, candidate_run.judge_id))
    base_judge = names.get(base_run.judge_id, base_run.judge_id)
    candidate_judge = names.get(candidate_run.judge_id, candidate_run.judge_id)
    if base_run.judge_id != candidate_run.judge_id:
        raise ValueError(
            f"cannot compare runs scored by different judges: baseline run {base_run.id} was "
            f"judged by {base_judge!r} and candidate run {candidate_run.id} by "
            f"{candidate_judge!r}; their scores are not on a common scale, so any difference "
            "between the runs would measure the judge swap as much as the agent"
        )

    # Same judge, different rubric is the same problem wearing a disguise, and a nastier
    # one: the pull request that edits the judge prompt is exactly the pull request whose
    # score movement is not about the agent. Only refuse when both versions are known --
    # a run recorded before the column existed carries None, and refusing on that would
    # break every gate the moment it upgraded, over a version that was never in conflict,
    # only unrecorded.
    base_prompt = base_run.judge_prompt_version
    candidate_prompt = candidate_run.judge_prompt_version
    if base_prompt is not None and candidate_prompt is not None and base_prompt != candidate_prompt:
        raise ValueError(
            f"cannot compare runs scored by different versions of the judge prompt: baseline "
            f"run {base_run.id} was graded with {base_prompt} and candidate run "
            f"{candidate_run.id} with {candidate_prompt}; the rubric moved between them, so "
            "the difference measures the grader as much as the agent -- record a fresh "
            "baseline under the current prompt"
        )

    scores, judge_errors = _scores_by_run(session, (base_run.id, candidate_run.id))
    base_scores = scores.get(base_run.id, {})
    candidate_scores = scores.get(candidate_run.id, {})

    # Pair on task key: replay creates new trajectories every run, so trajectory ids
    # never line up. Tasks only one run scored are reported, not silently dropped.
    shared = sorted(base_scores.keys() & candidate_scores.keys())
    base_only = sorted(base_scores.keys() - candidate_scores.keys())
    candidate_only = sorted(candidate_scores.keys() - base_scores.keys())

    task_deltas = [
        TaskScoreDelta(
            task_key=task_key,
            base_score=base_scores[task_key].score,
            candidate_score=candidate_scores[task_key].score,
            delta=candidate_scores[task_key].score - base_scores[task_key].score,
            base_trajectories=base_scores[task_key].trajectories,
            candidate_trajectories=candidate_scores[task_key].trajectories,
        )
        for task_key in shared
    ]

    # Built in sorted-key order: the bootstrap is seeded, so a stable input order is
    # what makes the same pair of runs produce the same interval every time.
    deltas = [row.delta for row in task_deltas]
    interval = bootstrap_mean_ci(
        deltas, iterations=iterations, confidence=confidence, seed=seed
    ).interval
    outcome = _classify(interval)

    # Means over the shared tasks only, so the headline delta is exactly the
    # difference between the two numbers printed beside it. Averaging each run over
    # everything it scored would let a task the candidate skipped move the delta.
    base_score = fmean([base_scores[key].score for key in shared]) if shared else None
    candidate_score = fmean([candidate_scores[key].score for key in shared]) if shared else None
    delta = (
        candidate_score - base_score
        if base_score is not None and candidate_score is not None
        else None
    )
    low, high = interval if interval is not None else (None, None)

    return RunComparison(
        base_run_id=base_run.id,
        candidate_run_id=candidate_run.id,
        judge_id=base_run.judge_id,
        judge_name=base_judge,
        # Equal or one-sided by this point; prefer whichever is known so an upgraded
        # baseline still reports the rubric the candidate was actually graded under.
        judge_prompt_version=candidate_prompt or base_prompt,
        tasks_compared=len(shared),
        base_score=base_score,
        candidate_score=candidate_score,
        delta=delta,
        delta_ci_low=low,
        delta_ci_high=high,
        outcome=outcome,
        # The one bit CI acts on. An improvement or an inconclusive result never
        # blocks: this gate exists to catch regressions, not to police churn.
        blocks_merge=outcome == "regression",
        summary=_summarize(
            outcome,
            tasks_compared=len(shared),
            base_score=base_score,
            candidate_score=candidate_score,
            delta=delta,
            interval=interval,
            confidence=confidence,
            excluded=len(base_only) + len(candidate_only),
        ),
        base_only_tasks=base_only,
        candidate_only_tasks=candidate_only,
        base_judge_errors=judge_errors.get(base_run.id, 0),
        candidate_judge_errors=judge_errors.get(candidate_run.id, 0),
        # Worst regressions first: the first row is the one worth opening.
        task_deltas=sorted(task_deltas, key=lambda row: (row.delta, row.task_key)),
        input_tokens=base_run.input_tokens + candidate_run.input_tokens,
        output_tokens=base_run.output_tokens + candidate_run.output_tokens,
        generated_at=utcnow(),
    )
