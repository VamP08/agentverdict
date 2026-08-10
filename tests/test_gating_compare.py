"""The regression gate's arithmetic: what is scored, what is paired, and when it blocks.

Every number below is one a reader can check by hand. A verdict is worth
``fail 0.0 / borderline 0.5 / pass 1.0``; a task scores the mean of its trajectories
in a run; a run scores the mean of its tasks. Verdicts are seeded directly, so
nothing here calls a model or touches the network.

The assertions that matter most are the ones about *not* blocking. A candidate whose
suite average fell, but whose bootstrap interval still contains zero, has to come
back ``inconclusive`` with ``blocks_merge`` false — a gate that fires on noise is
switched off within a week, and a switched-off gate catches nothing. That case is
asserted explicitly rather than left to follow from the arithmetic, and so is its
boundary: an interval whose upper edge lands exactly on zero must not block either.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from agentverdict.gating.compare import compare_runs
from agentverdict.models import EvalRun, Judge, JudgeVerdict, Task, Trajectory
from agentverdict.schemas import RunComparison, TaskScoreDelta

#: Six task keys. Their sorted order is the order ``compare_runs`` feeds the
#: per-task deltas to the seeded bootstrap, so writing suites in this order keeps
#: the intervals below reproducible.
SUITE: tuple[str, ...] = tuple(f"gate-{position}" for position in range(1, 7))


def _suite(*verdicts: str | None) -> dict[str, list[str | None]]:
    """Map the SUITE keys, in order, onto one trajectory each."""
    return {key: [verdict] for key, verdict in zip(SUITE, verdicts, strict=True)}


def _each(verdict: str | None) -> dict[str, list[str | None]]:
    """Every task in SUITE judged the same way."""
    return _suite(*([verdict] * len(SUITE)))


def _by_key(comparison: RunComparison) -> dict[str, TaskScoreDelta]:
    return {row.task_key: row for row in comparison.task_deltas}


def _trajectory_ids(session: Session, run: EvalRun) -> set[str]:
    """Every trajectory this run produced a verdict row for."""
    return set(
        session.scalars(
            select(JudgeVerdict.trajectory_id).where(JudgeVerdict.eval_run_id == run.id)
        )
    )


@pytest.fixture
def judge(make_judge: Callable[..., Judge]) -> Judge:
    return make_judge(name="gate-judge", model="llama-test-model")


@pytest.fixture
def seed_run(
    make_eval_run: Callable[..., EvalRun],
    make_task: Callable[..., Task],
    make_trajectory: Callable[..., Trajectory],
    make_verdict: Callable[..., JudgeVerdict],
) -> Callable[..., EvalRun]:
    """Store an eval run scoring the named tasks with the given verdicts.

    ``verdicts`` maps a task key to one entry per trajectory, where ``None`` records
    a judge call that failed rather than a decision it reached. Tasks are created
    once and reused across calls — that is the only thing a baseline and a candidate
    genuinely share — while every call mints brand-new trajectories, exactly as
    replay does.
    """
    tasks: dict[str, Task] = {}

    def _seed(
        judge: Judge, verdicts: Mapping[str, Sequence[str | None]], **kwargs: Any
    ) -> EvalRun:
        run = make_eval_run(judge, **kwargs)
        for key, per_trajectory in verdicts.items():
            if key not in tasks:
                tasks[key] = make_task(key=key)
            for verdict in per_trajectory:
                make_verdict(judge, make_trajectory(task=tasks[key]), verdict, eval_run=run)
        return run

    return _seed


# --- scoring a suite ----------------------------------------------------------


def test_a_task_scores_the_mean_of_its_trajectories(
    session: Session, judge: Judge, seed_run: Callable[..., EvalRun]
) -> None:
    base = seed_run(judge, {"repeated": ["pass", "fail"], "single": ["pass"]})
    candidate = seed_run(judge, {"repeated": ["pass", "pass"], "single": ["pass"]})

    comparison = compare_runs(session, base, candidate)

    repeated = _by_key(comparison)["repeated"]
    assert repeated.base_score == 0.5  # one pass and one fail, averaged
    assert repeated.base_trajectories == 2
    assert repeated.candidate_score == 1.0
    assert repeated.candidate_trajectories == 2
    assert repeated.delta == 0.5
    # The run score averages tasks, not trajectories: sampling a task twice must
    # not give it twice the say in the suite.
    assert comparison.base_score == 0.75
    assert comparison.candidate_score == 1.0


def test_errored_verdicts_are_skipped_rather_than_scored_as_failures(
    session: Session, judge: Judge, seed_run: Callable[..., EvalRun]
) -> None:
    base = seed_run(judge, {"flaky": ["pass", None], "steady": ["pass"]})
    candidate = seed_run(judge, {"flaky": ["pass"], "steady": ["pass"]})

    comparison = compare_runs(session, base, candidate)

    flaky = _by_key(comparison)["flaky"]
    # A call that never returned is not an opinion: counting it as a fail would
    # score 0.5 here and turn a rate limit into a regression.
    assert flaky.base_score == 1.0
    assert flaky.base_trajectories == 1
    assert comparison.base_score == 1.0
    assert comparison.delta == 0.0
    assert comparison.outcome == "inconclusive"
    assert comparison.blocks_merge is False


def test_a_task_whose_verdicts_all_errored_is_absent_rather_than_zero(
    session: Session, judge: Judge, seed_run: Callable[..., EvalRun]
) -> None:
    base = seed_run(judge, {"scored": ["pass"], "steady": ["pass"], "broken": [None, None]})
    candidate = seed_run(judge, {"scored": ["pass"], "steady": ["pass"], "broken": ["fail"]})

    comparison = compare_runs(session, base, candidate)

    # The baseline has no opinion at all about "broken", so it pairs with nothing.
    assert comparison.candidate_only_tasks == ["broken"]
    assert comparison.base_only_tasks == []
    assert comparison.tasks_compared == 2
    assert [row.task_key for row in comparison.task_deltas] == ["scored", "steady"]
    # Scoring the all-errored task 0.0 would drag the baseline to 2/3 and report
    # the candidate as an improvement over a run that simply failed to complete.
    assert comparison.base_score == 1.0
    assert comparison.candidate_score == 1.0
    assert comparison.delta == 0.0
    assert comparison.outcome == "inconclusive"


def test_re_judging_one_trajectory_does_not_give_its_task_extra_weight(
    session: Session,
    judge: Judge,
    seed_run: Callable[..., EvalRun],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    base = seed_run(judge, {"steady": ["pass"], "rejudged": ["pass"]})
    candidate = seed_run(judge, {"steady": ["pass"], "rejudged": ["fail"]})

    # A second verdict for a trajectory the same run already scored: it averages
    # into that trajectory rather than counting as another sample of the task.
    rejudged = session.scalar(
        select(Trajectory)
        .join(JudgeVerdict, JudgeVerdict.trajectory_id == Trajectory.id)
        .join(Task, Trajectory.task_id == Task.id)
        .where(JudgeVerdict.eval_run_id == candidate.id, Task.key == "rejudged")
    )
    assert rejudged is not None
    make_verdict(judge, rejudged, "pass", eval_run=candidate)

    comparison = compare_runs(session, base, candidate)

    row = _by_key(comparison)["rejudged"]
    assert row.candidate_trajectories == 1  # one trajectory, two verdict rows
    assert row.candidate_score == 0.5  # fail and pass, averaged within it
    assert comparison.candidate_score == 0.75


# --- pairing by task ----------------------------------------------------------


def test_runs_pair_by_task_even_though_they_share_no_trajectory(
    session: Session, judge: Judge, seed_run: Callable[..., EvalRun]
) -> None:
    base = seed_run(judge, _each("fail"))
    candidate = seed_run(judge, _each("pass"))

    # The premise: replay records fresh trajectories every time, so the two runs
    # have no row in common. Pairing on trajectory id would compare nothing.
    base_trajectories = _trajectory_ids(session, base)
    candidate_trajectories = _trajectory_ids(session, candidate)
    assert len(base_trajectories) == len(SUITE)
    assert len(candidate_trajectories) == len(SUITE)
    assert base_trajectories & candidate_trajectories == set()

    comparison = compare_runs(session, base, candidate)

    assert comparison.tasks_compared == len(SUITE)
    assert [row.task_key for row in comparison.task_deltas] == sorted(SUITE)
    assert comparison.base_only_tasks == []
    assert comparison.candidate_only_tasks == []


def test_tasks_only_one_run_scored_are_named_and_left_out(
    session: Session, judge: Judge, seed_run: Callable[..., EvalRun]
) -> None:
    base = seed_run(judge, {"shared-1": ["pass"], "shared-2": ["fail"], "retired": ["fail"]})
    candidate = seed_run(judge, {"shared-1": ["pass"], "shared-2": ["fail"], "added": ["pass"]})

    comparison = compare_runs(session, base, candidate)

    assert comparison.base_only_tasks == ["retired"]
    assert comparison.candidate_only_tasks == ["added"]
    assert comparison.tasks_compared == 2
    assert [row.task_key for row in comparison.task_deltas] == ["shared-1", "shared-2"]
    # Neither run changed on the tasks they share. Averaging the unshared pair in
    # would read 1/3 against 2/3 and invent an improvement out of a suite edit.
    assert comparison.base_score == 0.5
    assert comparison.candidate_score == 0.5
    assert comparison.delta == 0.0
    assert comparison.outcome == "inconclusive"
    assert "scored in only one of the two runs" in comparison.summary


def test_suite_scores_and_delta_cover_the_shared_tasks_only(
    session: Session, judge: Judge, seed_run: Callable[..., EvalRun]
) -> None:
    base = seed_run(judge, {"shared-1": ["pass"], "shared-2": ["fail"], "retired": ["fail"]})
    candidate = seed_run(judge, {"shared-1": ["pass"], "shared-2": ["pass"], "added": ["fail"]})

    comparison = compare_runs(session, base, candidate)

    assert comparison.base_score == 0.5  # (1.0 + 0.0) / 2, with "retired" left out
    assert comparison.candidate_score == 1.0  # (1.0 + 1.0) / 2, with "added" left out
    assert comparison.base_score != pytest.approx(1 / 3)  # what all three would give
    assert comparison.candidate_score != pytest.approx(2 / 3)
    # The headline delta is exactly the difference between the two numbers printed
    # beside it, so a reader can check the report by subtracting.
    assert comparison.delta == comparison.candidate_score - comparison.base_score
    assert comparison.delta == 0.5


def test_verdicts_from_other_runs_do_not_leak_in(
    session: Session, judge: Judge, seed_run: Callable[..., EvalRun]
) -> None:
    base = seed_run(judge, _each("pass"))
    candidate = seed_run(judge, _each("pass"))
    # A third run over the same tasks, judged far worse. It is not one of the two
    # runs being compared, so it must not touch either score.
    noise = seed_run(judge, _each("fail"))

    comparison = compare_runs(session, base, candidate)

    assert comparison.tasks_compared == len(SUITE)
    assert comparison.base_score == 1.0
    assert comparison.candidate_score == 1.0
    assert comparison.delta == 0.0
    assert comparison.blocks_merge is False
    assert noise.id not in (comparison.base_run_id, comparison.candidate_run_id)


# --- the three outcomes -------------------------------------------------------


def test_an_across_the_board_drop_is_a_regression_that_blocks(
    session: Session, judge: Judge, seed_run: Callable[..., EvalRun]
) -> None:
    base = seed_run(judge, _each("pass"))
    candidate = seed_run(judge, _each("borderline"))

    comparison = compare_runs(session, base, candidate)

    assert comparison.base_score == 1.0
    assert comparison.candidate_score == 0.5
    assert comparison.delta == -0.5
    # Every task moved by exactly -0.5, so every resample of the tasks does too:
    # the interval collapses onto the drop and sits entirely below zero.
    assert comparison.delta_ci_low == -0.5
    assert comparison.delta_ci_high == -0.5
    assert comparison.outcome == "regression"
    assert comparison.blocks_merge is True
    assert comparison.summary.startswith("Regression:")
    assert "the merge is blocked" in comparison.summary
    assert "not blocked" not in comparison.summary


def test_a_drop_of_uneven_size_still_regresses(
    session: Session, judge: Judge, seed_run: Callable[..., EvalRun]
) -> None:
    base = seed_run(judge, _each("pass"))
    # Per task: -1.0, -0.5, -0.5, -1.0, -0.5, -1.0. Nothing improved, so however
    # the tasks are re-drawn the average stays negative.
    candidate = seed_run(
        judge, _suite("fail", "borderline", "borderline", "fail", "borderline", "fail")
    )

    comparison = compare_runs(session, base, candidate)

    assert comparison.candidate_score == 0.25
    assert comparison.delta == -0.75
    assert comparison.delta_ci_low is not None
    assert comparison.delta_ci_high is not None
    assert comparison.delta_ci_low <= comparison.delta_ci_high < 0.0
    # A real interval, not the degenerate one a constant drop produces.
    assert comparison.delta_ci_low < comparison.delta_ci_high
    assert comparison.outcome == "regression"
    assert comparison.blocks_merge is True


def test_an_across_the_board_gain_is_an_improvement_that_never_blocks(
    session: Session, judge: Judge, seed_run: Callable[..., EvalRun]
) -> None:
    base = seed_run(judge, _each("fail"))
    candidate = seed_run(judge, _each("pass"))

    comparison = compare_runs(session, base, candidate)

    assert comparison.base_score == 0.0
    assert comparison.candidate_score == 1.0
    assert comparison.delta == 1.0
    assert comparison.delta_ci_low == 1.0
    assert comparison.delta_ci_high == 1.0
    assert comparison.outcome == "improvement"
    # This gate exists to catch regressions, not to police change of any kind.
    assert comparison.blocks_merge is False
    assert comparison.summary.startswith("Improvement:")
    assert "the merge is not blocked" in comparison.summary


def test_deltas_straddling_zero_are_inconclusive_even_though_the_mean_fell(
    session: Session, judge: Judge, seed_run: Callable[..., EvalRun]
) -> None:
    """Three tasks got worse, two got better, one held: the suite average drops.

    This is the case the whole design exists for. A gate that put two means side by
    side would block the pull request here — but re-drawing the six tasks at random
    turns the drop into a gain often enough that the interval still contains zero,
    so nothing has been measured and nothing may be blocked.
    """
    base = seed_run(judge, _suite("pass", "pass", "pass", "fail", "fail", "pass"))
    candidate = seed_run(judge, _suite("fail", "fail", "fail", "pass", "pass", "pass"))

    comparison = compare_runs(session, base, candidate)

    assert comparison.base_score == pytest.approx(4 / 6)
    assert comparison.candidate_score == 0.5
    assert comparison.delta is not None
    assert comparison.delta < 0.0  # the headline number really did fall
    assert comparison.delta_ci_low is not None
    assert comparison.delta_ci_high is not None
    assert comparison.delta_ci_low < 0.0 < comparison.delta_ci_high
    assert comparison.outcome == "inconclusive"
    assert comparison.blocks_merge is False
    assert comparison.summary.startswith("Inconclusive:")
    assert "still includes zero" in comparison.summary
    assert "the merge is not blocked" in comparison.summary

    # And it is not an artifact of one lucky seed: this suite cannot tell the two
    # runs apart however the resampling is drawn.
    for seed in (1, 7, 42):
        assert compare_runs(session, base, candidate, seed=seed).outcome == "inconclusive"
        assert compare_runs(session, base, candidate, seed=seed).blocks_merge is False


def test_an_interval_that_only_touches_zero_does_not_block(
    session: Session, judge: Judge, seed_run: Callable[..., EvalRun]
) -> None:
    """One task collapsed, the other held — and the interval's top edge is zero.

    ``regression`` needs the whole interval *below* zero. One that reaches zero
    still admits "nothing changed" among the outcomes it cannot rule out, so the
    boundary is strict even though the mean fell by half a verdict.
    """
    base = seed_run(judge, {"dropped": ["pass"], "held": ["pass"]})
    candidate = seed_run(judge, {"dropped": ["fail"], "held": ["pass"]})

    comparison = compare_runs(session, base, candidate)

    assert comparison.delta == -0.5
    assert comparison.delta_ci_low == -1.0
    assert comparison.delta_ci_high == 0.0
    assert comparison.outcome == "inconclusive"
    assert comparison.blocks_merge is False


def test_two_shared_tasks_are_enough_to_resample(
    session: Session, judge: Judge, seed_run: Callable[..., EvalRun]
) -> None:
    """The smallest suite the bootstrap can work on still reaches a verdict."""
    base = seed_run(judge, {"first": ["pass"], "second": ["pass"]})
    candidate = seed_run(judge, {"first": ["fail"], "second": ["fail"]})

    comparison = compare_runs(session, base, candidate)

    assert comparison.tasks_compared == 2
    assert (comparison.delta_ci_low, comparison.delta_ci_high) == (-1.0, -1.0)
    assert comparison.outcome == "regression"
    assert comparison.blocks_merge is True


# --- too little data ----------------------------------------------------------


def test_one_shared_task_yields_no_interval_and_no_verdict(
    session: Session, judge: Judge, seed_run: Callable[..., EvalRun]
) -> None:
    base = seed_run(judge, {"only": ["pass"], "base-extra": ["pass"]})
    candidate = seed_run(judge, {"only": ["fail"], "candidate-extra": ["pass"]})

    comparison = compare_runs(session, base, candidate)

    assert comparison.tasks_compared == 1
    assert comparison.base_score == 1.0
    assert comparison.candidate_score == 0.0
    assert comparison.delta == -1.0  # the drop is still reported...
    assert comparison.delta_ci_low is None  # ...but one task resamples to itself
    assert comparison.delta_ci_high is None
    assert comparison.outcome == "inconclusive"
    assert comparison.blocks_merge is False
    assert "too few to resample" in comparison.summary


def test_runs_with_no_task_in_common_have_nothing_to_compare(
    session: Session, judge: Judge, seed_run: Callable[..., EvalRun]
) -> None:
    base = seed_run(judge, {"old-1": ["pass"], "old-2": ["pass"]})
    candidate = seed_run(judge, {"new-1": ["fail"], "new-2": ["fail"]})

    comparison = compare_runs(session, base, candidate)

    assert comparison.tasks_compared == 0
    assert comparison.base_score is None  # not 0.0: nothing was measured
    assert comparison.candidate_score is None
    assert comparison.delta is None
    assert comparison.delta_ci_low is None
    assert comparison.delta_ci_high is None
    assert comparison.task_deltas == []
    assert comparison.base_only_tasks == ["old-1", "old-2"]
    assert comparison.candidate_only_tasks == ["new-1", "new-2"]
    assert comparison.outcome == "inconclusive"
    assert comparison.blocks_merge is False
    assert "share no task with a usable verdict" in comparison.summary


# --- refusals, ordering, reproducibility --------------------------------------


def test_comparing_runs_from_two_judges_raises_naming_both(
    session: Session,
    judge: Judge,
    make_judge: Callable[..., Judge],
    seed_run: Callable[..., EvalRun],
) -> None:
    rival = make_judge(name="rival-judge", model="llama-test-model")
    base = seed_run(judge, _each("pass"))
    candidate = seed_run(rival, _each("fail"))

    with pytest.raises(ValueError) as raised:
        compare_runs(session, base, candidate)

    # Two judges disagree with each other about as much as two humans do, so the
    # message has to name them: "the comparison failed" is not actionable.
    message = str(raised.value)
    assert "gate-judge" in message
    assert "rival-judge" in message
    assert base.id in message
    assert candidate.id in message


def test_comparing_runs_graded_by_two_rubrics_raises_naming_both(
    session: Session, judge: Judge, seed_run: Callable[..., EvalRun]
) -> None:
    # Same judge, edited prompt. This is the disguised version of the mismatch above,
    # and the more dangerous one: the pull request that rewrites the rubric is exactly
    # the pull request whose score movement has nothing to do with the agent.
    base = seed_run(judge, _each("pass"), judge_prompt_version="aaaaaaaaaaaa")
    candidate = seed_run(judge, _each("fail"), judge_prompt_version="bbbbbbbbbbbb")

    with pytest.raises(ValueError) as raised:
        compare_runs(session, base, candidate)

    message = str(raised.value)
    assert "aaaaaaaaaaaa" in message
    assert "bbbbbbbbbbbb" in message
    # The reader has to be told what to do about it, or they will assume the gate broke.
    assert "fresh baseline" in message


def test_a_run_without_a_recorded_rubric_still_compares(
    session: Session, judge: Judge, seed_run: Callable[..., EvalRun]
) -> None:
    # Runs stored before the version column existed carry None. Refusing on that would
    # break every gate the moment it upgraded, over a rubric that was never in conflict
    # -- only unrecorded. The comparison proceeds and reports the version it does know.
    base = seed_run(judge, _each("pass"), judge_prompt_version=None)
    candidate = seed_run(judge, _each("pass"), judge_prompt_version="bbbbbbbbbbbb")

    comparison = compare_runs(session, base, candidate)

    assert comparison.tasks_compared == len(SUITE)
    assert comparison.judge_prompt_version == "bbbbbbbbbbbb"


def test_matching_rubrics_are_reported_once(
    session: Session, judge: Judge, seed_run: Callable[..., EvalRun]
) -> None:
    base = seed_run(judge, _each("pass"), judge_prompt_version="cccccccccccc")
    candidate = seed_run(judge, _each("pass"), judge_prompt_version="cccccccccccc")

    assert compare_runs(session, base, candidate).judge_prompt_version == "cccccccccccc"


def test_task_deltas_are_ordered_worst_first(
    session: Session, judge: Judge, seed_run: Callable[..., EvalRun]
) -> None:
    base = seed_run(
        judge, {"alpha": ["pass"], "bravo": ["pass"], "charlie": ["pass"], "echo": ["pass"]}
    )
    candidate = seed_run(
        judge,
        {"alpha": ["fail"], "bravo": ["borderline"], "charlie": ["pass"], "echo": ["borderline"]},
    )

    comparison = compare_runs(session, base, candidate)

    # The biggest drop first, ties broken by key so the order never wobbles.
    assert [row.task_key for row in comparison.task_deltas] == [
        "alpha",
        "bravo",
        "echo",
        "charlie",
    ]
    assert [row.delta for row in comparison.task_deltas] == [-1.0, -0.5, -0.5, 0.0]


def test_the_same_two_runs_always_compare_the_same_way(
    session: Session, judge: Judge, seed_run: Callable[..., EvalRun]
) -> None:
    """A gate whose answer changed between re-runs would be argued with, not obeyed."""
    base = seed_run(judge, _suite("pass", "pass", "pass", "fail", "fail", "pass"))
    candidate = seed_run(judge, _suite("fail", "fail", "fail", "pass", "pass", "pass"))

    first = compare_runs(session, base, candidate)
    second = compare_runs(session, base, candidate)

    assert (first.delta_ci_low, first.delta_ci_high) == (second.delta_ci_low, second.delta_ci_high)
    assert first.outcome == second.outcome
    assert first.blocks_merge == second.blocks_merge
    assert first.summary == second.summary
    assert [row.model_dump() for row in first.task_deltas] == [
        row.model_dump() for row in second.task_deltas
    ]


def test_the_comparison_records_the_judge_and_the_token_cost(
    session: Session, judge: Judge, seed_run: Callable[..., EvalRun]
) -> None:
    base = seed_run(judge, _each("pass"), input_tokens=1200, output_tokens=300)
    candidate = seed_run(judge, _each("pass"), input_tokens=800, output_tokens=150)

    comparison = compare_runs(session, base, candidate)

    assert comparison.base_run_id == base.id
    assert comparison.candidate_run_id == candidate.id
    # A gate is only as trustworthy as the judge behind it, so the report names it.
    assert comparison.judge_id == judge.id
    assert comparison.judge_name == "gate-judge"
    assert comparison.input_tokens == 2000
    assert comparison.output_tokens == 450
    assert comparison.generated_at.tzinfo is not None
