"""Calibration report tests: what gets compared, what gets excluded, and why.

The report is the part that decides which rows the statistics ever see, so the
assertions below are about selection rather than arithmetic: the human majority
rule, the most-recent-verdict rule, error and tie exclusion, scoping, the
inter-annotator ceiling, and the disagreement drill-down. Everything is seeded
directly into the test session — nothing here touches the network or a model.
"""

from collections.abc import Callable
from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session

from agentverdict.calibration.report import build_calibration_report, human_majority
from agentverdict.judging.prompts import PROMPT_VERSION
from agentverdict.models import EvalRun, HumanLabel, Judge, JudgeVerdict, Rubric, Task, Trajectory
from agentverdict.rubric import RUBRIC_NAME, RUBRIC_VERSION, ensure_rubric

# Fixed timestamps make "most recent" a property of the data, not of clock speed.
EARLIER = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
LATER = datetime(2026, 1, 2, 9, 0, tzinfo=UTC)


def _detached_labels(*verdicts: str) -> list[HumanLabel]:
    """Unsaved labels from distinct annotators, for unit-testing the majority rule."""
    return [
        HumanLabel(trajectory_id="t", annotator=f"annotator-{index}", verdict=verdict)
        for index, verdict in enumerate(verdicts)
    ]


# --- the majority rule -------------------------------------------------------


def test_human_majority_needs_a_clear_winner() -> None:
    assert human_majority([]) is None
    assert human_majority(_detached_labels("pass")) == "pass"
    assert human_majority(_detached_labels("pass", "pass", "fail")) == "pass"
    # A plurality is enough as long as it is unambiguous.
    assert human_majority(_detached_labels("pass", "pass", "fail", "borderline")) == "pass"
    # Level at the top: no majority to hold the judge to, whichever way it is read.
    assert human_majority(_detached_labels("pass", "fail")) is None
    assert human_majority(_detached_labels("pass", "fail", "borderline")) is None
    assert human_majority(_detached_labels("pass", "pass", "fail", "fail", "borderline")) is None


def test_ties_are_excluded_and_counted_while_majorities_are_used(
    session: Session,
    make_judge: Callable[..., Judge],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    judge = make_judge()
    clear = make_trajectory()  # 2-1 for pass
    tied = make_trajectory()  # 1-1, no majority
    solo = make_trajectory()  # a single annotator is a majority of one

    make_label(clear, "alice", "pass")
    make_label(clear, "bob", "pass")
    make_label(clear, "carol", "fail")
    make_label(tied, "alice", "pass")
    make_label(tied, "bob", "fail")
    make_label(solo, "alice", "borderline")

    # The judge fails `clear`: if the majority were read as "fail" (the last
    # annotator) this would score as agreement instead of a 2-step miss.
    make_verdict(judge, clear, "fail")
    make_verdict(judge, tied, "pass")
    make_verdict(judge, solo, "borderline")

    report = build_calibration_report(session, judge)

    assert report.judged_trajectories == 3
    assert report.labeled_trajectories == 3
    assert report.ties_excluded == 1
    assert report.compared == 2  # clear + solo; the tied one is dropped
    assert report.agreement.n == 2
    assert report.agreement.observed_agreement == 0.5  # solo matched, clear did not

    assert len(report.disagreements) == 1
    miss = report.disagreements[0]
    assert miss.trajectory_id == clear.id
    assert miss.human_verdict == "pass"  # the 2-1 majority, not the dissenting "fail"
    assert miss.judge_verdict == "fail"
    assert miss.distance == 2
    assert miss.annotators == ["alice", "bob", "carol"]
    assert tied.id not in {row.trajectory_id for row in report.disagreements}


def test_labels_outside_the_verdict_vocabulary_are_not_compared(
    session: Session,
    make_judge: Callable[..., Judge],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    judge = make_judge()
    usable = make_trajectory()
    odd = make_trajectory()
    make_label(usable, "alice", "pass")
    make_label(odd, "alice", "unsure")  # not one of fail/borderline/pass
    make_verdict(judge, usable, "pass")
    make_verdict(judge, odd, "pass")

    report = build_calibration_report(session, judge)

    assert report.labeled_trajectories == 2  # it is still a labeled trajectory
    assert report.judged_trajectories == 2
    assert report.compared == 1  # but there is no comparable verdict on it
    assert report.agreement.n == 1
    assert report.disagreements == []


# --- which judge verdict counts ----------------------------------------------


def test_only_the_most_recent_judge_verdict_is_compared(
    session: Session,
    make_judge: Callable[..., Judge],
    make_eval_run: Callable[..., EvalRun],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    judge = make_judge()
    trajectory = make_trajectory()
    make_label(trajectory, "alice", "pass")

    # Insert the newer verdict first, so "last row written wins" would pick the
    # stale "fail" and report a disagreement that no longer exists.
    make_verdict(
        judge,
        trajectory,
        "pass",
        eval_run=make_eval_run(judge),
        created_at=LATER,
        rationale="Refund issued after verifying the order.",
    )
    make_verdict(
        judge,
        trajectory,
        "fail",
        eval_run=make_eval_run(judge),
        created_at=EARLIER,
        rationale="Stale verdict from the old prompt.",
    )

    report = build_calibration_report(session, judge)

    assert report.judged_trajectories == 1  # one trajectory, not two verdicts
    assert report.compared == 1
    assert report.agreement.matrix["pass"]["pass"] == 1
    assert report.agreement.matrix["pass"]["fail"] == 0
    assert report.agreement.observed_agreement == 1.0
    assert report.disagreements == []


def test_errored_verdicts_are_excluded_and_counted(
    session: Session,
    make_judge: Callable[..., Judge],
    make_eval_run: Callable[..., EvalRun],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    judge = make_judge()
    clean = make_trajectory()
    retried = make_trajectory()
    make_label(clean, "alice", "pass")
    make_label(retried, "alice", "fail")

    make_verdict(judge, clean, "pass", created_at=EARLIER)
    # An older real decision followed by a newer recorded API failure: the
    # failure is not a decision, so it must not displace the decision.
    make_verdict(judge, retried, "fail", eval_run=make_eval_run(judge), created_at=EARLIER)
    make_verdict(
        judge,
        retried,
        None,
        eval_run=make_eval_run(judge),
        created_at=LATER,
        error="HTTP 500 from the judge model",
    )

    report = build_calibration_report(session, judge)

    assert report.judge_errors == 1
    assert report.judged_trajectories == 2
    assert report.compared == 2
    assert report.agreement.n == 2
    assert report.agreement.observed_agreement == 1.0
    assert report.disagreements == []


def test_a_trajectory_with_only_an_errored_verdict_is_not_judged(
    session: Session,
    make_judge: Callable[..., Judge],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    judge = make_judge()
    trajectory = make_trajectory()
    make_label(trajectory, "alice", "pass")
    make_verdict(judge, trajectory, None, error="connection reset")

    report = build_calibration_report(session, judge)

    assert report.judge_errors == 1
    assert report.judged_trajectories == 0
    assert report.labeled_trajectories == 1
    assert report.compared == 0
    assert report.agreement.cohens_kappa is None


def test_another_judges_verdicts_are_ignored(
    session: Session,
    make_judge: Callable[..., Judge],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    ours = make_judge(name="ours")
    theirs = make_judge(name="theirs")
    trajectory = make_trajectory()
    make_label(trajectory, "alice", "pass")
    make_verdict(theirs, trajectory, "fail", created_at=LATER)
    make_verdict(ours, trajectory, "pass", created_at=EARLIER)

    report = build_calibration_report(session, ours)

    assert report.judge_id == ours.id
    assert report.judged_trajectories == 1
    assert report.judge_errors == 0
    assert report.compared == 1
    assert report.agreement.observed_agreement == 1.0


# --- scoping -----------------------------------------------------------------


def test_eval_run_scope_selects_that_runs_verdicts(
    session: Session,
    make_judge: Callable[..., Judge],
    make_eval_run: Callable[..., EvalRun],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    judge = make_judge()
    trajectory = make_trajectory()
    unjudged = make_trajectory()
    make_label(trajectory, "alice", "pass")
    make_label(unjudged, "alice", "pass")

    first_run = make_eval_run(judge)
    second_run = make_eval_run(judge)
    make_verdict(judge, trajectory, "pass", eval_run=first_run, created_at=EARLIER)
    make_verdict(judge, trajectory, "fail", eval_run=second_run, created_at=LATER)

    unscoped = build_calibration_report(session, judge)
    assert unscoped.eval_run_id is None
    assert unscoped.compared == 1
    assert unscoped.agreement.matrix["pass"]["fail"] == 1  # the newer run wins

    scoped_first = build_calibration_report(session, judge, eval_run_id=first_run.id)
    assert scoped_first.eval_run_id == first_run.id
    assert scoped_first.compared == 1
    assert scoped_first.agreement.matrix["pass"]["pass"] == 1
    assert scoped_first.agreement.matrix["pass"]["fail"] == 0
    assert scoped_first.disagreements == []

    scoped_second = build_calibration_report(session, judge, eval_run_id=second_run.id)
    assert scoped_second.compared == 1
    assert [row.judge_verdict for row in scoped_second.disagreements] == ["fail"]

    # The run scope narrows the judge's side only: human labels belong to the
    # dataset, not to a run, so the labeled count is the same in every scope.
    assert unscoped.labeled_trajectories == 2
    assert scoped_first.labeled_trajectories == 2
    assert scoped_second.labeled_trajectories == 2


def test_unknown_eval_run_scope_compares_nothing(
    session: Session,
    make_judge: Callable[..., Judge],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    judge = make_judge()
    trajectory = make_trajectory()
    make_label(trajectory, "alice", "pass")
    make_verdict(judge, trajectory, "pass")

    report = build_calibration_report(session, judge, eval_run_id="no-such-run")

    assert report.judged_trajectories == 0
    assert report.compared == 0
    assert report.agreement.cohens_kappa is None


def test_task_key_scopes_both_sides_of_the_comparison(
    session: Session,
    make_judge: Callable[..., Judge],
    make_task: Callable[..., Task],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    judge = make_judge()
    refunds = make_task(key="refund-simple-01")
    orders = make_task(key="order-status-01")
    agreeing = make_trajectory(task=refunds)
    disagreeing = make_trajectory(task=orders)
    make_label(agreeing, "alice", "pass")
    make_label(disagreeing, "alice", "pass")
    make_verdict(judge, agreeing, "pass")
    make_verdict(judge, disagreeing, "fail")

    whole = build_calibration_report(session, judge)
    assert whole.task_key is None
    assert (whole.judged_trajectories, whole.labeled_trajectories, whole.compared) == (2, 2, 2)
    assert whole.agreement.observed_agreement == 0.5

    only_refunds = build_calibration_report(session, judge, task_key="refund-simple-01")
    assert only_refunds.task_key == "refund-simple-01"
    assert (only_refunds.judged_trajectories, only_refunds.labeled_trajectories) == (1, 1)
    assert only_refunds.compared == 1
    assert only_refunds.agreement.observed_agreement == 1.0
    assert only_refunds.disagreements == []

    only_orders = build_calibration_report(session, judge, task_key="order-status-01")
    assert only_orders.compared == 1
    assert only_orders.agreement.observed_agreement == 0.0
    assert [row.task_key for row in only_orders.disagreements] == ["order-status-01"]
    assert [row.trajectory_id for row in only_orders.disagreements] == [disagreeing.id]

    missing = build_calibration_report(session, judge, task_key="no-such-task")
    assert missing.judged_trajectories == 0
    assert missing.labeled_trajectories == 0
    assert missing.compared == 0


# --- counts and headline statistics ------------------------------------------


def test_counts_and_kappa_match_what_was_seeded(
    session: Session,
    make_judge: Callable[..., Judge],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    judge = make_judge(name="groq-70b", model="llama-test-model")

    # Four comparable trajectories: 2 pass/pass, 1 fail/fail, 1 fail/pass.
    # p_o = 3/4 = 0.75. Human marginals pass 0.5 / fail 0.5; judge marginals
    # pass 0.75 / fail 0.25. p_e = 0.5*0.75 + 0.5*0.25 = 0.5.
    # kappa = (0.75 - 0.5) / (1 - 0.5) = 0.5.
    comparable = [("pass", "pass"), ("pass", "pass"), ("fail", "fail"), ("fail", "pass")]
    for index, (human, judged) in enumerate(comparable):
        trajectory = make_trajectory()
        make_label(trajectory, f"annotator-{index}", human)
        make_verdict(judge, trajectory, judged, rationale=f"decision {index}")

    labeled_only = make_trajectory()
    make_label(labeled_only, "alice", "pass")
    judged_only = make_trajectory()
    make_verdict(judge, judged_only, "pass")
    failed = make_trajectory()
    make_verdict(judge, failed, None, error="timeout")

    report = build_calibration_report(session, judge)

    assert report.judge_id == judge.id
    assert report.judge_name == "groq-70b"
    assert report.judge_model == "llama-test-model"
    assert report.judged_trajectories == 5  # 4 comparable + 1 judged-only
    assert report.labeled_trajectories == 5  # 4 comparable + 1 labeled-only
    assert report.compared == 4
    assert report.judge_errors == 1
    assert report.ties_excluded == 0
    assert report.agreement.n == report.compared

    assert report.agreement.observed_agreement == pytest.approx(0.75)
    assert report.agreement.cohens_kappa == pytest.approx(0.5)
    assert report.agreement.interpretation == "moderate"
    # Four items cannot support a bootstrap interval: a large share of resamples
    # draw a rater who never varies, leaving kappa undefined for them. Reporting
    # the percentiles of the survivors would be a confident number computed from a
    # filtered, flattering subset, so the interval is withheld and the usable
    # count is published instead.
    assert report.agreement.kappa_ci_low is None
    assert report.agreement.kappa_ci_high is None
    assert report.agreement.bootstrap_iterations == 1000
    assert 0 < report.agreement.bootstrap_usable < 1000

    # matrix[human][judge]: the one miss is a human "fail" the judge passed.
    assert report.agreement.matrix["fail"]["pass"] == 1
    assert report.agreement.matrix["pass"]["fail"] == 0
    assert report.agreement.matrix["pass"]["pass"] == 2
    assert report.agreement.matrix["fail"]["fail"] == 1

    per_class = {row.label: row for row in report.agreement.per_class}
    assert per_class["pass"].support == 2  # humans said pass twice
    assert per_class["pass"].predicted == 3  # the judge said pass three times
    assert per_class["borderline"].support == 0
    assert per_class["borderline"].precision is None


def test_report_is_reproducible_for_a_fixed_seed(
    session: Session,
    make_judge: Callable[..., Judge],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    judge = make_judge()
    for index, (human, judged) in enumerate(
        [("pass", "pass"), ("pass", "fail"), ("fail", "fail"), ("borderline", "pass")]
    ):
        trajectory = make_trajectory()
        make_label(trajectory, f"annotator-{index}", human)
        make_verdict(judge, trajectory, judged)

    first = build_calibration_report(session, judge, seed=11)
    second = build_calibration_report(session, judge, seed=11)

    assert first.agreement.kappa_ci_low == second.agreement.kappa_ci_low
    assert first.agreement.kappa_ci_high == second.agreement.kappa_ci_high
    assert [row.trajectory_id for row in first.disagreements] == [
        row.trajectory_id for row in second.disagreements
    ]


# --- disagreement drill-down -------------------------------------------------


def test_disagreements_are_worst_first_and_carry_the_rationale(
    session: Session,
    make_judge: Callable[..., Judge],
    make_task: Callable[..., Task],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    judge = make_judge()
    # Task keys are the documented tie-break within a distance band, so they are
    # chosen to be out of alphabetical order relative to the seeding order.
    seeded = [
        ("zz-flip", "pass", "fail", "Called a good refund a failure."),
        ("mm-near", "borderline", "pass", "Read a hedge as a clean pass."),
        ("aa-flip", "fail", "pass", "Missed that the refund never went through."),
        ("bb-near", "fail", "borderline", "Softened a clear failure."),
    ]
    trajectories: dict[str, Trajectory] = {}
    for key, human, judged, rationale in seeded:
        trajectory = make_trajectory(task=make_task(key=key))
        trajectories[key] = trajectory
        make_label(trajectory, "bob", human)
        make_label(trajectory, "alice", human)
        make_verdict(judge, trajectory, judged, rationale=rationale)

    matching = make_trajectory(task=make_task(key="cc-match"))
    make_label(matching, "alice", "pass")
    make_verdict(judge, matching, "pass", rationale="Agreed with the humans.")

    report = build_calibration_report(session, judge)

    assert report.compared == 5
    # Both pass/fail flips (distance 2) come before both near misses (distance 1),
    # and within each band the task key breaks the tie.
    assert [row.task_key for row in report.disagreements] == [
        "aa-flip",
        "zz-flip",
        "bb-near",
        "mm-near",
    ]
    assert [row.distance for row in report.disagreements] == [2, 2, 1, 1]
    assert matching.id not in {row.trajectory_id for row in report.disagreements}

    worst = report.disagreements[0]
    assert worst.trajectory_id == trajectories["aa-flip"].id
    assert worst.human_verdict == "fail"
    assert worst.judge_verdict == "pass"
    assert worst.judge_rationale == "Missed that the refund never went through."
    assert worst.annotators == ["alice", "bob"]  # sorted, not insertion order


def test_disagreements_respect_max_disagreements(
    session: Session,
    make_judge: Callable[..., Judge],
    make_task: Callable[..., Task],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    judge = make_judge()
    for key, human, judged in [
        ("near-01", "borderline", "pass"),
        ("flip-01", "pass", "fail"),
        ("near-02", "fail", "borderline"),
    ]:
        trajectory = make_trajectory(task=make_task(key=key))
        make_label(trajectory, "alice", human)
        make_verdict(judge, trajectory, judged)

    capped = build_calibration_report(session, judge, max_disagreements=1)
    assert len(capped.disagreements) == 1
    assert capped.disagreements[0].task_key == "flip-01"  # the cap keeps the worst
    assert capped.compared == 3  # the cap never changes the statistics

    none_shown = build_calibration_report(session, judge, max_disagreements=0)
    assert none_shown.disagreements == []
    assert none_shown.compared == 3
    assert none_shown.agreement.n == 3

    everything = build_calibration_report(session, judge, max_disagreements=25)
    assert len(everything.disagreements) == 3


def test_disagreement_without_a_rationale_is_still_listed(
    session: Session,
    make_judge: Callable[..., Judge],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    judge = make_judge()
    trajectory = make_trajectory()
    make_label(trajectory, "alice", "pass")
    make_verdict(judge, trajectory, "fail")

    report = build_calibration_report(session, judge)

    assert len(report.disagreements) == 1
    assert report.disagreements[0].judge_rationale is None


# --- the human ceiling -------------------------------------------------------


def test_human_ceiling_scores_overlapping_pairs_only(
    session: Session,
    make_judge: Callable[..., Judge],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    judge = make_judge()
    first = make_trajectory()
    second = make_trajectory()
    third = make_trajectory()
    isolated = make_trajectory()

    for trajectory, verdict in [(first, "pass"), (second, "fail"), (third, "pass")]:
        make_label(trajectory, "alice", verdict)
    for trajectory, verdict in [(first, "pass"), (second, "fail"), (third, "fail")]:
        make_label(trajectory, "bob", verdict)
    for trajectory, verdict in [(first, "pass"), (second, "fail")]:
        make_label(trajectory, "dave", verdict)
    make_label(isolated, "carol", "pass")  # overlaps nobody
    # The ceiling is scoped to trajectories the judge actually scored, so that the
    # human rows and the per-rater judge rows describe one item set.
    for trajectory in (first, second, third, isolated):
        make_verdict(judge, trajectory, "pass")

    report = build_calibration_report(session, judge)

    # Widest overlap first, then alphabetically. Carol never shares a trajectory
    # with anyone, so no pair involving her exists at all.
    assert [(row.annotator_a, row.annotator_b, row.n) for row in report.human_ceiling] == [
        ("alice", "bob", 3),
        ("alice", "dave", 2),
        ("bob", "dave", 2),
    ]
    assert all("carol" not in (row.annotator_a, row.annotator_b) for row in report.human_ceiling)

    # alice vs bob on 3 shared trajectories: (pass,pass), (fail,fail), (pass,fail).
    # p_o = 2/3. alice marginals pass 2/3, fail 1/3; bob marginals pass 1/3, fail 2/3.
    # p_e = (2/3)(1/3) + (1/3)(2/3) = 4/9.
    # kappa = (2/3 - 4/9) / (1 - 4/9) = (2/9) / (5/9) = 0.4.
    alice_bob = report.human_ceiling[0]
    assert alice_bob.observed_agreement == pytest.approx(2 / 3)
    assert alice_bob.cohens_kappa == pytest.approx(0.4)

    # alice vs dave agreed on both shared trajectories, one pass and one fail.
    alice_dave = report.human_ceiling[1]
    assert alice_dave.observed_agreement == pytest.approx(1.0)
    assert alice_dave.cohens_kappa == pytest.approx(1.0)

    # The split on `third` leaves it without a majority.
    assert report.ties_excluded == 1
    assert report.labeled_trajectories == 4


def test_human_ceiling_is_empty_without_double_labeling(
    session: Session,
    make_judge: Callable[..., Judge],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
) -> None:
    judge = make_judge()
    make_label(make_trajectory(), "alice", "pass")
    make_label(make_trajectory(), "bob", "fail")

    report = build_calibration_report(session, judge)

    assert report.human_ceiling == []
    assert report.labeled_trajectories == 2


# --- the empty state ---------------------------------------------------------


def test_empty_dataset_reports_undefined_not_zero(
    session: Session, make_judge: Callable[..., Judge]
) -> None:
    judge = make_judge(name="fresh-judge", model="llama-test-model")

    report = build_calibration_report(session, judge)

    assert report.judge_name == "fresh-judge"
    assert report.compared == 0
    assert report.judged_trajectories == 0
    assert report.labeled_trajectories == 0
    assert report.judge_errors == 0
    assert report.ties_excluded == 0
    assert report.agreement.n == 0
    assert report.agreement.observed_agreement is None
    assert report.agreement.cohens_kappa is None
    assert report.agreement.weighted_kappa is None
    assert report.agreement.kappa_ci_low is None
    assert report.agreement.kappa_ci_high is None
    assert report.agreement.interpretation == "not enough data"
    assert report.human_ceiling == []
    assert report.disagreements == []
    assert all(row.precision is None for row in report.agreement.per_class)
    assert all(row.support == 0 for row in report.agreement.per_class)
    # Nothing in scope means no rulebook to name and, above all, no warning: a mismatch
    # reported over an empty comparison would be a finding computed over nothing.
    assert report.label_rubrics == {}
    assert report.judge_prompt_versions == {}
    assert report.rubric_warning is None


def test_judged_but_unlabeled_reports_no_statistics(
    session: Session,
    make_judge: Callable[..., Judge],
    make_trajectory: Callable[..., Trajectory],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    judge = make_judge()
    for _ in range(3):
        make_verdict(judge, make_trajectory(), "pass")

    report = build_calibration_report(session, judge)

    assert report.judged_trajectories == 3
    assert report.labeled_trajectories == 0
    assert report.compared == 0
    assert report.agreement.n == 0
    assert report.agreement.cohens_kappa is None
    assert report.agreement.observed_agreement is None


# --- rubric provenance --------------------------------------------------------


def test_report_names_both_rubric_rounds_and_warns_about_the_mix(
    session: Session,
    make_judge: Callable[..., Judge],
    make_eval_run: Callable[..., EvalRun],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    """A report over round-one and round-two labels says so, in the data and in prose.

    This is the exact averaging-across-a-rubric-change hazard the rubric_id column was
    activated to prevent: one kappa over labels made under two rulebooks, with nothing
    on the report indicating it. The warning annotates — it must never filter, because
    quietly dropping a round would change the figures without saying which rulebook the
    survivors describe.
    """
    judge = make_judge()
    run = make_eval_run(judge, judge_prompt_version=PROMPT_VERSION)
    in_force = f"{RUBRIC_NAME} v{RUBRIC_VERSION}"

    round_one = make_trajectory()
    make_label(round_one, "momo", "pass")  # rubric_id NULL: the pre-rubric round
    make_verdict(judge, round_one, "pass", eval_run=run)

    round_two = make_trajectory()
    make_label(round_two, "momo", "pass", rubric_id=ensure_rubric(session).id)
    make_verdict(judge, round_two, "pass", eval_run=run)

    report = build_calibration_report(session, judge)

    assert report.label_rubrics == {in_force: 1, "unrecorded": 1}
    assert report.judge_prompt_versions == {PROMPT_VERSION: 2}
    assert report.rubric_warning is not None
    assert "2 rubric versions" in report.rubric_warning
    assert in_force in report.rubric_warning
    assert "unrecorded" in report.rubric_warning
    assert report.rubric_warning.isascii()  # the CLI echoes it verbatim
    # Provenance annotates; it never filters.
    assert report.compared == 2


def test_a_round_one_only_report_is_unrecorded_but_not_called_mixed(
    session: Session,
    make_judge: Callable[..., Judge],
    make_eval_run: Callable[..., EvalRun],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    """The pre-rubric corpus is named for what it is without being told it is mixed.

    A warning that fires on every round-one report would be read once and never again.
    "unrecorded" in the tally is the honest statement; inventing a mismatch out of an
    absence would be a finding computed over nothing.
    """
    judge = make_judge()
    trajectory = make_trajectory()
    make_label(trajectory, "momo", "pass")
    run = make_eval_run(judge, judge_prompt_version=PROMPT_VERSION)
    make_verdict(judge, trajectory, "pass", eval_run=run)
    # A stamped label on a trajectory the judge never scored moves no figure in this
    # report, so it must not put a second rulebook on it either.
    unjudged = make_trajectory()
    make_label(unjudged, "momo", "pass", rubric_id=ensure_rubric(session).id)

    report = build_calibration_report(session, judge)

    assert report.label_rubrics == {"unrecorded": 1}
    assert report.judge_prompt_versions == {PROMPT_VERSION: 1}
    assert report.rubric_warning is None


def test_verdicts_from_two_prompts_are_counted_per_run_and_warned_about(
    session: Session,
    make_judge: Callable[..., Judge],
    make_eval_run: Callable[..., EvalRun],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    """The prompt version comes from each verdict's run, never from the module constant.

    A judge re-run after a prompt edit leaves verdicts graded under both fingerprints in
    the store; the report has to count what each run recorded, or it would stamp today's
    rules onto history.
    """
    judge = make_judge()
    stale_hash = "aaaaaaaaaaaa"  # never a real fingerprint: PROMPT_VERSION would collide
    first, second = make_trajectory(), make_trajectory()
    make_label(first, "momo", "pass")
    make_label(second, "momo", "pass")
    make_verdict(
        judge, first, "pass", eval_run=make_eval_run(judge, judge_prompt_version=stale_hash)
    )
    make_verdict(
        judge, second, "pass", eval_run=make_eval_run(judge, judge_prompt_version=PROMPT_VERSION)
    )

    report = build_calibration_report(session, judge)

    assert report.judge_prompt_versions == {PROMPT_VERSION: 1, stale_hash: 1}
    assert report.rubric_warning is not None
    assert "2 judge prompt versions" in report.rubric_warning
    assert stale_hash in report.rubric_warning


def test_an_unstamped_run_is_reported_unrecorded_never_a_mismatch(
    session: Session,
    make_judge: Callable[..., Judge],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    """A run from before the version column existed is unknowable, not stale.

    It may well have been graded by the prompt in force now; claiming otherwise would
    invent the provenance the nullable column exists to refuse. The tally still says
    "unrecorded", so nothing is hidden — it is just not called a conflict.
    """
    judge = make_judge()
    trajectory = make_trajectory()
    make_label(trajectory, "momo", "pass", rubric_id=ensure_rubric(session).id)
    make_verdict(judge, trajectory, "pass")  # fixture default run carries no version

    report = build_calibration_report(session, judge)

    assert report.judge_prompt_versions == {"unrecorded": 1}
    assert report.label_rubrics == {f"{RUBRIC_NAME} v{RUBRIC_VERSION}": 1}
    assert report.rubric_warning is None


def test_labels_under_a_superseded_rubric_version_warn_against_the_one_in_force(
    session: Session,
    make_judge: Callable[..., Judge],
    make_eval_run: Callable[..., EvalRun],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    """A recorded-but-old rubric version is a fact, and the report says it plainly.

    Unlike an unrecorded one, a label pointing at a superseded rubric row proves the
    annotator was shown different rules than the ones now in force — that is stale by
    evidence, not by assumption, and it warns even though the report itself spans only
    one version.
    """
    judge = make_judge()
    superseded = Rubric(name=RUBRIC_NAME, version=RUBRIC_VERSION - 1, criteria=[])
    session.add(superseded)
    session.commit()
    trajectory = make_trajectory()
    make_label(trajectory, "momo", "pass", rubric_id=superseded.id)
    run = make_eval_run(judge, judge_prompt_version=PROMPT_VERSION)
    make_verdict(judge, trajectory, "pass", eval_run=run)

    report = build_calibration_report(session, judge)

    assert report.label_rubrics == {f"{RUBRIC_NAME} v{RUBRIC_VERSION - 1}": 1}
    assert report.rubric_warning is not None
    assert f"rubric in force is {RUBRIC_NAME} v{RUBRIC_VERSION}" in report.rubric_warning
    assert f"{RUBRIC_NAME} v{RUBRIC_VERSION - 1}" in report.rubric_warning


def test_scoping_to_one_round_names_one_rulebook_and_drops_the_warning(
    session: Session,
    make_judge: Callable[..., Judge],
    make_eval_run: Callable[..., EvalRun],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    """The way out the warning names has to work: scope to one round, warning gone.

    The annotator scope is how a second round is scored on its own. The label tally
    follows the scope, because a label outside it moves no figure in the report; the
    judge's prompt tally does not, because which prompt graded a verdict is a property
    of the run it belongs to, not of whose labels are being compared against.
    """
    judge = make_judge()
    run = make_eval_run(judge, judge_prompt_version=PROMPT_VERSION)
    in_force = f"{RUBRIC_NAME} v{RUBRIC_VERSION}"

    round_one = make_trajectory()
    make_label(round_one, "momo", "pass")  # rubric_id NULL: the pre-rubric round
    make_verdict(judge, round_one, "pass", eval_run=run)
    round_two = make_trajectory()
    make_label(round_two, "momo-r2", "pass", rubric_id=ensure_rubric(session).id)
    make_verdict(judge, round_two, "pass", eval_run=run)

    mixed = build_calibration_report(session, judge)
    assert mixed.label_rubrics == {in_force: 1, "unrecorded": 1}
    assert mixed.rubric_warning is not None

    scoped = build_calibration_report(session, judge, annotators=["momo-r2"])

    assert scoped.compared == 1
    assert scoped.label_rubrics == {in_force: 1}
    # Both verdicts stay: the judge side is scoped by judge and eval run, never by
    # annotator, and both graded rows still stand behind the judge-side figures.
    assert scoped.judge_prompt_versions == {PROMPT_VERSION: 2}
    assert scoped.rubric_warning is None
