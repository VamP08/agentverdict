"""The judge scored against each annotator individually, on their shared items.

The headline kappa is computed against the human *majority*, and a majority does
not exist when the annotators tie — so those trajectories are dropped from
``compared``. With two annotators every disagreement is a tie, which means the
judge is graded only on the items the humans found easy while the human figures
underneath it are computed over everything judged. Two numbers measured on
different samples cannot be read against each other, and the project's own dataset
showed exactly how badly that misleads: a headline kappa of 0.786 against a human
ceiling of 0.523 looked like a judge beating its annotators, when on the identical
items it scored 0.337 against one of them.

These tests seed that failure deliberately: a panel that splits on one trajectory,
so the tied item is excluded from the headline and *included* in every per-rater
row. The per-rater sample is therefore larger than ``compared``, which is the
whole point — that is the item the headline threw away.
"""

from collections.abc import Callable

import pytest
from sqlalchemy.orm import Session

from agentverdict.calibration.report import build_calibration_report
from agentverdict.models import EvalRun, HumanLabel, Judge, JudgeVerdict, Task, Trajectory


@pytest.fixture
def split_panel(
    make_judge: Callable[..., Judge],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> tuple[Judge, dict[str, Trajectory]]:
    """Two annotators who agree twice and split once, with a judge on all three.

    * ``agreed_pass`` — vamp pass, momo pass; judge pass.
    * ``split``       — vamp pass, momo fail; judge fail. No majority: a tie.
    * ``agreed_fail`` — vamp fail, momo fail; judge fail.

    The headline sees two trajectories and calls the judge perfect. Per rater, the
    judge matched momo on all three and missed vamp on the split one.
    """
    judge = make_judge(name="groq-70b", model="llama-test-model")
    trajectories = {
        "agreed_pass": make_trajectory(),
        "split": make_trajectory(),
        "agreed_fail": make_trajectory(),
    }
    make_label(trajectories["agreed_pass"], "vamp", "pass")
    make_label(trajectories["agreed_pass"], "momo", "pass")
    make_label(trajectories["split"], "vamp", "pass")
    make_label(trajectories["split"], "momo", "fail")
    make_label(trajectories["agreed_fail"], "vamp", "fail")
    make_label(trajectories["agreed_fail"], "momo", "fail")

    make_verdict(judge, trajectories["agreed_pass"], "pass", rationale="Refund was verified.")
    make_verdict(judge, trajectories["split"], "fail", rationale="Stopped without confirming.")
    make_verdict(judge, trajectories["agreed_fail"], "fail", rationale="Never looked it up.")
    return judge, trajectories


# --- the bug this feature fixes ----------------------------------------------


def test_tied_trajectory_leaves_the_headline_but_stays_in_every_per_rater_row(
    session: Session, split_panel: tuple[Judge, dict[str, Trajectory]]
) -> None:
    judge, trajectories = split_panel

    report = build_calibration_report(session, judge)

    # The headline: the split trajectory has no majority, so it is excluded.
    assert report.judged_trajectories == 3
    assert report.labeled_trajectories == 3
    assert report.ties_excluded == 1
    assert report.compared == 2
    assert report.agreement.n == 2

    matrix = report.agreement.matrix
    assert sum(sum(row.values()) for row in matrix.values()) == 2
    # Nail down *which* two. Resolving the tie to vamp's "pass" would put a count
    # in [pass][fail]; resolving it to momo's "fail" would make [fail][fail] 2.
    assert matrix["pass"]["fail"] == 0
    assert matrix["pass"]["pass"] == 1
    assert matrix["fail"]["fail"] == 1
    assert trajectories["split"].id not in {row.trajectory_id for row in report.disagreements}
    assert report.disagreements == []

    # The per-rater rows: the same judge, one more item each — the tied one.
    rows = {row.annotator: row for row in report.judge_vs_annotator}
    assert sorted(rows) == ["momo", "vamp"]
    assert rows["vamp"].n == 3
    assert rows["momo"].n == 3
    assert all(row.n > report.compared for row in report.judge_vs_annotator)
    assert all(row.n == report.compared + 1 for row in report.judge_vs_annotator)

    # vamp's raw agreement can only be 2/3 if the tied trajectory is counted:
    # on the two compared trajectories alone the judge matched vamp on both.
    assert rows["vamp"].observed_agreement == pytest.approx(2 / 3)
    assert rows["momo"].observed_agreement == pytest.approx(1.0)


def test_per_rater_rows_undo_the_superhuman_illusion(
    session: Session, split_panel: tuple[Judge, dict[str, Trajectory]]
) -> None:
    """The headline beats the human ceiling; on equal items the judge does not."""
    judge, _ = split_panel

    report = build_calibration_report(session, judge)

    rows = {row.annotator: row for row in report.judge_vs_annotator}
    ceiling = report.human_ceiling[0]
    assert (ceiling.annotator_a, ceiling.annotator_b, ceiling.n) == ("momo", "vamp", 3)

    # momo vs vamp on 3 shared items: (pass,pass), (fail,pass), (fail,fail).
    # p_o = 2/3. momo marginals pass 1/3, fail 2/3; vamp pass 2/3, fail 1/3.
    # p_e = (2/3)(1/3) + (1/3)(2/3) = 4/9, so kappa = (2/9)/(5/9) = 0.4.
    assert ceiling.cohens_kappa == pytest.approx(0.4)
    # The judge agreed with the humans on both surviving items, so the headline is
    # a perfect 1.0 — measured on a strictly easier sample than the ceiling above.
    assert report.agreement.cohens_kappa == pytest.approx(1.0)
    assert report.agreement.cohens_kappa > ceiling.cohens_kappa

    # On the identical items the humans were scored on, the judge is no better
    # than they are: exactly the ceiling against vamp, perfect only against momo.
    assert rows["vamp"].cohens_kappa == pytest.approx(0.4)
    assert rows["momo"].cohens_kappa == pytest.approx(1.0)
    assert rows["vamp"].cohens_kappa <= ceiling.cohens_kappa

    # The held-out baseline is already scoped to everything judged, so it and the
    # per-rater rows describe the same three trajectories — unlike the headline.
    assert [(row.annotator, row.n) for row in report.human_baseline] == [
        ("momo", 3),
        ("vamp", 3),
    ]
    assert {row.n for row in report.judge_vs_annotator} == {
        row.n for row in report.human_baseline
    }


def test_per_rater_pairs_the_annotators_own_verdict_not_the_majority(
    session: Session,
    make_judge: Callable[..., Judge],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    """A dissenter is scored on what they said, never on what the panel said."""
    judge = make_judge()
    trajectory = make_trajectory()
    make_label(trajectory, "alice", "pass")
    make_label(trajectory, "bob", "pass")
    make_label(trajectory, "dissenter", "fail")  # outvoted 2-1
    make_verdict(judge, trajectory, "pass")

    report = build_calibration_report(session, judge)

    assert report.compared == 1
    assert report.agreement.observed_agreement == pytest.approx(1.0)

    rows = {row.annotator: row for row in report.judge_vs_annotator}
    assert sorted(rows) == ["alice", "bob", "dissenter"]
    assert rows["alice"].observed_agreement == pytest.approx(1.0)
    # Against the majority the judge looks flawless; against the person who
    # actually disagreed with it, it is wrong on the only item they share.
    assert rows["dissenter"].observed_agreement == pytest.approx(0.0)
    assert all(row.n == 1 for row in report.judge_vs_annotator)


# --- membership and ordering --------------------------------------------------


def test_rows_are_widest_overlap_first_and_skip_annotators_the_judge_never_met(
    session: Session,
    make_judge: Callable[..., Judge],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    judge = make_judge()
    first, second, third = make_trajectory(), make_trajectory(), make_trajectory()
    unjudged = make_trajectory()

    for trajectory, verdict in [(first, "pass"), (second, "fail"), (third, "pass")]:
        make_label(trajectory, "alice", verdict)
    for trajectory, verdict in [(first, "pass"), (second, "fail")]:
        make_label(trajectory, "bob", verdict)
    for trajectory, verdict in [(first, "pass"), (second, "fail"), (third, "fail")]:
        make_label(trajectory, "zoe", verdict)
    make_label(unjudged, "carol", "pass")  # nothing the judge ever scored

    make_verdict(judge, first, "pass")
    make_verdict(judge, second, "fail")
    make_verdict(judge, third, "pass")

    report = build_calibration_report(session, judge)

    # Widest overlap first; alphabetical inside a tie, so alice precedes zoe.
    assert [(row.annotator, row.n) for row in report.judge_vs_annotator] == [
        ("alice", 3),
        ("zoe", 3),
        ("bob", 2),
    ]
    # carol shares no trajectory with the judge: there is nothing to compute, and
    # an empty row would read as an annotator the judge never agrees with.
    assert "carol" not in {row.annotator for row in report.judge_vs_annotator}
    assert report.labeled_trajectories == 4  # carol's trajectory is still labeled


def test_undefined_kappa_is_none_not_zero(
    session: Session,
    make_judge: Callable[..., Judge],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    """A rater who never varies has no chance model, so kappa has no answer."""
    judge = make_judge()
    first, second, third = make_trajectory(), make_trajectory(), make_trajectory()

    for trajectory in (first, second, third):
        make_label(trajectory, "flat", "pass")  # same verdict every time
    for trajectory, verdict in [(first, "pass"), (second, "pass"), (third, "fail")]:
        make_label(trajectory, "varied", verdict)
    make_verdict(judge, first, "pass")
    make_verdict(judge, second, "pass")
    make_verdict(judge, third, "fail")

    report = build_calibration_report(session, judge)

    rows = {row.annotator: row for row in report.judge_vs_annotator}
    assert rows["flat"].n == 3
    # 0.0 would be read as "slight agreement" when the judge in fact matched
    # this annotator on two of three items; the honest answer is "no answer".
    assert rows["flat"].cohens_kappa is None
    assert rows["flat"].observed_agreement == pytest.approx(2 / 3)

    assert rows["varied"].cohens_kappa == pytest.approx(1.0)
    assert rows["varied"].observed_agreement == pytest.approx(1.0)

    # The panel split on `third`, so the per-rater rows again outnumber the headline.
    assert report.ties_excluded == 1
    assert report.compared == 2


def test_unusable_verdicts_never_enter_a_per_rater_row(
    session: Session,
    make_judge: Callable[..., Judge],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    """Verdicts outside fail/borderline/pass are dropped from both sides."""
    judge = make_judge()
    usable = make_trajectory()
    odd_judge_verdict = make_trajectory()

    make_label(usable, "alice", "pass")
    make_label(usable, "bob", "unsure")  # not in the vocabulary
    make_label(odd_judge_verdict, "alice", "fail")
    make_label(odd_judge_verdict, "bob", "fail")
    make_verdict(judge, usable, "pass")
    make_verdict(judge, odd_judge_verdict, "weird")  # nor is this

    report = build_calibration_report(session, judge)

    assert report.judge_errors == 0  # a strange string is not a recorded failure
    assert report.judged_trajectories == 2
    # alice keeps only the one comparable pair; bob has no comparable pair at all,
    # so he gets no row rather than a row built from nothing.
    assert [(row.annotator, row.n) for row in report.judge_vs_annotator] == [("alice", 1)]


# --- nothing to compare -------------------------------------------------------


def test_no_rows_without_a_judge_verdict_or_a_label(
    session: Session,
    make_judge: Callable[..., Judge],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    fresh = make_judge(name="fresh-judge")
    assert build_calibration_report(session, fresh).judge_vs_annotator == []

    labeled_only = make_judge(name="labeled-only")
    make_label(make_trajectory(), "alice", "pass")
    assert build_calibration_report(session, labeled_only).judge_vs_annotator == []

    judged_only = make_judge(name="judged-only")
    make_verdict(judged_only, make_trajectory(), "pass")
    report = build_calibration_report(session, judged_only)
    assert report.judged_trajectories == 1
    assert report.judge_vs_annotator == []


def test_rows_survive_when_the_headline_has_nothing_to_show(
    session: Session,
    make_judge: Callable[..., Judge],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    """Every trajectory tied, so the headline is empty and the rows are all there is."""
    judge = make_judge()
    first, second = make_trajectory(), make_trajectory()
    make_label(first, "vamp", "pass")
    make_label(first, "momo", "fail")
    make_label(second, "vamp", "borderline")
    make_label(second, "momo", "pass")
    make_verdict(judge, first, "pass")
    make_verdict(judge, second, "pass")

    report = build_calibration_report(session, judge)

    assert report.ties_excluded == 2
    assert report.compared == 0
    assert report.agreement.cohens_kappa is None
    # Before this existed the report said precisely nothing about a judge that had
    # in fact scored both trajectories. Now it says the judge tracked vamp once
    # and momo once.
    assert [(row.annotator, row.n) for row in report.judge_vs_annotator] == [
        ("momo", 2),
        ("vamp", 2),
    ]
    rows = {row.annotator: row for row in report.judge_vs_annotator}
    assert rows["vamp"].observed_agreement == pytest.approx(0.5)
    assert rows["momo"].observed_agreement == pytest.approx(0.5)


def test_per_rater_rows_follow_the_eval_run_and_task_scopes(
    session: Session,
    make_judge: Callable[..., Judge],
    make_eval_run: Callable[..., EvalRun],
    make_task: Callable[..., Task],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    """The rows read the same judge verdicts the headline does, filters included."""
    judge = make_judge()
    refunds = make_trajectory(task=make_task(key="refund-simple-01"))
    orders = make_trajectory(task=make_task(key="order-status-01"))
    make_label(refunds, "alice", "pass")
    make_label(orders, "alice", "pass")
    old_run = make_eval_run(judge)
    make_verdict(judge, refunds, "pass", eval_run=old_run)
    make_verdict(judge, orders, "fail", eval_run=old_run)

    whole = build_calibration_report(session, judge)
    assert [(row.annotator, row.n) for row in whole.judge_vs_annotator] == [("alice", 2)]

    by_task = build_calibration_report(session, judge, task_key="refund-simple-01")
    assert [(row.annotator, row.n) for row in by_task.judge_vs_annotator] == [("alice", 1)]
    assert by_task.judge_vs_annotator[0].observed_agreement == pytest.approx(1.0)

    by_run = build_calibration_report(session, judge, eval_run_id="no-such-run")
    assert by_run.judge_vs_annotator == []
