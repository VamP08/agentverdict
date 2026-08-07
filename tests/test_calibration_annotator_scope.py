"""Scoping a calibration report to a named set of annotators.

Once a rubric ambiguity is written down, a second annotation round measures a
different thing than the first: the same trajectories, re-read under a rule that
did not exist the first time. Overwriting the original labels would destroy the
evidence that the ambiguity mattered, so the re-labels are recorded under their own
annotator names (``vamp-r2``, ``momo-r2``) and each round is scored separately.

The filter has to reach *everything* — the majority, the tie count, the ceiling,
the held-out baseline, the per-rater rows and ``labeled_trajectories`` — or the
report would mix rounds in some figures and not others, which is worse than not
scoping at all. These tests pin each of those, plus the two edges: naming nobody
means everybody, and naming somebody who has never labeled anything is an empty
report rather than an error.
"""

from collections.abc import Callable

import pytest
from sqlalchemy.orm import Session

from agentverdict.calibration.report import build_calibration_report
from agentverdict.models import HumanLabel, Judge, JudgeVerdict, Task, Trajectory

ROUND_ONE = ["vamp", "momo"]
ROUND_TWO = ["vamp-r2", "momo-r2"]


@pytest.fixture
def two_annotators(
    make_judge: Callable[..., Judge],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> tuple[Judge, dict[str, Trajectory]]:
    """A panel that ties twice, plus one trajectory only momo labeled.

    * ``agreed``   — vamp pass, momo pass; judge pass.
    * ``split_up`` — vamp pass, momo fail; judge fail. Tie.
    * ``split_dn`` — vamp fail, momo pass; judge fail. Tie.
    * ``momo_only``— momo pass; judge pass.

    Unscoped this yields 2 ties and 2 compared. Scoped to vamp alone, both ties
    resolve to vamp's own verdict and ``momo_only`` drops out of the label set.
    """
    judge = make_judge(name="groq-70b", model="llama-test-model")
    trajectories = {
        "agreed": make_trajectory(),
        "split_up": make_trajectory(),
        "split_dn": make_trajectory(),
        "momo_only": make_trajectory(),
    }
    make_label(trajectories["agreed"], "vamp", "pass")
    make_label(trajectories["agreed"], "momo", "pass")
    make_label(trajectories["split_up"], "vamp", "pass")
    make_label(trajectories["split_up"], "momo", "fail")
    make_label(trajectories["split_dn"], "vamp", "fail")
    make_label(trajectories["split_dn"], "momo", "pass")
    make_label(trajectories["momo_only"], "momo", "pass")

    make_verdict(judge, trajectories["agreed"], "pass")
    make_verdict(judge, trajectories["split_up"], "fail")
    make_verdict(judge, trajectories["split_dn"], "fail")
    make_verdict(judge, trajectories["momo_only"], "pass")
    return judge, trajectories


# --- what the scope changes ---------------------------------------------------


def test_scoping_to_one_annotator_resolves_the_ties_and_grows_the_comparison(
    session: Session, two_annotators: tuple[Judge, dict[str, Trajectory]]
) -> None:
    judge, trajectories = two_annotators

    everyone = build_calibration_report(session, judge)
    assert everyone.annotators == []
    assert everyone.labeled_trajectories == 4
    assert everyone.ties_excluded == 2
    assert everyone.compared == 2

    vamp_only = build_calibration_report(session, judge, annotators=["vamp"])

    # A panel of one is never split, so both former ties now carry a majority —
    # vamp's own verdict — and the comparison grows even though fewer labels count.
    assert vamp_only.ties_excluded == 0
    assert vamp_only.compared == 3
    assert vamp_only.compared > everyone.compared
    assert vamp_only.ties_excluded < everyone.ties_excluded
    # momo's solo trajectory has no in-scope label at all, so it stops being a
    # labeled trajectory for this report.
    assert vamp_only.labeled_trajectories == 3
    assert vamp_only.agreement.n == 3

    # The judge's side is untouched: an annotator filter selects labels, not runs.
    assert vamp_only.judged_trajectories == everyone.judged_trajectories == 4
    assert vamp_only.judge_errors == 0

    # Scoping surfaces a disagreement the tie had been hiding: vamp passed
    # `split_up`, the judge failed it.
    assert [row.trajectory_id for row in vamp_only.disagreements] == [
        trajectories["split_up"].id
    ]
    assert vamp_only.disagreements[0].human_verdict == "pass"
    assert vamp_only.disagreements[0].judge_verdict == "fail"
    assert vamp_only.disagreements[0].annotators == ["vamp"]


def test_scoping_narrows_the_per_rater_ceiling_and_baseline_blocks(
    session: Session, two_annotators: tuple[Judge, dict[str, Trajectory]]
) -> None:
    judge, _ = two_annotators

    everyone = build_calibration_report(session, judge)
    assert [(row.annotator, row.n) for row in everyone.judge_vs_annotator] == [
        ("momo", 4),
        ("vamp", 3),
    ]
    assert [(row.annotator_a, row.annotator_b) for row in everyone.human_ceiling] == [
        ("momo", "vamp")
    ]
    assert [row.annotator for row in everyone.human_baseline] == ["momo", "vamp"]

    vamp_only = build_calibration_report(session, judge, annotators=["vamp"])

    assert [(row.annotator, row.n) for row in vamp_only.judge_vs_annotator] == [("vamp", 3)]
    # One annotator has nobody to be compared against, so both human blocks are
    # empty rather than being quietly computed against the out-of-scope round.
    assert vamp_only.human_ceiling == []
    assert vamp_only.human_baseline == []


def test_scope_is_echoed_sorted_and_deduplicated(
    session: Session, two_annotators: tuple[Judge, dict[str, Trajectory]]
) -> None:
    judge, _ = two_annotators

    report = build_calibration_report(session, judge, annotators=["vamp", "momo", "vamp"])

    assert report.annotators == ["momo", "vamp"]
    # Naming the whole panel is the same measurement as naming nobody.
    everyone = build_calibration_report(session, judge)
    assert (report.compared, report.ties_excluded) == (everyone.compared, everyone.ties_excluded)
    assert report.labeled_trajectories == everyone.labeled_trajectories
    assert report.agreement.cohens_kappa == everyone.agreement.cohens_kappa


def test_an_empty_scope_means_every_annotator(
    session: Session, two_annotators: tuple[Judge, dict[str, Trajectory]]
) -> None:
    """No caller can have meant "score nobody", so an empty list is not a filter."""
    judge, _ = two_annotators

    empty = build_calibration_report(session, judge, annotators=[])
    unscoped = build_calibration_report(session, judge, annotators=None)

    assert empty.annotators == []
    assert empty.labeled_trajectories == unscoped.labeled_trajectories == 4
    assert empty.compared == unscoped.compared == 2
    assert [row.annotator for row in empty.judge_vs_annotator] == ["momo", "vamp"]


def test_unknown_annotator_yields_an_empty_comparison_not_an_error(
    session: Session, two_annotators: tuple[Judge, dict[str, Trajectory]]
) -> None:
    """Asking for a round nobody has annotated yet is a legitimate question."""
    judge, _ = two_annotators

    report = build_calibration_report(session, judge, annotators=["nobody", "vamp-r2"])

    assert report.annotators == ["nobody", "vamp-r2"]  # echoed, so the filter is visible
    assert report.labeled_trajectories == 0
    assert report.compared == 0
    assert report.ties_excluded == 0
    assert report.judge_vs_annotator == []
    assert report.human_ceiling == []
    assert report.human_baseline == []
    assert report.disagreements == []
    assert report.agreement.n == 0
    assert report.agreement.cohens_kappa is None
    assert report.agreement.interpretation == "not enough data"
    # The judge's own coverage is unchanged, which is what makes the empty result
    # readable as "wrong names" rather than "unjudged dataset".
    assert report.judged_trajectories == 4


def test_scope_combines_with_the_task_filter(
    session: Session,
    make_judge: Callable[..., Judge],
    make_task: Callable[..., Task],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    judge = make_judge()
    refunds = make_trajectory(task=make_task(key="refund-simple-01"))
    orders = make_trajectory(task=make_task(key="order-status-01"))
    for trajectory in (refunds, orders):
        make_label(trajectory, "vamp", "pass")
        make_label(trajectory, "momo", "fail")
        make_verdict(judge, trajectory, "pass")

    both = build_calibration_report(
        session, judge, task_key="refund-simple-01", annotators=["vamp"]
    )

    assert both.task_key == "refund-simple-01"
    assert both.annotators == ["vamp"]
    assert both.labeled_trajectories == 1
    assert both.compared == 1
    assert both.agreement.observed_agreement == pytest.approx(1.0)
    assert [(row.annotator, row.n) for row in both.judge_vs_annotator] == [("vamp", 1)]


# --- two annotation rounds ----------------------------------------------------


@pytest.fixture
def two_rounds(
    make_judge: Callable[..., Judge],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> tuple[Judge, dict[str, Trajectory]]:
    """The same three trajectories labeled twice, before and after the rubric fix.

    ``unfinished`` is the trajectory the rubric rule now settles: in round one vamp
    read a stopped-but-correct conversation as a pass and momo as borderline, which
    is a tie. In round two, with the rule written down, both call it borderline —
    and so does the judge, whose prompt states the same rule.
    """
    judge = make_judge(name="groq-70b", model="llama-test-model")
    trajectories = {
        "clean": make_trajectory(),
        "unfinished": make_trajectory(),
        "broken": make_trajectory(),
    }
    round_one = {
        "clean": ("pass", "pass"),
        "unfinished": ("pass", "borderline"),  # the rule nobody had written down yet
        "broken": ("fail", "fail"),
    }
    round_two = {
        "clean": ("pass", "pass"),
        "unfinished": ("borderline", "borderline"),  # settled by the rubric rule
        "broken": ("fail", "fail"),
    }
    for name, (vamp_verdict, momo_verdict) in round_one.items():
        make_label(trajectories[name], "vamp", vamp_verdict)
        make_label(trajectories[name], "momo", momo_verdict)
    for name, (vamp_verdict, momo_verdict) in round_two.items():
        make_label(trajectories[name], "vamp-r2", vamp_verdict)
        make_label(trajectories[name], "momo-r2", momo_verdict)

    make_verdict(judge, trajectories["clean"], "pass", rationale="Refund verified and issued.")
    make_verdict(
        judge,
        trajectories["unfinished"],
        "borderline",
        rationale="Correct conduct, but the job was left unfinished.",
    )
    make_verdict(judge, trajectories["broken"], "fail", rationale="Never looked the order up.")
    return judge, trajectories


def test_each_round_is_scored_on_its_own_labels(
    session: Session, two_rounds: tuple[Judge, dict[str, Trajectory]]
) -> None:
    judge, _ = two_rounds

    first = build_calibration_report(session, judge, annotators=ROUND_ONE)
    second = build_calibration_report(session, judge, annotators=ROUND_TWO)

    assert first.annotators == ["momo", "vamp"]
    assert second.annotators == ["momo-r2", "vamp-r2"]
    # Neither report can see the other round's names anywhere.
    assert [row.annotator for row in first.judge_vs_annotator] == ["momo", "vamp"]
    assert [row.annotator for row in second.judge_vs_annotator] == ["momo-r2", "vamp-r2"]
    assert [row.annotator for row in first.human_baseline] == ["momo", "vamp"]
    assert [(row.annotator_a, row.annotator_b) for row in second.human_ceiling] == [
        ("momo-r2", "vamp-r2")
    ]
    # Round one: the rubric ambiguity is a tie, so the headline never sees it —
    # it reports a judge that disagreed with nobody, on a dataset where one of the
    # two annotators disagreed with it.
    assert first.disagreements == []
    vamp_row = next(row for row in first.judge_vs_annotator if row.annotator == "vamp")
    assert vamp_row.observed_agreement == pytest.approx(2 / 3)

    assert first.labeled_trajectories == 3
    assert first.ties_excluded == 1
    assert first.compared == 2

    # Round two: the rule is written down, the panel agrees, nothing is dropped.
    assert second.labeled_trajectories == 3
    assert second.ties_excluded == 0
    assert second.compared == 3


def test_the_two_rounds_report_different_numbers(
    session: Session, two_rounds: tuple[Judge, dict[str, Trajectory]]
) -> None:
    judge, _ = two_rounds

    first = build_calibration_report(session, judge, annotators=ROUND_ONE)
    second = build_calibration_report(session, judge, annotators=ROUND_TWO)

    assert (first.compared, first.ties_excluded) != (second.compared, second.ties_excluded)

    # The headline is blind to the improvement: it was already perfect in round
    # one, because the item the annotators disagreed about had been thrown away.
    assert first.agreement.observed_agreement == pytest.approx(1.0)
    assert second.agreement.observed_agreement == pytest.approx(1.0)

    # The per-rater rows are where the clarification shows up. vamp missed the
    # unfinished conversation in round one and matched the judge in round two.
    first_rows = {row.annotator: row for row in first.judge_vs_annotator}
    second_rows = {row.annotator: row for row in second.judge_vs_annotator}
    assert all(row.n == 3 for row in first.judge_vs_annotator)
    assert all(row.n == 3 for row in second.judge_vs_annotator)
    assert first_rows["vamp"].observed_agreement == pytest.approx(2 / 3)
    assert second_rows["vamp-r2"].observed_agreement == pytest.approx(1.0)
    assert first_rows["momo"].observed_agreement == pytest.approx(1.0)
    assert second_rows["momo-r2"].observed_agreement == pytest.approx(1.0)

    # And so does the human ceiling: the round-one split is gone.
    assert first.human_ceiling[0].observed_agreement == pytest.approx(2 / 3)
    assert second.human_ceiling[0].observed_agreement == pytest.approx(1.0)


def test_round_one_labels_are_preserved_rather_than_replaced(
    session: Session, two_rounds: tuple[Judge, dict[str, Trajectory]]
) -> None:
    """The pre-clarification record still exists, and an unscoped report mixes it in."""
    judge, _ = two_rounds

    mixed = build_calibration_report(session, judge)

    assert mixed.annotators == []
    assert mixed.labeled_trajectories == 3
    assert [row.annotator for row in mixed.judge_vs_annotator] == [
        "momo",
        "momo-r2",
        "vamp",
        "vamp-r2",
    ]
    # Round one is still readable in full: the pair is there with its original
    # disagreement, which is the evidence that the ambiguity was real.
    ceiling = {(row.annotator_a, row.annotator_b): row for row in mixed.human_ceiling}
    assert ("momo", "vamp") in ceiling
    assert ceiling[("momo", "vamp")].observed_agreement == pytest.approx(2 / 3)

    # But mixing them is not a measurement of either round: three round-two votes
    # outvote vamp's round-one pass, so the ambiguity silently disappears.
    assert mixed.ties_excluded == 0
    assert mixed.compared == 3
    assert build_calibration_report(session, judge, annotators=ROUND_ONE).ties_excluded == 1
