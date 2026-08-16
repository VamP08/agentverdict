"""Per-criterion agreement: what gets compared, what is excluded, and what is not a fault.

The verdict token is a summary of a reading, and two raters can read a run identically and
summarise it differently. These tests are about the two ways that shows up in the report:

* a criterion one side did not answer is **excluded and counted**, never filled in with
  0.0 — absence says the question did not arise, and 0.0 says the agent failed it;
* a verdict split where both sides answered every criterion either of them reached, and
  answered all of it alike, is counted in ``rule_disagreements``, because the difference
  then lives in the rule that turns answers into a verdict rather than in anyone's
  perception of the run — while a split whose two sides were barely compared (a one-sided
  criterion, or no shared answers at all) goes to ``rule_undetermined``, since a thin
  overlap cannot carry the claim that the two sides read the run the same way.

Everything is seeded into the test session; nothing here calls a model.
"""

from collections.abc import Callable

import pytest
from sqlalchemy.orm import Session

from agentverdict.calibration.report import build_calibration_report
from agentverdict.cli import _print_calibration_report
from agentverdict.models import HumanLabel, Judge, JudgeVerdict, Task, Trajectory
from agentverdict.rubric import CRITERIA, validate_scores
from agentverdict.schemas import CalibrationReport, CriterionAgreement


def _scores(**answers: float) -> dict[str, float]:
    """Criterion answers, checked against the rubric the way every real writer checks them.

    Seeding straight into the table bypasses the API schema and the labeling form, so a
    typo here would sit in the fixture looking like data, be dropped by the report as an
    unknown key, and leave a test that proves nothing while passing.
    """
    return validate_scores(answers)


def _criterion(report: CalibrationReport, key: str) -> CriterionAgreement:
    """The report's row for one criterion. Every criterion always has exactly one."""
    return next(row for row in report.criteria if row.key == key)


# --- absence is not zero -----------------------------------------------------


def test_an_unanswered_consent_is_excluded_from_agreement_never_scored_zero(
    session: Session,
    make_judge: Callable[..., Judge],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    """A key one side left out moves the excluded count and leaves the statistics alone.

    This is the whole undefined-vs-zero discipline in one assertion. ``consent_obtained``
    is absent whenever no irreversible action was taken, which is most of the corpus; a
    scheme that wrote 0.0 for both "the agent acted without asking" and "the question
    never arose" would report a consent violation on every look-up, and — worse — would
    report the two raters as *agreeing* about it whenever the human said no.
    """
    judge = make_judge()
    # Four trajectories both sides scored: 3 agreements and 1 miss.
    # p_o = 3/4 = 0.75. Human marginals 1.0 -> 0.5, 0.0 -> 0.5; judge 1.0 -> 0.75,
    # 0.0 -> 0.25. p_e = 0.5*0.75 + 0.5*0.25 = 0.5. kappa = (0.75 - 0.5) / 0.5 = 0.5.
    for human, judged in [(1.0, 1.0), (1.0, 1.0), (0.0, 0.0), (0.0, 1.0)]:
        trajectory = make_trajectory()
        make_label(
            trajectory,
            "momo",
            "pass",
            rubric_scores=_scores(goal_achieved=1.0, consent_obtained=human),
        )
        make_verdict(
            judge,
            trajectory,
            "pass",
            rubric_scores=_scores(goal_achieved=1.0, consent_obtained=judged),
        )

    before = build_calibration_report(session, judge)
    consent_before = _criterion(before, "consent_obtained")
    assert before.compared == 4
    assert consent_before.n == 4
    assert consent_before.observed_agreement == pytest.approx(0.75)
    assert consent_before.cohens_kappa == pytest.approx(0.5)
    assert before.criteria_unanswered["consent_obtained"] == 0

    # A read-only run: the annotator answered "no" and the judge left the key out.
    read_only = make_trajectory()
    make_label(
        read_only,
        "momo",
        "pass",
        rubric_scores=_scores(goal_achieved=1.0, consent_obtained=0.0),
    )
    make_verdict(judge, read_only, "pass", rubric_scores=_scores(goal_achieved=1.0))

    after = build_calibration_report(session, judge)
    consent_after = _criterion(after, "consent_obtained")

    assert after.compared == 5
    assert after.criteria_unanswered["consent_obtained"] == 1  # the excluded count moves
    assert consent_after.n == 4  # and the sample it was measured on does not
    assert consent_after.observed_agreement == pytest.approx(0.75)
    assert consent_after.cohens_kappa == pytest.approx(0.5)
    # Coerced to 0.0, the judge's silence would have matched the annotator's "no": five
    # pairs, p_o = 0.8, p_e = 0.48, kappa = 0.615. A better number about a worse claim.
    assert (consent_after.human_positive, consent_after.judge_positive) == (2, 3)

    # goal_achieved was answered on all five, which is what makes the gap readable.
    assert _criterion(after, "goal_achieved").n == 5
    for row in after.criteria:
        assert row.n + after.criteria_unanswered[row.key] == after.compared


def test_a_criterion_the_annotators_split_on_has_no_panel_answer(
    session: Session,
    make_judge: Callable[..., Judge],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    """A split panel holds no position, so the judge is not scored against one either.

    Picking a side by annotator order would settle the hardest calls alphabetically. The
    drop is counted rather than hidden, because a criterion's kappa is only readable next
    to how much of the sample reached it.
    """
    judge = make_judge()

    split = make_trajectory()
    for annotator, clarity in [("momo", 1.0), ("vamp", 0.0)]:
        make_label(
            split,
            annotator,
            "pass",
            rubric_scores=_scores(goal_achieved=1.0, communication_clear=clarity),
        )
    make_verdict(
        judge,
        split,
        "pass",
        rubric_scores=_scores(goal_achieved=1.0, communication_clear=1.0),
    )

    # The same disagreement with a third reader: 2-1 is an answer, 1-1 is not.
    resolved = make_trajectory()
    for annotator, clarity in [("momo", 1.0), ("vamp", 0.0), ("carol", 1.0)]:
        make_label(
            resolved,
            annotator,
            "pass",
            rubric_scores=_scores(goal_achieved=1.0, communication_clear=clarity),
        )
    make_verdict(
        judge,
        resolved,
        "pass",
        rubric_scores=_scores(goal_achieved=1.0, communication_clear=1.0),
    )

    report = build_calibration_report(session, judge)

    assert report.compared == 2
    assert _criterion(report, "goal_achieved").n == 2
    assert _criterion(report, "communication_clear").n == 1
    assert report.criteria_unanswered["communication_clear"] == 1
    assert report.criteria_unanswered["goal_achieved"] == 0


def test_scores_the_rubric_does_not_recognise_are_dropped_rather_than_repaired(
    session: Session,
    make_judge: Callable[..., Judge],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    """Rows written under an older rubric, or by a hand-built POST, are still read.

    Deliberately seeded past ``_scores``: the report reads the stored blob directly and
    cannot assume every writer validated. Snapping 0.7 to the nearest permitted score
    would invent an opinion nobody held, and keeping "vibes" would print a column with no
    counterpart on the other side. Both read as unanswered, which is what they are.
    """
    judge = make_judge()
    trajectory = make_trajectory()
    make_label(trajectory, "momo", "pass", rubric_scores={"goal_achieved": 1.0, "vibes": 1.0})
    make_verdict(
        judge,
        trajectory,
        "pass",
        rubric_scores={"goal_achieved": 1.0, "tools_correct": 0.7},
    )

    report = build_calibration_report(session, judge)

    assert _criterion(report, "goal_achieved").n == 1
    assert _criterion(report, "tools_correct").n == 0
    assert [row.key for row in report.criteria] == [c.key for c in CRITERIA]


# --- what the rows say when there is nothing to say --------------------------


def test_a_criterion_everyone_answers_alike_reports_no_kappa_but_shows_its_marginals(
    session: Session,
    make_judge: Callable[..., Judge],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    """Kappa is undefined for a rater who never varies, and says so instead of 0.000.

    Most runs use their tools correctly, so this is the ordinary state of the column
    rather than an edge case. Chance-corrected agreement has no answer when one rater has
    no variance, and the marginals travel with the row precisely so that a reader can
    tell "they never disagreed" from "there was nothing to disagree about".
    """
    judge = make_judge()
    for _ in range(4):
        trajectory = make_trajectory()
        make_label(trajectory, "momo", "pass", rubric_scores=_scores(tools_correct=1.0))
        make_verdict(judge, trajectory, "pass", rubric_scores=_scores(tools_correct=1.0))

    row = _criterion(build_calibration_report(session, judge), "tools_correct")

    assert row.n == 4
    assert row.observed_agreement == 1.0
    assert row.cohens_kappa is None
    assert (row.human_positive, row.judge_positive) == (4, 4)


def test_every_criterion_gets_a_row_even_when_nobody_answered_one(
    session: Session,
    make_judge: Callable[..., Judge],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    """Round one was collected before the criteria existed, and the report survives it.

    A row reading n=0 says the column is empty; omitting the row says the rubric has four
    criteria. The first is a fact about the data and the second is a lie about the rules.
    """
    judge = make_judge()
    for _ in range(3):
        trajectory = make_trajectory()
        make_label(trajectory, "momo", "pass")
        make_verdict(judge, trajectory, "pass")

    report = build_calibration_report(session, judge)

    assert report.compared == 3
    assert [row.key for row in report.criteria] == [c.key for c in CRITERIA]
    assert all(row.n == 0 for row in report.criteria)
    assert all(row.cohens_kappa is None for row in report.criteria)
    assert all(row.observed_agreement is None for row in report.criteria)
    assert report.criteria_unanswered == {c.key: 3 for c in CRITERIA}
    assert report.rule_disagreements == 0
    assert report.rule_undetermined == 0  # only disagreements can be undetermined


# --- rule disagreement vs. perception disagreement ---------------------------


def test_a_verdict_split_with_identical_criteria_is_counted_as_a_rubric_fault(
    session: Session,
    make_judge: Callable[..., Judge],
    make_task: Callable[..., Task],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    """The order-status-01 case, and why it must not be filed under judge error.

    On trajectory eb524ef3 the agent looked the order up, reported status and ETA as the
    task asked, then offered tracking details and stopped. Both annotators said `pass`;
    the judge said `borderline` and its rationale agreed with them about every fact of the
    run. Both sides answer ended_waiting 1.0 and goal_achieved 1.0 — identical readings,
    different verdicts — so the difference lives in the rule that turns answers into a
    verdict, and re-prompting the judge would move nothing.

    The perception split beside it is the same verdict distance and the opposite cause:
    there the two sides do not agree about what the transcript shows. Only that one is a
    judge problem, and the counts are what separate them.
    """
    judge = make_judge()
    agreed = _scores(
        goal_achieved=1.0, tools_correct=1.0, ended_waiting=1.0, communication_clear=1.0
    )

    rule = make_trajectory(task=make_task(key="order-status-01"))
    make_label(rule, "momo", "pass", rubric_scores=agreed)
    make_label(rule, "vamp", "pass", rubric_scores=agreed)
    make_verdict(
        judge,
        rule,
        "borderline",
        rubric_scores=agreed,
        rationale="Status and ETA reported accurately. However, the conversation is unfinished.",
    )

    perception = make_trajectory(task=make_task(key="refund-simple-01"))
    read_as_done = _scores(goal_achieved=1.0, tools_correct=1.0)
    make_label(perception, "momo", "pass", rubric_scores=read_as_done)
    make_label(perception, "vamp", "pass", rubric_scores=read_as_done)
    make_verdict(
        judge,
        perception,
        "borderline",
        rubric_scores=_scores(goal_achieved=0.5, tools_correct=1.0),
    )

    # One the two sides settled outright, to show agreement is never counted here.
    settled = make_trajectory(task=make_task(key="angry-customer-01"))
    make_label(settled, "momo", "pass", rubric_scores=agreed)
    make_verdict(judge, settled, "pass", rubric_scores=agreed)

    report = build_calibration_report(session, judge)

    assert report.compared == 3
    assert report.disagreements_total == 2
    assert report.rule_disagreements == 1
    assert report.rule_undetermined == 0  # both splits here were fully measured
    # With nothing undetermined, the remainder is what is left for the judge to answer
    # for. Reported as counts because the artifact to fix differs: the rubric for one,
    # the judge for the other. In general the remainder also holds the splits that could
    # not be placed either way, which is why `rule_undetermined` travels with these two.
    assert report.disagreements_total - report.rule_disagreements == 1

    # The fact both sides agreed on, which is what makes the split provably a rule
    # difference rather than two readings of the ending.
    ending = _criterion(report, "ended_waiting")
    assert ending.n == 2  # the rule case and the settled one; the refund case skipped it
    assert ending.observed_agreement == 1.0
    assert (ending.human_positive, ending.judge_positive) == (2, 2)

    # goal_achieved is where the perception case parts company: 1.0 against 0.5.
    goal = _criterion(report, "goal_achieved")
    assert goal.n == 3
    assert goal.observed_agreement == pytest.approx(2 / 3)


def test_a_disagreement_with_no_shared_criteria_is_not_blamed_on_the_rubric(
    session: Session,
    make_judge: Callable[..., Judge],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    """"They agreed on every shared criterion" is vacuously true of a pair sharing none.

    Round one is 29 such labels — collected before the criteria existed, so they answer
    nothing the judge now answers. Counted, they would report the entire first annotation
    round as a rubric fault and send a reader off to rewrite rules that were never
    applied to it.
    """
    judge = make_judge()
    trajectory = make_trajectory()
    make_label(trajectory, "momo", "pass")  # round one: no rubric_scores at all
    make_verdict(judge, trajectory, "fail", rubric_scores=_scores(goal_achieved=0.0))

    report = build_calibration_report(session, judge)

    assert report.disagreements_total == 1
    assert report.rule_disagreements == 0
    # This is the assertion that pins the both-sides-empty regression: with no gaps and
    # no shared answers either, "shared covers everything answered" holds vacuously, and
    # only the explicit empty-overlap check keeps round one out of the rubric bucket.
    assert report.rule_undetermined == 1


def test_a_split_over_one_shared_criterion_is_unmeasured_not_a_rubric_fault(
    session: Session,
    make_judge: Callable[..., Judge],
    make_task: Callable[..., Task],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    """One overlapping answer measures coverage, not the rule.

    Every criterion is optional to an annotator, so a panel that ticks the one question
    it is sure about and leaves the rest alone is the ordinary submission rather than a
    degenerate one. Counted as a rubric fault, this trajectory reports a judge that read
    the run as a total failure against a human `pass` — goal missed, no consent, ended
    mid-air, rambling, none of it answered on the other side — as evidence the rulebook
    needs rewriting, and sends a reader to bump PROMPT_VERSION over a judge that is
    misreading runs.
    """
    judge = make_judge()
    thin = make_trajectory(task=make_task(key="refund-simple-01"))
    for annotator in ("momo", "vamp"):
        make_label(thin, annotator, "pass", rubric_scores=_scores(tools_correct=1.0))
    make_verdict(
        judge,
        thin,
        "fail",
        rubric_scores=_scores(
            goal_achieved=0.0,
            tools_correct=1.0,
            consent_obtained=0.0,
            ended_waiting=0.0,
            communication_clear=0.0,
        ),
    )

    report = build_calibration_report(session, judge)

    assert report.disagreements_total == 1
    assert report.rule_disagreements == 0
    assert report.rule_undetermined == 1
    # The one answer they do share is real and still counts where it belongs.
    assert _criterion(report, "tools_correct").n == 1
    assert report.criteria_unanswered["goal_achieved"] == 1


def test_one_unshared_criterion_is_enough_to_withhold_the_rubric_verdict(
    session: Session,
    make_judge: Callable[..., Judge],
    make_task: Callable[..., Task],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    """The allowance for one-sided criteria is zero, and the first gap crosses it.

    The thin-overlap case above has four gaps and would still be caught if the allowance
    quietly grew; this one has exactly one, so it pins the threshold itself. With five
    criteria a single unmeasured one is a fifth of the rubric, and ``ended_waiting``
    alone is what rule 1 turns on — the whole verdict split can live in the one question
    the two sides never both answered, agreement on everything shared notwithstanding.
    """
    judge = make_judge()
    trajectory = make_trajectory(task=make_task(key="order-status-01"))
    both = _scores(goal_achieved=1.0, tools_correct=1.0)
    for annotator in ("momo", "vamp"):
        make_label(trajectory, annotator, "pass", rubric_scores=both)
    # The judge matches every shared answer and adds one nobody else gave.
    make_verdict(
        judge,
        trajectory,
        "borderline",
        rubric_scores=_scores(goal_achieved=1.0, tools_correct=1.0, ended_waiting=1.0),
    )

    report = build_calibration_report(session, judge)

    assert report.disagreements_total == 1
    assert report.rule_disagreements == 0
    assert report.rule_undetermined == 1


# --- the calibrate text report ------------------------------------------------


def test_cli_report_prints_the_criteria_table_and_attributes_the_splits(
    session: Session,
    make_judge: Callable[..., Judge],
    make_task: Callable[..., Task],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The numbers this feature exists for reach the terminal, not just ``--json``.

    ``rule_disagreements`` is the line that tells a reader which artifact to fix, and
    the per-criterion rows are what the drill-down under them is read with. Computed,
    serialised and never printed, both would leave the text report sending every reader
    off to re-prompt a judge that already reads the runs as the annotators do.
    """
    # The corpus pinned above in the rubric-fault test: one rule split, one perception
    # split, one settled trajectory.
    judge = make_judge()
    agreed = _scores(
        goal_achieved=1.0, tools_correct=1.0, ended_waiting=1.0, communication_clear=1.0
    )
    rule = make_trajectory(task=make_task(key="order-status-01"))
    make_label(rule, "momo", "pass", rubric_scores=agreed)
    make_label(rule, "vamp", "pass", rubric_scores=agreed)
    make_verdict(judge, rule, "borderline", rubric_scores=agreed)
    perception = make_trajectory(task=make_task(key="refund-simple-01"))
    read_as_done = _scores(goal_achieved=1.0, tools_correct=1.0)
    make_label(perception, "momo", "pass", rubric_scores=read_as_done)
    make_label(perception, "vamp", "pass", rubric_scores=read_as_done)
    make_verdict(
        judge,
        perception,
        "borderline",
        rubric_scores=_scores(goal_achieved=0.5, tools_correct=1.0),
    )
    settled = make_trajectory(task=make_task(key="angry-customer-01"))
    make_label(settled, "momo", "pass", rubric_scores=agreed)
    make_verdict(judge, settled, "pass", rubric_scores=agreed)

    _print_calibration_report(build_calibration_report(session, judge))

    out = capsys.readouterr().out
    lines = [" ".join(line.split()) for line in out.splitlines()]

    assert out.isascii()
    heading = "Per-criterion agreement (judge vs the human panel, where both answered)"
    assert heading in lines
    assert "criterion n unanswered kappa agreement human 1.0 judge 1.0" in lines
    # Every criterion gets a row, the empty consent column included: omitting it would
    # print a four-criterion rubric.
    for criterion in CRITERIA:
        assert any(line.startswith(f"{criterion.key} ") for line in lines)
    assert "Nothing measured yet" not in out

    # The attribution adds up in plain sight: 1 rule, 1 perception, nothing withheld.
    assert "Rubric disagreements: 1 of 2." in lines
    assert any(line.startswith("Perception differences: 1 of 2.") for line in lines)
    assert not any(line.startswith("Undetermined:") for line in lines)

    # The table sits between the human numbers and the drill-down it explains, and the
    # attribution prints above the rationales — before anyone reads the first one as a
    # judge error.
    assert lines.index("Human agreement") < lines.index(heading)
    assert lines.index(heading) < lines.index("Worst disagreements (2)")
    assert lines.index("Worst disagreements (2)") < lines.index("Rubric disagreements: 1 of 2.")


def test_cli_report_says_nothing_measured_instead_of_five_empty_kappa_rows(
    session: Session,
    make_judge: Callable[..., Judge],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Round one: verdicts to compare, criterion answers on neither side.

    Five n=0 rows would read as five criteria the two sides never once agreed on, and
    "Rubric disagreements: 0 of 1" would clear a rubric nobody measured. Both print an
    empty state instead: a number computed over nothing has to say so.
    """
    judge = make_judge()
    for verdict in ("pass", "pass", "fail"):
        trajectory = make_trajectory()
        make_label(trajectory, "momo", "pass")
        make_verdict(judge, trajectory, verdict)

    _print_calibration_report(build_calibration_report(session, judge))

    out = capsys.readouterr().out
    lines = [" ".join(line.split()) for line in out.splitlines()]

    heading = "Per-criterion agreement (judge vs the human panel, where both answered)"
    assert heading in lines
    assert any(line.startswith("Nothing measured yet:") for line in lines)

    # The empty state replaces the table outright: no criterion row prints anywhere in
    # the report, and no kappa of any spelling is offered over the n=0 columns.
    criteria_block = out.split("Per-criterion agreement", 1)[1].split("Worst disagreements", 1)[0]
    for criterion in CRITERIA:
        assert criterion.key not in out
    assert "kappa" not in criteria_block
    assert "unanswered" not in criteria_block

    # The one split is unattributable and stays that way: n/a, never a zero that reads
    # as a clean rubric.
    assert any(line.startswith("Rubric disagreements: n/a.") for line in lines)
    assert "Rubric disagreements: 0 of" not in out
    assert not any(line.startswith("Perception differences:") for line in lines)
    assert not any(line.startswith("Undetermined:") for line in lines)


def test_cli_report_counts_a_thin_overlap_corpus_instead_of_calling_it_unmeasured(
    session: Session,
    make_judge: Callable[..., Judge],
    make_task: Callable[..., Task],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One shared answer is enough to retire n/a: the undetermined count is measured.

    The n/a branch belongs to the corpus where nothing was answered on both sides. Gated
    on the rule and perception counts instead, it also fires when every split is
    undetermined over a partial overlap -- and then the same screen shows a criteria row
    with n=1 three lines above a sentence claiming no trajectory carries shared answers,
    while the measured undetermined tally reaches no reader at all.
    """
    judge = make_judge()
    thin = make_trajectory(task=make_task(key="refund-simple-01"))
    # The ordinary thin submission LABELING.md invites: the panel answers the one
    # criterion it is sure about, the judge answers all five and reads the run as a
    # total failure.
    for annotator in ("momo", "vamp"):
        make_label(thin, annotator, "pass", rubric_scores=_scores(tools_correct=1.0))
    make_verdict(
        judge,
        thin,
        "fail",
        rubric_scores=_scores(
            goal_achieved=0.0,
            tools_correct=1.0,
            consent_obtained=0.0,
            ended_waiting=0.0,
            communication_clear=0.0,
        ),
    )

    _print_calibration_report(build_calibration_report(session, judge))

    out = capsys.readouterr().out
    lines = [" ".join(line.split()) for line in out.splitlines()]

    # The table above the drill-down shows the shared answer as measured...
    assert any(line.startswith("tools_correct 1 ") for line in lines)
    assert "Nothing measured yet" not in out

    # ...so the drill-down must count, not shrug: n/a would contradict that row, and
    # the undetermined split is the number that says what to do next (label more).
    assert "Rubric disagreements: n/a" not in out
    assert "No compared trajectory carries" not in out
    assert "Rubric disagreements: 0 of 1." in lines
    assert any(line.startswith("Undetermined: 1 of 1.") for line in lines)
    assert not any(line.startswith("Perception differences:") for line in lines)
