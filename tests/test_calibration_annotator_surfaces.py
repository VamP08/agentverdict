"""Where the per-rater comparison and the annotator scope reach a caller.

Three surfaces carry the same two additions, so each is asserted where a user
actually meets it: the repeatable ``?annotator=`` query parameter on the JSON
endpoint, the per-rater table on the calibration page, and the per-rater block in
the terminal report. The page tests matter most: the block has to appear precisely
when there is a judge and an annotator on the same trajectory, including the case
where every trajectory tied and the headline can say nothing at all.
"""

from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from agentverdict.calibration.report import build_calibration_report
from agentverdict.cli import _print_calibration_report
from agentverdict.judging.prompts import PROMPT_VERSION
from agentverdict.models import EvalRun, HumanLabel, Judge, JudgeVerdict, Trajectory
from agentverdict.rubric import RUBRIC_NAME, RUBRIC_VERSION, ensure_rubric

#: The heading the per-rater table renders under, and the start of the next block.
PER_RATER_HEADING = "<h2>Judge against each annotator</h2>"


def _section(html: str, heading: str) -> str:
    """The markup between a heading and the next ``<h2>``, so assertions stay local."""
    assert heading in html
    return html.split(heading, 1)[1].split("<h2>", 1)[0]


@pytest.fixture
def split_panel(
    make_judge: Callable[..., Judge],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> tuple[Judge, dict[str, Trajectory]]:
    """vamp and momo agree twice and split once; the judge scored all three.

    Headline: 2 compared, 1 tie excluded, a perfect kappa. Per rater: 3 items each,
    kappa 0.400 against vamp and 1.000 against momo. The numbers are pinned in
    ``test_calibration_per_rater.py``; here they are only used as recognisable
    output.
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


# --- GET /api/judges/{id}/calibration?annotator= ------------------------------


def test_calibration_endpoint_reports_the_per_rater_rows(
    client: TestClient, split_panel: tuple[Judge, dict[str, Trajectory]]
) -> None:
    judge, _ = split_panel

    body = client.get(f"/api/judges/{judge.id}/calibration").json()

    assert body["annotators"] == []  # no filter asked for, none reported
    assert body["compared"] == 2
    assert body["ties_excluded"] == 1
    assert [(row["annotator"], row["n"]) for row in body["judge_vs_annotator"]] == [
        ("momo", 3),
        ("vamp", 3),
    ]
    rows = {row["annotator"]: row for row in body["judge_vs_annotator"]}
    # Each row is scored on one more trajectory than the headline: the tied one.
    assert all(row["n"] > body["compared"] for row in body["judge_vs_annotator"])
    assert rows["vamp"]["cohens_kappa"] == pytest.approx(0.4)
    assert rows["momo"]["cohens_kappa"] == pytest.approx(1.0)
    assert rows["vamp"]["observed_agreement"] == pytest.approx(2 / 3)


def test_calibration_endpoint_accepts_a_repeated_annotator_parameter(
    client: TestClient, split_panel: tuple[Judge, dict[str, Trajectory]]
) -> None:
    judge, _ = split_panel

    response = client.get(
        f"/api/judges/{judge.id}/calibration", params={"annotator": ["vamp", "momo"]}
    )

    assert response.status_code == 200
    body = response.json()
    # The scope is echoed back sorted, so a caller can see what was measured.
    assert body["annotators"] == ["momo", "vamp"]
    assert body["compared"] == 2
    assert body["ties_excluded"] == 1
    assert [row["annotator"] for row in body["judge_vs_annotator"]] == ["momo", "vamp"]


def test_calibration_endpoint_scopes_to_a_single_annotator(
    client: TestClient, split_panel: tuple[Judge, dict[str, Trajectory]]
) -> None:
    judge, trajectories = split_panel

    body = client.get(
        f"/api/judges/{judge.id}/calibration", params={"annotator": "vamp"}
    ).json()

    assert body["annotators"] == ["vamp"]
    # A panel of one never ties, so the trajectory the headline had dropped comes
    # back — and with it the disagreement the tie was hiding.
    assert body["ties_excluded"] == 0
    assert body["compared"] == 3
    assert body["labeled_trajectories"] == 3
    assert body["judged_trajectories"] == 3
    assert [row["trajectory_id"] for row in body["disagreements"]] == [
        trajectories["split"].id
    ]
    assert [row["annotator"] for row in body["judge_vs_annotator"]] == ["vamp"]
    assert body["human_ceiling"] == []
    assert body["human_baseline"] == []


def test_calibration_endpoint_unknown_annotator_is_an_empty_report_not_an_error(
    client: TestClient, split_panel: tuple[Judge, dict[str, Trajectory]]
) -> None:
    judge, _ = split_panel

    response = client.get(
        f"/api/judges/{judge.id}/calibration", params={"annotator": "vamp-r2"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["annotators"] == ["vamp-r2"]
    assert body["labeled_trajectories"] == 0
    assert body["compared"] == 0
    assert body["judge_vs_annotator"] == []
    assert body["agreement"]["cohens_kappa"] is None
    # The judge's coverage is untouched, which is what makes the empty comparison
    # readable as a mistyped name rather than as an unjudged dataset.
    assert body["judged_trajectories"] == 3


def test_calibration_endpoint_404s_still_win_over_the_annotator_scope(
    client: TestClient,
    split_panel: tuple[Judge, dict[str, Trajectory]],
    make_judge: Callable[..., Judge],
    make_eval_run: Callable[..., EvalRun],
) -> None:
    judge, _ = split_panel
    foreign_run = make_eval_run(make_judge(name="other-judge"))
    scope = {"annotator": ["vamp", "momo"]}

    unknown_judge = client.get("/api/judges/does-not-exist/calibration", params=scope)
    assert unknown_judge.status_code == 404
    assert "not found" in unknown_judge.json()["detail"]

    unknown_run = client.get(
        f"/api/judges/{judge.id}/calibration",
        params={**scope, "eval_run_id": "no-such-run"},
    )
    assert unknown_run.status_code == 404
    assert "Eval run" in unknown_run.json()["detail"]

    wrong_owner = client.get(
        f"/api/judges/{judge.id}/calibration",
        params={**scope, "eval_run_id": foreign_run.id},
    )
    assert wrong_owner.status_code == 404
    assert "does not belong" in wrong_owner.json()["detail"]


# --- GET /calibration/{id} ----------------------------------------------------


def test_calibration_detail_renders_the_per_rater_table(
    client: TestClient, split_panel: tuple[Judge, dict[str, Trajectory]]
) -> None:
    judge, _ = split_panel

    response = client.get(f"/calibration/{judge.id}")

    assert response.status_code == 200
    html = response.text
    section = _section(html, PER_RATER_HEADING)

    assert "vamp" in section
    assert "momo" in section
    # Three items per rater against a headline computed on two: the extra one is
    # the trajectory the annotators tied on.
    assert '<td class="num">3</td>' in section
    assert '<td class="num">0.400</td>' in section  # the judge against vamp
    assert '<td class="num">1.000</td>' in section  # the judge against momo
    assert '<td class="num">66.7%</td>' in section

    # The page says why the table exists, so the headline above it is not misread.
    assert "scored on exactly the trajectories that annotator and the judge both rated" in section
    assert "The headline kappa cannot be" in section
    assert "every trajectory where the annotators tied" in section

    # It sits between the headline and the human numbers it shares an item set with.
    assert html.index("Agreement with the human majority") < html.index(PER_RATER_HEADING)
    assert html.index(PER_RATER_HEADING) < html.index("<h2>Human agreement</h2>")


def test_calibration_detail_renders_the_per_rater_table_when_everything_tied(
    client: TestClient,
    make_judge: Callable[..., Judge],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    """The case the feature was built for: no majority anywhere, so no headline."""
    judge = make_judge(name="groq-70b")
    trajectory = make_trajectory()
    make_label(trajectory, "vamp", "pass")
    make_label(trajectory, "momo", "fail")
    make_verdict(judge, trajectory, "pass")

    html = client.get(f"/calibration/{judge.id}").text

    # The headline has nothing: one labeled trajectory, one judged, zero compared.
    assert "Nothing to compare yet" in html
    assert "Confusion matrix" not in html
    assert "with no human majority" in html

    # The per-rater block still reports what the judge actually did.
    section = _section(html, PER_RATER_HEADING)
    assert "vamp" in section
    assert "momo" in section
    assert "n/a" in section  # one item cannot support a kappa; never a fake 0.000


def test_calibration_detail_omits_the_per_rater_table_with_nothing_to_compare(
    client: TestClient,
    make_judge: Callable[..., Judge],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    fresh = make_judge(name="fresh-judge")
    assert PER_RATER_HEADING not in client.get(f"/calibration/{fresh.id}").text

    # Labels exist, but this judge has scored none of those trajectories.
    labeled_only = make_judge(name="labeled-only")
    make_label(make_trajectory(), "vamp", "pass")
    assert PER_RATER_HEADING not in client.get(f"/calibration/{labeled_only.id}").text

    # Verdicts exist, but on trajectories nobody labeled.
    judged_only = make_judge(name="judged-only")
    make_verdict(judged_only, make_trajectory(), "pass")
    html = client.get(f"/calibration/{judged_only.id}").text
    assert PER_RATER_HEADING not in html
    assert "Nothing to compare yet" in html


# --- agentverdict calibrate ---------------------------------------------------


def test_cli_report_prints_the_per_rater_block_with_its_caveat(
    session: Session,
    split_panel: tuple[Judge, dict[str, Trajectory]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    judge, _ = split_panel
    report = build_calibration_report(session, judge)

    _print_calibration_report(report)

    out = capsys.readouterr().out
    lines = [" ".join(line.split()) for line in out.splitlines()]

    assert "Judge against each annotator (same items, like-for-like)" in lines
    assert "momo n=3 kappa 1.000 agreement 100.0%" in lines
    assert "vamp n=3 kappa 0.400 agreement 66.7%" in lines
    # The caveat is the point: without it the block reads as three comparable
    # numbers, and the headline above is not one of them.
    assert any("is measured on a different sample" in line for line in lines)

    # Printed after the headline and before the human numbers it can be read with.
    headline = lines.index("Agreement with the human majority")
    per_rater = lines.index("Judge against each annotator (same items, like-for-like)")
    humans = lines.index("Human agreement")
    assert headline < per_rater < humans


def test_cli_report_names_the_annotator_scope_it_used(
    session: Session,
    split_panel: tuple[Judge, dict[str, Trajectory]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    judge, _ = split_panel
    report = build_calibration_report(session, judge, annotators=["vamp", "momo"])

    _print_calibration_report(report)

    lines = [" ".join(line.split()) for line in capsys.readouterr().out.splitlines()]
    assert "scope annotators momo, vamp" in lines
    assert "compared 3" not in lines  # the pair still ties, so the headline keeps 2
    assert "compared 2" in lines


def test_cli_report_blames_the_annotator_filter_for_an_empty_report(
    session: Session,
    split_panel: tuple[Judge, dict[str, Trajectory]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A mistyped name looks exactly like an unlabeled dataset unless it is named."""
    judge, _ = split_panel
    report = build_calibration_report(session, judge, annotators=["vamp-r2"])

    _print_calibration_report(report)

    out = capsys.readouterr().out
    lines = [" ".join(line.split()) for line in out.splitlines()]
    assert "scope annotators vamp-r2" in lines
    assert "No agreement statistics yet" in lines
    assert any(
        "Only labels from vamp-r2 count here; drop --annotator" in line for line in lines
    )
    # No per-rater block: there is no annotator in scope to compare against.
    assert not any("Judge against each annotator" in line for line in lines)


def test_cli_report_warns_about_a_rubric_mix_until_the_scope_names_one_round(
    session: Session,
    make_judge: Callable[..., Judge],
    make_eval_run: Callable[..., EvalRun],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The mixed-rounds warning prints before any statistic, and scoping clears it.

    The warning's own advice is to scope the report to one annotation round, so the
    annotator scope has to actually silence it: a warning that survived a reader
    following its instructions would be read as noise from the second sighting on.
    """
    judge = make_judge()
    run = make_eval_run(judge, judge_prompt_version=PROMPT_VERSION)
    in_force = f"{RUBRIC_NAME} v{RUBRIC_VERSION}"
    round_one = make_trajectory()
    make_label(round_one, "vamp", "pass")  # rubric_id NULL: labeled before the rubric row
    make_verdict(judge, round_one, "pass", eval_run=run)
    round_two = make_trajectory()
    make_label(round_two, "vamp-r2", "pass", rubric_id=ensure_rubric(session).id)
    make_verdict(judge, round_two, "pass", eval_run=run)

    _print_calibration_report(build_calibration_report(session, judge))

    mixed = capsys.readouterr().out
    lines = [" ".join(line.split()) for line in mixed.splitlines()]
    assert mixed.isascii()
    assert f"rubric {in_force} (1 label), unrecorded (1 label)" in lines
    assert any(
        line.startswith("WARNING: This report spans more than one rulebook") for line in lines
    )
    # Above the first statistic: a reader who has already taken the headline kappa at
    # face value cannot be un-misled by a footnote.
    assert mixed.index("WARNING") < mixed.index("Agreement with the human majority")

    _print_calibration_report(build_calibration_report(session, judge, annotators=["vamp-r2"]))

    scoped = capsys.readouterr().out
    scoped_lines = [" ".join(line.split()) for line in scoped.splitlines()]
    assert f"rubric {in_force} (1 label)" in scoped_lines
    assert "WARNING" not in scoped
