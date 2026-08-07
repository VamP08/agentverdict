"""The calibration surfaces: the JSON endpoint and the server-rendered lab pages.

These assert what a caller actually receives — the compared count, the kappa, the
matrix orientation on the rendered page, and the link that takes a reviewer from a
disagreement to the trajectory that produced it. The empty state gets its own
tests: rendering a zero-filled matrix would read as a finding rather than as
"nothing has been measured yet".
"""

from collections.abc import Callable
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from agentverdict.models import EvalRun, HumanLabel, Judge, JudgeVerdict, Task, Trajectory

#: Four comparable trajectories: 2 pass/pass, 1 fail/fail, 1 fail called pass.
#: p_o = 3/4 = 0.75. Human marginals pass 0.50 / fail 0.50; judge marginals
#: pass 0.75 / fail 0.25, so p_e = 0.50*0.75 + 0.50*0.25 = 0.50 and
#: kappa = (0.75 - 0.50) / (1 - 0.50) = 0.5 ("moderate").
SEEDED = [
    ("refund-clean-01", "pass", "pass", "Verified the order before refunding."),
    ("refund-clean-02", "pass", "pass", "Followed the refund policy exactly."),
    ("order-status-01", "fail", "fail", "Never looked the order up."),
    ("refund-missing-01", "fail", "pass", "Sounded confident, so I passed it."),
]
EXPECTED_KAPPA = 0.5
EXPECTED_OBSERVED = 0.75


@pytest.fixture
def calibrated_judge(
    make_judge: Callable[..., Judge],
    make_task: Callable[..., Task],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> tuple[Judge, dict[str, Trajectory]]:
    """A judge with the SEEDED comparison above, plus the trajectories by task key."""
    judge = make_judge(name="groq-70b", model="llama-test-model")
    trajectories: dict[str, Trajectory] = {}
    for key, human, judged, rationale in SEEDED:
        trajectory = make_trajectory(task=make_task(key=key))
        trajectories[key] = trajectory
        make_label(trajectory, "alice", human)
        make_verdict(judge, trajectory, judged, rationale=rationale)
    return judge, trajectories


# --- GET /api/judges/{id} -----------------------------------------------------


def test_get_judge_returns_the_judge(
    client: TestClient, make_judge: Callable[..., Judge]
) -> None:
    judge = make_judge(name="groq-70b", model="llama-test-model", description="reference judge")

    response = client.get(f"/api/judges/{judge.id}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == judge.id
    assert body["name"] == "groq-70b"
    assert body["model"] == "llama-test-model"
    assert body["description"] == "reference judge"


def test_get_judge_unknown_id_returns_404(client: TestClient) -> None:
    response = client.get("/api/judges/does-not-exist")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


# --- GET /api/judges/{id}/calibration ----------------------------------------


def test_calibration_endpoint_reports_the_seeded_agreement(
    client: TestClient, calibrated_judge: tuple[Judge, dict[str, Trajectory]]
) -> None:
    judge, trajectories = calibrated_judge

    response = client.get(f"/api/judges/{judge.id}/calibration")

    assert response.status_code == 200
    body = response.json()
    assert body["judge_id"] == judge.id
    assert body["judge_name"] == "groq-70b"
    assert body["judge_model"] == "llama-test-model"
    assert body["eval_run_id"] is None
    assert body["task_key"] is None
    assert body["compared"] == 4
    assert body["judged_trajectories"] == 4
    assert body["labeled_trajectories"] == 4
    assert body["judge_errors"] == 0
    assert body["ties_excluded"] == 0

    agreement = body["agreement"]
    assert agreement["n"] == 4
    assert agreement["cohens_kappa"] == pytest.approx(EXPECTED_KAPPA)
    assert agreement["observed_agreement"] == pytest.approx(EXPECTED_OBSERVED)
    assert agreement["interpretation"] == "moderate"
    # matrix[human][judge]: the single miss is a human "fail" the judge passed.
    assert agreement["matrix"]["fail"]["pass"] == 1
    assert agreement["matrix"]["pass"]["fail"] == 0
    assert agreement["matrix"]["pass"]["pass"] == 2

    assert len(body["disagreements"]) == 1
    miss = body["disagreements"][0]
    assert miss["trajectory_id"] == trajectories["refund-missing-01"].id
    assert miss["task_key"] == "refund-missing-01"
    assert miss["human_verdict"] == "fail"
    assert miss["judge_verdict"] == "pass"
    assert miss["distance"] == 2
    assert miss["judge_rationale"] == "Sounded confident, so I passed it."
    assert miss["annotators"] == ["alice"]


def test_calibration_endpoint_accepts_a_task_key_filter(
    client: TestClient, calibrated_judge: tuple[Judge, dict[str, Trajectory]]
) -> None:
    judge, _ = calibrated_judge

    response = client.get(
        f"/api/judges/{judge.id}/calibration", params={"task_key": "refund-missing-01"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task_key"] == "refund-missing-01"
    assert body["compared"] == 1
    assert body["agreement"]["observed_agreement"] == 0.0
    # One item is not enough to resample, so there is no interval to report.
    assert body["agreement"]["kappa_ci_low"] is None


def test_calibration_endpoint_unknown_judge_returns_404(client: TestClient) -> None:
    response = client.get("/api/judges/does-not-exist/calibration")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_calibration_endpoint_unknown_eval_run_returns_404(
    client: TestClient, calibrated_judge: tuple[Judge, dict[str, Trajectory]]
) -> None:
    judge, _ = calibrated_judge

    response = client.get(
        f"/api/judges/{judge.id}/calibration", params={"eval_run_id": "no-such-run"}
    )

    # Not an empty report: zero comparisons would read as a judge with nothing
    # to answer for, when in fact the caller mistyped a run id.
    assert response.status_code == 404
    assert "Eval run" in response.json()["detail"]


def test_calibration_endpoint_rejects_another_judges_eval_run(
    client: TestClient,
    calibrated_judge: tuple[Judge, dict[str, Trajectory]],
    make_judge: Callable[..., Judge],
    make_eval_run: Callable[..., EvalRun],
) -> None:
    judge, _ = calibrated_judge
    stranger = make_judge(name="other-judge")
    foreign_run = make_eval_run(stranger)

    response = client.get(
        f"/api/judges/{judge.id}/calibration", params={"eval_run_id": foreign_run.id}
    )

    assert response.status_code == 404
    assert "does not belong" in response.json()["detail"]


def test_calibration_endpoint_scopes_to_one_eval_run(
    client: TestClient,
    make_judge: Callable[..., Judge],
    make_eval_run: Callable[..., EvalRun],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    judge = make_judge()
    trajectory = make_trajectory()
    make_label(trajectory, "alice", "pass")
    first_run = make_eval_run(judge)
    second_run = make_eval_run(judge)
    make_verdict(
        judge,
        trajectory,
        "fail",
        eval_run=first_run,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    make_verdict(
        judge,
        trajectory,
        "pass",
        eval_run=second_run,
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    old = client.get(
        f"/api/judges/{judge.id}/calibration", params={"eval_run_id": first_run.id}
    ).json()
    new = client.get(
        f"/api/judges/{judge.id}/calibration", params={"eval_run_id": second_run.id}
    ).json()

    assert old["eval_run_id"] == first_run.id
    assert old["compared"] == 1
    assert old["agreement"]["matrix"]["pass"]["fail"] == 1
    assert [row["judge_verdict"] for row in old["disagreements"]] == ["fail"]

    assert new["compared"] == 1
    assert new["agreement"]["matrix"]["pass"]["pass"] == 1
    assert new["disagreements"] == []


def test_calibration_endpoint_empty_state_is_a_200_with_nulls(
    client: TestClient, make_judge: Callable[..., Judge]
) -> None:
    judge = make_judge(name="fresh-judge")

    response = client.get(f"/api/judges/{judge.id}/calibration")

    assert response.status_code == 200
    body = response.json()
    assert body["compared"] == 0
    assert body["agreement"]["n"] == 0
    assert body["agreement"]["cohens_kappa"] is None
    assert body["agreement"]["observed_agreement"] is None
    assert body["agreement"]["interpretation"] == "not enough data"
    assert body["human_ceiling"] == []
    assert body["disagreements"] == []


# --- GET /calibration ---------------------------------------------------------


def test_calibration_index_lists_the_seeded_judge(
    client: TestClient, calibrated_judge: tuple[Judge, dict[str, Trajectory]]
) -> None:
    judge, _ = calibrated_judge

    response = client.get("/calibration")

    assert response.status_code == 200
    html = response.text
    assert "Judge calibration" in html
    assert "groq-70b" in html
    assert "llama-test-model" in html
    assert f'href="/calibration/{judge.id}"' in html
    assert "0.500" in html  # the kappa, rendered to three places
    assert "moderate" in html


def test_calibration_index_marks_an_uncalibrated_judge(
    client: TestClient,
    calibrated_judge: tuple[Judge, dict[str, Trajectory]],
    make_judge: Callable[..., Judge],
) -> None:
    judge, _ = calibrated_judge
    fresh = make_judge(name="zz-fresh-judge")

    html = client.get("/calibration").text

    assert f'href="/calibration/{judge.id}"' in html
    assert f'href="/calibration/{fresh.id}"' in html
    assert "not yet calibrated" in html
    assert "n/a" in html  # an undefined kappa never renders as 0.000
    assert "<strong>1</strong> of <strong>2</strong>" in html


def test_calibration_index_renders_without_judges(client: TestClient) -> None:
    response = client.get("/calibration")
    assert response.status_code == 200
    assert "No judges registered yet." in response.text
    assert "<table" not in response.text  # no empty grid of dashes to misread


# --- GET /calibration/{id} ----------------------------------------------------


def test_calibration_detail_renders_matrix_and_drill_down(
    client: TestClient, calibrated_judge: tuple[Judge, dict[str, Trajectory]]
) -> None:
    judge, trajectories = calibrated_judge

    response = client.get(f"/calibration/{judge.id}")

    assert response.status_code == 200
    html = response.text

    # Headline and coverage.
    assert "groq-70b" in html
    assert "0.500" in html
    assert "moderate" in html
    assert "75.0%" in html  # observed agreement

    # The matrix, with its orientation spelled out for the reader.
    assert "Confusion matrix" in html
    assert 'class="matrix"' in html
    assert "human \\ judge" in html
    assert 'title="humans said fail, judge said pass"' in html
    assert 'title="humans said pass, judge said pass"' in html

    # The drill-down: one row, linking to the trajectory that produced it.
    assert "Disagreements (1)" in html
    missed = trajectories["refund-missing-01"]
    assert f'href="/label/{missed.id}"' in html
    assert "Sounded confident, so I passed it." in html
    assert "refund-missing-01" in html
    # Trajectories the judge got right must not appear in the drill-down list.
    agreed_id = trajectories["refund-clean-01"].id
    assert f'href="/label/{agreed_id}"' not in html

    # Per-verdict breakdown is rendered once there is something to break down.
    assert "Per-verdict breakdown" in html


def test_calibration_detail_shows_the_human_ceiling(
    client: TestClient,
    make_judge: Callable[..., Judge],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    judge = make_judge()
    agreed = make_trajectory()
    split = make_trajectory()
    make_label(agreed, "alice", "pass")
    make_label(agreed, "bob", "pass")
    make_label(split, "alice", "pass")
    make_label(split, "bob", "fail")
    make_verdict(judge, agreed, "pass")

    html = client.get(f"/calibration/{judge.id}").text

    assert "Human agreement" in html
    assert "alice" in html
    assert "bob" in html
    # alice and bob share both trajectories and agree on one of them: raw
    # agreement 50%. Kappa is undefined rather than 0.0 — alice never varied, and
    # a rater with no variance has no chance model to correct against.
    assert "50.0%" in html
    # Both framings are present: the held-out baseline the judge should be read
    # against, and pairwise agreement for spotting a divergent grader.
    assert "majority of the others" in html
    assert "Annotator against annotator" in html
    # Meanwhile the judge's own kappa is undefined on its single compared
    # trajectory: "n/a" and a stated reason, never a fabricated 1.000.
    assert "n/a" in html
    assert "No confidence interval" in html


def test_calibration_detail_empty_state_has_no_zero_filled_table(
    client: TestClient, make_judge: Callable[..., Judge]
) -> None:
    judge = make_judge(name="fresh-judge")

    response = client.get(f"/calibration/{judge.id}")

    assert response.status_code == 200
    html = response.text
    assert "Nothing to compare yet" in html
    # None of the statistics tables may render: zeros would look like findings.
    assert "Confusion matrix" not in html
    assert 'class="matrix"' not in html
    assert "Per-verdict breakdown" not in html
    assert "Disagreements" not in html
    # Instead it names the two commands that produce something to compare.
    assert "agentverdict judge run fresh-judge" in html
    assert 'href="/label"' in html


def test_calibration_detail_empty_state_explains_a_one_sided_dataset(
    client: TestClient,
    make_judge: Callable[..., Judge],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
) -> None:
    judge = make_judge()
    make_label(make_trajectory(), "alice", "pass")
    make_label(make_trajectory(), "alice", "fail")

    html = client.get(f"/calibration/{judge.id}").text

    assert "Nothing to compare yet" in html
    assert "this judge has not scored any of them yet" in html
    assert "Confusion matrix" not in html


def test_calibration_detail_escapes_a_rationale_containing_markup(
    client: TestClient,
    make_judge: Callable[..., Judge],
    make_trajectory: Callable[..., Trajectory],
    make_label: Callable[..., HumanLabel],
    make_verdict: Callable[..., JudgeVerdict],
) -> None:
    judge = make_judge()
    trajectory = make_trajectory()
    make_label(trajectory, "alice", "pass")
    make_verdict(
        judge,
        trajectory,
        "fail",
        rationale='<script>alert("xss")</script> tools & policy <b>ignored</b>',
    )

    response = client.get(f"/calibration/{judge.id}")

    assert response.status_code == 200
    html = response.text
    # A judge rationale is model output: it must never reach the page as markup.
    assert "<script>alert" not in html
    assert "<b>ignored</b>" not in html
    assert "&lt;script&gt;" in html
    assert "&lt;/script&gt;" in html
    assert "&lt;b&gt;ignored&lt;/b&gt;" in html
    assert "tools &amp; policy" in html


def test_calibration_detail_unknown_judge_returns_404(client: TestClient) -> None:
    assert client.get("/calibration/does-not-exist").status_code == 404
