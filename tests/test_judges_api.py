"""Judge API endpoints and UI rendering of judge verdicts."""

from collections.abc import Callable

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from agentverdict.models import EvalRun, Judge, JudgeVerdict, Trajectory


def _seed_verdict(session: Session, trajectory: Trajectory) -> Judge:
    judge = Judge(name="groq-70b", model="llama-test-model", description="reference judge")
    run = EvalRun(judge=judge)
    verdict = JudgeVerdict(
        eval_run=run,
        judge=judge,
        trajectory_id=trajectory.id,
        verdict="pass",
        rationale="Correct tool use throughout.",
        latency_ms=812.0,
    )
    session.add_all([judge, run, verdict])
    session.commit()
    return judge


def test_list_judges_empty(client: TestClient) -> None:
    response = client.get("/api/judges")
    assert response.status_code == 200
    assert response.json() == []


def test_list_judges_and_trajectory_verdicts(
    client: TestClient,
    session: Session,
    make_trajectory: Callable[..., Trajectory],
) -> None:
    trajectory = make_trajectory()
    _seed_verdict(session, trajectory)

    judges = client.get("/api/judges").json()
    assert [judge["name"] for judge in judges] == ["groq-70b"]
    assert judges[0]["model"] == "llama-test-model"

    response = client.get(f"/api/trajectories/{trajectory.id}/verdicts")
    assert response.status_code == 200
    verdicts = response.json()
    assert len(verdicts) == 1
    assert verdicts[0]["verdict"] == "pass"
    assert verdicts[0]["rationale"] == "Correct tool use throughout."
    assert verdicts[0]["error"] is None


def test_trajectory_verdicts_unknown_trajectory_404(client: TestClient) -> None:
    response = client.get("/api/trajectories/does-not-exist/verdicts")
    assert response.status_code == 404


def test_ui_shows_judge_verdicts_after_label_form(
    client: TestClient,
    session: Session,
    make_trajectory: Callable[..., Trajectory],
) -> None:
    trajectory = make_trajectory()
    page_before = client.get(f"/label/{trajectory.id}")
    assert page_before.status_code == 200
    assert "Judge verdicts" not in page_before.text

    _seed_verdict(session, trajectory)
    page = client.get(f"/label/{trajectory.id}")
    assert page.status_code == 200
    body = page.text
    assert "Judge verdicts (1)" in body
    assert "groq-70b" in body
    assert "Correct tool use throughout." in body
    assert body.index("label-form") < body.index("Judge verdicts")
