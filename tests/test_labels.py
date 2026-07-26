"""API tests for /api/trajectories/{id}/labels."""

from collections.abc import Callable

from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from agentverdict.models import HumanLabel, Trajectory

LABEL_PAYLOAD = {
    "annotator": "alice",
    "verdict": "pass",
    "rationale": "Verified the order before refunding.",
    "rubric_scores": {"accuracy": 1.0, "safety": 0.5},
    "time_spent_s": 12.5,
}


def test_create_label(
    client: TestClient, make_trajectory: Callable[..., Trajectory]
) -> None:
    trajectory = make_trajectory()
    response = client.post(f"/api/trajectories/{trajectory.id}/labels", json=LABEL_PAYLOAD)
    assert response.status_code == 201
    body = response.json()
    assert body["trajectory_id"] == trajectory.id
    assert body["annotator"] == "alice"
    assert body["verdict"] == "pass"
    assert body["rationale"] == LABEL_PAYLOAD["rationale"]
    assert body["rubric_scores"] == LABEL_PAYLOAD["rubric_scores"]
    assert body["time_spent_s"] == 12.5
    assert body["id"]


def test_resubmit_same_annotator_replaces_in_place(
    client: TestClient,
    make_trajectory: Callable[..., Trajectory],
    session_factory: sessionmaker[Session],
) -> None:
    trajectory = make_trajectory()
    first = client.post(f"/api/trajectories/{trajectory.id}/labels", json=LABEL_PAYLOAD).json()

    updated = dict(LABEL_PAYLOAD, verdict="fail", rationale="Missed the confirmation step.")
    response = client.post(f"/api/trajectories/{trajectory.id}/labels", json=updated)
    assert response.status_code == 201
    second = response.json()
    assert second["id"] == first["id"]
    assert second["verdict"] == "fail"

    labels = client.get(f"/api/trajectories/{trajectory.id}/labels").json()
    assert len(labels) == 1
    assert labels[0]["verdict"] == "fail"
    assert labels[0]["rationale"] == "Missed the confirmation step."

    with session_factory() as check:
        row_count = check.scalar(select(func.count()).select_from(HumanLabel))
    assert row_count == 1


def test_second_annotator_adds_a_row(
    client: TestClient, make_trajectory: Callable[..., Trajectory]
) -> None:
    trajectory = make_trajectory()
    client.post(f"/api/trajectories/{trajectory.id}/labels", json=LABEL_PAYLOAD)
    response = client.post(
        f"/api/trajectories/{trajectory.id}/labels",
        json={"annotator": "bob", "verdict": "borderline"},
    )
    assert response.status_code == 201

    labels = client.get(f"/api/trajectories/{trajectory.id}/labels").json()
    assert len(labels) == 2
    assert {label["annotator"] for label in labels} == {"alice", "bob"}
    verdicts = {label["annotator"]: label["verdict"] for label in labels}
    assert verdicts == {"alice": "pass", "bob": "borderline"}


def test_unknown_trajectory_returns_404(client: TestClient) -> None:
    assert client.post("/api/trajectories/missing/labels", json=LABEL_PAYLOAD).status_code == 404
    assert client.get("/api/trajectories/missing/labels").status_code == 404


def test_invalid_verdict_returns_422(
    client: TestClient, make_trajectory: Callable[..., Trajectory]
) -> None:
    trajectory = make_trajectory()
    response = client.post(
        f"/api/trajectories/{trajectory.id}/labels",
        json={"annotator": "alice", "verdict": "excellent"},
    )
    assert response.status_code == 422
