"""Tests for the server-rendered labeling UI under /label."""

from collections.abc import Callable

from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from agentverdict.models import HumanLabel, Task, Trajectory


def test_queue_lists_task_keys_unlabeled_first(
    client: TestClient,
    session: Session,
    make_task: Callable[..., Task],
    make_trajectory: Callable[..., Trajectory],
) -> None:
    labeled_task = make_task(key="refund-simple-01")
    unlabeled_task = make_task(key="order-status-01")
    labeled = make_trajectory(task=labeled_task)
    unlabeled = make_trajectory(task=unlabeled_task)
    session.add(HumanLabel(trajectory_id=labeled.id, annotator="alice", verdict="pass"))
    session.commit()

    response = client.get("/label")
    assert response.status_code == 200
    html = response.text
    assert "refund-simple-01" in html
    assert "order-status-01" in html
    assert f"/label/{unlabeled.id}" in html
    assert f"/label/{labeled.id}" in html
    assert html.index("order-status-01") < html.index("refund-simple-01")


def test_queue_renders_when_empty(client: TestClient) -> None:
    response = client.get("/label")
    assert response.status_code == 200
    assert "No trajectories yet." in response.text


def test_detail_shows_prompt_and_steps(
    client: TestClient,
    make_task: Callable[..., Task],
    make_trajectory: Callable[..., Trajectory],
) -> None:
    task = make_task(
        key="refund-simple-01",
        prompt="Handle a refund for a damaged AeroMouse without skipping verification.",
    )
    trajectory = make_trajectory(task=task)

    response = client.get(f"/label/{trajectory.id}")
    assert response.status_code == 200
    html = response.text
    assert task.prompt in html
    assert "refund-simple-01" in html
    assert "lookup_order" in html


def test_detail_unknown_trajectory_returns_404(client: TestClient) -> None:
    assert client.get("/label/missing").status_code == 404


def test_post_label_persists_and_redirects_to_queue(
    client: TestClient,
    make_trajectory: Callable[..., Trajectory],
    session_factory: sessionmaker[Session],
) -> None:
    trajectory = make_trajectory()
    response = client.post(
        f"/label/{trajectory.id}",
        data={
            "annotator": "alice",
            "verdict": "pass",
            "rationale": "Order verified before refunding.",
        },
        follow_redirects=False,
    )
    assert response.status_code in (302, 303)
    assert response.headers["location"] == "/label"

    with session_factory() as check:
        labels = check.scalars(
            select(HumanLabel).where(HumanLabel.trajectory_id == trajectory.id)
        ).all()
    assert len(labels) == 1
    assert labels[0].annotator == "alice"
    assert labels[0].verdict == "pass"
    assert labels[0].rationale == "Order verified before refunding."


def test_post_label_redirects_to_next_unlabeled(
    client: TestClient, make_trajectory: Callable[..., Trajectory]
) -> None:
    first = make_trajectory()
    second = make_trajectory()
    response = client.post(
        f"/label/{first.id}",
        data={"annotator": "alice", "verdict": "fail"},
        follow_redirects=False,
    )
    assert response.status_code in (302, 303)
    assert response.headers["location"] == f"/label/{second.id}"


def test_post_label_invalid_form_does_not_persist(
    client: TestClient,
    make_trajectory: Callable[..., Trajectory],
    session_factory: sessionmaker[Session],
) -> None:
    trajectory = make_trajectory()
    response = client.post(
        f"/label/{trajectory.id}",
        data={"annotator": "alice", "verdict": "maybe"},
        follow_redirects=False,
    )
    assert response.status_code == 400

    with session_factory() as check:
        label_count = check.scalar(select(func.count()).select_from(HumanLabel))
    assert label_count == 0


def test_post_label_unknown_trajectory_returns_404(client: TestClient) -> None:
    response = client.post(
        "/label/missing",
        data={"annotator": "alice", "verdict": "pass"},
        follow_redirects=False,
    )
    assert response.status_code == 404
