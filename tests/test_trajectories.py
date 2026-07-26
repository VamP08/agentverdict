"""API tests for /api/trajectories."""

from collections.abc import Callable
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from agentverdict.models import HumanLabel, Task, Trajectory

STEPS: list[dict[str, Any]] = [
    {"type": "user_message", "content": {"text": "Hi, I want a refund for order A123"}},
    {"type": "tool_call", "content": {"name": "lookup_order", "arguments": {"order_id": "A123"}}},
    {
        "type": "tool_result",
        "content": {"name": "lookup_order", "result": {"status": "delivered"}, "is_error": False},
    },
    {"type": "assistant_message", "content": {"text": "Your refund has been initiated."}},
]


def _create_task(client: TestClient, key: str = "refund-simple-01") -> dict[str, Any]:
    response = client.post(
        "/api/tasks", json={"key": key, "prompt": "Handle a refund for a damaged item."}
    )
    assert response.status_code == 201
    return response.json()


def test_create_via_task_key(client: TestClient) -> None:
    task = _create_task(client)
    response = client.post("/api/trajectories", json={"task_key": task["key"], "steps": STEPS})
    assert response.status_code == 201
    body = response.json()
    assert body["task_id"] == task["id"]
    assert body["source"] == "api"
    assert body["status"] == "completed"
    assert len(body["steps"]) == len(STEPS)


def test_create_via_task_id(client: TestClient) -> None:
    task = _create_task(client)
    payload = {
        "task_id": task["id"],
        "agent_config": {"model": "llama-3.3-70b", "prompt_version": "v2"},
        "source": "manual",
        "status": "truncated",
        "meta": {"note": "run cut off after four steps"},
        "steps": STEPS,
    }
    response = client.post("/api/trajectories", json=payload)
    assert response.status_code == 201
    body = response.json()
    assert body["task_id"] == task["id"]
    assert body["agent_config"] == payload["agent_config"]
    assert body["source"] == "manual"
    assert body["status"] == "truncated"
    assert body["meta"] == payload["meta"]


def test_steps_come_back_ordered_with_position_indexes(client: TestClient) -> None:
    task = _create_task(client)
    created = client.post(
        "/api/trajectories", json={"task_key": task["key"], "steps": STEPS}
    ).json()
    response = client.get(f"/api/trajectories/{created['id']}")
    assert response.status_code == 200
    steps = response.json()["steps"]
    assert [step["index"] for step in steps] == list(range(len(STEPS)))
    assert [step["type"] for step in steps] == [step["type"] for step in STEPS]
    assert [step["content"] for step in steps] == [step["content"] for step in STEPS]


def test_neither_task_reference_returns_400(client: TestClient) -> None:
    response = client.post("/api/trajectories", json={"steps": []})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "task_id" in detail
    assert "task_key" in detail


def test_both_task_references_return_400(client: TestClient) -> None:
    task = _create_task(client)
    response = client.post(
        "/api/trajectories",
        json={"task_id": task["id"], "task_key": task["key"], "steps": []},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "task_id" in detail
    assert "task_key" in detail


def test_unknown_task_returns_404(client: TestClient) -> None:
    by_id = client.post("/api/trajectories", json={"task_id": "missing", "steps": []})
    assert by_id.status_code == 404
    by_key = client.post("/api/trajectories", json={"task_key": "missing", "steps": []})
    assert by_key.status_code == 404


def test_invalid_step_type_returns_422(client: TestClient) -> None:
    task = _create_task(client)
    response = client.post(
        "/api/trajectories",
        json={"task_key": task["key"], "steps": [{"type": "thought", "content": {}}]},
    )
    assert response.status_code == 422


def test_invalid_status_returns_422(client: TestClient) -> None:
    task = _create_task(client)
    response = client.post(
        "/api/trajectories", json={"task_key": task["key"], "status": "running", "steps": []}
    )
    assert response.status_code == 422


def test_get_unknown_trajectory_returns_404(client: TestClient) -> None:
    assert client.get("/api/trajectories/missing").status_code == 404


def test_list_filters_and_summary_fields(
    client: TestClient,
    session: Session,
    make_task: Callable[..., Task],
    make_trajectory: Callable[..., Trajectory],
) -> None:
    task_a = make_task(key="refund-simple-01")
    task_b = make_task(key="order-status-01")
    t1 = make_trajectory(task=task_a)
    t2 = make_trajectory(task=task_a, steps=[("user_message", {"text": "Where is my order?"})])
    t3 = make_trajectory(task=task_b)
    session.add(HumanLabel(trajectory_id=t2.id, annotator="alice", verdict="pass"))
    session.commit()

    response = client.get("/api/trajectories")
    assert response.status_code == 200
    rows = response.json()
    assert {row["id"] for row in rows} == {t1.id, t2.id, t3.id}
    by_id = {row["id"]: row for row in rows}
    assert by_id[t1.id]["task_key"] == "refund-simple-01"
    assert by_id[t1.id]["step_count"] == 4
    assert by_id[t1.id]["label_count"] == 0
    assert by_id[t2.id]["step_count"] == 1
    assert by_id[t2.id]["label_count"] == 1
    assert by_id[t3.id]["task_key"] == "order-status-01"

    filtered = client.get("/api/trajectories", params={"task_id": task_a.id}).json()
    assert {row["id"] for row in filtered} == {t1.id, t2.id}

    labeled = client.get("/api/trajectories", params={"labeled": True}).json()
    assert {row["id"] for row in labeled} == {t2.id}

    unlabeled = client.get("/api/trajectories", params={"labeled": False}).json()
    assert {row["id"] for row in unlabeled} == {t1.id, t3.id}
