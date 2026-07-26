"""API tests for /api/tasks."""

from fastapi.testclient import TestClient

TASK_PAYLOAD = {
    "key": "refund-simple-01",
    "prompt": "A customer reports a damaged item and asks for a refund.",
    "tools_spec": [{"name": "lookup_order", "description": "Fetch an order by id."}],
    "expected_outcome": "Refund issued and confirmed by email.",
    "tags": ["refunds", "happy-path"],
}


def test_create_task_returns_created_row(client: TestClient) -> None:
    response = client.post("/api/tasks", json=TASK_PAYLOAD)
    assert response.status_code == 201
    body = response.json()
    assert body["key"] == TASK_PAYLOAD["key"]
    assert body["prompt"] == TASK_PAYLOAD["prompt"]
    assert body["tools_spec"] == TASK_PAYLOAD["tools_spec"]
    assert body["expected_outcome"] == TASK_PAYLOAD["expected_outcome"]
    assert body["tags"] == TASK_PAYLOAD["tags"]
    assert body["id"]
    assert body["created_at"]


def test_create_task_applies_defaults(client: TestClient) -> None:
    response = client.post(
        "/api/tasks", json={"key": "minimal-01", "prompt": "Check an order status."}
    )
    assert response.status_code == 201
    body = response.json()
    assert body["tools_spec"] == []
    assert body["tags"] == []
    assert body["expected_outcome"] is None


def test_duplicate_key_returns_409(client: TestClient) -> None:
    assert client.post("/api/tasks", json=TASK_PAYLOAD).status_code == 201
    response = client.post("/api/tasks", json=TASK_PAYLOAD)
    assert response.status_code == 409
    assert TASK_PAYLOAD["key"] in response.json()["detail"]


def test_duplicate_key_does_not_add_a_row(client: TestClient) -> None:
    client.post("/api/tasks", json=TASK_PAYLOAD)
    client.post("/api/tasks", json=TASK_PAYLOAD)
    body = client.get("/api/tasks").json()
    assert len(body) == 1


def test_list_tasks(client: TestClient) -> None:
    client.post("/api/tasks", json=TASK_PAYLOAD)
    client.post("/api/tasks", json=dict(TASK_PAYLOAD, key="order-status-01"))
    response = client.get("/api/tasks")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert {task["key"] for task in body} == {"refund-simple-01", "order-status-01"}


def test_get_task_by_id(client: TestClient) -> None:
    created = client.post("/api/tasks", json=TASK_PAYLOAD).json()
    response = client.get(f"/api/tasks/{created['id']}")
    assert response.status_code == 200
    assert response.json() == created


def test_get_unknown_task_returns_404(client: TestClient) -> None:
    assert client.get("/api/tasks/doesnotexist").status_code == 404


def test_invalid_body_returns_422(client: TestClient) -> None:
    assert client.post("/api/tasks", json={"prompt": "missing key"}).status_code == 422
    assert client.post("/api/tasks", json={"key": "no-prompt-01"}).status_code == 422
    assert client.post("/api/tasks", json={"key": "", "prompt": "empty key"}).status_code == 422
