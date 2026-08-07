"""``Task.user_message`` must survive every write path.

The prompt describes the scenario for graders and states what the agent *should*
do; the agent under test sees ``user_message`` instead. Any write path that drops
the field silently reintroduces the answer leak the column exists to prevent, so
each one is pinned here.
"""

import json
from collections.abc import Callable
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from agentverdict.agents.base import opening_message
from agentverdict.importer import export_jsonl, import_jsonl
from agentverdict.models import Task

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "examples" / "sample_trajectories.jsonl"

OPENING = "Hi, order NW-1001 arrived cracked. Can I get a refund?"
GRADER_PROMPT = "The agent should look up the order, verify ownership, then refund it."


def _bundle_line(**task_overrides: object) -> str:
    task = {"key": "leak-check-01", "prompt": GRADER_PROMPT, "user_message": OPENING}
    task.update(task_overrides)
    return json.dumps(
        {
            "task": task,
            "trajectory": {
                "steps": [{"type": "user_message", "content": {"text": OPENING}}],
            },
        }
    )


def test_import_persists_user_message(session: Session, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.jsonl"
    bundle.write_text(_bundle_line() + "\n", encoding="utf-8")

    report = import_jsonl(session, bundle)
    session.commit()

    assert report.errors == []
    task = session.scalar(select(Task).where(Task.key == "leak-check-01"))
    assert task is not None
    assert task.user_message == OPENING
    # The whole point: the agent must not be handed the grader-facing prompt.
    assert opening_message(task) == OPENING


def test_sample_bundle_imports_with_user_messages(session: Session) -> None:
    import_jsonl(session, SAMPLE_PATH)
    session.commit()

    tasks = list(session.scalars(select(Task)))
    assert tasks, "sample bundle should create tasks"
    for task in tasks:
        assert task.user_message, f"task {task.key} imported without a user_message"
        assert opening_message(task) != task.prompt


def test_export_round_trip_preserves_user_message(session: Session, tmp_path: Path) -> None:
    bundle = tmp_path / "in.jsonl"
    bundle.write_text(_bundle_line() + "\n", encoding="utf-8")
    import_jsonl(session, bundle)
    session.commit()

    out = tmp_path / "out.jsonl"
    assert export_jsonl(session, out) == 1
    exported = json.loads(out.read_text(encoding="utf-8").strip())
    assert exported["task"]["user_message"] == OPENING


def test_api_persists_user_message(client: TestClient) -> None:
    response = client.post(
        "/api/tasks",
        json={"key": "api-leak-check", "prompt": GRADER_PROMPT, "user_message": OPENING},
    )
    assert response.status_code == 201
    assert response.json()["user_message"] == OPENING

    fetched = client.get(f"/api/tasks/{response.json()['id']}")
    assert fetched.json()["user_message"] == OPENING


def test_opening_message_falls_back_to_prompt(make_task: Callable[..., Task]) -> None:
    task = make_task(key="no-opening-line", prompt=GRADER_PROMPT)
    assert task.user_message is None
    assert opening_message(task) == GRADER_PROMPT
