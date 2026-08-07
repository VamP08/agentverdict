"""Task replay: persisting agent runs as trajectories, then judging exactly those runs.

The adapter here is a deterministic stub — the Groq reference agent has its own
tests — so these exercise the harness itself: ordering, repeats, filters, failure
isolation, and the ``replay -> run_eval(trajectory_ids=...)`` handoff the CLI makes.
"""

import json
from collections.abc import Callable

import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from agentverdict.agents.base import AgentAdapter, AgentRunError, AgentRunResult
from agentverdict.agents.replay import run_replay
from agentverdict.judging.client import GroqChatClient
from agentverdict.judging.runner import run_eval
from agentverdict.models import Judge, JudgeVerdict, Task, Trajectory
from agentverdict.schemas import ReplayReport, StepIn

STEP_TYPES = ["system", "user_message", "tool_call", "tool_result", "assistant_message"]


class StubAdapter:
    """A deterministic agent under test: fixed steps per task, scripted failures.

    ``raise_for`` names tasks whose ``run`` blows up (no trajectory at all);
    ``error_for`` names tasks that come back with ``status="error"`` — a real,
    partial run that the harness must still persist.
    """

    def __init__(
        self,
        name: str = "stub-agent",
        raise_for: tuple[str, ...] = (),
        error_for: tuple[str, ...] = (),
        input_tokens: int = 70,
        output_tokens: int = 13,
    ) -> None:
        self.name = name
        self.raise_for = set(raise_for)
        self.error_for = set(error_for)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.calls: list[str] = []

    def run(self, task: Task) -> AgentRunResult:
        self.calls.append(task.key)
        attempt = self.calls.count(task.key)
        if task.key in self.raise_for:
            raise AgentRunError(f"adapter refused {task.key}")
        steps = [
            StepIn(type="system", content={"text": "You are a support agent."}),
            StepIn(type="user_message", content={"text": f"opening line for {task.key}"}),
            StepIn(
                type="tool_call",
                content={"name": "lookup_order", "arguments": {"order_id": task.key}},
            ),
            StepIn(
                type="tool_result",
                content={
                    "name": "lookup_order",
                    "result": {"status": "delivered"},
                    "is_error": False,
                },
            ),
            StepIn(type="assistant_message", content={"text": f"{task.key} attempt {attempt}"}),
        ]
        return AgentRunResult(
            steps=steps,
            status="error" if task.key in self.error_for else "completed",
            agent_config={"adapter": self.name, "model": "stub-model"},
            meta={"attempt": attempt},
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )


def _task_order(session: Session) -> list[str]:
    """The task keys in the order the replay runner walks them."""
    return list(session.scalars(select(Task.key).order_by(Task.created_at, Task.id)))


def _stored(session: Session, report: ReplayReport) -> list[Trajectory]:
    rows = [session.get(Trajectory, trajectory_id) for trajectory_id in report.trajectory_ids]
    assert all(row is not None for row in rows), "report ids must all resolve to trajectories"
    return [row for row in rows if row is not None]


# --- persistence -------------------------------------------------------------


def test_replay_records_trajectories_with_source_replay(
    session: Session, make_task: Callable[..., Task]
) -> None:
    make_task(key="replay-a")
    make_task(key="replay-b")
    adapter = StubAdapter()
    assert isinstance(adapter, AgentAdapter)

    report = run_replay(session, adapter)

    expected_order = _task_order(session)
    assert sorted(expected_order) == ["replay-a", "replay-b"]
    assert adapter.calls == expected_order
    assert report.adapter == "stub-agent"
    assert report.tasks_attempted == 2
    assert report.trajectories_created == 2
    assert report.error_count == 0
    assert report.errors == []
    assert report.input_tokens == 140
    assert report.output_tokens == 26

    stored = _stored(session, report)
    assert [trajectory.task.key for trajectory in stored] == expected_order
    assert {trajectory.id for trajectory in stored} == set(
        session.scalars(select(Trajectory.id))
    )
    for trajectory in stored:
        assert trajectory.source == "replay"
        assert trajectory.status == "completed"
        assert trajectory.agent_config == {"adapter": "stub-agent", "model": "stub-model"}
        assert [step.type for step in trajectory.steps] == STEP_TYPES
        assert [step.index for step in trajectory.steps] == [0, 1, 2, 3, 4]
        assert trajectory.steps[2].content["arguments"] == {"order_id": trajectory.task.key}
        assert trajectory.started_at is not None
        assert trajectory.completed_at is not None


def test_repeats_create_several_trajectories_per_task(
    session: Session, make_task: Callable[..., Task]
) -> None:
    make_task(key="replay-a")
    make_task(key="replay-b")
    adapter = StubAdapter()

    report = run_replay(session, adapter, repeats=3)

    first, second = _task_order(session)
    # Repeats run back to back, so the call order is grouped by task.
    assert adapter.calls == [first, first, first, second, second, second]
    assert report.tasks_attempted == 2
    assert report.trajectories_created == 6
    assert len(report.trajectory_ids) == 6
    assert len(set(report.trajectory_ids)) == 6

    per_task = dict(
        session.execute(
            select(Task.key, func.count(Trajectory.id))
            .join(Trajectory, Trajectory.task_id == Task.id)
            .group_by(Task.key)
        ).all()
    )
    assert per_task == {first: 3, second: 3}
    assert [t.meta["attempt"] for t in _stored(session, report)[:3]] == [1, 2, 3]


def test_task_key_filter_replays_one_task(
    session: Session, make_task: Callable[..., Task]
) -> None:
    make_task(key="replay-a")
    make_task(key="replay-b")
    make_task(key="replay-c")
    adapter = StubAdapter()

    report = run_replay(session, adapter, task_key="replay-b")

    assert adapter.calls == ["replay-b"]
    assert report.tasks_attempted == 1
    assert report.trajectories_created == 1
    assert _stored(session, report)[0].task.key == "replay-b"


def test_limit_bounds_tasks_not_trajectories(
    session: Session, make_task: Callable[..., Task]
) -> None:
    make_task(key="replay-a")
    make_task(key="replay-b")
    make_task(key="replay-c")
    adapter = StubAdapter()

    report = run_replay(session, adapter, limit=2, repeats=2)

    assert report.tasks_attempted == 2
    assert report.trajectories_created == 4
    assert set(adapter.calls) == set(_task_order(session)[:2])
    assert session.scalar(select(func.count()).select_from(Trajectory)) == 4


# --- failure isolation -------------------------------------------------------


def test_adapter_failure_is_recorded_and_the_sweep_continues(
    session: Session, make_task: Callable[..., Task]
) -> None:
    make_task(key="replay-a")
    make_task(key="replay-b")
    make_task(key="replay-c")
    adapter = StubAdapter(raise_for=("replay-b",))
    events: list[tuple[int, int, str, str | None, str | None]] = []

    def record(
        position: int,
        total: int,
        task: Task,
        trajectory: Trajectory | None,
        error: str | None,
    ) -> None:
        events.append(
            (position, total, task.key, None if trajectory is None else trajectory.status, error)
        )

    report = run_replay(session, adapter, on_progress=record)

    assert adapter.calls == _task_order(session)  # the failure did not abort the run
    assert report.tasks_attempted == 3
    assert report.trajectories_created == 2
    assert report.error_count == 1
    assert len(report.errors) == 1
    assert report.errors[0].startswith("task replay-b:")
    assert "adapter refused replay-b" in report.errors[0]
    assert session.scalar(select(func.count()).select_from(Trajectory)) == 2
    assert "replay-b" not in {trajectory.task.key for trajectory in _stored(session, report)}

    assert [event[0] for event in events] == [1, 2, 3]
    assert {event[1] for event in events} == {3}
    failed = [event for event in events if event[2] == "replay-b"]
    succeeded = [event for event in events if event[2] != "replay-b"]
    assert len(failed) == 1
    # Exactly one of trajectory/error is set on every progress event.
    assert failed[0][3] is None and failed[0][4] is not None
    assert len(succeeded) == 2
    assert all(event[3] == "completed" and event[4] is None for event in succeeded)


def test_error_status_runs_are_persisted_and_counted(
    session: Session, make_task: Callable[..., Task]
) -> None:
    make_task(key="replay-a")
    make_task(key="replay-b")
    adapter = StubAdapter(error_for=("replay-a",))

    report = run_replay(session, adapter)

    # A broken run still leaves partial steps worth judging, so it is a trajectory...
    assert report.trajectories_created == 2
    assert len(report.trajectory_ids) == 2
    # ...counted as an error, but not as a runner error with a message.
    assert report.error_count == 1
    assert report.errors == []

    by_key = {trajectory.task.key: trajectory for trajectory in _stored(session, report)}
    assert by_key["replay-a"].status == "error"
    assert by_key["replay-b"].status == "completed"
    assert len(by_key["replay-a"].steps) == len(STEP_TYPES)


def test_replay_without_tasks_does_nothing(session: Session) -> None:
    adapter = StubAdapter()
    report = run_replay(session, adapter)
    assert adapter.calls == []
    assert report.tasks_attempted == 0
    assert report.trajectories_created == 0
    assert report.trajectory_ids == []
    assert report.errors == []


# --- replay -> judge handoff -------------------------------------------------


def _chat_body(content: str, prompt_tokens: int = 100, completion_tokens: int = 20) -> dict:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def _decision(verdict: str = "pass", rationale: str = "Looked the order up first.") -> str:
    return json.dumps({"verdict": verdict, "rationale": rationale, "rubric_scores": {}})


def _make_client(handler: Callable[[httpx.Request], httpx.Response]) -> GroqChatClient:
    return GroqChatClient(
        api_key="test-key",
        base_url="https://groq.test/openai/v1",
        transport=httpx.MockTransport(handler),
    )


@pytest.fixture
def judge(session: Session) -> Judge:
    row = Judge(name="replay-judge", model="test-model")
    session.add(row)
    session.commit()
    return row


def test_eval_judges_only_the_trajectories_just_replayed(
    session: Session,
    judge: Judge,
    make_task: Callable[..., Task],
    make_trajectory: Callable[..., Trajectory],
) -> None:
    task_a = make_task(key="replay-a")
    make_task(key="replay-b")
    # An older run of the same task: it must survive untouched and unjudged.
    stale = make_trajectory(task=task_a, source="import")

    report = run_replay(session, StubAdapter())
    assert report.trajectories_created == 2

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_body(_decision()))

    with _make_client(handler) as client:
        run = run_eval(session, judge, trajectory_ids=report.trajectory_ids, client=client)

    assert run.status == "completed"
    assert run.trajectory_count == 2
    assert run.error_count == 0
    assert run.verdict_counts == {"pass": 2}
    assert run.input_tokens == 200

    judged = set(session.scalars(select(JudgeVerdict.trajectory_id)))
    assert judged == set(report.trajectory_ids)
    assert stale.id not in judged
    assert session.scalar(select(func.count()).select_from(Trajectory)) == 3


def test_eval_with_no_new_trajectories_judges_nothing(
    session: Session,
    judge: Judge,
    make_trajectory: Callable[..., Trajectory],
) -> None:
    make_trajectory()

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("the judge must not be called when there is nothing to judge")

    with _make_client(handler) as client:
        run = run_eval(session, judge, trajectory_ids=[], client=client)

    assert run.status == "completed"
    assert run.trajectory_count == 0
    assert run.verdict_counts == {}
    assert session.scalar(select(func.count()).select_from(JudgeVerdict)) == 0
