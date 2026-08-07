"""The agent-under-test contract.

An adapter turns a stored task into a recorded run. Everything the replay harness
needs to persist a trajectory comes back in ``AgentRunResult`` — adapters never
touch the database, which keeps them trivial to test and easy to swap (the Groq
reference agent today, a customer's own agent tomorrow).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from agentverdict.models import Task, utcnow
from agentverdict.schemas import StepIn, TrajectoryStatus


class AgentRunError(RuntimeError):
    """The adapter could not produce a run at all (as opposed to a failed run)."""


@dataclass
class AgentRunResult:
    """One recorded attempt at a task.

    ``status`` mirrors the trajectory vocabulary: ``completed`` when the agent
    finished on its own, ``truncated`` when it hit the step budget, ``error``
    when the run broke partway (the partial steps are still worth judging).
    """

    steps: list[StepIn]
    status: TrajectoryStatus = "completed"
    agent_config: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=utcnow)
    completed_at: datetime = field(default_factory=utcnow)
    input_tokens: int = 0
    output_tokens: int = 0


@runtime_checkable
class AgentAdapter(Protocol):
    """Anything that can attempt a task and hand back an ordered step list."""

    name: str

    def run(self, task: Task) -> AgentRunResult:
        """Attempt ``task`` once. Raise AgentRunError only if no run was produced."""
        ...


def opening_message(task: Task) -> str:
    """The first user turn an agent should see.

    ``task.prompt`` is written for graders and states what the agent *should* do,
    so replaying against it would leak the answer. Tasks therefore carry a
    first-person ``user_message``; the prompt is only a last-resort fallback.
    """
    if task.user_message and task.user_message.strip():
        return task.user_message.strip()
    return task.prompt
