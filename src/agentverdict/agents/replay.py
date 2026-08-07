"""Task replay: run an agent under test against stored tasks and record trajectories.

The sweep is deliberately unkillable. Adapters call networks and parse model output,
so any single attempt may blow up; the harness records what went wrong and moves on to
the next task rather than losing the runs it already produced. Ordering is fixed
(tasks by ``created_at``, ``id``; repeats back to back) so a replay is reproducible.
"""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from agentverdict.agents.base import AgentAdapter, AgentRunError, AgentRunResult
from agentverdict.models import Step, Task, Trajectory
from agentverdict.schemas import ReplayReport

# Called once per attempt with (position, total, task, trajectory, error). Exactly one
# of ``trajectory`` / ``error`` is set: a trajectory when the attempt was persisted
# (whatever its status), an error string when the attempt produced no trajectory at all.
ProgressCallback = Callable[[int, int, Task, Trajectory | None, str | None], None]


def _detail(exc: Exception) -> str:
    """A single short line describing an exception, safe to embed in a report."""
    for line in str(exc).splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:200]
    return exc.__class__.__name__


def _persist(session: Session, task: Task, result: AgentRunResult) -> Trajectory:
    """Write one agent run as a replay trajectory with position-indexed steps."""
    trajectory = Trajectory(
        task=task,
        agent_config=result.agent_config,
        source="replay",
        status=result.status,
        meta=result.meta,
        started_at=result.started_at,
        completed_at=result.completed_at,
    )
    for index, step in enumerate(result.steps):
        trajectory.steps.append(Step(index=index, type=step.type, content=step.content))
    session.add(trajectory)
    session.flush()
    return trajectory


def run_replay(
    session: Session,
    adapter: AgentAdapter,
    *,
    task_key: str | None = None,
    limit: int | None = None,
    repeats: int = 1,
    on_progress: ProgressCallback | None = None,
) -> ReplayReport:
    """Replay stored tasks against ``adapter`` and persist the resulting trajectories.

    Tasks are selected in ``(created_at, id)`` order, narrowed to ``task_key`` when given
    and capped by ``limit`` — ``limit`` bounds *tasks*, not trajectories, so ``limit=5``
    with ``repeats=3`` produces up to fifteen trajectories. Each task is attempted
    ``repeats`` times back to back; ``report.trajectory_ids`` is in creation order.

    Failure accounting distinguishes two very different things:

    * A **runner error** — ``adapter.run`` raised ``AgentRunError``, blew up unexpectedly,
      or the trajectory could not be written — yields no trajectory. It is appended to
      ``report.errors`` as ``"task {key}: {detail}"`` and counted in ``error_count``,
      and the sweep continues with the next attempt. Each persist runs inside its own
      savepoint, so a database-level failure on one trajectory rolls back only that
      trajectory instead of poisoning the whole sweep.
    * A **result whose status is ``"error"``** is not a runner error. The agent broke
      partway but left real, partial steps that are still worth judging, so the
      trajectory is persisted normally and counted in ``trajectories_created``. It is
      *also* counted in ``error_count`` so the CLI can surface it, but it is *not*
      appended to ``report.errors`` — the recorded trajectory itself carries the detail.
      ``error_count`` may therefore exceed ``len(report.errors)``.

    Token counts accumulate for every run the adapter returns, including runs that
    later failed to persist: those tokens were really spent.
    """
    stmt = select(Task).order_by(Task.created_at, Task.id)
    if task_key is not None:
        stmt = stmt.where(Task.key == task_key)
    if limit is not None:
        stmt = stmt.limit(limit)
    tasks = list(session.scalars(stmt))

    report = ReplayReport(adapter=adapter.name)
    total = len(tasks) * max(repeats, 0)
    position = 0
    for task in tasks:
        for attempt in range(repeats):
            if attempt == 0:
                report.tasks_attempted += 1
            position += 1
            trajectory: Trajectory | None = None
            error: str | None = None
            try:
                result = adapter.run(task)
            except AgentRunError as exc:
                error = f"task {task.key}: {_detail(exc)}"
            except Exception as exc:  # a broken adapter never aborts the sweep
                error = f"task {task.key}: unexpected {exc.__class__.__name__}: {_detail(exc)}"
            else:
                report.input_tokens += result.input_tokens
                report.output_tokens += result.output_tokens
                try:
                    # Per-attempt savepoint, mirroring the importer's per-line handling:
                    # a constraint violation rolls back this trajectory alone.
                    with session.begin_nested():
                        trajectory = _persist(session, task, result)
                except SQLAlchemyError as exc:
                    trajectory = None
                    error = f"task {task.key}: database error: {_detail(exc)}"
                else:
                    report.trajectories_created += 1
                    report.trajectory_ids.append(trajectory.id)
                    if result.status == "error":
                        # A real partial trajectory, still worth judging: surfaced in the
                        # count but not in errors (see the failure accounting above).
                        report.error_count += 1
            if error is not None:
                report.errors.append(error)
                report.error_count += 1
            if on_progress is not None:
                on_progress(position, total, task, trajectory, error)
    session.flush()
    return report
