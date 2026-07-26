"""JSONL trajectory bundle import/export and dataset statistics.

Bundle format (see DESIGN.md): one JSON object per line, holding an embedded task
(upserted by ``key``; existing tasks are never overwritten) and one trajectory with
its ordered steps. Import is line-tolerant: a malformed or invalid line is recorded
in ``ImportReport.errors`` and skipped while the remaining lines still import.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, selectinload

from agentverdict.models import HumanLabel, Step, Task, Trajectory
from agentverdict.schemas import (
    ImportReport,
    StatsReport,
    StepIn,
    TaskCreate,
    TrajectorySource,
    TrajectoryStatus,
)


class _TrajectoryIn(BaseModel):
    """The ``trajectory`` object of a bundle line (the task travels separately)."""

    agent_config: dict[str, Any] = Field(default_factory=dict)
    source: TrajectorySource = "import"
    status: TrajectoryStatus = "completed"
    meta: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    steps: list[StepIn] = Field(default_factory=list)


class _BundleLine(BaseModel):
    """One JSONL line: an embedded task plus a single trajectory."""

    task: TaskCreate
    trajectory: _TrajectoryIn


def _describe(exc: ValidationError) -> str:
    """Flatten a pydantic ValidationError into a single human-readable string."""
    parts = []
    for error in exc.errors():
        loc = ".".join(str(piece) for piece in error["loc"]) or "<root>"
        parts.append(f"{loc}: {error['msg']}")
    return "; ".join(parts)


def _get_or_create_task(
    session: Session,
    payload: TaskCreate,
    cache: dict[str, Task],
    report: ImportReport,
) -> Task:
    """Upsert a task by key: create if missing, never overwrite existing fields."""
    task = cache.get(payload.key)
    if task is None:
        task = session.scalar(select(Task).where(Task.key == payload.key))
    if task is None:
        task = Task(
            key=payload.key,
            prompt=payload.prompt,
            tools_spec=payload.tools_spec,
            expected_outcome=payload.expected_outcome,
            tags=payload.tags,
        )
        session.add(task)
        report.tasks_created += 1
    cache[payload.key] = task
    return task


def import_jsonl(session: Session, path: str | Path) -> ImportReport:
    """Import a JSONL trajectory bundle, skipping (and reporting) invalid lines."""
    report = ImportReport()
    task_cache: dict[str, Task] = {}
    with Path(path).open(encoding="utf-8-sig") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                report.errors.append(f"line {lineno}: invalid JSON: {exc}")
                continue
            try:
                bundle = _BundleLine.model_validate(payload)
            except ValidationError as exc:
                report.errors.append(f"line {lineno}: {_describe(exc)}")
                continue
            spec = bundle.trajectory
            line_report = ImportReport()
            try:
                # Per-line savepoint: a database-level failure (e.g. a constraint
                # violation schema validation cannot see) rolls back only this line,
                # preserving the line-tolerance contract.
                with session.begin_nested():
                    task = _get_or_create_task(session, bundle.task, task_cache, line_report)
                    trajectory = Trajectory(
                        task=task,
                        agent_config=spec.agent_config,
                        source=spec.source,
                        status=spec.status,
                        meta=spec.meta,
                        started_at=spec.started_at,
                        completed_at=spec.completed_at,
                    )
                    for index, step in enumerate(spec.steps):
                        trajectory.steps.append(
                            Step(index=index, type=step.type, content=step.content)
                        )
                    session.add(trajectory)
                    session.flush()
            except SQLAlchemyError as exc:
                # The savepoint rolled back; drop the cached task so a later line
                # re-resolves it from the database.
                task_cache.pop(bundle.task.key, None)
                detail = str(exc).splitlines()[0][:200]
                report.errors.append(f"line {lineno}: database error: {detail}")
                continue
            report.tasks_created += line_report.tasks_created
            report.trajectories_created += 1
            report.steps_created += len(spec.steps)
    session.flush()
    return report


def _as_bundle(trajectory: Trajectory) -> dict[str, Any]:
    """Serialize one trajectory (with its task and steps) as a bundle-line dict."""
    task = trajectory.task
    spec: dict[str, Any] = {
        "agent_config": trajectory.agent_config,
        "source": trajectory.source,
        "status": trajectory.status,
        "meta": trajectory.meta,
    }
    if trajectory.started_at is not None:
        spec["started_at"] = trajectory.started_at.isoformat()
    if trajectory.completed_at is not None:
        spec["completed_at"] = trajectory.completed_at.isoformat()
    spec["steps"] = [{"type": step.type, "content": step.content} for step in trajectory.steps]
    return {
        "task": {
            "key": task.key,
            "prompt": task.prompt,
            "tools_spec": task.tools_spec,
            "expected_outcome": task.expected_outcome,
            "tags": task.tags,
        },
        "trajectory": spec,
    }


def export_jsonl(session: Session, path: str | Path, task_key: str | None = None) -> int:
    """Write trajectories (optionally only those of ``task_key``) as a JSONL bundle.

    Returns the number of trajectories written.
    """
    stmt = (
        select(Trajectory)
        .join(Task, Trajectory.task_id == Task.id)
        .options(selectinload(Trajectory.steps), selectinload(Trajectory.task))
        .order_by(Trajectory.created_at, Trajectory.id)
    )
    if task_key is not None:
        stmt = stmt.where(Task.key == task_key)
    count = 0
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for trajectory in session.scalars(stmt):
            handle.write(json.dumps(_as_bundle(trajectory), ensure_ascii=False) + "\n")
            count += 1
    return count


def compute_stats(session: Session) -> StatsReport:
    """Count tasks, trajectories, and labels; include a verdict histogram."""
    task_count = session.scalar(select(func.count()).select_from(Task)) or 0
    trajectory_count = session.scalar(select(func.count()).select_from(Trajectory)) or 0
    label_count = session.scalar(select(func.count()).select_from(HumanLabel)) or 0
    labeled_trajectory_count = (
        session.scalar(select(func.count(func.distinct(HumanLabel.trajectory_id)))) or 0
    )
    verdict_rows = session.execute(
        select(HumanLabel.verdict, func.count()).group_by(HumanLabel.verdict)
    ).all()
    return StatsReport(
        task_count=task_count,
        trajectory_count=trajectory_count,
        labeled_trajectory_count=labeled_trajectory_count,
        label_count=label_count,
        labels_by_verdict={verdict: count for verdict, count in verdict_rows},
    )
