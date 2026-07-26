"""JSON routes for trajectories (recorded multi-step agent runs)."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from agentverdict.db import get_session
from agentverdict.models import HumanLabel, Step, Task, Trajectory
from agentverdict.schemas import TrajectoryCreate, TrajectoryRead, TrajectorySummary

router = APIRouter(prefix="/api/trajectories", tags=["trajectories"])

SessionDep = Annotated[Session, Depends(get_session)]


def _resolve_task(payload: TrajectoryCreate, session: Session) -> Task:
    """Resolve the task referenced by exactly one of task_id / task_key."""
    if payload.task_id is not None and payload.task_key is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide either task_id or task_key, not both",
        )
    if payload.task_id is None and payload.task_key is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide exactly one of task_id or task_key",
        )
    if payload.task_id is not None:
        task = session.get(Task, payload.task_id)
        if task is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {payload.task_id!r} not found",
            )
        return task
    task = session.scalar(select(Task).where(Task.key == payload.task_key))
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with key {payload.task_key!r} not found",
        )
    return task


@router.post("", response_model=TrajectoryRead, status_code=status.HTTP_201_CREATED)
def create_trajectory(payload: TrajectoryCreate, session: SessionDep) -> TrajectoryRead:
    """Create a trajectory with nested steps; steps are indexed by list position."""
    task = _resolve_task(payload, session)
    trajectory = Trajectory(
        task_id=task.id,
        agent_config=payload.agent_config,
        source=payload.source,
        status=payload.status,
        meta=payload.meta,
        started_at=payload.started_at,
        completed_at=payload.completed_at,
        steps=[
            Step(index=index, type=step.type, content=step.content)
            for index, step in enumerate(payload.steps)
        ],
    )
    session.add(trajectory)
    session.flush()
    return TrajectoryRead.model_validate(trajectory)


@router.get("", response_model=list[TrajectorySummary])
def list_trajectories(
    session: SessionDep,
    task_id: str | None = None,
    labeled: bool | None = None,
) -> list[TrajectorySummary]:
    """List trajectory summaries with step/label counts, optionally filtered."""
    step_counts = (
        select(Step.trajectory_id, func.count(Step.id).label("n"))
        .group_by(Step.trajectory_id)
        .subquery()
    )
    label_counts = (
        select(HumanLabel.trajectory_id, func.count(HumanLabel.id).label("n"))
        .group_by(HumanLabel.trajectory_id)
        .subquery()
    )
    stmt = (
        select(
            Trajectory,
            Task.key,
            func.coalesce(step_counts.c.n, 0),
            func.coalesce(label_counts.c.n, 0),
        )
        .join(Task, Trajectory.task_id == Task.id)
        .outerjoin(step_counts, step_counts.c.trajectory_id == Trajectory.id)
        .outerjoin(label_counts, label_counts.c.trajectory_id == Trajectory.id)
        .order_by(Trajectory.created_at, Trajectory.id)
    )
    if task_id is not None:
        stmt = stmt.where(Trajectory.task_id == task_id)
    if labeled is True:
        stmt = stmt.where(label_counts.c.n > 0)
    elif labeled is False:
        stmt = stmt.where(label_counts.c.n.is_(None))
    rows = session.execute(stmt).all()
    return [
        TrajectorySummary(
            id=trajectory.id,
            task_id=trajectory.task_id,
            task_key=task_key,
            source=trajectory.source,
            status=trajectory.status,
            step_count=step_count,
            label_count=label_count,
            created_at=trajectory.created_at,
        )
        for trajectory, task_key, step_count, label_count in rows
    ]


@router.get("/{trajectory_id}", response_model=TrajectoryRead)
def get_trajectory(trajectory_id: str, session: SessionDep) -> TrajectoryRead:
    """Fetch a full trajectory (with ordered steps) by id; 404 if missing."""
    trajectory = session.get(
        Trajectory, trajectory_id, options=[selectinload(Trajectory.steps)]
    )
    if trajectory is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trajectory {trajectory_id!r} not found",
        )
    return TrajectoryRead.model_validate(trajectory)
