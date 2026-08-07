"""JSON routes for tasks (scenarios the agent-under-test should perform)."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from agentverdict.db import get_session
from agentverdict.models import Task
from agentverdict.schemas import TaskCreate, TaskRead

router = APIRouter(prefix="/api/tasks", tags=["tasks"])

SessionDep = Annotated[Session, Depends(get_session)]


@router.post("", response_model=TaskRead, status_code=status.HTTP_201_CREATED)
def create_task(payload: TaskCreate, session: SessionDep) -> TaskRead:
    """Create a task; 409 if a task with the same key already exists."""
    existing = session.scalar(select(Task).where(Task.key == payload.key))
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Task with key {payload.key!r} already exists",
        )
    task = Task(
        key=payload.key,
        prompt=payload.prompt,
        user_message=payload.user_message,
        tools_spec=payload.tools_spec,
        expected_outcome=payload.expected_outcome,
        tags=payload.tags,
    )
    session.add(task)
    session.flush()
    return TaskRead.model_validate(task)


@router.get("", response_model=list[TaskRead])
def list_tasks(session: SessionDep) -> list[TaskRead]:
    """List all tasks, oldest first."""
    tasks = session.scalars(select(Task).order_by(Task.created_at, Task.id)).all()
    return [TaskRead.model_validate(task) for task in tasks]


@router.get("/{task_id}", response_model=TaskRead)
def get_task(task_id: str, session: SessionDep) -> TaskRead:
    """Fetch a single task by id; 404 if missing."""
    task = session.get(Task, task_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id!r} not found",
        )
    return TaskRead.model_validate(task)
