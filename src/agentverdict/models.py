"""SQLAlchemy models for the AgentVerdict core domain.

Milestone 1 scope: the golden-dataset workbench — tasks, recorded agent trajectories,
and human labels. Judge verdicts, eval runs, and calibration tables arrive in later
milestones (see DESIGN.md).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def new_id() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Rubric(Base):
    """A versioned set of grading criteria. Dormant in M1 (no routes yet)."""

    __tablename__ = "rubrics"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_rubric_name_version"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(200))
    version: Mapped[int] = mapped_column(Integer, default=1)
    criteria: Mapped[list] = mapped_column(JSON, default=list)  # [{key, description, weight}]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Task(Base):
    """A scenario the agent-under-test should perform."""

    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    key: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    prompt: Mapped[str] = mapped_column(Text)
    tools_spec: Mapped[list] = mapped_column(JSON, default=list)
    expected_outcome: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    trajectories: Mapped[list[Trajectory]] = relationship(back_populates="task")


class Trajectory(Base):
    """One recorded run of an agent on a task."""

    __tablename__ = "trajectories"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"), index=True)
    agent_config: Mapped[dict] = mapped_column(JSON, default=dict)
    source: Mapped[str] = mapped_column(String(32), default="api")
    status: Mapped[str] = mapped_column(String(32), default="completed")
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    task: Mapped[Task] = relationship(back_populates="trajectories")
    steps: Mapped[list[Step]] = relationship(
        back_populates="trajectory",
        cascade="all, delete-orphan",
        order_by="Step.index",
    )
    labels: Mapped[list[HumanLabel]] = relationship(
        back_populates="trajectory", cascade="all, delete-orphan"
    )


class Step(Base):
    """An ordered entry within a trajectory (message, tool call, or tool result)."""

    __tablename__ = "trajectory_steps"
    __table_args__ = (
        UniqueConstraint("trajectory_id", "index", name="uq_step_trajectory_index"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    trajectory_id: Mapped[str] = mapped_column(ForeignKey("trajectories.id"), index=True)
    index: Mapped[int] = mapped_column(Integer)
    type: Mapped[str] = mapped_column(String(32))
    content: Mapped[dict] = mapped_column(JSON, default=dict)

    trajectory: Mapped[Trajectory] = relationship(back_populates="steps")


class HumanLabel(Base):
    """A human judgment of a trajectory. One label per (trajectory, annotator)."""

    __tablename__ = "human_labels"
    __table_args__ = (
        UniqueConstraint("trajectory_id", "annotator", name="uq_label_trajectory_annotator"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    trajectory_id: Mapped[str] = mapped_column(ForeignKey("trajectories.id"), index=True)
    annotator: Mapped[str] = mapped_column(String(100))
    verdict: Mapped[str] = mapped_column(String(16))  # pass | fail | borderline
    rubric_id: Mapped[str | None] = mapped_column(ForeignKey("rubrics.id"), nullable=True)
    rubric_scores: Mapped[dict] = mapped_column(JSON, default=dict)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    time_spent_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    trajectory: Mapped[Trajectory] = relationship(back_populates="labels")
