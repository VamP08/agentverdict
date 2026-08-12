"""SQLAlchemy models for the AgentVerdict core domain.

Milestone 1 scope: the golden-dataset workbench — tasks, recorded agent trajectories,
and human labels. Judge verdicts, eval runs, and calibration tables arrive in later
milestones (see DESIGN.md).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def new_id() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(UTC)


class TZDateTime(TypeDecorator):
    """Store datetimes as naive UTC; return them tz-aware (UTC) on load.

    SQLite discards tzinfo on write and Postgres re-interprets it against the
    connection timezone, so normalizing at the type boundary keeps timestamps
    identical across backends and across API create/read round trips.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value  # naive input is treated as UTC
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC)


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
    created_at: Mapped[datetime] = mapped_column(TZDateTime(), default=utcnow)


class Task(Base):
    """A scenario the agent-under-test should perform."""

    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    key: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    prompt: Mapped[str] = mapped_column(Text)
    # The opening customer message a replayed agent receives. The prompt above
    # describes the scenario for graders and would leak expected behavior to the
    # agent under test, so replay uses this instead (falling back to prompt).
    user_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    tools_spec: Mapped[list] = mapped_column(JSON, default=list)
    expected_outcome: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(TZDateTime(), default=utcnow)

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
    started_at: Mapped[datetime | None] = mapped_column(TZDateTime(), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(TZDateTime(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(TZDateTime(), default=utcnow)

    task: Mapped[Task] = relationship(back_populates="trajectories")
    steps: Mapped[list[Step]] = relationship(
        back_populates="trajectory",
        cascade="all, delete-orphan",
        order_by="Step.index",
    )
    labels: Mapped[list[HumanLabel]] = relationship(
        back_populates="trajectory", cascade="all, delete-orphan"
    )
    judge_verdicts: Mapped[list[JudgeVerdict]] = relationship(
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
    created_at: Mapped[datetime] = mapped_column(TZDateTime(), default=utcnow)

    trajectory: Mapped[Trajectory] = relationship(back_populates="labels")


class Judge(Base):
    """A configured LLM judge: a model plus a prompting recipe."""

    __tablename__ = "judges"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    model: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    config: Mapped[dict] = mapped_column(JSON, default=dict)  # e.g. {"temperature": 0.0}
    created_at: Mapped[datetime] = mapped_column(TZDateTime(), default=utcnow)

    eval_runs: Mapped[list[EvalRun]] = relationship(back_populates="judge")
    verdicts: Mapped[list[JudgeVerdict]] = relationship(back_populates="judge")
    bias_probe_runs: Mapped[list[BiasProbeRun]] = relationship(back_populates="judge")


class EvalRun(Base):
    """One batch of judging: a judge applied to a set of trajectories."""

    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    judge_id: Mapped[str] = mapped_column(ForeignKey("judges.id"), index=True)
    task_key: Mapped[str | None] = mapped_column(String(200), nullable=True)
    # Fingerprint of the judge prompt this run was graded with (judging.prompts
    # .PROMPT_VERSION). Nullable because runs recorded before the column existed cannot
    # be attributed to a version after the fact, and inventing one for them would be a
    # lie the regression gate then trusts.
    judge_prompt_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="running")  # running|completed|failed
    trajectory_count: Mapped[int] = mapped_column(Integer, default=0)
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    verdict_counts: Mapped[dict] = mapped_column(JSON, default=dict)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)  # e.g. naive human-agreement figures
    started_at: Mapped[datetime] = mapped_column(TZDateTime(), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(TZDateTime(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(TZDateTime(), default=utcnow)

    judge: Mapped[Judge] = relationship(back_populates="eval_runs")
    verdicts: Mapped[list[JudgeVerdict]] = relationship(
        back_populates="eval_run", cascade="all, delete-orphan"
    )


class JudgeVerdict(Base):
    """One judge decision (or recorded failure) for one trajectory in one run."""

    __tablename__ = "judge_verdicts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    eval_run_id: Mapped[str] = mapped_column(ForeignKey("eval_runs.id"), index=True)
    judge_id: Mapped[str] = mapped_column(ForeignKey("judges.id"), index=True)
    trajectory_id: Mapped[str] = mapped_column(ForeignKey("trajectories.id"), index=True)
    verdict: Mapped[str | None] = mapped_column(String(16), nullable=True)  # null when errored
    rubric_scores: Mapped[dict] = mapped_column(JSON, default=dict)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_response: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(TZDateTime(), default=utcnow)

    eval_run: Mapped[EvalRun] = relationship(back_populates="verdicts")
    judge: Mapped[Judge] = relationship(back_populates="verdicts")
    trajectory: Mapped[Trajectory] = relationship(back_populates="judge_verdicts")


class BiasProbeRun(Base):
    """One bias probe: a judge re-asked about the same trajectories under perturbed prompts.

    Deliberately a table of its own rather than an ``EvalRun`` carrying a flag. Eval
    runs feed calibration and the regression gate, and a verdict on a transcript that
    was reordered or padded on purpose is not a verdict on the run.
    """

    __tablename__ = "bias_probe_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    judge_id: Mapped[str] = mapped_column(ForeignKey("judges.id"), index=True)
    probe: Mapped[str] = mapped_column(String(32))  # order | length
    status: Mapped[str] = mapped_column(String(16), default="running")  # running|completed|failed
    # No default: the arm size is the experiment, not a counter. A row claiming zero
    # repeats would describe a run that made no calls, and the probe refuses to run
    # below two in the first place (at R=1 the control arm measures nothing).
    repeats: Mapped[int] = mapped_column(Integer)
    trajectory_count: Mapped[int] = mapped_column(Integer, default=0)
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    # The *unperturbed* fingerprint (judging.prompts.PROMPT_VERSION), because the probe
    # reports on the judge that runs in production; the departure from it is named per
    # row by BiasProbeResult.variant. Nullable for the same reason it is on EvalRun.
    judge_prompt_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    seed: Mapped[int] = mapped_column(Integer, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime] = mapped_column(TZDateTime(), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(TZDateTime(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(TZDateTime(), default=utcnow)

    judge: Mapped[Judge] = relationship(back_populates="bias_probe_runs")
    results: Mapped[list[BiasProbeResult]] = relationship(
        back_populates="probe_run", cascade="all, delete-orphan"
    )


class BiasProbeResult(Base):
    """One judge call in one arm of a probe, or the failure that replaced it.

    Parallel to ``JudgeVerdict`` and separate from it on purpose: nothing that queries
    ``judge_verdicts`` may pick these rows up, whatever filter it forgets.
    """

    __tablename__ = "bias_probe_results"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    probe_run_id: Mapped[str] = mapped_column(ForeignKey("bias_probe_runs.id"), index=True)
    trajectory_id: Mapped[str] = mapped_column(ForeignKey("trajectories.id"), index=True)
    variant: Mapped[str] = mapped_column(String(32))  # identity | format_null | padded_2x | ...
    repeat: Mapped[int] = mapped_column(Integer)  # 0-based draw within the arm
    verdict: Mapped[str | None] = mapped_column(String(16), nullable=True)  # null when errored
    # Kept for the same reason JudgeVerdict keeps it: a probe whose only output is a
    # flip count says that a reading changed but never which one, so nobody can act on it.
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # sha256 of the rendered user message, and the delivery check: an arm that hashes to
    # identity's never applied its perturbation and reports `not_applied`, never "no bias".
    prompt_sha: Mapped[str] = mapped_column(String(64))
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(TZDateTime(), default=utcnow)

    probe_run: Mapped[BiasProbeRun] = relationship(back_populates="results")
    # No `trajectory` relationship, deliberately. Assigning one would cascade the loaded
    # Trajectory and its steps into the session that writes probe rows, which is the only
    # route by which filler could ever reach the golden dataset. The foreign key is
    # enough: these rows are written from the id alone and read back by joining.
