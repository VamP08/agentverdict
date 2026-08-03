"""Pydantic v2 schemas shared by the API, importer, CLI, and web UI."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

StepType = Literal["user_message", "assistant_message", "tool_call", "tool_result", "system"]
TrajectoryStatus = Literal["completed", "error", "truncated"]
TrajectorySource = Literal["api", "import", "manual", "langfuse"]
LabelVerdict = Literal["pass", "fail", "borderline"]


# --- Tasks -------------------------------------------------------------------


class TaskCreate(BaseModel):
    key: str = Field(min_length=1, max_length=200)
    prompt: str = Field(min_length=1)
    tools_spec: list[dict[str, Any]] = Field(default_factory=list)
    expected_outcome: str | None = None
    tags: list[str] = Field(default_factory=list)


class TaskRead(TaskCreate):
    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: datetime


# --- Trajectories ------------------------------------------------------------


class StepIn(BaseModel):
    type: StepType
    content: dict[str, Any] = Field(default_factory=dict)


class StepRead(StepIn):
    model_config = ConfigDict(from_attributes=True)

    id: str
    index: int


class TrajectoryCreate(BaseModel):
    """Create a trajectory. Exactly one of task_id / task_key must reference an existing task."""

    task_id: str | None = None
    task_key: str | None = None
    agent_config: dict[str, Any] = Field(default_factory=dict)
    source: TrajectorySource = "api"
    status: TrajectoryStatus = "completed"
    meta: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    steps: list[StepIn] = Field(default_factory=list)


class TrajectoryRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    task_id: str
    agent_config: dict[str, Any]
    source: str
    status: str
    meta: dict[str, Any]
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime
    steps: list[StepRead] = Field(default_factory=list)


class TrajectorySummary(BaseModel):
    """List-view row; constructed manually in routes (label_count is computed)."""

    id: str
    task_id: str
    task_key: str
    source: str
    status: str
    step_count: int
    label_count: int
    created_at: datetime


# --- Labels ------------------------------------------------------------------


class LabelCreate(BaseModel):
    annotator: str = Field(min_length=1, max_length=100)
    verdict: LabelVerdict
    rubric_id: str | None = None
    rubric_scores: dict[str, float] = Field(default_factory=dict)
    rationale: str | None = None
    time_spent_s: float | None = None


class LabelRead(LabelCreate):
    model_config = ConfigDict(from_attributes=True)

    id: str
    trajectory_id: str
    created_at: datetime


# --- Judging (M2) ------------------------------------------------------------


class JudgeDecision(BaseModel):
    """The strict-JSON contract the LLM judge must answer with."""

    verdict: LabelVerdict
    rationale: str = Field(min_length=1)
    rubric_scores: dict[str, float] = Field(default_factory=dict)


class JudgeCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=200)
    description: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class JudgeRead(JudgeCreate):
    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: datetime


class JudgeVerdictRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    eval_run_id: str
    judge_id: str
    trajectory_id: str
    verdict: str | None
    rubric_scores: dict[str, float] = Field(default_factory=dict)
    rationale: str | None
    error: str | None
    latency_ms: float | None
    input_tokens: int
    output_tokens: int
    created_at: datetime


class EvalRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    judge_id: str
    task_key: str | None
    status: str
    trajectory_count: int
    error_count: int
    verdict_counts: dict[str, int] = Field(default_factory=dict)
    input_tokens: int
    output_tokens: int
    meta: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime
    completed_at: datetime | None
    created_at: datetime


# --- Import / stats ----------------------------------------------------------


class ImportReport(BaseModel):
    tasks_created: int = 0
    trajectories_created: int = 0
    steps_created: int = 0
    errors: list[str] = Field(default_factory=list)


class StatsReport(BaseModel):
    task_count: int
    trajectory_count: int
    labeled_trajectory_count: int
    label_count: int
    labels_by_verdict: dict[str, int] = Field(default_factory=dict)
