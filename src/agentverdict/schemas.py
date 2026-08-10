"""Pydantic v2 schemas shared by the API, importer, CLI, and web UI."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

StepType = Literal["user_message", "assistant_message", "tool_call", "tool_result", "system"]
TrajectoryStatus = Literal["completed", "error", "truncated"]
TrajectorySource = Literal["api", "import", "manual", "replay", "langfuse"]
LabelVerdict = Literal["pass", "fail", "borderline"]


# --- Tasks -------------------------------------------------------------------


class TaskCreate(BaseModel):
    key: str = Field(min_length=1, max_length=200)
    prompt: str = Field(min_length=1)
    user_message: str | None = None
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


# --- Calibration (M3) --------------------------------------------------------


class ClassMetricsRead(BaseModel):
    label: str
    support: int
    predicted: int
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None


class AgreementRead(BaseModel):
    """One rater pair's agreement, serialized for the API and templates."""

    n: int
    labels: list[str] = Field(default_factory=list)
    matrix: dict[str, dict[str, int]] = Field(default_factory=dict)
    observed_agreement: float | None = None
    cohens_kappa: float | None = None
    weighted_kappa: float | None = None
    kappa_ci_low: float | None = None
    kappa_ci_high: float | None = None
    interpretation: str = "not enough data"
    per_class: list[ClassMetricsRead] = Field(default_factory=list)
    # Resamples that produced a defined kappa, out of those attempted. A shortfall
    # is why an interval may be missing, and is worth showing rather than hiding.
    bootstrap_usable: int = 0
    bootstrap_iterations: int = 0


class PairUncertainty(BaseModel):
    """Bootstrap uncertainty for a secondary agreement row.

    These rows are often computed over a handful of items, and the report tells
    readers to prefer them over the headline — so they carry the same interval and
    the same discard count. A bare kappa of 1.000 over three trajectories, shown
    without either, is exactly the overclaiming this project exists to catch.
    """

    kappa_ci_low: float | None = None
    kappa_ci_high: float | None = None
    bootstrap_usable: int = 0
    bootstrap_iterations: int = 0


class AnnotatorPairAgreement(PairUncertainty):
    """Agreement between two individual humans, on the items both labeled."""

    annotator_a: str
    annotator_b: str
    n: int
    observed_agreement: float | None = None
    cohens_kappa: float | None = None


class JudgeAnnotatorAgreement(PairUncertainty):
    """The judge against one annotator, on exactly the items both of them rated.

    The headline kappa is computed against the human *majority*, which silently
    drops trajectories where the annotators tied — so with two annotators the judge
    is scored only on the items they agreed about, an easier set than the one the
    annotators are scored on. That makes the headline number and the human figures
    non-comparable. These rows put every rater on the same items so the three
    numbers can honestly sit side by side.
    """

    annotator: str
    n: int
    observed_agreement: float | None = None
    cohens_kappa: float | None = None


class HeldOutAnnotatorAgreement(PairUncertainty):
    """One human scored the same way the judge is: against a majority of others.

    The judge is compared against the aggregated human majority, which is quieter
    than any single annotator. Comparing that to raw annotator-vs-annotator
    agreement is unfair to the judge, so each annotator is also held out and
    scored against the majority of the rest — the like-for-like reference point.
    """

    annotator: str
    n: int
    observed_agreement: float | None = None
    cohens_kappa: float | None = None


class DisagreementRow(BaseModel):
    """One trajectory where the judge and the humans reached different verdicts."""

    trajectory_id: str
    task_key: str
    human_verdict: str
    judge_verdict: str
    distance: int  # steps apart on the ordinal scale; 2 = pass vs fail
    annotators: list[str] = Field(default_factory=list)
    judge_rationale: str | None = None


class CalibrationReport(BaseModel):
    """How far a judge can be trusted, measured against the human golden set."""

    judge_id: str
    judge_name: str
    judge_model: str
    eval_run_id: str | None = None
    task_key: str | None = None
    annotators: list[str] = Field(default_factory=list)  # scope filter, empty = all
    generated_at: datetime

    # Annotators who actually contributed an in-scope label. A scope naming two
    # people where only one has labels still collapses the panel to one rater, so
    # this — not len(annotators) — is what tells a reader a "majority" is really
    # a single opinion.
    annotators_present: list[str] = Field(default_factory=list)

    judged_trajectories: int = 0  # trajectories this judge produced a verdict for
    labeled_trajectories: int = 0  # trajectories carrying at least one human label
    compared: int = 0  # trajectories with both, i.e. the sample kappa is computed on
    judge_errors: int = 0  # verdict rows recording a failed call
    ties_excluded: int = 0  # labeled trajectories with no human majority

    agreement: AgreementRead
    judge_vs_annotator: list[JudgeAnnotatorAgreement] = Field(default_factory=list)
    human_ceiling: list[AnnotatorPairAgreement] = Field(default_factory=list)
    human_baseline: list[HeldOutAnnotatorAgreement] = Field(default_factory=list)
    disagreements: list[DisagreementRow] = Field(default_factory=list)
    disagreements_total: int = 0  # before `disagreements` was truncated for display


# --- Regression gating (M5) --------------------------------------------------


class TaskScoreDelta(BaseModel):
    """One task's mean score in each run, and the change between them."""

    task_key: str
    base_score: float
    candidate_score: float
    delta: float
    base_trajectories: int
    candidate_trajectories: int


class RunComparison(BaseModel):
    """Whether a candidate eval run is worse than its baseline, and by how much.

    The verdict is deliberately three-valued. A suite score always moves a little
    between runs, so "the mean went down" is not evidence of anything; only an
    interval lying entirely below zero says the drop survives resampling the
    tasks. Anything else is `inconclusive`, which must not block a merge.
    """

    base_run_id: str
    candidate_run_id: str
    judge_id: str
    judge_name: str
    #: Fingerprint of the judge prompt both runs were graded with. Identical by
    #: construction — the comparison refuses to run across two versions of it — so one
    #: field, not two. None for runs recorded before the version was tracked, which is
    #: the one case where the rubric behind these numbers is genuinely unknown.
    judge_prompt_version: str | None = None

    tasks_compared: int
    base_score: float | None = None  # mean over compared tasks, 0.0-1.0
    candidate_score: float | None = None
    delta: float | None = None
    delta_ci_low: float | None = None
    delta_ci_high: float | None = None

    #: "regression" (interval entirely below zero), "improvement" (entirely
    #: above), or "inconclusive" (straddles zero, or too little data to resample).
    outcome: Literal["regression", "improvement", "inconclusive"] = "inconclusive"
    #: True only for "regression" — the single bit a CI gate acts on.
    blocks_merge: bool = False
    summary: str = ""

    base_only_tasks: list[str] = Field(default_factory=list)
    candidate_only_tasks: list[str] = Field(default_factory=list)
    task_deltas: list[TaskScoreDelta] = Field(default_factory=list)

    # Verdict rows where the judge call failed. Counting a failed call as `fail`
    # would turn a rate limit into a regression, so they are excluded from the
    # scores — but silently excluding them biases the result upward exactly when
    # it matters, because the calls that fail are the long, hard transcripts. A
    # run that errored on its worst tasks scores better than it deserves, so the
    # counts travel with the comparison and every surface reports them.
    base_judge_errors: int = 0
    candidate_judge_errors: int = 0

    input_tokens: int = 0
    output_tokens: int = 0
    generated_at: datetime


# --- Replay (M2) -------------------------------------------------------------


class ReplayReport(BaseModel):
    """Outcome of running an agent against a set of stored tasks."""

    adapter: str
    tasks_attempted: int = 0
    trajectories_created: int = 0
    error_count: int = 0
    trajectory_ids: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


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
