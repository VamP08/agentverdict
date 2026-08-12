"""Eval-run orchestration: judge a set of trajectories and persist the verdicts.

One bad call never aborts a run — failures are recorded as error verdicts and the
run keeps going. Aggregates (verdict histogram, tokens, naive human agreement)
land on the EvalRun row.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from agentverdict.judging.client import GroqChatClient, JudgeClientError
from agentverdict.judging.prompts import CORRECTION_PROMPT, PROMPT_VERSION, build_messages
from agentverdict.models import EvalRun, HumanLabel, Judge, JudgeVerdict, Task, Trajectory, utcnow
from agentverdict.schemas import JudgeDecision

ProgressCallback = Callable[[int, int, Trajectory, JudgeVerdict], None]


def parse_decision(content: str) -> JudgeDecision:
    """Parse the judge's answer, raising ValueError with a useful message."""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"judge answered with invalid JSON: {exc}") from exc
    try:
        return JudgeDecision.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"judge JSON did not match the decision contract: {exc}") from exc


@dataclass(frozen=True)
class JudgeCall:
    """One completed judging call, with the corrective retry folded into the totals."""

    decision: JudgeDecision
    content: str  # the answer that parsed, which is what gets stored as the raw response
    latency_ms: float
    input_tokens: int
    output_tokens: int
    retried: bool


def judge_messages(
    client: GroqChatClient,
    judge: Judge,
    messages: list[dict[str, str]],
) -> JudgeCall:
    """Ask the judge about one already-rendered prompt, retrying once on malformed JSON.

    The single place that decides how this judge is called: which temperature, which
    correction turn, how a two-call answer is totalled. ``judge_trajectory`` is this
    function over the production prompt, and the bias probe is this function over a
    deliberately perturbed one — so a probe can never end up measuring a judge that
    production no longer runs.
    """
    temperature = float((judge.config or {}).get("temperature", 0.0))
    first = client.chat_json(judge.model, messages, temperature=temperature)
    retry = None
    try:
        decision = parse_decision(first.content)
    except ValueError:
        retry_messages = [
            *messages,
            {"role": "assistant", "content": first.content},
            {"role": "user", "content": CORRECTION_PROMPT},
        ]
        retry = client.chat_json(judge.model, retry_messages, temperature=temperature)
        decision = parse_decision(retry.content)
    return JudgeCall(
        decision=decision,
        content=(retry or first).content,
        latency_ms=first.latency_ms + (retry.latency_ms if retry else 0.0),
        input_tokens=first.input_tokens + (retry.input_tokens if retry else 0),
        output_tokens=first.output_tokens + (retry.output_tokens if retry else 0),
        retried=retry is not None,
    )


def judge_trajectory(
    client: GroqChatClient,
    judge: Judge,
    task: Task,
    trajectory: Trajectory,
) -> JudgeCall:
    """Judge one trajectory under the production prompt."""
    return judge_messages(client, judge, build_messages(task, trajectory))


def _human_majority(labels: list[HumanLabel]) -> str | None:
    """The majority human verdict for a trajectory, or None when empty or tied."""
    if not labels:
        return None
    counts = Counter(label.verdict for label in labels)
    ranked = counts.most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0]


def run_eval(
    session: Session,
    judge: Judge,
    *,
    task_key: str | None = None,
    limit: int | None = None,
    trajectory_ids: list[str] | None = None,
    client: GroqChatClient | None = None,
    on_progress: ProgressCallback | None = None,
) -> EvalRun:
    """Judge trajectories (optionally filtered) and return the completed EvalRun.

    ``trajectory_ids`` scores exactly those trajectories — used by ``eval`` to judge
    only the runs it just replayed. An empty list judges nothing.
    """
    stmt = (
        select(Trajectory)
        .join(Task, Trajectory.task_id == Task.id)
        .options(
            selectinload(Trajectory.steps),
            selectinload(Trajectory.task),
            selectinload(Trajectory.labels),
        )
        .order_by(Trajectory.created_at, Trajectory.id)
    )
    if task_key is not None:
        stmt = stmt.where(Task.key == task_key)
    if trajectory_ids is not None:
        # in_([]) is a valid empty-set filter in SQLAlchemy 2.0 and judges nothing.
        stmt = stmt.where(Trajectory.id.in_(trajectory_ids))
    if limit is not None:
        stmt = stmt.limit(limit)
    trajectories = list(session.scalars(stmt))

    # Stamped at creation, not at completion: the prompt in force when the grading
    # started is the one the verdicts were produced under, and a run that dies partway
    # still has to be attributable to a grader.
    run = EvalRun(
        judge=judge, task_key=task_key, status="running", judge_prompt_version=PROMPT_VERSION
    )
    session.add(run)
    session.flush()

    owns_client = client is None
    if owns_client:
        client = GroqChatClient()
    verdict_counts: Counter[str] = Counter()
    agreement_total = 0
    agreement_hits = 0
    try:
        for position, trajectory in enumerate(trajectories, start=1):
            # Explicit zeros: column defaults only apply at flush, and the error
            # path below reads these before that happens.
            verdict_row = JudgeVerdict(
                eval_run=run, judge=judge, trajectory=trajectory, input_tokens=0, output_tokens=0
            )
            try:
                call = judge_trajectory(client, judge, trajectory.task, trajectory)
            except (JudgeClientError, ValueError) as exc:
                verdict_row.error = str(exc)[:1000]
                run.error_count += 1
            else:
                decision = call.decision
                verdict_row.verdict = decision.verdict
                verdict_row.rationale = decision.rationale
                verdict_row.rubric_scores = decision.rubric_scores
                verdict_row.raw_response = {"content": call.content}
                verdict_row.latency_ms = call.latency_ms
                verdict_row.input_tokens = call.input_tokens
                verdict_row.output_tokens = call.output_tokens
                verdict_counts[decision.verdict] += 1
                majority = _human_majority(trajectory.labels)
                if majority is not None:
                    agreement_total += 1
                    if majority == decision.verdict:
                        agreement_hits += 1
            run.trajectory_count += 1
            run.input_tokens += verdict_row.input_tokens
            run.output_tokens += verdict_row.output_tokens
            session.add(verdict_row)
            session.flush()
            if on_progress is not None:
                on_progress(position, len(trajectories), trajectory, verdict_row)
    finally:
        if owns_client:
            client.close()

    run.verdict_counts = dict(verdict_counts)
    if agreement_total:
        run.meta = {
            "human_agreement": {
                "agree": agreement_hits,
                "total": agreement_total,
                "rate": round(agreement_hits / agreement_total, 4),
            }
        }
    run.status = "completed"
    run.completed_at = utcnow()
    session.flush()
    return run
