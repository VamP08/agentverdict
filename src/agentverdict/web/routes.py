"""Server-rendered labeling UI: queue, trajectory detail, and label submission."""

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote, unquote

from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from agentverdict.db import get_session
from agentverdict.models import HumanLabel, Step, Task, Trajectory
from agentverdict.schemas import TrajectorySummary

VERDICTS = ("pass", "fail", "borderline")
ANNOTATOR_COOKIE = "agentverdict_annotator"
COOKIE_MAX_AGE_S = 60 * 60 * 24 * 90

router = APIRouter(tags=["labeling-ui"])

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

SessionDep = Annotated[Session, Depends(get_session)]


def _pretty_json(value: Any) -> str:
    """Render step content as stable, human-readable JSON for <pre> blocks."""
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def _format_dt(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.strftime("%Y-%m-%d %H:%M")


templates.env.filters["tojson_pretty"] = _pretty_json
templates.env.filters["fmt_dt"] = _format_dt


def _get_trajectory_or_404(trajectory_id: str, session: Session) -> Trajectory:
    trajectory = session.get(Trajectory, trajectory_id)
    if trajectory is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trajectory {trajectory_id!r} not found",
        )
    return trajectory


def _next_unlabeled_id(session: Session, exclude_id: str | None = None) -> str | None:
    """The id of the oldest trajectory with no labels at all, or None when none remain."""
    has_label = select(HumanLabel.id).where(HumanLabel.trajectory_id == Trajectory.id).exists()
    stmt = select(Trajectory.id).where(~has_label)
    if exclude_id is not None:
        stmt = stmt.where(Trajectory.id != exclude_id)
    stmt = stmt.order_by(Trajectory.created_at, Trajectory.id).limit(1)
    return session.scalars(stmt).first()


def _render_trajectory(
    request: Request,
    session: Session,
    trajectory: Trajectory,
    *,
    form_annotator: str = "",
    form_verdict: str = "",
    form_rationale: str = "",
    form_error: str | None = None,
    status_code: int = 200,
) -> Response:
    labels = sorted(trajectory.labels, key=lambda label: (label.created_at, label.id))
    context = {
        "trajectory": trajectory,
        "task": trajectory.task,
        "steps": trajectory.steps,
        "labels": labels,
        "next_unlabeled_id": _next_unlabeled_id(session, exclude_id=trajectory.id),
        "form_annotator": form_annotator,
        "form_verdict": form_verdict,
        "form_rationale": form_rationale,
        "form_error": form_error,
    }
    return templates.TemplateResponse(request, "trajectory.html", context, status_code=status_code)


@router.get("/label", response_class=HTMLResponse)
def label_queue(request: Request, session: SessionDep) -> Response:
    """Queue of trajectories to label: unlabeled first, then labeled."""
    step_count = (
        select(func.count(Step.id)).where(Step.trajectory_id == Trajectory.id).scalar_subquery()
    )
    label_count = (
        select(func.count(HumanLabel.id))
        .where(HumanLabel.trajectory_id == Trajectory.id)
        .scalar_subquery()
    )
    rows = session.execute(
        select(
            Trajectory.id,
            Trajectory.task_id,
            Trajectory.source,
            Trajectory.status,
            Trajectory.created_at,
            Task.key.label("task_key"),
            step_count.label("step_count"),
            label_count.label("label_count"),
        ).join(Task, Trajectory.task_id == Task.id)
    ).all()
    summaries = [
        TrajectorySummary(
            id=row.id,
            task_id=row.task_id,
            task_key=row.task_key,
            source=row.source,
            status=row.status,
            step_count=row.step_count,
            label_count=row.label_count,
            created_at=row.created_at,
        )
        for row in rows
    ]
    summaries.sort(key=lambda s: (s.label_count > 0, s.created_at, s.id))
    labeled_count = sum(1 for s in summaries if s.label_count > 0)
    first_unlabeled_id = next((s.id for s in summaries if s.label_count == 0), None)
    context = {
        "trajectories": summaries,
        "total_count": len(summaries),
        "labeled_count": labeled_count,
        "first_unlabeled_id": first_unlabeled_id,
    }
    return templates.TemplateResponse(request, "queue.html", context)


@router.get("/label/{trajectory_id}", response_class=HTMLResponse)
def label_detail(request: Request, trajectory_id: str, session: SessionDep) -> Response:
    """Rendered trajectory steps plus the label form, prefilled from the annotator cookie."""
    trajectory = _get_trajectory_or_404(trajectory_id, session)
    last_annotator = unquote(request.cookies.get(ANNOTATOR_COOKIE, ""))
    return _render_trajectory(request, session, trajectory, form_annotator=last_annotator)


@router.post("/label/{trajectory_id}", response_class=HTMLResponse)
def label_submit(
    request: Request,
    trajectory_id: str,
    session: SessionDep,
    annotator: Annotated[str, Form()] = "",
    verdict: Annotated[str, Form()] = "",
    rationale: Annotated[str, Form()] = "",
) -> Response:
    """Save (create-or-replace) a label, then redirect to the next unlabeled trajectory."""
    trajectory = _get_trajectory_or_404(trajectory_id, session)

    annotator = annotator.strip()
    verdict = verdict.strip()
    rationale = rationale.strip()

    error: str | None = None
    if not annotator:
        error = "Enter an annotator name."
    elif len(annotator) > 100:
        error = "Annotator name must be 100 characters or fewer."
    elif verdict not in VERDICTS:
        error = "Choose a verdict: pass, fail, or borderline."
    if error is not None:
        return _render_trajectory(
            request,
            session,
            trajectory,
            form_annotator=annotator,
            form_verdict=verdict,
            form_rationale=rationale,
            form_error=error,
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    label = session.scalar(
        select(HumanLabel).where(
            HumanLabel.trajectory_id == trajectory.id,
            HumanLabel.annotator == annotator,
        )
    )
    if label is None:
        session.add(
            HumanLabel(
                trajectory_id=trajectory.id,
                annotator=annotator,
                verdict=verdict,
                rationale=rationale or None,
            )
        )
    else:
        label.verdict = verdict
        label.rubric_id = None
        label.rubric_scores = {}
        label.rationale = rationale or None
        label.time_spent_s = None
    session.flush()

    next_id = _next_unlabeled_id(session)
    target = f"/label/{next_id}" if next_id is not None else "/label"
    response = RedirectResponse(url=target, status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(
        ANNOTATOR_COOKIE,
        quote(annotator),
        max_age=COOKIE_MAX_AGE_S,
        httponly=True,
        samesite="lax",
    )
    return response
