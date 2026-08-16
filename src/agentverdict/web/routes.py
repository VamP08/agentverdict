"""Server-rendered UI: the labeling queue, trajectory detail, and the calibration lab."""

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote, unquote

from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from agentverdict.calibration.report import build_calibration_report
from agentverdict.calibration.stats import VERDICT_ORDER, interpret_kappa
from agentverdict.db import get_session
from agentverdict.models import HumanLabel, Judge, JudgeVerdict, Step, Task, Trajectory
from agentverdict.rubric import (
    CRITERIA,
    RubricDriftError,
    ensure_rubric,
    score_label,
    validate_scores,
)
from agentverdict.schemas import AgreementRead, TrajectorySummary

VERDICTS = ("pass", "fail", "borderline")
ANNOTATOR_COOKIE = "agentverdict_annotator"
COOKIE_MAX_AGE_S = 60 * 60 * 24 * 90

#: Prefix for one criterion's radio group. The keys come from ``CRITERIA``, so no field
#: name here can be hand-typed; the prefix is what keeps a future criterion called
#: ``verdict`` or ``annotator`` from quietly overwriting the field it collides with.
CRITERION_FIELD_PREFIX = "criterion_"

#: The scores as an annotator reads them, keyed by the string the radio submits so the
#: word and the value cannot drift apart. A row of buttons labeled 1.0 / 0.5 / 0.0 asks a
#: person to translate the rubric on every screen. A value with no word here falls back to
#: the number, which is plain rather than pretty: what is shown is what gets recorded.
SCORE_WORDS = {"1.0": "yes", "0.5": "partly", "0.0": "no"}

#: The option offered for a criterion that can fail to arise. Its value is empty on
#: purpose: the field posts, the parser drops it, and the key never reaches
#: ``rubric_scores``. 0.0 is an answer -- "the agent took an irreversible action without
#: agreement" -- and spending it on a run that never faced the question would report a
#: consent violation on every look-up in the corpus.
ABSENT_WORD = "did not arise"
ABSENT_HINT = (
    '"did not arise" means nothing irreversible happened; '
    '"no" means it acted without asking.'
)

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


def _format_number(value: float | None, digits: int = 3) -> str:
    """Statistics print as 'n/a' when undefined, never as a misleading zero."""
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _format_percent(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def _kappa_badge(value: float | None) -> str:
    """Badge class for a kappa, so its band reads at a glance in the verdict palette."""
    if value is None:
        return "type-system"
    if value <= 0.40:  # worse than chance, slight, fair
        return "verdict-fail"
    if value <= 0.60:  # moderate
        return "verdict-borderline"
    return "verdict-pass"  # substantial, almost perfect


templates.env.filters["tojson_pretty"] = _pretty_json
templates.env.filters["fmt_dt"] = _format_dt
templates.env.filters["fmt_num"] = _format_number
templates.env.filters["fmt_pct"] = _format_percent
templates.env.filters["kappa_word"] = interpret_kappa
templates.env.filters["kappa_badge"] = _kappa_badge


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


def _criteria_view(submitted: Mapping[str, str]) -> list[dict[str, Any]]:
    """``CRITERIA`` as the form's radio groups, with ``submitted`` answers reselected.

    Assembled here because the word on a button and the value it posts have to come from
    the tuple the judge's prompt is rendered from. A template spelling out five groups
    would be a second copy of the rubric, and it is the copy nobody diffs that drifts.
    """
    groups: list[dict[str, Any]] = []
    for criterion in CRITERIA:
        values = [score_label(value) for value in criterion.values]
        if not criterion.applicable_always:
            values.append("")  # the "did not arise" option; see ABSENT_WORD
        # Nothing is checked until an annotator answers: every criterion is optional, and a
        # pre-selected option is an answer nobody gave. `selected` is None for a group that
        # was never posted and "" for one answered "did not arise".
        selected = submitted.get(criterion.key)
        groups.append(
            {
                "key": criterion.key,
                "field": CRITERION_FIELD_PREFIX + criterion.key,
                "question": criterion.description,
                "hint": None if criterion.applicable_always else ABSENT_HINT,
                "options": [
                    {
                        "value": value,
                        "word": ABSENT_WORD if value == "" else SCORE_WORDS.get(value, value),
                        "checked": selected is not None and value == selected,
                    }
                    for value in values
                ],
            }
        )
    return groups


def _submitted_scores(fields: Mapping[str, Any]) -> dict[str, str]:
    """The criterion fields exactly as posted; a group nobody touched has no key at all.

    Present-and-empty is kept, because that is what the "did not arise" option posts and
    it is the one answer an annotator has to click deliberately. Dropping it here would
    make a re-rendered form forget it.
    """
    submitted: dict[str, str] = {}
    for criterion in CRITERIA:
        raw = fields.get(CRITERION_FIELD_PREFIX + criterion.key)
        if raw is not None:
            submitted[criterion.key] = str(raw).strip()
    return submitted


def _parse_scores(submitted: Mapping[str, str]) -> dict[str, float]:
    """Submitted strings as rubric scores, raising ValueError on anything off the rubric.

    An unanswered criterion contributes no key rather than a 0.0. Absence records that the
    question did not arise, which is a different statement about a run than "the agent
    acted without agreement" -- and every read-only trajectory in the corpus sits on that
    difference.
    """
    scores = {key: float(raw) for key, raw in submitted.items() if raw}
    return validate_scores(scores)


def _render_trajectory(
    request: Request,
    session: Session,
    trajectory: Trajectory,
    *,
    form_annotator: str = "",
    form_verdict: str = "",
    form_rationale: str = "",
    form_scores: Mapping[str, str] | None = None,
    form_error: str | None = None,
    status_code: int = 200,
) -> Response:
    labels = sorted(trajectory.labels, key=lambda label: (label.created_at, label.id))
    judge_verdicts = session.execute(
        select(JudgeVerdict, Judge)
        .join(Judge, JudgeVerdict.judge_id == Judge.id)
        .where(JudgeVerdict.trajectory_id == trajectory.id)
        .order_by(JudgeVerdict.created_at.desc(), JudgeVerdict.id)
    ).all()
    context = {
        "trajectory": trajectory,
        "task": trajectory.task,
        "steps": trajectory.steps,
        "labels": labels,
        "judge_verdicts": judge_verdicts,
        "next_unlabeled_id": _next_unlabeled_id(session, exclude_id=trajectory.id),
        "form_annotator": form_annotator,
        "form_verdict": form_verdict,
        "form_rationale": form_rationale,
        "form_error": form_error,
        "criteria": _criteria_view(form_scores or {}),
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
    """Rendered trajectory steps plus the label form, prefilled from this annotator's own label.

    Reopening a run you have already judged shows what you said, not a blank form. That is
    worth more than the convenience: submitting replaces the whole row, so a blank form is
    a delete button wearing the shape of an edit. An annotator who came back to fix a typo
    in the rationale would silently drop five criterion answers they had spent time on,
    and nothing anywhere would say so.

    Prefill is scoped to the annotator named in the cookie. Showing one person's answers to
    another would be the anchoring this project takes some trouble to avoid -- the two
    annotators are measured against each other precisely because they judged independently.
    """
    trajectory = _get_trajectory_or_404(trajectory_id, session)
    last_annotator = unquote(request.cookies.get(ANNOTATOR_COOKIE, ""))
    mine = next(
        (label for label in trajectory.labels if label.annotator == last_annotator), None
    )
    if mine is None:
        return _render_trajectory(request, session, trajectory, form_annotator=last_annotator)
    return _render_trajectory(
        request,
        session,
        trajectory,
        form_annotator=last_annotator,
        form_verdict=mine.verdict,
        form_rationale=mine.rationale or "",
        # Back through the same string form the radios post, so a re-submit round-trips
        # what is stored rather than a re-rendering of it.
        form_scores={key: score_label(value) for key, value in (mine.rubric_scores or {}).items()},
    )


class LabelForm(BaseModel):
    """What the labeling form posts.

    ``extra="allow"`` is what carries the criterion groups: their field names are built
    from ``CRITERIA`` and so cannot appear in a hand-typed signature -- five named
    parameters here would be a second copy of the rubric, drifting silently the first time
    a criterion is renamed. :func:`_submitted_scores` picks them back out by key.
    """

    model_config = ConfigDict(extra="allow")

    annotator: str = ""
    verdict: str = ""
    rationale: str = ""


@router.post("/label/{trajectory_id}", response_class=HTMLResponse)
def label_submit(
    request: Request,
    trajectory_id: str,
    session: SessionDep,
    form: Annotated[LabelForm, Form()],
) -> Response:
    """Save (create-or-replace) a label, then redirect to the next unlabeled trajectory."""
    trajectory = _get_trajectory_or_404(trajectory_id, session)

    annotator = form.annotator.strip()
    verdict = form.verdict.strip()
    rationale = form.rationale.strip()
    submitted = _submitted_scores(form.model_extra or {})

    error: str | None = None
    scores: dict[str, float] = {}
    if not annotator:
        error = "Enter an annotator name."
    elif len(annotator) > 100:
        error = "Annotator name must be 100 characters or fewer."
    elif verdict not in VERDICTS:
        error = "Choose a verdict: pass, fail, or borderline."
    else:
        try:
            # Only a hand-built POST can land here -- the radios render from CRITERIA --
            # but a label carrying a criterion nobody else was asked about would reach
            # calibration as a column with no counterpart and still print a kappa.
            scores = _parse_scores(submitted)
        except ValueError as exc:
            error = f"Criterion scores were not accepted: {exc}"
    if error is not None:
        return _render_trajectory(
            request,
            session,
            trajectory,
            form_annotator=annotator,
            form_verdict=verdict,
            form_rationale=rationale,
            form_scores=submitted,
            form_error=error,
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    # Which rulebook was on screen, recorded per label rather than inferred later. It is
    # what lets a report say two rounds were collected under different rules instead of
    # averaging across a rubric change. Round-one labels carry no rubric_id at all, and
    # that gap is the honest record of rules that were never written down.
    try:
        rubric = ensure_rubric(session)
    except RubricDriftError:
        # No label can be honestly stamped while the stored rulebook disagrees with the
        # one on screen, so this submission could not save either way. But the annotator
        # is not who fixes it: a bare 500 costs them the rationale they just typed, says
        # nothing, and costs it again on every retry.
        return _render_trajectory(
            request,
            session,
            trajectory,
            form_annotator=annotator,
            form_verdict=verdict,
            form_rationale=rationale,
            form_scores=submitted,
            form_error=(
                "The stored rubric no longer matches the criteria this build defines, "
                "so no label can be recorded against it. Nothing you typed is lost. "
                "Ask an operator to run: agentverdict init-db"
            ),
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
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
                rubric_id=rubric.id,
                rubric_scores=scores,
                rationale=rationale or None,
            )
        )
    else:
        label.verdict = verdict
        label.rubric_id = rubric.id
        label.rubric_scores = scores
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


def _matrix_view(agreement: AgreementRead) -> dict[str, Any]:
    """Reshape the confusion matrix for rendering: human rows, judge columns, totals.

    The template stays declarative — diagonal (agreement) cells and the marginal
    totals are decided here rather than recomputed in Jinja.
    """
    labels = list(agreement.labels) or list(VERDICT_ORDER)
    rows: list[dict[str, Any]] = []
    for human in labels:
        counts = agreement.matrix.get(human, {})
        cells = []
        for judge_verdict in labels:
            count = counts.get(judge_verdict, 0)
            classes = []
            if judge_verdict == human:
                classes.append("diagonal")  # judge matched the humans
            if not count:
                classes.append("zero")
            cells.append(
                {"judge": judge_verdict, "count": count, "classes": " ".join(classes)}
            )
        rows.append(
            {"human": human, "cells": cells, "total": sum(cell["count"] for cell in cells)}
        )
    column_totals = [
        {"judge": judge_verdict, "count": sum(row["cells"][position]["count"] for row in rows)}
        for position, judge_verdict in enumerate(labels)
    ]
    total = sum(row["total"] for row in rows)
    matched = sum(agreement.matrix.get(label, {}).get(label, 0) for label in labels)
    return {
        "labels": labels,
        "rows": rows,
        "column_totals": column_totals,
        "total": total,
        "matched": matched,
        "mismatched": total - matched,
    }


@router.get("/calibration", response_class=HTMLResponse)
def calibration_index(request: Request, session: SessionDep) -> Response:
    """Every registered judge with its headline agreement against the human labels."""
    judges = list(session.scalars(select(Judge).order_by(Judge.name, Judge.id)))
    reports = [build_calibration_report(session, judge) for judge in judges]
    context = {
        "reports": reports,
        "judge_count": len(reports),
        "calibrated_count": sum(1 for report in reports if report.compared > 0),
    }
    return templates.TemplateResponse(request, "calibration_index.html", context)


@router.get("/calibration/{judge_id}", response_class=HTMLResponse)
def calibration_detail(request: Request, judge_id: str, session: SessionDep) -> Response:
    """One judge in full: matrix, per-class metrics, per-rater rows, ceiling, disagreements.

    The per-rater block is what keeps the page honest. The headline kappa is scored
    against the human majority, which silently drops annotator ties, so it is not
    computed on the same items as the human figures underneath it; the per-rater rows
    put the judge and each annotator on one shared item set.
    """
    judge = session.get(Judge, judge_id)
    if judge is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Judge {judge_id!r} not found",
        )
    report = build_calibration_report(session, judge)
    context = {
        "judge": judge,
        "report": report,
        "matrix": _matrix_view(report.agreement),
    }
    return templates.TemplateResponse(request, "calibration_detail.html", context)
