"""Read-only judge endpoints; judges are created via the CLI in M2.

Calibration is served from here too: it is pure analysis over stored verdicts and
human labels, so the endpoint never calls a model and needs no API key.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from agentverdict.calibration.report import build_calibration_report
from agentverdict.db import get_session
from agentverdict.models import EvalRun, Judge, JudgeVerdict, Trajectory
from agentverdict.schemas import CalibrationReport, JudgeRead, JudgeVerdictRead

router = APIRouter(prefix="/api", tags=["judges"])

SessionDep = Annotated[Session, Depends(get_session)]


def _get_judge_or_404(judge_id: str, session: Session) -> Judge:
    judge = session.get(Judge, judge_id)
    if judge is None:
        raise HTTPException(status_code=404, detail=f"Judge {judge_id!r} not found")
    return judge


@router.get("/judges", response_model=list[JudgeRead])
def list_judges(session: SessionDep) -> list[Judge]:
    return list(session.scalars(select(Judge).order_by(Judge.name)))


@router.get("/judges/{judge_id}", response_model=JudgeRead)
def get_judge(judge_id: str, session: SessionDep) -> Judge:
    """Fetch a single judge by id; 404 if missing."""
    return _get_judge_or_404(judge_id, session)


@router.get("/judges/{judge_id}/calibration", response_model=CalibrationReport)
def get_judge_calibration(
    judge_id: str,
    session: SessionDep,
    eval_run_id: str | None = None,
    task_key: str | None = None,
) -> CalibrationReport:
    """Score a judge against the human-labeled golden set.

    Optionally scoped to a single eval run and/or a single task key. An unknown
    eval run is a 404 rather than an empty report: silently reporting zero
    comparisons would read as a judge with nothing to answer for.
    """
    judge = _get_judge_or_404(judge_id, session)
    if eval_run_id is not None:
        eval_run = session.get(EvalRun, eval_run_id)
        if eval_run is None:
            raise HTTPException(status_code=404, detail=f"Eval run {eval_run_id!r} not found")
        if eval_run.judge_id != judge.id:
            raise HTTPException(
                status_code=404,
                detail=f"Eval run {eval_run_id!r} does not belong to judge {judge.name!r}",
            )
    return build_calibration_report(session, judge, eval_run_id=eval_run_id, task_key=task_key)


@router.get("/trajectories/{trajectory_id}/verdicts", response_model=list[JudgeVerdictRead])
def list_trajectory_verdicts(trajectory_id: str, session: SessionDep) -> list[JudgeVerdict]:
    if session.get(Trajectory, trajectory_id) is None:
        raise HTTPException(status_code=404, detail="Trajectory not found")
    stmt = (
        select(JudgeVerdict)
        .where(JudgeVerdict.trajectory_id == trajectory_id)
        .order_by(JudgeVerdict.created_at.desc(), JudgeVerdict.id)
    )
    return list(session.scalars(stmt))
