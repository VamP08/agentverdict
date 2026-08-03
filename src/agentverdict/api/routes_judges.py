"""Read-only judge endpoints; judges are created via the CLI in M2."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from agentverdict.db import get_session
from agentverdict.models import Judge, JudgeVerdict, Trajectory
from agentverdict.schemas import JudgeRead, JudgeVerdictRead

router = APIRouter(prefix="/api", tags=["judges"])

SessionDep = Annotated[Session, Depends(get_session)]


@router.get("/judges", response_model=list[JudgeRead])
def list_judges(session: SessionDep) -> list[Judge]:
    return list(session.scalars(select(Judge).order_by(Judge.name)))


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
