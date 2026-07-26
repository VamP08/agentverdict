"""JSON routes for human labels on trajectories."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from agentverdict.db import get_session
from agentverdict.models import HumanLabel, Rubric, Trajectory
from agentverdict.schemas import LabelCreate, LabelRead

router = APIRouter(prefix="/api/trajectories", tags=["labels"])

SessionDep = Annotated[Session, Depends(get_session)]


def _get_trajectory_or_404(trajectory_id: str, session: Session) -> Trajectory:
    trajectory = session.get(Trajectory, trajectory_id)
    if trajectory is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trajectory {trajectory_id!r} not found",
        )
    return trajectory


@router.post(
    "/{trajectory_id}/labels",
    response_model=LabelRead,
    status_code=status.HTTP_201_CREATED,
)
def submit_label(trajectory_id: str, payload: LabelCreate, session: SessionDep) -> LabelRead:
    """Create a label; a re-submission by the same annotator updates the existing row."""
    _get_trajectory_or_404(trajectory_id, session)
    if payload.rubric_id is not None and session.get(Rubric, payload.rubric_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Rubric {payload.rubric_id!r} not found",
        )
    label = session.scalar(
        select(HumanLabel).where(
            HumanLabel.trajectory_id == trajectory_id,
            HumanLabel.annotator == payload.annotator,
        )
    )
    if label is None:
        label = HumanLabel(
            trajectory_id=trajectory_id,
            annotator=payload.annotator,
            verdict=payload.verdict,
            rubric_id=payload.rubric_id,
            rubric_scores=payload.rubric_scores,
            rationale=payload.rationale,
            time_spent_s=payload.time_spent_s,
        )
        session.add(label)
    else:
        label.verdict = payload.verdict
        label.rubric_id = payload.rubric_id
        label.rubric_scores = payload.rubric_scores
        label.rationale = payload.rationale
        label.time_spent_s = payload.time_spent_s
    session.flush()
    return LabelRead.model_validate(label)


@router.get("/{trajectory_id}/labels", response_model=list[LabelRead])
def list_labels(trajectory_id: str, session: SessionDep) -> list[LabelRead]:
    """List all labels for a trajectory; 404 if the trajectory doesn't exist."""
    _get_trajectory_or_404(trajectory_id, session)
    labels = session.scalars(
        select(HumanLabel)
        .where(HumanLabel.trajectory_id == trajectory_id)
        .order_by(HumanLabel.created_at, HumanLabel.id)
    ).all()
    return [LabelRead.model_validate(label) for label in labels]
