"""JSON routes for human labels on trajectories."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from agentverdict.db import get_session
from agentverdict.models import HumanLabel, Rubric, Trajectory
from agentverdict.rubric import validate_scores
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
    # A criterion nobody defined, or a 0.7 on a yes/no question, is refused rather than
    # stored: either one is read back by the calibration report as an answer, the first as
    # a column the judge was never asked to fill and the second as a category of its own in
    # the confusion matrix. Omitted keys stay omitted -- absence is how a question that did
    # not arise is recorded, and nothing here fills one in.
    try:
        scores = validate_scores(payload.rubric_scores)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    # A label that answers criteria was made under some rulebook, and which one is what
    # lets a report refuse to average across a rubric change. The web form stamps the
    # current rubric on every save; an API caller that scored criteria without naming a
    # rubric gets the same treatment rather than arriving provenance-blind. A bare
    # verdict is left unstamped -- that is what a round-one label legitimately looks
    # like, and inventing provenance for it would claim the annotator saw criteria
    # that did not exist when they judged.
    rubric_id = payload.rubric_id
    if rubric_id is None and scores:
        from agentverdict.rubric import ensure_rubric

        rubric_id = ensure_rubric(session).id
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
            rubric_id=rubric_id,
            rubric_scores=scores,
            rationale=payload.rationale,
            time_spent_s=payload.time_spent_s,
        )
        session.add(label)
    else:
        label.verdict = payload.verdict
        label.rubric_id = rubric_id
        label.rubric_scores = scores
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
