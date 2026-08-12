"""Read-only judge endpoints; judges are created via the CLI in M2.

Calibration (M3), the bias probes (M3 part 2) and run comparison (M5) are served
from here too. All three are pure analysis over rows that are already stored, so
these endpoints never call a model and need no API key — which is what makes them
safe to call from CI, and what keeps a page refresh from re-running a probe that
costs tokens.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from agentverdict.bias.runner import build_bias_probe_report
from agentverdict.calibration.report import build_calibration_report
from agentverdict.db import get_session
from agentverdict.gating.compare import compare_runs
from agentverdict.models import EvalRun, Judge, JudgeVerdict, Trajectory
from agentverdict.schemas import (
    BiasProbeReport,
    CalibrationReport,
    JudgeRead,
    JudgeVerdictRead,
    RunComparison,
)

router = APIRouter(prefix="/api", tags=["judges"])

SessionDep = Annotated[Session, Depends(get_session)]


def _get_judge_or_404(judge_id: str, session: Session) -> Judge:
    judge = session.get(Judge, judge_id)
    if judge is None:
        raise HTTPException(status_code=404, detail=f"Judge {judge_id!r} not found")
    return judge


def _get_eval_run_or_404(eval_run_id: str, session: Session) -> EvalRun:
    """Resolve an eval run, or 404 naming the id that could not be found.

    The id is echoed back because callers pass two of them to ``/comparisons``,
    and "one of your run ids is wrong" is not an answer anybody can act on.
    """
    eval_run = session.get(EvalRun, eval_run_id)
    if eval_run is None:
        raise HTTPException(status_code=404, detail=f"Eval run {eval_run_id!r} not found")
    return eval_run


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
    annotator: Annotated[
        list[str] | None,
        Query(
            description=(
                "Restrict human labels to these annotators; repeat the parameter for each "
                "one. Omit to use every annotator. Scoping a report to one annotation round "
                "keeps rounds from being averaged together."
            )
        ),
    ] = None,
) -> CalibrationReport:
    """Score a judge against the human-labeled golden set.

    Optionally scoped to a single eval run, a single task key, and/or a named set
    of annotators. An unknown eval run is a 404 rather than an empty report:
    silently reporting zero comparisons would read as a judge with nothing to
    answer for. An unknown annotator is not an error — it simply contributes no
    labels, which the report's coverage counts make visible.
    """
    judge = _get_judge_or_404(judge_id, session)
    if eval_run_id is not None:
        eval_run = _get_eval_run_or_404(eval_run_id, session)
        if eval_run.judge_id != judge.id:
            raise HTTPException(
                status_code=404,
                detail=f"Eval run {eval_run_id!r} does not belong to judge {judge.name!r}",
            )
    return build_calibration_report(
        session,
        judge,
        eval_run_id=eval_run_id,
        task_key=task_key,
        annotators=annotator,
    )


@router.get("/judges/{judge_id}/bias", response_model=BiasProbeReport)
def get_judge_bias(
    judge_id: str,
    session: SessionDep,
    probe: Annotated[
        str,
        Query(
            description=(
                "Which sensitivity to report: 'order' (the prompt's sections are "
                "re-ordered) or 'length' (filler is appended to agent turns). One "
                "report describes one probe, because their arms are not comparable."
            )
        ),
    ] = "order",
) -> BiasProbeReport:
    """Read back the latest stored bias probe for a judge.

    This endpoint reads probe rows and nothing else — it never calls a model, so
    refreshing the calibration page costs nothing and cannot start a sweep. The
    run it describes is the newest completed one, or the newest failed one when
    none completed; `agentverdict probe` is what produces them.

    A judge nobody has probed yet gets an empty report rather than an error: no
    probe has run, which is a state to describe rather than a fault. Every
    statistic in that report is null instead of zero, so a client cannot render it
    as a judge that passed a test that was never administered. An unknown judge is
    still a 404, and an unknown probe name a 400 naming the ones that exist —
    answering a typo with an empty report would read exactly like a clean bill of
    health.
    """
    judge = _get_judge_or_404(judge_id, session)
    try:
        return build_bias_probe_report(session, judge, probe=probe)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/comparisons", response_model=RunComparison)
def compare_eval_runs(
    session: SessionDep,
    base_run_id: Annotated[
        str,
        Query(description="Eval run holding the accepted quality bar, e.g. the last run on main."),
    ],
    candidate_run_id: Annotated[
        str,
        Query(description="Eval run under test, e.g. the one this pull request produced."),
    ],
) -> RunComparison:
    """Ask whether a candidate eval run is worse than its baseline.

    The answer is three-valued (`regression`, `improvement`, `inconclusive`) and
    only `regression` sets `blocks_merge`; the per-task deltas and the bootstrap
    interval behind that call come back with it, so a reader can check the verdict
    instead of trusting it.

    Both ids must name stored runs: an unknown id is a 404 rather than an empty
    comparison, because a gate that answers "inconclusive" to a typo is a gate that
    silently stops gating. Two runs scored by different judges — or by different
    versions of the judge prompt, which is the same problem wearing the same
    judge's name — are a 400: the scores would come off different measuring
    instruments, so their difference would describe the grader rather than the
    agent, and that is a request worth rejecting rather than a server fault.

    Reads only stored verdicts, so it calls no model and costs nothing to poll.
    """
    base_run = _get_eval_run_or_404(base_run_id, session)
    candidate_run = _get_eval_run_or_404(candidate_run_id, session)
    try:
        return compare_runs(session, base_run, candidate_run)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
