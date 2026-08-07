"""Typer command-line interface: init-db, import, export, stats, serve, judge, replay, eval.

Schema creation goes through Alembic (``agentverdict.migrate``) so upgrades of an
existing database work; ``db.init_db()`` remains the fast create_all path for tests.

Heavy or optional machinery — alembic, uvicorn, the agents package, the judging
package — is imported inside the command bodies so plain data commands start fast
and work without a model API key.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from agentverdict.db import get_session_factory
from agentverdict.importer import compute_stats, export_jsonl, import_jsonl
from agentverdict.models import EvalRun, Judge, JudgeVerdict, Task, Trajectory
from agentverdict.schemas import ReplayReport

if TYPE_CHECKING:  # import-time cost avoided; only needed for annotations
    from agentverdict.agents.base import AgentAdapter

#: Adapter names accepted by ``--adapter``. Adding an agent means adding a builder here.
ADAPTERS = ("groq-agent",)

app = typer.Typer(
    name="agentverdict",
    help="AgentVerdict: golden-dataset workbench for tool-calling agent evaluation.",
    no_args_is_help=True,
)
judge_app = typer.Typer(help="Manage LLM judges and evaluation runs.", no_args_is_help=True)
app.add_typer(judge_app, name="judge")


@contextmanager
def _open_session() -> Iterator[Session]:
    """Yield a session against an up-to-date database; commit on success."""
    from agentverdict.migrate import upgrade_to_head

    upgrade_to_head()
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _require_judge(session: Session, name: str) -> Judge:
    """Look up a judge by name, exiting with a helpful message when it is missing."""
    judge = session.scalar(select(Judge).where(Judge.name == name))
    if judge is None:
        typer.echo(
            f"No judge named '{name}'. Register it with: agentverdict judge add {name}",
            err=True,
        )
        raise typer.Exit(code=1)
    return judge


def _build_adapter(adapter: str, model: str | None) -> AgentAdapter:
    """Construct the named agent-under-test, wired to the bundled mock tool fixture."""
    if adapter not in ADAPTERS:
        typer.echo(
            f"Unknown adapter '{adapter}'. Valid adapters: {', '.join(ADAPTERS)}",
            err=True,
        )
        raise typer.Exit(code=1)

    from agentverdict.agents.groq_agent import GroqReferenceAgent
    from agentverdict.agents.mock_tools import MockToolRegistry, default_fixture_path

    # Load the fixture by explicit path so a missing or malformed file is reported
    # here, rather than silently yielding an agent whose every tool call errors.
    fixture = default_fixture_path()
    try:
        registry = MockToolRegistry.from_file(fixture)
    except (OSError, ValueError) as exc:
        typer.echo(f"Could not load mock tools from {fixture}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    return GroqReferenceAgent(model=model, registry=registry)


def _replay_progress(
    position: int,
    total: int,
    task: Task,
    trajectory: Trajectory | None,
    error: str | None,
) -> None:
    """Print one replayed attempt as it lands (exactly one of trajectory/error is set)."""
    if trajectory is None:
        typer.echo(f"  [{position}/{total}] {task.key}: ERROR: {(error or 'unknown')[:80]}")
        return
    typer.echo(
        f"  [{position}/{total}] {task.key}: {trajectory.status} ({len(trajectory.steps)} steps)"
    )


def _judge_progress(position: int, total: int, trajectory: Trajectory, row: JudgeVerdict) -> None:
    """Print one judged trajectory as it lands."""
    outcome = row.verdict or f"ERROR: {(row.error or 'unknown')[:60]}"
    typer.echo(f"  [{position}/{total}] {trajectory.task.key} {trajectory.id[:8]}: {outcome}")


def _print_replay_summary(report: ReplayReport) -> None:
    noun = "trajectory" if report.trajectories_created == 1 else "trajectories"
    tasks = "task" if report.tasks_attempted == 1 else "tasks"
    typer.echo(
        f"Replay complete: {report.trajectories_created} {noun} created from "
        f"{report.tasks_attempted} {tasks} via {report.adapter}"
    )
    if report.error_count:
        typer.echo(f"  errors      {report.error_count}")
        for error in report.errors:
            typer.echo(f"    ! {error}", err=True)
    typer.echo(f"  tokens      {report.input_tokens} in / {report.output_tokens} out")


def _print_eval_summary(run: EvalRun) -> None:
    typer.echo(f"Eval run {run.id[:8]} completed: {run.trajectory_count} trajectories")
    for verdict, count in sorted(run.verdict_counts.items()):
        typer.echo(f"  {verdict:<11} {count}")
    if run.error_count:
        typer.echo(f"  errors      {run.error_count}")
    typer.echo(f"  tokens      {run.input_tokens} in / {run.output_tokens} out")
    agreement = (run.meta or {}).get("human_agreement")
    if agreement:
        typer.echo(
            f"  human agreement: {agreement['agree']}/{agreement['total']}"
            f" ({agreement['rate']:.0%})"
        )


@app.command("init-db")
def init_db_cmd() -> None:
    """Create or upgrade the database schema (safe to run repeatedly)."""
    from agentverdict.migrate import upgrade_to_head

    upgrade_to_head()
    typer.echo("Database schema is up to date.")


@app.command("import")
def import_cmd(
    path: Annotated[Path, typer.Argument(help="Path to a JSONL trajectory bundle.")],
) -> None:
    """Import a JSONL trajectory bundle and print an import summary."""
    if not path.is_file():
        typer.echo(f"No such file: {path}", err=True)
        raise typer.Exit(code=1)
    with _open_session() as session:
        report = import_jsonl(session, path)
    typer.echo(f"Imported {path}:")
    typer.echo(f"  tasks created:        {report.tasks_created}")
    typer.echo(f"  trajectories created: {report.trajectories_created}")
    typer.echo(f"  steps created:        {report.steps_created}")
    if report.errors:
        typer.echo(f"  lines skipped:        {len(report.errors)}")
        for error in report.errors:
            typer.echo(f"    ! {error}", err=True)


@app.command("export")
def export_cmd(
    path: Annotated[Path, typer.Argument(help="Destination JSONL file.")],
    task_key: Annotated[
        str | None,
        typer.Option("--task-key", help="Only export trajectories for this task key."),
    ] = None,
) -> None:
    """Export trajectories (optionally filtered by task key) to a JSONL bundle."""
    with _open_session() as session:
        count = export_jsonl(session, path, task_key=task_key)
    noun = "trajectory" if count == 1 else "trajectories"
    scope = f" for task '{task_key}'" if task_key else ""
    typer.echo(f"Wrote {count} {noun}{scope} to {path}")


@app.command()
def stats() -> None:
    """Print dataset counts, label coverage, and the verdict histogram."""
    with _open_session() as session:
        report = compute_stats(session)
    if report.trajectory_count:
        coverage = f"{report.labeled_trajectory_count / report.trajectory_count:.0%}"
    else:
        coverage = "n/a"
    typer.echo(f"Tasks:                {report.task_count}")
    typer.echo(f"Trajectories:         {report.trajectory_count}")
    typer.echo(f"Labeled trajectories: {report.labeled_trajectory_count} ({coverage} coverage)")
    typer.echo(f"Labels:               {report.label_count}")
    if report.labels_by_verdict:
        typer.echo("Labels by verdict:")
        for verdict, count in sorted(report.labels_by_verdict.items()):
            typer.echo(f"  {verdict:<11} {count}")


@judge_app.command("add")
def judge_add(
    name: Annotated[str, typer.Argument(help="Unique judge name, e.g. groq-70b.")],
    model: Annotated[
        str | None, typer.Option(help="Model id (default: the configured judge model).")
    ] = None,
    description: Annotated[str | None, typer.Option(help="Free-text description.")] = None,
) -> None:
    """Register a judge (a model plus prompting recipe)."""
    from agentverdict.config import get_settings

    with _open_session() as session:
        existing = session.scalar(select(Judge).where(Judge.name == name))
        if existing is not None:
            typer.echo(f"Judge '{name}' already exists (model {existing.model}).", err=True)
            raise typer.Exit(code=1)
        judge = Judge(
            name=name, model=model or get_settings().judge_model, description=description
        )
        session.add(judge)
    typer.echo(f"Registered judge '{name}' using model {judge.model}.")


@judge_app.command("list")
def judge_list() -> None:
    """List registered judges with their verdict counts."""
    with _open_session() as session:
        rows = session.execute(
            select(Judge, func.count(JudgeVerdict.id))
            .outerjoin(JudgeVerdict, JudgeVerdict.judge_id == Judge.id)
            .group_by(Judge.id)
            .order_by(Judge.name)
        ).all()
        if not rows:
            typer.echo("No judges registered. Add one with: agentverdict judge add NAME")
            return
        for judge, verdict_count in rows:
            suffix = f" — {judge.description}" if judge.description else ""
            typer.echo(f"{judge.name}  [{judge.model}]  verdicts: {verdict_count}{suffix}")


@judge_app.command("run")
def judge_run(
    name: Annotated[str, typer.Argument(help="Name of a registered judge.")],
    task_key: Annotated[
        str | None, typer.Option("--task-key", help="Only judge trajectories of this task.")
    ] = None,
    limit: Annotated[
        int | None, typer.Option(help="Judge at most this many trajectories.", min=1)
    ] = None,
) -> None:
    """Judge stored trajectories with an LLM and print a scored summary."""
    from agentverdict.judging.client import JudgeClientError
    from agentverdict.judging.runner import run_eval

    try:
        with _open_session() as session:
            judge = _require_judge(session, name)
            run = run_eval(
                session, judge, task_key=task_key, limit=limit, on_progress=_judge_progress
            )
    except JudgeClientError as exc:
        typer.echo(f"Judge run failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    _print_eval_summary(run)


@app.command()
def replay(
    adapter: Annotated[
        str, typer.Option("--adapter", help=f"Agent under test ({', '.join(ADAPTERS)}).")
    ] = "groq-agent",
    task_key: Annotated[
        str | None, typer.Option("--task-key", help="Only replay this task key.")
    ] = None,
    limit: Annotated[
        int | None, typer.Option(help="Replay at most this many tasks.", min=1)
    ] = None,
    repeats: Annotated[int, typer.Option(help="Attempts per task.", min=1)] = 1,
    model: Annotated[
        str | None, typer.Option(help="Model id for the agent (default: the configured model).")
    ] = None,
) -> None:
    """Run an agent against stored tasks and record the trajectories it produces."""
    from agentverdict.agents.replay import run_replay
    from agentverdict.judging.client import JudgeClientError

    try:
        agent = _build_adapter(adapter, model)
        typer.echo(f"Replaying tasks with '{agent.name}'...")
        with _open_session() as session:
            report = run_replay(
                session,
                agent,
                task_key=task_key,
                limit=limit,
                repeats=repeats,
                on_progress=_replay_progress,
            )
    except JudgeClientError as exc:
        typer.echo(f"Replay failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    _print_replay_summary(report)


@app.command("eval")
def eval_cmd(
    judge_name: Annotated[str, typer.Option("--judge", help="Name of a registered judge.")],
    adapter: Annotated[
        str, typer.Option("--adapter", help=f"Agent under test ({', '.join(ADAPTERS)}).")
    ] = "groq-agent",
    task_key: Annotated[
        str | None, typer.Option("--task-key", help="Only use this task key.")
    ] = None,
    limit: Annotated[
        int | None, typer.Option(help="Cap tasks replayed / rows judged.", min=1)
    ] = None,
    repeats: Annotated[int, typer.Option(help="Attempts per task.", min=1)] = 1,
    model: Annotated[
        str | None, typer.Option(help="Model id for the agent (default: the configured model).")
    ] = None,
    skip_replay: Annotated[
        bool,
        typer.Option("--skip-replay", help="Judge existing trajectories instead of replaying."),
    ] = False,
) -> None:
    """Replay tasks and judge exactly those runs: the end-to-end evaluation."""
    from agentverdict.agents.replay import run_replay
    from agentverdict.judging.client import JudgeClientError
    from agentverdict.judging.runner import run_eval

    try:
        with _open_session() as session:
            # Resolve the judge before spending any tokens on replay.
            judge = _require_judge(session, judge_name)

            if skip_replay:
                typer.echo(f"Judging stored trajectories with judge '{judge.name}'...")
                run = run_eval(
                    session, judge, task_key=task_key, limit=limit, on_progress=_judge_progress
                )
            else:
                agent = _build_adapter(adapter, model)
                typer.echo(f"Replaying tasks with '{agent.name}'...")
                report = run_replay(
                    session,
                    agent,
                    task_key=task_key,
                    limit=limit,
                    repeats=repeats,
                    on_progress=_replay_progress,
                )
                # Commit the fresh trajectories so a later judging failure cannot
                # discard runs that already cost real model calls.
                session.commit()
                _print_replay_summary(report)
                if not report.trajectory_ids:
                    typer.echo("No trajectories were created, so there is nothing to judge.")
                    return
                fresh = len(report.trajectory_ids)
                typer.echo(
                    f"Judging {fresh} new {'trajectory' if fresh == 1 else 'trajectories'}"
                    f" with judge '{judge.name}'..."
                )
                # trajectory_ids pins the judged set to exactly what was just replayed;
                # task_key rides along only to record the filter on the eval run. `limit`
                # already bounded the replay, so re-applying it here would judge fewer
                # trajectories than were created.
                run = run_eval(
                    session,
                    judge,
                    task_key=task_key,
                    trajectory_ids=report.trajectory_ids,
                    on_progress=_judge_progress,
                )
    except JudgeClientError as exc:
        typer.echo(f"Eval failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    _print_eval_summary(run)


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Bind port.")] = 8000,
    reload: Annotated[bool, typer.Option("--reload", help="Reload on code changes.")] = False,
) -> None:
    """Run the API and labeling UI with uvicorn."""
    import uvicorn

    from agentverdict.migrate import upgrade_to_head

    upgrade_to_head()
    uvicorn.run("agentverdict.api.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
