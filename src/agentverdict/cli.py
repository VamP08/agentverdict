"""Typer command-line interface: init-db, import, export, stats, serve.

FastAPI/uvicorn are imported lazily inside ``serve`` so plain data commands
start fast and work without a running server.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from agentverdict.db import get_session_factory, init_db
from agentverdict.importer import compute_stats, export_jsonl, import_jsonl
from agentverdict.models import Judge, JudgeVerdict, Trajectory

app = typer.Typer(
    name="agentverdict",
    help="AgentVerdict: golden-dataset workbench for tool-calling agent evaluation.",
    no_args_is_help=True,
)
judge_app = typer.Typer(help="Manage LLM judges and evaluation runs.", no_args_is_help=True)
app.add_typer(judge_app, name="judge")


@contextmanager
def _open_session() -> Iterator[Session]:
    """Yield a session against an initialized database; commit on success."""
    init_db()
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@app.command("init-db")
def init_db_cmd() -> None:
    """Create all database tables (safe to run repeatedly)."""
    init_db()
    typer.echo("Database tables created.")


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
        int | None, typer.Option(help="Judge at most this many trajectories.")
    ] = None,
) -> None:
    """Judge stored trajectories with an LLM and print a scored summary."""
    from agentverdict.judging.client import JudgeClientError
    from agentverdict.judging.runner import run_eval

    def _progress(position: int, total: int, trajectory: Trajectory, row: JudgeVerdict) -> None:
        outcome = row.verdict or f"ERROR: {(row.error or 'unknown')[:60]}"
        typer.echo(f"  [{position}/{total}] {trajectory.task.key} {trajectory.id[:8]}: {outcome}")

    try:
        with _open_session() as session:
            judge = session.scalar(select(Judge).where(Judge.name == name))
            if judge is None:
                typer.echo(
                    f"No judge named '{name}'. Register it with: agentverdict judge add {name}",
                    err=True,
                )
                raise typer.Exit(code=1)
            run = run_eval(session, judge, task_key=task_key, limit=limit, on_progress=_progress)
    except JudgeClientError as exc:
        typer.echo(f"Judge run failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

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


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Bind port.")] = 8000,
    reload: Annotated[bool, typer.Option("--reload", help="Reload on code changes.")] = False,
) -> None:
    """Run the API and labeling UI with uvicorn."""
    import uvicorn

    init_db()
    uvicorn.run("agentverdict.api.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
