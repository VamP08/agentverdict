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
from sqlalchemy.orm import Session

from agentverdict.db import get_session_factory, init_db
from agentverdict.importer import compute_stats, export_jsonl, import_jsonl

app = typer.Typer(
    name="agentverdict",
    help="AgentVerdict: golden-dataset workbench for tool-calling agent evaluation.",
    no_args_is_help=True,
)


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
