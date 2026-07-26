"""Tests for JSONL import/export and dataset stats, using the shipped sample bundle."""

import json
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from agentverdict.importer import compute_stats, export_jsonl, import_jsonl
from agentverdict.models import Base, HumanLabel, Step, Task, Trajectory

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "examples" / "sample_trajectories.jsonl"

VALID_LINE: dict[str, Any] = {
    "task": {"key": "tmp-task-01", "prompt": "Check the status of an order."},
    "trajectory": {
        "status": "completed",
        "steps": [{"type": "user_message", "content": {"text": "Where is my order?"}}],
    },
}


def _sample_lines() -> list[dict[str, Any]]:
    text = SAMPLE_PATH.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _unique_task_keys(lines: list[dict[str, Any]]) -> set[str]:
    return {line["task"]["key"] for line in lines}


def _total_steps(lines: list[dict[str, Any]]) -> int:
    return sum(len(line["trajectory"]["steps"]) for line in lines)


def test_import_sample_bundle_matches_file_contents(session: Session) -> None:
    lines = _sample_lines()
    assert lines, "sample bundle must not be empty"

    report = import_jsonl(session, SAMPLE_PATH)
    session.commit()

    assert report.errors == []
    assert report.tasks_created == len(_unique_task_keys(lines))
    assert report.trajectories_created == len(lines)
    assert report.steps_created == _total_steps(lines)

    assert session.scalar(select(func.count()).select_from(Task)) == report.tasks_created
    assert session.scalar(select(func.count()).select_from(Trajectory)) == len(lines)
    assert session.scalar(select(func.count()).select_from(Step)) == report.steps_created


def test_reimport_does_not_duplicate_tasks(session: Session) -> None:
    lines = _sample_lines()
    import_jsonl(session, SAMPLE_PATH)
    session.commit()

    second = import_jsonl(session, SAMPLE_PATH)
    session.commit()

    assert second.errors == []
    assert second.tasks_created == 0
    assert second.trajectories_created == len(lines)
    task_count = session.scalar(select(func.count()).select_from(Task))
    assert task_count == len(_unique_task_keys(lines))


def test_malformed_lines_are_reported_and_valid_lines_import(
    session: Session, tmp_path: Path
) -> None:
    invalid_schema_line = {
        "task": {"key": "tmp-task-02", "prompt": "x"},
        "trajectory": {"steps": [{"type": "thought", "content": {}}]},
    }
    bundle = tmp_path / "bundle.jsonl"
    bundle.write_text(
        '{"task": broken\n'
        + json.dumps(VALID_LINE)
        + "\n"
        + json.dumps(invalid_schema_line)
        + "\n",
        encoding="utf-8",
    )

    report = import_jsonl(session, bundle)
    session.commit()

    assert len(report.errors) == 2
    assert report.errors[0].startswith("line 1")
    assert report.errors[1].startswith("line 3")
    assert report.tasks_created == 1
    assert report.trajectories_created == 1
    assert report.steps_created == 1
    assert session.scalar(select(func.count()).select_from(Trajectory)) == 1
    assert session.scalar(select(Task.key)) == "tmp-task-01"


def test_export_reimport_roundtrip_preserves_counts(session: Session, tmp_path: Path) -> None:
    lines = _sample_lines()
    import_jsonl(session, SAMPLE_PATH)
    session.commit()

    out = tmp_path / "export.jsonl"
    written = export_jsonl(session, out)
    assert written == len(lines)

    fresh_engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(fresh_engine)
    fresh_factory = sessionmaker(bind=fresh_engine, expire_on_commit=False)
    try:
        with fresh_factory() as fresh:
            report = import_jsonl(fresh, out)
            fresh.commit()
            assert report.errors == []
            assert report.trajectories_created == len(lines)
            assert report.steps_created == _total_steps(lines)
            trajectory_count = fresh.scalar(select(func.count()).select_from(Trajectory))
            step_count = fresh.scalar(select(func.count()).select_from(Step))
            task_count = fresh.scalar(select(func.count()).select_from(Task))
            assert trajectory_count == len(lines)
            assert step_count == _total_steps(lines)
            assert task_count == len(_unique_task_keys(lines))
    finally:
        fresh_engine.dispose()


def test_export_filters_by_task_key(session: Session, tmp_path: Path) -> None:
    lines = _sample_lines()
    import_jsonl(session, SAMPLE_PATH)
    session.commit()

    key = lines[0]["task"]["key"]
    expected = sum(1 for line in lines if line["task"]["key"] == key)
    out = tmp_path / "filtered.jsonl"
    assert export_jsonl(session, out, task_key=key) == expected

    exported = [
        json.loads(line)
        for line in out.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(exported) == expected
    assert all(line["task"]["key"] == key for line in exported)


def test_compute_stats_before_and_after_labeling(session: Session) -> None:
    lines = _sample_lines()
    import_jsonl(session, SAMPLE_PATH)
    session.commit()

    before = compute_stats(session)
    assert before.task_count == len(_unique_task_keys(lines))
    assert before.trajectory_count == len(lines)
    assert before.label_count == 0
    assert before.labeled_trajectory_count == 0
    assert before.labels_by_verdict == {}

    ids = session.scalars(select(Trajectory.id).order_by(Trajectory.id)).all()
    session.add_all(
        [
            HumanLabel(trajectory_id=ids[0], annotator="alice", verdict="pass"),
            HumanLabel(trajectory_id=ids[0], annotator="bob", verdict="fail"),
            HumanLabel(trajectory_id=ids[1], annotator="alice", verdict="pass"),
        ]
    )
    session.commit()

    after = compute_stats(session)
    assert after.task_count == before.task_count
    assert after.trajectory_count == before.trajectory_count
    assert after.label_count == 3
    assert after.labeled_trajectory_count == 2
    assert after.labels_by_verdict == {"pass": 2, "fail": 1}


def test_existing_task_fields_are_not_overwritten(session: Session, tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    first.write_text(json.dumps(VALID_LINE) + "\n", encoding="utf-8")
    import_jsonl(session, first)
    session.commit()

    modified = json.loads(json.dumps(VALID_LINE))
    modified["task"]["prompt"] = "A completely different prompt."
    modified["task"]["tags"] = ["changed"]
    second = tmp_path / "second.jsonl"
    second.write_text(json.dumps(modified) + "\n", encoding="utf-8")
    report = import_jsonl(session, second)
    session.commit()

    assert report.errors == []
    assert report.tasks_created == 0
    assert report.trajectories_created == 1
    task = session.scalar(select(Task).where(Task.key == "tmp-task-01"))
    assert task is not None
    assert task.prompt == VALID_LINE["task"]["prompt"]
    assert task.tags == []
