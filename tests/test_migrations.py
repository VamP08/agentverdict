"""Alembic migrations must describe exactly the schema the models declare.

M1 created tables with ``create_all``; from M2 on, a deployed database is upgraded
by migrations instead. Nothing keeps the two in step automatically, so this compares
a freshly migrated database against ``Base.metadata`` and fails on any drift.

Everything runs against a throwaway SQLite file under ``tmp_path``. The fixture
temporarily repoints the application settings at it and restores the cached
settings and engine afterwards, so the rest of the suite is unaffected.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect, text

from agentverdict.config import get_settings
from agentverdict.db import reset_engine
from agentverdict.migrate import BASELINE_REVISION, build_alembic_config, upgrade_to_head
from agentverdict.models import Base


@pytest.fixture
def migrated_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A scratch SQLite database upgraded to head, with global caches restored after."""
    url = f"sqlite:///{(tmp_path / 'migrated.db').as_posix()}"
    # upgrade_to_head takes the URL directly; the env var covers anything that reads
    # settings instead (the alembic env falls back to them when nothing is injected).
    monkeypatch.setenv("AGENTVERDICT_DATABASE_URL", url)
    get_settings.cache_clear()
    reset_engine()
    try:
        upgrade_to_head(url)
        yield url
    finally:
        # Leave the process-wide caches exactly as they were found.
        reset_engine()
        get_settings.cache_clear()


def _describe(diffs: list[Any]) -> str:
    """Flatten alembic's diff tuples (column diffs arrive nested) into readable lines."""
    lines: list[str] = []
    for diff in diffs:
        if isinstance(diff, list):
            lines.extend(f"  {entry}" for entry in diff)
        else:
            lines.append(f"  {diff}")
    return "\n".join(lines)


def _diff_against_models(url: str) -> list[Any]:
    engine = create_engine(url)
    try:
        with engine.connect() as connection:
            # compare_type mirrors migrations/env.py, so the test sees the same
            # differences an `alembic revision --autogenerate` would.
            context = MigrationContext.configure(connection, opts={"compare_type": True})
            return compare_metadata(context, Base.metadata)
    finally:
        engine.dispose()


def test_migrations_create_every_modelled_table(migrated_url: str) -> None:
    engine = create_engine(migrated_url)
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()

    assert tables == set(Base.metadata.tables) | {"alembic_version"}


def test_migrations_match_the_models(migrated_url: str) -> None:
    diffs = _diff_against_models(migrated_url)
    assert diffs == [], (
        "Alembic migrations have drifted from the SQLAlchemy models. Add a revision "
        f"covering these differences:\n{_describe(diffs)}"
    )


def test_upgrade_to_head_is_idempotent(migrated_url: str) -> None:
    config = build_alembic_config(migrated_url)
    assert config.attributes["sqlalchemy_url"] == migrated_url

    upgrade_to_head(migrated_url)  # already at head: a no-op, not an error

    assert _diff_against_models(migrated_url) == []


def _scratch_url(tmp_path: Path, name: str) -> str:
    return f"sqlite:///{(tmp_path / name).as_posix()}"


def test_database_predating_migrations_is_adopted(tmp_path: Path) -> None:
    """A database left at the pre-migration schema upgrades instead of colliding.

    Releases before Alembic built their tables with ``create_all`` and recorded no
    revision, so a plain upgrade would try to create tables that already exist.
    """
    url = _scratch_url(tmp_path, "legacy.db")
    command.upgrade(build_alembic_config(url), BASELINE_REVISION)
    engine = create_engine(url)
    try:
        # Strip the stamp: what an install that never ran Alembic looks like.
        with engine.begin() as connection:
            connection.execute(text("DROP TABLE alembic_version"))
        with engine.connect() as connection:
            before = {column["name"] for column in inspect(connection).get_columns("tasks")}
    finally:
        engine.dispose()
    assert "user_message" not in before

    upgrade_to_head(url)

    engine = create_engine(url)
    try:
        after = {column["name"] for column in inspect(engine).get_columns("tasks")}
    finally:
        engine.dispose()
    assert "user_message" in after
    assert _diff_against_models(url) == []


def test_database_already_matching_the_models_is_stamped(tmp_path: Path) -> None:
    """A create_all database that happens to be current is stamped, not re-created."""
    url = _scratch_url(tmp_path, "create_all.db")
    engine = create_engine(url)
    try:
        Base.metadata.create_all(engine)
    finally:
        engine.dispose()

    upgrade_to_head(url)  # must not raise "table already exists" or "duplicate column"

    assert _diff_against_models(url) == []
