"""Programmatic Alembic access: build a Config and upgrade a database to head.

This module lives outside the ``migrations`` package so the script directory stays a
pure Alembic environment, and it is imported only by the CLI and the tests — the
FastAPI app never pulls Alembic in.

``script_location`` is resolved relative to this file, so migrations run identically
from a source checkout and from an installed wheel (where there is no repo-root
``alembic.ini`` to anchor on).
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from agentverdict.config import get_settings
from agentverdict.models import Base

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

#: The revision representing the schema as it was before migrations existed.
BASELINE_REVISION = "0001_initial_schema"

#: Presence of this table means the database predates (or already has) the schema.
SENTINEL_TABLE = "tasks"


def _escape(value: str) -> str:
    """Escape percent signs: ConfigParser reads option values as pyformat templates."""
    return value.replace("%", "%%")


def build_alembic_config(database_url: str | None = None) -> Config:
    """Return an Alembic Config pointed at the packaged migrations and a database.

    The URL falls back to ``AGENTVERDICT_DATABASE_URL`` (via settings) when omitted.
    It is published both as the ``sqlalchemy.url`` option and, unescaped, as the
    ``sqlalchemy_url`` attribute that ``env.py`` prefers.
    """
    url = database_url or get_settings().database_url
    config = Config()
    config.set_main_option("script_location", _escape(str(MIGRATIONS_DIR)))
    config.set_main_option("sqlalchemy.url", _escape(url))
    config.attributes["sqlalchemy_url"] = url
    return config


def _adopt_unversioned_database(url: str, config: Config) -> None:
    """Bring a pre-migrations database under Alembic control, in place.

    Releases before migrations existed built their schema with ``create_all`` and
    left no ``alembic_version`` table, so a plain upgrade would try to create
    tables that are already there and abort. Such a database is stamped instead:
    with the baseline when it still predates a later column, or with ``head`` when
    ``create_all`` happened to produce the current schema already. Any table added
    by a later release but missing here is created first, since the baseline
    revision claims all of them exist.

    Databases that Alembic already tracks, and empty ones, are left untouched.
    """
    engine = create_engine(url)
    try:
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        if "alembic_version" in tables or SENTINEL_TABLE not in tables:
            return
        Base.metadata.create_all(engine)
        columns = {column["name"] for column in inspect(engine).get_columns(SENTINEL_TABLE)}
        already_current = "user_message" in columns
    finally:
        engine.dispose()
    command.stamp(config, "head" if already_current else BASELINE_REVISION)


def upgrade_to_head(database_url: str | None = None) -> None:
    """Apply every pending migration to the target database (a no-op when current).

    Safe to run against a fresh database, one Alembic already tracks, or one built
    by a release that predates migrations entirely.
    """
    url = database_url or get_settings().database_url
    config = build_alembic_config(url)
    _adopt_unversioned_database(url, config)
    command.upgrade(config, "head")
