"""Alembic environment for AgentVerdict.

The database URL is resolved from application settings (``AGENTVERDICT_DATABASE_URL``)
rather than from ``alembic.ini``, so the CLI, the test suite, and manual ``alembic``
invocations all target the same database. A URL or an open connection injected
programmatically — ``config.set_main_option("sqlalchemy.url", ...)``,
``config.attributes["sqlalchemy_url"]``, ``config.attributes["connection"]`` — wins over
settings, which is how tests point migrations at a scratch database.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import Connection, create_engine, pool

from agentverdict.config import get_settings
from agentverdict.models import Base

# The placeholder alembic writes into a freshly generated alembic.ini; treated as absent.
PLACEHOLDER_URL_PREFIX = "driver://"

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def resolve_url() -> str:
    """Return the database URL: injected value first, then alembic.ini, then settings."""
    injected = config.attributes.get("sqlalchemy_url")
    if isinstance(injected, str) and injected.strip():
        return injected
    # get_section (unlike get_main_option) tolerates a Config with no [alembic] section.
    configured = config.get_section(config.config_ini_section, {}).get("sqlalchemy.url")
    if configured and not configured.startswith(PLACEHOLDER_URL_PREFIX):
        return configured
    return get_settings().database_url


def run_migrations_offline() -> None:
    """Render migrations as SQL against a URL, without creating an Engine or DBAPI."""
    context.configure(
        url=resolve_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations(connection: Connection) -> None:
    """Configure the migration context on an open connection and run it."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        # SQLite cannot ALTER columns in place; batch mode rewrites the table instead.
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live connection, reusing an injected one when given."""
    injected = config.attributes.get("connection")
    if isinstance(injected, Connection):
        run_migrations(injected)
        return

    engine = create_engine(resolve_url(), poolclass=pool.NullPool)
    try:
        with engine.connect() as connection:
            run_migrations(connection)
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
