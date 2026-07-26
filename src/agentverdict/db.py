"""Engine and session management.

The engine is created lazily from settings so tests can point AGENTVERDICT_DATABASE_URL
at a throwaway database (or override the FastAPI dependency entirely).
"""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from agentverdict.config import get_settings
from agentverdict.models import Base

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        url = get_settings().database_url
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, connect_args=connect_args)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _session_factory


def init_db() -> None:
    """Create all tables. M1 uses create_all; Alembic arrives with the first migration."""
    Base.metadata.create_all(get_engine())


def reset_engine() -> None:
    """Drop the cached engine/session factory (used by tests and the CLI)."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a session that commits on success, rolls back on error."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
