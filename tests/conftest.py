"""Shared fixtures: in-memory SQLite database, app wiring, and data factories."""

import itertools
from collections.abc import Callable, Generator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from agentverdict.api.app import create_app
from agentverdict.db import get_session
from agentverdict.models import Base, Step, Task, Trajectory

_key_counter = itertools.count(1)

DEFAULT_STEPS: list[tuple[str, dict[str, Any]]] = [
    ("user_message", {"text": "Hi, I want a refund for order A123"}),
    ("tool_call", {"name": "lookup_order", "arguments": {"order_id": "A123"}}),
    (
        "tool_result",
        {"name": "lookup_order", "result": {"status": "delivered"}, "is_error": False},
    ),
    ("assistant_message", {"text": "Your refund has been initiated."}),
]


@pytest.fixture
def engine() -> Generator[Engine, None, None]:
    """A fresh in-memory SQLite engine with all tables created."""
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(test_engine)
    yield test_engine
    test_engine.dispose()


@pytest.fixture
def session_factory(engine: Engine) -> sessionmaker[Session]:
    """Session factory bound to the per-test engine."""
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def session(session_factory: sessionmaker[Session]) -> Generator[Session, None, None]:
    """A session for direct database setup and verification within a test."""
    db = session_factory()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def app(session_factory: sessionmaker[Session]) -> FastAPI:
    """The application with its session dependency pointed at the test database.

    The override mirrors production semantics: commit at request end, roll back
    on exception, always close.
    """
    application = create_app()

    def override_get_session() -> Generator[Session, None, None]:
        db = session_factory()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    application.dependency_overrides[get_session] = override_get_session
    return application


@pytest.fixture
def client(app: FastAPI) -> Generator[TestClient, None, None]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def make_task(session: Session) -> Callable[..., Task]:
    """Factory committing a Task row; generates a unique key when none is given."""

    def _make(
        key: str | None = None,
        prompt: str = "Help the customer get a refund for a damaged item.",
        **kwargs: Any,
    ) -> Task:
        if key is None:
            key = f"task-{next(_key_counter):03d}"
        task = Task(key=key, prompt=prompt, **kwargs)
        session.add(task)
        session.commit()
        return task

    return _make


@pytest.fixture
def make_trajectory(
    session: Session, make_task: Callable[..., Task]
) -> Callable[..., Trajectory]:
    """Factory committing a Trajectory (with ordered steps) for a task."""

    def _make(
        task: Task | None = None,
        steps: list[tuple[str, dict[str, Any]]] | None = None,
        **kwargs: Any,
    ) -> Trajectory:
        if task is None:
            task = make_task()
        if steps is None:
            steps = DEFAULT_STEPS
        trajectory = Trajectory(
            task_id=task.id,
            steps=[
                Step(index=index, type=step_type, content=content)
                for index, (step_type, content) in enumerate(steps)
            ],
            **kwargs,
        )
        session.add(trajectory)
        session.commit()
        return trajectory

    return _make
