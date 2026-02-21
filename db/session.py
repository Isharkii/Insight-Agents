"""
db/session.py

SQLAlchemy engine and session factory.
"""

from __future__ import annotations

import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from db.config import resolve_database_url


def _get_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _validate_env() -> str:
    """Resolve and validate database configuration. Raises if env is misconfigured."""
    return resolve_database_url()


def create_db_engine() -> Engine:
    database_url = _validate_env()
    if not database_url.startswith("postgresql"):
        raise RuntimeError("Only PostgreSQL URLs are supported.")

    return create_engine(
        database_url,
        echo=_get_bool_env("SQL_ECHO", default=False),
        pool_pre_ping=True,
        pool_recycle=_get_int_env("DB_POOL_RECYCLE", 1800),
        pool_size=_get_int_env("DB_POOL_SIZE", 5),
        max_overflow=_get_int_env("DB_MAX_OVERFLOW", 10),
    )


_engine: Engine | None = None
_session_factory: sessionmaker | None = None


def get_engine() -> Engine:
    """Return the shared engine, creating it on first call."""
    global _engine
    if _engine is None:
        _engine = create_db_engine()
    return _engine


def _get_session_factory() -> sessionmaker:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(),
            class_=Session,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
    return _session_factory


def SessionLocal() -> Session:
    """Lazy session factory. Drop-in replacement for a sessionmaker() call."""
    return _get_session_factory()()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def __getattr__(name: str) -> object:
    # Provides lazy access to `engine` for callers that import it directly
    # (e.g. `from db.session import engine`). The engine is not created until
    # the attribute is first accessed.
    if name == "engine":
        return get_engine()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
