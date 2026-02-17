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


def create_db_engine() -> Engine:
    database_url = resolve_database_url()
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


engine: Engine = create_db_engine()

SessionLocal = sessionmaker(
    bind=engine,
    class_=Session,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
