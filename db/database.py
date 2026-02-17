"""
db/database.py

Backward-compatible re-export of session primitives.
Prefer importing from `db.session` in new code.
"""

from db.session import SessionLocal, create_db_engine, engine, get_db

__all__ = ["engine", "SessionLocal", "get_db", "create_db_engine"]
