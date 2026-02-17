"""
Shared environment-driven database configuration helpers.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env_files() -> None:
    """
    Load simple KEY=VALUE pairs from `.env` and `.env.local` (if present).
    Existing process environment variables are not overwritten.
    """

    project_root = Path(__file__).resolve().parents[1]
    for filename in (".env", ".env.local"):
        env_path = project_root / filename
        if not env_path.exists():
            continue

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def normalize_postgres_url(url: str) -> str:
    """
    Normalize postgres URLs to SQLAlchemy's recommended psycopg driver form.
    """

    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def resolve_database_url() -> str:
    """
    Resolve database URL using environment variables and optional .env files.

    Priority:
    1) DATABASE_URL
    2) CLOUD_DATABASE_URL when ENVIRONMENT is cloud-like
    3) LOCAL_DATABASE_URL
    """

    load_env_files()

    direct_url = os.getenv("DATABASE_URL")
    if direct_url:
        return normalize_postgres_url(direct_url)

    environment = os.getenv("ENVIRONMENT", "local").strip().lower()
    cloud_like_envs = {"prod", "production", "staging", "cloud"}

    cloud_url = os.getenv("CLOUD_DATABASE_URL")
    if environment in cloud_like_envs and cloud_url:
        return normalize_postgres_url(cloud_url)

    local_url = os.getenv("LOCAL_DATABASE_URL")
    if local_url:
        return normalize_postgres_url(local_url)

    raise RuntimeError(
        "No database URL configured. Set DATABASE_URL, or configure "
        "LOCAL_DATABASE_URL / CLOUD_DATABASE_URL."
    )
