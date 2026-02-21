from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from db.base import Base
from db.config import load_env_files, normalize_postgres_url, resolve_database_url
from db.models import (  # noqa: F401  — imports trigger Base.metadata registration
    BusinessRiskScore,
    CanonicalInsightRecord,
    Client,
    ComputedKPI,
    Dataset,
    ForecastMetric,
    IngestionJob,
    MappingConfig,
)

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _resolve_database_url() -> str:
    """
    Resolve DB URL for migrations.

    Priority:
    1) `-x db_url=...` override for one-off migration targets
    2) ALEMBIC_DATABASE_URL
    3) sqlalchemy.url from alembic.ini
    4) DATABASE_URL
    5) CLOUD_DATABASE_URL in cloud-like environments
    6) LOCAL_DATABASE_URL
    """

    load_env_files()

    x_args = context.get_x_argument(as_dictionary=True)
    if "db_url" in x_args and x_args["db_url"]:
        url = normalize_postgres_url(x_args["db_url"])
    else:
        alembic_url = os.getenv("ALEMBIC_DATABASE_URL")
        if alembic_url:
            url = normalize_postgres_url(alembic_url)
        else:
            ini_url = (config.get_main_option("sqlalchemy.url") or "").strip()
            if ini_url:
                url = normalize_postgres_url(ini_url)
            else:
                url = resolve_database_url()

    if not url.startswith("postgresql"):
        raise RuntimeError("Alembic is configured for PostgreSQL URLs only.")

    return url


def run_migrations_offline() -> None:
    url = _resolve_database_url()

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _resolve_database_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        future=True,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
