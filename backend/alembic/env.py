"""Alembic environment configuration for OpenSourceLens.

Supports dual-dialect execution (PostgreSQL in production and SQLite in local/testing).
Automatically sets render_as_batch=True for SQLite schema migrations.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.database.base import Base
import backend.app.database.models  # Registers all ORM declarative models
from backend.app.database.session import get_database_url

# Alembic Config object
config = context.config

# Interpret logging configuration
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Model metadata for autogenerate and migrations
target_metadata = Base.metadata


def get_url() -> str:
    """Retrieve database URL dynamically with CLI and Config override support."""
    # Support -x db_url=sqlite:///... or -x db_path=...
    x_args = context.get_x_argument(as_dictionary=True)
    if "db_url" in x_args:
        return x_args["db_url"]
    if "db_path" in x_args:
        resolved_path = Path(x_args["db_path"]).resolve()
        return f"sqlite:///{resolved_path}"
    
    main_url = config.get_main_option("sqlalchemy.url")
    if main_url and not main_url.startswith("driver://"):
        return main_url

    return os.getenv("DATABASE_URL") or get_database_url()


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    url = get_url()
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = url

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
