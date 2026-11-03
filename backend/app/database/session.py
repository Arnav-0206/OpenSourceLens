"""Database engine and session management for OpenSourceLens.

Supports PostgreSQL via DATABASE_URL environment variable with automatic
fallback to SQLite for local development and offline unit testing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.database.base import Base

# Default SQLite database path in data/ directory
DEFAULT_SQLITE_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "opensource_lens.db"
)


def get_database_url() -> str:
    """Resolve the effective database connection URL.

    Returns:
        DATABASE_URL from environment if set; otherwise SQLite file URL.
    """
    env_url = os.getenv("DATABASE_URL")
    if env_url and env_url.strip():
        # Standardize modern postgres driver prefix if needed
        if env_url.startswith("postgres://"):
            return env_url.replace("postgres://", "postgresql://", 1)
        return env_url

    DEFAULT_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{DEFAULT_SQLITE_PATH.as_posix()}"


def build_engine(url: Optional[str] = None) -> Engine:
    """Build a configured SQLAlchemy engine for PostgreSQL or SQLite.

    Args:
        url: Database URL. If None, resolved via get_database_url().

    Returns:
        Configured SQLAlchemy Engine.
    """
    db_url = url or get_database_url()
    connect_args = {}

    if db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        return create_engine(db_url, connect_args=connect_args)

    # PostgreSQL pool settings
    return create_engine(
        db_url,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
    )


engine: Engine = build_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db(target_engine: Optional[Engine] = None) -> None:
    """Create all registered database tables in the target engine.

    Args:
        target_engine: Engine to create tables on. Defaults to global engine.
    """
    eng = target_engine or engine
    Base.metadata.create_all(bind=eng)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a transactional database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
