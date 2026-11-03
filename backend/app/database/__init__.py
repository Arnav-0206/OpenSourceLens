"""Database module initialization for OpenSourceLens."""

from backend.app.database.base import Base
from backend.app.database.models import (
    Issue,
    IssueLabel,
    Prediction,
    RelatedIssuesCache,
    Repository,
    RepositoryAnalytics,
)
from backend.app.database.session import (
    SessionLocal,
    build_engine,
    engine,
    get_database_url,
    get_db,
    init_db,
)

__all__ = [
    "Base",
    "Repository",
    "Issue",
    "IssueLabel",
    "RepositoryAnalytics",
    "Prediction",
    "RelatedIssuesCache",
    "engine",
    "SessionLocal",
    "build_engine",
    "get_database_url",
    "get_db",
    "init_db",
]
