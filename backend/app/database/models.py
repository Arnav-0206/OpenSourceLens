"""SQLAlchemy 2.0 ORM Declarative Models for OpenSourceLens.

Supports dual-driver execution (PostgreSQL for production, SQLite for local
testing and offline experimentation).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database.base import Base


# Reusable surrogate PK type: BIGSERIAL on PostgreSQL, INTEGER PRIMARY KEY on SQLite
PK_BIGINT = BigInteger().with_variant(Integer, "sqlite")


class Repository(Base):
    """Represents a GitHub repository monitored by OpenSourceLens."""

    __tablename__ = "repositories"

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=False, comment="GitHub repository ID"
    )
    repo_name: Mapped[str] = mapped_column(
        String(120), unique=True, index=True, nullable=False, comment="owner/repo"
    )
    owner: Mapped[str] = mapped_column(String(80), nullable=False)
    name: Mapped[str] = mapped_column(String(80), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    primary_language: Mapped[Optional[str]] = mapped_column(
        String(60), nullable=True
    )
    stars_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    forks_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    open_issues_count: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    github_created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    github_updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    collected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    issues: Mapped[List[Issue]] = relationship(
        "Issue", back_populates="repository", cascade="all, delete-orphan"
    )
    analytics: Mapped[List[RepositoryAnalytics]] = relationship(
        "RepositoryAnalytics",
        back_populates="repository",
        cascade="all, delete-orphan",
    )


class Issue(Base):
    """Represents a GitHub issue with cleaned text, NLP attributes, and resolution metrics."""

    __tablename__ = "issues"

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=False, comment="GitHub issue ID"
    )
    repository_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("repositories.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    repo_name: Mapped[str] = mapped_column(
        String(120), index=True, nullable=False
    )
    issue_number: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    body: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    clean_title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    clean_body: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    combined_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    clean_title_word_count: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    clean_body_word_count: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )

    # Lifecycle & Author Metadata
    state: Mapped[str] = mapped_column(
        String(30), index=True, nullable=False, comment="open or closed"
    )
    state_reason: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )
    author: Mapped[str] = mapped_column(String(100), nullable=False)
    author_type: Mapped[str] = mapped_column(
        String(40), default="User", nullable=False
    )
    author_association: Mapped[str] = mapped_column(
        String(40), default="NONE", nullable=False
    )
    is_internal_author: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    is_bot_author: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True, nullable=False
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    closed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), index=True, nullable=True
    )

    # Activity Counts
    comments_count: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    reactions_count: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    labels_count: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )

    # Content Characteristics
    has_code_block: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    code_block_count: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    has_url: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    has_stack_trace: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )

    # Canonical Classification Category
    canonical_label: Mapped[str] = mapped_column(
        String(50), index=True, default="other", nullable=False
    )
    canonical_label_id: Mapped[int] = mapped_column(
        Integer, default=-1, nullable=False
    )
    is_classified_category: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    is_unlabeled: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )

    # Resolution Metrics & Velocity Tiers
    resolution_time_hours: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    resolution_time_days: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    speed_tier: Mapped[Optional[str]] = mapped_column(
        String(30), nullable=True, index=True
    )
    resolution_type: Mapped[Optional[str]] = mapped_column(
        String(30), nullable=True
    )
    is_censored: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    is_immediate_closure: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    duration_at_risk_days: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )

    # Composite Indexes
    __table_args__ = (
        Index("ix_issues_repo_number", "repo_name", "issue_number", unique=True),
        Index("ix_issues_repo_state", "repo_name", "state"),
        Index("ix_issues_repo_canonical", "repo_name", "canonical_label"),
        Index("ix_issues_repo_speed_tier", "repo_name", "speed_tier"),
    )

    # Relationships
    repository: Mapped[Repository] = relationship(
        "Repository", back_populates="issues"
    )
    labels: Mapped[List[IssueLabel]] = relationship(
        "IssueLabel", back_populates="issue", cascade="all, delete-orphan"
    )
    predictions: Mapped[List[Prediction]] = relationship(
        "Prediction", back_populates="issue"
    )
    source_related: Mapped[List[RelatedIssuesCache]] = relationship(
        "RelatedIssuesCache",
        foreign_keys="[RelatedIssuesCache.source_issue_id]",
        back_populates="source_issue",
        cascade="all, delete-orphan",
    )
    target_related: Mapped[List[RelatedIssuesCache]] = relationship(
        "RelatedIssuesCache",
        foreign_keys="[RelatedIssuesCache.target_issue_id]",
        back_populates="target_issue",
        cascade="all, delete-orphan",
    )


class IssueLabel(Base):
    """Normalized labels associated with GitHub issues."""

    __tablename__ = "issue_labels"

    id: Mapped[int] = mapped_column(
        PK_BIGINT, primary_key=True, autoincrement=True
    )
    issue_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("issues.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    label_name: Mapped[str] = mapped_column(
        String(100), index=True, nullable=False
    )
    is_canonical: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    canonical_category: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )

    # Relationships
    issue: Mapped[Issue] = relationship("Issue", back_populates="labels")


class RepositoryAnalytics(Base):
    """Aggregated resolution metrics and distribution snapshots per repository."""

    __tablename__ = "repository_analytics"

    id: Mapped[int] = mapped_column(
        PK_BIGINT, primary_key=True, autoincrement=True
    )
    repository_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("repositories.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    repo_name: Mapped[str] = mapped_column(
        String(120), index=True, nullable=False
    )
    snapshot_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    total_issues: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    open_issues: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    closed_issues: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    category_distribution: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON, nullable=True
    )
    median_resolution_hours: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    mean_resolution_hours: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    p25_resolution_hours: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    p75_resolution_hours: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    survival_rate_30d: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    speed_tier_distribution: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index(
            "ix_analytics_repo_date", "repository_id", "snapshot_date"
        ),
    )

    # Relationships
    repository: Mapped[Repository] = relationship(
        "Repository", back_populates="analytics"
    )


class Prediction(Base):
    """Inference audit records for ML classification and resolution advisory predictions."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(
        PK_BIGINT, primary_key=True, autoincrement=True
    )
    issue_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        ForeignKey("issues.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    repo_name: Mapped[Optional[str]] = mapped_column(
        String(120), nullable=True
    )
    input_title: Mapped[str] = mapped_column(Text, nullable=False)
    input_body: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    model_name: Mapped[str] = mapped_column(String(80), nullable=False)
    task_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="classification, related_issues, or resolution_prediction",
    )
    predicted_label: Mapped[Optional[str]] = mapped_column(
        String(80), nullable=True
    )
    confidence_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    probabilities: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON, nullable=True
    )
    inference_time_ms: Mapped[float] = mapped_column(
        Float, default=0.0, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    issue: Mapped[Optional[Issue]] = relationship(
        "Issue", back_populates="predictions"
    )


class RelatedIssuesCache(Base):
    """Precomputed semantic similarity scores and top-k rankings between issue pairs."""

    __tablename__ = "related_issues_cache"

    id: Mapped[int] = mapped_column(
        PK_BIGINT, primary_key=True, autoincrement=True
    )
    source_issue_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("issues.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    target_issue_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("issues.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    similarity_score: Mapped[float] = mapped_column(Float, nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    model_version: Mapped[str] = mapped_column(
        String(60), default="all-MiniLM-L6-v2", nullable=False
    )
    calculated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_related_source_rank", "source_issue_id", "rank"),
        UniqueConstraint(
            "source_issue_id",
            "target_issue_id",
            name="uq_related_source_target",
        ),
    )

    # Relationships
    source_issue: Mapped[Issue] = relationship(
        "Issue",
        foreign_keys=[source_issue_id],
        back_populates="source_related",
    )
    target_issue: Mapped[Issue] = relationship(
        "Issue",
        foreign_keys=[target_issue_id],
        back_populates="target_related",
    )
