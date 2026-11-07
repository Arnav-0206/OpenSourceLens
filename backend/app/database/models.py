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
    CheckConstraint,
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
    text,
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

    # Phase B: Additive columns for universal dynamic ingestion
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True, server_default="1", nullable=False
    )
    is_archived: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="0", nullable=False
    )
    is_fork: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="0", nullable=False
    )
    upstream_repo: Mapped[Optional[str]] = mapped_column(
        String(120), nullable=True
    )
    last_synced_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
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
    collection_runs: Mapped[List[CollectionRun]] = relationship(
        "CollectionRun",
        back_populates="repository",
        cascade="all, delete-orphan",
    )
    analysis_runs: Mapped[List[AnalysisRun]] = relationship(
        "AnalysisRun",
        back_populates="repository",
        cascade="all, delete-orphan",
    )
    commits: Mapped[List[Commit]] = relationship(
        "Commit",
        back_populates="repository",
        cascade="all, delete-orphan",
    )
    contributors: Mapped[List[Contributor]] = relationship(
        "Contributor",
        back_populates="repository",
        cascade="all, delete-orphan",
    )
    pull_requests: Mapped[List[PullRequest]] = relationship(
        "PullRequest",
        back_populates="repository",
        cascade="all, delete-orphan",
    )
    releases: Mapped[List[Release]] = relationship(
        "Release",
        back_populates="repository",
        cascade="all, delete-orphan",
    )
    issue_comments: Mapped[List[IssueComment]] = relationship(
        "IssueComment",
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
    issue_comments: Mapped[List[IssueComment]] = relationship(
        "IssueComment",
        back_populates="issue",
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


# ---------------------------------------------------------------------------
# Phase B: Universal Multi-Entity Models & Dynamic Ingestion Infrastructure
# ---------------------------------------------------------------------------


class CollectionRun(Base):
    """Audit log and distributed job-lease state for repository data ingestion runs."""

    __tablename__ = "collection_runs"

    id: Mapped[int] = mapped_column(
        PK_BIGINT, primary_key=True, autoincrement=True
    )
    repository_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("repositories.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    mode: Mapped[str] = mapped_column(
        String(32), default="standard_diagnostic", nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(32), default="pending", index=True, nullable=False
    )
    worker_id: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    leased_until: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    lease_version: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    max_retries: Mapped[int] = mapped_column(Integer, default=2, nullable=False)
    is_partial: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    coverage_summary: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON, nullable=True
    )
    items_collected: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON, nullable=True
    )
    rate_limit_consumed: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_collection_runs_repo_started", "repository_id", "started_at"),
    )

    # Relationships
    repository: Mapped[Repository] = relationship(
        "Repository", back_populates="collection_runs"
    )
    analysis_runs: Mapped[List[AnalysisRun]] = relationship(
        "AnalysisRun", back_populates="collection_run"
    )


class AnalysisRun(Base):
    """Audit log of multi-domain statistical, NLP, and analytical engine evaluations."""

    __tablename__ = "analysis_runs"

    id: Mapped[int] = mapped_column(
        PK_BIGINT, primary_key=True, autoincrement=True
    )
    repository_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("repositories.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    collection_run_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        ForeignKey("collection_runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(
        String(32), default="pending", nullable=False
    )
    execution_time_ms: Mapped[float] = mapped_column(
        Float, default=0.0, nullable=False
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    repository: Mapped[Repository] = relationship(
        "Repository", back_populates="analysis_runs"
    )
    collection_run: Mapped[Optional[CollectionRun]] = relationship(
        "CollectionRun", back_populates="analysis_runs"
    )


class Commit(Base):
    """Represents an individual Git commit harvested from the repository."""

    __tablename__ = "commits"

    id: Mapped[int] = mapped_column(
        PK_BIGINT, primary_key=True, autoincrement=True
    )
    repository_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("repositories.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    sha: Mapped[str] = mapped_column(String(40), nullable=False)
    author_name: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    author_email: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    author_login: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    author_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    author_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    additions: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    deletions: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_changes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "repository_id", "sha", name="uq_commit_repo_sha"
        ),
        Index("ix_commits_repo_date", "repository_id", "author_date"),
    )

    # Relationships
    repository: Mapped[Repository] = relationship(
        "Repository", back_populates="commits"
    )


class Contributor(Base):
    """Represents a contributor across commits, pull requests, reviews, or issues."""

    __tablename__ = "contributors"

    id: Mapped[int] = mapped_column(
        PK_BIGINT, primary_key=True, autoincrement=True
    )
    repository_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("repositories.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    login: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    github_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    git_email: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    avatar_url: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    primary_role: Mapped[str] = mapped_column(
        String(40), default="contributor", nullable=False
    )
    total_commits: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_prs_authored: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    total_prs_reviewed: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    total_issues_opened: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    first_contribution_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_contribution_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    tenure_days: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    is_newcomer: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    is_core: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_bot: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_unlinked: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    identity_confidence: Mapped[float] = mapped_column(
        Float, default=1.0, nullable=False
    )

    __table_args__ = (
        Index(
            "uq_contributors_repo_github_id",
            "repository_id",
            "github_id",
            unique=True,
            sqlite_where=text("github_id IS NOT NULL"),
            postgresql_where=text("github_id IS NOT NULL"),
        ),
        Index(
            "uq_contributors_repo_unlinked_email",
            "repository_id",
            "git_email",
            unique=True,
            sqlite_where=text("github_id IS NULL AND git_email IS NOT NULL"),
            postgresql_where=text("github_id IS NULL AND git_email IS NOT NULL"),
        ),
        Index("ix_contributors_repo_commits", "repository_id", "total_commits"),
    )

    # Relationships
    repository: Mapped[Repository] = relationship(
        "Repository", back_populates="contributors"
    )
    activities: Mapped[List[ContributorActivity]] = relationship(
        "ContributorActivity",
        back_populates="contributor",
        cascade="all, delete-orphan",
    )


class ContributorActivity(Base):
    """Normalized timestamped activity log for newcomer retention and participation modeling."""

    __tablename__ = "contributor_activities"

    id: Mapped[int] = mapped_column(
        PK_BIGINT, primary_key=True, autoincrement=True
    )
    contributor_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("contributors.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    repository_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("repositories.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    activity_type: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # commit, pr_opened, pr_merged, review_submitted, issue_opened, comment_created
    activity_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    entity_id: Mapped[str] = mapped_column(
        String(64), nullable=False
    )  # commit sha, pr number, review id, issue number, comment id
    details: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON, nullable=True
    )

    __table_args__ = (
        UniqueConstraint(
            "repository_id",
            "activity_type",
            "entity_id",
            name="uq_activity_natural_key",
        ),
        Index("ix_contrib_act_date", "contributor_id", "activity_date"),
        Index(
            "ix_repo_act_type_date",
            "repository_id",
            "activity_type",
            "activity_date",
        ),
    )

    # Relationships
    contributor: Mapped[Contributor] = relationship(
        "Contributor", back_populates="activities"
    )


class PullRequest(Base):
    """Represents a GitHub pull request with lifecycle turnaround and review dynamics."""

    __tablename__ = "pull_requests"

    id: Mapped[int] = mapped_column(
        PK_BIGINT, primary_key=True, autoincrement=True
    )
    repository_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("repositories.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    pr_number: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    state: Mapped[str] = mapped_column(String(20), nullable=False)  # open, closed
    author: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    author_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    closed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    merged_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    time_to_merge_hours: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    time_to_first_review_hours: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    comments_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    review_comments_count: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    additions: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    deletions: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    changed_files: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )

    __table_args__ = (
        UniqueConstraint(
            "repository_id", "pr_number", name="uq_pr_repo_number"
        ),
        Index(
            "ix_pr_repo_state_merged",
            "repository_id",
            "state",
            "merged_at",
        ),
    )

    # Relationships
    repository: Mapped[Repository] = relationship(
        "Repository", back_populates="pull_requests"
    )
    reviews: Mapped[List[PullRequestReview]] = relationship(
        "PullRequestReview",
        back_populates="pull_request",
        cascade="all, delete-orphan",
    )
    review_comments: Mapped[List[PullRequestReviewComment]] = relationship(
        "PullRequestReviewComment",
        back_populates="pull_request",
        cascade="all, delete-orphan",
    )
    issue_comments: Mapped[List[IssueComment]] = relationship(
        "IssueComment", back_populates="pull_request"
    )


class PullRequestReview(Base):
    """Represents a formal code review decision envelope submitted on a pull request."""

    __tablename__ = "pull_request_reviews"

    id: Mapped[int] = mapped_column(
        PK_BIGINT, primary_key=True, autoincrement=True
    )
    pull_request_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("pull_requests.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    review_id: Mapped[int] = mapped_column(
        BigInteger, unique=True, index=True, nullable=False
    )
    author: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    author_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    state: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # APPROVED, CHANGES_REQUESTED, COMMENTED, DISMISSED
    submitted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    body: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_pr_reviews_pr_submitted", "pull_request_id", "submitted_at"),
    )

    # Relationships
    pull_request: Mapped[PullRequest] = relationship(
        "PullRequest", back_populates="reviews"
    )
    review_comments: Mapped[List[PullRequestReviewComment]] = relationship(
        "PullRequestReviewComment", back_populates="review"
    )


class PullRequestReviewComment(Base):
    """Represents an inline diff comment tied to specific code lines in a pull request."""

    __tablename__ = "pull_request_review_comments"

    id: Mapped[int] = mapped_column(
        PK_BIGINT, primary_key=True, autoincrement=True
    )
    pull_request_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("pull_requests.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    review_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        ForeignKey("pull_request_reviews.id", ondelete="SET NULL"),
        nullable=True,
    )
    comment_id: Mapped[int] = mapped_column(
        BigInteger, unique=True, index=True, nullable=False
    )
    author: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    author_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    line: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    commit_id: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    body: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    pull_request: Mapped[PullRequest] = relationship(
        "PullRequest", back_populates="review_comments"
    )
    review: Mapped[Optional[PullRequestReview]] = relationship(
        "PullRequestReview", back_populates="review_comments"
    )


class Release(Base):
    """Represents a software release tag and milestone publication cadence."""

    __tablename__ = "releases"

    id: Mapped[int] = mapped_column(
        PK_BIGINT, primary_key=True, autoincrement=True
    )
    repository_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("repositories.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    release_id: Mapped[int] = mapped_column(
        BigInteger, unique=True, index=True, nullable=False
    )
    tag_name: Mapped[str] = mapped_column(String(100), nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    published_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    is_prerelease: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    body: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_releases_repo_published", "repository_id", "published_at"),
    )

    # Relationships
    repository: Mapped[Repository] = relationship(
        "Repository", back_populates="releases"
    )


class IssueComment(Base):
    """Represents general timeline discussion comments on issues and pull requests."""

    __tablename__ = "issue_comments"

    id: Mapped[int] = mapped_column(
        PK_BIGINT, primary_key=True, autoincrement=True
    )
    repository_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("repositories.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    comment_id: Mapped[int] = mapped_column(
        BigInteger, unique=True, index=True, nullable=False
    )
    issue_number: Mapped[int] = mapped_column(Integer, nullable=False)
    issue_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        ForeignKey("issues.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    pull_request_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        ForeignKey("pull_requests.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    author: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    author_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    author_association: Mapped[Optional[str]] = mapped_column(
        String(40), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    body: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "NOT (issue_id IS NOT NULL AND pull_request_id IS NOT NULL)",
            name="chk_issue_comment_parent",
        ),
        Index("ix_issue_comments_repo_num", "repository_id", "issue_number"),
        Index("ix_issue_comments_created", "created_at"),
    )

    # Relationships
    repository: Mapped[Repository] = relationship(
        "Repository", back_populates="issue_comments"
    )
    issue: Mapped[Optional[Issue]] = relationship(
        "Issue", back_populates="issue_comments"
    )
    pull_request: Mapped[Optional[PullRequest]] = relationship(
        "PullRequest", back_populates="issue_comments"
    )

