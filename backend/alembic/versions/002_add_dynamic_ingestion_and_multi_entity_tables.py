"""Add dynamic ingestion and multi-entity tables.

Revision ID: 002_add_dynamic_tables
Revises: 001_initial_benchmark_schema
Create Date: 2026-11-08 01:56:00.000000

Adds the universal multi-entity data model:
- Additive columns on repositories (is_active, is_archived, is_fork, upstream_repo, last_synced_at)
- collection_runs
- analysis_runs
- commits
- contributors (with partial unique indexes)
- contributor_activities
- pull_requests
- pull_request_reviews
- pull_request_review_comments
- releases
- issue_comments (with mutual exclusion CHECK constraint)
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "002_add_dynamic_tables"
down_revision: Union[str, None] = "001_initial_benchmark_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

PK_TYPE = sa.BigInteger().with_variant(sa.Integer(), "sqlite")


def upgrade() -> None:
    # 1. Additive columns on repositories
    with op.batch_alter_table("repositories") as batch_op:
        batch_op.add_column(sa.Column("is_active", sa.Boolean(), nullable=False, server_default="1"))
        batch_op.add_column(sa.Column("is_archived", sa.Boolean(), nullable=False, server_default="0"))
        batch_op.add_column(sa.Column("is_fork", sa.Boolean(), nullable=False, server_default="0"))
        batch_op.add_column(sa.Column("upstream_repo", sa.String(length=120), nullable=True))
        batch_op.add_column(sa.Column("last_synced_at", sa.DateTime(timezone=True), nullable=True))

    # 2. collection_runs table
    op.create_table(
        "collection_runs",
        sa.Column("id", PK_TYPE, autoincrement=True, nullable=False),
        sa.Column("repository_id", sa.BigInteger(), nullable=False),
        sa.Column("mode", sa.String(length=32), nullable=False, server_default="standard_diagnostic"),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("worker_id", sa.String(length=80), nullable=True),
        sa.Column("leased_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lease_version", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_retries", sa.Integer(), nullable=False, server_default="2"),
        sa.Column("is_partial", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("coverage_summary", sa.JSON(), nullable=True),
        sa.Column("items_collected", sa.JSON(), nullable=True),
        sa.Column("rate_limit_consumed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["repository_id"], ["repositories.id"], name="fk_collection_runs_repo_id", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_collection_runs"),
    )
    op.create_index("ix_collection_runs_leased_until", "collection_runs", ["leased_until"])
    op.create_index("ix_collection_runs_repo_started", "collection_runs", ["repository_id", "started_at"])
    op.create_index("ix_collection_runs_repository_id", "collection_runs", ["repository_id"])
    op.create_index("ix_collection_runs_status", "collection_runs", ["status"])

    # 3. analysis_runs table
    op.create_table(
        "analysis_runs",
        sa.Column("id", PK_TYPE, autoincrement=True, nullable=False),
        sa.Column("repository_id", sa.BigInteger(), nullable=False),
        sa.Column("collection_run_id", sa.BigInteger(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("execution_time_ms", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["collection_run_id"], ["collection_runs.id"], name="fk_analysis_runs_collection_run_id", ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["repository_id"], ["repositories.id"], name="fk_analysis_runs_repo_id", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_analysis_runs"),
    )
    op.create_index("ix_analysis_runs_repository_id", "analysis_runs", ["repository_id"])

    # 4. commits table
    op.create_table(
        "commits",
        sa.Column("id", PK_TYPE, autoincrement=True, nullable=False),
        sa.Column("repository_id", sa.BigInteger(), nullable=False),
        sa.Column("sha", sa.String(length=40), nullable=False),
        sa.Column("author_name", sa.String(length=120), nullable=True),
        sa.Column("author_email", sa.String(length=120), nullable=True),
        sa.Column("author_login", sa.String(length=80), nullable=True),
        sa.Column("author_id", sa.BigInteger(), nullable=True),
        sa.Column("author_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("additions", sa.Integer(), nullable=True),
        sa.Column("deletions", sa.Integer(), nullable=True),
        sa.Column("total_changes", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["repository_id"], ["repositories.id"], name="fk_commits_repository_id", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_commits"),
        sa.UniqueConstraint("repository_id", "sha", name="uq_commit_repo_sha"),
    )
    op.create_index("ix_commits_repo_date", "commits", ["repository_id", "author_date"])
    op.create_index("ix_commits_repository_id", "commits", ["repository_id"])

    # 5. contributors table
    op.create_table(
        "contributors",
        sa.Column("id", PK_TYPE, autoincrement=True, nullable=False),
        sa.Column("repository_id", sa.BigInteger(), nullable=False),
        sa.Column("login", sa.String(length=80), nullable=True),
        sa.Column("github_id", sa.BigInteger(), nullable=True),
        sa.Column("git_email", sa.String(length=120), nullable=True),
        sa.Column("avatar_url", sa.String(length=255), nullable=True),
        sa.Column("primary_role", sa.String(length=40), nullable=False, server_default="contributor"),
        sa.Column("total_commits", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_prs_authored", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_prs_reviewed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_issues_opened", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("first_contribution_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_contribution_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("tenure_days", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("is_newcomer", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("is_core", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("is_bot", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("is_unlinked", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("identity_confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.ForeignKeyConstraint(["repository_id"], ["repositories.id"], name="fk_contributors_repository_id", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_contributors"),
    )
    op.create_index("ix_contributors_repo_commits", "contributors", ["repository_id", "total_commits"])
    op.create_index("ix_contributors_repository_id", "contributors", ["repository_id"])
    op.create_index(
        "uq_contributors_repo_github_id",
        "contributors",
        ["repository_id", "github_id"],
        unique=True,
        sqlite_where=sa.text("github_id IS NOT NULL"),
        postgresql_where=sa.text("github_id IS NOT NULL"),
    )
    op.create_index(
        "uq_contributors_repo_unlinked_email",
        "contributors",
        ["repository_id", "git_email"],
        unique=True,
        sqlite_where=sa.text("github_id IS NULL AND git_email IS NOT NULL"),
        postgresql_where=sa.text("github_id IS NULL AND git_email IS NOT NULL"),
    )

    # 6. contributor_activities table
    op.create_table(
        "contributor_activities",
        sa.Column("id", PK_TYPE, autoincrement=True, nullable=False),
        sa.Column("contributor_id", sa.BigInteger(), nullable=False),
        sa.Column("repository_id", sa.BigInteger(), nullable=False),
        sa.Column("activity_type", sa.String(length=32), nullable=False),
        sa.Column("activity_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("entity_id", sa.String(length=64), nullable=False),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["contributor_id"], ["contributors.id"], name="fk_contrib_act_contributor_id", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["repository_id"], ["repositories.id"], name="fk_contrib_act_repository_id", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_contributor_activities"),
        sa.UniqueConstraint("repository_id", "activity_type", "entity_id", name="uq_activity_natural_key"),
    )
    op.create_index("ix_contrib_act_date", "contributor_activities", ["contributor_id", "activity_date"])
    op.create_index("ix_repo_act_type_date", "contributor_activities", ["repository_id", "activity_type", "activity_date"])

    # 7. pull_requests table
    op.create_table(
        "pull_requests",
        sa.Column("id", PK_TYPE, autoincrement=True, nullable=False),
        sa.Column("repository_id", sa.BigInteger(), nullable=False),
        sa.Column("pr_number", sa.Integer(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("state", sa.String(length=20), nullable=False),
        sa.Column("author", sa.String(length=80), nullable=True),
        sa.Column("author_id", sa.BigInteger(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("merged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("time_to_merge_hours", sa.Float(), nullable=True),
        sa.Column("time_to_first_review_hours", sa.Float(), nullable=True),
        sa.Column("comments_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("review_comments_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("additions", sa.Integer(), nullable=True),
        sa.Column("deletions", sa.Integer(), nullable=True),
        sa.Column("changed_files", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["repository_id"], ["repositories.id"], name="fk_pull_requests_repository_id", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_pull_requests"),
        sa.UniqueConstraint("repository_id", "pr_number", name="uq_pr_repo_number"),
    )
    op.create_index("ix_pr_repo_state_merged", "pull_requests", ["repository_id", "state", "merged_at"])
    op.create_index("ix_pull_requests_repository_id", "pull_requests", ["repository_id"])

    # 8. pull_request_reviews table
    op.create_table(
        "pull_request_reviews",
        sa.Column("id", PK_TYPE, autoincrement=True, nullable=False),
        sa.Column("pull_request_id", sa.BigInteger(), nullable=False),
        sa.Column("review_id", sa.BigInteger(), nullable=False),
        sa.Column("author", sa.String(length=80), nullable=True),
        sa.Column("author_id", sa.BigInteger(), nullable=True),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("body", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["pull_request_id"], ["pull_requests.id"], name="fk_pr_reviews_pull_request_id", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_pull_request_reviews"),
        sa.UniqueConstraint("review_id", name="uq_pr_reviews_review_id"),
    )
    op.create_index("ix_pr_reviews_pr_submitted", "pull_request_reviews", ["pull_request_id", "submitted_at"])
    op.create_index("ix_pull_request_reviews_pull_request_id", "pull_request_reviews", ["pull_request_id"])
    op.create_index("ix_pull_request_reviews_review_id", "pull_request_reviews", ["review_id"])

    # 9. pull_request_review_comments table
    op.create_table(
        "pull_request_review_comments",
        sa.Column("id", PK_TYPE, autoincrement=True, nullable=False),
        sa.Column("pull_request_id", sa.BigInteger(), nullable=False),
        sa.Column("review_id", sa.BigInteger(), nullable=True),
        sa.Column("comment_id", sa.BigInteger(), nullable=False),
        sa.Column("author", sa.String(length=80), nullable=True),
        sa.Column("author_id", sa.BigInteger(), nullable=True),
        sa.Column("path", sa.String(length=500), nullable=True),
        sa.Column("line", sa.Integer(), nullable=True),
        sa.Column("commit_id", sa.String(length=40), nullable=True),
        sa.Column("body", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["pull_request_id"], ["pull_requests.id"], name="fk_pr_review_comments_pr_id", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["review_id"], ["pull_request_reviews.id"], name="fk_pr_review_comments_review_id", ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id", name="pk_pull_request_review_comments"),
        sa.UniqueConstraint("comment_id", name="uq_pr_review_comments_comment_id"),
    )
    op.create_index("ix_pr_review_comments_pr_id", "pull_request_review_comments", ["pull_request_id"])
    op.create_index("ix_pr_review_comments_review_id", "pull_request_review_comments", ["review_id"])
    op.create_index("ix_pr_review_comments_comment_id", "pull_request_review_comments", ["comment_id"])

    # 10. releases table
    op.create_table(
        "releases",
        sa.Column("id", PK_TYPE, autoincrement=True, nullable=False),
        sa.Column("repository_id", sa.BigInteger(), nullable=False),
        sa.Column("release_id", sa.BigInteger(), nullable=False),
        sa.Column("tag_name", sa.String(length=100), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_prerelease", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("body", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["repository_id"], ["repositories.id"], name="fk_releases_repository_id", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_releases"),
        sa.UniqueConstraint("release_id", name="uq_releases_release_id"),
    )
    op.create_index("ix_releases_repo_published", "releases", ["repository_id", "published_at"])
    op.create_index("ix_releases_repository_id", "releases", ["repository_id"])
    op.create_index("ix_releases_release_id", "releases", ["release_id"])

    # 11. issue_comments table (with CHECK constraint)
    op.create_table(
        "issue_comments",
        sa.Column("id", PK_TYPE, autoincrement=True, nullable=False),
        sa.Column("repository_id", sa.BigInteger(), nullable=False),
        sa.Column("comment_id", sa.BigInteger(), nullable=False),
        sa.Column("issue_number", sa.Integer(), nullable=False),
        sa.Column("issue_id", sa.BigInteger(), nullable=True),
        sa.Column("pull_request_id", sa.BigInteger(), nullable=True),
        sa.Column("author", sa.String(length=80), nullable=True),
        sa.Column("author_id", sa.BigInteger(), nullable=True),
        sa.Column("author_association", sa.String(length=40), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("body", sa.Text(), nullable=True),
        sa.CheckConstraint("NOT (issue_id IS NOT NULL AND pull_request_id IS NOT NULL)", name="chk_issue_comment_parent"),
        sa.ForeignKeyConstraint(["issue_id"], ["issues.id"], name="fk_issue_comments_issue_id", ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["pull_request_id"], ["pull_requests.id"], name="fk_issue_comments_pr_id", ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["repository_id"], ["repositories.id"], name="fk_issue_comments_repository_id", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_issue_comments"),
        sa.UniqueConstraint("comment_id", name="uq_issue_comments_comment_id"),
    )
    op.create_index("ix_issue_comments_created", "issue_comments", ["created_at"])
    op.create_index("ix_issue_comments_issue_id", "issue_comments", ["issue_id"])
    op.create_index("ix_issue_comments_pull_request_id", "issue_comments", ["pull_request_id"])
    op.create_index("ix_issue_comments_repo_num", "issue_comments", ["repository_id", "issue_number"])
    op.create_index("ix_issue_comments_repository_id", "issue_comments", ["repository_id"])
    op.create_index("ix_issue_comments_comment_id", "issue_comments", ["comment_id"])


def downgrade() -> None:
    op.drop_table("issue_comments")
    op.drop_table("releases")
    op.drop_table("pull_request_review_comments")
    op.drop_table("pull_request_reviews")
    op.drop_table("pull_requests")
    op.drop_table("contributor_activities")
    op.drop_table("contributors")
    op.drop_table("commits")
    op.drop_table("analysis_runs")
    op.drop_table("collection_runs")

    with op.batch_alter_table("repositories") as batch_op:
        batch_op.drop_column("last_synced_at")
        batch_op.drop_column("upstream_repo")
        batch_op.drop_column("is_fork")
        batch_op.drop_column("is_archived")
        batch_op.drop_column("is_active")
