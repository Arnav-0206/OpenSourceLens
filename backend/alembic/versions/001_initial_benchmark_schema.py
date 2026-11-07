"""Initial benchmark schema baseline.

Revision ID: 001_initial_benchmark_schema
Revises: 
Create Date: 2026-11-08 01:55:00.000000

Establishes the initial Phase 11 baseline schema containing:
- repositories
- issues
- issue_labels
- repository_analytics
- predictions
- related_issues_cache
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "001_initial_benchmark_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

PK_TYPE = sa.BigInteger().with_variant(sa.Integer(), "sqlite")


def upgrade() -> None:
    # 1. repositories table
    op.create_table(
        "repositories",
        sa.Column("id", sa.BigInteger(), autoincrement=False, nullable=False),
        sa.Column("repo_name", sa.String(length=120), nullable=False),
        sa.Column("owner", sa.String(length=80), nullable=False),
        sa.Column("name", sa.String(length=80), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("primary_language", sa.String(length=60), nullable=True),
        sa.Column("stars_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("forks_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("open_issues_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("github_created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("github_updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("collected_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_repositories"),
    )
    op.create_index("ix_repositories_repo_name", "repositories", ["repo_name"], unique=True)

    # 2. issues table
    op.create_table(
        "issues",
        sa.Column("id", sa.BigInteger(), autoincrement=False, nullable=False),
        sa.Column("repository_id", sa.BigInteger(), nullable=False),
        sa.Column("repo_name", sa.String(length=120), nullable=False),
        sa.Column("issue_number", sa.Integer(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("body", sa.Text(), nullable=True),
        sa.Column("clean_title", sa.Text(), nullable=True),
        sa.Column("clean_body", sa.Text(), nullable=True),
        sa.Column("combined_text", sa.Text(), nullable=True),
        sa.Column("clean_title_word_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("clean_body_word_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("state", sa.String(length=30), nullable=False),
        sa.Column("state_reason", sa.String(length=50), nullable=True),
        sa.Column("author", sa.String(length=100), nullable=False),
        sa.Column("author_type", sa.String(length=40), nullable=False, server_default="User"),
        sa.Column("author_association", sa.String(length=40), nullable=False, server_default="NONE"),
        sa.Column("is_internal_author", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("is_bot_author", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("comments_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("reactions_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("labels_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("has_code_block", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("code_block_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("has_url", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("has_stack_trace", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("canonical_label", sa.String(length=50), nullable=False),
        sa.Column("canonical_label_id", sa.Integer(), nullable=False),
        sa.Column("is_classified_category", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("is_unlabeled", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("resolution_time_hours", sa.Float(), nullable=True),
        sa.Column("resolution_time_days", sa.Float(), nullable=True),
        sa.Column("speed_tier", sa.String(length=30), nullable=True),
        sa.Column("resolution_type", sa.String(length=30), nullable=True),
        sa.Column("is_censored", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("is_immediate_closure", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("duration_at_risk_days", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(["repository_id"], ["repositories.id"], name="fk_issues_repository_id", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_issues"),
        sa.UniqueConstraint("repository_id", "issue_number", name="uq_issues_repo_issue_number"),
    )
    op.create_index("ix_issues_canonical_label", "issues", ["canonical_label"])
    op.create_index("ix_issues_created_at", "issues", ["created_at"])
    op.create_index("ix_issues_repo_canonical", "issues", ["repo_name", "canonical_label"])
    op.create_index("ix_issues_repo_issue_number", "issues", ["repo_name", "issue_number"])
    op.create_index("ix_issues_repo_speed_tier", "issues", ["repo_name", "speed_tier"])
    op.create_index("ix_issues_repo_state", "issues", ["repo_name", "state"])
    op.create_index("ix_issues_repository_id", "issues", ["repository_id"])
    op.create_index("ix_issues_speed_tier", "issues", ["speed_tier"])
    op.create_index("ix_issues_state", "issues", ["state"])

    # 3. issue_labels table
    op.create_table(
        "issue_labels",
        sa.Column("id", PK_TYPE, autoincrement=True, nullable=False),
        sa.Column("issue_id", sa.BigInteger(), nullable=False),
        sa.Column("raw_label", sa.String(length=120), nullable=False),
        sa.Column("normalized_label", sa.String(length=120), nullable=False),
        sa.Column("color", sa.String(length=10), nullable=True),
        sa.Column("description", sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(["issue_id"], ["issues.id"], name="fk_issue_labels_issue_id", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_issue_labels"),
    )
    op.create_index("ix_issue_labels_issue_id", "issue_labels", ["issue_id"])
    op.create_index("ix_issue_labels_normalized_label", "issue_labels", ["normalized_label"])
    op.create_index("ix_issue_labels_raw_label", "issue_labels", ["raw_label"])

    # 4. repository_analytics table
    op.create_table(
        "repository_analytics",
        sa.Column("id", PK_TYPE, autoincrement=True, nullable=False),
        sa.Column("repository_id", sa.BigInteger(), nullable=False),
        sa.Column("repo_name", sa.String(length=120), nullable=False),
        sa.Column("snapshot_date", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("total_issues", sa.Integer(), nullable=False),
        sa.Column("closed_issues", sa.Integer(), nullable=False),
        sa.Column("open_issues", sa.Integer(), nullable=False),
        sa.Column("closure_rate", sa.Float(), nullable=False),
        sa.Column("category_bug_count", sa.Integer(), nullable=False),
        sa.Column("category_feature_count", sa.Integer(), nullable=False),
        sa.Column("category_other_count", sa.Integer(), nullable=False),
        sa.Column("median_resolution_hours", sa.Float(), nullable=True),
        sa.Column("mean_resolution_hours", sa.Float(), nullable=True),
        sa.Column("p75_resolution_hours", sa.Float(), nullable=True),
        sa.Column("p90_resolution_hours", sa.Float(), nullable=True),
        sa.Column("iqr_resolution_hours", sa.Float(), nullable=True),
        sa.Column("fast_tier_count", sa.Integer(), nullable=False),
        sa.Column("medium_tier_count", sa.Integer(), nullable=False),
        sa.Column("slow_tier_count", sa.Integer(), nullable=False),
        sa.Column("unlabeled_count", sa.Integer(), nullable=False),
        sa.Column("unlabeled_ratio", sa.Float(), nullable=False),
        sa.Column("internal_author_count", sa.Integer(), nullable=False),
        sa.Column("external_author_count", sa.Integer(), nullable=False),
        sa.Column("bot_author_count", sa.Integer(), nullable=False),
        sa.Column("internal_median_hours", sa.Float(), nullable=True),
        sa.Column("external_median_hours", sa.Float(), nullable=True),
        sa.Column("avg_comments_per_issue", sa.Float(), nullable=False),
        sa.Column("avg_body_words", sa.Float(), nullable=False),
        sa.Column("avg_title_words", sa.Float(), nullable=False),
        sa.Column("code_block_ratio", sa.Float(), nullable=False),
        sa.Column("has_url_ratio", sa.Float(), nullable=False),
        sa.Column("survival_probabilities", sa.JSON(), nullable=True),
        sa.Column("monthly_inflow_outflow", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["repository_id"], ["repositories.id"], name="fk_repo_analytics_repository_id", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_repository_analytics"),
        sa.UniqueConstraint("repository_id", "snapshot_date", name="uq_repo_analytics_snapshot"),
    )
    op.create_index("ix_repository_analytics_repo_name", "repository_analytics", ["repo_name"])
    op.create_index("ix_repository_analytics_repository_id", "repository_analytics", ["repository_id"])
    op.create_index("ix_repository_analytics_snapshot_date", "repository_analytics", ["snapshot_date"])

    # 5. predictions table
    op.create_table(
        "predictions",
        sa.Column("id", PK_TYPE, autoincrement=True, nullable=False),
        sa.Column("issue_id", sa.BigInteger(), nullable=True),
        sa.Column("repo_name", sa.String(length=120), nullable=True),
        sa.Column("model_name", sa.String(length=80), nullable=False),
        sa.Column("model_version", sa.String(length=40), nullable=False),
        sa.Column("task_type", sa.String(length=50), nullable=False),
        sa.Column("predicted_label", sa.String(length=80), nullable=True),
        sa.Column("confidence_score", sa.Float(), nullable=True),
        sa.Column("probabilities", sa.JSON(), nullable=True),
        sa.Column("inference_time_ms", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["issue_id"], ["issues.id"], name="fk_predictions_issue_id", ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id", name="pk_predictions"),
    )
    op.create_index("ix_predictions_created_at", "predictions", ["created_at"])
    op.create_index("ix_predictions_issue_id", "predictions", ["issue_id"])
    op.create_index("ix_predictions_model_name", "predictions", ["model_name"])
    op.create_index("ix_predictions_repo_name", "predictions", ["repo_name"])
    op.create_index("ix_predictions_task_type", "predictions", ["task_type"])

    # 6. related_issues_cache table
    op.create_table(
        "related_issues_cache",
        sa.Column("id", PK_TYPE, autoincrement=True, nullable=False),
        sa.Column("source_issue_id", sa.BigInteger(), nullable=False),
        sa.Column("target_issue_id", sa.BigInteger(), nullable=False),
        sa.Column("similarity_score", sa.Float(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("model_version", sa.String(length=60), nullable=False, server_default="all-MiniLM-L6-v2"),
        sa.Column("calculated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["source_issue_id"], ["issues.id"], name="fk_related_source_issue_id", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["target_issue_id"], ["issues.id"], name="fk_related_target_issue_id", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_related_issues_cache"),
        sa.UniqueConstraint("source_issue_id", "target_issue_id", name="uq_related_source_target"),
    )
    op.create_index("ix_related_issues_cache_source_issue_id", "related_issues_cache", ["source_issue_id"])
    op.create_index("ix_related_issues_cache_target_issue_id", "related_issues_cache", ["target_issue_id"])
    op.create_index("ix_related_source_rank", "related_issues_cache", ["source_issue_id", "rank"])


def downgrade() -> None:
    op.drop_table("related_issues_cache")
    op.drop_table("predictions")
    op.drop_table("repository_analytics")
    op.drop_table("issue_labels")
    op.drop_table("issues")
    op.drop_table("repositories")
