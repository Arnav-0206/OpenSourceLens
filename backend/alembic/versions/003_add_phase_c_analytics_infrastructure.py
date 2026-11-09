"""Add Phase C analytics infrastructure and lifecycle models.

Revision ID: 003_add_phase_c_analytics_infrastructure
Revises: 002_add_dynamic_tables
Create Date: 2026-11-10 00:10:00.000000

Adds the Phase C analytical foundation schema:
- Additive columns on repositories: analytics_source_revision, last_analyzed_at
- Additive column on commits: committer_date (nullable, historical benchmark rows stay NULL)
- Additive columns on analysis_runs: requested_source_revision, analytics_source_revision,
  requested_methodology_version, methodology_version, analysis_reference_time,
  source_content_digest, worker_id, leased_until, lease_version, retry_count,
  max_retries, metrics_payload, coverage_summary, confidence_summary, rule_interpretations
- Partial unique index uq_analysis_runs_active_repo on analysis_runs(repository_id) WHERE status IN ('pending', 'running')
- analysis_run_attempts table with CHECK constraint on outcome and cascade delete on analysis_runs.id
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "003_add_phase_c_analytics_infrastructure"
down_revision: Union[str, None] = "002_add_dynamic_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

PK_TYPE = sa.BigInteger().with_variant(sa.Integer(), "sqlite")


def upgrade() -> None:
    # 1. Additive columns on repositories
    with op.batch_alter_table("repositories") as batch_op:
        batch_op.add_column(
            sa.Column(
                "analytics_source_revision",
                sa.Integer(),
                nullable=False,
                server_default="1",
            )
        )
        batch_op.add_column(
            sa.Column(
                "last_analyzed_at",
                sa.DateTime(timezone=True),
                nullable=True,
            )
        )

    # 2. Additive column on commits
    with op.batch_alter_table("commits") as batch_op:
        batch_op.add_column(
            sa.Column(
                "committer_date",
                sa.DateTime(timezone=True),
                nullable=True,
            )
        )

    # 3. Additive columns on analysis_runs
    with op.batch_alter_table("analysis_runs") as batch_op:
        batch_op.add_column(
            sa.Column(
                "requested_source_revision",
                sa.Integer(),
                nullable=False,
                server_default="1",
            )
        )
        batch_op.add_column(
            sa.Column(
                "analytics_source_revision",
                sa.Integer(),
                nullable=True,
            )
        )
        batch_op.add_column(
            sa.Column(
                "requested_methodology_version",
                sa.String(length=50),
                nullable=True,
            )
        )
        batch_op.add_column(
            sa.Column(
                "methodology_version",
                sa.String(length=50),
                nullable=True,
            )
        )
        batch_op.add_column(
            sa.Column(
                "analysis_reference_time",
                sa.DateTime(timezone=True),
                nullable=True,
            )
        )
        batch_op.add_column(
            sa.Column(
                "source_content_digest",
                sa.String(length=64),
                nullable=True,
            )
        )
        batch_op.add_column(
            sa.Column(
                "worker_id",
                sa.String(length=100),
                nullable=True,
            )
        )
        batch_op.add_column(
            sa.Column(
                "leased_until",
                sa.DateTime(timezone=True),
                nullable=True,
            )
        )
        batch_op.add_column(
            sa.Column(
                "lease_version",
                sa.Integer(),
                nullable=False,
                server_default="0",
            )
        )
        batch_op.add_column(
            sa.Column(
                "retry_count",
                sa.Integer(),
                nullable=False,
                server_default="0",
            )
        )
        batch_op.add_column(
            sa.Column(
                "max_retries",
                sa.Integer(),
                nullable=False,
                server_default="3",
            )
        )
        batch_op.add_column(
            sa.Column(
                "metrics_payload",
                sa.JSON(),
                nullable=True,
            )
        )
        batch_op.add_column(
            sa.Column(
                "coverage_summary",
                sa.JSON(),
                nullable=True,
            )
        )
        batch_op.add_column(
            sa.Column(
                "confidence_summary",
                sa.JSON(),
                nullable=True,
            )
        )
        batch_op.add_column(
            sa.Column(
                "rule_interpretations",
                sa.JSON(),
                nullable=True,
            )
        )

    # 4. Partial unique index on analysis_runs
    op.create_index(
        "uq_analysis_runs_active_repo",
        "analysis_runs",
        ["repository_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('pending', 'running')"),
        sqlite_where=sa.text("status IN ('pending', 'running')"),
    )

    # 5. analysis_run_attempts table
    op.create_table(
        "analysis_run_attempts",
        sa.Column("id", PK_TYPE, autoincrement=True, nullable=False),
        sa.Column("analysis_run_id", sa.BigInteger(), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("worker_id", sa.String(length=100), nullable=False),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "outcome",
            sa.String(length=30),
            nullable=False,
            server_default="running",
        ),
        sa.Column("error_category", sa.String(length=50), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("interruption_reason", sa.String(length=100), nullable=True),
        sa.CheckConstraint(
            "outcome IN ('running', 'completed', 'failed', 'lease_expired', 'interrupted')",
            name="ck_attempt_outcome",
        ),
        sa.ForeignKeyConstraint(
            ["analysis_run_id"],
            ["analysis_runs.id"],
            name="fk_analysis_run_attempts_run_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_analysis_run_attempts"),
        sa.UniqueConstraint(
            "analysis_run_id", "attempt_number", name="uq_attempt_run_number"
        ),
    )
    op.create_index(
        "ix_analysis_run_attempts_analysis_run_id",
        "analysis_run_attempts",
        ["analysis_run_id"],
    )
    op.create_index(
        "ix_attempts_run_outcome",
        "analysis_run_attempts",
        ["analysis_run_id", "outcome"],
    )


def downgrade() -> None:
    # 1. Drop analysis_run_attempts
    op.drop_index(
        "ix_attempts_run_outcome", table_name="analysis_run_attempts"
    )
    op.drop_index(
        "ix_analysis_run_attempts_analysis_run_id",
        table_name="analysis_run_attempts",
    )
    op.drop_table("analysis_run_attempts")

    # 2. Drop partial unique index on analysis_runs
    op.drop_index(
        "uq_analysis_runs_active_repo", table_name="analysis_runs"
    )

    # 3. Drop columns on analysis_runs
    with op.batch_alter_table("analysis_runs") as batch_op:
        batch_op.drop_column("rule_interpretations")
        batch_op.drop_column("confidence_summary")
        batch_op.drop_column("coverage_summary")
        batch_op.drop_column("metrics_payload")
        batch_op.drop_column("max_retries")
        batch_op.drop_column("retry_count")
        batch_op.drop_column("lease_version")
        batch_op.drop_column("leased_until")
        batch_op.drop_column("worker_id")
        batch_op.drop_column("source_content_digest")
        batch_op.drop_column("analysis_reference_time")
        batch_op.drop_column("methodology_version")
        batch_op.drop_column("requested_methodology_version")
        batch_op.drop_column("analytics_source_revision")
        batch_op.drop_column("requested_source_revision")

    # 4. Drop committer_date on commits
    with op.batch_alter_table("commits") as batch_op:
        batch_op.drop_column("committer_date")

    # 5. Drop columns on repositories
    with op.batch_alter_table("repositories") as batch_op:
        batch_op.drop_column("last_analyzed_at")
        batch_op.drop_column("analytics_source_revision")
