"""Add Discussions, PR depth attributes, and release tag fallback columns.

Revision ID: 004_add_discussions_and_pr_depth
Revises: 003_add_phase_c_analytics_infrastructure
Create Date: 2026-11-16 13:40:00.000000

Adds Stage 3 schema extensions:
- Additive columns on pull_requests: body, head_branch, base_branch, is_draft, labels, requested_reviewers
- Additive columns on releases: is_tag_fallback, commit_sha
- New tables: discussions, discussion_comments
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "004_add_discussions_and_pr_depth"
down_revision: Union[str, None] = "003_add_phase_c_analytics_infrastructure"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

PK_TYPE = sa.BigInteger().with_variant(sa.Integer(), "sqlite")


def upgrade() -> None:
    # 1. Additive columns on pull_requests
    with op.batch_alter_table("pull_requests") as batch_op:
        batch_op.add_column(
            sa.Column("body", sa.Text(), nullable=True)
        )
        batch_op.add_column(
            sa.Column("head_branch", sa.String(120), nullable=True)
        )
        batch_op.add_column(
            sa.Column("base_branch", sa.String(120), nullable=True)
        )
        batch_op.add_column(
            sa.Column(
                "is_draft",
                sa.Boolean(),
                nullable=False,
                server_default="0",
            )
        )
        batch_op.add_column(
            sa.Column(
                "labels",
                sa.JSON(),
                nullable=False,
                server_default="[]",
            )
        )
        batch_op.add_column(
            sa.Column(
                "requested_reviewers",
                sa.JSON(),
                nullable=False,
                server_default="[]",
            )
        )

    # 2. Additive columns on releases
    with op.batch_alter_table("releases") as batch_op:
        batch_op.add_column(
            sa.Column(
                "is_tag_fallback",
                sa.Boolean(),
                nullable=False,
                server_default="0",
            )
        )
        batch_op.add_column(
            sa.Column("commit_sha", sa.String(40), nullable=True)
        )

    # 3. New table: discussions
    op.create_table(
        "discussions",
        sa.Column("id", PK_TYPE, primary_key=True, autoincrement=True),
        sa.Column(
            "repository_id",
            sa.BigInteger(),
            sa.ForeignKey("repositories.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("discussion_number", sa.Integer(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("body", sa.Text(), nullable=True),
        sa.Column("author", sa.String(80), nullable=True),
        sa.Column("author_id", sa.BigInteger(), nullable=True),
        sa.Column("category_id", sa.String(80), nullable=True),
        sa.Column("category_name", sa.String(80), nullable=False),
        sa.Column(
            "category_is_answerable",
            sa.Boolean(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "is_answered",
            sa.Boolean(),
            nullable=False,
            server_default="0",
        ),
        sa.Column("answered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("answer_author", sa.String(80), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "comments_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "upvote_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "reactions_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.UniqueConstraint(
            "repository_id",
            "discussion_number",
            name="uq_discussions_repo_number",
        ),
    )
    op.create_index(
        "ix_discussions_repo_created",
        "discussions",
        ["repository_id", "created_at"],
    )

    # 4. New table: discussion_comments
    op.create_table(
        "discussion_comments",
        sa.Column("id", PK_TYPE, primary_key=True, autoincrement=True),
        sa.Column(
            "discussion_id",
            sa.BigInteger(),
            sa.ForeignKey("discussions.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "repository_id",
            sa.BigInteger(),
            sa.ForeignKey("repositories.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("comment_id", sa.String(80), nullable=False, unique=True, index=True),
        sa.Column("author", sa.String(80), nullable=True),
        sa.Column("body", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "is_answer",
            sa.Boolean(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "upvote_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )


def downgrade() -> None:
    op.drop_table("discussion_comments")
    op.drop_index("ix_discussions_repo_created", table_name="discussions")
    op.drop_table("discussions")

    with op.batch_alter_table("releases") as batch_op:
        batch_op.drop_column("commit_sha")
        batch_op.drop_column("is_tag_fallback")

    with op.batch_alter_table("pull_requests") as batch_op:
        batch_op.drop_column("requested_reviewers")
        batch_op.drop_column("labels")
        batch_op.drop_column("is_draft")
        batch_op.drop_column("base_branch")
        batch_op.drop_column("head_branch")
        batch_op.drop_column("body")
