"""Add issue_embeddings and discussion_embeddings tables for dynamic vector retrieval.

Revision ID: 005_issue_and_discussion_embeddings
Revises: 004_add_discussions_and_pr_depth
Create Date: 2026-11-16 15:15:00.000000

Adds Stage 5 vector store infrastructure:
- issue_embeddings: 384-dimensional binary float32 vector blobs for repository issues
- discussion_embeddings: 384-dimensional binary float32 vector blobs for repository discussions
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "005_issue_and_discussion_embeddings"
down_revision: Union[str, None] = "004_add_discussions_and_pr_depth"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create issue_embeddings table
    op.create_table(
        "issue_embeddings",
        sa.Column(
            "id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            primary_key=True,
            autoincrement=True,
        ),
        sa.Column(
            "repository_id",
            sa.BigInteger(),
            sa.ForeignKey("repositories.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "issue_id",
            sa.BigInteger(),
            sa.ForeignKey("issues.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("embedding_blob", sa.LargeBinary(), nullable=False),
        sa.Column(
            "model_version",
            sa.String(length=60),
            nullable=False,
            server_default="all-MiniLM-L6-v2",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_issue_embeddings_repo",
        "issue_embeddings",
        ["repository_id"],
        unique=False,
    )
    op.create_index(
        "ix_issue_embeddings_issue",
        "issue_embeddings",
        ["issue_id"],
        unique=True,
    )

    # 2. Create discussion_embeddings table
    op.create_table(
        "discussion_embeddings",
        sa.Column(
            "id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            primary_key=True,
            autoincrement=True,
        ),
        sa.Column(
            "repository_id",
            sa.BigInteger(),
            sa.ForeignKey("repositories.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "discussion_id",
            sa.BigInteger(),
            sa.ForeignKey("discussions.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("embedding_blob", sa.LargeBinary(), nullable=False),
        sa.Column(
            "model_version",
            sa.String(length=60),
            nullable=False,
            server_default="all-MiniLM-L6-v2",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_discussion_embeddings_repo",
        "discussion_embeddings",
        ["repository_id"],
        unique=False,
    )
    op.create_index(
        "ix_discussion_embeddings_discussion",
        "discussion_embeddings",
        ["discussion_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_discussion_embeddings_discussion", table_name="discussion_embeddings")
    op.drop_index("ix_discussion_embeddings_repo", table_name="discussion_embeddings")
    op.drop_table("discussion_embeddings")

    op.drop_index("ix_issue_embeddings_issue", table_name="issue_embeddings")
    op.drop_index("ix_issue_embeddings_repo", table_name="issue_embeddings")
    op.drop_table("issue_embeddings")
