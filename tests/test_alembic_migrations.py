"""Tests for Phase B Alembic database migrations.

Verifies:
1. Automated pre-migration backup creation and integrity.
2. Baseline schema stamp on staging database.
3. Migration upgrade to head (002_add_dynamic_tables) with SQLite batch mode.
4. Retention and readability of all 2,328 benchmark issues and 4 repositories.
5. Verification of all 10 new tables and additive columns.
6. Verification of the IssueComment CHECK constraint for all 3 valid states and rejection of dual parents.
7. Verification of migration rollback (downgrade -1) and clean re-upgrade.
"""

import gc
import shutil
import sqlite3
from pathlib import Path
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from backend.app.database.models import (
    Issue,
    IssueComment,
    Repository,
    PullRequest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"
PRIMARY_DB = PROJECT_ROOT / "data" / "opensource_lens.db"
BACKUP_DIR = PROJECT_ROOT / "data" / "backups"
PRE_PHASE_B_BACKUP = BACKUP_DIR / "opensource_lens_pre_phase_b_backup.db"


@pytest.fixture(scope="module")
def staging_db(tmp_path_factory):
    """Create a temporary isolated copy of the pre-migration baseline database for staging migration tests."""
    temp_dir = tmp_path_factory.mktemp("migration_test")
    staging_file = temp_dir / "test_migration_staging.db"
    
    source_db = PRE_PHASE_B_BACKUP if PRE_PHASE_B_BACKUP.exists() else PRIMARY_DB
    assert source_db.exists(), f"Source database {source_db} must exist for staging test"
    shutil.copy2(source_db, staging_file)
    
    # Configure alembic config pointing to this staging DB
    alembic_cfg = Config(str(ALEMBIC_INI))
    alembic_cfg.set_main_option("script_location", str(PROJECT_ROOT / "backend" / "alembic"))
    alembic_cfg.set_main_option("sqlalchemy.url", f"sqlite:///{staging_file}")
    
    yield staging_file, alembic_cfg
    
    # Force garbage collection to close any lingering SQLite handles on Windows
    gc.collect()


def test_pre_migration_backup_exists():
    """Verify that the pre-migration backup file exists in data/backups/."""
    assert BACKUP_DIR.exists(), "Backup directory data/backups/ must exist"
    backups = list(BACKUP_DIR.glob("*.db"))
    assert len(backups) >= 1, "At least one pre-migration database backup must exist in data/backups/"
    backup_file = backups[0]
    assert backup_file.stat().st_size > 10_000_000, f"Backup file {backup_file} is smaller than expected"


def test_staging_migration_lifecycle(staging_db):
    """Test full migration lifecycle: stamp baseline -> upgrade head -> verify -> downgrade -> re-upgrade."""
    staging_file, alembic_cfg = staging_db

    # Step 1: Stamp baseline
    command.stamp(alembic_cfg, "001_initial_benchmark_schema")
    
    # Verify current revision is 001
    conn = sqlite3.connect(staging_file)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT version_num FROM alembic_version;")
        row = cursor.fetchone()
        assert row is not None and row[0] == "001_initial_benchmark_schema"
    finally:
        conn.close()

    # Step 2: Upgrade to head (002_add_dynamic_tables)
    command.upgrade(alembic_cfg, "head")

    conn = sqlite3.connect(staging_file)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT version_num FROM alembic_version;")
        row = cursor.fetchone()
        assert row is not None and row[0] == "002_add_dynamic_tables"

        # Step 3: Verify all 16 tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = set(r[0] for r in cursor.fetchall())
        expected_tables = {
            "repositories",
            "issues",
            "issue_labels",
            "repository_analytics",
            "predictions",
            "related_issues_cache",
            "collection_runs",
            "analysis_runs",
            "commits",
            "contributors",
            "contributor_activities",
            "pull_requests",
            "pull_request_reviews",
            "pull_request_review_comments",
            "releases",
            "issue_comments",
            "alembic_version",
        }
        for exp in expected_tables:
            assert exp in tables, f"Table '{exp}' was not created by migration"

        # Step 4: Verify existing benchmark data is 100% preserved
        cursor.execute("SELECT count(*) FROM repositories;")
        assert cursor.fetchone()[0] == 4, "Repositories table must retain all 4 benchmark repos"

        cursor.execute("SELECT count(*) FROM issues;")
        assert cursor.fetchone()[0] == 2328, "Issues table must retain all 2,328 benchmark issues"

        # Step 5: Verify additive columns on repositories
        cursor.execute("PRAGMA table_info(repositories);")
        repo_cols = {r[1]: r for r in cursor.fetchall()}
        assert "is_active" in repo_cols
        assert "is_archived" in repo_cols
        assert "is_fork" in repo_cols
        assert "upstream_repo" in repo_cols
        assert "last_synced_at" in repo_cols
    finally:
        conn.close()

    # Step 6: Test downgrade -1
    command.downgrade(alembic_cfg, "-1")
    conn = sqlite3.connect(staging_file)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT version_num FROM alembic_version;")
        assert cursor.fetchone()[0] == "001_initial_benchmark_schema"

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = set(r[0] for r in cursor.fetchall())
        assert "collection_runs" not in tables
        assert "commits" not in tables
        assert "pull_requests" not in tables
    finally:
        conn.close()

    # Step 7: Re-upgrade to head
    command.upgrade(alembic_cfg, "head")
    conn = sqlite3.connect(staging_file)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT version_num FROM alembic_version;")
        assert cursor.fetchone()[0] == "002_add_dynamic_tables"
    finally:
        conn.close()


def test_issue_comment_check_constraint(staging_db):
    """Test that issue_comments strictly enforces the mutual exclusion CHECK constraint across all states."""
    staging_file, _ = staging_db
    engine = create_engine(f"sqlite:///{staging_file}")

    try:
        with Session(engine) as session:
            # Fetch an existing repository and issue
            repo = session.scalar(select(Repository))
            assert repo is not None
            issue = session.scalar(select(Issue).where(Issue.repository_id == repo.id))
            assert issue is not None

            # Create a sample pull request
            pr = PullRequest(
                repository_id=repo.id,
                pr_number=9999,
                title="Test PR",
                state="open",
                created_at=issue.created_at,
            )
            session.add(pr)
            session.commit()
            session.refresh(pr)

            # State 1: Comment with issue_id only (valid)
            comment_issue = IssueComment(
                repository_id=repo.id,
                comment_id=100001,
                issue_number=issue.issue_number,
                issue_id=issue.id,
                pull_request_id=None,
                author="testuser",
                created_at=issue.created_at,
                body="Comment on issue",
            )
            session.add(comment_issue)
            session.commit()
            assert comment_issue.id is not None

            # State 2: Comment with pull_request_id only (valid)
            comment_pr = IssueComment(
                repository_id=repo.id,
                comment_id=100002,
                issue_number=pr.pr_number,
                issue_id=None,
                pull_request_id=pr.id,
                author="testuser",
                created_at=pr.created_at,
                body="Comment on PR",
            )
            session.add(comment_pr)
            session.commit()
            assert comment_pr.id is not None

            # State 3: Comment with both NULL (valid: parent not yet linked)
            comment_unlinked = IssueComment(
                repository_id=repo.id,
                comment_id=100003,
                issue_number=8888,
                issue_id=None,
                pull_request_id=None,
                author="testuser",
                created_at=pr.created_at,
                body="Unlinked comment",
            )
            session.add(comment_unlinked)
            session.commit()
            assert comment_unlinked.id is not None

            # State 4: Comment with BOTH issue_id and pull_request_id (INVALID - must fail CHECK constraint)
            comment_invalid = IssueComment(
                repository_id=repo.id,
                comment_id=100004,
                issue_number=9999,
                issue_id=issue.id,
                pull_request_id=pr.id,
                author="testuser",
                created_at=pr.created_at,
                body="Invalid dual parent comment",
            )
            session.add(comment_invalid)
            with pytest.raises(Exception) as exc_info:
                session.commit()
            session.rollback()
            assert "CHECK constraint failed" in str(exc_info.value) or "chk_issue_comment_parent" in str(exc_info.value)
    finally:
        engine.dispose()
