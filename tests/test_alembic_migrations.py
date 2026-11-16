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
    AnalysisRun,
    AnalysisRunAttempt,
    Commit,
    Contributor,
    Discussion,
    DiscussionComment,
    Issue,
    IssueComment,
    IssueLabel,
    Prediction,
    PullRequest,
    Release,
    Repository,
    RepositoryAnalytics,
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

    # Step 2: Upgrade to 002_add_dynamic_tables
    command.upgrade(alembic_cfg, "002_add_dynamic_tables")

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
        assert cursor.fetchone()[0] == "005_issue_and_discussion_embeddings"

        # Verify discussions, discussion_comments, issue_embeddings, and discussion_embeddings tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = set(r[0] for r in cursor.fetchall())
        assert "discussions" in tables
        assert "discussion_comments" in tables
        assert "issue_embeddings" in tables
        assert "discussion_embeddings" in tables

        # Verify additive columns on pull_requests
        cursor.execute("PRAGMA table_info(pull_requests);")
        pr_cols = {r[1] for r in cursor.fetchall()}
        assert "body" in pr_cols
        assert "head_branch" in pr_cols
        assert "base_branch" in pr_cols
        assert "is_draft" in pr_cols
        assert "labels" in pr_cols
        assert "requested_reviewers" in pr_cols

        # Verify additive columns on releases
        cursor.execute("PRAGMA table_info(releases);")
        rel_cols = {r[1] for r in cursor.fetchall()}
        assert "is_tag_fallback" in rel_cols
        assert "commit_sha" in rel_cols
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


def test_migration_003_lifecycle(staging_db):
    """Test Phase C Migration 003: upgrade from 002 -> verify schema & data -> downgrade -> re-upgrade."""
    staging_file, alembic_cfg = staging_db

    # Ensure baseline is at 002 before testing 003
    command.downgrade(alembic_cfg, "002_add_dynamic_tables")

    # Step 1: Upgrade to 003
    command.upgrade(alembic_cfg, "003_add_phase_c_analytics_infrastructure")

    conn = sqlite3.connect(staging_file)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT version_num FROM alembic_version;")
        assert cursor.fetchone()[0] == "003_add_phase_c_analytics_infrastructure"

        # Verify benchmark issues preserved
        cursor.execute("SELECT count(*) FROM issues;")
        assert cursor.fetchone()[0] == 2328

        # Verify repository additions
        cursor.execute("PRAGMA table_info(repositories);")
        repo_cols = {r[1]: r for r in cursor.fetchall()}
        assert "analytics_source_revision" in repo_cols
        assert "last_analyzed_at" in repo_cols

        # Verify commits committer_date addition and NULL preservation
        cursor.execute("PRAGMA table_info(commits);")
        commit_cols = {r[1]: r for r in cursor.fetchall()}
        assert "committer_date" in commit_cols
        cursor.execute("SELECT count(*) FROM commits WHERE committer_date IS NOT NULL;")
        assert cursor.fetchone()[0] == 0, "Historical commits must have NULL committer_date"

        # Verify analysis_runs additions
        cursor.execute("PRAGMA table_info(analysis_runs);")
        ar_cols = {r[1]: r for r in cursor.fetchall()}
        assert "requested_source_revision" in ar_cols
        assert "analytics_source_revision" in ar_cols
        assert "requested_methodology_version" in ar_cols
        assert "methodology_version" in ar_cols
        assert "analysis_reference_time" in ar_cols
        assert "source_content_digest" in ar_cols
        assert "worker_id" in ar_cols
        assert "leased_until" in ar_cols
        assert "lease_version" in ar_cols
        assert "retry_count" in ar_cols
        assert "max_retries" in ar_cols
        assert "metrics_payload" in ar_cols
        assert "coverage_summary" in ar_cols
        assert "confidence_summary" in ar_cols
        assert "rule_interpretations" in ar_cols

        # Verify partial unique index
        cursor.execute("PRAGMA index_list(analysis_runs);")
        indexes = {idx[1] for idx in cursor.fetchall()}
        assert "uq_analysis_runs_active_repo" in indexes

        # Verify analysis_run_attempts table
        cursor.execute("PRAGMA table_info(analysis_run_attempts);")
        att_cols = {r[1]: r for r in cursor.fetchall()}
        assert "outcome" in att_cols
        assert "analysis_run_id" in att_cols
    finally:
        conn.close()

    # Step 2: Downgrade to 002
    command.downgrade(alembic_cfg, "002_add_dynamic_tables")

    conn = sqlite3.connect(staging_file)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT version_num FROM alembic_version;")
        assert cursor.fetchone()[0] == "002_add_dynamic_tables"

        cursor.execute("PRAGMA table_info(analysis_runs);")
        ar_cols_down = {r[1]: r for r in cursor.fetchall()}
        assert "requested_source_revision" not in ar_cols_down

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='analysis_run_attempts';")
        assert len(cursor.fetchall()) == 0
    finally:
        conn.close()

    # Step 3: Re-upgrade to 003
    command.upgrade(alembic_cfg, "003_add_phase_c_analytics_infrastructure")

    conn = sqlite3.connect(staging_file)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT version_num FROM alembic_version;")
        assert cursor.fetchone()[0] == "003_add_phase_c_analytics_infrastructure"

        cursor.execute("SELECT count(*) FROM issues;")
        assert cursor.fetchone()[0] == 2328
    finally:
        conn.close()


def test_migration_004_lifecycle(staging_db):
    """Test Stage 3 Migration 004: upgrade from 003 -> verify schema -> downgrade -> re-upgrade."""
    staging_file, alembic_cfg = staging_db

    # Step 1: Upgrade to 004
    command.upgrade(alembic_cfg, "004_add_discussions_and_pr_depth")

    conn = sqlite3.connect(staging_file)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT version_num FROM alembic_version;")
        assert cursor.fetchone()[0] == "004_add_discussions_and_pr_depth"

        # Verify discussions and discussion_comments exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = set(r[0] for r in cursor.fetchall())
        assert "discussions" in tables
        assert "discussion_comments" in tables

        # Verify pull_requests additions
        cursor.execute("PRAGMA table_info(pull_requests);")
        pr_cols = {r[1] for r in cursor.fetchall()}
        assert "body" in pr_cols
        assert "head_branch" in pr_cols
        assert "base_branch" in pr_cols
        assert "is_draft" in pr_cols
        assert "labels" in pr_cols
        assert "requested_reviewers" in pr_cols

        # Verify releases additions
        cursor.execute("PRAGMA table_info(releases);")
        rel_cols = {r[1] for r in cursor.fetchall()}
        assert "is_tag_fallback" in rel_cols
        assert "commit_sha" in rel_cols
    finally:
        conn.close()

    # Step 2: Downgrade back to 003
    command.downgrade(alembic_cfg, "003_add_phase_c_analytics_infrastructure")

    conn = sqlite3.connect(staging_file)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT version_num FROM alembic_version;")
        assert cursor.fetchone()[0] == "003_add_phase_c_analytics_infrastructure"

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('discussions', 'discussion_comments');")
        assert len(cursor.fetchall()) == 0

        cursor.execute("PRAGMA table_info(pull_requests);")
        pr_cols_down = {r[1] for r in cursor.fetchall()}
        assert "is_draft" not in pr_cols_down
        assert "head_branch" not in pr_cols_down

        cursor.execute("PRAGMA table_info(releases);")
        rel_cols_down = {r[1] for r in cursor.fetchall()}
        assert "is_tag_fallback" not in rel_cols_down
    finally:
        conn.close()

    # Step 3: Re-upgrade to 004
    command.upgrade(alembic_cfg, "004_add_discussions_and_pr_depth")

    conn = sqlite3.connect(staging_file)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT version_num FROM alembic_version;")
        assert cursor.fetchone()[0] == "004_add_discussions_and_pr_depth"

        cursor.execute("SELECT count(*) FROM issues;")
        assert cursor.fetchone()[0] == 2328
    finally:
        conn.close()


def test_migration_005_lifecycle(staging_db):
    """Test Stage 5 Migration 005: upgrade to 005 -> verify issue_embeddings and discussion_embeddings -> downgrade -> re-upgrade to head."""
    staging_file, alembic_cfg = staging_db

    # Step 1: Upgrade to head (005)
    command.upgrade(alembic_cfg, "head")

    conn = sqlite3.connect(staging_file)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT version_num FROM alembic_version;")
        assert cursor.fetchone()[0] == "005_issue_and_discussion_embeddings"

        # Verify issue_embeddings and discussion_embeddings exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = set(r[0] for r in cursor.fetchall())
        assert "issue_embeddings" in tables
        assert "discussion_embeddings" in tables

        # Verify columns on issue_embeddings
        cursor.execute("PRAGMA table_info(issue_embeddings);")
        cols = {r[1] for r in cursor.fetchall()}
        assert "id" in cols
        assert "repository_id" in cols
        assert "issue_id" in cols
        assert "embedding_blob" in cols
        assert "model_version" in cols
        assert "created_at" in cols

        # Verify columns on discussion_embeddings
        cursor.execute("PRAGMA table_info(discussion_embeddings);")
        d_cols = {r[1] for r in cursor.fetchall()}
        assert "id" in d_cols
        assert "repository_id" in d_cols
        assert "discussion_id" in d_cols
        assert "embedding_blob" in d_cols
        assert "model_version" in d_cols
        assert "created_at" in d_cols
    finally:
        conn.close()

    # Step 2: Downgrade to 004
    command.downgrade(alembic_cfg, "004_add_discussions_and_pr_depth")
    conn = sqlite3.connect(staging_file)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT version_num FROM alembic_version;")
        assert cursor.fetchone()[0] == "004_add_discussions_and_pr_depth"

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('issue_embeddings', 'discussion_embeddings');")
        assert len(cursor.fetchall()) == 0
    finally:
        conn.close()

    # Step 3: Re-upgrade to head
    command.upgrade(alembic_cfg, "head")
    conn = sqlite3.connect(staging_file)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT version_num FROM alembic_version;")
        assert cursor.fetchone()[0] == "005_issue_and_discussion_embeddings"
    finally:
        conn.close()


def test_fresh_database_upgrade_head_and_orm_queries(tmp_path):
    """Verify that a brand new empty database upgraded via alembic upgrade head matches ORM models 100%."""
    from datetime import datetime, timezone

    fresh_db_file = tmp_path / "fresh_alembic_test.db"
    alembic_cfg = Config(str(ALEMBIC_INI))
    alembic_cfg.set_main_option("script_location", str(PROJECT_ROOT / "backend" / "alembic"))
    alembic_cfg.set_main_option("sqlalchemy.url", f"sqlite:///{fresh_db_file}")

    # Upgrade to head on empty database
    command.upgrade(alembic_cfg, "head")

    # Connect with SQLAlchemy ORM
    engine = create_engine(f"sqlite:///{fresh_db_file}")
    try:
        with Session(engine) as session:
            # Verify ORM queries on all previously drifting tables execute without OperationalError
            labels = session.scalars(select(IssueLabel)).all()
            assert labels == []

            predictions = session.scalars(select(Prediction)).all()
            assert predictions == []

            analytics = session.scalars(select(RepositoryAnalytics)).all()
            assert analytics == []

            runs = session.scalars(select(AnalysisRun)).all()
            assert runs == []

            discussions = session.scalars(select(Discussion)).all()
            assert discussions == []

            disc_comments = session.scalars(select(DiscussionComment)).all()
            assert disc_comments == []

            # Verify complete write and read lifecycle with ORM models
            now = datetime.now(timezone.utc)
            repo = Repository(
                id=101,
                repo_name="test-org/fresh-repo",
                owner="test-org",
                name="fresh-repo",
                stars_count=42,
                forks_count=5,
                open_issues_count=1,
            )
            session.add(repo)
            session.flush()

            issue = Issue(
                id=202,
                repository_id=repo.id,
                repo_name=repo.repo_name,
                issue_number=1,
                title="Fresh DB test issue",
                state="open",
                author="tester",
                created_at=now,
                canonical_label="bug",
                canonical_label_id=1,
            )
            session.add(issue)
            session.flush()

            label = IssueLabel(
                issue_id=issue.id,
                label_name="bug",
                is_canonical=True,
                canonical_category="bug",
            )
            session.add(label)

            pred = Prediction(
                issue_id=issue.id,
                repo_name=repo.repo_name,
                input_title=issue.title,
                input_body=None,
                model_name="distilbert",
                task_type="classification",
                predicted_label="bug",
                confidence_score=0.95,
            )
            session.add(pred)

            repo_ana = RepositoryAnalytics(
                repository_id=repo.id,
                repo_name=repo.repo_name,
                total_issues=1,
                open_issues=1,
                closed_issues=0,
                category_distribution={"bug": 1},
                speed_tier_distribution={"< 1 Day": 1},
            )
            session.add(repo_ana)

            pr = PullRequest(
                repository_id=repo.id,
                pr_number=1,
                title="Fresh DB test PR",
                body="PR description body text",
                state="open",
                head_branch="feature-x",
                base_branch="main",
                is_draft=True,
                labels=["enhancement"],
                requested_reviewers=["lead-reviewer"],
                created_at=now,
            )
            session.add(pr)

            rel = Release(
                repository_id=repo.id,
                release_id=99901,
                tag_name="v1.0.0",
                name="v1.0.0",
                published_at=now,
                is_prerelease=False,
                is_tag_fallback=True,
                commit_sha="a1b2c3d4e5f67890123456789012345678901234",
            )
            session.add(rel)

            disc = Discussion(
                repository_id=repo.id,
                discussion_number=1,
                title="Welcome to discussions",
                body="Q&A thread",
                author="maintainer",
                category_name="Q&A",
                category_is_answerable=True,
                is_answered=True,
                answered_at=now,
                answer_author="contributor",
                created_at=now,
                comments_count=1,
                upvote_count=5,
            )
            session.add(disc)
            session.flush()

            disc_comment = DiscussionComment(
                discussion_id=disc.id,
                repository_id=repo.id,
                comment_id="DC_123456",
                author="contributor",
                body="Here is the answer",
                created_at=now,
                is_answer=True,
                upvote_count=3,
            )
            session.add(disc_comment)
            session.commit()

            # Verify persisted rows match expected ORM properties
            queried_pr = session.scalar(select(PullRequest).where(PullRequest.repository_id == repo.id))
            assert queried_pr is not None
            assert queried_pr.head_branch == "feature-x"
            assert queried_pr.is_draft is True
            assert queried_pr.labels == ["enhancement"]
            assert queried_pr.requested_reviewers == ["lead-reviewer"]

            queried_rel = session.scalar(select(Release).where(Release.release_id == 99901))
            assert queried_rel is not None
            assert queried_rel.is_tag_fallback is True
            assert queried_rel.commit_sha == "a1b2c3d4e5f67890123456789012345678901234"

            queried_disc = session.scalar(select(Discussion).where(Discussion.repository_id == repo.id))
            assert queried_disc is not None
            assert queried_disc.is_answered is True
            assert queried_disc.category_name == "Q&A"
            assert queried_disc.upvote_count == 5

            queried_comment = session.scalar(select(DiscussionComment).where(DiscussionComment.comment_id == "DC_123456"))
            assert queried_comment is not None
            assert queried_comment.is_answer is True

            # Verify persisted rows match expected ORM properties
            queried_label = session.scalar(select(IssueLabel).where(IssueLabel.issue_id == issue.id))
            assert queried_label is not None
            assert queried_label.label_name == "bug"
            assert queried_label.is_canonical is True

            queried_pred = session.scalar(select(Prediction).where(Prediction.issue_id == issue.id))
            assert queried_pred is not None
            assert queried_pred.input_title == "Fresh DB test issue"
            assert queried_pred.predicted_label == "bug"

            queried_ana = session.scalar(
                select(RepositoryAnalytics).where(RepositoryAnalytics.repository_id == repo.id)
            )
            assert queried_ana is not None
            assert queried_ana.category_distribution == {"bug": 1}
    finally:
        engine.dispose()
        gc.collect()


