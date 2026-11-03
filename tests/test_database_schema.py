"""Tests for OpenSourceLens database schema, SQLAlchemy 2.0 ORM models, and DDL generation."""

from datetime import datetime, timezone
import unittest

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.database.base import Base
from backend.app.database.models import (
    Issue,
    IssueLabel,
    Prediction,
    RelatedIssuesCache,
    Repository,
    RepositoryAnalytics,
)
from backend.app.database.schema_generator import generate_ddl_for_dialect
from backend.app.database.session import build_engine, get_database_url


class TestDatabaseSchema(unittest.TestCase):
    """Test suite for database models, constraints, and dialect compatibility."""

    @classmethod
    def setUpClass(cls):
        """Set up in-memory SQLite engine and session factory for testing."""
        cls.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(bind=cls.engine)
        cls.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=cls.engine
        )

    def test_metadata_tables_registered(self):
        """Ensure all 6 core entity tables are registered in SQLAlchemy metadata."""
        expected_tables = {
            "repositories",
            "issues",
            "issue_labels",
            "repository_analytics",
            "predictions",
            "related_issues_cache",
        }
        actual_tables = set(Base.metadata.tables.keys())
        self.assertTrue(expected_tables.issubset(actual_tables))

    def test_ddl_generation_postgresql(self):
        """Ensure PostgreSQL DDL generates valid CREATE TABLE and CREATE INDEX statements."""
        ddl = generate_ddl_for_dialect("postgresql")
        self.assertIn("CREATE TABLE repositories", ddl)
        self.assertIn("CREATE TABLE issues", ddl)
        self.assertIn("CREATE TABLE issue_labels", ddl)
        self.assertIn("CREATE TABLE repository_analytics", ddl)
        self.assertIn("CREATE TABLE predictions", ddl)
        self.assertIn("CREATE TABLE related_issues_cache", ddl)
        self.assertIn("ix_issues_repo_number", ddl)
        self.assertIn("ix_issues_repo_state", ddl)
        self.assertIn("ix_related_source_rank", ddl)
        self.assertIn("JSON", ddl)

    def test_ddl_generation_sqlite(self):
        """Ensure SQLite DDL generates cleanly without syntax errors."""
        ddl = generate_ddl_for_dialect("sqlite")
        self.assertIn("CREATE TABLE repositories", ddl)
        self.assertIn("CREATE TABLE issues", ddl)
        self.assertIn("CREATE TABLE predictions", ddl)

    def test_database_url_resolution(self):
        """Test URL resolution default and engine instantiation."""
        url = get_database_url()
        self.assertTrue(url.startswith("sqlite") or url.startswith("postgresql"))
        eng = build_engine(url)
        self.assertIsNotNone(eng)

    def test_orm_crud_and_relationships(self):
        """Test full transactional CRUD, relationship navigation, and integrity."""
        session: Session = self.SessionLocal()
        now = datetime.now(timezone.utc)

        try:
            # 1. Create Repository
            repo = Repository(
                id=41881900,
                repo_name="microsoft/vscode",
                owner="microsoft",
                name="vscode",
                description="Visual Studio Code",
                primary_language="TypeScript",
                stars_count=193939,
                forks_count=45875,
                open_issues_count=21922,
            )
            session.add(repo)
            session.commit()

            # 2. Add Issue
            issue_1 = Issue(
                id=10001,
                repository_id=repo.id,
                repo_name=repo.repo_name,
                issue_number=101,
                title="Crash on startup in window manager",
                body="Steps to reproduce...",
                clean_title="crash on startup in window manager",
                clean_body="steps to reproduce",
                combined_text="crash on startup in window manager steps to reproduce",
                clean_title_word_count=6,
                clean_body_word_count=3,
                state="open",
                author="dev_tester",
                author_type="User",
                author_association="CONTRIBUTOR",
                is_internal_author=True,
                is_bot_author=False,
                created_at=now,
                canonical_label="bug",
                canonical_label_id=0,
                speed_tier="1-7 Days",
                is_censored=True,
            )
            session.add(issue_1)

            issue_2 = Issue(
                id=10002,
                repository_id=repo.id,
                repo_name=repo.repo_name,
                issue_number=102,
                title="Add support for dark high-contrast themes",
                body="Feature request details...",
                clean_title="add support for dark high-contrast themes",
                clean_body="feature request details",
                combined_text="add support for dark high-contrast themes feature request details",
                clean_title_word_count=6,
                clean_body_word_count=3,
                state="closed",
                author="designer_user",
                author_type="User",
                author_association="NONE",
                is_internal_author=False,
                is_bot_author=False,
                created_at=now,
                closed_at=now,
                resolution_time_hours=24.5,
                resolution_time_days=1.02,
                speed_tier="1-7 Days",
                canonical_label="enhancement",
                canonical_label_id=1,
                is_censored=False,
            )
            session.add(issue_2)
            session.commit()

            # 3. Add Labels to Issue 1
            label_1 = IssueLabel(
                issue_id=issue_1.id,
                label_name="bug",
                is_canonical=True,
                canonical_category="bug",
            )
            label_2 = IssueLabel(
                issue_id=issue_1.id,
                label_name="window-management",
                is_canonical=False,
                canonical_category=None,
            )
            session.add_all([label_1, label_2])

            # 4. Add Repository Analytics Snapshot
            analytics = RepositoryAnalytics(
                repository_id=repo.id,
                repo_name=repo.repo_name,
                snapshot_date=now,
                total_issues=2,
                open_issues=1,
                closed_issues=1,
                category_distribution={"bug": 1, "enhancement": 1},
                median_resolution_hours=24.5,
                mean_resolution_hours=24.5,
                p25_resolution_hours=24.5,
                p75_resolution_hours=24.5,
                survival_rate_30d=0.5,
                speed_tier_distribution={"1-7 Days": 2},
            )
            session.add(analytics)

            # 5. Add Predictions (one linked to issue_1, one arbitrary frontend input)
            pred_linked = Prediction(
                issue_id=issue_1.id,
                repo_name=repo.repo_name,
                input_title=issue_1.title,
                input_body=issue_1.body,
                model_name="distilbert",
                task_type="classification",
                predicted_label="bug",
                confidence_score=0.942,
                probabilities={"bug": 0.942, "enhancement": 0.045, "other": 0.013},
                inference_time_ms=18.4,
            )
            pred_unlinked = Prediction(
                issue_id=None,
                repo_name=None,
                input_title="Ad hoc user issue query",
                input_body="No repo context provided",
                model_name="tfidf_svm",
                task_type="classification",
                predicted_label="question",
                confidence_score=0.812,
                inference_time_ms=2.1,
            )
            session.add_all([pred_linked, pred_unlinked])

            # 6. Add Related Issues Cache Pairing
            related = RelatedIssuesCache(
                source_issue_id=issue_1.id,
                target_issue_id=issue_2.id,
                similarity_score=0.785,
                rank=1,
                model_version="all-MiniLM-L6-v2",
            )
            session.add(related)
            session.commit()

            # 7. Verify Queries and Relationships
            fetched_repo = session.scalar(
                select(Repository).where(Repository.repo_name == "microsoft/vscode")
            )
            self.assertIsNotNone(fetched_repo)
            self.assertEqual(len(fetched_repo.issues), 2)
            self.assertEqual(len(fetched_repo.analytics), 1)

            fetched_issue = session.scalar(
                select(Issue).where(
                    Issue.repo_name == "microsoft/vscode",
                    Issue.issue_number == 101,
                )
            )
            self.assertIsNotNone(fetched_issue)
            self.assertEqual(len(fetched_issue.labels), 2)
            self.assertEqual(fetched_issue.repository.name, "vscode")
            self.assertEqual(len(fetched_issue.predictions), 1)
            self.assertEqual(len(fetched_issue.source_related), 1)
            self.assertEqual(
                fetched_issue.source_related[0].target_issue.title,
                "Add support for dark high-contrast themes",
            )

            # 8. Test repr
            self.assertIn("Repository(id=41881900)", repr(fetched_repo))
            self.assertIn("Issue(id=10001)", repr(fetched_issue))

        finally:
            session.close()

    def test_duplicate_issue_number_raises_integrity_error(self):
        """Ensure (repo_name, issue_number) unique constraint is enforced."""
        from sqlalchemy.exc import IntegrityError
        session: Session = self.SessionLocal()
        now = datetime.now(timezone.utc)
        try:
            repo = Repository(
                id=99999,
                repo_name="org/dup-test",
                owner="org",
                name="dup-test",
            )
            session.add(repo)
            session.commit()

            i1 = Issue(
                id=9901,
                repository_id=repo.id,
                repo_name=repo.repo_name,
                issue_number=1,
                title="First",
                state="open",
                author="user1",
                created_at=now,
            )
            session.add(i1)
            session.commit()

            i2 = Issue(
                id=9902,
                repository_id=repo.id,
                repo_name=repo.repo_name,
                issue_number=1,  # Duplicate!
                title="Second with duplicate number",
                state="open",
                author="user2",
                created_at=now,
            )
            session.add(i2)
            with self.assertRaises(IntegrityError):
                session.commit()
            session.rollback()
        finally:
            session.close()

    def test_duplicate_related_issues_pairing_raises_integrity_error(self):
        """Ensure (source_issue_id, target_issue_id) unique constraint is enforced."""
        from sqlalchemy.exc import IntegrityError
        session: Session = self.SessionLocal()
        now = datetime.now(timezone.utc)
        try:
            repo = Repository(
                id=88888,
                repo_name="org/rel-test",
                owner="org",
                name="rel-test",
            )
            session.add(repo)
            session.commit()

            i1 = Issue(
                id=8801,
                repository_id=repo.id,
                repo_name=repo.repo_name,
                issue_number=1,
                title="Issue 1",
                state="open",
                author="u1",
                created_at=now,
            )
            i2 = Issue(
                id=8802,
                repository_id=repo.id,
                repo_name=repo.repo_name,
                issue_number=2,
                title="Issue 2",
                state="open",
                author="u2",
                created_at=now,
            )
            session.add_all([i1, i2])
            session.commit()

            rel1 = RelatedIssuesCache(
                source_issue_id=i1.id,
                target_issue_id=i2.id,
                similarity_score=0.9,
                rank=1,
            )
            session.add(rel1)
            session.commit()

            rel2 = RelatedIssuesCache(
                source_issue_id=i1.id,
                target_issue_id=i2.id,
                similarity_score=0.85,
                rank=2,
            )
            session.add(rel2)
            with self.assertRaises(IntegrityError):
                session.commit()
            session.rollback()
        finally:
            session.close()

    def test_cascade_delete_repository(self):
        """Ensure deleting a repository cascades to issues and labels."""
        session: Session = self.SessionLocal()
        now = datetime.now(timezone.utc)
        try:
            repo = Repository(
                id=77777,
                repo_name="org/cascade-test",
                owner="org",
                name="cascade-test",
            )
            session.add(repo)
            session.commit()

            issue = Issue(
                id=7701,
                repository_id=repo.id,
                repo_name=repo.repo_name,
                issue_number=1,
                title="Cascade issue",
                state="open",
                author="u1",
                created_at=now,
            )
            session.add(issue)
            session.commit()

            label = IssueLabel(
                issue_id=issue.id,
                label_name="test-label",
            )
            session.add(label)
            session.commit()

            # Delete repository
            session.delete(repo)
            session.commit()

            # Verify issues and labels are gone
            remaining_issues = session.scalars(
                select(Issue).where(Issue.repository_id == 77777)
            ).all()
            self.assertEqual(len(remaining_issues), 0)

            remaining_labels = session.scalars(
                select(IssueLabel).where(IssueLabel.issue_id == 7701)
            ).all()
            self.assertEqual(len(remaining_labels), 0)
        finally:
            session.close()


if __name__ == "__main__":
    unittest.main()
