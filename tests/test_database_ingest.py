"""Unit tests for OpenSourceLens database ingestion pipeline."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.database.base import Base
from backend.app.database.ingest import (
    ingest_issue_labels,
    ingest_issues,
    ingest_related_issues_cache,
    ingest_repositories,
    ingest_repository_analytics,
    run_ingest_pipeline,
)
from backend.app.database.models import (
    Issue,
    IssueLabel,
    RelatedIssuesCache,
    Repository,
    RepositoryAnalytics,
)


class TestDatabaseIngest(unittest.TestCase):
    """Test suite for ETL data ingestion pipeline."""

    def setUp(self):
        """Set up an isolated temporary SQLite database and session factory."""
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = os.path.join(self.temp_dir.name, "test_ingest.db")
        self.db_url = f"sqlite:///{self.db_path}"
        self.engine = create_engine(self.db_url)
        Base.metadata.create_all(bind=self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.session: Session = self.SessionLocal()

    def tearDown(self):
        """Clean up database session, engine, and directory."""
        self.session.close()
        self.engine.dispose()
        self.temp_dir.cleanup()

    def test_ingest_repositories(self):
        """Verify repositories ingestion loads the 4 canonical repositories correctly."""
        repos = ingest_repositories(self.session)
        self.assertEqual(len(repos), 4)

        repo_names = {r.repo_name for r in repos}
        expected = {
            "microsoft/vscode",
            "pytorch/pytorch",
            "kubernetes/kubernetes",
            "huggingface/transformers",
        }
        self.assertEqual(repo_names, expected)

        # Check VS Code attributes
        vscode = self.session.scalar(select(Repository).where(Repository.repo_name == "microsoft/vscode"))
        self.assertIsNotNone(vscode)
        self.assertEqual(vscode.id, 41881900)
        self.assertEqual(vscode.primary_language, "TypeScript")
        self.assertGreater(vscode.stars_count, 100000)

    def test_ingest_issues_and_foreign_keys(self):
        """Verify issue ingestion links issues to repositories with proper data types."""
        ingest_repositories(self.session)
        issues_count = ingest_issues(self.session, batch_size=200)
        self.assertEqual(issues_count, 2328)

        # Verify database record count
        db_count = len(self.session.scalars(select(Issue)).all())
        self.assertEqual(db_count, 2328)

        # Check an open issue (censored, resolution_time_hours is None)
        open_issue = self.session.scalar(select(Issue).where(Issue.state == "open"))
        self.assertIsNotNone(open_issue)
        self.assertTrue(open_issue.is_censored)
        self.assertIsNone(open_issue.resolution_time_hours)
        self.assertIsNone(open_issue.closed_at)

        # Check a closed issue (resolution_time_hours is a positive float)
        closed_issue = self.session.scalar(select(Issue).where(Issue.state == "closed"))
        self.assertIsNotNone(closed_issue)
        self.assertFalse(closed_issue.is_censored)
        self.assertIsNotNone(closed_issue.resolution_time_hours)
        self.assertGreater(closed_issue.resolution_time_hours, 0.0)
        self.assertIsNotNone(closed_issue.closed_at)

        # Check repository relationship
        self.assertIsNotNone(closed_issue.repository)
        self.assertEqual(closed_issue.repository.id, closed_issue.repository_id)

    def test_ingest_issue_labels(self):
        """Verify issue labels are normalized and stored."""
        ingest_repositories(self.session)
        ingest_issues(self.session)
        labels_count = ingest_issue_labels(self.session, batch_size=500)
        self.assertGreater(labels_count, 0)

        # Inspect canonical labels
        canonical_label = self.session.scalar(select(IssueLabel).where(IssueLabel.is_canonical == True))
        self.assertIsNotNone(canonical_label)
        self.assertIn(canonical_label.canonical_category, ["bug", "feature", "documentation", "question", "other"])

    def test_ingest_repository_analytics(self):
        """Verify repository analytics snapshot aggregates match issue data."""
        ingest_repositories(self.session)
        ingest_issues(self.session)
        analytics_records = ingest_repository_analytics(self.session)
        self.assertEqual(len(analytics_records), 4)

        for rec in analytics_records:
            self.assertEqual(rec.total_issues, rec.open_issues + rec.closed_issues)
            self.assertIsNotNone(rec.category_distribution)
            self.assertIsNotNone(rec.median_resolution_hours)
            self.assertGreater(rec.median_resolution_hours, 0.0)
            self.assertIsNotNone(rec.survival_rate_30d)

    def test_ingest_related_issues_cache(self):
        """Verify semantic similarity calculation, self-exclusion, and ranking."""
        ingest_repositories(self.session)
        ingest_issues(self.session)
        top_k = 3
        edge_count = ingest_related_issues_cache(self.session, top_k=top_k, batch_size=500)

        # 2,328 issues * top_k 3 = 6,984 edges
        self.assertEqual(edge_count, 2328 * top_k)

        # Verify ranks and self-exclusion for a sample issue
        sample_issue = self.session.scalar(select(Issue))
        related_edges = self.session.scalars(
            select(RelatedIssuesCache)
            .where(RelatedIssuesCache.source_issue_id == sample_issue.id)
            .order_by(RelatedIssuesCache.rank)
        ).all()

        self.assertEqual(len(related_edges), top_k)
        ranks = [e.rank for e in related_edges]
        self.assertEqual(ranks, [1, 2, 3])

        # Self-exclusion check: source cannot equal target
        for edge in related_edges:
            self.assertNotEqual(edge.source_issue_id, edge.target_issue_id)
            self.assertGreaterEqual(edge.similarity_score, -1.0)
            self.assertLessEqual(edge.similarity_score, 1.0)

    def test_full_pipeline_idempotency(self):
        """Verify running run_ingest_pipeline completes cleanly and is idempotent."""
        summary = run_ingest_pipeline(self.session, clean=True, top_k=2, target_engine=self.engine)
        self.assertEqual(summary["status"], "success")
        self.assertEqual(summary["repositories_count"], 4)
        self.assertEqual(summary["issues_count"], 2328)
        self.assertEqual(summary["analytics_snapshots"], 4)
        self.assertEqual(summary["related_edges_count"], 2328 * 2)

        # Run second time without clean
        summary_2 = run_ingest_pipeline(self.session, clean=False, top_k=2, target_engine=self.engine)
        self.assertEqual(summary_2["status"], "success")
        self.assertEqual(summary_2["issues_count"], 2328)


if __name__ == "__main__":
    unittest.main()
