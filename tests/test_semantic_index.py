"""Unit and integration tests for dynamic per-repository semantic index service."""

from datetime import datetime, timezone
import unittest

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.app.database.base import Base
from backend.app.database.models import (
    Discussion,
    DiscussionEmbedding,
    Issue,
    IssueEmbedding,
    Repository,
)
from backend.app.services.nlp.semantic_index_service import (
    HIGH_DUPLICATE_THRESHOLD,
    SemanticIndexService,
)


class TestSemanticIndexService(unittest.TestCase):
    """Test suite for SemanticIndexService indexing, vector retrieval, and duplicate search."""

    @classmethod
    def setUpClass(cls):
        cls.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(cls.engine)
        cls.service = SemanticIndexService()

    def setUp(self):
        self.session = Session(self.engine)
        # Create test repository
        self.repo = Repository(
            id=123456,
            repo_name="test-org/vector-repo",
            owner="test-org",
            name="vector-repo",
            stars_count=100,
            forks_count=20,
            open_issues_count=5,
        )
        self.session.add(self.repo)
        self.session.commit()

    def tearDown(self):
        self.session.close()
        # Clean up database tables for next test
        with self.engine.begin() as conn:
            for table in reversed(Base.metadata.sorted_tables):
                conn.execute(table.delete())

    def test_index_issues_and_query_vectors(self):
        """Verify issue embedding generation, blob persistence, and vector extraction."""
        now = datetime.now(timezone.utc)
        issue1 = Issue(
            id=101,
            repository_id=self.repo.id,
            repo_name=self.repo.repo_name,
            issue_number=1,
            title="CUDA out of memory error during tensor backward pass",
            body="PyTorch allocator failed to reserve 2GB GPU RAM during backpropagation",
            state="open",
            author="ai-dev",
            created_at=now,
        )
        issue2 = Issue(
            id=102,
            repository_id=self.repo.id,
            repo_name=self.repo.repo_name,
            issue_number=2,
            title="Add support for dark theme in settings",
            body="Feature request for dark mode styling in the UI navigation pane",
            state="closed",
            author="ui-designer",
            created_at=now,
        )
        self.session.add_all([issue1, issue2])
        self.session.commit()

        # Run indexing
        indexed_count = self.service.index_repository_issues(self.repo.id, self.session)
        self.assertEqual(indexed_count, 2)

        # Check rows in issue_embeddings
        stored = self.session.query(IssueEmbedding).filter(IssueEmbedding.repository_id == self.repo.id).all()
        self.assertEqual(len(stored), 2)
        for s in stored:
            self.assertEqual(len(s.embedding_blob), 384 * 4)  # 384 float32 = 1,536 bytes
            self.assertEqual(s.model_version, "all-MiniLM-L6-v2")

        # Second indexing call should do nothing (idempotent / incremental)
        reindex_count = self.service.index_repository_issues(self.repo.id, self.session)
        self.assertEqual(reindex_count, 0)

        # Load vectors matrix
        matrix, meta = self.service.get_repository_vectors(self.repo.id, self.session)
        self.assertEqual(matrix.shape, (2, 384))
        self.assertEqual(len(meta), 2)

        # Check strict L2 normalization
        norms = np.linalg.norm(matrix, axis=1)
        for n in norms:
            self.assertAlmostEqual(float(n), 1.0, places=4)

        # Query similar - Duplicate match for Issue 1
        query = "CUDA out of memory in backpropagation on GPU"
        res = self.service.query_similar(self.repo.id, query, self.session, top_k=2)
        self.assertGreaterEqual(res["total_candidates_found"], 1)
        top = res["candidates"][0]
        self.assertEqual(top["number"], 1)
        self.assertEqual(top["entity_type"], "issue")
        self.assertGreaterEqual(top["similarity_score"], 0.65)
        self.assertIn("duplicate", res["primary_recommendation"].lower())

    def test_discussion_indexing_and_bridge_search(self):
        """Verify discussions are indexed and returned alongside issues in unified search."""
        now = datetime.now(timezone.utc)
        disc = Discussion(
            id=201,
            repository_id=self.repo.id,
            discussion_number=5,
            title="RFC: Architecture proposal for custom plugin system",
            body="Community discussion regarding plugin sandboxing and hooks API",
            category_name="RFC",
            author="contributor-42",
            created_at=now,
        )
        self.session.add(disc)
        self.session.commit()

        indexed_disc = self.service.index_repository_discussions(self.repo.id, self.session)
        self.assertEqual(indexed_disc, 1)

        # Query similar matching the RFC
        res = self.service.query_similar(
            self.repo.id,
            "plugin hooks and sandbox architecture",
            self.session,
            include_discussions=True,
        )
        self.assertGreaterEqual(res["total_candidates_found"], 1)
        match = res["candidates"][0]
        self.assertEqual(match["entity_type"], "discussion")
        self.assertEqual(match["number"], 5)
        self.assertEqual(match["category"], "RFC")


if __name__ == "__main__":
    unittest.main()
