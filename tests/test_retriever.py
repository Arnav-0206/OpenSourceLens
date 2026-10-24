"""
Unit tests for RelatedIssueRetriever (Phase 7, Step 7.4).
"""

from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from ml.related_issues.retriever import RelatedIssueRetriever
from ml.related_issues.store import IssueEmbeddingStore


class DummyEmbedder:
    """Mock embedder returning deterministic vectors for testing."""
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def format_issue_text(self, title: str, body: str = "") -> str:
        return f"Title: {title} | Description: {body}"

    def encode_text(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        # Hash text to deterministic vector
        val = sum(ord(c) for c in text) % self.dimension
        vec[val] = 1.0
        return vec


class TestRelatedIssueRetriever(unittest.TestCase):
    """Test suite for top-K issue retrieval, self-match exclusion, and filtering."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store_dir = Path(self.temp_dir) / "test_store"

        # Create dummy corpus of 25 issues with 384 dimensions
        np.random.seed(42)
        raw_embs = np.random.randn(25, 384).astype(np.float32)
        self.dummy_embs = raw_embs / np.linalg.norm(raw_embs, axis=1, keepdims=True)

        self.dummy_df = pd.DataFrame({
            "issue_id": [2000 + i for i in range(25)],
            "issue_number": [100 + i for i in range(25)],
            "repo_name": ["repo_alpha"] * 15 + ["repo_beta"] * 10,
            "title": [f"Feature request or bug title {i}" for i in range(25)],
            "clean_title": [f"feature request or bug title {i}" for i in range(25)],
            "clean_body": [f"detailed description body {i}" for i in range(25)],
            "canonical_label": ["bug"] * 12 + ["feature"] * 8 + ["other"] * 5,
            "state": ["closed"] * 18 + ["open"] * 7,
            "created_at": ["2026-01-01T00:00:00Z"] * 25,
        })

        # Save store
        self.store = IssueEmbeddingStore(self.store_dir)
        self.store.save(self.dummy_embs, self.dummy_df)

        self.mock_embedder = DummyEmbedder(384)
        self.retriever = RelatedIssueRetriever(
            store=self.store,
            embedder=self.mock_embedder,
            store_dir=self.store_dir,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_retrieve_by_vector_structure_and_order(self):
        """Test candidate structure and descending order of similarity scores."""
        query_vec = self.dummy_embs[0]
        results = self.retriever.retrieve_by_vector(query_vec, top_k=5)

        self.assertEqual(len(results), 5)
        # Check required fields
        required_keys = {
            "rank", "issue_id", "issue_number", "repo_name",
            "title", "clean_title", "canonical_label", "state",
            "created_at", "similarity_score"
        }
        for item in results:
            self.assertTrue(required_keys.issubset(item.keys()))

        # Check descending order
        scores = [r["similarity_score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        # Vector 0 should match itself with score 1.0 when not excluded
        self.assertEqual(results[0]["issue_id"], 2000)
        self.assertAlmostEqual(results[0]["similarity_score"], 1.0, places=4)

    def test_retrieve_by_id_self_exclusion(self):
        """Test that query by issue_id STRICTLY excludes the query issue from results."""
        query_id = 2000
        results = self.retriever.retrieve_by_id(issue_id=query_id, top_k=5)

        self.assertEqual(len(results), 5)
        returned_ids = [r["issue_id"] for r in results]
        self.assertNotIn(query_id, returned_ids)
        self.assertNotIn(str(query_id), [str(x) for x in returned_ids])

        # Rank 1 must be the second closest neighbor, not self
        self.assertNotEqual(results[0]["issue_id"], query_id)

    def test_retrieve_by_text(self):
        """Test retrieving candidates for arbitrary text title and description."""
        results = self.retriever.retrieve_by_text(
            title="memory leak in allocator",
            body="process memory continuously increases",
            top_k=3,
        )
        self.assertEqual(len(results), 3)
        scores = [r["similarity_score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_repo_filter(self):
        """Test restricting candidate results to a specific repository."""
        query_vec = self.dummy_embs[0]
        results_beta = self.retriever.retrieve_by_vector(
            query_vec, top_k=5, repo_filter="repo_beta"
        )
        self.assertEqual(len(results_beta), 5)
        for r in results_beta:
            self.assertEqual(r["repo_name"], "repo_beta")

    def test_state_filter(self):
        """Test filtering by issue state ('open' or 'closed')."""
        query_vec = self.dummy_embs[0]
        results_open = self.retriever.retrieve_by_vector(
            query_vec, top_k=5, state_filter="open"
        )
        self.assertEqual(len(results_open), 5)
        for r in results_open:
            self.assertEqual(r["state"], "open")

    def test_top_k_larger_than_corpus(self):
        """Test that requesting K larger than corpus size caps gracefully."""
        query_vec = self.dummy_embs[0]
        results = self.retriever.retrieve_by_vector(query_vec, top_k=100)
        self.assertEqual(len(results), 25)

    def test_benchmark_retrieval(self):
        """Test retrieval benchmarking routine."""
        bench = self.retriever.benchmark_retrieval(
            sample_queries=5, top_k_values=[1, 3, 5]
        )
        self.assertIn("top1_mean", bench)
        self.assertIn("vector_latencies_ms", bench)
        self.assertIn("text_latencies_ms", bench)
        self.assertEqual(len(bench["top1_scores"]), 5)

    def test_plot_topk_diagnostics(self):
        """Test diagnostic visualization generation."""
        bench = self.retriever.benchmark_retrieval(
            sample_queries=5, top_k_values=[1, 3, 5]
        )
        fig_path = Path(self.temp_dir) / "test_fig26.png"
        out = self.retriever.plot_topk_diagnostics(benchmark=bench, save_path=fig_path)
        self.assertTrue(out.exists())
        self.assertGreater(out.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
