"""
Unit tests for RelatedIssuePipeline (Phase 7, Step 7.7).
"""

from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from ml.related_issues.pipeline import (
    RelatedIssuePipeline,
    build_github_issue_url,
)
from ml.related_issues.store import IssueEmbeddingStore


class DummyPipelineEmbedder:
    """Mock embedder returning deterministic vectors for pipeline testing."""
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def format_issue_text(self, title: str, body: str = "") -> str:
        return f"Title: {title} | Description: {body}"

    def encode_text(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        val = sum(ord(c) for c in text) % self.dimension
        vec[val] = 1.0
        return vec


class TestRelatedIssuePipeline(unittest.TestCase):
    """Test suite for unified related issue production pipeline."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store_dir = Path(self.temp_dir) / "test_store"

        # Create dummy corpus of 20 issues
        np.random.seed(42)
        raw_embs = np.random.randn(20, 384).astype(np.float32)
        dummy_embs = raw_embs / np.linalg.norm(raw_embs, axis=1, keepdims=True)

        dummy_df = pd.DataFrame({
            "issue_id": [3000 + i for i in range(20)],
            "issue_number": [500 + i for i in range(20)],
            "repo_name": ["microsoft/vscode"] * 10 + ["pytorch/pytorch"] * 10,
            "title": [f"VSCode or PyTorch Issue {i}" for i in range(20)],
            "clean_title": [f"vscode or pytorch issue {i}" for i in range(20)],
            "clean_body": [f"Sample description details {i}" for i in range(20)],
            "canonical_label": ["bug"] * 10 + ["feature"] * 10,
            "state": ["open"] * 10 + ["closed"] * 10,
            "created_at": ["2026-01-01T12:00:00Z"] * 20,
        })

        self.store = IssueEmbeddingStore(self.store_dir)
        self.store.save(dummy_embs, dummy_df)

        self.mock_embedder = DummyPipelineEmbedder(384)
        self.pipeline = RelatedIssuePipeline(
            store_dir=self.store_dir,
            embedder=self.mock_embedder,
            store=self.store,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_build_github_issue_url(self):
        """Test URL formatting for valid and missing issue numbers."""
        self.assertEqual(
            build_github_issue_url("microsoft/vscode", 1234),
            "https://github.com/microsoft/vscode/issues/1234",
        )
        self.assertEqual(
            build_github_issue_url("pytorch/pytorch", None),
            "https://github.com/pytorch/pytorch/issues",
        )
        self.assertEqual(
            build_github_issue_url("pytorch/pytorch", float("nan")),
            "https://github.com/pytorch/pytorch/issues",
        )

    def test_find_related_by_text_schema(self):
        """Test standard JSON response schema for natural language query."""
        res = self.pipeline.find_related_by_text(
            title="Syntax highlighting error in Python extension",
            body="Keywords are not highlighted properly in dark mode",
            repo_name="microsoft/vscode",
            top_k=3,
            min_confidence="ALL",
        )

        self.assertIn("query", res)
        self.assertIn("summary", res)
        self.assertIn("related_issues", res)
        self.assertIn("execution_metadata", res)

        self.assertEqual(res["query"]["query_type"], "text")
        self.assertEqual(res["query"]["repo_scope"], "microsoft/vscode")
        self.assertGreaterEqual(len(res["related_issues"]), 1)

        # Check candidate structure
        first = res["related_issues"][0]
        expected_keys = [
            "rank", "issue_id", "issue_number", "repo_name",
            "title", "clean_title", "canonical_label", "state",
            "created_at", "html_url", "similarity_score",
            "confidence_tier", "tier_label", "confidence",
            "action_recommendation", "is_potential_duplicate",
        ]
        for k in expected_keys:
            self.assertIn(k, first)

        self.assertTrue(first["html_url"].startswith("https://github.com/microsoft/vscode/issues/"))

    def test_find_related_by_id_self_exclusion(self):
        """Test query by ID strictly excludes the query issue from results."""
        query_id = 3000
        res = self.pipeline.find_related_by_id(
            issue_id=query_id,
            top_k=5,
            min_confidence="ALL",
        )

        self.assertEqual(res["query"]["query_type"], "issue_id")
        self.assertEqual(res["query"]["issue_id"], query_id)

        returned_ids = [c["issue_id"] for c in res["related_issues"]]
        self.assertNotIn(query_id, returned_ids)

    def test_confidence_threshold_filtering(self):
        """Test filtering candidates by minimum confidence tier."""
        # ALL returns items even if weak
        res_all = self.pipeline.find_related_by_text(
            title="Generic query",
            top_k=5,
            min_confidence="ALL",
        )
        # HIGH requires >= 0.70
        res_high = self.pipeline.find_related_by_text(
            title="Generic query",
            top_k=5,
            min_confidence="HIGH",
        )

        for c in res_high["related_issues"]:
            self.assertEqual(c["confidence_tier"], "HIGH")
            self.assertGreaterEqual(c["similarity_score"], 0.70)

        self.assertLessEqual(len(res_high["related_issues"]), len(res_all["related_issues"]))

    def test_duplicate_flag_logic(self):
        """Test that has_potential_duplicate is correctly flagged."""
        # Query matching vector 0 exactly (score = 1.0)
        # Use vector 0 as text hash
        vec0 = self.store.embeddings[0]
        # Query retriever directly to test formatting
        raw = self.pipeline.retriever.retrieve_by_vector(vec0, top_k=2)
        payload = self.pipeline._format_payload(
            query_info={"query_type": "vector"},
            raw_candidates=raw,
            latency_ms=1.5,
            min_confidence="LOW",
        )

        self.assertTrue(payload["summary"]["has_potential_duplicate"])
        self.assertIn("Potential duplicate detected", payload["summary"]["primary_recommendation"])

    def test_empty_candidates_handling(self):
        """Test payload when no candidates pass threshold."""
        payload = self.pipeline._format_payload(
            query_info={"query_type": "text"},
            raw_candidates=[],
            latency_ms=0.5,
            min_confidence="HIGH",
        )

        self.assertEqual(payload["summary"]["total_candidates_found"], 0)
        self.assertFalse(payload["summary"]["has_potential_duplicate"])
        self.assertEqual(payload["summary"]["highest_similarity_score"], 0.0)
        self.assertIn("No related issues found", payload["summary"]["primary_recommendation"])


if __name__ == "__main__":
    unittest.main()
