"""Integration Test Suite for Semantic Similarity & Related Issues Pipeline (Phase 11, Step 11.5).

Verifies the complete integration of the dense Sentence-Transformers embedding retrieval system:
1. Dynamic semantic search via POST /api/v1/related with all-MiniLM-L6-v2 embeddings.
2. Cosine similarity score validation and descending rank ordering.
3. Repository-scoped candidate filtering (e.g. repo_name="pytorch/pytorch").
4. State filtering (open vs. closed candidate issues).
5. Combined multi-criteria filtering (repository + issue state).
6. Parameter bounds for top_k candidate slicing (1 to 20).
7. Relational edge cache retrieval via GET /api/v1/related/{issue_id} (O(1) database lookup).
8. Boundary 404 handling for non-existent issue identifiers.
9. Model registry introspection for sentence_transformer subsystem.
"""

from __future__ import annotations

import unittest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.models import Issue, RelatedIssuesCache
from backend.app.database.session import SessionLocal
from backend.app.main import app


class TestSimilarityPipelineIntegration(unittest.TestCase):
    """Integration test suite for dense sentence embedding retrieval and relational cache."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Prewarm models to prime memory and embedder
        cls.client.post("/api/v1/models/prewarm")

    def setUp(self):
        self.db: Session = SessionLocal()

    def tearDown(self):
        self.db.close()

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Dynamic Semantic Similarity Search
    # ──────────────────────────────────────────────────────────────────────────

    def test_dynamic_similarity_search_contract(self):
        """Verify POST /api/v1/related returns ranked candidate matches with cosine scores."""
        payload = {
            "title": "CUDA out of memory error during backward pass",
            "body": "When executing matrix multiply on Ampere GPU, memory spikes and throws CUDA OOM.",
            "top_k": 5,
        }
        response = self.client.post("/api/v1/related", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["model_version"], "all-MiniLM-L6-v2")
        self.assertEqual(data["total_matches"], 5)
        self.assertEqual(len(data["matches"]), 5)
        self.assertGreater(data["retrieval_time_ms"], 0.0)

        # Verify score ordering and bounds
        scores = [m["similarity_score"] for m in data["matches"]]
        for score in scores:
            self.assertGreaterEqual(score, -1.0)
            self.assertLessEqual(score, 1.0)

        # Check descending order
        self.assertEqual(scores, sorted(scores, reverse=True))

        # Check match fields
        for idx, match in enumerate(data["matches"], start=1):
            self.assertEqual(match["rank"], idx)
            self.assertIn("issue_id", match)
            self.assertIn("repo_name", match)
            self.assertIn("issue_number", match)
            self.assertIn("title", match)
            self.assertIn(match["state"], ["open", "closed"])
            self.assertIn("canonical_label", match)

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Candidate Pool Scoping & Filtering
    # ──────────────────────────────────────────────────────────────────────────

    def test_repository_scoped_similarity_search(self):
        """Verify repo_name filter restricts candidates strictly to the target repository."""
        target_repo = "pytorch/pytorch"
        payload = {
            "title": "Segmentation fault in tensor dispatch kernel",
            "repo_name": target_repo,
            "top_k": 4,
        }
        response = self.client.post("/api/v1/related", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(len(data["matches"]), 4)
        for match in data["matches"]:
            self.assertEqual(match["repo_name"], target_repo)

    def test_state_filtered_similarity_search(self):
        """Verify state filter restricts candidates to either open or closed issues."""
        # Test closed filter
        closed_payload = {
            "title": "Terminal process crashes with exit code 139",
            "state": "closed",
            "top_k": 4,
        }
        closed_res = self.client.post("/api/v1/related", json=closed_payload)
        self.assertEqual(closed_res.status_code, 200)
        for match in closed_res.json()["matches"]:
            self.assertEqual(match["state"], "closed")

        # Test open filter
        open_payload = {
            "title": "Terminal process crashes with exit code 139",
            "state": "open",
            "top_k": 4,
        }
        open_res = self.client.post("/api/v1/related", json=open_payload)
        self.assertEqual(open_res.status_code, 200)
        for match in open_res.json()["matches"]:
            self.assertEqual(match["state"], "open")

    def test_combined_repo_and_state_filtering(self):
        """Verify combining repo_name and state filters applies simultaneous constraints."""
        payload = {
            "title": "Ingress controller routing deadlock under heavy traffic",
            "repo_name": "kubernetes/kubernetes",
            "state": "closed",
            "top_k": 3,
        }
        response = self.client.post("/api/v1/related", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        for match in data["matches"]:
            self.assertEqual(match["repo_name"], "kubernetes/kubernetes")
            self.assertEqual(match["state"], "closed")

    def test_top_k_parameter_slicing(self):
        """Verify requested top_k parameter governs the exact candidate count returned."""
        for k in [2, 5, 8]:
            payload = {
                "title": "Memory leak during multi-worker processing",
                "top_k": k,
            }
            res = self.client.post("/api/v1/related", json=payload)
            self.assertEqual(res.status_code, 200)
            self.assertEqual(len(res.json()["matches"]), k)

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Relational Edge Cache Retrieval (O(1))
    # ──────────────────────────────────────────────────────────────────────────

    def test_cached_related_issues_lookup(self):
        """Verify GET /api/v1/related/{issue_id} fetches precomputed edges from database cache."""
        # Find a valid source issue that has precomputed edges
        cached_edge = self.db.scalar(select(RelatedIssuesCache))
        self.assertIsNotNone(cached_edge, "Database should contain precomputed related issue edges")

        source_id = cached_edge.source_issue_id
        response = self.client.get(f"/api/v1/related/{source_id}")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["query_issue_id"], source_id)
        self.assertGreater(data["total_matches"], 0)
        self.assertLessEqual(data["total_matches"], 5)
        self.assertLess(data["retrieval_time_ms"], 100.0)  # O(1) database lookup

        scores = [m["similarity_score"] for m in data["matches"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_cached_lookup_nonexistent_issue_returns_404(self):
        """Verify requesting cached related issues for a non-existent issue ID returns 404."""
        response = self.client.get("/api/v1/related/999999999999")
        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertIn("detail", data)

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Model Registry Metadata Contract
    # ──────────────────────────────────────────────────────────────────────────

    def test_model_registry_sentence_transformer_metadata(self):
        """Verify GET /api/v1/models/sentence_transformer returns dense retriever metadata."""
        response = self.client.get("/api/v1/models/sentence_transformer")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["model_id"], "sentence_transformer")
        self.assertEqual(data["task_type"], "semantic_retrieval")
        self.assertEqual(data["status"], "ready")
        self.assertEqual(data["details"]["embedding_dimension"], 384)
        self.assertEqual(data["details"]["cached_embeddings_count"], 2328)
        self.assertTrue(data["details"]["normalized"])


if __name__ == "__main__":
    unittest.main()
