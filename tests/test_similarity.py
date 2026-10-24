"""
Unit tests for SimilarityEngine (Phase 7, Step 7.3).
"""

from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from ml.related_issues.similarity import SimilarityEngine
from ml.related_issues.store import IssueEmbeddingStore


class TestSimilarityEngine(unittest.TestCase):
    """Test suite for cosine similarity calculations, repo filtering, and distributions."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.engine = SimilarityEngine()

        # Deterministic test vectors
        np.random.seed(42)
        # 384-dimensional vectors
        self.v_a = np.zeros(384, dtype=np.float32)
        self.v_a[0] = 1.0  # Unit vector along axis 0

        self.v_b = np.zeros(384, dtype=np.float32)
        self.v_b[0] = 1.0  # Identical to v_a

        self.v_c = np.zeros(384, dtype=np.float32)
        self.v_c[1] = 1.0  # Orthogonal to v_a

        self.v_d = np.zeros(384, dtype=np.float32)
        self.v_d[0] = -1.0  # Opposite to v_a

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_normalize_vectors(self):
        """Test L2 normalization on 1D and 2D arrays."""
        unnorm_1d = np.array([3.0, 4.0], dtype=np.float32)
        norm_1d = self.engine.normalize_vectors(unnorm_1d)
        np.testing.assert_allclose(norm_1d, np.array([0.6, 0.8], dtype=np.float32), atol=1e-5)
        self.assertAlmostEqual(np.linalg.norm(norm_1d), 1.0, places=5)

        unnorm_2d = np.array([[3.0, 4.0], [1.0, 1.0]], dtype=np.float32)
        norm_2d = self.engine.normalize_vectors(unnorm_2d)
        self.assertEqual(norm_2d.shape, (2, 2))
        np.testing.assert_allclose(np.linalg.norm(norm_2d, axis=1), np.ones(2), atol=1e-5)

    def test_cosine_similarity_properties(self):
        """Test fundamental mathematical properties: identical, orthogonal, opposite."""
        corpus = np.stack([self.v_b, self.v_c, self.v_d])  # (3, 384)
        sims = self.engine.cosine_similarity(self.v_a, corpus)

        self.assertEqual(sims.shape, (3,))
        self.assertAlmostEqual(sims[0], 1.0, places=5)   # Identical
        self.assertAlmostEqual(sims[1], 0.0, places=5)   # Orthogonal
        self.assertAlmostEqual(sims[2], -1.0, places=5)  # Opposite

    def test_pairwise_similarity(self):
        """Test pairwise scalar similarity."""
        self.assertAlmostEqual(self.engine.pairwise_similarity(self.v_a, self.v_b), 1.0, places=5)
        self.assertAlmostEqual(self.engine.pairwise_similarity(self.v_a, self.v_c), 0.0, places=5)
        self.assertAlmostEqual(self.engine.pairwise_similarity(self.v_a, self.v_d), -1.0, places=5)

    def test_batch_cosine_similarity(self):
        """Test batch matrix multiplication (M, D) x (N, D) -> (M, N)."""
        queries = np.stack([self.v_a, self.v_c])           # (2, 384)
        corpus = np.stack([self.v_b, self.v_c, self.v_d])  # (3, 384)

        batch_sims = self.engine.batch_cosine_similarity(queries, corpus)
        self.assertEqual(batch_sims.shape, (2, 3))
        # Query 0 (v_a) vs corpus
        np.testing.assert_allclose(batch_sims[0], [1.0, 0.0, -1.0], atol=1e-5)
        # Query 1 (v_c) vs corpus
        np.testing.assert_allclose(batch_sims[1], [0.0, 1.0, 0.0], atol=1e-5)

    def test_unnormalized_inputs_handling(self):
        """Test that unnormalized vectors are handled cleanly with assume_normalized=False."""
        unnorm_q = self.v_a * 5.5
        unnorm_corpus = np.stack([self.v_b * 2.0, self.v_c * 10.0, self.v_d * 0.5])

        sims = self.engine.cosine_similarity(unnorm_q, unnorm_corpus, assume_normalized=False)
        np.testing.assert_allclose(sims, [1.0, 0.0, -1.0], atol=1e-5)

    def test_dimension_mismatch_exception(self):
        """Test dimension mismatch raises ValueError."""
        wrong_dim_q = np.ones(50, dtype=np.float32)
        corpus = np.ones((10, 384), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.engine.cosine_similarity(wrong_dim_q, corpus)

    def test_score_corpus_with_store(self):
        """Test scoring against an IssueEmbeddingStore instance with repo filtering."""
        store_dir = Path(self.temp_dir) / "test_store"
        store = IssueEmbeddingStore(store_dir)

        dummy_embs = np.random.randn(15, 384).astype(np.float32)
        dummy_embs = dummy_embs / np.linalg.norm(dummy_embs, axis=1, keepdims=True)

        dummy_df = pd.DataFrame({
            "issue_id": list(range(15)),
            "repo_name": ["repo_1"] * 10 + ["repo_2"] * 5,
            "title": [f"Title {i}" for i in range(15)],
        })
        store.save(dummy_embs, dummy_df)

        # Global score
        q = dummy_embs[0]
        scores_all, meta_all = self.engine.score_corpus(q, store)
        self.assertEqual(len(scores_all), 15)
        self.assertEqual(len(meta_all), 15)
        self.assertAlmostEqual(scores_all[0], 1.0, places=4)

        # Repo-filtered score
        scores_sub, meta_sub = self.engine.score_corpus(q, store, repo_filter="repo_2")
        self.assertEqual(len(scores_sub), 5)
        self.assertEqual(len(meta_sub), 5)
        self.assertTrue((meta_sub["repo_name"] == "repo_2").all())

    def test_intra_vs_inter_repo_analysis(self):
        """Test distribution analysis and statistical testing logic."""
        store_dir = Path(self.temp_dir) / "test_store_dist"
        store = IssueEmbeddingStore(store_dir)

        # Make repo_A clustered near [1, 0, ...] and repo_B clustered near [0, 1, ...]
        np.random.seed(42)
        cluster_a = np.zeros((20, 384), dtype=np.float32)
        cluster_a[:, 0] = 1.0
        cluster_a += np.random.normal(0, 0.05, (20, 384))
        cluster_a /= np.linalg.norm(cluster_a, axis=1, keepdims=True)

        cluster_b = np.zeros((20, 384), dtype=np.float32)
        cluster_b[:, 1] = 1.0
        cluster_b += np.random.normal(0, 0.05, (20, 384))
        cluster_b /= np.linalg.norm(cluster_b, axis=1, keepdims=True)

        all_embs = np.vstack([cluster_a, cluster_b])
        all_df = pd.DataFrame({
            "issue_id": list(range(40)),
            "repo_name": ["repo_a"] * 20 + ["repo_b"] * 20,
            "title": [f"Issue {i}" for i in range(40)],
        })
        store.save(all_embs, all_df)

        analysis = self.engine.analyze_intra_vs_inter_repo_distributions(store, sample_size=40)
        self.assertIn("intra_repo_stats", analysis)
        self.assertIn("inter_repo_stats", analysis)
        self.assertGreater(analysis["separation_margin"], 0.0)
        self.assertGreater(analysis["intra_repo_stats"]["mean"], analysis["inter_repo_stats"]["mean"])
        self.assertIn("cohens_d", analysis)
        self.assertIn("latency_scaling_us", analysis)

    def test_plot_similarity_diagnostics(self):
        """Test generating diagnostic plot fig25."""
        fig_path = Path(self.temp_dir) / "test_fig25.png"
        intra = np.random.normal(0.45, 0.1, 1000)
        inter = np.random.normal(0.20, 0.1, 3000)
        latency = {100: 15.2, 500: 42.1, 1000: 81.3, 2328: 165.4}

        out = self.engine.plot_similarity_diagnostics(intra, inter, latency, save_path=fig_path)
        self.assertTrue(out.exists())
        self.assertGreater(out.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
