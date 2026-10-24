"""
Unit tests for RetrievalEvaluator and LexicalRetrievalBaseline (Phase 7, Step 7.6).
"""

from pathlib import Path
import shutil
import tempfile
import unittest

import pandas as pd

from ml.related_issues.evaluator import (
    LexicalRetrievalBaseline,
    RetrievalEvaluator,
    compute_hit_rate_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
)


class TestRetrievalEvaluator(unittest.TestCase):
    """Test suite for IR evaluation metrics, ground-truth mining, and lexical baseline."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.evaluator = RetrievalEvaluator()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_compute_mrr(self):
        """Test Mean Reciprocal Rank calculation."""
        ranks = [1, 2, 4, None]
        expected_mrr = (1.0/1.0 + 1.0/2.0 + 1.0/4.0 + 0.0) / 4.0  # 1.75 / 4 = 0.4375
        self.assertAlmostEqual(compute_mrr(ranks), expected_mrr, places=4)
        self.assertEqual(compute_mrr([]), 0.0)
        self.assertEqual(compute_mrr([None, None]), 0.0)

    def test_compute_hit_rate_at_k(self):
        """Test Hit Rate @ K (Recall @ K) across various K thresholds."""
        ranks = [1, 3, 5, 8, None]
        self.assertAlmostEqual(compute_hit_rate_at_k(ranks, k=1), 1.0 / 5.0, places=4)
        self.assertAlmostEqual(compute_hit_rate_at_k(ranks, k=3), 2.0 / 5.0, places=4)
        self.assertAlmostEqual(compute_hit_rate_at_k(ranks, k=5), 3.0 / 5.0, places=4)
        self.assertAlmostEqual(compute_hit_rate_at_k(ranks, k=10), 4.0 / 5.0, places=4)
        self.assertEqual(compute_hit_rate_at_k([], k=5), 0.0)

    def test_compute_ndcg_at_k(self):
        """Test position-discounted NDCG @ K."""
        # Rank 1 yields ideal NDCG = 1.0
        self.assertAlmostEqual(compute_ndcg_at_k([1], k=1), 1.0, places=4)
        self.assertAlmostEqual(compute_ndcg_at_k([1], k=5), 1.0, places=4)

        # Rank 2 discount: 1 / log2(3)
        self.assertAlmostEqual(compute_ndcg_at_k([2], k=1), 0.0, places=4)  # Rank 2 is outside k=1
        self.assertGreater(compute_ndcg_at_k([2], k=3), 0.0)
        self.assertLess(compute_ndcg_at_k([2], k=3), 1.0)

    def test_compute_precision_at_k(self):
        """Test single-target Precision @ K."""
        ranks = [1, 2, None]
        # At k=1: only first query is a hit -> 1/1, others 0 -> mean = 1/3
        self.assertAlmostEqual(compute_precision_at_k(ranks, k=1), 1.0 / 3.0, places=4)
        # At k=2: first two are hits (each 1/2), third 0 -> mean = (0.5 + 0.5 + 0)/3 = 1/3
        self.assertAlmostEqual(compute_precision_at_k(ranks, k=2), 1.0 / 3.0, places=4)

    def test_lexical_retrieval_baseline(self):
        """Test fitting and keyword retrieval of TF-IDF baseline."""
        dummy_df = pd.DataFrame({
            "issue_id": [10, 20, 30],
            "issue_number": [101, 102, 103],
            "repo_name": ["repo_x", "repo_x", "repo_y"],
            "title": [
                "CUDA out of memory in backprop",
                "TypeScript extension crash on save",
                "Kubernetes pod scheduling failure",
            ],
            "clean_title": [
                "cuda out of memory in backprop",
                "typescript extension crash on save",
                "kubernetes pod scheduling failure",
            ],
            "clean_body": [
                "allocation error on GPU device",
                "editor hangs when saving large file",
                "node affinity rejected deployment",
            ],
        })

        baseline = LexicalRetrievalBaseline()
        baseline.fit(dummy_df)

        # Retrieve CUDA query
        results = baseline.retrieve("cuda memory gpu", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["issue_id"], 10)
        self.assertGreater(results[0]["similarity_score"], 0.0)

        # Exclusion test
        results_ex = baseline.retrieve("cuda memory gpu", top_k=2, exclude_issue_ids={10})
        for r in results_ex:
            self.assertNotEqual(r["issue_id"], 10)

    def test_mine_linked_reference_pairs(self):
        """Test mining intra-repo cross-references."""
        df = pd.DataFrame({
            "issue_id": [100, 200, 300],
            "issue_number": [1, 2, 3],
            "repo_name": ["repo_test", "repo_test", "repo_other"],
            "title": ["Bug in parser", "Duplicate of #1 in parser", "Unrelated issue"],
            "clean_title": ["bug in parser", "duplicate of 1 in parser", "unrelated issue"],
            "body": ["Some details", "See issue #1 for repro steps", "Details"],
            "clean_body": ["some details", "see issue 1 for repro steps", "details"],
        })

        pairs = self.evaluator.mine_linked_reference_pairs(df)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["query_id"], 200)
        self.assertEqual(pairs[0]["target_id"], 100)
        self.assertEqual(pairs[0]["target_number"], 1)

    def test_create_title_to_corpus_evaluation_set(self):
        """Test creating title perturbation evaluation set."""
        df = pd.DataFrame({
            "issue_id": [10, 20, 30],
            "issue_number": [1, 2, 3],
            "repo_name": ["r1", "r1", "r2"],
            "title": ["Title A", "Title B", "Title C"],
            "clean_title": ["title a", "title b", "title c"],
            "clean_body": ["body a", "body b", "body c"],
        })

        eval_set = self.evaluator.create_title_to_corpus_evaluation_set(df, n_queries=2, random_seed=42)
        self.assertEqual(len(eval_set), 2)
        self.assertIn("query_title", eval_set[0])
        self.assertIn("target_id", eval_set[0])
        self.assertEqual(eval_set[0]["evaluation_type"], "title_perturbation")

    def test_plot_comparative_benchmark(self):
        """Test comparative diagnostic plot generation."""
        fig_path = Path(self.temp_dir) / "test_fig28.png"
        neural_summary = {
            "mrr": 0.725,
            "hit_rate_at_1": 0.65,
            "hit_rate_at_3": 0.80,
            "hit_rate_at_5": 0.88,
            "hit_rate_at_10": 0.94,
            "ndcg_at_5": 0.765,
        }
        lexical_summary = {
            "mrr": 0.450,
            "hit_rate_at_1": 0.38,
            "hit_rate_at_3": 0.52,
            "hit_rate_at_5": 0.60,
            "hit_rate_at_10": 0.68,
            "ndcg_at_5": 0.495,
        }

        out = self.evaluator.plot_comparative_benchmark(
            neural_summary=neural_summary,
            lexical_summary=lexical_summary,
            save_path=fig_path,
        )
        self.assertTrue(out.exists())
        self.assertGreater(out.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
