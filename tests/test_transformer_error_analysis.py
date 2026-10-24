"""Unit Tests for Transformer Error Analysis Pipeline (Step 6.7)."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from ml.deep_learning.error_analyzer import TransformerErrorAnalyzer


class TestTransformerErrorAnalysis(unittest.TestCase):
    """Test suite for TransformerErrorAnalyzer, diagnostic metrics, and export generation."""

    @classmethod
    def setUpClass(cls):
        """Set up analyzer with canonical test artifacts."""
        cls.analyzer = TransformerErrorAnalyzer(
            predictions_path="data/processed/transformer/test_predictions.csv",
            splits_path="data/processed/splits/test.csv",
            classical_model_path="models/final_classical_nlp_pipeline.joblib",
        )
        cls.analyzer.load_data()

    def test_data_loading_and_error_counts(self):
        """Verify test predictions loading and exact misclassification count."""
        self.assertEqual(len(self.analyzer.df_all), 385)
        self.assertEqual(len(self.analyzer.df_errors), 95)
        self.assertEqual(int((~self.analyzer.df_all["is_correct"]).sum()), 95)
        self.assertEqual(int(self.analyzer.df_all["is_correct"].sum()), 290)

    def test_failure_pair_taxonomy(self):
        """Verify confusion pair taxonomy counts and high-level groupings."""
        tax = self.analyzer.compute_failure_taxonomy()
        self.assertEqual(tax["total_test_samples"], 385)
        self.assertEqual(tax["total_errors"], 95)
        self.assertEqual(tax["overall_error_rate_pct"], 24.68)

        pairs = tax["pair_taxonomy"]
        self.assertIn("other -> bug", pairs)
        self.assertIn("bug -> other", pairs)
        self.assertIn("other -> feature", pairs)
        self.assertIn("feature -> other", pairs)
        self.assertIn("bug -> feature", pairs)
        self.assertIn("feature -> bug", pairs)

        # Exact pair counts
        self.assertEqual(pairs["other -> bug"]["count"], 34)
        self.assertEqual(pairs["bug -> other"]["count"], 31)
        self.assertEqual(pairs["other -> feature"]["count"], 15)
        self.assertEqual(pairs["feature -> other"]["count"], 12)
        self.assertEqual(pairs["bug -> feature"]["count"], 2)
        self.assertEqual(pairs["feature -> bug"]["count"], 1)

        # High level grouping (92 other-spillover, 3 direct bug-feature)
        grouping = tax["high_level_grouping"]
        self.assertEqual(grouping["other_class_spillover_count"], 92)
        self.assertEqual(grouping["direct_bug_feature_confusion_count"], 3)

    def test_length_and_truncation_impact(self):
        """Verify word count quartile trends and truncation analysis."""
        length_stats = self.analyzer.compute_length_and_truncation_impact()
        quartiles = length_stats["length_quartile_analysis"]
        self.assertEqual(len(quartiles), 4)

        # Errors should have higher mean/median word count than correct samples
        self.assertGreater(length_stats["mean_word_count_errors"], length_stats["mean_word_count_correct"])
        self.assertGreater(length_stats["median_word_count_errors"], length_stats["median_word_count_correct"])

        # Q4 error rate is higher than Q1
        q1_rate = quartiles["Q1 (Short)"]["error_rate_pct"]
        q4_rate = quartiles["Q4 (Long)"]["error_rate_pct"]
        self.assertGreater(q4_rate, q1_rate)

    def test_repository_breakdown(self):
        """Verify per-repository sample counts and error distribution."""
        repo_stats = self.analyzer.compute_repository_breakdown()
        self.assertEqual(len(repo_stats), 4)
        total_samples = sum(s["sample_count"] for s in repo_stats.values())
        total_errors = sum(s["error_count"] for s in repo_stats.values())
        self.assertEqual(total_samples, 385)
        self.assertEqual(total_errors, 95)

        # transformers has lowest error rate
        self.assertLess(repo_stats["huggingface/transformers"]["error_rate_pct"], 10.0)

    def test_confidence_diagnostics(self):
        """Verify confidence distribution, high-confidence error count, and calibration."""
        conf = self.analyzer.compute_confidence_diagnostics()
        self.assertGreater(conf["mean_confidence_correct"], conf["mean_confidence_errors"])
        self.assertEqual(conf["high_confidence_error_count"], 27)
        self.assertGreaterEqual(len(conf["high_confidence_examples"]), 1)

    def test_cross_model_intersection(self):
        """Verify cross-model counts against Classical ML."""
        cross = self.analyzer.compute_cross_model_intersection()
        self.assertEqual(cross["both_failed_count"], 55)
        self.assertEqual(cross["transformer_unique_wins_count"], 16)
        self.assertEqual(cross["classical_unique_wins_count"], 40)

    def test_exports_and_visualizations(self):
        """Verify misclassified CSV, JSON report, and 4-panel diagnostic plot export."""
        self.analyzer.classify_error_taxonomy()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fig_path = tmp_path / "test_fig23.png"
            csv_path = tmp_path / "test_errors.csv"
            json_path = tmp_path / "test_analysis.json"

            # Test plot export
            self.analyzer.plot_error_distributions(save_path=fig_path)
            self.assertTrue(fig_path.exists())
            self.assertGreater(fig_path.stat().st_size, 1000)

            # Test CSV export
            self.analyzer.save_misclassified_csv(save_path=csv_path)
            self.assertTrue(csv_path.exists())
            df_err = pd.read_csv(csv_path)
            self.assertEqual(len(df_err), 95)
            self.assertIn("error_category", df_err.columns)

            # Test JSON export
            self.analyzer.save_error_analysis_json(save_path=json_path)
            self.assertTrue(json_path.exists())
            with open(json_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertIn("failure_pair_taxonomy", loaded)
            self.assertIn("key_takeaways", loaded)


if __name__ == "__main__":
    unittest.main()
