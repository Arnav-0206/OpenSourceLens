"""
OpenSourceLens — Unit Tests for Resolution Detailed Model Evaluation (Phase 8, Step 8.6).

Validates residual diagnostics, normality tests, Breusch-Pagan heteroskedasticity,
subgroup disaggregation across repositories and categories, ordinal classification bounds,
Figure 33 generation, and diagnostics manifest export.
"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ml.resolution_analysis.evaluation import ResolutionEvaluator


class TestResolutionEvaluator(unittest.TestCase):
    """Test suite for ResolutionEvaluator."""

    @classmethod
    def setUpClass(cls):
        """Initializes the evaluator and loads holdout test data and models once."""
        cls.evaluator = ResolutionEvaluator()
        cls.df_test = cls.evaluator.load_test_data_and_models(test_size=0.2, random_state=42)
        cls.evaluator.evaluate_residuals()
        cls.evaluator.evaluate_subgroups()
        cls.evaluator.evaluate_ordinal_classification()

    def test_evaluator_initialization(self):
        """Verifies test set alignment, sample count, and model loading."""
        self.assertIsNotNone(self.evaluator.df_test)
        self.assertEqual(len(self.evaluator.df_test), 227)
        self.assertIsNotNone(self.evaluator.regressor_model)
        self.assertIsNotNone(self.evaluator.classifier_model)
        self.assertEqual(len(self.evaluator.reg_preds), 227)
        self.assertEqual(len(self.evaluator.clf_preds), 227)
        self.assertEqual(len(self.evaluator.residuals), 227)

    def test_residual_diagnostics_metrics(self):
        """Validates residual summary statistics and hypothesis tests."""
        diag = self.evaluator.residual_diagnostics
        self.assertIn("mean", diag)
        self.assertIn("std", diag)
        self.assertIn("skewness", diag)
        self.assertIn("excess_kurtosis", diag)
        self.assertIn("normality_omnibus", diag)
        self.assertIn("breusch_pagan_heteroskedasticity", diag)

        self.assertFalse(np.isnan(diag["mean"]))
        self.assertFalse(np.isnan(diag["std"]))
        self.assertGreater(diag["std"], 0.0)

        # Normality and heteroskedasticity p-values must be valid probabilities
        self.assertGreaterEqual(diag["normality_omnibus"]["p_value"], 0.0)
        self.assertLessEqual(diag["normality_omnibus"]["p_value"], 1.0)
        self.assertGreaterEqual(diag["breusch_pagan_heteroskedasticity"]["p_value"], 0.0)
        self.assertLessEqual(diag["breusch_pagan_heteroskedasticity"]["p_value"], 1.0)

    def test_extreme_cases_extraction(self):
        """Validates that Top 5 under and over predictions are extracted with valid metadata."""
        diag = self.evaluator.residual_diagnostics
        self.assertIn("top_under_predictions", diag)
        self.assertIn("top_over_predictions", diag)
        self.assertEqual(len(diag["top_under_predictions"]), 5)
        self.assertEqual(len(diag["top_over_predictions"]), 5)

        for case in diag["top_under_predictions"]:
            self.assertGreater(case["residual_log"], 0.0)
            self.assertGreater(len(case["title"]), 0)
            self.assertIn(case["repo_name"], ["microsoft/vscode", "pytorch/pytorch", "kubernetes/kubernetes", "huggingface/transformers"])

        for case in diag["top_over_predictions"]:
            self.assertLess(case["residual_log"], 0.0)
            self.assertGreater(len(case["title"]), 0)

    def test_subgroup_evaluation_coverage(self):
        """Verifies evaluation across all 4 repositories and major categories."""
        subgroups = self.evaluator.subgroup_results
        self.assertIn("by_repository", subgroups)
        self.assertIn("by_canonical_label", subgroups)

        repos = subgroups["by_repository"]
        self.assertEqual(len(repos), 4)
        for repo_name in ["microsoft/vscode", "pytorch/pytorch", "kubernetes/kubernetes", "huggingface/transformers"]:
            self.assertIn(repo_name, repos)
            self.assertGreater(repos[repo_name]["n_samples"], 30)
            self.assertGreaterEqual(repos[repo_name]["test_accuracy"], 0.0)
            self.assertLessEqual(repos[repo_name]["test_accuracy"], 1.0)

    def test_ordinal_metrics_bounds(self):
        """Validates ordinal accuracy bounds (Exact <= Within 1 tier <= 1.0)."""
        ord_res = self.evaluator.ordinal_results
        exact = ord_res["exact_accuracy"]
        adjacent = ord_res["adjacent_accuracy_within_1_tier"]
        severe = ord_res["severe_error_rate_ge_2_tiers"]
        extreme = ord_res["extreme_error_rate_3_tiers"]

        self.assertGreaterEqual(exact, 0.0)
        self.assertLessEqual(exact, 1.0)
        self.assertGreaterEqual(adjacent, exact)
        self.assertLessEqual(adjacent, 1.0)
        self.assertAlmostEqual(adjacent + severe, 1.0, places=4)
        self.assertLessEqual(extreme, severe)
        self.assertGreater(ord_res["adjacent_accuracy_within_1_tier"], 0.75)  # > 75% within 1 tier

    def test_weighted_kappa_bounds(self):
        """Validates linear and quadratic weighted Cohen's Kappa bounds."""
        ord_res = self.evaluator.ordinal_results
        lin_kappa = ord_res["linear_weighted_cohen_kappa"]
        quad_kappa = ord_res["quadratic_weighted_cohen_kappa"]

        self.assertGreaterEqual(lin_kappa, -1.0)
        self.assertLessEqual(lin_kappa, 1.0)
        self.assertGreaterEqual(quad_kappa, -1.0)
        self.assertLessEqual(quad_kappa, 1.0)
        self.assertGreater(lin_kappa, 0.20)  # Positive ordinal agreement

    def test_diagnostics_manifest_export(self):
        """Verifies JSON diagnostics manifest export and validity."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_diag.json"
            exported = self.evaluator.export_diagnostics_manifest(out_file)
            self.assertTrue(exported.exists())

            with open(exported, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("metadata", data)
            self.assertIn("residual_diagnostics", data)
            self.assertIn("subgroup_performance", data)
            self.assertIn("ordinal_classification_analysis", data)

    def test_figure_generation(self):
        """Verifies Figure 33 4-panel generation and file size."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_plot = Path(tmp_dir) / "test_fig33.png"
            gen_plot = self.evaluator.generate_figures(out_plot)
            self.assertTrue(gen_plot.exists())
            self.assertGreater(gen_plot.stat().st_size, 50000)


if __name__ == "__main__":
    unittest.main()
