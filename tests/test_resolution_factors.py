"""
OpenSourceLens — Unit Tests for Resolution Factor Investigation (Phase 8, Step 8.4).

Validates multivariate OLS regression modeling, VIF multicollinearity diagnostics,
ANOVA factor family variance decomposition, interaction testing, figure generation,
and manifest export.
"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from ml.resolution_analysis.factors import (
    FACTOR_FAMILIES,
    ResolutionFactorAnalyzer,
)


class TestResolutionFactorAnalyzer(unittest.TestCase):
    """Test suite for ResolutionFactorAnalyzer."""

    @classmethod
    def setUpClass(cls):
        """Initializes the analyzer with production datasets."""
        cls.analyzer = ResolutionFactorAnalyzer()
        cls.df = cls.analyzer.load_data()

    def test_factor_analyzer_initialization(self):
        """Verifies dataset loading, matrix dimensions, and feature alignment."""
        self.assertIsNotNone(self.analyzer.df_features)
        self.assertGreater(len(self.analyzer.df_features), 500)
        self.assertIsNotNone(self.analyzer.X_matrix)
        self.assertIsNotNone(self.analyzer.y_vector)
        self.assertEqual(len(self.analyzer.X_matrix), len(self.analyzer.y_vector))
        self.assertGreater(len(self.analyzer.feature_names), 15)
        # Ensure redundant author association dummies were excluded to prevent dummy variable trap
        for col in self.analyzer.feature_names:
            self.assertFalse(col.startswith("author_association_"))

    def test_ols_fit_metrics_bounds(self):
        """Validates R², Adjusted R², F-statistic, and degrees of freedom."""
        res = self.analyzer.fit_ols()
        self.assertIn("r_squared", res)
        self.assertIn("adj_r_squared", res)
        self.assertIn("f_statistic", res)
        self.assertIn("coefficients", res)

        r2 = res["r_squared"]
        adj_r2 = res["adj_r_squared"]
        self.assertGreaterEqual(r2, 0.0)
        self.assertLessEqual(r2, 1.0)
        self.assertLessEqual(adj_r2, r2)
        self.assertGreater(res["f_statistic"], 0.0)
        self.assertGreater(res["rss"], 0.0)
        self.assertGreater(res["tss"], res["rss"])
        self.assertEqual(res["n_samples"], len(self.analyzer.df_features))

    def test_coefficients_and_standard_errors(self):
        """Validates that all coefficients and standard errors are finite and SE > 0."""
        res = self.analyzer.fit_ols()
        coefs = res["coefficients"]
        self.assertEqual(len(coefs), len(self.analyzer.feature_names) + 1)  # +1 for intercept

        for c in coefs:
            self.assertFalse(np.isnan(c["coefficient"]))
            self.assertFalse(np.isinf(c["coefficient"]))
            self.assertFalse(np.isnan(c["std_error"]))
            self.assertGreater(c["std_error"], 0.0)
            self.assertFalse(np.isnan(c["t_statistic"]))

    def test_p_values_and_confidence_intervals(self):
        """Verifies p-values are bounded in [0, 1] and CI lower < CI upper."""
        res = self.analyzer.fit_ols()
        for c in res["coefficients"]:
            self.assertGreaterEqual(c["p_value"], 0.0)
            self.assertLessEqual(c["p_value"], 1.0)
            self.assertLess(c["ci_lower_95"], c["ci_upper_95"])
            self.assertIsInstance(c["is_significant_05"], bool)

    def test_vif_computation(self):
        """Verifies VIF computation for all features."""
        vifs = self.analyzer.compute_vif()
        self.assertEqual(len(vifs), len(self.analyzer.feature_names))
        for feat, vif in vifs.items():
            self.assertGreaterEqual(vif, 1.0)
            self.assertFalse(np.isnan(vif))

    def test_anova_variance_decomposition(self):
        """Verifies ANOVA variance decomposition across the 5 factor families."""
        anova = self.analyzer.compute_anova_decomposition()
        self.assertIn("factor_families", anova)
        families = anova["factor_families"]
        self.assertEqual(len(families), len(FACTOR_FAMILIES))

        total_pct = sum(f["pct_explained_variance"] for f in families)
        self.assertAlmostEqual(total_pct, 100.0, places=3)

        for f in families:
            self.assertGreaterEqual(f["sum_of_squares"], 0.0)
            self.assertGreaterEqual(f["partial_eta_squared"], 0.0)
            self.assertLessEqual(f["partial_eta_squared"], 1.0)
            self.assertGreaterEqual(f["p_value"], 0.0)
            self.assertLessEqual(f["p_value"], 1.0)

    def test_interaction_effects(self):
        """Verifies repo x category interaction test results."""
        interaction = self.analyzer.compute_interaction_effects()
        self.assertIn("f_statistic", interaction)
        self.assertIn("p_value", interaction)
        self.assertIn("n_interaction_terms", interaction)
        self.assertGreater(interaction["n_interaction_terms"], 0)
        self.assertGreater(interaction["f_statistic"], 0.0)
        self.assertGreaterEqual(interaction["p_value"], 0.0)
        self.assertLessEqual(interaction["p_value"], 1.0)

    def test_manifest_export(self):
        """Verifies JSON manifest export and serializability."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_manifest.json"
            exported = self.analyzer.export_analysis_manifest(out_file)
            self.assertTrue(exported.exists())
            self.assertGreater(exported.stat().st_size, 1000)

            with open(exported, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("ols_model_summary", data)
            self.assertIn("multicollinearity_vif", data)
            self.assertIn("anova_variance_decomposition", data)
            self.assertIn("interaction_effects", data)
            self.assertIn("coefficients", data)

    def test_plot_generation(self):
        """Verifies Figure 31 4-panel generation and file size."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_plot = Path(tmp_dir) / "test_fig31.png"
            gen_plot = self.analyzer.generate_figures(out_plot)
            self.assertTrue(gen_plot.exists())
            self.assertGreater(gen_plot.stat().st_size, 50000)  # > 50 KB


if __name__ == "__main__":
    unittest.main()
