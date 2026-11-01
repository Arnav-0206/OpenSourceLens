"""
Unit tests for OpenSourceLens Phase 8, Step 8.2: Resolution Distribution Analysis.

Validates:
- Statistical moments and percentiles calculation
- Log-transformation variance stabilization (skewness and kurtosis reduction)
- Subgroup turnaround comparisons across categories, repositories, and closure types
- Non-parametric hypothesis tests (Kruskal-Wallis, Mann-Whitney U)
- Kaplan-Meier product-limit survival curve mathematical properties (monotonicity, boundaries)
- Publication Figure 29 generation and JSON manifest export
"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from ml.resolution_analysis.distribution import ResolutionDistributionAnalyzer

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestResolutionDistributionAnalysis(unittest.TestCase):
    """Test suite for ResolutionDistributionAnalyzer."""

    def setUp(self):
        """Set up synthetic test fixtures representing diverse repos, categories, and censored data."""
        np.random.seed(42)

        # Create synthetic dataset with known properties
        records = []
        # Repositories with distinct turnaround scales
        # VS Code: fast (exponential scale 10 hours)
        for i in range(40):
            h = float(np.random.exponential(10.0))
            records.append({
                "issue_id": 1000 + i,
                "repo_name": "microsoft/vscode",
                "canonical_label": "bug" if i % 2 == 0 else "feature",
                "state": "closed",
                "state_reason": "completed",
                "is_censored": False,
                "event_observed": 1,
                "resolution_time_hours": h,
                "resolution_time_days": h / 24.0,
                "duration_at_risk_days": h / 24.0,
                "resolution_time_log_hours": np.log(h + 1.0),
                "is_internal_author": (i % 3 == 0),
                "resolution_type": "completed",
            })

        # Kubernetes: slow (exponential scale 150 hours)
        for i in range(40):
            h = float(np.random.exponential(150.0))
            records.append({
                "issue_id": 2000 + i,
                "repo_name": "kubernetes/kubernetes",
                "canonical_label": "bug" if i % 2 == 0 else "other",
                "state": "closed",
                "state_reason": "completed",
                "is_censored": False,
                "event_observed": 1,
                "resolution_time_hours": h,
                "resolution_time_days": h / 24.0,
                "duration_at_risk_days": h / 24.0,
                "resolution_time_log_hours": np.log(h + 1.0),
                "is_internal_author": (i % 2 == 0),
                "resolution_type": "completed",
            })

        # Open / Censored issues
        for i in range(30):
            risk_days = float(np.random.uniform(1.0, 60.0))
            records.append({
                "issue_id": 3000 + i,
                "repo_name": "microsoft/vscode" if i % 2 == 0 else "kubernetes/kubernetes",
                "canonical_label": "feature",
                "state": "open",
                "state_reason": None,
                "is_censored": True,
                "event_observed": 0,
                "resolution_time_hours": np.nan,
                "resolution_time_days": np.nan,
                "duration_at_risk_days": risk_days,
                "resolution_time_log_hours": np.nan,
                "is_internal_author": False,
                "resolution_type": "unresolved",
            })

        self.sample_df = pd.DataFrame(records)
        self.analyzer = ResolutionDistributionAnalyzer(self.sample_df)

    def test_initialization_exceptions(self):
        """Verify initialization errors on missing files, bad types, or missing columns."""
        with self.assertRaises(FileNotFoundError):
            ResolutionDistributionAnalyzer("non_existent_file.parquet")

        with self.assertRaises(TypeError):
            ResolutionDistributionAnalyzer(12345)

        with self.assertRaises(ValueError):
            ResolutionDistributionAnalyzer(pd.DataFrame())

        bad_df = pd.DataFrame({"some_col": [1, 2, 3]})
        with self.assertRaises(KeyError):
            ResolutionDistributionAnalyzer(bad_df)

    def test_compute_moments(self):
        """Verify statistical moments calculation (mean, median, quantiles, skewness, kurtosis)."""
        moments = self.analyzer.compute_moments("resolution_time_hours")

        self.assertEqual(moments["count"], 80)
        self.assertGreater(moments["mean"], 0.0)
        self.assertGreater(moments["median"], 0.0)
        self.assertGreater(moments["iqr"], 0.0)
        self.assertIn("p50", moments["quantiles"])
        self.assertIn("p90", moments["quantiles"])
        self.assertIn("skewness", moments)
        self.assertIn("excess_kurtosis", moments)

    def test_log_variance_stabilization(self):
        """Verify log transformation significantly stabilizes positive right skewness."""
        moments_raw = self.analyzer.compute_moments("resolution_time_hours")
        moments_log = self.analyzer.compute_moments("resolution_time_log_hours")

        # Raw hours should have high positive skewness (> 1.0)
        self.assertGreater(moments_raw["skewness"], 1.0)
        # Log-transformed skewness should be drastically reduced
        self.assertLess(abs(moments_log["skewness"]), abs(moments_raw["skewness"]))

    def test_compare_subgroups(self):
        """Verify subgroup turnaround statistics across repositories and categories."""
        subgroups = self.analyzer.compare_subgroups()

        self.assertIn("by_repository", subgroups)
        self.assertIn("by_category", subgroups)
        self.assertIn("by_resolution_type", subgroups)
        self.assertIn("by_author_type", subgroups)

        repo_stats = subgroups["by_repository"]
        self.assertIn("microsoft/vscode", repo_stats)
        self.assertIn("kubernetes/kubernetes", repo_stats)

        # VS Code synthetic data (scale 10) should have lower median than Kubernetes (scale 150)
        self.assertLess(
            repo_stats["microsoft/vscode"]["median_hours"],
            repo_stats["kubernetes/kubernetes"]["median_hours"],
        )

    def test_perform_hypothesis_tests(self):
        """Verify Kruskal-Wallis hypothesis tests return valid statistics and p-values."""
        tests = self.analyzer.perform_hypothesis_tests()

        self.assertIn("kruskal_wallis_by_repository", tests)
        repo_test = tests["kruskal_wallis_by_repository"]
        self.assertGreater(repo_test["h_statistic"], 0.0)
        self.assertGreaterEqual(repo_test["p_value"], 0.0)
        self.assertLessEqual(repo_test["p_value"], 1.0)
        self.assertTrue(repo_test["is_statistically_significant"])

    def test_kaplan_meier_mathematical_properties(self):
        """Verify Kaplan-Meier estimator satisfies boundary and monotonicity axioms."""
        km_curves = self.analyzer.estimate_survival_curves()
        self.assertIn("Overall", km_curves)
        km_df = km_curves["Overall"]

        # 1. Boundary: S(0) == 1.0
        self.assertEqual(km_df.iloc[0]["time_days"], 0.0)
        self.assertEqual(km_df.iloc[0]["survival_probability"], 1.0)

        # 2. Bounded range: 0.0 <= S(t) <= 1.0
        probs = km_df["survival_probability"].values
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

        # 3. Monotonic non-increasing: S(t_k+1) <= S(t_k)
        diffs = np.diff(probs)
        self.assertTrue(np.all(diffs <= 1e-7), f"Violations found: {diffs[diffs > 1e-7]}")

    def test_kaplan_meier_stratified_by_group(self):
        """Verify stratified Kaplan-Meier curves are generated for each repository."""
        km_by_repo = self.analyzer.estimate_survival_curves(group_by="repo_name")

        self.assertIn("microsoft/vscode", km_by_repo)
        self.assertIn("kubernetes/kubernetes", km_by_repo)

        vs_km = km_by_repo["microsoft/vscode"]
        k8s_km = km_by_repo["kubernetes/kubernetes"]

        # VS Code resolves faster, so its survival probability S(t) should decay faster (lower S(t) at day 5)
        vs_p5 = vs_km[vs_km["time_days"] >= 5.0].iloc[0]["survival_probability"]
        k8s_p5 = k8s_km[k8s_km["time_days"] >= 5.0].iloc[0]["survival_probability"]
        self.assertLess(vs_p5, k8s_p5)

    def test_generate_fig29_visualization(self):
        """Verify Figure 29 generates a valid, non-empty PNG file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_png = Path(tmp_dir) / "test_fig29.png"
            res_path = self.analyzer.generate_fig29_visualization(out_png)

            self.assertTrue(res_path.exists())
            self.assertGreater(res_path.stat().st_size, 10000)

    def test_export_analysis_manifest(self):
        """Verify JSON analysis manifest exports all required analytical fields."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_json = Path(tmp_dir) / "test_analysis.json"
            res_path = self.analyzer.export_analysis_manifest(out_json)

            self.assertTrue(res_path.exists())
            with open(res_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data["phase"], "Phase 8: Resolution Analysis & Prediction")
            self.assertEqual(data["step"], "Step 8.2: Analyze Resolution Distribution")
            self.assertIn("statistical_moments", data)
            self.assertIn("subgroup_comparisons", data)
            self.assertIn("hypothesis_testing", data)


if __name__ == "__main__":
    unittest.main()
