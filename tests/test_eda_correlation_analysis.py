"""Unit tests for ml.eda.correlation_analysis (IssueCorrelationAnalyzer)."""

import unittest
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np

from ml.eda.correlation_analysis import IssueCorrelationAnalyzer


class TestIssueCorrelationAnalyzer(unittest.TestCase):
    """Test suite for IssueCorrelationAnalyzer."""

    def setUp(self):
        """Construct synthetic test dataset with known linear and non-linear associations."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        # y is linearly correlated with x
        y = 2.0 * x + 1.0
        # z is non-linearly monotonic (exponential)
        z = np.exp(x / 3.0)
        # noise is uncorrelated
        noise = np.array([3.1, 1.2, 8.4, 2.5, 9.1, 4.2, 1.1, 6.5, 3.0, 5.5])
        # const has zero variance
        const = np.zeros(10)

        self.mock_df = pd.DataFrame({
            "issue_id": list(range(1, 11)),
            "feature_x": x,
            "feature_y": y,
            "feature_z": z,
            "noise": noise,
            "constant_col": const,
            "flag_bool": [True, False, True, False, True, False, True, False, True, False],
            "state": ["open", "closed", "closed", "closed", "closed", "closed", "closed", "closed", "open", "closed"],
            "resolution_time_days": [np.nan, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, np.nan, 18.0],
            "comments_count": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        })

    def test_init_from_dataframe(self):
        analyzer = IssueCorrelationAnalyzer(self.mock_df)
        self.assertEqual(len(analyzer.df), 10)
        self.assertEqual(analyzer.source_path, "DataFrame in-memory")
        # Ensure boolean flag was converted to float
        self.assertEqual(analyzer.df["flag_bool"].dtype, float)

    def test_init_from_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            self.mock_df.to_csv(csv_path, index=False)
            analyzer = IssueCorrelationAnalyzer(csv_path)
            self.assertEqual(len(analyzer.df), 10)
            self.assertEqual(analyzer.source_path, str(csv_path))

    def test_init_from_parquet_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            parquet_path = Path(tmpdir) / "test_data.parquet"
            self.mock_df.to_csv(csv_path, index=False)
            parquet_path.write_bytes(b"MOCK_PARQUET_FALLBACK")
            analyzer = IssueCorrelationAnalyzer(parquet_path)
            self.assertEqual(len(analyzer.df), 10)

    def test_init_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            IssueCorrelationAnalyzer("non_existent_file.csv")

    def test_init_invalid_type(self):
        with self.assertRaises(TypeError):
            IssueCorrelationAnalyzer(12345)  # type: ignore

    def test_init_empty_dataframe(self):
        with self.assertRaises(ValueError):
            IssueCorrelationAnalyzer(pd.DataFrame())

    def test_get_available_features(self):
        analyzer = IssueCorrelationAnalyzer(self.mock_df)
        avail = analyzer.get_available_features(["feature_x", "feature_y", "constant_col", "missing_col"])
        self.assertIn("feature_x", avail)
        self.assertIn("feature_y", avail)
        # constant_col has zero variance and missing_col is absent; both must be excluded
        self.assertNotIn("constant_col", avail)
        self.assertNotIn("missing_col", avail)

    def test_get_correlation_matrix_pearson(self):
        analyzer = IssueCorrelationAnalyzer(self.mock_df)
        corr = analyzer.get_correlation_matrix(method="pearson", features=["feature_x", "feature_y", "noise"])

        self.assertEqual(corr.shape, (3, 3))
        # Diagonal is 1.0
        self.assertAlmostEqual(corr.loc["feature_x", "feature_x"], 1.0, places=3)
        # Perfect linear correlation between x and y
        self.assertAlmostEqual(corr.loc["feature_x", "feature_y"], 1.0, places=3)
        # Symmetric
        self.assertAlmostEqual(corr.loc["feature_x", "feature_y"], corr.loc["feature_y", "feature_x"], places=3)

    def test_get_correlation_matrix_spearman(self):
        analyzer = IssueCorrelationAnalyzer(self.mock_df)
        corr = analyzer.get_correlation_matrix(method="spearman", features=["feature_x", "feature_z"])

        # Monotonic relationship between x and z = exp(x/3) has perfect Spearman rank corr = 1.0
        self.assertAlmostEqual(corr.loc["feature_x", "feature_z"], 1.0, places=3)

    def test_get_correlation_matrix_invalid_method(self):
        analyzer = IssueCorrelationAnalyzer(self.mock_df)
        with self.assertRaises(ValueError):
            analyzer.get_correlation_matrix(method="unsupported_method")

    def test_get_target_correlations(self):
        analyzer = IssueCorrelationAnalyzer(self.mock_df)
        target_df = analyzer.get_target_correlations(
            target="feature_x",
            features=["feature_y", "feature_z", "noise"]
        )

        self.assertEqual(len(target_df), 3)
        self.assertIn("feature", target_df.columns)
        self.assertIn("pearson_corr", target_df.columns)
        self.assertIn("spearman_corr", target_df.columns)
        self.assertIn("diff_spearman_pearson", target_df.columns)

        # feature_y is highest correlated
        top = target_df.iloc[0]
        self.assertEqual(top["feature"], "feature_y")
        self.assertAlmostEqual(top["pearson_corr"], 1.0, places=3)

    def test_get_target_correlations_subset_closed(self):
        analyzer = IssueCorrelationAnalyzer(self.mock_df)
        target_df = analyzer.get_target_correlations(
            target="resolution_time_days",
            subset="closed",
            features=["feature_x", "comments_count"]
        )
        self.assertGreater(len(target_df), 0)

    def test_get_target_correlations_missing_target(self):
        analyzer = IssueCorrelationAnalyzer(self.mock_df)
        target_df = analyzer.get_target_correlations(target="non_existent_target")
        self.assertTrue(target_df.empty)

    def test_get_top_correlated_pairs(self):
        analyzer = IssueCorrelationAnalyzer(self.mock_df)
        pairs_df = analyzer.get_top_correlated_pairs(
            threshold=0.80,
            method="pearson",
            features=["feature_x", "feature_y", "feature_z", "noise"]
        )

        self.assertGreaterEqual(len(pairs_df), 1)
        top = pairs_df.iloc[0]
        # x and y have r = 1.0
        self.assertEqual(set([top["feature_1"], top["feature_2"]]), {"feature_x", "feature_y"})
        self.assertAlmostEqual(top["correlation"], 1.0, places=3)
        self.assertEqual(top["relationship"], "Positive")

    def test_get_multicollinearity_report(self):
        analyzer = IssueCorrelationAnalyzer(self.mock_df)
        report = analyzer.get_multicollinearity_report(threshold=0.85)

        self.assertEqual(report["threshold"], 0.85)
        self.assertGreaterEqual(report["total_high_risk_pairs"], 1)
        self.assertIn("feature_x", report["affected_features"])
        self.assertIn("feature_y", report["affected_features"])

        high_risk = report["high_risk_pairs"][0]
        self.assertEqual(high_risk["risk_level"], "Severe Multicollinearity")
        self.assertIn("recommendation", high_risk)

    def test_get_correlation_summary(self):
        analyzer = IssueCorrelationAnalyzer(self.mock_df)
        summary = analyzer.get_correlation_summary()

        self.assertIn("features_analyzed", summary)
        self.assertIn("total_feature_pairs", summary)
        self.assertIn("mean_abs_correlation", summary)
        self.assertIn("highest_positive_pair", summary)

    def test_compute_full_correlation_analysis(self):
        analyzer = IssueCorrelationAnalyzer(self.mock_df)
        full = analyzer.compute_full_correlation_analysis()

        self.assertIn("correlation_summary", full)
        self.assertIn("target_correlations_resolution", full)
        self.assertIn("target_correlations_comments", full)
        self.assertIn("top_correlated_pairs_pearson", full)
        self.assertIn("top_correlated_pairs_spearman", full)
        self.assertIn("multicollinearity_report", full)

    def test_generate_summary_report(self):
        analyzer = IssueCorrelationAnalyzer(self.mock_df)
        report = analyzer.generate_summary_report()

        self.assertIsInstance(report, str)
        self.assertIn("# OpenSourceLens — Correlation & Multicollinearity Analysis Summary", report)
        self.assertIn("Overview & Inter-Feature Dynamics", report)
        self.assertIn("Target Associations", report)
        self.assertIn("Multicollinearity & ML Modeling Diagnostics", report)

    def test_edge_case_zero_variance(self):
        df_const = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [2.0, 2.0, 2.0]})
        analyzer = IssueCorrelationAnalyzer(df_const)
        corr = analyzer.get_correlation_matrix()
        self.assertTrue(corr.empty)

    def test_live_master_dataset_integration(self):
        csv_path = Path("data/processed/issues_master.csv")
        if not csv_path.exists():
            self.skipTest("Master dataset CSV not found.")

        analyzer = IssueCorrelationAnalyzer(csv_path)
        self.assertEqual(len(analyzer.df), 2328)

        corr = analyzer.get_correlation_matrix()
        self.assertGreater(len(corr.columns), 10)

        res_corr = analyzer.get_target_correlations(target="resolution_time_days", subset="closed")
        self.assertGreater(len(res_corr), 5)

        # comments_count should be in top resolution predictors
        top_features = res_corr["feature"].head(5).tolist()
        self.assertIn("comments_count", top_features)

        collinear = analyzer.get_multicollinearity_report(threshold=0.70)
        self.assertGreaterEqual(collinear["total_high_risk_pairs"], 1)

        report = analyzer.generate_summary_report()
        self.assertIn("comments_count", report)
        self.assertIn("interaction_total", report)


if __name__ == "__main__":
    unittest.main()
