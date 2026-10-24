"""Unit tests for ml.eda.resolution_analysis (IssueResolutionAnalyzer)."""

import unittest
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np

from ml.eda.resolution_analysis import IssueResolutionAnalyzer


class TestIssueResolutionAnalyzer(unittest.TestCase):
    """Test suite for IssueResolutionAnalyzer."""

    def setUp(self):
        """Construct synthetic test dataset with varied resolution times, reasons, and repos."""
        self.mock_df = pd.DataFrame({
            "issue_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "repo_name": [
                "repo_a", "repo_a", "repo_a", "repo_a",
                "repo_b", "repo_b", "repo_b", "repo_b",
            ],
            "state": ["open", "closed", "closed", "closed", "open", "closed", "open", "closed"],
            "state_reason": [
                "unresolved", "completed", "duplicate", "completed",
                "unresolved", "not_planned", "unresolved", "completed",
            ],
            "canonical_label": ["bug", "bug", "feature", "other", "feature", "bug", "other", "feature"],
            "comments_count": [0, 1, 3, 5, 8, 15, 0, 2],
            "resolution_time_days": [np.nan, 0.5, 0.2, 5.0, np.nan, 20.0, np.nan, 2.0],
            "resolution_time_hours": [np.nan, 12.0, 4.8, 120.0, np.nan, 480.0, np.nan, 48.0],
            "issue_age_days": [10.0, np.nan, np.nan, np.nan, 45.0, np.nan, 5.0, np.nan],
        })

    def test_init_from_dataframe(self):
        analyzer = IssueResolutionAnalyzer(self.mock_df)
        self.assertEqual(len(analyzer.df), 8)
        self.assertEqual(len(analyzer.closed_df), 5)
        self.assertEqual(analyzer.source_path, "DataFrame in-memory")

    def test_init_from_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            self.mock_df.to_csv(csv_path, index=False)
            analyzer = IssueResolutionAnalyzer(csv_path)
            self.assertEqual(len(analyzer.df), 8)
            self.assertEqual(len(analyzer.closed_df), 5)
            self.assertEqual(analyzer.source_path, str(csv_path))

    def test_init_from_parquet_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            parquet_path = Path(tmpdir) / "test_data.parquet"
            self.mock_df.to_csv(csv_path, index=False)
            # Create a mock parquet file placeholder that will trigger fallback to csv
            parquet_path.write_bytes(b"INVALID_PARQUET_MAGIC")
            analyzer = IssueResolutionAnalyzer(parquet_path)
            self.assertEqual(len(analyzer.df), 8)

    def test_init_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            IssueResolutionAnalyzer("non_existent_file.csv")

    def test_init_invalid_type(self):
        with self.assertRaises(TypeError):
            IssueResolutionAnalyzer(9999)  # type: ignore

    def test_init_empty_dataframe(self):
        with self.assertRaises(ValueError):
            IssueResolutionAnalyzer(pd.DataFrame())

    def test_init_missing_columns(self):
        bad_df = pd.DataFrame({"state": ["closed"], "comments_count": [1]})
        with self.assertRaises(KeyError):
            IssueResolutionAnalyzer(bad_df)

    def test_compute_numeric_stats_empty(self):
        stats = IssueResolutionAnalyzer._compute_numeric_stats(pd.Series([], dtype=float))
        self.assertEqual(stats["count"], 0)
        self.assertEqual(stats["mean"], 0.0)

    def test_get_resolution_distribution_days(self):
        analyzer = IssueResolutionAnalyzer(self.mock_df)
        dist = analyzer.get_resolution_distribution(unit="days")

        self.assertEqual(dist["count"], 5)
        self.assertEqual(dist["min"], 0.2)
        self.assertEqual(dist["max"], 20.0)
        self.assertAlmostEqual(dist["median"], 2.0, places=1)
        self.assertIn("skewness", dist)
        self.assertIn("iqr", dist)

    def test_get_resolution_distribution_hours(self):
        analyzer = IssueResolutionAnalyzer(self.mock_df)
        dist = analyzer.get_resolution_distribution(unit="hours")

        self.assertEqual(dist["count"], 5)
        self.assertEqual(dist["min"], 4.8)
        self.assertEqual(dist["max"], 480.0)
        self.assertAlmostEqual(dist["median"], 48.0, places=1)

    def test_get_resolution_speed_buckets(self):
        analyzer = IssueResolutionAnalyzer(self.mock_df)
        buckets_df = analyzer.get_resolution_speed_buckets()

        self.assertEqual(len(buckets_df), len(IssueResolutionAnalyzer.RESOLUTION_SPEED_LABELS))
        self.assertEqual(buckets_df["closed_issue_count"].sum(), 5)
        self.assertAlmostEqual(buckets_df["percentage"].sum(), 100.0, places=1)
        self.assertEqual(buckets_df["cumulative_count"].iloc[-1], 5)

        # Issues with <= 1 day are 0.5 and 0.2 (2 issues)
        under_1 = buckets_df[buckets_df["speed_tier"] == "< 1 Day"].iloc[0]
        self.assertEqual(under_1["closed_issue_count"], 2)

    def test_get_resolution_by_repository(self):
        analyzer = IssueResolutionAnalyzer(self.mock_df)
        repo_df = analyzer.get_resolution_by_repository()

        self.assertEqual(len(repo_df), 2)
        repos = set(repo_df["repo_name"])
        self.assertEqual(repos, {"repo_a", "repo_b"})

        repo_a = repo_df[repo_df["repo_name"] == "repo_a"].iloc[0]
        self.assertEqual(repo_a["closed_count"], 3)  # indices 1, 2, 3
        self.assertEqual(repo_a["fast_closure_count"], 2)  # 0.5 and 0.2
        self.assertAlmostEqual(repo_a["fast_closure_pct"], 66.67, places=1)

    def test_get_resolution_by_reason(self):
        analyzer = IssueResolutionAnalyzer(self.mock_df)
        reason_df = analyzer.get_resolution_by_reason()

        self.assertEqual(len(reason_df), 3)
        reasons = set(reason_df["state_reason"])
        self.assertEqual(reasons, {"completed", "duplicate", "not_planned"})

        dup_row = reason_df[reason_df["state_reason"] == "duplicate"].iloc[0]
        self.assertEqual(dup_row["closed_count"], 1)
        self.assertEqual(dup_row["median_days"], 0.2)

    def test_get_resolution_by_label(self):
        analyzer = IssueResolutionAnalyzer(self.mock_df)
        label_df = analyzer.get_resolution_by_label()

        self.assertEqual(len(label_df), 3)
        labels = set(label_df["canonical_label"])
        self.assertEqual(labels, {"bug", "feature", "other"})

    def test_get_resolution_by_comment_volume(self):
        analyzer = IssueResolutionAnalyzer(self.mock_df)
        comment_df = analyzer.get_resolution_by_comment_volume()

        self.assertEqual(len(comment_df), len(IssueResolutionAnalyzer.COMMENT_BUCKET_LABELS))
        self.assertEqual(comment_df["closed_count"].sum(), 5)

    def test_get_backlog_aging(self):
        analyzer = IssueResolutionAnalyzer(self.mock_df)
        aging = analyzer.get_backlog_aging()

        self.assertEqual(aging["count"], 3)  # open issues: indices 0, 4, 6
        self.assertEqual(aging["min"], 5.0)
        self.assertEqual(aging["max"], 45.0)
        self.assertEqual(aging["median"], 10.0)

    def test_compute_full_resolution_analysis(self):
        analyzer = IssueResolutionAnalyzer(self.mock_df)
        full = analyzer.compute_full_resolution_analysis()

        self.assertIn("resolution_distribution_days", full)
        self.assertIn("resolution_distribution_hours", full)
        self.assertIn("speed_buckets", full)
        self.assertIn("resolution_by_repository", full)
        self.assertIn("resolution_by_reason", full)
        self.assertIn("resolution_by_label", full)
        self.assertIn("resolution_by_comment_volume", full)
        self.assertIn("backlog_aging", full)

    def test_generate_summary_report(self):
        analyzer = IssueResolutionAnalyzer(self.mock_df)
        report = analyzer.generate_summary_report()

        self.assertIsInstance(report, str)
        self.assertIn("# OpenSourceLens — Resolution-Time Analysis Summary", report)
        self.assertIn("Overall Resolution Distribution", report)
        self.assertIn("Operational Resolution Speed Tiers", report)
        self.assertIn("repo_a", report)

    def test_edge_case_no_closed_issues(self):
        all_open = self.mock_df[self.mock_df["state"] == "open"].copy()
        analyzer = IssueResolutionAnalyzer(all_open)

        dist = analyzer.get_resolution_distribution(unit="days")
        self.assertEqual(dist["count"], 0)
        speed_df = analyzer.get_resolution_speed_buckets()
        self.assertTrue(speed_df.empty or speed_df["closed_issue_count"].sum() == 0)

    def test_live_master_dataset_integration(self):
        csv_path = Path("data/processed/issues_master.csv")
        if not csv_path.exists():
            self.skipTest("Master dataset CSV not found.")

        analyzer = IssueResolutionAnalyzer(csv_path)
        self.assertEqual(len(analyzer.df), 2328)
        self.assertEqual(len(analyzer.closed_df), 1135)

        dist = analyzer.get_resolution_distribution(unit="days")
        self.assertEqual(dist["count"], 1135)
        self.assertAlmostEqual(dist["mean"], 8.35, places=1)
        self.assertAlmostEqual(dist["median"], 1.75, places=1)

        speed_df = analyzer.get_resolution_speed_buckets()
        self.assertEqual(speed_df["closed_issue_count"].sum(), 1135)

        repos_df = analyzer.get_resolution_by_repository()
        self.assertEqual(len(repos_df), 4)

        reasons_df = analyzer.get_resolution_by_reason()
        self.assertGreaterEqual(len(reasons_df), 3)

        report = analyzer.generate_summary_report()
        self.assertIn("kubernetes/kubernetes", report)
        self.assertIn("microsoft/vscode", report)


if __name__ == "__main__":
    unittest.main()
