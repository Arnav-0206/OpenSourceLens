"""Unit tests for ml.eda.comment_analysis (IssueCommentAnalyzer)."""

import json
import unittest
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np

from ml.eda.comment_analysis import IssueCommentAnalyzer


class TestIssueCommentAnalyzer(unittest.TestCase):
    """Test suite for IssueCommentAnalyzer."""

    def setUp(self):
        """Construct synthetic test dataset with varied comment counts and reactions."""
        self.mock_df = pd.DataFrame({
            "issue_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "repo_name": [
                "repo_a", "repo_a", "repo_a", "repo_a",
                "repo_b", "repo_b", "repo_b", "repo_b",
            ],
            "state": ["open", "closed", "closed", "closed", "open", "closed", "open", "closed"],
            "canonical_label": ["bug", "bug", "feature", "other", "feature", "bug", "other", "feature"],
            "comments_count": [0, 1, 3, 5, 8, 15, 0, 2],
            "reactions_count": [0, 2, 0, 5, 10, 20, 0, 1],
            "interaction_total": [0, 3, 3, 10, 18, 35, 0, 3],
            "reactions_per_comment": [0.0, 2.0, 0.0, 1.0, 1.25, 1.33, 0.0, 0.5],
            "resolution_time_days": [np.nan, 1.5, 4.0, 10.0, np.nan, 25.0, np.nan, 3.0],
        })

    def test_init_from_dataframe(self):
        analyzer = IssueCommentAnalyzer(self.mock_df)
        self.assertEqual(len(analyzer.df), 8)
        self.assertEqual(analyzer.source_path, "DataFrame in-memory")

    def test_init_from_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            self.mock_df.to_csv(csv_path, index=False)
            analyzer = IssueCommentAnalyzer(csv_path)
            self.assertEqual(len(analyzer.df), 8)
            self.assertEqual(analyzer.source_path, str(csv_path))

    def test_init_from_parquet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            parquet_path = Path(tmpdir) / "test_data.parquet"
            self.mock_df.to_csv(csv_path, index=False)
            try:
                self.mock_df.to_parquet(parquet_path, index=False)
                analyzer = IssueCommentAnalyzer(parquet_path)
                self.assertEqual(len(analyzer.df), 8)
            except Exception:
                parquet_path.write_bytes(b"mock_parquet")
                analyzer = IssueCommentAnalyzer(parquet_path)
                self.assertEqual(len(analyzer.df), 8)

    def test_init_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            IssueCommentAnalyzer("non_existent_file.parquet")

    def test_init_invalid_type(self):
        with self.assertRaises(TypeError):
            IssueCommentAnalyzer(12345)  # type: ignore

    def test_init_empty_dataframe(self):
        with self.assertRaises(ValueError):
            IssueCommentAnalyzer(pd.DataFrame())

    def test_init_missing_comments_count(self):
        bad_df = pd.DataFrame({"repo_name": ["a"], "state": ["open"]})
        with self.assertRaises(KeyError):
            IssueCommentAnalyzer(bad_df)

    def test_compute_numeric_stats_empty(self):
        stats = IssueCommentAnalyzer._compute_numeric_stats(pd.Series([], dtype=float))
        self.assertEqual(stats["count"], 0)
        self.assertEqual(stats["mean"], 0.0)

    def test_comment_volume_distribution(self):
        analyzer = IssueCommentAnalyzer(self.mock_df)
        dist = analyzer.get_comment_volume_distribution()

        self.assertEqual(dist["count"], 8)
        self.assertEqual(dist["min"], 0.0)
        self.assertEqual(dist["max"], 15.0)
        self.assertGreater(dist["mean"], 0)
        self.assertIn("median", dist)
        self.assertIn("skewness", dist)
        self.assertIn("p90", dist)

    def test_comment_buckets(self):
        analyzer = IssueCommentAnalyzer(self.mock_df)
        buckets_df = analyzer.get_comment_buckets()

        self.assertEqual(len(buckets_df), len(IssueCommentAnalyzer.COMMENT_BUCKET_LABELS))
        self.assertEqual(buckets_df["issue_count"].sum(), 8)
        self.assertAlmostEqual(buckets_df["percentage"].sum(), 100.0, places=1)
        self.assertEqual(buckets_df["cumulative_count"].iloc[-1], 8)
        self.assertAlmostEqual(buckets_df["cumulative_percentage"].iloc[-1], 100.0, places=1)

        # 0 comments should be 2 issues (indices 0 and 6)
        row_zero = buckets_df[buckets_df["bucket"] == "0 Comments"].iloc[0]
        self.assertEqual(row_zero["issue_count"], 2)

    def test_comments_by_repository(self):
        analyzer = IssueCommentAnalyzer(self.mock_df)
        repo_df = analyzer.get_comments_by_repository()

        self.assertEqual(len(repo_df), 2)
        repos = set(repo_df["repo_name"])
        self.assertIn("repo_a", repos)
        self.assertIn("repo_b", repos)

        repo_a = repo_df[repo_df["repo_name"] == "repo_a"].iloc[0]
        self.assertEqual(repo_a["issue_count"], 4)
        self.assertEqual(repo_a["total_comments"], 9)  # 0 + 1 + 3 + 5
        self.assertEqual(repo_a["zero_comment_count"], 1)
        self.assertEqual(repo_a["zero_comment_rate_pct"], 25.0)

    def test_comments_by_state(self):
        analyzer = IssueCommentAnalyzer(self.mock_df)
        state_df = analyzer.get_comments_by_state()

        self.assertEqual(len(state_df), 2)
        states = set(state_df["state"])
        self.assertEqual(states, {"open", "closed"})

        open_row = state_df[state_df["state"] == "open"].iloc[0]
        self.assertEqual(open_row["issue_count"], 3)  # indices 0, 4, 6
        self.assertEqual(open_row["zero_comment_count"], 2)  # indices 0 and 6

    def test_comments_by_label(self):
        analyzer = IssueCommentAnalyzer(self.mock_df)
        label_df = analyzer.get_comments_by_label()

        self.assertEqual(len(label_df), 3)
        labels = set(label_df["label"])
        self.assertEqual(labels, {"bug", "feature", "other"})

    def test_comments_vs_resolution(self):
        analyzer = IssueCommentAnalyzer(self.mock_df)
        res_df = analyzer.get_comments_vs_resolution()

        self.assertEqual(len(res_df), len(IssueCommentAnalyzer.COMMENT_BUCKET_LABELS))
        # 5 closed issues in mock_df
        self.assertEqual(res_df["closed_issue_count"].sum(), 5)

        # 11+ bucket has 1 closed issue with 25.0 days
        bucket_11 = res_df[res_df["bucket"] == "11+ Comments"].iloc[0]
        self.assertEqual(bucket_11["closed_issue_count"], 1)
        self.assertEqual(bucket_11["median_resolution_days"], 25.0)

    def test_reaction_dynamics(self):
        analyzer = IssueCommentAnalyzer(self.mock_df)
        dynamics = analyzer.get_reaction_dynamics()

        self.assertIn("reactions_count", dynamics)
        self.assertIn("interaction_total", dynamics)
        self.assertIn("reactions_per_comment", dynamics)
        # 3 issues with 0 reactions (indices 0, 2, 6)
        self.assertEqual(dynamics["zero_reaction_count"], 3)
        self.assertAlmostEqual(dynamics["zero_reaction_rate_pct"], 37.5, places=1)

    def test_sampled_comments_profile_with_mock_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_file = Path(tmpdir) / "comments_1.json"
            sample_data = [
                {"author_association": "MEMBER", "body": "Hello world from maintainer"},
                {"author_association": "CONTRIBUTOR", "body": "Thank you for reviewing this pull request"},
            ]
            with open(sample_file, "w", encoding="utf-8") as f:
                json.dump(sample_data, f)

            analyzer = IssueCommentAnalyzer(self.mock_df, comments_dir=tmpdir)
            profile = analyzer.get_sampled_comments_profile()

            self.assertTrue(profile["available"])
            self.assertEqual(profile["total_files"], 1)
            self.assertEqual(profile["total_comments"], 2)
            self.assertEqual(profile["author_associations"]["MEMBER"], 1)
            self.assertEqual(profile["author_associations"]["CONTRIBUTOR"], 1)
            self.assertGreater(profile["avg_char_length"], 0)
            self.assertGreater(profile["avg_word_length"], 0)

    def test_sampled_comments_profile_missing_directory(self):
        analyzer = IssueCommentAnalyzer(self.mock_df, comments_dir="non_existent_comments_dir")
        profile = analyzer.get_sampled_comments_profile()

        self.assertFalse(profile["available"])
        self.assertEqual(profile["total_files"], 0)
        self.assertEqual(profile["total_comments"], 0)

    def test_compute_full_comment_analysis(self):
        analyzer = IssueCommentAnalyzer(self.mock_df, comments_dir=None)
        full = analyzer.compute_full_comment_analysis()

        self.assertIn("comment_volume_distribution", full)
        self.assertIn("comment_buckets", full)
        self.assertIn("comments_by_repository", full)
        self.assertIn("comments_by_state", full)
        self.assertIn("comments_by_label", full)
        self.assertIn("comments_vs_resolution", full)
        self.assertIn("reaction_dynamics", full)
        self.assertIn("sampled_comments_profile", full)

    def test_generate_summary_report(self):
        analyzer = IssueCommentAnalyzer(self.mock_df, comments_dir=None)
        report = analyzer.generate_summary_report()

        self.assertIsInstance(report, str)
        self.assertIn("# OpenSourceLens — Comment Analysis Summary", report)
        self.assertIn("Overall Comment Distribution", report)
        self.assertIn("Comment Volume Buckets", report)
        self.assertIn("repo_a", report)

    def test_live_master_dataset_integration(self):
        master_path = Path("data/processed/issues_master.csv")
        if not master_path.exists():
            master_path = Path("data/processed/issues_master.parquet")
        if not master_path.exists():
            self.skipTest("Master dataset file not available for live testing.")

        analyzer = IssueCommentAnalyzer(master_path, comments_dir="data/raw/comments")
        self.assertEqual(len(analyzer.df), 2328)

        dist = analyzer.get_comment_volume_distribution()
        self.assertEqual(dist["count"], 2328)
        self.assertAlmostEqual(dist["mean"], 2.78, places=1)
        self.assertEqual(dist["median"], 2.0)
        self.assertEqual(dist["max"], 66.0)

        buckets = analyzer.get_comment_buckets()
        self.assertEqual(buckets["issue_count"].sum(), 2328)

        repos = analyzer.get_comments_by_repository()
        self.assertEqual(len(repos), 4)

        report = analyzer.generate_summary_report()
        self.assertIn("kubernetes", report)


if __name__ == "__main__":
    unittest.main()
