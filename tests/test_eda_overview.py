"""Unit tests for ml.eda.overview (DatasetOverviewAnalyzer)."""

import json
import unittest
import tempfile
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np

from ml.eda.overview import DatasetOverviewAnalyzer


class TestDatasetOverviewAnalyzer(unittest.TestCase):
    """Test suite for DatasetOverviewAnalyzer."""

    def setUp(self):
        """Construct synthetic test DataFrame mimicking issues_master schema."""
        self.mock_df = pd.DataFrame({
            "repo_name": ["repo/a", "repo/a", "repo/b", "repo/b"],
            "issue_id": [101, 102, 103, 104],
            "title": ["Bug in login", "Feature request", "Crash on start", "Typo in docs"],
            "body": ["App crashes with error", "Add dark mode please", "Stack trace follows", "Doc fix"],
            "clean_title": ["Bug in login", "Feature request", "Crash on start", "Typo in docs"],
            "clean_body": ["App crashes with error", "Add dark mode please", "Stack trace follows", "Doc fix"],
            "state": ["open", "closed", "closed", "open"],
            "state_reason": [None, "completed", "completed", None],
            "created_at": pd.to_datetime([
                "2026-01-01T00:00:00Z",
                "2026-01-05T00:00:00Z",
                "2026-01-10T00:00:00Z",
                "2026-01-20T00:00:00Z",
            ], utc=True),
            "closed_at": pd.to_datetime([
                None,
                "2026-01-07T00:00:00Z",
                "2026-01-15T00:00:00Z",
                None,
            ], utc=True),
            "resolution_time_days": [np.nan, 2.0, 5.0, np.nan],
            "clean_title_word_count": [3, 2, 3, 3],
            "clean_body_word_count": [4, 4, 3, 2],
            "comments_count": [1, 3, 0, 2],
            "reactions_count": [0, 1, 0, 0],
            "canonical_label": ["bug", "feature", "bug", "documentation"],
            "is_classified_category": [True, True, True, True],
            "has_code_block": [True, False, True, False],
            "has_stack_trace": [False, False, True, False],
            "has_url": [False, True, False, False],
        })

    def test_init_from_dataframe(self):
        """Test initialization directly from DataFrame."""
        analyzer = DatasetOverviewAnalyzer(self.mock_df)
        self.assertEqual(len(analyzer.df), 4)
        self.assertEqual(len(analyzer.df.columns), 20)
        self.assertEqual(analyzer.source_path, "DataFrame in-memory")

    def test_init_from_parquet_and_csv(self):
        """Test initialization from temporary Parquet and CSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p_path = Path(tmpdir) / "test.parquet"
            c_path = Path(tmpdir) / "test.csv"

            self.mock_df.to_csv(c_path, index=False)
            try:
                self.mock_df.to_parquet(p_path)
                analyzer_p = DatasetOverviewAnalyzer(p_path)
                self.assertEqual(len(analyzer_p.df), 4)
            except Exception:
                p_path.write_bytes(b"mock_parquet")
                analyzer_p = DatasetOverviewAnalyzer(p_path)
                self.assertEqual(len(analyzer_p.df), 4)

            analyzer_c = DatasetOverviewAnalyzer(c_path)
            self.assertEqual(len(analyzer_c.df), 4)

    def test_init_errors(self):
        """Test file not found, empty DataFrame, and type error handling."""
        with self.assertRaises(FileNotFoundError):
            DatasetOverviewAnalyzer("non_existent_file.parquet")

        with self.assertRaises(ValueError):
            DatasetOverviewAnalyzer(pd.DataFrame())

        with self.assertRaises(TypeError):
            DatasetOverviewAnalyzer(12345)  # type: ignore

    def test_get_dimensions_summary(self):
        """Test computation of dimensions, memory footprint, and type breakdown."""
        analyzer = DatasetOverviewAnalyzer(self.mock_df)
        dims = analyzer.get_dimensions_summary()

        self.assertEqual(dims["total_rows"], 4)
        self.assertEqual(dims["total_columns"], 20)
        self.assertGreater(dims["memory_bytes"], 0)
        self.assertIn("dtype_breakdown", dims)

        dt = dims["dtype_breakdown"]
        self.assertEqual(dt["boolean"]["count"], 4)
        self.assertEqual(dt["datetime"]["count"], 2)
        self.assertIn("clean_body_word_count", dt["numeric_integer"]["columns"])

    def test_get_missing_summary(self):
        """Test missing value calculation and filtering."""
        analyzer = DatasetOverviewAnalyzer(self.mock_df)
        missing_df = analyzer.get_missing_summary(only_missing=False)

        self.assertEqual(len(missing_df), 20)
        res_row = missing_df[missing_df["column"] == "resolution_time_days"].iloc[0]
        self.assertEqual(res_row["missing_count"], 2)
        self.assertEqual(res_row["missing_pct"], 50.0)

        only_miss = analyzer.get_missing_summary(only_missing=True)
        self.assertTrue((only_miss["missing_count"] > 0).all())
        self.assertIn("state_reason", only_miss["column"].values)
        self.assertIn("closed_at", only_miss["column"].values)

    def test_get_repository_distribution(self):
        """Test repository counts and open/closed breakdowns."""
        analyzer = DatasetOverviewAnalyzer(self.mock_df)
        repo_df = analyzer.get_repository_distribution()

        self.assertEqual(len(repo_df), 2)
        row_a = repo_df[repo_df["repo_name"] == "repo/a"].iloc[0]
        self.assertEqual(row_a["count"], 2)
        self.assertEqual(row_a["pct"], 50.0)
        self.assertEqual(row_a["open_count"], 1)
        self.assertEqual(row_a["closed_count"], 1)

    def test_get_state_distribution(self):
        """Test global and per-repository state distributions."""
        analyzer = DatasetOverviewAnalyzer(self.mock_df)
        state_df = analyzer.get_state_distribution(by_repo=False)

        self.assertEqual(len(state_df), 2)
        self.assertEqual(int(state_df[state_df["state"] == "open"]["count"].iloc[0]), 2)
        self.assertEqual(int(state_df[state_df["state"] == "closed"]["count"].iloc[0]), 2)

        repo_state_df = analyzer.get_state_distribution(by_repo=True)
        self.assertEqual(len(repo_state_df), 2)
        self.assertIn("open", repo_state_df.columns)
        self.assertIn("closed", repo_state_df.columns)
        self.assertIn("open_pct", repo_state_df.columns)

    def test_get_label_distribution(self):
        """Test canonical label frequency aggregation."""
        analyzer = DatasetOverviewAnalyzer(self.mock_df)
        label_df = analyzer.get_label_distribution()

        bug_row = label_df[label_df["canonical_label"] == "bug"].iloc[0]
        self.assertEqual(bug_row["count"], 2)
        self.assertEqual(bug_row["pct"], 50.0)
        self.assertTrue(bug_row["is_classified_category"])

    def test_get_temporal_summary(self):
        """Test global and repository temporal span computations."""
        analyzer = DatasetOverviewAnalyzer(self.mock_df)
        temp = analyzer.get_temporal_summary()

        self.assertTrue(temp["global_earliest"].startswith("2026-01-01"))
        self.assertTrue(temp["global_latest"].startswith("2026-01-20"))
        self.assertEqual(temp["global_span_days"], 19.0)
        self.assertIn("repo/a", temp["by_repository"])
        self.assertEqual(temp["by_repository"]["repo/a"]["span_days"], 4.0)

    def test_get_numeric_summary(self):
        """Test 8-point numerical summary statistics calculation."""
        analyzer = DatasetOverviewAnalyzer(self.mock_df)
        num_df = analyzer.get_numeric_summary(["clean_body_word_count", "resolution_time_days"])

        self.assertEqual(len(num_df), 2)
        body_row = num_df[num_df["feature"] == "clean_body_word_count"].iloc[0]
        self.assertEqual(body_row["count"], 4)
        self.assertEqual(body_row["missing"], 0)
        self.assertEqual(body_row["mean"], 3.25)
        self.assertEqual(body_row["median"], 3.5)
        self.assertEqual(body_row["min"], 2.0)
        self.assertEqual(body_row["max"], 4.0)

        res_row = num_df[num_df["feature"] == "resolution_time_days"].iloc[0]
        self.assertEqual(res_row["count"], 2)
        self.assertEqual(res_row["missing"], 2)
        self.assertEqual(res_row["mean"], 3.5)
        self.assertEqual(res_row["median"], 3.5)

    def test_get_boolean_prevalence(self):
        """Test boolean indicator feature prevalence proportions."""
        analyzer = DatasetOverviewAnalyzer(self.mock_df)
        bool_df = analyzer.get_boolean_prevalence(["has_code_block", "has_stack_trace"])

        self.assertEqual(len(bool_df), 2)
        cb_row = bool_df[bool_df["feature"] == "has_code_block"].iloc[0]
        self.assertEqual(cb_row["true_count"], 2)
        self.assertEqual(cb_row["false_count"], 2)
        self.assertEqual(cb_row["true_pct"], 50.0)

        st_row = bool_df[bool_df["feature"] == "has_stack_trace"].iloc[0]
        self.assertEqual(st_row["true_count"], 1)
        self.assertEqual(st_row["false_count"], 3)
        self.assertEqual(st_row["true_pct"], 25.0)

    def test_compute_full_overview_and_json_export(self):
        """Test full dictionary compilation and JSON export."""
        analyzer = DatasetOverviewAnalyzer(self.mock_df)
        overview = analyzer.compute_full_overview()

        self.assertIn("metadata", overview)
        self.assertIn("dimensions", overview)
        self.assertIn("missing_features", overview)
        self.assertIn("repository_distribution", overview)
        self.assertIn("label_distribution", overview)

        # Test JSON serializability
        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = Path(tmpdir) / "overview.json"
            analyzer.save_summary_json(json_file)
            self.assertTrue(json_file.exists())

            with open(json_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["dimensions"]["total_rows"], 4)

    def test_generate_markdown_report(self):
        """Test generation and saving of Markdown summary report."""
        analyzer = DatasetOverviewAnalyzer(self.mock_df)
        with tempfile.TemporaryDirectory() as tmpdir:
            md_file = Path(tmpdir) / "report.md"
            report = analyzer.generate_markdown_report(md_file)

            self.assertTrue(md_file.exists())
            self.assertIn("# OpenSourceLens", report)
            self.assertIn("Executive Summary", report)
            self.assertIn("Target Label Distribution", report)
            self.assertIn("Missing Value Profile", report)


if __name__ == "__main__":
    unittest.main()
