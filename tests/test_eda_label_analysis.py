"""Unit tests for ml.eda.label_analysis (IssueLabelAnalyzer)."""

import json
import unittest
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np

from ml.eda.label_analysis import IssueLabelAnalyzer


class TestIssueLabelAnalyzer(unittest.TestCase):
    """Test suite for IssueLabelAnalyzer."""

    def setUp(self):
        """Construct synthetic test dataset with various label configurations."""
        self.mock_df = pd.DataFrame({
            "repo_name": ["repo/a", "repo/a", "repo/b", "repo/b", "repo/b"],
            "issue_id": [201, 202, 203, 204, 205],
            "state": ["open", "closed", "closed", "open", "closed"],
            "raw_labels": [
                ["kind/bug", "sig/node"],
                ["kind/bug", "sig/node", "triage/accepted"],
                ["feature", "ui"],
                [],
                ["feature"],
            ],
            "labels_count": [2, 3, 2, 0, 1],
            "canonical_label": ["bug", "bug", "feature", "unlabeled", "feature"],
            "is_classified_category": [True, True, True, False, True],
            "resolution_time_days": [np.nan, 3.5, 1.2, np.nan, 0.8],
        })

    def test_init_from_dataframe(self):
        """Test initialization directly from DataFrame."""
        analyzer = IssueLabelAnalyzer(self.mock_df)
        self.assertEqual(len(analyzer.df), 5)
        self.assertIn("raw_labels_clean", analyzer.df.columns)
        self.assertIn("label_bucket", analyzer.df.columns)

    def test_init_from_parquet_and_csv(self):
        """Test initialization from temporary Parquet and CSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p_path = Path(tmpdir) / "test.parquet"
            c_path = Path(tmpdir) / "test.csv"

            # Parquet preserves list type; CSV saves as string representation
            self.mock_df.to_csv(c_path, index=False)
            try:
                self.mock_df.to_parquet(p_path)
                analyzer_p = IssueLabelAnalyzer(p_path)
                self.assertEqual(len(analyzer_p.df), 5)
            except Exception:
                p_path.write_bytes(b"mock_parquet")
                analyzer_p = IssueLabelAnalyzer(p_path)
                self.assertEqual(len(analyzer_p.df), 5)

            analyzer_c = IssueLabelAnalyzer(c_path)
            self.assertEqual(len(analyzer_c.df), 5)
            self.assertEqual(analyzer_c.df["raw_labels_clean"].iloc[0], ["kind/bug", "sig/node"])

    def test_init_errors(self):
        """Test file not found, empty DataFrame, and type error handling."""
        with self.assertRaises(FileNotFoundError):
            IssueLabelAnalyzer("non_existent_file.parquet")

        with self.assertRaises(ValueError):
            IssueLabelAnalyzer(pd.DataFrame())

        with self.assertRaises(TypeError):
            IssueLabelAnalyzer(12345)  # type: ignore

    def test_parse_labels_list(self):
        """Test robust label string/list/array parsing."""
        self.assertEqual(IssueLabelAnalyzer._parse_labels_list(None), [])
        self.assertEqual(IssueLabelAnalyzer._parse_labels_list(["a", "b"]), ["a", "b"])
        self.assertEqual(IssueLabelAnalyzer._parse_labels_list(np.array(["x", "y"])), ["x", "y"])
        self.assertEqual(IssueLabelAnalyzer._parse_labels_list("['tag1', 'tag2']"), ["tag1", "tag2"])
        self.assertEqual(IssueLabelAnalyzer._parse_labels_list("single_label"), ["single_label"])

    def test_get_label_cardinality_summary(self):
        """Test summary statistics and cardinality buckets."""
        analyzer = IssueLabelAnalyzer(self.mock_df)
        summary = analyzer.get_label_cardinality_summary()

        self.assertEqual(summary["statistics"]["count"], 5)
        self.assertEqual(summary["statistics"]["mean"], 1.6)
        self.assertEqual(summary["statistics"]["max"], 3.0)
        self.assertEqual(summary["statistics"]["min"], 0.0)

        self.assertEqual(summary["unlabeled_issues"]["count"], 1)
        self.assertEqual(summary["single_label_issues"]["count"], 1)
        self.assertEqual(summary["multi_label_issues"]["count"], 3)

        bucket_dict = {b["bucket"]: b["count"] for b in summary["buckets"]}
        self.assertEqual(bucket_dict.get("0 (Unlabeled)"), 1)
        self.assertEqual(bucket_dict.get("1 Label"), 1)
        self.assertEqual(bucket_dict.get("2-3 Labels"), 3)

    def test_get_top_raw_labels_global(self):
        """Test global raw label frequencies and rankings."""
        analyzer = IssueLabelAnalyzer(self.mock_df)
        top_df = analyzer.get_top_raw_labels(top_n=5, by_repo=False)

        self.assertGreater(len(top_df), 0)
        top_label = top_df.iloc[0]
        # kind/bug, sig/node, and feature each appear 2 times
        self.assertIn(top_label["label"], ["kind/bug", "sig/node", "feature"])
        self.assertEqual(top_label["count"], 2)

    def test_get_top_raw_labels_by_repo(self):
        """Test repository decomposition of raw labels."""
        analyzer = IssueLabelAnalyzer(self.mock_df)
        repo_labels = analyzer.get_top_raw_labels(top_n=5, by_repo=True)

        self.assertIn("repo/a", repo_labels)
        self.assertIn("repo/b", repo_labels)
        repo_a_labels = [r["label"] for r in repo_labels["repo/a"]]
        self.assertIn("kind/bug", repo_a_labels)
        self.assertIn("sig/node", repo_a_labels)

    def test_get_label_co_occurrences(self):
        """Test calculation of pairwise label co-occurrences."""
        analyzer = IssueLabelAnalyzer(self.mock_df)
        co_df = analyzer.get_label_co_occurrences(top_n=5)

        self.assertGreater(len(co_df), 0)
        # kind/bug and sig/node co-occur twice (issues 201 and 202)
        top_co = co_df.iloc[0]
        self.assertEqual(top_co["label_1"], "kind/bug")
        self.assertEqual(top_co["label_2"], "sig/node")
        self.assertEqual(top_co["co_occurrence_count"], 2)

    def test_get_canonical_category_breakdown(self):
        """Test canonical label counts and candidate subset."""
        analyzer = IssueLabelAnalyzer(self.mock_df)
        cat_df = analyzer.get_canonical_category_breakdown()

        self.assertEqual(len(cat_df), 3)
        bug_row = cat_df[cat_df["canonical_label"] == "bug"].iloc[0]
        self.assertEqual(bug_row["count"], 2)
        self.assertTrue(bug_row["is_classified_category"])

        unlab_row = cat_df[cat_df["canonical_label"] == "unlabeled"].iloc[0]
        self.assertEqual(unlab_row["count"], 1)
        self.assertFalse(unlab_row["is_classified_category"])

    def test_get_cardinality_vs_resolution(self):
        """Test resolution duration stats across label count buckets."""
        analyzer = IssueLabelAnalyzer(self.mock_df)
        res_df = analyzer.get_cardinality_vs_resolution()

        self.assertGreater(len(res_df), 0)
        b_2_3 = res_df[res_df["label_bucket"] == "2-3 Labels"].iloc[0]
        # Issues 202 (3.5) and 203 (1.2) are closed in bucket 2-3 Labels
        self.assertEqual(b_2_3["count"], 2)
        self.assertEqual(b_2_3["mean"], 2.35)

    def test_get_cardinality_vs_state(self):
        """Test closure rates across label buckets."""
        analyzer = IssueLabelAnalyzer(self.mock_df)
        state_df = analyzer.get_cardinality_vs_state()

        self.assertGreater(len(state_df), 0)
        self.assertIn("closure_rate", state_df.columns)
        b_2_3 = state_df[state_df["label_bucket"] == "2-3 Labels"].iloc[0]
        self.assertEqual(b_2_3["total"], 3)
        self.assertEqual(b_2_3["closed"], 2)
        self.assertEqual(b_2_3["closure_rate"], 0.6667)

    def test_compute_full_label_analysis_and_json(self):
        """Test compiling complete analysis and exporting JSON."""
        analyzer = IssueLabelAnalyzer(self.mock_df)
        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = Path(tmpdir) / "label_analysis.json"
            analyzer.save_summary_json(json_file)

            self.assertTrue(json_file.exists())
            with open(json_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertIn("cardinality_summary", loaded)
            self.assertIn("top_raw_labels_global", loaded)
            self.assertIn("label_co_occurrences", loaded)
            self.assertIn("canonical_category_breakdown", loaded)

    def test_generate_markdown_report(self):
        """Test generating and saving Markdown label report."""
        analyzer = IssueLabelAnalyzer(self.mock_df)
        with tempfile.TemporaryDirectory() as tmpdir:
            md_file = Path(tmpdir) / "label_report.md"
            report = analyzer.generate_markdown_report(md_file)

            self.assertTrue(md_file.exists())
            self.assertIn("# OpenSourceLens — Phase 3, Step 3.3", report)
            self.assertIn("Label Cardinality & Density Distribution", report)
            self.assertIn("Top 25 Raw Labels", report)
            self.assertIn("Pairwise Label Co-Occurrences", report)


if __name__ == "__main__":
    unittest.main()
