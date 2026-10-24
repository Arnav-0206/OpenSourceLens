"""Unit tests for ml.eda.repo_comparison (RepositoryComparisonAnalyzer)."""

import json
import unittest
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np

from ml.eda.repo_comparison import RepositoryComparisonAnalyzer


class TestRepositoryComparisonAnalyzer(unittest.TestCase):
    """Test suite for RepositoryComparisonAnalyzer."""

    def setUp(self):
        """Construct synthetic multi-repository test dataset."""
        self.mock_df = pd.DataFrame({
            "repo_name": ["repo/alpha", "repo/alpha", "repo/beta", "repo/beta", "repo/beta"],
            "issue_id": [401, 402, 403, 404, 405],
            "state": ["open", "closed", "closed", "open", "closed"],
            "state_reason": ["unresolved", "completed", "completed", "reopened", "duplicate"],
            "canonical_label": ["bug", "bug", "feature", "documentation", "feature"],
            "created_at": pd.to_datetime([
                "2026-01-01T00:00:00Z",
                "2026-01-03T00:00:00Z",
                "2026-01-01T00:00:00Z",
                "2026-01-05T00:00:00Z",
                "2026-01-11T00:00:00Z",
            ], utc=True),
            "resolution_time_days": [np.nan, 2.0, 4.0, np.nan, 0.5],
            "clean_title_word_count": [4, 5, 6, 3, 5],
            "clean_body_word_count": [100, 150, 200, 80, 120],
            "has_code_block": [True, True, False, False, True],
            "has_stack_trace": [True, False, False, False, False],
            "has_url": [False, True, True, False, True],
            "title_has_bracket_tag": [True, False, False, False, False],
            "comments_count": [3, 5, 2, 8, 1],
            "reactions_count": [0, 2, 1, 0, 0],
            "is_internal_author": [True, False, False, False, True],
            "is_bot_author": [False, False, False, True, False],
            "author": ["alice", "bob", "charlie", "bot_user", "alice"],
        })

    def test_init_from_dataframe(self):
        """Test initialization with DataFrame."""
        analyzer = RepositoryComparisonAnalyzer(self.mock_df)
        self.assertEqual(len(analyzer.df), 5)
        self.assertEqual(analyzer.repo_col, "repo_name")

    def test_init_from_parquet_and_csv(self):
        """Test initialization from temporary Parquet and CSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p_path = Path(tmpdir) / "test.parquet"
            c_path = Path(tmpdir) / "test.csv"

            self.mock_df.to_csv(c_path, index=False)
            try:
                self.mock_df.to_parquet(p_path)
                analyzer_p = RepositoryComparisonAnalyzer(p_path)
                self.assertEqual(len(analyzer_p.df), 5)
            except Exception:
                p_path.write_bytes(b"mock_parquet")
                analyzer_p = RepositoryComparisonAnalyzer(p_path)
                self.assertEqual(len(analyzer_p.df), 5)

            analyzer_c = RepositoryComparisonAnalyzer(c_path)
            self.assertEqual(len(analyzer_c.df), 5)

    def test_init_errors(self):
        """Test file not found, empty DataFrame, and missing repository column errors."""
        with self.assertRaises(FileNotFoundError):
            RepositoryComparisonAnalyzer("non_existent_file.parquet")

        with self.assertRaises(ValueError):
            RepositoryComparisonAnalyzer(pd.DataFrame())

        with self.assertRaises(TypeError):
            RepositoryComparisonAnalyzer(12345)  # type: ignore

        with self.assertRaises(KeyError):
            RepositoryComparisonAnalyzer(pd.DataFrame({"issue_id": [1, 2]}))

    def test_get_overview_scorecard(self):
        """Test high-level repository scorecard calculation."""
        analyzer = RepositoryComparisonAnalyzer(self.mock_df)
        scorecard = analyzer.get_overview_scorecard()

        self.assertEqual(len(scorecard), 2)
        row_beta = scorecard[scorecard["repo_name"] == "repo/beta"].iloc[0]
        self.assertEqual(row_beta["total_issues"], 3)
        self.assertEqual(row_beta["pct_of_dataset"], 60.0)
        self.assertEqual(row_beta["closed_issues"], 2)
        self.assertEqual(row_beta["closure_rate"], 0.6667)
        self.assertEqual(row_beta["span_days"], 10.0)

    def test_get_lifecycle_comparison(self):
        """Test resolution time and closure reason comparison."""
        analyzer = RepositoryComparisonAnalyzer(self.mock_df)
        life_df = analyzer.get_lifecycle_comparison()

        self.assertEqual(len(life_df), 2)
        row_alpha = life_df[life_df["repo_name"] == "repo/alpha"].iloc[0]
        self.assertEqual(row_alpha["closed_issues"], 1)
        self.assertEqual(row_alpha["median_resolution_days"], 2.0)
        self.assertEqual(row_alpha["completed_pct"], 100.0)

        row_beta = life_df[life_df["repo_name"] == "repo/beta"].iloc[0]
        self.assertEqual(row_beta["closed_issues"], 2)
        self.assertEqual(row_beta["median_resolution_days"], 2.25)  # median of 4.0 and 0.5
        self.assertEqual(row_beta["completed_pct"], 50.0)
        self.assertEqual(row_beta["duplicate_pct"], 50.0)

    def test_get_label_profile_comparison(self):
        """Test label category shares and candidate proportions."""
        analyzer = RepositoryComparisonAnalyzer(self.mock_df)
        label_df = analyzer.get_label_profile_comparison()

        self.assertEqual(len(label_df), 2)
        alpha_lbl = label_df[label_df["repo_name"] == "repo/alpha"].iloc[0]
        self.assertEqual(alpha_lbl["bug_pct"], 100.0)
        self.assertEqual(alpha_lbl["feature_pct"], 0.0)
        self.assertEqual(alpha_lbl["candidate_pct"], 100.0)

        beta_lbl = label_df[label_df["repo_name"] == "repo/beta"].iloc[0]
        self.assertEqual(beta_lbl["feature_pct"], 66.67)
        self.assertEqual(beta_lbl["documentation_pct"], 33.33)

    def test_get_text_complexity_comparison(self):
        """Test word counts and code/trace indicators."""
        analyzer = RepositoryComparisonAnalyzer(self.mock_df)
        text_df = analyzer.get_text_complexity_comparison()

        self.assertEqual(len(text_df), 2)
        alpha_text = text_df[text_df["repo_name"] == "repo/alpha"].iloc[0]
        self.assertEqual(alpha_text["code_block_pct"], 100.0)
        self.assertEqual(alpha_text["stack_trace_pct"], 50.0)
        self.assertEqual(alpha_text["bracket_tag_pct"], 50.0)

        beta_text = text_df[text_df["repo_name"] == "repo/beta"].iloc[0]
        self.assertEqual(beta_text["code_block_pct"], 33.33)
        self.assertEqual(beta_text["stack_trace_pct"], 0.0)

    def test_get_collaboration_comparison(self):
        """Test comments, reactions, author profiles, and contributor diversity."""
        analyzer = RepositoryComparisonAnalyzer(self.mock_df)
        collab_df = analyzer.get_collaboration_comparison()

        self.assertEqual(len(collab_df), 2)
        alpha_c = collab_df[collab_df["repo_name"] == "repo/alpha"].iloc[0]
        self.assertEqual(alpha_c["comments_mean"], 4.0)
        self.assertEqual(alpha_c["internal_author_pct"], 50.0)
        self.assertEqual(alpha_c["unique_authors_count"], 2)
        self.assertEqual(alpha_c["author_diversity_pct"], 100.0)

        beta_c = collab_df[collab_df["repo_name"] == "repo/beta"].iloc[0]
        self.assertEqual(beta_c["bot_author_pct"], 33.33)
        self.assertEqual(beta_c["unique_authors_count"], 3)  # charlie, bot_user, alice
        self.assertEqual(beta_c["author_diversity_pct"], 100.0)

    def test_compute_full_comparison_and_json(self):
        """Test compiling complete analysis and exporting JSON."""
        analyzer = RepositoryComparisonAnalyzer(self.mock_df)
        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = Path(tmpdir) / "comparison.json"
            analyzer.save_summary_json(json_file)

            self.assertTrue(json_file.exists())
            with open(json_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertIn("overview_scorecard", loaded)
            self.assertIn("lifecycle_and_resolution", loaded)
            self.assertIn("text_complexity", loaded)

    def test_generate_markdown_report(self):
        """Test generating and saving Markdown comparison report."""
        analyzer = RepositoryComparisonAnalyzer(self.mock_df)
        with tempfile.TemporaryDirectory() as tmpdir:
            md_file = Path(tmpdir) / "comparison_report.md"
            report = analyzer.generate_markdown_report(md_file)

            self.assertTrue(md_file.exists())
            self.assertIn("# OpenSourceLens — Phase 3, Step 3.5", report)
            self.assertIn("High-Level Repository Scorecard", report)
            self.assertIn("Lifecycle Dynamics & Resolution Speed", report)
            self.assertIn("Collaboration, Authorship & Contributor Diversity", report)


if __name__ == "__main__":
    unittest.main()
