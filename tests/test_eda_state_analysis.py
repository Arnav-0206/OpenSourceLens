"""Unit tests for ml.eda.state_analysis (IssueStateAnalyzer)."""

import json
import unittest
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np

from ml.eda.state_analysis import IssueStateAnalyzer


class TestIssueStateAnalyzer(unittest.TestCase):
    """Test suite for IssueStateAnalyzer."""

    def setUp(self):
        """Construct synthetic test dataset mimicking issues_master schema."""
        self.mock_df = pd.DataFrame({
            "repo_name": ["repo/a", "repo/a", "repo/b", "repo/b", "repo/b"],
            "issue_id": [101, 102, 103, 104, 105],
            "state": ["open", "closed", "closed", "open", "closed"],
            "state_reason": ["unresolved", "completed", "duplicate", "reopened", "not_planned"],
            "canonical_label": ["bug", "bug", "feature", "feature", "bug"],
            "resolution_time_days": [np.nan, 3.0, 0.5, np.nan, 1.5],
            "issue_age_days": [10.0, 3.0, 0.5, 40.0, 1.5],
            "comments_count": [2, 4, 1, 10, 0],
            "reactions_count": [0, 2, 0, 1, 0],
            "interaction_total": [2, 6, 1, 11, 0],
        })

    def test_init_from_dataframe(self):
        """Test initialization with DataFrame."""
        analyzer = IssueStateAnalyzer(self.mock_df)
        self.assertEqual(len(analyzer.df), 5)
        self.assertEqual(analyzer.source_path, "DataFrame in-memory")

    def test_init_from_parquet_and_csv(self):
        """Test initialization from temporary Parquet and CSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p_path = Path(tmpdir) / "test.parquet"
            c_path = Path(tmpdir) / "test.csv"

            self.mock_df.to_csv(c_path, index=False)
            try:
                self.mock_df.to_parquet(p_path)
                analyzer_p = IssueStateAnalyzer(p_path)
                self.assertEqual(len(analyzer_p.df), 5)
            except Exception:
                p_path.write_bytes(b"mock_parquet")
                analyzer_p = IssueStateAnalyzer(p_path)
                self.assertEqual(len(analyzer_p.df), 5)

            analyzer_c = IssueStateAnalyzer(c_path)
            self.assertEqual(len(analyzer_c.df), 5)

    def test_init_errors(self):
        """Test invalid inputs and missing 'state' column."""
        with self.assertRaises(FileNotFoundError):
            IssueStateAnalyzer("non_existent_file.parquet")

        with self.assertRaises(ValueError):
            IssueStateAnalyzer(pd.DataFrame())

        with self.assertRaises(TypeError):
            IssueStateAnalyzer(12345)  # type: ignore

        with self.assertRaises(KeyError):
            IssueStateAnalyzer(pd.DataFrame({"repo_name": ["a", "b"]}))

    def test_get_global_state_summary(self):
        """Test global counts and closure rate calculation."""
        analyzer = IssueStateAnalyzer(self.mock_df)
        summary = analyzer.get_global_state_summary()

        self.assertEqual(summary["total_issues"], 5)
        self.assertEqual(summary["open_count"], 2)
        self.assertEqual(summary["closed_count"], 3)
        self.assertEqual(summary["open_pct"], 40.0)
        self.assertEqual(summary["closed_pct"], 60.0)
        self.assertEqual(summary["closure_rate"], 0.6)

    def test_get_state_by_repository(self):
        """Test cross-tabulation by repository."""
        analyzer = IssueStateAnalyzer(self.mock_df)
        repo_df = analyzer.get_state_by_repository()

        self.assertEqual(len(repo_df), 2)
        repo_a = repo_df[repo_df["repo_name"] == "repo/a"].iloc[0]
        self.assertEqual(repo_a["total"], 2)
        self.assertEqual(repo_a["open"], 1)
        self.assertEqual(repo_a["closed"], 1)
        self.assertEqual(repo_a["closure_rate"], 0.5)

        repo_b = repo_df[repo_df["repo_name"] == "repo/b"].iloc[0]
        self.assertEqual(repo_b["total"], 3)
        self.assertEqual(repo_b["closed"], 2)
        self.assertEqual(repo_b["closure_rate"], 0.6667)

    def test_get_state_reason_breakdown(self):
        """Test decomposition into closed and open state reasons."""
        analyzer = IssueStateAnalyzer(self.mock_df)
        reasons = analyzer.get_state_reason_breakdown()

        self.assertEqual(len(reasons["closed_reasons"]), 3)
        completed = next(r for r in reasons["closed_reasons"] if r["reason"] == "completed")
        self.assertEqual(completed["count"], 1)
        self.assertEqual(completed["pct_of_closed"], 33.33)

        self.assertEqual(len(reasons["open_reasons"]), 2)
        reopened = next(r for r in reasons["open_reasons"] if r["reason"] == "reopened")
        self.assertEqual(reopened["count"], 1)
        self.assertEqual(reopened["pct_of_open"], 50.0)

    def test_get_closure_reasons_by_repository(self):
        """Test repo × closure reason cross-tabulation."""
        analyzer = IssueStateAnalyzer(self.mock_df)
        ct = analyzer.get_closure_reasons_by_repository()

        self.assertEqual(len(ct), 2)
        row_b = ct[ct["repo_name"] == "repo/b"].iloc[0]
        self.assertEqual(row_b["total_closed"], 2)
        self.assertEqual(row_b["duplicate"], 1)
        self.assertEqual(row_b["not_planned"], 1)
        self.assertEqual(row_b["duplicate_pct"], 50.0)

    def test_get_state_by_label(self):
        """Test label closure rate disparities."""
        analyzer = IssueStateAnalyzer(self.mock_df)
        label_df = analyzer.get_state_by_label()

        self.assertEqual(len(label_df), 2)
        bug_row = label_df[label_df["canonical_label"] == "bug"].iloc[0]
        self.assertEqual(bug_row["total"], 3)
        self.assertEqual(bug_row["closed"], 2)
        self.assertEqual(bug_row["closure_rate"], 0.6667)

        feat_row = label_df[label_df["canonical_label"] == "feature"].iloc[0]
        self.assertEqual(feat_row["total"], 2)
        self.assertEqual(feat_row["open"], 1)
        self.assertEqual(feat_row["closure_rate"], 0.5)

    def test_get_resolution_by_closure_reason(self):
        """Test resolution time statistics partitioned by closure reason."""
        analyzer = IssueStateAnalyzer(self.mock_df)
        res_df = analyzer.get_resolution_by_closure_reason()

        self.assertGreaterEqual(len(res_df), 3)
        comp = res_df[res_df["state_reason"] == "completed"].iloc[0]
        self.assertEqual(comp["count"], 1)
        self.assertEqual(comp["mean"], 3.0)

        dup = res_df[res_df["state_reason"] == "duplicate"].iloc[0]
        self.assertEqual(dup["count"], 1)
        self.assertEqual(dup["mean"], 0.5)

        all_closed = res_df[res_df["state_reason"] == "ALL_CLOSED"].iloc[0]
        self.assertEqual(all_closed["count"], 3)
        self.assertEqual(all_closed["median"], 1.5)

    def test_get_backlog_aging_summary(self):
        """Test open issue backlog age and closed issue lifespan calculations."""
        analyzer = IssueStateAnalyzer(self.mock_df)
        aging = analyzer.get_backlog_aging_summary()

        self.assertIn("closed_issues_lifespan", aging)
        self.assertIn("open_issues_backlog_age", aging)
        self.assertIn("by_repository", aging)

        open_age = aging["open_issues_backlog_age"]
        self.assertEqual(open_age["count"], 2)
        self.assertEqual(open_age["median"], 25.0)
        self.assertEqual(open_age["max"], 40.0)

    def test_get_engagement_by_state(self):
        """Test comments and reactions comparison between states."""
        analyzer = IssueStateAnalyzer(self.mock_df)
        eng_df = analyzer.get_engagement_by_state()

        self.assertGreater(len(eng_df), 0)
        open_comments = eng_df[(eng_df["metric"] == "comments_count") & (eng_df["state"] == "open")].iloc[0]
        self.assertEqual(open_comments["count"], 2)
        self.assertEqual(open_comments["mean"], 6.0)

    def test_compute_full_state_analysis_and_json(self):
        """Test compiling complete state analysis and saving to JSON."""
        analyzer = IssueStateAnalyzer(self.mock_df)
        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = Path(tmpdir) / "state_analysis.json"
            analyzer.save_summary_json(json_file)

            self.assertTrue(json_file.exists())
            with open(json_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["global_state_summary"]["total_issues"], 5)
            self.assertIn("state_reason_breakdown", loaded)
            self.assertIn("resolution_duration_by_closure_reason", loaded)

    def test_generate_markdown_report(self):
        """Test generating and saving Markdown state analysis report."""
        analyzer = IssueStateAnalyzer(self.mock_df)
        with tempfile.TemporaryDirectory() as tmpdir:
            md_file = Path(tmpdir) / "state_report.md"
            report = analyzer.generate_markdown_report(md_file)

            self.assertTrue(md_file.exists())
            self.assertIn("# OpenSourceLens — Phase 3, Step 3.2", report)
            self.assertIn("Global Closure Rate", report)
            self.assertIn("Detailed State Reason Decomposition", report)
            self.assertIn("Resolution Duration by Closure Reason", report)


if __name__ == "__main__":
    unittest.main()
