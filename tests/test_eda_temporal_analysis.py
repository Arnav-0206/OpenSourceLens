"""Unit tests for ml.eda.temporal_analysis (IssueTemporalAnalyzer)."""

import json
import unittest
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np

from ml.eda.temporal_analysis import IssueTemporalAnalyzer


class TestIssueTemporalAnalyzer(unittest.TestCase):
    """Test suite for IssueTemporalAnalyzer."""

    def setUp(self):
        """Construct synthetic test dataset with known timestamps and repositories."""
        self.mock_df = pd.DataFrame({
            "repo_name": ["repo/a", "repo/a", "repo/b", "repo/b", "repo/b"],
            "issue_id": [301, 302, 303, 304, 305],
            "state": ["open", "closed", "closed", "open", "closed"],
            "created_at": pd.to_datetime([
                "2026-01-05T08:30:00Z",  # Monday, 08:00
                "2026-01-08T13:45:00Z",  # Thursday, 13:00
                "2026-02-10T22:15:00Z",  # Tuesday, 22:00
                "2026-02-15T00:10:00Z",  # Sunday (Weekend), 00:00
                "2026-03-20T14:00:00Z",  # Friday, 14:00
            ], utc=True),
        })

    def test_init_from_dataframe(self):
        """Test initialization with DataFrame."""
        analyzer = IssueTemporalAnalyzer(self.mock_df)
        self.assertEqual(len(analyzer.df), 5)
        self.assertIn("created_at_dt", analyzer.df.columns)
        self.assertIn("created_year_month", analyzer.df.columns)
        self.assertIn("is_weekend", analyzer.df.columns)

    def test_init_from_parquet_and_csv(self):
        """Test initialization from temporary Parquet and CSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p_path = Path(tmpdir) / "test.parquet"
            c_path = Path(tmpdir) / "test.csv"

            self.mock_df.to_csv(c_path, index=False)
            try:
                self.mock_df.to_parquet(p_path)
                analyzer_p = IssueTemporalAnalyzer(p_path)
                self.assertEqual(len(analyzer_p.df), 5)
            except Exception:
                p_path.write_bytes(b"mock_parquet")
                analyzer_p = IssueTemporalAnalyzer(p_path)
                self.assertEqual(len(analyzer_p.df), 5)

            analyzer_c = IssueTemporalAnalyzer(c_path)
            self.assertEqual(len(analyzer_c.df), 5)

    def test_init_errors(self):
        """Test file not found, empty DataFrame, and missing 'created_at' errors."""
        with self.assertRaises(FileNotFoundError):
            IssueTemporalAnalyzer("non_existent_file.parquet")

        with self.assertRaises(ValueError):
            IssueTemporalAnalyzer(pd.DataFrame())

        with self.assertRaises(TypeError):
            IssueTemporalAnalyzer(12345)  # type: ignore

        with self.assertRaises(KeyError):
            IssueTemporalAnalyzer(pd.DataFrame({"repo_name": ["a", "b"]}))

    def test_get_monthly_trends_global(self):
        """Test global Year-Month volume trends."""
        analyzer = IssueTemporalAnalyzer(self.mock_df)
        monthly_df = analyzer.get_monthly_trends(by_repo=False)

        self.assertEqual(len(monthly_df), 3)  # Jan, Feb, Mar 2026
        jan_row = monthly_df[monthly_df["year_month"] == "2026-01"].iloc[0]
        self.assertEqual(jan_row["issue_count"], 2)
        self.assertEqual(jan_row["pct_of_total"], 40.0)

        mar_row = monthly_df[monthly_df["year_month"] == "2026-03"].iloc[0]
        self.assertEqual(mar_row["cumulative_count"], 5)
        self.assertEqual(mar_row["cumulative_pct"], 100.0)

    def test_get_monthly_trends_by_repo(self):
        """Test repository decomposition of monthly trends."""
        analyzer = IssueTemporalAnalyzer(self.mock_df)
        ct = analyzer.get_monthly_trends(by_repo=True)

        self.assertEqual(len(ct), 3)
        self.assertIn("repo/a", ct.columns)
        self.assertIn("repo/b", ct.columns)
        self.assertIn("total", ct.columns)

    def test_get_weekly_trends(self):
        """Test weekly aggregation."""
        analyzer = IssueTemporalAnalyzer(self.mock_df)
        weekly_df = analyzer.get_weekly_trends(year=2026)

        self.assertGreater(len(weekly_df), 0)
        self.assertEqual(weekly_df["issue_count"].sum(), 5)

    def test_get_day_of_week_distribution(self):
        """Test day-of-week breakdown and weekend proportions."""
        analyzer = IssueTemporalAnalyzer(self.mock_df)
        dow_df = analyzer.get_day_of_week_distribution(by_repo=False)

        self.assertEqual(len(dow_df), 7)
        sun_row = dow_df[dow_df["day_name"] == "Sunday"].iloc[0]
        self.assertEqual(sun_row["issue_count"], 1)
        self.assertTrue(sun_row["is_weekend"])

        mon_row = dow_df[dow_df["day_name"] == "Monday"].iloc[0]
        self.assertEqual(mon_row["issue_count"], 1)
        self.assertFalse(mon_row["is_weekend"])

    def test_get_day_of_week_by_repo(self):
        """Test repository decomposition of day of week."""
        analyzer = IssueTemporalAnalyzer(self.mock_df)
        dow_repo_df = analyzer.get_day_of_week_distribution(by_repo=True)

        self.assertEqual(len(dow_repo_df), 7)
        self.assertIn("day_name", dow_repo_df.columns)
        self.assertIn("total", dow_repo_df.columns)

    def test_get_hourly_distribution(self):
        """Test 24-hour UTC distribution."""
        analyzer = IssueTemporalAnalyzer(self.mock_df)
        hourly_df = analyzer.get_hourly_distribution()

        self.assertEqual(len(hourly_df), 24)
        h8_row = hourly_df[hourly_df["hour_utc"] == 8].iloc[0]
        self.assertEqual(h8_row["issue_count"], 1)

    def test_get_state_evolution_over_time(self):
        """Test quarterly state evolution."""
        analyzer = IssueTemporalAnalyzer(self.mock_df)
        evo_df = analyzer.get_state_evolution_over_time()

        self.assertEqual(len(evo_df), 1)  # All are 2026-Q1
        row = evo_df.iloc[0]
        self.assertEqual(row["period"], "2026-Q1")
        self.assertEqual(row["total"], 5)
        self.assertEqual(row["open"], 2)
        self.assertEqual(row["closed"], 3)

    def test_get_repository_temporal_profiles(self):
        """Test repository spans, boundaries, and velocity."""
        analyzer = IssueTemporalAnalyzer(self.mock_df)
        profiles_df = analyzer.get_repository_temporal_profiles()

        self.assertEqual(len(profiles_df), 2)
        self.assertIn("velocity_issues_per_day", profiles_df.columns)
        self.assertIn("span_days", profiles_df.columns)

    def test_compute_full_temporal_analysis_and_json(self):
        """Test compiling complete analysis and exporting JSON."""
        analyzer = IssueTemporalAnalyzer(self.mock_df)
        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = Path(tmpdir) / "temporal_analysis.json"
            analyzer.save_summary_json(json_file)

            self.assertTrue(json_file.exists())
            with open(json_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["global_boundaries"]["total_issues"], 5)
            self.assertIn("seasonality", loaded)
            self.assertIn("monthly_trends_global", loaded)

    def test_generate_markdown_report(self):
        """Test generating and saving Markdown temporal report."""
        analyzer = IssueTemporalAnalyzer(self.mock_df)
        with tempfile.TemporaryDirectory() as tmpdir:
            md_file = Path(tmpdir) / "temporal_report.md"
            report = analyzer.generate_markdown_report(md_file)

            self.assertTrue(md_file.exists())
            self.assertIn("# OpenSourceLens — Phase 3, Step 3.4", report)
            self.assertIn("Repository Temporal Profiles", report)
            self.assertIn("Day-of-Week Seasonality", report)
            self.assertIn("Diurnal Hourly Distribution", report)


if __name__ == "__main__":
    unittest.main()
