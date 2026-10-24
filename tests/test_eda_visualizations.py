"""Unit tests for ml.eda.visualizations (IssueVisualizer)."""

import unittest
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np

from ml.eda.visualizations import IssueVisualizer


class TestIssueVisualizer(unittest.TestCase):
    """Test suite for IssueVisualizer."""

    def setUp(self):
        """Construct synthetic test dataset with varied categories, dates, and metrics."""
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
            "created_at": [
                "2026-01-01T10:00:00Z", "2026-01-05T12:00:00Z", "2026-01-10T14:00:00Z", "2026-01-15T16:00:00Z",
                "2026-02-01T08:00:00Z", "2026-02-05T10:00:00Z", "2026-02-10T12:00:00Z", "2026-02-15T14:00:00Z",
            ],
            "clean_title_word_count": [5, 8, 4, 6, 7, 9, 3, 5],
            "clean_body_word_count": [25, 120, 45, 80, 210, 65, 30, 95],
            "comments_count": [0, 1, 3, 5, 8, 15, 0, 2],
            "reactions_count": [0, 2, 0, 5, 10, 20, 0, 1],
            "interaction_total": [0, 3, 3, 10, 18, 35, 0, 3],
            "labels_count": [1, 2, 1, 3, 2, 1, 2, 1],
            "code_block_count": [0, 2, 0, 1, 4, 1, 0, 1],
            "url_count": [0, 1, 0, 2, 1, 0, 0, 1],
            "has_code_block": [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
            "is_weekend": [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            "resolution_time_days": [np.nan, 0.5, 0.2, 5.0, np.nan, 20.0, np.nan, 2.0],
        })

    def test_init_from_dataframe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = IssueVisualizer(self.mock_df, output_dir=tmpdir)
            self.assertEqual(len(visualizer.df), 8)
            self.assertEqual(visualizer.source_path, "DataFrame in-memory")
            self.assertTrue(Path(tmpdir).exists())

    def test_init_from_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            out_dir = Path(tmpdir) / "figures"
            self.mock_df.to_csv(csv_path, index=False)
            visualizer = IssueVisualizer(csv_path, output_dir=out_dir)
            self.assertEqual(len(visualizer.df), 8)
            self.assertEqual(visualizer.source_path, str(csv_path))
            self.assertTrue(out_dir.exists())

    def test_init_from_parquet_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            parquet_path = Path(tmpdir) / "test_data.parquet"
            out_dir = Path(tmpdir) / "figures"
            self.mock_df.to_csv(csv_path, index=False)
            parquet_path.write_bytes(b"MOCK_PARQUET_FALLBACK")
            visualizer = IssueVisualizer(parquet_path, output_dir=out_dir)
            self.assertEqual(len(visualizer.df), 8)

    def test_init_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            IssueVisualizer("non_existent_file.csv")

    def test_init_invalid_type(self):
        with self.assertRaises(TypeError):
            IssueVisualizer(12345)  # type: ignore

    def test_init_empty_dataframe(self):
        with self.assertRaises(ValueError):
            IssueVisualizer(pd.DataFrame())

    def test_plot_label_distribution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = IssueVisualizer(self.mock_df, output_dir=tmpdir)
            target = visualizer.plot_label_distribution()
            self.assertTrue(target.exists())
            self.assertGreater(target.stat().st_size, 1000)
            self.assertEqual(target.name, "fig1_label_distribution.png")

    def test_plot_state_and_reasons(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = IssueVisualizer(self.mock_df, output_dir=tmpdir)
            target = visualizer.plot_state_and_reasons()
            self.assertTrue(target.exists())
            self.assertGreater(target.stat().st_size, 1000)
            self.assertEqual(target.name, "fig2_state_reasons.png")

    def test_plot_repository_comparison(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = IssueVisualizer(self.mock_df, output_dir=tmpdir)
            target = visualizer.plot_repository_comparison()
            self.assertTrue(target.exists())
            self.assertGreater(target.stat().st_size, 1000)
            self.assertEqual(target.name, "fig3_repo_comparison.png")

    def test_plot_temporal_trends(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = IssueVisualizer(self.mock_df, output_dir=tmpdir)
            target = visualizer.plot_temporal_trends()
            self.assertTrue(target.exists())
            self.assertGreater(target.stat().st_size, 1000)
            self.assertEqual(target.name, "fig4_temporal_volume.png")

    def test_plot_text_distributions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = IssueVisualizer(self.mock_df, output_dir=tmpdir)
            target = visualizer.plot_text_distributions()
            self.assertTrue(target.exists())
            self.assertGreater(target.stat().st_size, 1000)
            self.assertEqual(target.name, "fig5_text_distributions.png")

    def test_plot_comment_buckets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = IssueVisualizer(self.mock_df, output_dir=tmpdir)
            target = visualizer.plot_comment_buckets()
            self.assertTrue(target.exists())
            self.assertGreater(target.stat().st_size, 1000)
            self.assertEqual(target.name, "fig6_comment_buckets.png")

    def test_plot_resolution_dynamics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = IssueVisualizer(self.mock_df, output_dir=tmpdir)
            target = visualizer.plot_resolution_dynamics()
            self.assertTrue(target.exists())
            self.assertGreater(target.stat().st_size, 1000)
            self.assertEqual(target.name, "fig7_resolution_dynamics.png")

    def test_plot_correlation_heatmap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = IssueVisualizer(self.mock_df, output_dir=tmpdir)
            target = visualizer.plot_correlation_heatmap()
            self.assertTrue(target.exists())
            self.assertGreater(target.stat().st_size, 1000)
            self.assertEqual(target.name, "fig8_correlation_heatmap.png")

    def test_plot_master_dashboard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = IssueVisualizer(self.mock_df, output_dir=tmpdir)
            target = visualizer.plot_master_dashboard()
            self.assertTrue(target.exists())
            self.assertGreater(target.stat().st_size, 1000)
            self.assertEqual(target.name, "fig9_master_eda_dashboard.png")

    def test_generate_all_visualizations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = IssueVisualizer(self.mock_df, output_dir=tmpdir)
            figures = visualizer.generate_all_visualizations()
            self.assertEqual(len(figures), 9)
            for fig_key, fig_path_str in figures.items():
                p = Path(fig_path_str)
                self.assertTrue(p.exists())
                self.assertGreater(p.stat().st_size, 1000)

    def test_edge_case_minimal_columns(self):
        minimal_df = pd.DataFrame({"title": ["Crash"], "comments_count": [1]})
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = IssueVisualizer(minimal_df, output_dir=tmpdir)
            target = visualizer.plot_label_distribution()
            self.assertTrue(target.exists())

    def test_live_master_dataset_batch_generation(self):
        csv_path = Path("data/processed/issues_master.csv")
        if not csv_path.exists():
            self.skipTest("Master dataset CSV not found.")

        out_figures_dir = Path("ml/eda/figures")
        visualizer = IssueVisualizer(csv_path, output_dir=out_figures_dir)
        figures = visualizer.generate_all_visualizations()

        self.assertEqual(len(figures), 9)
        for name, path_str in figures.items():
            p = Path(path_str)
            self.assertTrue(p.exists(), f"Figure {name} was not created at {p}")
            self.assertGreater(p.stat().st_size, 5000, f"Figure {name} size unexpectedly small")


if __name__ == "__main__":
    unittest.main()
