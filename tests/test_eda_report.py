"""Unit tests for ml.eda.report (EDAFindingsReporter)."""

import json
import unittest
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np

from ml.eda.report import EDAFindingsReporter


class TestEDAFindingsReporter(unittest.TestCase):
    """Test suite for EDAFindingsReporter."""

    def setUp(self):
        """Construct synthetic test dataset with necessary columns for reporting."""
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
            "title": ["Crash bug", "Bug fix", "Add feature", "Other issue", "Feature idea", "Error in engine", "Other question", "Add test"],
            "body": ["Detailed description of crash", "Fix details", "Feature proposal", "Other notes", "More info", "Stack trace", "Notes", "Test info"],
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
            reporter = EDAFindingsReporter(self.mock_df, eda_dir=tmpdir)
            self.assertEqual(len(reporter.df), 8)
            self.assertEqual(reporter.source_path, "DataFrame in-memory")

    def test_init_from_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            self.mock_df.to_csv(csv_path, index=False)
            reporter = EDAFindingsReporter(csv_path, eda_dir=tmpdir)
            self.assertEqual(len(reporter.df), 8)
            self.assertEqual(reporter.source_path, str(csv_path))

    def test_init_from_parquet_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            parquet_path = Path(tmpdir) / "test_data.parquet"
            self.mock_df.to_csv(csv_path, index=False)
            parquet_path.write_bytes(b"MOCK_PARQUET_FALLBACK")
            reporter = EDAFindingsReporter(parquet_path, eda_dir=tmpdir)
            self.assertEqual(len(reporter.df), 8)

    def test_init_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            EDAFindingsReporter("non_existent_file.csv")

    def test_init_empty_dataframe(self):
        with self.assertRaises(ValueError):
            EDAFindingsReporter(pd.DataFrame())

    def test_generate_markdown_findings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = EDAFindingsReporter(self.mock_df, eda_dir=tmpdir)
            md = reporter.generate_markdown_findings()
            self.assertIsInstance(md, str)
            self.assertIn("# OpenSourceLens — Phase 3: Exploratory Data Analysis (EDA) Findings", md)
            self.assertIn("1. Dataset Health, Scope & Dimensionality", md)
            self.assertIn("2. Issue Lifecycle & Closure Decomposition", md)
            self.assertIn("3. Canonical Taxonomy & Class Imbalance Analysis", md)
            self.assertIn("11. Strategic Blueprint for Phase 4", md)
            self.assertIn("figures/fig1_label_distribution.png", md)

    def test_generate_jupyter_notebook(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = EDAFindingsReporter(self.mock_df, eda_dir=tmpdir)
            nb = reporter.generate_jupyter_notebook()
            self.assertIsInstance(nb, dict)
            self.assertEqual(nb["nbformat"], 4)
            self.assertIn("cells", nb)
            self.assertGreaterEqual(len(nb["cells"]), 15)

            # Check that cells have code and markdown types
            types = set(c["cell_type"] for c in nb["cells"])
            self.assertEqual(types, {"code", "markdown"})

    def test_export_all_custom_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nb_target = Path(tmpdir) / "test_notebook.ipynb"
            md_target = Path(tmpdir) / "test_findings.md"

            reporter = EDAFindingsReporter(self.mock_df, eda_dir=tmpdir)
            outputs = reporter.export_all(notebook_path=nb_target, markdown_path=md_target)

            self.assertTrue(outputs["notebook"].exists())
            self.assertTrue(outputs["findings_report"].exists())

            # Validate notebook is valid json
            nb_parsed = json.loads(outputs["notebook"].read_text(encoding="utf-8"))
            self.assertEqual(nb_parsed["nbformat"], 4)

            # Validate markdown content
            md_text = outputs["findings_report"].read_text(encoding="utf-8")
            self.assertIn("Phase 3: Exploratory Data Analysis (EDA) Findings", md_text)

    def test_live_master_dataset_export(self):
        csv_path = Path("data/processed/issues_master.csv")
        if not csv_path.exists():
            self.skipTest("Master dataset CSV not found.")

        eda_target_dir = Path("ml/eda")
        reporter = EDAFindingsReporter(csv_path, eda_dir=eda_target_dir)
        outputs = reporter.export_all()

        nb_path = outputs["notebook"]
        md_path = outputs["findings_report"]

        self.assertTrue(nb_path.exists())
        self.assertTrue(md_path.exists())
        self.assertGreater(nb_path.stat().st_size, 5000)
        self.assertGreater(md_path.stat().st_size, 5000)

        # Assert specific empirical facts in generated report
        md_text = md_path.read_text(encoding="utf-8")
        self.assertIn("2,328 issues", md_text)
        self.assertIn("microsoft/vscode", md_text)
        self.assertIn("kubernetes/kubernetes", md_text)
        self.assertIn("figures/fig9_master_eda_dashboard.png", md_text)


if __name__ == "__main__":
    unittest.main()
