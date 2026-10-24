"""Unit Tests for Multi-Paradigm Model Comparison Pipeline (Step 6.6)."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from ml.deep_learning.comparator import ModelComparator


class TestModelComparison(unittest.TestCase):
    """Test suite for ModelComparator, agreement metrics, and comparative visualizations."""

    @classmethod
    def setUpClass(cls):
        """Set up comparator with project files."""
        cls.comparator = ModelComparator(
            baseline_path="data/processed/evaluation/baseline_comparison.json",
            classical_path="data/processed/evaluation/final_classical_nlp_evaluation.json",
            transformer_path="data/processed/evaluation/transformer_test_evaluation.json",
            test_splits_path="data/processed/splits/test.csv",
            test_predictions_path="data/processed/transformer/test_predictions.csv",
            classical_model_path="models/final_classical_nlp_pipeline.joblib",
        )
        cls.comparator.load_metrics()

    def test_metrics_loading(self):
        """Verify baseline, classical, and transformer JSON benchmark loading."""
        self.assertIsInstance(self.comparator.baseline_data, dict)
        self.assertIsInstance(self.comparator.classical_data, dict)
        self.assertIsInstance(self.comparator.transformer_data, dict)
        self.assertIn("overall_accuracy", self.comparator.transformer_data)
        self.assertIn("macro_f1", self.comparator.transformer_data)

    def test_agreement_analysis(self):
        """Verify Cohen's Kappa and contingency matrix integrity."""
        agreement = self.comparator.compute_agreement_analysis()
        self.assertIn("raw_agreement_rate", agreement)
        self.assertIn("cohen_kappa", agreement)
        self.assertIn("contingency", agreement)
        self.assertIn("ensemble_ceiling_pct", agreement)

        # Range checks
        self.assertGreaterEqual(agreement["cohen_kappa"], 0.5)  # Substantial agreement
        self.assertLessEqual(agreement["cohen_kappa"], 1.0)
        self.assertGreaterEqual(agreement["raw_agreement_rate"], 0.75)

        # Contingency sums match sample count
        contingency = agreement["contingency"]
        total_contingency = (
            contingency["both_correct"]
            + contingency["both_wrong"]
            + contingency["classical_only_correct"]
            + contingency["transformer_only_correct"]
        )
        self.assertEqual(total_contingency, agreement["sample_count"])

        # Ensemble ceiling should be strictly >= either individual accuracy
        self.assertGreaterEqual(agreement["ensemble_ceiling_pct"], 81.56)

    def test_build_comparison_table(self):
        """Verify unified comparison DataFrame structure and contents."""
        df = self.comparator.build_comparison_table()
        self.assertEqual(len(df), 3)  # Phase 4, Phase 5, Phase 6
        expected_cols = [
            "phase",
            "model_name",
            "paradigm",
            "accuracy",
            "macro_f1",
            "latency_ms_per_sample",
            "model_size_mb",
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)

        # Verify phase ordering
        self.assertEqual(df.iloc[0]["phase"], "Phase 4")
        self.assertEqual(df.iloc[1]["phase"], "Phase 5")
        self.assertEqual(df.iloc[2]["phase"], "Phase 6")

    def test_visualization_and_json_export(self):
        """Verify 3-panel figure generation and JSON serialization."""
        self.comparator.compute_agreement_analysis()
        self.comparator.build_comparison_table()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fig_path = tmp_path / "test_fig22.png"
            json_path = tmp_path / "test_benchmark.json"

            # Test plot export
            self.comparator.plot_comparison_figure(save_path=fig_path)
            self.assertTrue(fig_path.exists())
            self.assertGreater(fig_path.stat().st_size, 1000)

            # Test JSON export
            self.comparator.save_benchmark_json(save_path=json_path)
            self.assertTrue(json_path.exists())
            with open(json_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertIn("models", loaded)
            self.assertIn("trade_off_synthesis", loaded)
            self.assertEqual(len(loaded["models"]), 3)


if __name__ == "__main__":
    unittest.main()
