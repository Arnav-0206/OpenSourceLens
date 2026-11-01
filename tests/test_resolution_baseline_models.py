"""
OpenSourceLens — Unit Tests for Resolution Baseline Prediction Modeling (Phase 8, Step 8.5).

Validates data loading and stratified splitting, regressor and classifier benchmarks,
champion model selection and joblib persistence, figure generation, and manifest export.
"""

import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np

from ml.resolution_analysis.baseline_models import (
    SPEED_TIER_ORDER,
    ResolutionBaselineTrainer,
)


class TestResolutionBaselineTrainer(unittest.TestCase):
    """Test suite for ResolutionBaselineTrainer."""

    @classmethod
    def setUpClass(cls):
        """Initializes the trainer, loads data, and trains benchmarks once."""
        cls.trainer = ResolutionBaselineTrainer()
        cls.trainer.load_and_split_data(test_size=0.2, random_state=42)
        cls.trainer.train_regressors()
        cls.trainer.train_classifiers()

    def test_data_loading_and_split_shapes(self):
        """Verifies dataset merge and 80/20 stratified split shapes."""
        self.assertEqual(len(self.trainer.X_train), 908)
        self.assertEqual(len(self.trainer.X_test), 227)
        self.assertEqual(len(self.trainer.y_reg_train), 908)
        self.assertEqual(len(self.trainer.y_reg_test), 227)
        self.assertEqual(len(self.trainer.y_clf_train), 908)
        self.assertEqual(len(self.trainer.y_clf_test), 227)
        self.assertGreater(len(self.trainer.feature_names), 15)

    def test_regressors_training_and_metrics(self):
        """Validates that candidate regressors train and compute valid metrics."""
        benchmarks = self.trainer.train_regressors()
        self.assertIn("dummy_mean", benchmarks)
        self.assertIn("ridge", benchmarks)
        self.assertIn("random_forest", benchmarks)
        self.assertIn("gradient_boosting", benchmarks)

        for name, metrics in benchmarks.items():
            self.assertIn("test_r2", metrics)
            self.assertIn("test_mae_log_hours", metrics)
            self.assertIn("test_rmse_log_hours", metrics)
            self.assertFalse(np.isnan(metrics["test_r2"]))
            self.assertFalse(np.isnan(metrics["test_mae_log_hours"]))
            self.assertGreater(metrics["test_mae_log_hours"], 0.0)

    def test_classifiers_training_and_metrics(self):
        """Validates that candidate classifiers train and compute valid metrics."""
        benchmarks = self.trainer.train_classifiers()
        self.assertIn("dummy_prior", benchmarks)
        self.assertIn("logistic_regression", benchmarks)
        self.assertIn("random_forest", benchmarks)
        self.assertIn("gradient_boosting", benchmarks)

        for name, metrics in benchmarks.items():
            self.assertIn("test_accuracy", metrics)
            self.assertIn("test_macro_f1", metrics)
            self.assertGreaterEqual(metrics["test_accuracy"], 0.0)
            self.assertLessEqual(metrics["test_accuracy"], 1.0)
            self.assertGreaterEqual(metrics["test_macro_f1"], 0.0)
            self.assertLessEqual(metrics["test_macro_f1"], 1.0)
            self.assertEqual(len(metrics["confusion_matrix"]), len(SPEED_TIER_ORDER))

    def test_baseline_superiority(self):
        """Ensures champion models strictly outperform naive dummy baselines."""
        reg_bench = self.trainer.regression_benchmarks
        clf_bench = self.trainer.classification_benchmarks

        champ_reg = self.trainer.champion_regressor_name
        dummy_r2 = reg_bench["dummy_mean"]["test_r2"]
        champ_r2 = reg_bench[champ_reg]["test_r2"]
        self.assertGreater(champ_r2, dummy_r2)
        self.assertGreater(champ_r2, 0.20)  # Champion regressor achieves > 0.20 R²

        champ_clf = self.trainer.champion_classifier_name
        dummy_f1 = clf_bench["dummy_prior"]["test_macro_f1"]
        champ_f1 = clf_bench[champ_clf]["test_macro_f1"]
        self.assertGreater(champ_f1, dummy_f1)
        self.assertGreater(champ_f1, 0.35)  # Champion classifier achieves > 0.35 Macro F1

    def test_champion_model_serialization(self):
        """Validates serialization and re-loading of champion models."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = ResolutionBaselineTrainer(models_dir=tmp_dir)
            trainer.load_and_split_data()
            trainer.train_regressors()
            trainer.train_classifiers()
            saved = trainer.save_models()

            self.assertTrue(Path(saved["regressor"]).exists())
            self.assertTrue(Path(saved["classifier"]).exists())

            # Load and verify inference
            reg_obj = joblib.load(saved["regressor"])
            self.assertIn("model", reg_obj)
            self.assertIn("feature_names", reg_obj)
            sample_reg_pred = reg_obj["model"].predict(trainer.X_test[:2])
            self.assertEqual(len(sample_reg_pred), 2)

            clf_obj = joblib.load(saved["classifier"])
            self.assertIn("model", clf_obj)
            self.assertIn("classes", clf_obj)
            sample_clf_pred = clf_obj["model"].predict(trainer.X_test[:2])
            self.assertEqual(len(sample_clf_pred), 2)

    def test_manifest_export(self):
        """Verifies JSON benchmark manifest export and schema."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_manifest.json"
            exported = self.trainer.export_benchmark_manifest(out_file)
            self.assertTrue(exported.exists())

            with open(exported, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("metadata", data)
            self.assertIn("regression_benchmarks", data)
            self.assertIn("classification_benchmarks", data)
            self.assertEqual(data["metadata"]["n_samples_train"], 908)
            self.assertEqual(data["metadata"]["n_samples_test"], 227)

    def test_plot_generation(self):
        """Verifies Figure 32 4-panel generation and size."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_plot = Path(tmp_dir) / "test_fig32.png"
            gen_plot = self.trainer.generate_figures(out_plot)
            self.assertTrue(gen_plot.exists())
            self.assertGreater(gen_plot.stat().st_size, 50000)


if __name__ == "__main__":
    unittest.main()
