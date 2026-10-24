"""Unit tests for ml.nlp.title_body_experiment (TitleBodyExperimentRunner)."""

import unittest
import tempfile
import json
from pathlib import Path

import pandas as pd
import numpy as np

from ml.nlp.title_body_experiment import TitleBodyExperimentRunner


class TestTitleBodyExperimentRunner(unittest.TestCase):
    """Test suite for TitleBodyExperimentRunner."""

    def setUp(self):
        """Construct synthetic test DataFrame."""
        self.runner = TitleBodyExperimentRunner(max_features=200, random_state=42)

        # 9 samples: 3 bug, 3 feature, 3 other
        self.mock_train = pd.DataFrame({
            "clean_title": [
                "Crash on startup error",
                "Segmentation fault fatal bug",
                "NullPointer crash boot",
                "Add dark mode theme support",
                "New feature export button",
                "Enhance user interface support",
                "Update documentation readme",
                "Refactor internal code logic",
                "License and governance update",
            ],
            "clean_body": [
                "Application terminates with error code 1",
                "Memory violation in loop reproduction steps",
                "Boot loader crashes unexpectedly",
                "Please add a dark mode toggle to preferences",
                "Support markdown export option in file menu",
                "Improve button alignment and colors",
                "Fix typos in installation guide",
                "Clean up unused helper functions",
                "Clarify contributor guidelines",
            ],
            "target_label": ["bug", "bug", "bug", "feature", "feature", "feature", "other", "other", "other"],
        })

        self.mock_test = pd.DataFrame({
            "clean_title": ["Another crash error", "Feature request button", "Typo in readme"],
            "clean_body": ["Reproduced crash on startup", "Please add button", "Documentation fix"],
            "target_label": ["bug", "feature", "other"],
        })

    def test_prepare_text_series_single(self):
        series = self.runner.prepare_text_series(self.mock_train, "title_only")
        self.assertEqual(len(series), 9)
        self.assertIn("crash on startup error", series.iloc[0])

        body_series = self.runner.prepare_text_series(self.mock_train, "body_only")
        self.assertEqual(len(body_series), 9)
        self.assertIn("application terminates", body_series.iloc[0])

        combo_series = self.runner.prepare_text_series(self.mock_train, "title_body_1x")
        self.assertEqual(len(combo_series), 9)
        self.assertIn("crash on startup error", combo_series.iloc[0])
        self.assertIn("application terminates", combo_series.iloc[0])

    def test_prepare_text_series_dual(self):
        titles, bodies = self.runner.prepare_text_series(self.mock_train, "dual_channel_union")
        self.assertEqual(len(titles), 9)
        self.assertEqual(len(bodies), 9)

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            self.runner.prepare_text_series(self.mock_train, "invalid_strategy")

    def test_fit_transform_strategy(self):
        X_tr, X_te = self.runner._fit_transform_strategy(self.mock_train, "title_body_1x", self.mock_test)
        self.assertEqual(X_tr.shape[0], 9)
        self.assertEqual(X_te.shape[0], 3)

        X_tr_dual, X_te_dual = self.runner._fit_transform_strategy(self.mock_train, "dual_channel_union", self.mock_test)
        self.assertEqual(X_tr_dual.shape[0], 9)
        self.assertEqual(X_te_dual.shape[0], 3)

    def test_run_cross_validation_experiments(self):
        cv_results = self.runner.run_cross_validation_experiments(self.mock_train, cv=3)

        self.assertEqual(cv_results["cv_folds"], 3)
        self.assertEqual(len(cv_results["ranked_strategies"]), 6)
        self.assertIn("best_strategy", cv_results)

        for res in cv_results["ranked_strategies"]:
            self.assertIn("strategy", res)
            self.assertIn("mean_macro_f1", res)
            self.assertIn("mean_accuracy", res)

    def test_evaluate_best_on_test(self):
        test_eval = self.runner.evaluate_best_on_test(
            self.mock_train, self.mock_test, best_strategy="title_body_1x"
        )

        self.assertEqual(test_eval["test_sample_count"], 3)
        self.assertIn("accuracy", test_eval)
        self.assertIn("macro_f1", test_eval)
        self.assertIn("per_class", test_eval)

    def test_plot_comparison(self):
        cv_results = self.runner.run_cross_validation_experiments(self.mock_train, cv=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_img = Path(tmpdir) / "test_fig.png"
            saved_file = self.runner.plot_comparison(cv_results, output_path=out_img)
            self.assertTrue(saved_file.exists())
            self.assertGreater(saved_file.stat().st_size, 1000)

    def test_save_results(self):
        cv_results = self.runner.run_cross_validation_experiments(self.mock_train, cv=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = Path(tmpdir) / "test_results.json"
            saved_file = self.runner.save_results(cv_results, output_path=out_json)
            self.assertTrue(saved_file.exists())

            with open(saved_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["cv_folds"], 3)


if __name__ == "__main__":
    unittest.main()
