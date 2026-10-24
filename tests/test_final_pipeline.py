"""Unit Tests for Final Classical NLP Pipeline (Step 5.5)."""

from pathlib import Path
import tempfile
import unittest
import json

import pandas as pd
import numpy as np

from ml.nlp.final_pipeline import FinalClassicalNLPPipeline, FinalPipelineEvaluator


class TestFinalClassicalNLPPipeline(unittest.TestCase):
    """Test suite for FinalClassicalNLPPipeline and FinalPipelineEvaluator."""

    def setUp(self):
        """Set up synthetic datasets for testing."""
        self.mock_train_data = pd.DataFrame(
            {
                "title": [
                    "Fix null pointer in auth.py",
                    "Add dark mode support to editor",
                    "Bump dependencies in requirements.txt",
                    "Crash when rendering complex table",
                    "Implement OAuth2 login flow",
                    "Update README installation instructions",
                    "Segmentation fault on large array input",
                    "Add export to CSV button",
                    "Refactor utility helpers chore",
                    "Memory leak in image processing worker",
                    "Support multi-language localized strings",
                    "Format code using black and ruff",
                    "NullPointerException in issue_ref #456",
                    "Feature request: add copy button",
                    "Chore: clean up deprecated flags",
                ],
                "body": [
                    "NullPointerException occurs at line 42 when user is anonymous.",
                    "Users have requested a high contrast dark mode theme option.",
                    "Automated chore bumping packages in requirements.txt.",
                    "Application terminates with exit code 139 when rendering table.",
                    "Users should be able to authenticate using Google or GitHub.",
                    "Documentation is outdated for Windows Powershell setups.",
                    "SIGSEGV occurs in C++ backend when tensor size exceeds 1GB.",
                    "Allow exporting analytics reports directly to comma-separated CSV.",
                    "Routine code maintenance and cleanup of dead helper methods.",
                    "Resident set size grows continuously over 24 hours of operation.",
                    "Add translations for Spanish, French, and Japanese interfaces.",
                    "Run automated code formatter on commit hooks.",
                    "Crash when handling issue reference #456 in file_path utils.py.",
                    "Allow users to easily copy text directly from the dashboard.",
                    "Remove obsolete configuration flags from settings.json.",
                ],
                "canonical_label": [
                    "bug",
                    "feature",
                    "other",
                    "bug",
                    "feature",
                    "other",
                    "bug",
                    "feature",
                    "other",
                    "bug",
                    "feature",
                    "other",
                    "bug",
                    "feature",
                    "other",
                ],
            }
        )

        self.mock_test_data = pd.DataFrame(
            {
                "title": [
                    "IndexError out of bounds in parser",
                    "Support webhook notifications on build failure",
                    "Clean up deprecated configuration flags",
                ],
                "body": [
                    "Parser crashes when token list is empty.",
                    "Send HTTP POST payload to configured Slack endpoint.",
                    "Remove legacy settings from config.json schema.",
                ],
                "canonical_label": ["bug", "feature", "other"],
            }
        )

    def test_initialization(self):
        """Verify pipeline initialization defaults and state."""
        pipe = FinalClassicalNLPPipeline(max_features=1000, random_state=42)
        self.assertEqual(pipe.max_features, 1000)
        self.assertEqual(pipe.random_state, 42)
        self.assertEqual(pipe.title_multiplier, 3)
        self.assertFalse(pipe.is_fitted)
        self.assertIsNone(pipe.calibrated_model)

    def test_unfitted_raises_runtime_error(self):
        """Verify calling predict on unfitted pipeline raises RuntimeError."""
        pipe = FinalClassicalNLPPipeline()
        with self.assertRaises(RuntimeError):
            pipe.predict(["Fix bug"])
        with self.assertRaises(RuntimeError):
            pipe.predict_proba(["Fix bug"])
        with self.assertRaises(RuntimeError):
            pipe.explain_prediction("Fix bug", "body")

    def test_fit_and_predict_flow(self):
        """Verify end-to-end fitting and diverse input format prediction."""
        pipe = FinalClassicalNLPPipeline(max_features=200, random_state=42)
        pipe.fit(self.mock_train_data)

        self.assertTrue(pipe.is_fitted)
        self.assertEqual(set(pipe.classes_), {"bug", "feature", "other"})
        self.assertGreater(len(pipe.feature_names_), 0)

        # 1. Predict on DataFrame
        df_preds = pipe.predict(self.mock_test_data)
        self.assertEqual(len(df_preds), len(self.mock_test_data))
        for pred in df_preds:
            self.assertIn(pred, pipe.classes_)

        # 2. Predict probabilities on list of tuples
        tuples_input = [
            ("NullPointerException on login", "Crash occurs at line 5"),
            ("Add dark theme", "New visual theme requested"),
        ]
        probas = pipe.predict_proba(tuples_input)
        self.assertEqual(probas.shape, (2, 3))
        # Verify probabilities sum to 1.0
        np.testing.assert_allclose(probas.sum(axis=1), np.ones(2), rtol=1e-5)

        # 3. Predict on list of dicts
        dict_input = [
            {"title": "Bug in API", "body": "Endpoint returns 500"},
        ]
        dict_preds = pipe.predict(dict_input)
        self.assertEqual(len(dict_preds), 1)

    def test_explain_prediction(self):
        """Verify explain_prediction provides top contributing tokens and weights."""
        pipe = FinalClassicalNLPPipeline(max_features=200, random_state=42)
        pipe.fit(self.mock_train_data)

        explanation = pipe.explain_prediction(
            title="NullPointerException in authentication module",
            body="Unexpected fatal crash with exit code 139 when token is empty.",
            top_k=3,
        )

        self.assertIn("predicted_class", explanation)
        self.assertIn("confidence", explanation)
        self.assertIn("class_probabilities", explanation)
        self.assertIn("top_contributing_features", explanation)
        self.assertLessEqual(len(explanation["top_contributing_features"]), 3)

        if explanation["top_contributing_features"]:
            feat_item = explanation["top_contributing_features"][0]
            self.assertIn("feature", feat_item)
            self.assertIn("contribution_score", feat_item)

    def test_save_and_load_pipeline(self):
        """Verify pipeline serialization and reloading produces identical inference."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "pipeline.joblib"
            manifest_path = Path(tmp_dir) / "manifest.json"

            pipe = FinalClassicalNLPPipeline(max_features=200, random_state=42)
            pipe.fit(self.mock_train_data)

            pipe.save(model_path, manifest_path)
            self.assertTrue(model_path.exists())
            self.assertTrue(manifest_path.exists())

            # Load pipeline
            loaded_pipe = FinalClassicalNLPPipeline.load(model_path)
            self.assertTrue(loaded_pipe.is_fitted)
            self.assertEqual(loaded_pipe.classes_, pipe.classes_)

            # Compare predictions
            orig_preds = pipe.predict(self.mock_test_data)
            loaded_preds = loaded_pipe.predict(self.mock_test_data)
            np.testing.assert_array_equal(orig_preds, loaded_preds)

            orig_probas = pipe.predict_proba(self.mock_test_data)
            loaded_probas = loaded_pipe.predict_proba(self.mock_test_data)
            np.testing.assert_allclose(orig_probas, loaded_probas, rtol=1e-5)

    def test_evaluator_and_head_to_head(self):
        """Verify FinalPipelineEvaluator generates metrics, deltas, and plot."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fig_path = Path(tmp_dir) / "fig16.png"
            json_path = Path(tmp_dir) / "eval.json"

            pipe = FinalClassicalNLPPipeline(max_features=200, random_state=42)
            pipe.fit(self.mock_train_data)

            evaluator = FinalPipelineEvaluator(pipe)
            test_metrics = evaluator.evaluate_on_test(self.mock_test_data)

            self.assertIn("accuracy", test_metrics)
            self.assertIn("macro_f1", test_metrics)
            self.assertIn("per_class", test_metrics)

            head_to_head = evaluator.compute_head_to_head(test_metrics)
            self.assertIn("phase4_baseline", head_to_head)
            self.assertIn("phase5_pipeline", head_to_head)
            self.assertIn("deltas", head_to_head)

            # Test plot
            saved_fig = evaluator.plot_comparison(head_to_head, fig_path)
            self.assertTrue(saved_fig.exists())
            self.assertGreater(saved_fig.stat().st_size, 1000)

            # Test save results
            saved_json = evaluator.save_results(test_metrics, head_to_head, json_path)
            self.assertTrue(saved_json.exists())


if __name__ == "__main__":
    unittest.main()
