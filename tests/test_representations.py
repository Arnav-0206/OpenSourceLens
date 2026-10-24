"""Unit Tests for Text Representation Experiments (Step 5.3)."""

from pathlib import Path
import tempfile
import unittest
import json

import pandas as pd
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer

from ml.nlp.representations import TextRepresentationExperimentRunner


class TestTextRepresentationExperiments(unittest.TestCase):
    """Test suite for TextRepresentationExperimentRunner."""

    def setUp(self):
        """Set up synthetic datasets for testing."""
        self.mock_train_data = pd.DataFrame(
            {
                "title": [
                    "Fix null pointer in auth module",
                    "Add dark mode support to editor",
                    "Bump dependencies to latest version",
                    "Crash when rendering complex table",
                    "Implement OAuth2 login flow",
                    "Update README installation instructions",
                    "Segmentation fault on large array input",
                    "Add export to CSV button",
                    "Refactor utility helpers into separate package",
                    "Memory leak in image processing worker",
                    "Support multi-language localized strings",
                    "Format code using black and ruff",
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
        """Verify runner initialization defaults and properties."""
        runner = TextRepresentationExperimentRunner(max_features=2000, random_state=123)
        self.assertEqual(runner.max_features, 2000)
        self.assertEqual(runner.random_state, 123)
        self.assertEqual(runner.title_multiplier, 3)
        self.assertTrue(runner.use_enhanced_preprocessing)
        self.assertIsNotNone(runner.preprocessor)
        self.assertEqual(len(runner.STRATEGIES), 8)

    def test_build_vectorizer_all_strategies(self):
        """Verify build_vectorizer constructs valid pipelines for all strategies."""
        runner = TextRepresentationExperimentRunner(max_features=3000)

        for strategy in runner.STRATEGIES:
            vec = runner.build_vectorizer(strategy, corpus_size=100)
            if strategy == "hybrid_word_char":
                self.assertIsInstance(vec, FeatureUnion)
                self.assertEqual(len(vec.transformer_list), 2)
            else:
                self.assertIsInstance(vec, TfidfVectorizer)

    def test_build_vectorizer_unknown_raises(self):
        """Verify unknown strategy name raises ValueError."""
        runner = TextRepresentationExperimentRunner()
        with self.assertRaises(ValueError):
            runner.build_vectorizer("unknown_magic_representation")

    def test_hybrid_feature_budget(self):
        """Verify hybrid FeatureUnion maintains the total max_features budget."""
        runner = TextRepresentationExperimentRunner(max_features=5000)
        hybrid_vec = runner.build_vectorizer("hybrid_word_char")
        self.assertIsInstance(hybrid_vec, FeatureUnion)

        word_vec = hybrid_vec.transformer_list[0][1]
        char_vec = hybrid_vec.transformer_list[1][1]

        self.assertEqual(word_vec.max_features, 3500)
        self.assertEqual(char_vec.max_features, 1500)
        self.assertEqual(word_vec.max_features + char_vec.max_features, 5000)

    def test_prepare_text(self):
        """Verify prepare_text handles title repetition and cleaning."""
        runner = TextRepresentationExperimentRunner(title_multiplier=2, use_enhanced_preprocessing=True)
        composed = runner.prepare_text(
            title="Fix bug #123 in auth.py",
            body="Unexpected error at line 10.",
        )
        # Check issue ref replacement and repetition
        self.assertIn("issue_ref", composed)
        self.assertIn("file_path", composed)

    def test_prepare_dataset(self):
        """Verify prepare_dataset produces matched texts and labels."""
        runner = TextRepresentationExperimentRunner()
        texts, labels = runner.prepare_dataset(self.mock_train_data)
        self.assertEqual(len(texts), len(self.mock_train_data))
        self.assertEqual(len(labels), len(self.mock_train_data))
        self.assertEqual(set(runner.classes_), {"bug", "feature", "other"})

    def test_cross_validation_small_dataset(self):
        """Verify 3-fold cross-validation executes on mock data for all strategies."""
        runner = TextRepresentationExperimentRunner(max_features=500, random_state=42)
        cv_results = runner.run_cross_validation(self.mock_train_data, n_splits=3)

        self.assertEqual(len(cv_results), 8)
        for strategy, metrics in cv_results.items():
            self.assertIn("macro_f1_mean", metrics)
            self.assertIn("macro_f1_std", metrics)
            self.assertIn("accuracy_mean", metrics)
            self.assertIn("per_class_f1_mean", metrics)
            self.assertGreaterEqual(metrics["macro_f1_mean"], 0.0)
            self.assertLessEqual(metrics["macro_f1_mean"], 1.0)
            for cls in ["bug", "feature", "other"]:
                self.assertIn(cls, metrics["per_class_f1_mean"])

    def test_evaluate_on_test_small_dataset(self):
        """Verify test evaluation fits and scores correctly on held-out test data."""
        runner = TextRepresentationExperimentRunner(max_features=500, random_state=42)
        test_results = runner.evaluate_on_test(self.mock_train_data, self.mock_test_data)

        self.assertEqual(len(test_results), 8)
        for strategy, metrics in test_results.items():
            self.assertIn("test_accuracy", metrics)
            self.assertIn("test_macro_f1", metrics)
            self.assertIn("test_per_class_f1", metrics)
            self.assertIn("feature_dimension", metrics)
            self.assertGreaterEqual(metrics["test_macro_f1"], 0.0)
            self.assertLessEqual(metrics["test_macro_f1"], 1.0)

    def test_plot_and_save_results(self):
        """Verify plot generation and JSON persistence work end-to-end."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fig_path = Path(tmp_dir) / "test_fig.png"
            json_path = Path(tmp_dir) / "test_results.json"

            runner = TextRepresentationExperimentRunner(max_features=500, random_state=42)
            runner.run_cross_validation(self.mock_train_data, n_splits=3)
            runner.evaluate_on_test(self.mock_train_data, self.mock_test_data)

            # Test plot creation
            saved_fig = runner.plot_comparison(fig_path)
            self.assertTrue(saved_fig.exists())
            self.assertGreater(saved_fig.stat().st_size, 1000)

            # Test JSON output
            saved_json = runner.save_results(json_path)
            self.assertTrue(saved_json.exists())

            with open(saved_json, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data["metadata"]["step"], "5.3")
            self.assertIn("champion_cv_strategy", data["metadata"])
            self.assertIn("cross_validation_results", data)
            self.assertIn("test_evaluation_results", data)


if __name__ == "__main__":
    unittest.main()
