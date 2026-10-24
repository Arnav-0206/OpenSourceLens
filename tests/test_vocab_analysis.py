"""Unit Tests for Vocabulary and Feature Salience Analysis (Step 5.4)."""

from pathlib import Path
import tempfile
import unittest
import json

import pandas as pd
import numpy as np

from ml.nlp.vocab_analysis import VocabularyFeatureAnalyzer


class TestVocabularyFeatureAnalyzer(unittest.TestCase):
    """Test suite for VocabularyFeatureAnalyzer."""

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
        """Verify initialization defaults and properties."""
        analyzer = VocabularyFeatureAnalyzer(max_features=1000, random_state=42)
        self.assertEqual(analyzer.max_features, 1000)
        self.assertEqual(analyzer.random_state, 42)
        self.assertEqual(analyzer.title_multiplier, 3)
        self.assertTrue(analyzer.use_enhanced_preprocessing)
        self.assertIsNotNone(analyzer.preprocessor)

    def test_prepare_dataset_column_resolution(self):
        """Verify column resolution works with canonical_label and clean_title."""
        analyzer = VocabularyFeatureAnalyzer()
        texts, labels = analyzer.prepare_dataset(self.mock_train_data)
        self.assertEqual(len(texts), 15)
        self.assertEqual(len(labels), 15)
        self.assertEqual(set(analyzer.classes_), {"bug", "feature", "other"})

    def test_analyze_features_and_salience(self):
        """Verify feature extraction, class coefficients, chi2, and domain tokens."""
        analyzer = VocabularyFeatureAnalyzer(max_features=200, random_state=42)
        results = analyzer.analyze_features_and_salience(
            self.mock_train_data,
            top_k_positive=5,
            top_k_negative=3,
        )

        # Check class coefficients
        self.assertIn("coefficients_by_class", results)
        for cls in ["bug", "feature", "other"]:
            self.assertIn(cls, results["coefficients_by_class"])
            pos = results["coefficients_by_class"][cls]["positive"]
            neg = results["coefficients_by_class"][cls]["negative"]
            self.assertGreater(len(pos), 0)
            self.assertGreater(len(neg), 0)
            # Verify positive weights >= negative weights
            self.assertGreaterEqual(pos[0]["weight"], neg[0]["weight"])

        # Check Chi2 rankings
        self.assertIn("top_chi2_features", results)
        self.assertGreater(len(results["top_chi2_features"]), 0)
        top_chi2 = results["top_chi2_features"][0]
        self.assertIn("feature", top_chi2)
        self.assertIn("chi2_score", top_chi2)
        self.assertIn("p_value", top_chi2)

        # Check domain token audit
        self.assertIn("domain_token_audit", results)
        self.assertIsInstance(results["domain_token_audit"], list)

    def test_evaluate_vocab_size_sensitivity(self):
        """Verify sensitivity curve evaluation across vocabulary capacities."""
        analyzer = VocabularyFeatureAnalyzer(random_state=42)
        curve_res = analyzer.evaluate_vocab_size_sensitivity(
            train_df=self.mock_train_data,
            test_df=self.mock_test_data,
            vocab_sizes=[50, 100],
            n_splits=3,
        )

        self.assertIn("curve_data", curve_res)
        self.assertEqual(len(curve_res["curve_data"]), 2)
        for pt in curve_res["curve_data"]:
            self.assertIn("vocab_size", pt)
            self.assertIn("cv_macro_f1_mean", pt)
            self.assertIn("cv_macro_f1_std", pt)
            self.assertIn("test_macro_f1", pt)
            self.assertGreaterEqual(pt["cv_macro_f1_mean"], 0.0)

    def test_plot_and_save_workflow(self):
        """Verify plot generation and JSON persistence work end-to-end."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fig14_path = Path(tmp_dir) / "fig14.png"
            fig15_path = Path(tmp_dir) / "fig15.png"
            json_path = Path(tmp_dir) / "results.json"

            analyzer = VocabularyFeatureAnalyzer(max_features=200, random_state=42)
            analyzer.analyze_features_and_salience(self.mock_train_data, top_k_positive=5, top_k_negative=3)
            analyzer.evaluate_vocab_size_sensitivity(self.mock_train_data, self.mock_test_data, vocab_sizes=[50, 100], n_splits=3)

            # Test plot 14
            saved_fig14 = analyzer.plot_top_features_per_class(fig14_path)
            self.assertTrue(saved_fig14.exists())
            self.assertGreater(saved_fig14.stat().st_size, 1000)

            # Test plot 15
            saved_fig15 = analyzer.plot_vocab_size_sensitivity(fig15_path)
            self.assertTrue(saved_fig15.exists())
            self.assertGreater(saved_fig15.stat().st_size, 1000)

            # Test JSON persistence
            saved_json = analyzer.save_results(json_path)
            self.assertTrue(saved_json.exists())

            with open(saved_json, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data["metadata"]["step"], "5.4")
            self.assertIn("coefficients_by_class", data)
            self.assertIn("chi2_top_discriminative_features", data)
            self.assertIn("domain_token_audit", data)
            self.assertIn("vocab_size_sensitivity_curve", data)

    def test_uninitialized_plots_raise(self):
        """Verify plotting without running analysis raises ValueError."""
        analyzer = VocabularyFeatureAnalyzer()
        with self.assertRaises(ValueError):
            analyzer.plot_top_features_per_class("dummy.png")
        with self.assertRaises(ValueError):
            analyzer.plot_vocab_size_sensitivity("dummy.png")


if __name__ == "__main__":
    unittest.main()
