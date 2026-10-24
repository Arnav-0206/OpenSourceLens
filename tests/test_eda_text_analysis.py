"""Unit tests for ml.eda.text_analysis (IssueTextAnalyzer)."""

import json
import unittest
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np

from ml.eda.text_analysis import IssueTextAnalyzer


class TestIssueTextAnalyzer(unittest.TestCase):
    """Test suite for IssueTextAnalyzer."""

    def setUp(self):
        """Construct synthetic test dataset with varied text lengths and keywords."""
        self.mock_df = pd.DataFrame({
            "issue_id": [501, 502, 503, 504, 505],
            "title": [
                "Crash on startup with error",
                "Feature request: add support for new architecture",
                "Failing test and flaky flaking execution",
                "Please add paper motivation contribution",
                "Minor typo in documentation",
            ],
            "clean_title": [
                "Crash on startup with error",
                "Feature request: add support for new architecture",
                "Failing test and flaky flaking execution",
                "Please add paper motivation contribution",
                "Minor typo in documentation",
            ],
            "body": [
                "Application failed with error stack trace following error error",
                "We propose new feature support for motivation and paper contribution",
                "The test is flaking on CI with failure error",
                "Motivation for this feature is high. We provide contribution paper",
                "Small typo fix in README",
            ],
            "clean_body": [
                "Application failed with error stack trace following error error",
                "We propose new feature support for motivation and paper contribution",
                "The test is flaking on CI with failure error",
                "Motivation for this feature is high. We provide contribution paper",
                "Small typo fix in README",
            ],
            "canonical_label": ["bug", "feature", "bug", "feature", "documentation"],
            "clean_title_word_count": [5, 7, 6, 5, 4],
            "clean_body_word_count": [10, 10, 8, 9, 5],
            "has_code_block": [True, False, True, False, False],
            "has_stack_trace": [True, False, False, False, False],
            "has_url": [False, True, False, True, False],
            "title_has_bracket_tag": [False, True, False, False, False],
            "avg_word_length_title": [5.2, 5.8, 6.1, 6.4, 5.0],
            "avg_word_length_body": [6.0, 6.2, 5.5, 6.1, 4.8],
        })

    def test_init_from_dataframe(self):
        """Test initialization with DataFrame."""
        analyzer = IssueTextAnalyzer(self.mock_df)
        self.assertEqual(len(analyzer.df), 5)
        self.assertIn("combined_text", analyzer.df.columns)
        self.assertIn("combined_word_count", analyzer.df.columns)

    def test_init_from_parquet_and_csv(self):
        """Test initialization from temporary Parquet and CSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p_path = Path(tmpdir) / "test.parquet"
            c_path = Path(tmpdir) / "test.csv"

            self.mock_df.to_csv(c_path, index=False)
            try:
                self.mock_df.to_parquet(p_path)
                analyzer_p = IssueTextAnalyzer(p_path)
                self.assertEqual(len(analyzer_p.df), 5)
            except Exception:
                p_path.write_bytes(b"mock_parquet")
                analyzer_p = IssueTextAnalyzer(p_path)
                self.assertEqual(len(analyzer_p.df), 5)

            analyzer_c = IssueTextAnalyzer(c_path)
            self.assertEqual(len(analyzer_c.df), 5)

    def test_init_errors(self):
        """Test file not found, empty DataFrame, and invalid type errors."""
        with self.assertRaises(FileNotFoundError):
            IssueTextAnalyzer("non_existent_file.parquet")

        with self.assertRaises(ValueError):
            IssueTextAnalyzer(pd.DataFrame())

        with self.assertRaises(TypeError):
            IssueTextAnalyzer(12345)  # type: ignore

    def test_tokenize(self):
        """Test word extraction, lowercasing, and length threshold."""
        tokens = IssueTextAnalyzer._tokenize("Error: Failed at line 42 with Exception!!")
        self.assertIn("error", tokens)
        self.assertIn("failed", tokens)
        self.assertIn("line", tokens)
        self.assertIn("exception", tokens)
        self.assertNotIn("at", tokens)  # length < 3

    def test_get_length_percentiles(self):
        """Test 11-point percentile calculation."""
        analyzer = IssueTextAnalyzer(self.mock_df)
        stats = analyzer.get_length_percentiles("clean_title_word_count")

        self.assertEqual(stats["count"], 5)
        self.assertEqual(stats["min"], 4.0)
        self.assertEqual(stats["max"], 7.0)
        self.assertEqual(stats["median"], 5.0)
        self.assertEqual(stats["mean"], 5.4)

    def test_get_transformer_budget_analysis(self):
        """Test retention and truncation rates across token budgets."""
        analyzer = IssueTextAnalyzer(self.mock_df)
        budgets_df = analyzer.get_transformer_budget_analysis([10, 20, 50])

        self.assertEqual(len(budgets_df), 3)
        b_50 = budgets_df[budgets_df["context_token_limit"] == 50].iloc[0]
        # All 5 issues have combined word count <= 17, so all fit in budget 50
        self.assertEqual(b_50["fully_retained_issues"], 5)
        self.assertEqual(b_50["retention_pct"], 100.0)

    def test_get_vocabulary_metrics(self):
        """Test total token count, unique vocabulary, and TTR."""
        analyzer = IssueTextAnalyzer(self.mock_df)
        vocab = analyzer.get_vocabulary_metrics()

        self.assertGreater(vocab["total_tokens_combined"], 0)
        self.assertGreater(vocab["unique_vocabulary_size"], 0)
        self.assertGreater(vocab["type_token_ratio_ttr"], 0.0)
        self.assertLessEqual(vocab["type_token_ratio_ttr"], 1.0)

    def test_get_top_ngrams(self):
        """Test extraction of top unigrams and bigrams."""
        analyzer = IssueTextAnalyzer(self.mock_df)
        unigrams_df = analyzer.get_top_ngrams(n=1, top_k=5)
        self.assertGreater(len(unigrams_df), 0)
        # 'error' appears multiple times
        self.assertIn("error", unigrams_df["ngram"].values)

        bigrams_df = analyzer.get_top_ngrams(n=2, top_k=5)
        self.assertGreater(len(bigrams_df), 0)

    def test_get_discriminative_keywords(self):
        """Test contrastive keyword extraction between bug and feature."""
        # Add repeated words so support threshold >= 15 is met in test
        repeated_df = pd.concat([self.mock_df] * 10, ignore_index=True)
        analyzer = IssueTextAnalyzer(repeated_df)
        discrim_df = analyzer.get_discriminative_keywords(top_k=10)

        self.assertGreater(len(discrim_df), 0)
        terms = discrim_df["term"].values
        self.assertTrue(any(t in terms for t in ["error", "motivation", "contribution", "feature"]))

    def test_get_structural_indicators(self):
        """Test structural indicator proportions."""
        analyzer = IssueTextAnalyzer(self.mock_df)
        struct_df = analyzer.get_structural_indicators()

        self.assertEqual(len(struct_df), 4)
        cb_row = struct_df[struct_df["indicator_feature"] == "has_code_block"].iloc[0]
        self.assertEqual(cb_row["true_count"], 2)
        self.assertEqual(cb_row["prevalence_pct"], 40.0)

    def test_compute_full_text_analysis_and_json(self):
        """Test compiling complete analysis and exporting JSON."""
        analyzer = IssueTextAnalyzer(self.mock_df)
        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = Path(tmpdir) / "text_analysis.json"
            analyzer.save_summary_json(json_file)

            self.assertTrue(json_file.exists())
            with open(json_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertIn("length_distributions", loaded)
            self.assertIn("transformer_token_budgets", loaded)
            self.assertIn("vocabulary_metrics", loaded)

    def test_generate_markdown_report(self):
        """Test generating and saving Markdown text report."""
        analyzer = IssueTextAnalyzer(self.mock_df)
        with tempfile.TemporaryDirectory() as tmpdir:
            md_file = Path(tmpdir) / "text_report.md"
            report = analyzer.generate_markdown_report(md_file)

            self.assertTrue(md_file.exists())
            self.assertIn("# OpenSourceLens — Phase 3, Step 3.6", report)
            self.assertIn("Text Length Percentile Distributions", report)
            self.assertIn("Transformer Token Budget & Truncation", report)
            self.assertIn("Discriminative Keywords", report)


if __name__ == "__main__":
    unittest.main()
