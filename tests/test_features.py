"""Unit tests for FeatureEngineer in ml.preprocessing.features."""

import unittest
import pandas as pd

from ml.preprocessing.features import FeatureEngineer


class TestFeatureEngineer(unittest.TestCase):
    """Test suite for FeatureEngineer."""

    def setUp(self):
        self.engineer = FeatureEngineer()
        self.sample_df = pd.DataFrame([
            {
                "issue_id": 1,
                "title": "[Bug] Crash in CUDA runtime",
                "clean_title": "Crash in CUDA runtime",
                "body": "Here is error: ```RuntimeError: CUDA out of memory``` from https://repro.com",
                "clean_body": "Here is error: <CODE_BLOCK> from <URL>",
                "clean_title_word_count": 4,
                "clean_body_word_count": 7,
                "author_association": "MEMBER",
                "author_type": "User",
                "comments_count": 3,
                "reactions_count": 2,
                "canonical_label": "bug",
            },
            {
                "issue_id": 2,
                "title": "Add support for fp8",
                "clean_title": "Add support for fp8",
                "body": "Is this possible? Please consider!",
                "clean_body": "Is this possible? Please consider!",
                "clean_title_word_count": 4,
                "clean_body_word_count": 5,
                "author_association": "NONE",
                "author_type": "User",
                "comments_count": 0,
                "reactions_count": 5,
                "canonical_label": "feature",
            },
            {
                "issue_id": 3,
                "title": "Dependabot bump version",
                "clean_title": "Dependabot bump version",
                "body": "",
                "clean_body": "",
                "clean_title_word_count": 3,
                "clean_body_word_count": 0,
                "author_association": "NONE",
                "author_type": "Bot",
                "comments_count": 0,
                "reactions_count": 0,
                "canonical_label": "other",
            },
        ])

    def test_extract_text_complexity_features(self):
        res = self.engineer.extract_text_complexity_features(self.sample_df)

        # Row 0: has code block, has url, has stack trace ("RuntimeError"), has bracket tag in raw title ("[Bug]")
        self.assertTrue(res.loc[0, "has_code_block"])
        self.assertGreaterEqual(res.loc[0, "code_block_count"], 1)
        self.assertTrue(res.loc[0, "has_url"])
        self.assertGreaterEqual(res.loc[0, "url_count"], 1)
        self.assertTrue(res.loc[0, "has_stack_trace"])
        self.assertTrue(res.loc[0, "title_has_bracket_tag"])
        self.assertGreater(res.loc[0, "uppercase_ratio_title"], 0.1)  # "CUDA" is uppercase

        # Row 1: has question mark, has exclamation mark, no code block
        self.assertFalse(res.loc[1, "has_code_block"])
        self.assertFalse(res.loc[1, "has_stack_trace"])
        self.assertFalse(res.loc[1, "title_has_bracket_tag"])
        self.assertEqual(res.loc[1, "question_mark_count"], 1)
        self.assertEqual(res.loc[1, "exclamation_mark_count"], 1)

        # Row 2: empty body -> 0 ratio, no crash
        self.assertEqual(res.loc[2, "body_to_title_word_ratio"], 0.0)

    def test_extract_authorship_features(self):
        res = self.engineer.extract_authorship_features(self.sample_df)

        # Row 0: MEMBER -> True
        self.assertTrue(res.loc[0, "is_internal_author"])
        self.assertFalse(res.loc[0, "is_bot_author"])

        # Row 1: NONE -> False
        self.assertFalse(res.loc[1, "is_internal_author"])
        self.assertFalse(res.loc[1, "is_bot_author"])

        # Row 2: Bot -> True
        self.assertFalse(res.loc[2, "is_internal_author"])
        self.assertTrue(res.loc[2, "is_bot_author"])

    def test_extract_interaction_features(self):
        res = self.engineer.extract_interaction_features(self.sample_df)

        # Row 0: 3 comments + 2 reactions = 5 total
        self.assertEqual(res.loc[0, "interaction_total"], 5)
        # reactions_per_comment: 2 / (3 + 1) = 0.5
        self.assertEqual(res.loc[0, "reactions_per_comment"], 0.5)

        # Row 1: 0 comments, 5 reactions -> 5 / (0 + 1) = 5.0
        self.assertEqual(res.loc[1, "interaction_total"], 5)
        self.assertEqual(res.loc[1, "reactions_per_comment"], 5.0)

    def test_engineer_features_end_to_end(self):
        final_df = self.engineer.engineer_features(self.sample_df)

        for col in FeatureEngineer.DERIVED_FEATURE_COLUMNS:
            self.assertIn(col, final_df.columns)
            # Ensure no NaNs in newly created columns
            self.assertEqual(final_df[col].isna().sum(), 0)

    def test_summarize_features(self):
        final_df = self.engineer.engineer_features(self.sample_df)
        summary = FeatureEngineer.summarize_features(final_df)

        self.assertEqual(summary["total_records"], 3)
        self.assertEqual(summary["text_features"]["has_code_block_count"], 1)
        self.assertEqual(summary["authorship_features"]["internal_authors_count"], 1)
        self.assertEqual(summary["authorship_features"]["bot_authors_count"], 1)

        # Category signal check
        self.assertIn("signals_by_category", summary)
        self.assertEqual(summary["signals_by_category"]["bugs"]["code_block_pct"], 100.0)
        self.assertEqual(summary["signals_by_category"]["features"]["code_block_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
