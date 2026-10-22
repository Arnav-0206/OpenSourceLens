"""Unit tests for IssueTextCleaner in ml.preprocessing.text_cleaner."""

import unittest
import pandas as pd

from ml.preprocessing.text_cleaner import IssueTextCleaner


class TestIssueTextCleaner(unittest.TestCase):
    """Test suite for IssueTextCleaner."""

    def setUp(self):
        self.cleaner = IssueTextCleaner()

    def test_strip_html_comments(self):
        raw = "Start <!-- this is a \n multi-line comment --> End"
        res = self.cleaner.strip_html_comments(raw)
        self.assertEqual(res.strip(), "Start   End")

    def test_strip_html_tags(self):
        raw = "<details><summary>Logs</summary><br>Error &amp; Warning</details>"
        res = self.cleaner.strip_html_tags(raw)
        self.assertNotIn("<details>", res)
        self.assertNotIn("</summary>", res)
        self.assertIn("Error & Warning", res)

    def test_normalize_code_blocks(self):
        raw = "Before\n```python\nimport torch\nx = torch.randn(10)\n```\nAfter"
        res = self.cleaner.normalize_code_blocks(raw)
        self.assertIn("<CODE_BLOCK>", res)
        self.assertNotIn("import torch", res)

    def test_normalize_inline_code(self):
        raw = "Call `torch.cuda.is_available()` before running."
        res = self.cleaner.normalize_inline_code(raw)
        self.assertIn("<CODE>", res)
        self.assertNotIn("torch.cuda", res)

    def test_normalize_urls(self):
        raw = "Check https://github.com/pytorch/pytorch/issues/12345 for more info."
        res = self.cleaner.normalize_urls(raw)
        self.assertIn("<URL>", res)
        self.assertNotIn("https://github.com", res)

    def test_normalize_user_mentions(self):
        raw = "Thanks @alice and @dev-team for fixing."
        res = self.cleaner.normalize_user_mentions(raw)
        self.assertIn("<USER>", res)
        self.assertNotIn("@alice", res)
        self.assertNotIn("@dev-team", res)

        # Ensure emails are not mangled
        email_text = "Contact support@example.com for help."
        res_email = self.cleaner.normalize_user_mentions(email_text)
        self.assertIn("support@example.com", res_email)

    def test_normalize_issue_references(self):
        raw = "Fixed in #1234 and related to GH-5678."
        res = self.cleaner.normalize_issue_references(raw)
        self.assertIn("<REF>", res)
        self.assertNotIn("#1234", res)
        self.assertNotIn("GH-5678", res)

    def test_normalize_whitespace(self):
        raw = "  Line 1\n\n\n\t  Line 2   with   spaces.  "
        res = self.cleaner.normalize_whitespace(raw)
        self.assertEqual(res, "Line 1 Line 2 with spaces.")

    def test_clean_text_full_pipeline(self):
        raw = (
            "<!-- Template header -->\n"
            "### Bug: Crash on GPU\n"
            "When running `model.forward()`, it crashes with:\n"
            "```python\n"
            "RuntimeError: CUDA error: out of memory\n"
            "```\n"
            "Reported by @maintainer in #999. See https://docs.python.org\n"
        )
        res = self.cleaner.clean_text(raw)

        self.assertNotIn("Template header", res)
        self.assertNotIn("RuntimeError", res)
        self.assertIn("<CODE_BLOCK>", res)
        self.assertIn("<CODE>", res)
        self.assertIn("<USER>", res)
        self.assertIn("<REF>", res)
        self.assertIn("<URL>", res)
        self.assertIn("Bug: Crash on GPU", res)

    def test_clean_text_edge_cases(self):
        # None and NaN
        self.assertEqual(self.cleaner.clean_text(None), "")
        self.assertEqual(self.cleaner.clean_text(float("nan")), "")

        # Empty and whitespace
        self.assertEqual(self.cleaner.clean_text(""), "")
        self.assertEqual(self.cleaner.clean_text("   \n\t  "), "")

        # Unicode and emoji preservation
        unicode_str = "Fix label ¯\\_(ツ)_/¯ with emoji 🚀"
        res = self.cleaner.clean_text(unicode_str)
        self.assertIn("¯\\_(ツ)_/¯", res)
        self.assertIn("🚀", res)

    def test_clean_dataframe(self):
        df = pd.DataFrame([
            {
                "issue_id": 1,
                "title": "Bug in editor scrolling",
                "body": "Code here: ```x = 1``` reported by @bob in #10",
            },
            {
                "issue_id": 2,
                "title": "Feature request: Dark mode",
                "body": None,  # null body test
            },
        ])

        df_clean = self.cleaner.clean_dataframe(df)

        self.assertIn("clean_title", df_clean.columns)
        self.assertIn("clean_body", df_clean.columns)
        self.assertIn("combined_text", df_clean.columns)
        self.assertIn("clean_title_word_count", df_clean.columns)
        self.assertIn("clean_body_word_count", df_clean.columns)

        # Row 0 assertions
        self.assertEqual(df_clean.iloc[0]["clean_title"], "Bug in editor scrolling")
        self.assertEqual(df_clean.iloc[0]["clean_title_word_count"], 4)
        self.assertIn("<CODE_BLOCK>", df_clean.iloc[0]["clean_body"])
        self.assertIn("<USER>", df_clean.iloc[0]["clean_body"])
        self.assertIn("<REF>", df_clean.iloc[0]["clean_body"])
        self.assertIn("Bug in editor scrolling", df_clean.iloc[0]["combined_text"])

        # Row 1 (null body) assertions
        self.assertEqual(df_clean.iloc[1]["clean_body"], "")
        self.assertEqual(df_clean.iloc[1]["clean_body_word_count"], 0)
        self.assertEqual(df_clean.iloc[1]["combined_text"], "Feature request: Dark mode")

    def test_summarize_cleaning(self):
        df_raw = pd.DataFrame([
            {"title": "Bug", "body": "<!-- comment --> Heavy log ```print(1)``` with https://example.com"}
        ])
        df_clean = self.cleaner.clean_dataframe(df_raw)
        metrics = IssueTextCleaner.summarize_cleaning(df_raw, df_clean)

        self.assertEqual(metrics["total_records"], 1)
        self.assertEqual(metrics["remaining_html_comments"], 0)
        self.assertEqual(metrics["remaining_raw_code_blocks"], 0)
        self.assertEqual(metrics["remaining_raw_urls"], 0)
        self.assertGreater(metrics["char_reduction_percent"], 0)


if __name__ == "__main__":
    unittest.main()
