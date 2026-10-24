"""Unit tests for ml.nlp.enhanced_preprocessor (EnhancedIssueTextPreprocessor)."""

import unittest
import pandas as pd

from ml.nlp.enhanced_preprocessor import EnhancedIssueTextPreprocessor


class TestEnhancedIssueTextPreprocessor(unittest.TestCase):
    """Test suite for EnhancedIssueTextPreprocessor."""

    def setUp(self):
        """Initialize default preprocessor."""
        self.preprocessor = EnhancedIssueTextPreprocessor()

    def test_normalize_hex_addresses(self):
        text = "Fatal crash at address 0x7fff5fbff8e0 during pointer dereference 0x10a2b3c4"
        cleaned = self.preprocessor.clean_text(text)
        self.assertIn("hex_addr", cleaned)
        self.assertNotIn("0x7fff5fbff8e0", cleaned)
        self.assertNotIn("0x10a2b3c4", cleaned)

    def test_normalize_commit_hashes(self):
        text = "Broken since commit a1b2c3d4e5f678901234567890abcdef12345678 in branch main"
        cleaned = self.preprocessor.clean_text(text)
        self.assertIn("commit_hash", cleaned)
        self.assertNotIn("a1b2c3d4e5f678901234567890abcdef12345678", cleaned)

    def test_normalize_versions(self):
        text = "Reproduced on kubernetes v1.28.3 and pytorch 2.1.0-rc1"
        cleaned = self.preprocessor.clean_text(text)
        self.assertIn("version_num", cleaned)
        self.assertNotIn("v1.28.3", cleaned)
        self.assertNotIn("2.1.0-rc1", cleaned)

    def test_normalize_issue_refs(self):
        text = "Regression introduced by #12345 and discussed in #678"
        cleaned = self.preprocessor.clean_text(text)
        self.assertIn("issue_ref", cleaned)
        self.assertNotIn("#12345", cleaned)
        self.assertNotIn("#678", cleaned)

    def test_normalize_file_paths(self):
        text = "Error located in src/compiler/backend.py and config/settings.yaml"
        cleaned = self.preprocessor.clean_text(text)
        self.assertIn("file_path", cleaned)
        self.assertNotIn("backend.py", cleaned)
        self.assertNotIn("settings.yaml", cleaned)

    def test_strip_template_headers(self):
        body = """
        ### Steps to reproduce
        1. Run build command
        ### Expected behavior
        Build should succeed
        ### Actual behavior
        Build fails with error
        ### Environment info
        Ubuntu 22.04
        """
        cleaned = self.preprocessor.clean_text(body, is_title=False)
        self.assertNotIn("steps to reproduce", cleaned)
        self.assertNotIn("expected behavior", cleaned)
        self.assertNotIn("actual behavior", cleaned)
        self.assertNotIn("environment info", cleaned)
        self.assertIn("build fails with error", cleaned)

    def test_preserve_negations(self):
        text = "The button cannot work and does not display anything, failed to open"
        cleaned = self.preprocessor.clean_text(text)
        # Crucial defect indicators must be preserved
        self.assertIn("cannot", cleaned)
        self.assertIn("not", cleaned)
        self.assertIn("failed", cleaned)

    def test_filter_github_noise(self):
        text = "Thanks in advance, please help me with this issue, any updates? ping"
        cleaned = self.preprocessor.clean_text(text)
        self.assertNotIn("thanks in advance", cleaned)
        self.assertNotIn("please", cleaned)
        self.assertNotIn("any updates", cleaned)
        self.assertNotIn("ping", cleaned)

    def test_stack_trace_condensation(self):
        body = """
        Traceback (most recent call last):
          File "/app/main.py", line 42, in run
            val = compute(x)
          File "/app/utils.py", line 105, in compute
            raise IndexError("List index out of bounds")
        IndexError: List index out of bounds
        """
        cleaned = self.preprocessor.clean_text(body, is_title=False)
        # Verbose frames should be condensed, but exception type retained
        self.assertNotIn("line 42", cleaned)
        self.assertNotIn("line 105", cleaned)
        self.assertIn("indexerror", cleaned)

    def test_preprocess_issue_structure(self):
        title = "Fatal crash on startup"
        body = "Application throws NullPointerException on boot"
        res = self.preprocessor.preprocess_issue(title, body)

        self.assertIn("clean_title", res)
        self.assertIn("clean_body", res)
        self.assertIn("combined_text", res)
        self.assertIn("fatal crash on startup", res["clean_title"])
        self.assertIn("nullpointerexception", res["clean_body"])

    def test_transform_dataframe(self):
        df = pd.DataFrame({
            "clean_title": ["Issue 1 title", "Issue 2 title"],
            "clean_body": ["Body 1 with 0x12345678", "Body 2 with v2.0.1"],
        })

        augmented = self.preprocessor.transform_dataframe(df)

        self.assertIn("enhanced_title", augmented.columns)
        self.assertIn("enhanced_body", augmented.columns)
        self.assertIn("enhanced_combined_text", augmented.columns)
        self.assertIn("enhanced_total_word_count", augmented.columns)
        self.assertEqual(len(augmented), 2)
        self.assertIn("hex_addr", augmented.loc[0, "enhanced_body"])
        self.assertIn("version_num", augmented.loc[1, "enhanced_body"])

    def test_empty_and_none_handling(self):
        self.assertEqual(self.preprocessor.clean_text(None), "")
        self.assertEqual(self.preprocessor.clean_text(""), "")
        self.assertEqual(self.preprocessor.clean_text("   "), "")


if __name__ == "__main__":
    unittest.main()
