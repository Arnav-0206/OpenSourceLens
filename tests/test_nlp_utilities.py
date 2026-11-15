"""Unit tests for backend NLP services (text cleaner and embedder)."""

import numpy as np
import pandas as pd
import pytest

from backend.app.services.nlp.text_cleaner import IssueTextCleaner
from backend.app.services.nlp.embedder import IssueEmbedder


class TestIssueTextCleaner:
    def setup_method(self):
        self.cleaner = IssueTextCleaner()

    def test_strip_html_comments(self):
        raw = "Start <!-- this is a \n multi-line comment --> End"
        res = self.cleaner.strip_html_comments(raw)
        assert res.strip() == "Start   End"

    def test_strip_html_tags(self):
        raw = "<details><summary>Logs</summary><br>Error &amp; Warning</details>"
        res = self.cleaner.strip_html_tags(raw)
        assert "<details>" not in res
        assert "</summary>" not in res
        assert "Error & Warning" in res

    def test_normalize_code_blocks(self):
        raw = "Before\n```python\nimport torch\nx = torch.randn(10)\n```\nAfter"
        res = self.cleaner.normalize_code_blocks(raw)
        assert "<CODE_BLOCK>" in res
        assert "import torch" not in res

    def test_normalize_inline_code(self):
        raw = "Call `torch.cuda.is_available()` before running."
        res = self.cleaner.normalize_inline_code(raw)
        assert "<CODE>" in res
        assert "torch.cuda" not in res

    def test_normalize_urls(self):
        raw = "Check https://github.com/pytorch/pytorch/issues/12345 for more info."
        res = self.cleaner.normalize_urls(raw)
        assert "<URL>" in res
        assert "https://github.com" not in res

    def test_normalize_user_mentions(self):
        raw = "Thanks @alice and @dev-team for fixing."
        res = self.cleaner.normalize_user_mentions(raw)
        assert "<USER>" in res
        assert "@alice" not in res
        assert "@dev-team" not in res
        # Emails preserved
        email_text = "Contact support@example.com for help."
        res_email = self.cleaner.normalize_user_mentions(email_text)
        assert "support@example.com" in res_email

    def test_normalize_issue_references(self):
        raw = "Fixed in #1234 and related to GH-5678."
        res = self.cleaner.normalize_issue_references(raw)
        assert "<REF>" in res

    def test_clean_dataframe(self):
        df = pd.DataFrame({
            "title": ["Crash in `main()` #10"],
            "body": ["See https://example.com <!-- note --> ```code```"],
        })
        res = self.cleaner.clean_dataframe(df)
        assert "clean_title" in res.columns
        assert "clean_body" in res.columns
        assert "<CODE>" in res["clean_title"].iloc[0]
        assert "<REF>" in res["clean_title"].iloc[0]
        assert "<URL>" in res["clean_body"].iloc[0]
        assert "<CODE_BLOCK>" in res["clean_body"].iloc[0]


class TestIssueEmbedder:
    @classmethod
    def setup_class(cls):
        cls.embedder = IssueEmbedder(model_name="all-MiniLM-L6-v2", normalize_embeddings=True)

    def test_dimension_and_properties(self):
        assert self.embedder.dimension == 384
        assert self.embedder.normalize_embeddings is True

    def test_format_issue_text(self):
        formatted = self.embedder.format_issue_text("Crash on launch", "Segmentation fault in core")
        assert formatted == "Title: Crash on launch | Description: Segmentation fault in core"
        formatted_no_body = self.embedder.format_issue_text("Short title", "")
        assert formatted_no_body == "Title: Short title"

    def test_encode_text_single(self):
        vec = self.embedder.encode_text("Title: Fix memory leak | Description: Buffer overflow")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (384,)
        assert vec.dtype == np.float32
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-4

    def test_encode_texts_batch(self):
        texts = [
            "Title: Memory leak | Description: Memory steadily grows",
            "Title: Add ARM64 | Description: Apple Silicon build support",
        ]
        embeddings = self.embedder.encode_texts(texts, batch_size=2)
        assert embeddings.shape == (2, 384)
        norms = np.linalg.norm(embeddings, axis=1)
        for n in norms:
            assert abs(float(n) - 1.0) < 1e-4
