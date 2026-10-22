"""Text Cleaning and Normalization Engine for OpenSourceLens.

Sanitizes raw GitHub issue titles and bodies by stripping HTML comments and
boilerplate templates, normalizing code blocks, URLs, user mentions, issue
references, standardizing whitespace, and constructing unified text features.
"""

import re
import sys
import html
import argparse
from typing import Dict, Any, Optional, Union

import pandas as pd


class IssueTextCleaner:
    """Performs modular text sanitization and normalization on issue titles and bodies."""

    # Pre-compiled regular expressions for performance on large issue corpora
    RE_HTML_COMMENT = re.compile(r"<!--[\s\S]*?-->")
    RE_HTML_TAG = re.compile(r"<[^>]+>")
    RE_CODE_BLOCK = re.compile(r"```[\s\S]*?```")
    RE_INLINE_CODE = re.compile(r"`[^`\n]+`")
    RE_URL = re.compile(r"https?://[^\s)]+")
    RE_USER_MENTION = re.compile(r"(?<!\w)@([a-zA-Z0-9_\-]+)")
    RE_ISSUE_REF = re.compile(r"(?<!\w)(?:#|GH-)([0-9]+)\b", re.IGNORECASE)
    RE_WHITESPACE = re.compile(r"[\r\n\t]+")
    RE_MULTISPACE = re.compile(r" {2,}")

    def __init__(
        self,
        strip_comments: bool = True,
        strip_tags: bool = True,
        normalize_code: bool = True,
        normalize_inline: bool = True,
        normalize_urls: bool = True,
        normalize_users: bool = True,
        normalize_refs: bool = True,
        lowercase: bool = False,
    ):
        """Initialize cleaner with transformation toggles."""
        self.do_strip_comments = strip_comments
        self.do_strip_tags = strip_tags
        self.do_normalize_code = normalize_code
        self.do_normalize_inline = normalize_inline
        self.do_normalize_urls = normalize_urls
        self.do_normalize_users = normalize_users
        self.do_normalize_refs = normalize_refs
        self.do_lowercase = lowercase

    def strip_html_comments(self, text: str) -> str:
        """Strip HTML comment blocks (e.g., <!-- instructions -->)."""
        return self.RE_HTML_COMMENT.sub(" ", text)

    def strip_html_tags(self, text: str) -> str:
        """Unescape HTML entities and strip HTML markup tags."""
        unescaped = html.unescape(text)
        return self.RE_HTML_TAG.sub(" ", unescaped)

    def normalize_code_blocks(self, text: str, replacement: str = " <CODE_BLOCK> ") -> str:
        """Replace multi-line fenced code blocks with a placeholder token."""
        return self.RE_CODE_BLOCK.sub(replacement, text)

    def normalize_inline_code(self, text: str, replacement: str = " <CODE> ") -> str:
        """Replace inline backtick code snippets with a placeholder token."""
        return self.RE_INLINE_CODE.sub(replacement, text)

    def normalize_urls(self, text: str, replacement: str = " <URL> ") -> str:
        """Replace HTTP and HTTPS URLs with a placeholder token."""
        return self.RE_URL.sub(replacement, text)

    def normalize_user_mentions(self, text: str, replacement: str = " <USER> ") -> str:
        """Replace @username mentions with a placeholder token."""
        return self.RE_USER_MENTION.sub(replacement, text)

    def normalize_issue_references(self, text: str, replacement: str = " <REF> ") -> str:
        """Replace #1234 or GH-1234 issue/PR references with a placeholder token."""
        return self.RE_ISSUE_REF.sub(replacement, text)

    def normalize_whitespace(self, text: str) -> str:
        """Collapse newlines, tabs, and multiple spaces into single space and strip."""
        collapsed = self.RE_WHITESPACE.sub(" ", text)
        collapsed = self.RE_MULTISPACE.sub(" ", collapsed)
        return collapsed.strip()

    def clean_text(self, text: Optional[str]) -> str:
        """Apply configured text sanitization pipeline to a string.

        Args:
            text: Raw input string (handles None, NaN, empty strings gracefully).

        Returns:
            Cleaned and normalized text string.
        """
        if text is None or pd.isna(text):
            return ""

        s = str(text)
        if not s.strip():
            return ""

        # 1. Strip HTML comments (template instructions) first
        if self.do_strip_comments:
            s = self.strip_html_comments(s)

        # 2. Normalize code blocks before stripping tags (use intermediate token to prevent HTML tag stripping)
        if self.do_normalize_code:
            s = self.normalize_code_blocks(s, replacement=" __TOKEN_CODE_BLOCK__ ")

        # 3. Normalize inline code (use intermediate token)
        if self.do_normalize_inline:
            s = self.normalize_inline_code(s, replacement=" __TOKEN_INLINE_CODE__ ")

        # 4. Strip remaining HTML tags and entities
        if self.do_strip_tags:
            s = self.strip_html_tags(s)

        # Restore token markers
        if self.do_normalize_code:
            s = s.replace("__TOKEN_CODE_BLOCK__", "<CODE_BLOCK>")
        if self.do_normalize_inline:
            s = s.replace("__TOKEN_INLINE_CODE__", "<CODE>")

        # 5. Normalize URLs
        if self.do_normalize_urls:
            s = self.normalize_urls(s)

        # 6. Normalize user mentions
        if self.do_normalize_users:
            s = self.normalize_user_mentions(s)

        # 7. Normalize issue references
        if self.do_normalize_refs:
            s = self.normalize_issue_references(s)

        # 8. Normalize whitespace
        s = self.normalize_whitespace(s)

        # 9. Optional lowercasing
        if self.do_lowercase:
            s = s.lower()

        return s

    def clean_dataframe(
        self,
        df: pd.DataFrame,
        title_col: str = "title",
        body_col: str = "body",
    ) -> pd.DataFrame:
        """Apply text cleaning across issues DataFrame.

        Creates:
            - clean_title: Cleaned title
            - clean_body: Cleaned body
            - combined_text: clean_title + " " + clean_body
            - clean_title_word_count: Word count of clean_title
            - clean_body_word_count: Word count of clean_body

        Args:
            df: Ingested issues DataFrame.
            title_col: Name of title column.
            body_col: Name of body column.

        Returns:
            New DataFrame with cleaned text columns.
        """
        out_df = df.copy()

        # Clean title
        if title_col in out_df.columns:
            out_df["clean_title"] = out_df[title_col].apply(self.clean_text)
        else:
            out_df["clean_title"] = ""

        # Clean body
        if body_col in out_df.columns:
            out_df["clean_body"] = out_df[body_col].apply(self.clean_text)
        else:
            out_df["clean_body"] = ""

        # Combined text
        out_df["combined_text"] = (
            out_df["clean_title"].str.strip() + " " + out_df["clean_body"].str.strip()
        ).str.strip()

        # Word counts
        out_df["clean_title_word_count"] = out_df["clean_title"].apply(lambda t: len(t.split()) if t else 0)
        out_df["clean_body_word_count"] = out_df["clean_body"].apply(lambda b: len(b.split()) if b else 0)

        return out_df

    @staticmethod
    def summarize_cleaning(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> Dict[str, Any]:
        """Compute before vs. after text cleaning metrics."""
        raw_body_chars = df_raw["body"].fillna("").str.len().sum()
        clean_body_chars = df_clean["clean_body"].str.len().sum()
        char_reduction_pct = round(100.0 * (raw_body_chars - clean_body_chars) / raw_body_chars, 2) if raw_body_chars > 0 else 0.0

        raw_body_words = df_raw["body"].fillna("").str.split().str.len().sum()
        clean_body_words = df_clean["clean_body_word_count"].sum()
        word_reduction_pct = round(100.0 * (raw_body_words - clean_body_words) / raw_body_words, 2) if raw_body_words > 0 else 0.0

        # Check remaining artifacts
        re_comment = re.compile(r"<!--[\s\S]*?-->")
        re_code = re.compile(r"```[\s\S]*?```")
        re_url = re.compile(r"https?://[^\s)]+")

        remaining_comments = int(df_clean["clean_body"].apply(lambda b: bool(re_comment.search(b))).sum())
        remaining_code = int(df_clean["clean_body"].apply(lambda b: bool(re_code.search(b))).sum())
        remaining_raw_urls = int(df_clean["clean_body"].apply(lambda b: bool(re_url.search(b))).sum())

        return {
            "total_records": len(df_clean),
            "raw_body_total_chars": int(raw_body_chars),
            "clean_body_total_chars": int(clean_body_chars),
            "char_reduction_percent": char_reduction_pct,
            "raw_body_total_words": int(raw_body_words),
            "clean_body_total_words": int(clean_body_words),
            "word_reduction_percent": word_reduction_pct,
            "avg_clean_title_words": round(float(df_clean["clean_title_word_count"].mean()), 1),
            "avg_clean_body_words": round(float(df_clean["clean_body_word_count"].mean()), 1),
            "remaining_html_comments": remaining_comments,
            "remaining_raw_code_blocks": remaining_code,
            "remaining_raw_urls": remaining_raw_urls,
        }


def main():
    """CLI entrypoint for testing text cleaning on raw dataset."""
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Clean raw issue text.")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw data directory")
    args = parser.parse_args()

    from ml.preprocessing.loader import RawDataLoader

    loader = RawDataLoader(raw_dir=args.data_dir)
    data = loader.load_all()
    df_raw = data["issues"]

    cleaner = IssueTextCleaner()
    df_clean = cleaner.clean_dataframe(df_raw)
    metrics = IssueTextCleaner.summarize_cleaning(df_raw, df_clean)

    print("=" * 65)
    print("OPENSOURCELENS — TEXT CLEANING SUMMARY")
    print("=" * 65)
    print(f"Total Issues Cleaned:        {metrics['total_records']:,}")
    print(f"Raw Body Characters:        {metrics['raw_body_total_chars']:,}")
    print(f"Clean Body Characters:      {metrics['clean_body_total_chars']:,} (-{metrics['char_reduction_percent']}%)")
    print(f"Raw Body Words:             {metrics['raw_body_total_words']:,}")
    print(f"Clean Body Words:           {metrics['clean_body_total_words']:,} (-{metrics['word_reduction_percent']}%)")
    print(f"Avg Clean Words Per Title:  {metrics['avg_clean_title_words']}")
    print(f"Avg Clean Words Per Body:   {metrics['avg_clean_body_words']}")
    print(f"Remaining HTML Comments:    {metrics['remaining_html_comments']} (Target: 0)")
    print(f"Remaining Raw Code Blocks:  {metrics['remaining_raw_code_blocks']} (Target: 0)")
    print(f"Remaining Raw URLs:         {metrics['remaining_raw_urls']} (Target: 0)")
    print("=" * 65)


if __name__ == "__main__":
    main()
