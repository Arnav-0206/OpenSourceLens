"""Feature Engineering Engine for OpenSourceLens.

Extracts high-signal derived features from the master issues dataset:
- Text structural & complexity metrics (code blocks, URLs, stack traces, brackets, word ratios).
- Authorship authority indicators (internal maintainers vs. bots).
- Community interaction ratios (reactions, comments, engagement density).
- Partitions features into creation-time static vs. lifecycle dynamic to prevent data leakage.
"""

import re
import sys
import argparse
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Extracts derived numerical, categorical, and structural features from issues."""

    # Keywords indicative of stack traces, errors, and crash reports
    STACK_TRACE_KEYWORDS = [
        "traceback",
        "runtimeerror",
        "exception",
        "error:",
        "failed:",
        "assertionerror",
        "nullpointer",
        "segmentation fault",
        "stack trace",
    ]
    RE_STACK_TRACE = re.compile("|".join(re.escape(k) for k in STACK_TRACE_KEYWORDS), re.IGNORECASE)

    # Bracket tag prefix regex: e.g. [Bug], [Feature Request], [CI], (Bug)
    RE_BRACKET_TAG = re.compile(r"^\s*(\[|\()[^\]\)]+(\]|\))")

    # Author roles associated with repository ownership/maintenance
    INTERNAL_ROLES = {"MEMBER", "OWNER", "COLLABORATOR"}

    DERIVED_FEATURE_COLUMNS = [
        "has_code_block",
        "code_block_count",
        "has_url",
        "url_count",
        "has_stack_trace",
        "title_has_bracket_tag",
        "uppercase_ratio_title",
        "question_mark_count",
        "exclamation_mark_count",
        "body_to_title_word_ratio",
        "avg_word_length_title",
        "avg_word_length_body",
        "is_internal_author",
        "is_bot_author",
        "interaction_total",
        "reactions_per_comment",
    ]

    STATIC_CREATION_FEATURES = [
        "has_code_block",
        "code_block_count",
        "has_url",
        "url_count",
        "has_stack_trace",
        "title_has_bracket_tag",
        "uppercase_ratio_title",
        "question_mark_count",
        "exclamation_mark_count",
        "body_to_title_word_ratio",
        "avg_word_length_title",
        "avg_word_length_body",
        "is_internal_author",
        "is_bot_author",
    ]

    DYNAMIC_LIFECYCLE_FEATURES = [
        "interaction_total",
        "reactions_per_comment",
    ]

    def extract_text_complexity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract text structural signals, complexity ratios, and syntactic indicators.

        Args:
            df: DataFrame containing clean_title, clean_body, and raw body.

        Returns:
            DataFrame enriched with text complexity columns.
        """
        out_df = df.copy()

        # Body source (clean_body has <CODE_BLOCK> and <URL> tokens; raw body has ``` and http)
        clean_body = out_df["clean_body"].fillna("") if "clean_body" in out_df.columns else pd.Series([""] * len(out_df))
        raw_body = out_df["body"].fillna("") if "body" in out_df.columns else pd.Series([""] * len(out_df))
        clean_title = out_df["clean_title"].fillna("") if "clean_title" in out_df.columns else pd.Series([""] * len(out_df))
        raw_title = out_df["title"].fillna("") if "title" in out_df.columns else pd.Series([""] * len(out_df))

        # 1. Code block features
        token_code_count = clean_body.str.count(r"<CODE_BLOCK>")
        raw_code_count = raw_body.str.count(r"```") // 2
        code_counts = np.maximum(token_code_count, raw_code_count)
        out_df["code_block_count"] = code_counts.astype("int64")
        out_df["has_code_block"] = (code_counts > 0).astype(bool)

        # 2. URL features
        token_url_count = clean_body.str.count(r"<URL>")
        raw_url_count = raw_body.str.count(r"https?://")
        url_counts = np.maximum(token_url_count, raw_url_count)
        out_df["url_count"] = url_counts.astype("int64")
        out_df["has_url"] = (url_counts > 0).astype(bool)

        # 3. Stack trace detection
        out_df["has_stack_trace"] = raw_body.apply(lambda b: bool(self.RE_STACK_TRACE.search(b))).astype(bool)

        # 4. Bracket tag prefix in title
        out_df["title_has_bracket_tag"] = raw_title.apply(lambda t: bool(self.RE_BRACKET_TAG.search(t))).astype(bool)

        # 5. Uppercase character ratio in title
        def compute_uppercase_ratio(title: str) -> float:
            clean = title.strip()
            if not clean:
                return 0.0
            alphas = [c for c in clean if c.isalpha()]
            if not alphas:
                return 0.0
            return round(sum(1 for c in alphas if c.isupper()) / len(alphas), 4)

        out_df["uppercase_ratio_title"] = clean_title.apply(compute_uppercase_ratio)

        # 6. Punctuation counts (title + body)
        out_df["question_mark_count"] = (clean_title.str.count(r"\?") + clean_body.str.count(r"\?")).astype("int64")
        out_df["exclamation_mark_count"] = (clean_title.str.count(r"!") + clean_body.str.count(r"!")).astype("int64")

        # 7. Word & character ratios
        title_words = out_df["clean_title_word_count"] if "clean_title_word_count" in out_df.columns else clean_title.apply(lambda t: len(t.split()))
        body_words = out_df["clean_body_word_count"] if "clean_body_word_count" in out_df.columns else clean_body.apply(lambda b: len(b.split()))

        # Body to title word ratio (safe division)
        out_df["body_to_title_word_ratio"] = (
            body_words / np.maximum(title_words, 1)
        ).round(2).astype("float64")

        # Average word length in title and body
        def avg_word_len(text: str) -> float:
            words = text.split()
            if not words:
                return 0.0
            total_chars = sum(len(w) for w in words)
            return round(total_chars / len(words), 2)

        out_df["avg_word_length_title"] = clean_title.apply(avg_word_len)
        out_df["avg_word_length_body"] = clean_body.apply(avg_word_len)

        return out_df

    def extract_authorship_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract author authority indicators and account type features.

        Args:
            df: DataFrame containing author_association and author_type.

        Returns:
            DataFrame enriched with authorship indicator columns.
        """
        out_df = df.copy()

        # Internal maintainer vs external user
        if "author_association" in out_df.columns:
            out_df["is_internal_author"] = out_df["author_association"].isin(self.INTERNAL_ROLES).astype(bool)
        else:
            out_df["is_internal_author"] = False

        # Bot account
        if "author_type" in out_df.columns:
            out_df["is_bot_author"] = (out_df["author_type"].str.lower() == "bot").astype(bool)
        else:
            out_df["is_bot_author"] = False

        return out_df

    def extract_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract engagement intensity and reaction density metrics.

        Args:
            df: DataFrame containing comments_count and reactions_count.

        Returns:
            DataFrame enriched with interaction columns.
        """
        out_df = df.copy()

        comments = out_df["comments_count"] if "comments_count" in out_df.columns else pd.Series(0, index=out_df.index)
        reactions = out_df["reactions_count"] if "reactions_count" in out_df.columns else pd.Series(0, index=out_df.index)

        out_df["interaction_total"] = (comments + reactions).astype("int64")

        # Reactions per comment (+ 1 denominator smoothing to prevent division by zero)
        out_df["reactions_per_comment"] = (reactions / (comments + 1.0)).round(3).astype("float64")

        return out_df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute full feature engineering pipeline sequentially.

        Args:
            df: Master issues DataFrame.

        Returns:
            Enriched DataFrame with all 16 derived features.
        """
        # 1. Text complexity features
        df_text = self.extract_text_complexity_features(df)

        # 2. Authorship features
        df_author = self.extract_authorship_features(df_text)

        # 3. Community interaction features
        df_final = self.extract_interaction_features(df_author)

        return df_final

    @classmethod
    def summarize_features(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute statistical summary for all engineered features."""
        if df.empty:
            return {}

        total = len(df)
        summary = {
            "total_records": total,
            "text_features": {
                "has_code_block_count": int(df["has_code_block"].sum()) if "has_code_block" in df.columns else 0,
                "has_code_block_pct": round(100.0 * df["has_code_block"].sum() / total, 2) if "has_code_block" in df.columns else 0.0,
                "has_stack_trace_count": int(df["has_stack_trace"].sum()) if "has_stack_trace" in df.columns else 0,
                "has_stack_trace_pct": round(100.0 * df["has_stack_trace"].sum() / total, 2) if "has_stack_trace" in df.columns else 0.0,
                "title_has_bracket_tag_count": int(df["title_has_bracket_tag"].sum()) if "title_has_bracket_tag" in df.columns else 0,
                "title_has_bracket_tag_pct": round(100.0 * df["title_has_bracket_tag"].sum() / total, 2) if "title_has_bracket_tag" in df.columns else 0.0,
                "mean_uppercase_ratio_title": round(float(df["uppercase_ratio_title"].mean()), 4) if "uppercase_ratio_title" in df.columns else 0.0,
                "mean_body_to_title_word_ratio": round(float(df["body_to_title_word_ratio"].mean()), 2) if "body_to_title_word_ratio" in df.columns else 0.0,
            },
            "authorship_features": {
                "internal_authors_count": int(df["is_internal_author"].sum()) if "is_internal_author" in df.columns else 0,
                "internal_authors_pct": round(100.0 * df["is_internal_author"].sum() / total, 2) if "is_internal_author" in df.columns else 0.0,
                "bot_authors_count": int(df["is_bot_author"].sum()) if "is_bot_author" in df.columns else 0,
                "bot_authors_pct": round(100.0 * df["is_bot_author"].sum() / total, 2) if "is_bot_author" in df.columns else 0.0,
            },
            "interaction_features": {
                "mean_interaction_total": round(float(df["interaction_total"].mean()), 2) if "interaction_total" in df.columns else 0.0,
                "median_interaction_total": round(float(df["interaction_total"].median()), 2) if "interaction_total" in df.columns else 0.0,
                "mean_reactions_per_comment": round(float(df["reactions_per_comment"].mean()), 3) if "reactions_per_comment" in df.columns else 0.0,
            },
        }

        # Category signal comparison: code blocks and stack traces in bugs vs features
        if "canonical_label" in df.columns and "has_code_block" in df.columns:
            bugs = df[df["canonical_label"] == "bug"]
            features = df[df["canonical_label"] == "feature"]

            summary["signals_by_category"] = {
                "bugs": {
                    "total": len(bugs),
                    "code_block_pct": round(100.0 * bugs["has_code_block"].sum() / len(bugs), 2) if len(bugs) else 0.0,
                    "stack_trace_pct": round(100.0 * bugs["has_stack_trace"].sum() / len(bugs), 2) if len(bugs) else 0.0,
                },
                "features": {
                    "total": len(features),
                    "code_block_pct": round(100.0 * features["has_code_block"].sum() / len(features), 2) if len(features) else 0.0,
                    "stack_trace_pct": round(100.0 * features["has_stack_trace"].sum() / len(features), 2) if len(features) else 0.0,
                },
            }

        return summary


def main():
    """CLI entrypoint for testing feature engineering on raw dataset."""
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Extract derived features from issues dataset.")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw data directory")
    args = parser.parse_args()

    from ml.preprocessing.dataset_builder import IssueDatasetBuilder

    builder = IssueDatasetBuilder()
    master_df, _ = builder.build(raw_dir=args.data_dir)

    engineer = FeatureEngineer()
    df_features = engineer.engineer_features(master_df)
    summary = FeatureEngineer.summarize_features(df_features)

    print("=" * 65)
    print("OPENSOURCELENS — DERIVED FEATURE ENGINEERING SUMMARY")
    print("=" * 65)
    print(f"Total Issues Processed:        {summary['total_records']:,}")
    print(f"Issues with Code Blocks:       {summary['text_features']['has_code_block_count']} ({summary['text_features']['has_code_block_pct']}%)")
    print(f"Issues with Stack Traces:      {summary['text_features']['has_stack_trace_count']} ({summary['text_features']['has_stack_trace_pct']}%)")
    print(f"Issues with Bracket Prefixes:  {summary['text_features']['title_has_bracket_tag_count']} ({summary['text_features']['title_has_bracket_tag_pct']}%)")
    print(f"Internal Maintainer Authors:   {summary['authorship_features']['internal_authors_count']} ({summary['authorship_features']['internal_authors_pct']}%)")
    print(f"Bot Authors:                   {summary['authorship_features']['bot_authors_count']} ({summary['authorship_features']['bot_authors_pct']}%)")
    print(f"Mean Total Interactions:       {summary['interaction_features']['mean_interaction_total']}")
    if "signals_by_category" in summary:
        print("-" * 65)
        print("CATEGORY SIGNAL COMPARISON (Bug vs. Feature):")
        b = summary["signals_by_category"]["bugs"]
        f = summary["signals_by_category"]["features"]
        print(f"  - Bugs (N={b['total']:,}):     Code Blocks: {b['code_block_pct']}%, Stack Traces: {b['stack_trace_pct']}%")
        print(f"  - Features (N={f['total']:,}): Code Blocks: {f['code_block_pct']}%, Stack Traces: {f['stack_trace_pct']}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
