"""Dataset Filtering and Deduplication Engine for OpenSourceLens.

Eliminates non-viable records from the issue dataset:
- Deduplicates exact (clean_title, clean_body) pairs.
- Filters verified spam, abuse, and invalid issues.
- Filters degenerate text (sub-threshold length / character count).
- Generates transparent audit summaries and retention statistics.
"""

import sys
import argparse
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd


class DataFilter:
    """Manages noise reduction, deduplication, and spam removal."""

    DEFAULT_SPAM_LABELS = ["spam", "abuse", "invalid"]

    def __init__(
        self,
        drop_duplicates: bool = True,
        filter_spam: bool = True,
        filter_degenerate: bool = True,
        min_words: int = 2,
        min_chars: int = 5,
        spam_labels: Optional[List[str]] = None,
    ):
        """Initialize data filter.

        Args:
            drop_duplicates: Whether to drop duplicate title/body pairs.
            filter_spam: Whether to drop issues labeled as spam or abuse.
            filter_degenerate: Whether to drop issues with sub-threshold text length.
            min_words: Minimum word count required in combined_text.
            min_chars: Minimum character count required in clean_title.
            spam_labels: Custom list of spam labels (defaults to ['spam', 'abuse', 'invalid']).
        """
        self.drop_duplicates = drop_duplicates
        self.filter_spam = filter_spam
        self.filter_degenerate = filter_degenerate
        self.min_words = min_words
        self.min_chars = min_chars
        self.spam_labels = [s.lower() for s in (spam_labels or self.DEFAULT_SPAM_LABELS)]

    def filter_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Drop exact duplicate records based on specified text columns.

        Args:
            df: Input DataFrame.
            subset: List of columns to check for duplicates (defaults to ['clean_title', 'clean_body']).

        Returns:
            Tuple of (retained_df, dropped_df).
        """
        target_subset = subset or ["clean_title", "clean_body"]
        # Fallback to title/body if clean columns not present
        actual_subset = [c for c in target_subset if c in df.columns]
        if not actual_subset:
            actual_subset = [c for c in ["title", "body"] if c in df.columns]

        if not actual_subset:
            return df.copy(), pd.DataFrame(columns=list(df.columns) + ["drop_reason"])

        dup_mask = df.duplicated(subset=actual_subset, keep="first")
        retained = df[~dup_mask].copy()
        dropped = df[dup_mask].copy()
        dropped["drop_reason"] = "exact_duplicate"

        return retained, dropped

    def filter_spam_labels(
        self,
        df: pd.DataFrame,
        spam_labels: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Filter out issues marked with spam, abuse, or invalid labels.

        Args:
            df: Input DataFrame.
            spam_labels: Optional list of spam label names.

        Returns:
            Tuple of (retained_df, dropped_df).
        """
        targets = set(s.lower() for s in (spam_labels or self.spam_labels))

        if "raw_labels" not in df.columns:
            return df.copy(), pd.DataFrame(columns=list(df.columns) + ["drop_reason"])

        def has_spam_label(labels) -> bool:
            if not isinstance(labels, list):
                return False
            for lbl in labels:
                name = lbl.lower() if isinstance(lbl, str) else str(lbl).lower()
                if name in targets:
                    return True
            return False

        spam_mask = df["raw_labels"].apply(has_spam_label)
        retained = df[~spam_mask].copy()
        dropped = df[spam_mask].copy()
        dropped["drop_reason"] = "spam_label"

        return retained, dropped

    def filter_degenerate_text(
        self,
        df: pd.DataFrame,
        min_words: int = 2,
        min_chars: int = 5,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Filter out issues with degenerate, sub-threshold text.

        Checks:
            - clean_title char length >= min_chars
            - combined_text word count >= min_words

        Args:
            df: Input DataFrame.
            min_words: Minimum word count threshold.
            min_chars: Minimum character length for title.

        Returns:
            Tuple of (retained_df, dropped_df).
        """
        title_col = "clean_title" if "clean_title" in df.columns else "title"
        text_col = "combined_text" if "combined_text" in df.columns else ("clean_body" if "clean_body" in df.columns else "body")

        if title_col not in df.columns or text_col not in df.columns:
            return df.copy(), pd.DataFrame(columns=list(df.columns) + ["drop_reason"])

        title_too_short = df[title_col].fillna("").str.strip().str.len() < min_chars
        words_too_few = df[text_col].fillna("").str.strip().apply(lambda s: len(s.split())) < min_words

        degenerate_mask = title_too_short | words_too_few
        retained = df[~degenerate_mask].copy()
        dropped = df[degenerate_mask].copy()
        dropped["drop_reason"] = "degenerate_text"

        return retained, dropped

    def filter_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute full filtering pipeline in sequence and generate audit summary.

        Pipeline order:
            1. Spam label removal
            2. Exact duplicate removal
            3. Degenerate text removal

        Args:
            df: Preprocessed issues DataFrame.

        Returns:
            Tuple of (clean_df, audit_summary_dict).
        """
        current_df = df.copy()
        all_dropped = []
        initial_count = len(df)

        # 1. Spam filtering
        if self.filter_spam:
            current_df, dropped_spam = self.filter_spam_labels(current_df, spam_labels=self.spam_labels)
            if not dropped_spam.empty:
                all_dropped.append(dropped_spam)

        # 2. Duplicate filtering
        if self.drop_duplicates:
            current_df, dropped_dups = self.filter_duplicates(current_df)
            if not dropped_dups.empty:
                all_dropped.append(dropped_dups)

        # 3. Degenerate text filtering
        if self.filter_degenerate:
            current_df, dropped_deg = self.filter_degenerate_text(
                current_df, min_words=self.min_words, min_chars=self.min_chars
            )
            if not dropped_deg.empty:
                all_dropped.append(dropped_deg)

        # Combine all dropped records
        dropped_df = pd.concat(all_dropped, ignore_index=True) if all_dropped else pd.DataFrame(columns=list(df.columns) + ["drop_reason"])
        final_count = len(current_df)
        total_dropped = len(dropped_df)
        retention_rate = round(100.0 * final_count / initial_count, 2) if initial_count else 0.0

        # Breakdown by reason
        reasons_breakdown = dropped_df["drop_reason"].value_counts().to_dict() if not dropped_df.empty else {}

        # Breakdown by repository
        repo_breakdown = {}
        if "repo_name" in dropped_df.columns:
            repo_breakdown = dropped_df["repo_name"].value_counts().to_dict()

        audit = {
            "initial_records": initial_count,
            "retained_records": final_count,
            "total_dropped": total_dropped,
            "retention_rate_percent": retention_rate,
            "dropped_by_reason": reasons_breakdown,
            "dropped_by_repository": repo_breakdown,
            "dropped_dataframe": dropped_df,
        }

        return current_df.reset_index(drop=True), audit

    @staticmethod
    def format_audit_report(audit: Dict[str, Any]) -> str:
        """Format filtering audit results as a Markdown summary."""
        lines = [
            "### Data Filtering & Noise Reduction Audit",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Initial Issue Count | **{audit['initial_records']:,}** |",
            f"| Retained Issue Count | **{audit['retained_records']:,}** |",
            f"| Total Dropped Records | **{audit['total_dropped']:,}** |",
            f"| Overall Retention Rate | **{audit['retention_rate_percent']}%** |",
            "",
            "#### Filtered Records by Reason",
            "",
            "| Drop Reason | Count | Notes |",
            "|---|---|---|",
        ]

        reasons = audit.get("dropped_by_reason", {})
        descriptions = {
            "spam_label": "Issues explicitly tagged with 'spam', 'abuse', or 'invalid' labels",
            "exact_duplicate": "Exact (title, body) duplicates, retaining first occurrence",
            "degenerate_text": "Sub-threshold text length (< 5 chars or < 2 words)",
        }
        for reason, count in reasons.items():
            desc = descriptions.get(reason, "Filtered record")
            lines.append(f"| `{reason}` | {count:,} | {desc} |")

        if not reasons:
            lines.append("| *None* | 0 | No records dropped |")

        lines.extend([
            "",
            "#### Filtered Records by Repository",
            "",
            "| Repository | Dropped Count |",
            "|---|---|",
        ])

        repos = audit.get("dropped_by_repository", {})
        for repo, count in repos.items():
            lines.append(f"| `{repo}` | {count:,} |")

        if not repos:
            lines.append("| *None* | 0 |")

        return "\n".join(lines)


def main():
    """CLI entrypoint for running filtering on dataset."""
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Filter irrelevant data and duplicates from dataset.")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw data directory")
    args = parser.parse_args()

    from ml.preprocessing.loader import RawDataLoader
    from ml.preprocessing.text_cleaner import IssueTextCleaner
    from ml.preprocessing.timestamp_cleaner import TimestampCleaner
    from ml.preprocessing.missing_handler import MissingValueHandler

    # Load and preprocess
    loader = RawDataLoader(raw_dir=args.data_dir)
    data = loader.load_all()
    df_raw = data["issues"]

    text_cleaner = IssueTextCleaner()
    df_text = text_cleaner.clean_dataframe(df_raw)

    ts_cleaner = TimestampCleaner()
    df_ts = ts_cleaner.clean_dataframe(df_text)

    missing_handler = MissingValueHandler()
    df_missing = missing_handler.clean_dataframe(df_ts)

    # Filter
    data_filter = DataFilter()
    df_clean, audit = data_filter.filter_dataframe(df_missing)

    print("=" * 65)
    print("OPENSOURCELENS — DATA FILTERING & DEDUPLICATION SUMMARY")
    print("=" * 65)
    print(f"Initial Records:       {audit['initial_records']:,}")
    print(f"Retained Records:      {audit['retained_records']:,} ({audit['retention_rate_percent']}%)")
    print(f"Total Records Dropped: {audit['total_dropped']:,}")
    print("Breakdown by Reason:")
    for reason, count in audit["dropped_by_reason"].items():
        print(f"  - {reason:18}: {count:,}")
    print("Breakdown by Repository:")
    for repo, count in audit["dropped_by_repository"].items():
        print(f"  - {repo:25}: {count:,}")
    print("=" * 65)


if __name__ == "__main__":
    main()
