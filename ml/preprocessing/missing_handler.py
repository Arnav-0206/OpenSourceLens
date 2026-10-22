"""Missing Value Imputation and Completeness Audit for OpenSourceLens.

Systematically handles missing values across the issue dataset:
- Imputes empty body text using issue titles with tracking flags.
- Standardizes state_reason for open and closed issues.
- Ensures label list integrity and flags unlabeled issues.
- Protects resolution duration targets from invalid imputation.
"""

import sys
import argparse
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd


class MissingValueHandler:
    """Manages missing value imputation and completeness auditing."""

    def __init__(
        self,
        impute_body_with_title: bool = True,
        open_state_reason: str = "unresolved",
        closed_state_reason_fallback: str = "closed_unspecified",
    ):
        """Initialize missing value handler.

        Args:
            impute_body_with_title: Whether to impute empty bodies using clean_title.
            open_state_reason: Replacement for null state_reason on open issues.
            closed_state_reason_fallback: Replacement for null state_reason on closed issues.
        """
        self.impute_body_with_title = impute_body_with_title
        self.open_state_reason = open_state_reason
        self.closed_state_reason_fallback = closed_state_reason_fallback

    def handle_body_missingness(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing or empty body text and update combined_text.

        Args:
            df: DataFrame containing clean_title and clean_body.

        Returns:
            DataFrame with imputed clean_body, is_body_imputed, and updated combined_text.
        """
        out_df = df.copy()

        # Identify missing or empty bodies
        has_body_col = "clean_body" in out_df.columns
        has_title_col = "clean_title" in out_df.columns

        if not has_body_col:
            out_df["clean_body"] = ""
            out_df["is_body_imputed"] = False
            return out_df

        # Empty body mask: null, NaN, or whitespace-only
        body_series = out_df["clean_body"]
        empty_mask = body_series.isna() | (body_series.fillna("").str.strip() == "")

        out_df["is_body_imputed"] = empty_mask.astype(bool)

        if self.impute_body_with_title and has_title_col:
            # Impute empty bodies with clean_title
            title_fallback = out_df["clean_title"].fillna("").str.strip()
            out_df.loc[empty_mask, "clean_body"] = title_fallback[empty_mask]

            # Recompute word count for imputed bodies
            if "clean_body_word_count" in out_df.columns:
                out_df.loc[empty_mask, "clean_body_word_count"] = (
                    out_df.loc[empty_mask, "clean_body"].apply(lambda b: len(b.split()) if b else 0)
                )

            # Recompute combined_text
            out_df["combined_text"] = (
                out_df["clean_title"].fillna("").str.strip()
                + " "
                + out_df["clean_body"].fillna("").str.strip()
            ).str.strip()

        return out_df

    def handle_state_reason(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize state_reason categorical values for open and closed issues.

        Args:
            df: DataFrame containing state and state_reason.

        Returns:
            DataFrame with non-null standardized state_reason.
        """
        out_df = df.copy()

        if "state_reason" not in out_df.columns:
            out_df["state_reason"] = self.open_state_reason
            return out_df

        if "state" not in out_df.columns:
            out_df["state_reason"] = out_df["state_reason"].fillna(self.closed_state_reason_fallback)
            return out_df

        # Open issues with null reason -> open_state_reason ('unresolved')
        open_mask = (out_df["state"] == "open") & (out_df["state_reason"].isna() | (out_df["state_reason"] == ""))
        out_df.loc[open_mask, "state_reason"] = self.open_state_reason

        # Closed issues with null reason -> closed_state_reason_fallback ('closed_unspecified')
        closed_mask = (out_df["state"] == "closed") & (out_df["state_reason"].isna() | (out_df["state_reason"] == ""))
        out_df.loc[closed_mask, "state_reason"] = self.closed_state_reason_fallback

        return out_df

    def handle_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure raw_labels contains valid lists and add is_unlabeled flag.

        Args:
            df: DataFrame containing raw_labels and labels_count.

        Returns:
            DataFrame with clean list objects and is_unlabeled boolean flag.
        """
        out_df = df.copy()

        if "raw_labels" in out_df.columns:
            # Replace None / NaN with empty list
            out_df["raw_labels"] = out_df["raw_labels"].apply(
                lambda x: x if isinstance(x, list) else []
            )

        if "labels_count" in out_df.columns:
            out_df["is_unlabeled"] = (out_df["labels_count"] == 0).astype(bool)
        elif "raw_labels" in out_df.columns:
            out_df["labels_count"] = out_df["raw_labels"].apply(len)
            out_df["is_unlabeled"] = (out_df["labels_count"] == 0).astype(bool)
        else:
            out_df["is_unlabeled"] = False

        return out_df

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute full missing value handling pipeline.

        Args:
            df: Issues DataFrame (processed by text and timestamp cleaners).

        Returns:
            DataFrame with missing values handled and tracking flags added.
        """
        # 1. Handle body missingness
        step1 = self.handle_body_missingness(df)

        # 2. Handle state_reason
        step2 = self.handle_state_reason(step1)

        # 3. Handle labels
        step3 = self.handle_labels(step2)

        return step3

    @staticmethod
    def audit_missingness(df: pd.DataFrame) -> Dict[str, Any]:
        """Audit null and missing counts across all columns."""
        if df.empty:
            return {"total_records": 0, "columns": {}}

        total = len(df)
        cols_summary = {}

        for col in df.columns:
            null_count = int(df[col].isna().sum())
            null_pct = round(100.0 * null_count / total, 2)
            cols_summary[col] = {
                "null_count": null_count,
                "null_percent": null_pct,
                "is_complete": (null_count == 0),
            }

        return {
            "total_records": total,
            "columns": cols_summary,
        }

    @staticmethod
    def summarize_imputation(df_before: pd.DataFrame, df_after: pd.DataFrame) -> Dict[str, Any]:
        """Summarize changes performed during imputation."""
        before_empty_bodies = int(
            df_before["clean_body"].isna().sum() + (df_before["clean_body"].fillna("").str.strip() == "").sum()
        ) if "clean_body" in df_before.columns else 0

        after_empty_bodies = int(
            df_after["clean_body"].isna().sum() + (df_after["clean_body"].fillna("").str.strip() == "").sum()
        ) if "clean_body" in df_after.columns else 0

        imputed_bodies = int(df_after["is_body_imputed"].sum()) if "is_body_imputed" in df_after.columns else 0
        unlabeled_count = int(df_after["is_unlabeled"].sum()) if "is_unlabeled" in df_after.columns else 0

        before_null_state_reasons = int(df_before["state_reason"].isna().sum()) if "state_reason" in df_before.columns else 0
        after_null_state_reasons = int(df_after["state_reason"].isna().sum()) if "state_reason" in df_after.columns else 0

        return {
            "total_records": len(df_after),
            "before_empty_bodies": before_empty_bodies,
            "after_empty_bodies": after_empty_bodies,
            "imputed_bodies_count": imputed_bodies,
            "before_null_state_reasons": before_null_state_reasons,
            "after_null_state_reasons": after_null_state_reasons,
            "unlabeled_issues_count": unlabeled_count,
            "unlabeled_percent": round(100.0 * unlabeled_count / len(df_after), 2) if len(df_after) else 0.0,
        }


def main():
    """CLI entrypoint for running missing value handling on dataset."""
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Handle missing values in issue dataset.")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw data directory")
    args = parser.parse_args()

    from ml.preprocessing.loader import RawDataLoader
    from ml.preprocessing.text_cleaner import IssueTextCleaner
    from ml.preprocessing.timestamp_cleaner import TimestampCleaner

    # Load and preprocess
    loader = RawDataLoader(raw_dir=args.data_dir)
    data = loader.load_all()
    df_raw = data["issues"]

    text_cleaner = IssueTextCleaner()
    df_text = text_cleaner.clean_dataframe(df_raw)

    ts_cleaner = TimestampCleaner()
    df_ts = ts_cleaner.clean_dataframe(df_text)

    # Missing value handling
    missing_handler = MissingValueHandler()
    df_final = missing_handler.clean_dataframe(df_ts)
    summary = MissingValueHandler.summarize_imputation(df_ts, df_final)

    print("=" * 65)
    print("OPENSOURCELENS — MISSING VALUE IMPUTATION SUMMARY")
    print("=" * 65)
    print(f"Total Issues Processed:        {summary['total_records']:,}")
    print(f"Empty Bodies Before:           {summary['before_empty_bodies']}")
    print(f"Empty Bodies After:            {summary['after_empty_bodies']} (Target: 0)")
    print(f"Bodies Imputed from Title:     {summary['imputed_bodies_count']}")
    print(f"State Reason Nulls Before:     {summary['before_null_state_reasons']}")
    print(f"State Reason Nulls After:      {summary['after_null_state_reasons']} (Target: 0)")
    print(f"Unlabeled Issues Tracked:      {summary['unlabeled_issues_count']} ({summary['unlabeled_percent']}%)")
    print("=" * 65)


if __name__ == "__main__":
    main()
