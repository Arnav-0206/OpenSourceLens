"""Timestamp Cleaning and Temporal Feature Engineering for OpenSourceLens.

Parses ISO 8601 strings into timezone-aware UTC datetime objects, validates
chronological invariants, computes resolution durations, issue age, and extracts
calendar/time-of-day features for downstream modeling.
"""

import sys
import argparse
from typing import Dict, Any, List, Optional, Union

import numpy as np
import pandas as pd


class TimestampCleaner:
    """Performs timestamp parsing, invariant verification, and temporal feature extraction."""

    DEFAULT_DATETIME_COLS = ["created_at", "updated_at", "closed_at"]
    DEFAULT_REFERENCE_DATE = "2026-10-19T12:00:00Z"

    def __init__(self, reference_date: Optional[Union[str, pd.Timestamp]] = None):
        """Initialize timestamp cleaner.

        Args:
            reference_date: Snapshot cutoff date used to calculate issue age.
                            Defaults to '2026-10-19T12:00:00Z'.
        """
        if reference_date is None:
            self.reference_date = pd.Timestamp(self.DEFAULT_REFERENCE_DATE, tz="UTC")
        elif isinstance(reference_date, str):
            self.reference_date = pd.Timestamp(reference_date, tz="UTC")
        elif isinstance(reference_date, pd.Timestamp):
            self.reference_date = reference_date.tz_convert("UTC") if reference_date.tzinfo else reference_date.tz_localize("UTC")
        else:
            self.reference_date = pd.Timestamp(self.DEFAULT_REFERENCE_DATE, tz="UTC")

    def parse_timestamps(
        self,
        df: pd.DataFrame,
        cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Convert ISO 8601 string columns to timezone-aware UTC datetimes.

        Args:
            df: Ingested issues DataFrame.
            cols: List of column names to parse (defaults to created_at, updated_at, closed_at).

        Returns:
            DataFrame with datetime64[ns, UTC] columns.
        """
        out_df = df.copy()
        target_cols = cols or self.DEFAULT_DATETIME_COLS

        for col in target_cols:
            if col in out_df.columns:
                out_df[col] = pd.to_datetime(out_df[col], utc=True, errors="coerce")

        return out_df

    def validate_chronological_invariants(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate logical temporal constraints across issue lifecycle.

        Checks:
            1. created_at <= updated_at for all records.
            2. created_at <= closed_at for closed issues with closed_at.
            3. closed_at is null for open issues.
            4. closed_at is present for closed issues.

        Args:
            df: DataFrame with parsed UTC datetime columns.

        Returns:
            Dictionary reporting anomaly counts and offending issue IDs.
        """
        if df.empty:
            return {
                "total_records": 0,
                "created_gt_updated_count": 0,
                "created_gt_closed_count": 0,
                "open_with_closed_at_count": 0,
                "closed_without_closed_at_count": 0,
                "is_valid": True,
            }

        # Invariant 1: created_at <= updated_at
        created_gt_updated = pd.Series([False] * len(df), index=df.index)
        if "created_at" in df.columns and "updated_at" in df.columns:
            valid_mask = df["created_at"].notna() & df["updated_at"].notna()
            created_gt_updated = valid_mask & (df["created_at"] > df["updated_at"])

        # Invariant 2: created_at <= closed_at
        created_gt_closed = pd.Series([False] * len(df), index=df.index)
        if "created_at" in df.columns and "closed_at" in df.columns:
            valid_mask = df["created_at"].notna() & df["closed_at"].notna()
            created_gt_closed = valid_mask & (df["created_at"] > df["closed_at"])

        # Invariant 3: closed_at is null for open issues
        open_with_closed_at = pd.Series([False] * len(df), index=df.index)
        if "state" in df.columns and "closed_at" in df.columns:
            open_with_closed_at = (df["state"] == "open") & df["closed_at"].notna()

        # Invariant 4: closed_at is present for closed issues
        closed_without_closed_at = pd.Series([False] * len(df), index=df.index)
        if "state" in df.columns and "closed_at" in df.columns:
            closed_without_closed_at = (df["state"] == "closed") & df["closed_at"].isna()

        anomalies_count = (
            int(created_gt_updated.sum())
            + int(created_gt_closed.sum())
            + int(open_with_closed_at.sum())
            + int(closed_without_closed_at.sum())
        )

        return {
            "total_records": len(df),
            "created_gt_updated_count": int(created_gt_updated.sum()),
            "created_gt_closed_count": int(created_gt_closed.sum()),
            "open_with_closed_at_count": int(open_with_closed_at.sum()),
            "closed_without_closed_at_count": int(closed_without_closed_at.sum()),
            "is_valid": (anomalies_count == 0),
        }

    def compute_resolution_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute resolution time in days and hours for closed issues.

        Sets resolution duration to NaN for open issues to prevent target leakage.

        Args:
            df: DataFrame with parsed UTC datetime columns.

        Returns:
            DataFrame with resolution_time_days and resolution_time_hours columns.
        """
        out_df = df.copy()

        if "created_at" in out_df.columns and "closed_at" in out_df.columns:
            # Mask for closed issues with valid timestamps
            closed_mask = out_df["closed_at"].notna() & out_df["created_at"].notna()
            if "state" in out_df.columns:
                closed_mask = closed_mask & (out_df["state"] == "closed")

            # Duration in seconds
            duration_secs = pd.Series(np.nan, index=out_df.index, dtype="float64")
            duration_secs[closed_mask] = (
                out_df.loc[closed_mask, "closed_at"] - out_df.loc[closed_mask, "created_at"]
            ).dt.total_seconds()

            # Ensure no negative durations due to clock drift (clip at 0.0)
            duration_secs = duration_secs.clip(lower=0.0)

            out_df["resolution_time_days"] = (duration_secs / 86400.0).round(4)
            out_df["resolution_time_hours"] = (duration_secs / 3600.0).round(2)
        else:
            out_df["resolution_time_days"] = np.nan
            out_df["resolution_time_hours"] = np.nan

        return out_df

    def compute_issue_age(
        self,
        df: pd.DataFrame,
        reference_date: Optional[Union[str, pd.Timestamp]] = None,
    ) -> pd.DataFrame:
        """Compute issue age in days relative to dataset reference timestamp.

        Args:
            df: DataFrame with parsed created_at column.
            reference_date: Optional cutoff date (overrides instance reference_date).

        Returns:
            DataFrame with issue_age_days column.
        """
        out_df = df.copy()
        ref = self.reference_date
        if reference_date is not None:
            if isinstance(reference_date, str):
                ref = pd.Timestamp(reference_date, tz="UTC")
            elif isinstance(reference_date, pd.Timestamp):
                ref = reference_date.tz_convert("UTC") if reference_date.tzinfo else reference_date.tz_localize("UTC")

        if "created_at" in out_df.columns:
            age_secs = (ref - out_df["created_at"]).dt.total_seconds().clip(lower=0.0)
            out_df["issue_age_days"] = (age_secs / 86400.0).round(2)
        else:
            out_df["issue_age_days"] = np.nan

        return out_df

    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract calendar, day-of-week, and time-of-day features from created_at.

        Adds:
            - created_year: Integer year
            - created_month: Integer month (1-12)
            - created_day: Integer day of month (1-31)
            - created_day_of_week: Integer day of week (0=Monday, 6=Sunday)
            - created_hour: Integer hour of day (0-23 UTC)
            - is_weekend: Boolean flag (True if Saturday or Sunday)

        Args:
            df: DataFrame with parsed created_at column.

        Returns:
            DataFrame with extracted temporal feature columns.
        """
        out_df = df.copy()

        if "created_at" in out_df.columns and pd.api.types.is_datetime64_any_dtype(out_df["created_at"]):
            dt = out_df["created_at"].dt
            out_df["created_year"] = dt.year.astype("Int64")
            out_df["created_month"] = dt.month.astype("Int64")
            out_df["created_day"] = dt.day.astype("Int64")
            out_df["created_day_of_week"] = dt.dayofweek.astype("Int64")
            out_df["created_hour"] = dt.hour.astype("Int64")
            out_df["is_weekend"] = (out_df["created_day_of_week"] >= 5).astype(bool)
        else:
            out_df["created_year"] = pd.NA
            out_df["created_month"] = pd.NA
            out_df["created_day"] = pd.NA
            out_df["created_day_of_week"] = pd.NA
            out_df["created_hour"] = pd.NA
            out_df["is_weekend"] = False

        return out_df

    def clean_dataframe(
        self,
        df: pd.DataFrame,
        reference_date: Optional[Union[str, pd.Timestamp]] = None,
    ) -> pd.DataFrame:
        """Execute full timestamp cleaning and feature extraction pipeline.

        Args:
            df: Ingested issues DataFrame.
            reference_date: Optional reference date for issue age computation.

        Returns:
            DataFrame with parsed UTC datetimes, resolution durations, and temporal features.
        """
        # 1. Parse timestamps to UTC
        parsed_df = self.parse_timestamps(df)

        # 2. Compute resolution time
        res_df = self.compute_resolution_time(parsed_df)

        # 3. Compute issue age
        age_df = self.compute_issue_age(res_df, reference_date=reference_date)

        # 4. Extract temporal calendar features
        clean_df = self.extract_temporal_features(age_df)

        return clean_df

    @staticmethod
    def summarize_timestamps(df_clean: pd.DataFrame) -> Dict[str, Any]:
        """Compute summary statistics for parsed timestamps and resolution durations."""
        if df_clean.empty:
            return {}

        total_issues = len(df_clean)
        closed_issues = df_clean[df_clean["state"] == "closed"] if "state" in df_clean.columns else df_clean[df_clean["closed_at"].notna()]
        open_issues = df_clean[df_clean["state"] == "open"] if "state" in df_clean.columns else df_clean[df_clean["closed_at"].isna()]

        res_days = closed_issues["resolution_time_days"].dropna() if "resolution_time_days" in closed_issues.columns else pd.Series([], dtype="float64")
        age_days = df_clean["issue_age_days"].dropna() if "issue_age_days" in df_clean.columns else pd.Series([], dtype="float64")

        # Day of week distribution
        dow_counts = {}
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        if "created_day_of_week" in df_clean.columns:
            for idx, name in enumerate(day_names):
                dow_counts[name] = int((df_clean["created_day_of_week"] == idx).sum())

        weekend_pct = 0.0
        if "is_weekend" in df_clean.columns:
            weekend_pct = round(100.0 * df_clean["is_weekend"].sum() / total_issues, 2) if total_issues else 0.0

        return {
            "total_issues": total_issues,
            "closed_issues_count": len(closed_issues),
            "open_issues_count": len(open_issues),
            "resolution_days": {
                "mean": round(float(res_days.mean()), 2) if not res_days.empty else 0.0,
                "median": round(float(res_days.median()), 2) if not res_days.empty else 0.0,
                "p25": round(float(res_days.quantile(0.25)), 2) if not res_days.empty else 0.0,
                "p75": round(float(res_days.quantile(0.75)), 2) if not res_days.empty else 0.0,
                "min": round(float(res_days.min()), 2) if not res_days.empty else 0.0,
                "max": round(float(res_days.max()), 2) if not res_days.empty else 0.0,
            },
            "issue_age_days": {
                "mean": round(float(age_days.mean()), 2) if not age_days.empty else 0.0,
                "median": round(float(age_days.median()), 2) if not age_days.empty else 0.0,
                "min": round(float(age_days.min()), 2) if not age_days.empty else 0.0,
                "max": round(float(age_days.max()), 2) if not age_days.empty else 0.0,
            },
            "day_of_week_distribution": dow_counts,
            "weekend_percent": weekend_pct,
        }


def main():
    """CLI entrypoint for testing timestamp cleaning on raw dataset."""
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Clean timestamps and engineer temporal features.")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw data directory")
    args = parser.parse_args()

    from ml.preprocessing.loader import RawDataLoader

    loader = RawDataLoader(raw_dir=args.data_dir)
    data = loader.load_all()
    df_raw = data["issues"]

    cleaner = TimestampCleaner()
    parsed_df = cleaner.parse_timestamps(df_raw)
    invariants = cleaner.validate_chronological_invariants(parsed_df)
    clean_df = cleaner.clean_dataframe(df_raw)
    summary = TimestampCleaner.summarize_timestamps(clean_df)

    print("=" * 65)
    print("OPENSOURCELENS — TIMESTAMP CLEANING & TEMPORAL AUDIT")
    print("=" * 65)
    print(f"Total Records Processed:       {invariants['total_records']:,}")
    print(f"Chronological Invariants Valid:{invariants['is_valid']}")
    print(f"Created > Updated Anomalies:   {invariants['created_gt_updated_count']}")
    print(f"Created > Closed Anomalies:    {invariants['created_gt_closed_count']}")
    print(f"Open with closed_at:           {invariants['open_with_closed_at_count']}")
    print("-" * 65)
    print(f"Closed Issues Analyzed:        {summary['closed_issues_count']:,}")
    print(f"Median Resolution Time:        {summary['resolution_days']['median']} days")
    print(f"Mean Resolution Time:          {summary['resolution_days']['mean']} days")
    print(f"Min Resolution Time:           {summary['resolution_days']['min']} days")
    print(f"Max Resolution Time:           {summary['resolution_days']['max']} days")
    print(f"Weekend Creation Ratio:        {summary['weekend_percent']}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
