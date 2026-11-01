"""
OpenSourceLens — Resolution Time Definition & Metric Formalization (Phase 8, Step 8.1).

Defines mathematical formulation of resolution duration, survival-analysis right-censoring
for open issues, sub-minute/bot closure detection, resolution reason disaggregation,
and dataset enrichment for downstream distribution analysis (Step 8.2) and modeling (Step 8.5).
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Standard snapshot reference cutoff for OpenSourceLens
DEFAULT_REFERENCE_DATE: str = "2026-10-19T12:00:00Z"

# Immediate closure threshold: closures under 300 seconds (5 minutes)
DEFAULT_IMMEDIATE_THRESHOLD_SECONDS: float = 300.0

# Operational Speed Tiers
SPEED_TIER_BINS: List[float] = [-0.001, 1.0, 7.0, 30.0, np.inf]
SPEED_TIER_LABELS: List[str] = [
    "< 1 Day",
    "1-7 Days",
    "7-30 Days",
    "30+ Days",
]


class ResolutionTimeDefiner:
    """Formalizes issue resolution metrics, censorship tracking, and data validation."""

    def __init__(
        self,
        reference_date: Optional[Union[str, pd.Timestamp]] = None,
        immediate_threshold_seconds: float = DEFAULT_IMMEDIATE_THRESHOLD_SECONDS,
    ):
        """Initialize the resolution time definer.

        Args:
            reference_date: Snapshot cutoff date for calculating open issue age and duration-at-risk.
                            Defaults to '2026-10-19T12:00:00Z'.
            immediate_threshold_seconds: Second threshold below which closures are marked immediate.
                                         Defaults to 300.0 (5 minutes).
        """
        if reference_date is None:
            self.reference_date = pd.Timestamp(DEFAULT_REFERENCE_DATE, tz="UTC")
        elif isinstance(reference_date, str):
            self.reference_date = pd.Timestamp(reference_date, tz="UTC")
        elif isinstance(reference_date, pd.Timestamp):
            self.reference_date = (
                reference_date if reference_date.tzinfo is not None else reference_date.tz_localize("UTC")
            )
        else:
            raise TypeError(f"Invalid reference_date type: {type(reference_date)}")

        self.immediate_threshold_seconds = float(immediate_threshold_seconds)

    def compute_resolution_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute exact continuous resolution durations, log features, and censorship flags.

        Args:
            df: Ingested master issues DataFrame.

        Returns:
            Enriched DataFrame with formal resolution variables.
        """
        out = df.copy()

        # Ensure datetime columns are timezone-aware UTC datetime64
        for col in ["created_at", "closed_at", "updated_at"]:
            if col in out.columns and not pd.api.types.is_datetime64_any_dtype(out[col]):
                out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")

        if "created_at" not in out.columns:
            raise KeyError("Column 'created_at' is required to compute resolution metrics.")

        has_closed = "closed_at" in out.columns
        has_state = "state" in out.columns

        # Masks
        if has_state:
            closed_mask = (out["state"] == "closed") & (out["closed_at"].notna() if has_closed else False)
            open_mask = (out["state"] == "open")
        elif has_closed:
            closed_mask = out["closed_at"].notna()
            open_mask = out["closed_at"].isna()
        else:
            closed_mask = pd.Series([False] * len(out), index=out.index)
            open_mask = pd.Series([True] * len(out), index=out.index)

        # 1. Exact Second-Level Duration
        res_seconds = pd.Series(np.nan, index=out.index, dtype="float64")
        if has_closed:
            diff_secs = (out.loc[closed_mask, "closed_at"] - out.loc[closed_mask, "created_at"]).dt.total_seconds()
            res_seconds.loc[closed_mask] = diff_secs.clip(lower=0.0)

        out["resolution_time_seconds"] = res_seconds

        # 2. Continuous Hour and Day Formulations
        out["resolution_time_hours"] = res_seconds / 3600.0
        out["resolution_time_days"] = res_seconds / 86400.0

        # 3. Log-Transformed Duration: ln(Delta t_hours + 1.0)
        # Smoothly handles zero-bounded and sub-hour values for classical regression
        out["resolution_time_log_hours"] = np.where(
            res_seconds.notna(),
            np.log(out["resolution_time_hours"] + 1.0),
            np.nan,
        )

        # 4. Immediate Closures Flag (< threshold seconds)
        out["is_immediate_closure"] = False
        if closed_mask.any():
            out.loc[closed_mask, "is_immediate_closure"] = (
                res_seconds.loc[closed_mask] < self.immediate_threshold_seconds
            )

        # 5. Survival Analysis Right-Censorship Tracking
        # is_censored: True for open issues (event has not yet occurred as of snapshot date)
        # event_observed: 1 for closed issues, 0 for open issues
        out["is_censored"] = open_mask
        out["event_observed"] = np.where(closed_mask, 1, 0)

        # 6. Duration at Risk (Exposure Time for Survival Modeling)
        # For closed issues: observed resolution duration
        # For open issues: time between creation and reference snapshot date
        at_risk_days = pd.Series(np.nan, index=out.index, dtype="float64")
        at_risk_days.loc[closed_mask] = out.loc[closed_mask, "resolution_time_days"]

        open_created = out.loc[open_mask, "created_at"]
        open_risk_secs = (self.reference_date - open_created).dt.total_seconds().clip(lower=0.0)
        at_risk_days.loc[open_mask] = open_risk_secs / 86400.0

        out["duration_at_risk_days"] = at_risk_days

        # 7. Operational Speed Tiers
        # Closed issues categorized into discrete speed bins; open issues marked 'Unresolved'
        tier_series = pd.Series("Unresolved", index=out.index, dtype="object")
        if closed_mask.any():
            cut_bins = pd.cut(
                out.loc[closed_mask, "resolution_time_days"],
                bins=SPEED_TIER_BINS,
                labels=SPEED_TIER_LABELS,
                right=False,
            ).astype(str)
            tier_series.loc[closed_mask] = cut_bins

        out["speed_tier"] = tier_series

        # 8. Resolution Type Disaggregation (completed vs rejected vs duplicate vs unresolved)
        out["resolution_type"] = self._derive_resolution_type(out, closed_mask, open_mask)

        return out

    @staticmethod
    def _derive_resolution_type(
        df: pd.DataFrame,
        closed_mask: pd.Series,
        open_mask: pd.Series,
    ) -> pd.Series:
        """Categorize issues into functional resolution types."""
        res_types = pd.Series("unresolved", index=df.index, dtype="object")

        if "state_reason" in df.columns:
            reason = df["state_reason"].astype(str).str.lower().fillna("")

            # Completed / Accepted defect or feature
            completed_mask = closed_mask & (reason.isin(["completed", "resolved"]) | (reason == ""))
            res_types.loc[completed_mask] = "completed"

            # Rejected / Won't fix / Not planned
            rejected_mask = closed_mask & (reason.isin(["not_planned", "wontfix", "not planned", "invalid"]))
            res_types.loc[rejected_mask] = "rejected"

            # Duplicate closure
            duplicate_mask = closed_mask & (reason.str.contains("duplicate"))
            res_types.loc[duplicate_mask] = "duplicate"
        else:
            res_types.loc[closed_mask] = "completed"

        res_types.loc[open_mask] = "unresolved"
        return res_types

    def validate_invariants(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Verify strict chronological and logical consistency invariants.

        Invariants checked:
        1. No negative durations (closed_at >= created_at).
        2. No closed_at on open issues.
        3. No missing closed_at on closed issues.
        4. is_censored aligns strictly with state == 'open'.
        5. resolution_time_seconds is null if and only if is_censored is True.

        Args:
            df: Enriched DataFrame.

        Returns:
            Dictionary with invariant status and anomaly counts.
        """
        total = len(df)
        if total == 0:
            return {"total_records": 0, "all_invariants_pass": True}

        # 1. Negative durations
        neg_count = 0
        if "created_at" in df.columns and "closed_at" in df.columns:
            valid_closed = df["created_at"].notna() & df["closed_at"].notna()
            neg_count = int((df.loc[valid_closed, "closed_at"] < df.loc[valid_closed, "created_at"]).sum())

        # 2. Open issues with closed_at
        open_with_closed = 0
        if "state" in df.columns and "closed_at" in df.columns:
            open_with_closed = int(((df["state"] == "open") & df["closed_at"].notna()).sum())

        # 3. Closed issues without closed_at
        closed_without_closed = 0
        if "state" in df.columns and "closed_at" in df.columns:
            closed_without_closed = int(((df["state"] == "closed") & df["closed_at"].isna()).sum())

        # 4. Censoring alignment
        censored_mismatch = 0
        if "is_censored" in df.columns and "state" in df.columns:
            expected_censored = (df["state"] == "open")
            censored_mismatch = int((df["is_censored"] != expected_censored).sum())

        # 5. Duration nullness matches censoring
        duration_censored_mismatch = 0
        if "resolution_time_seconds" in df.columns and "is_censored" in df.columns:
            duration_censored_mismatch = int((df["resolution_time_seconds"].isna() != df["is_censored"]).sum())

        all_pass = (
            neg_count == 0
            and open_with_closed == 0
            and closed_without_closed == 0
            and censored_mismatch == 0
            and duration_censored_mismatch == 0
        )

        return {
            "total_records": total,
            "negative_durations_count": neg_count,
            "open_with_closed_at_count": open_with_closed,
            "closed_without_closed_at_count": closed_without_closed,
            "censored_mismatch_count": censored_mismatch,
            "duration_censored_mismatch_count": duration_censored_mismatch,
            "all_invariants_pass": all_pass,
        }

    def summarize(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute comprehensive statistical profile of resolution metrics.

        Args:
            df: Enriched DataFrame.

        Returns:
            Dictionary containing descriptive statistics, percentiles, and segment breakdowns.
        """
        closed_df = df[df["is_censored"] == False] if "is_censored" in df.columns else df[df["state"] == "closed"]
        open_df = df[df["is_censored"] == True] if "is_censored" in df.columns else df[df["state"] == "open"]

        closed_hours = closed_df["resolution_time_hours"].dropna()
        closed_days = closed_df["resolution_time_days"].dropna()

        # Percentiles
        q_steps = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        hours_quantiles = {f"p{int(q*100)}": round(float(closed_hours.quantile(q)), 2) for q in q_steps} if not closed_hours.empty else {}
        days_quantiles = {f"p{int(q*100)}": round(float(closed_days.quantile(q)), 4) for q in q_steps} if not closed_days.empty else {}

        # Speed Tier Breakdown
        tier_counts = df["speed_tier"].value_counts().to_dict() if "speed_tier" in df.columns else {}

        # Resolution Type Breakdown
        type_counts = df["resolution_type"].value_counts().to_dict() if "resolution_type" in df.columns else {}

        # Immediate closures
        immediate_count = int(closed_df["is_immediate_closure"].sum()) if "is_immediate_closure" in closed_df.columns else 0
        immediate_pct = round((immediate_count / len(closed_df)) * 100.0, 2) if len(closed_df) > 0 else 0.0

        # Repository-level resolution turnaround
        repo_breakdown = {}
        if "repo_name" in df.columns:
            for repo, grp in df.groupby("repo_name"):
                grp_closed = grp[grp["is_censored"] == False]["resolution_time_hours"].dropna()
                repo_breakdown[repo] = {
                    "total_issues": len(grp),
                    "closed_count": len(grp_closed),
                    "open_count": len(grp[grp["is_censored"] == True]),
                    "median_hours": round(float(grp_closed.median()), 2) if not grp_closed.empty else 0.0,
                    "mean_hours": round(float(grp_closed.mean()), 2) if not grp_closed.empty else 0.0,
                    "std_hours": round(float(grp_closed.std()), 2) if len(grp_closed) > 1 else 0.0,
                }

        # Category-level resolution turnaround
        cat_breakdown = {}
        if "canonical_label" in df.columns:
            for cat, grp in df.groupby("canonical_label"):
                grp_closed = grp[grp["is_censored"] == False]["resolution_time_hours"].dropna()
                cat_breakdown[cat] = {
                    "total_issues": len(grp),
                    "closed_count": len(grp_closed),
                    "median_hours": round(float(grp_closed.median()), 2) if not grp_closed.empty else 0.0,
                    "mean_hours": round(float(grp_closed.mean()), 2) if not grp_closed.empty else 0.0,
                }

        return {
            "dataset_overview": {
                "total_issues": len(df),
                "closed_issues": len(closed_df),
                "open_issues": len(open_df),
                "closed_percentage": round((len(closed_df) / len(df)) * 100.0, 2) if len(df) > 0 else 0.0,
            },
            "resolution_statistics_hours": {
                "mean": round(float(closed_hours.mean()), 2) if not closed_hours.empty else 0.0,
                "median": round(float(closed_hours.median()), 2) if not closed_hours.empty else 0.0,
                "std": round(float(closed_hours.std()), 2) if len(closed_hours) > 1 else 0.0,
                "min": round(float(closed_hours.min()), 2) if not closed_hours.empty else 0.0,
                "max": round(float(closed_hours.max()), 2) if not closed_hours.empty else 0.0,
                "quantiles": hours_quantiles,
            },
            "resolution_statistics_days": {
                "mean": round(float(closed_days.mean()), 4) if not closed_days.empty else 0.0,
                "median": round(float(closed_days.median()), 4) if not closed_days.empty else 0.0,
                "std": round(float(closed_days.std()), 4) if len(closed_days) > 1 else 0.0,
                "min": round(float(closed_days.min()), 4) if not closed_days.empty else 0.0,
                "max": round(float(closed_days.max()), 4) if not closed_days.empty else 0.0,
                "quantiles": days_quantiles,
            },
            "immediate_closures": {
                "threshold_seconds": self.immediate_threshold_seconds,
                "count": immediate_count,
                "percentage_of_closed": immediate_pct,
            },
            "speed_tier_distribution": tier_counts,
            "resolution_type_distribution": type_counts,
            "by_repository": repo_breakdown,
            "by_category": cat_breakdown,
        }

    def save_enriched_dataset(
        self,
        df: pd.DataFrame,
        output_parquet: Union[str, Path] = "data/processed/issues_resolution.parquet",
        manifest_path: Optional[Union[str, Path]] = "data/processed/evaluation/resolution_definition_manifest.json",
    ) -> Tuple[Path, Optional[Path]]:
        """Compute metrics, validate, and serialize enriched dataset and manifest.

        Args:
            df: Input master issues DataFrame.
            output_parquet: Destination parquet path.
            manifest_path: Optional path to export JSON execution manifest.

        Returns:
            Tuple of (output_parquet_path, manifest_path).
        """
        enriched = self.compute_resolution_metrics(df)
        invariants = self.validate_invariants(enriched)

        if not invariants["all_invariants_pass"]:
            raise ValueError(f"Chronological invariant check failed: {invariants}")

        p_out = Path(output_parquet)
        p_out.parent.mkdir(parents=True, exist_ok=True)

        # Save Parquet and CSV
        enriched.to_parquet(p_out, index=False)
        csv_out = p_out.with_suffix(".csv")
        enriched.to_csv(csv_out, index=False)

        p_man = None
        if manifest_path:
            p_man = Path(manifest_path)
            p_man.parent.mkdir(parents=True, exist_ok=True)
            summary = self.summarize(enriched)

            manifest_content = {
                "phase": "Phase 8: Resolution Analysis & Prediction",
                "step": "Step 8.1: Define Resolution Time",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "reference_date_utc": self.reference_date.isoformat(),
                "immediate_threshold_seconds": self.immediate_threshold_seconds,
                "speed_tier_bins_days": SPEED_TIER_BINS,
                "speed_tier_labels": SPEED_TIER_LABELS,
                "invariant_checks": invariants,
                "summary_metrics": summary,
                "file_artifacts": {
                    "parquet_path": str(p_out),
                    "csv_path": str(csv_out),
                    "parquet_size_bytes": p_out.stat().st_size,
                    "csv_size_bytes": csv_out.stat().st_size,
                },
            }

            with open(p_man, "w", encoding="utf-8") as f:
                json.dump(manifest_content, f, indent=2)

        return p_out, p_man
