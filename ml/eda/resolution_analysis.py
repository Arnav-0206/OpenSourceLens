"""Issue Resolution-Time Analysis Module for OpenSourceLens EDA.

Provides comprehensive analytics on issue resolution durations (days and hours),
operational resolution speed tiers, repository turnaround disparities, label and
closure reason impacts, correlation with discussion intensity, and backlog aging for open issues.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import pandas as pd
import numpy as np


class IssueResolutionAnalyzer:
    """Analyzer focused on issue resolution times, turnaround speed, and active backlog aging."""

    RESOLUTION_SPEED_BINS = [-0.001, 1.0, 3.0, 7.0, 14.0, 30.0, np.inf]
    RESOLUTION_SPEED_LABELS = [
        "< 1 Day",
        "1-3 Days",
        "3-7 Days",
        "7-14 Days",
        "14-30 Days",
        "30+ Days",
    ]

    COMMENT_BUCKET_BINS = [-1, 0, 1, 3, 5, 10, np.inf]
    COMMENT_BUCKET_LABELS = [
        "0 Comments",
        "1 Comment",
        "2-3 Comments",
        "4-5 Comments",
        "6-10 Comments",
        "11+ Comments",
    ]

    def __init__(self, data: Union[pd.DataFrame, str, Path] = "data/processed/issues_master.csv"):
        """Initialize resolution analyzer with DataFrame or path to Parquet/CSV file.

        Args:
            data: Loaded pd.DataFrame or path to processed dataset file (.csv or .parquet).

        Raises:
            FileNotFoundError: If input file path does not exist.
            ValueError: If dataset is empty or required columns are missing.
        """
        if isinstance(data, (str, Path)):
            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(f"Dataset file not found at: {path}")

            if path.suffix == ".parquet":
                try:
                    self.df = pd.read_parquet(path)
                except Exception:
                    csv_alt = path.with_suffix(".csv")
                    if csv_alt.exists():
                        self.df = pd.read_csv(csv_alt)
                    else:
                        raise
            elif path.suffix == ".csv":
                self.df = pd.read_csv(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}. Expected .parquet or .csv")
            self.source_path = str(path)
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
            self.source_path = "DataFrame in-memory"
        else:
            raise TypeError(f"Expected DataFrame, str, or Path, got {type(data).__name__}")

        if self.df.empty:
            raise ValueError("Provided dataset is empty (0 rows).")

        if "state" not in self.df.columns or "resolution_time_days" not in self.df.columns:
            raise KeyError("Columns 'state' and 'resolution_time_days' are required for resolution analysis.")

        # Cache closed subset
        self.closed_df = self.df[self.df["state"] == "closed"].copy()

    @staticmethod
    def _compute_numeric_stats(series: pd.Series) -> Dict[str, Any]:
        """Compute robust summary statistics for a numeric Series."""
        clean = pd.to_numeric(series, errors="coerce").dropna()
        clean = clean[clean >= 0]
        count = len(clean)
        if count == 0:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "p10": 0.0,
                "p25": 0.0,
                "median": 0.0,
                "p75": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "max": 0.0,
                "iqr": 0.0,
                "skewness": 0.0,
            }

        q10 = float(clean.quantile(0.10))
        q25 = float(clean.quantile(0.25))
        median = float(clean.median())
        q75 = float(clean.quantile(0.75))
        q90 = float(clean.quantile(0.90))
        q95 = float(clean.quantile(0.95))
        q99 = float(clean.quantile(0.99))
        mean = float(clean.mean())
        std = float(clean.std()) if count > 1 else 0.0
        skew = float(clean.skew()) if count > 2 and std > 0 else 0.0

        return {
            "count": int(count),
            "mean": round(mean, 2),
            "std": round(std, 2),
            "min": round(float(clean.min()), 2),
            "p10": round(q10, 2),
            "p25": round(q25, 2),
            "median": round(median, 2),
            "p75": round(q75, 2),
            "p90": round(q90, 2),
            "p95": round(q95, 2),
            "p99": round(q99, 2),
            "max": round(float(clean.max()), 2),
            "iqr": round(q75 - q25, 2),
            "skewness": round(skew, 2),
        }

    def get_resolution_distribution(self, unit: str = "days") -> Dict[str, Any]:
        """Compute parametric and non-parametric distribution statistics for resolution duration.

        Args:
            unit: 'days' or 'hours'.

        Returns:
            Dictionary containing 11-point percentiles, mean, std, IQR, and skewness.
        """
        if self.closed_df.empty:
            return self._compute_numeric_stats(pd.Series(dtype=float))

        col = "resolution_time_days" if unit.lower() == "days" else "resolution_time_hours"
        if col not in self.closed_df.columns:
            if unit.lower() == "hours" and "resolution_time_days" in self.closed_df.columns:
                series = self.closed_df["resolution_time_days"] * 24.0
            else:
                return self._compute_numeric_stats(pd.Series(dtype=float))
        else:
            series = self.closed_df[col]

        return self._compute_numeric_stats(series)

    def get_resolution_speed_buckets(self) -> pd.DataFrame:
        """Categorize closed issues into standard operational turnaround duration tiers."""
        if self.closed_df.empty:
            return pd.DataFrame(columns=[
                "speed_tier", "closed_issue_count", "percentage", "cumulative_count", "cumulative_percentage"
            ])

        valid_days = pd.to_numeric(self.closed_df["resolution_time_days"], errors="coerce").dropna()
        valid_days = valid_days[valid_days >= 0]
        total = len(valid_days)
        if total == 0:
            return pd.DataFrame(columns=[
                "speed_tier", "closed_issue_count", "percentage", "cumulative_count", "cumulative_percentage"
            ])

        binned = pd.cut(
            valid_days,
            bins=self.RESOLUTION_SPEED_BINS,
            labels=self.RESOLUTION_SPEED_LABELS,
            right=True,
        )
        counts = binned.value_counts()[self.RESOLUTION_SPEED_LABELS]
        percentages = (counts / total * 100).round(2)
        cum_counts = counts.cumsum()
        cum_percentages = (cum_counts / total * 100).round(2)

        return pd.DataFrame({
            "speed_tier": self.RESOLUTION_SPEED_LABELS,
            "closed_issue_count": counts.values,
            "percentage": percentages.values,
            "cumulative_count": cum_counts.values,
            "cumulative_percentage": cum_percentages.values,
        })

    def get_resolution_by_repository(self) -> pd.DataFrame:
        """Analyze resolution duration variations across target repositories."""
        if "repo_name" not in self.df.columns or self.closed_df.empty:
            return pd.DataFrame()

        rows = []
        for repo, group in self.closed_df.groupby("repo_name"):
            res_days = pd.to_numeric(group["resolution_time_days"], errors="coerce").dropna()
            res_days = res_days[res_days >= 0]
            count = len(res_days)
            if count == 0:
                continue

            q25 = float(res_days.quantile(0.25))
            median = float(res_days.median())
            q75 = float(res_days.quantile(0.75))
            fast_close = int((res_days <= 1.0).sum())
            fast_rate = round(fast_close / count * 100, 2)

            rows.append({
                "repo_name": repo,
                "closed_count": count,
                "mean_days": round(float(res_days.mean()), 2),
                "median_days": round(median, 2),
                "std_days": round(float(res_days.std()), 2) if count > 1 else 0.0,
                "p25_days": round(q25, 2),
                "p75_days": round(q75, 2),
                "iqr_days": round(q75 - q25, 2),
                "max_days": round(float(res_days.max()), 2),
                "fast_closure_count": fast_close,
                "fast_closure_pct": fast_rate,
            })

        return pd.DataFrame(rows).sort_values(by="median_days", ascending=False).reset_index(drop=True)

    def get_resolution_by_reason(self) -> pd.DataFrame:
        """Evaluate resolution durations across GitHub state closure reasons."""
        if "state_reason" not in self.df.columns or self.closed_df.empty:
            return pd.DataFrame()

        rows = []
        for reason, group in self.closed_df.groupby("state_reason"):
            res_days = pd.to_numeric(group["resolution_time_days"], errors="coerce").dropna()
            res_days = res_days[res_days >= 0]
            count = len(res_days)
            if count == 0:
                continue

            q25 = float(res_days.quantile(0.25))
            median = float(res_days.median())
            q75 = float(res_days.quantile(0.75))

            rows.append({
                "state_reason": reason,
                "closed_count": count,
                "mean_days": round(float(res_days.mean()), 2),
                "median_days": round(median, 2),
                "std_days": round(float(res_days.std()), 2) if count > 1 else 0.0,
                "p25_days": round(q25, 2),
                "p75_days": round(q75, 2),
                "iqr_days": round(q75 - q25, 2),
                "max_days": round(float(res_days.max()), 2),
            })

        return pd.DataFrame(rows).sort_values(by="closed_count", ascending=False).reset_index(drop=True)

    def get_resolution_by_label(self) -> pd.DataFrame:
        """Compare resolution durations across canonical issue labels."""
        label_col = "canonical_label" if "canonical_label" in self.df.columns else "primary_label"
        if label_col not in self.df.columns or self.closed_df.empty:
            return pd.DataFrame()

        rows = []
        for label, group in self.closed_df.groupby(label_col):
            res_days = pd.to_numeric(group["resolution_time_days"], errors="coerce").dropna()
            res_days = res_days[res_days >= 0]
            count = len(res_days)
            if count == 0:
                continue

            q25 = float(res_days.quantile(0.25))
            median = float(res_days.median())
            q75 = float(res_days.quantile(0.75))

            rows.append({
                "canonical_label": label,
                "closed_count": count,
                "mean_days": round(float(res_days.mean()), 2),
                "median_days": round(median, 2),
                "std_days": round(float(res_days.std()), 2) if count > 1 else 0.0,
                "p25_days": round(q25, 2),
                "p75_days": round(q75, 2),
                "iqr_days": round(q75 - q25, 2),
                "max_days": round(float(res_days.max()), 2),
            })

        return pd.DataFrame(rows).sort_values(by="median_days", ascending=False).reset_index(drop=True)

    def get_resolution_by_comment_volume(self) -> pd.DataFrame:
        """Analyze resolution duration progression across comment volume buckets."""
        if "comments_count" not in self.df.columns or self.closed_df.empty:
            return pd.DataFrame()

        closed = self.closed_df.copy()
        closed["comment_bucket"] = pd.cut(
            closed["comments_count"],
            bins=self.COMMENT_BUCKET_BINS,
            labels=self.COMMENT_BUCKET_LABELS,
            right=True,
        )

        rows = []
        for bucket in self.COMMENT_BUCKET_LABELS:
            group = closed[closed["comment_bucket"] == bucket]
            res_days = pd.to_numeric(group["resolution_time_days"], errors="coerce").dropna()
            res_days = res_days[res_days >= 0]
            count = len(res_days)
            if count > 0:
                q25 = float(res_days.quantile(0.25))
                median = float(res_days.median())
                q75 = float(res_days.quantile(0.75))
                mean_v = round(float(res_days.mean()), 2)
                median_v = round(median, 2)
                p25_v = round(q25, 2)
                p75_v = round(q75, 2)
                iqr_v = round(q75 - q25, 2)
                max_v = round(float(res_days.max()), 2)
            else:
                mean_v = median_v = p25_v = p75_v = iqr_v = max_v = 0.0

            rows.append({
                "comment_bucket": bucket,
                "closed_count": count,
                "mean_days": mean_v,
                "median_days": median_v,
                "p25_days": p25_v,
                "p75_days": p75_v,
                "iqr_days": iqr_v,
                "max_days": max_v,
            })

        return pd.DataFrame(rows)

    def get_backlog_aging(self) -> Dict[str, Any]:
        """Compute backlog age statistics for open issues based on issue_age_days."""
        open_df = self.df[self.df["state"] == "open"]
        if open_df.empty or "issue_age_days" not in open_df.columns:
            return self._compute_numeric_stats(pd.Series(dtype=float))

        return self._compute_numeric_stats(open_df["issue_age_days"])

    def compute_full_resolution_analysis(self) -> Dict[str, Any]:
        """Aggregate all resolution analysis dimensions into a consolidated dictionary."""
        return {
            "resolution_distribution_days": self.get_resolution_distribution(unit="days"),
            "resolution_distribution_hours": self.get_resolution_distribution(unit="hours"),
            "speed_buckets": self.get_resolution_speed_buckets().to_dict(orient="records"),
            "resolution_by_repository": self.get_resolution_by_repository().to_dict(orient="records"),
            "resolution_by_reason": self.get_resolution_by_reason().to_dict(orient="records"),
            "resolution_by_label": self.get_resolution_by_label().to_dict(orient="records"),
            "resolution_by_comment_volume": self.get_resolution_by_comment_volume().to_dict(orient="records"),
            "backlog_aging": self.get_backlog_aging(),
        }

    def generate_summary_report(self) -> str:
        """Generate a concise, formatted text summary of the resolution time findings."""
        dist_days = self.get_resolution_distribution(unit="days")
        dist_hours = self.get_resolution_distribution(unit="hours")
        speed_df = self.get_resolution_speed_buckets()
        repo_df = self.get_resolution_by_repository()
        reason_df = self.get_resolution_by_reason()
        label_df = self.get_resolution_by_label()
        comment_df = self.get_resolution_by_comment_volume()
        backlog = self.get_backlog_aging()

        lines = [
            "# OpenSourceLens — Resolution-Time Analysis Summary",
            "",
            "## 1. Overall Resolution Distribution (Closed Issues)",
            f"- **Closed Issues Analyzed:** {dist_days['count']:,}",
            f"- **Mean Resolution Duration:** {dist_days['mean']} days ({dist_hours['mean']} hours)",
            f"- **Median Resolution Duration:** {dist_days['median']} days ({dist_hours['median']} hours)",
            f"- **Standard Deviation:** {dist_days['std']} days (Skewness: {dist_days['skewness']})",
            f"- **Percentiles (Days):** p10={dist_days['p10']}, p25={dist_days['p25']}, p50={dist_days['median']}, p75={dist_days['p75']}, p90={dist_days['p90']}, p95={dist_days['p95']}, p99={dist_days['p99']}, Max={dist_days['max']}",
            "",
            "## 2. Operational Resolution Speed Tiers",
            "| Speed Tier | Closed Issues | Percentage | Cumulative % |",
            "|---|---|---|---|",
        ]
        for _, r in speed_df.iterrows():
            lines.append(f"| {r['speed_tier']} | {r['closed_issue_count']:,} | {r['percentage']:.2f}% | {r['cumulative_percentage']:.2f}% |")

        lines.extend([
            "",
            "## 3. Resolution by Repository",
            "| Repository | Closed Issues | Mean Days | Median Days | p75 Days | Fast Closure (<24h) % |",
            "|---|---|---|---|---|---|",
        ])
        for _, r in repo_df.iterrows():
            lines.append(f"| `{r['repo_name']}` | {r['closed_count']:,} | {r['mean_days']:.2f} | {r['median_days']:.2f} | {r['p75_days']:.2f} | {r['fast_closure_pct']:.2f}% |")

        lines.extend([
            "",
            "## 4. Resolution by Closure Reason",
            "| State Reason | Closed Issues | Mean Days | Median Days | p75 Days | Max Days |",
            "|---|---|---|---|---|---|",
        ])
        for _, r in reason_df.iterrows():
            lines.append(f"| `{r['state_reason']}` | {r['closed_count']:,} | {r['mean_days']:.2f} | {r['median_days']:.2f} | {r['p75_days']:.2f} | {r['max_days']:.2f} |")

        lines.extend([
            "",
            "## 5. Resolution by Canonical Label",
            "| Label | Closed Issues | Mean Days | Median Days | p75 Days | Max Days |",
            "|---|---|---|---|---|---|",
        ])
        for _, r in label_df.iterrows():
            lines.append(f"| `{r['canonical_label']}` | {r['closed_count']:,} | {r['mean_days']:.2f} | {r['median_days']:.2f} | {r['p75_days']:.2f} | {r['max_days']:.2f} |")

        lines.extend([
            "",
            "## 6. Resolution Duration vs. Comment Volume",
            "| Comment Bucket | Closed Issues | Mean Days | Median Days | p75 Days |",
            "|---|---|---|---|---|",
        ])
        for _, r in comment_df.iterrows():
            lines.append(f"| {r['comment_bucket']} | {r['closed_count']:,} | {r['mean_days']:.2f} | {r['median_days']:.2f} | {r['p75_days']:.2f} |")

        lines.extend([
            "",
            "## 7. Active Backlog Aging (Open Issues)",
            f"- **Open Issues in Backlog:** {backlog['count']:,}",
            f"- **Mean Backlog Age:** {backlog['mean']} days",
            f"- **Median Backlog Age:** {backlog['median']} days (p75: {backlog['p75']} days, Max: {backlog['max']} days)",
        ])

        return "\n".join(lines)
