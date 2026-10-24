"""Issue Comment and Discussion Analysis Module for OpenSourceLens EDA.

Provides deep analytical capabilities for issue comment distributions, discussion volume
variations across repositories and canonical label categories, comment counts vs. resolution
time duration, reaction dynamics, and qualitative author association profiling from raw comments.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import pandas as pd
import numpy as np


class IssueCommentAnalyzer:
    """Analyzer focused on issue comments, community engagement, and discussion dynamics."""

    COMMENT_BUCKET_BINS = [-1, 0, 1, 3, 5, 10, np.inf]
    COMMENT_BUCKET_LABELS = [
        "0 Comments",
        "1 Comment",
        "2-3 Comments",
        "4-5 Comments",
        "6-10 Comments",
        "11+ Comments",
    ]

    def __init__(
        self,
        data: Union[pd.DataFrame, str, Path] = "data/processed/issues_master.parquet",
        comments_dir: Optional[Union[str, Path]] = "data/raw/comments",
    ):
        """Initialize comment analyzer with DataFrame or path, and optional raw comments directory.

        Args:
            data: Loaded pd.DataFrame or path to processed dataset file.
            comments_dir: Path to directory containing raw harvested comment JSON files.

        Raises:
            FileNotFoundError: If input dataset path does not exist.
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

        if "comments_count" not in self.df.columns:
            raise KeyError("Column 'comments_count' is required for comment analysis.")

        self.comments_dir = Path(comments_dir) if comments_dir else None

    @staticmethod
    def _compute_numeric_stats(series: pd.Series) -> Dict[str, Any]:
        """Compute comprehensive summary statistics for a numeric Series."""
        clean = pd.to_numeric(series, errors="coerce").dropna()
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

    def get_comment_volume_distribution(self) -> Dict[str, Any]:
        """Return parametric and non-parametric distribution statistics for issue comments."""
        return self._compute_numeric_stats(self.df["comments_count"])

    def get_comment_buckets(self) -> pd.DataFrame:
        """Categorize issues into discrete comment volume buckets with percentage and cumulative totals."""
        bucket_series = pd.cut(
            self.df["comments_count"],
            bins=self.COMMENT_BUCKET_BINS,
            labels=self.COMMENT_BUCKET_LABELS,
            right=True,
        )
        counts = bucket_series.value_counts()[self.COMMENT_BUCKET_LABELS]
        total = len(self.df)

        percentages = (counts / total * 100).round(2)
        cum_counts = counts.cumsum()
        cum_percentages = (cum_counts / total * 100).round(2)

        return pd.DataFrame({
            "bucket": self.COMMENT_BUCKET_LABELS,
            "issue_count": counts.values,
            "percentage": percentages.values,
            "cumulative_count": cum_counts.values,
            "cumulative_percentage": cum_percentages.values,
        })

    def get_comments_by_repository(self) -> pd.DataFrame:
        """Compute comment statistics grouped by target repository."""
        if "repo_name" not in self.df.columns:
            return pd.DataFrame()

        rows = []
        for repo, group in self.df.groupby("repo_name"):
            comments = pd.to_numeric(group["comments_count"], errors="coerce").dropna()
            total_issues = len(group)
            zero_comments = int((comments == 0).sum())
            zero_rate = round(zero_comments / total_issues * 100, 2) if total_issues > 0 else 0.0

            rows.append({
                "repo_name": repo,
                "issue_count": total_issues,
                "total_comments": int(comments.sum()),
                "mean_comments": round(float(comments.mean()), 2) if not comments.empty else 0.0,
                "median_comments": round(float(comments.median()), 2) if not comments.empty else 0.0,
                "std_comments": round(float(comments.std()), 2) if len(comments) > 1 else 0.0,
                "p75_comments": round(float(comments.quantile(0.75)), 2) if not comments.empty else 0.0,
                "max_comments": int(comments.max()) if not comments.empty else 0,
                "zero_comment_count": zero_comments,
                "zero_comment_rate_pct": zero_rate,
            })

        return pd.DataFrame(rows).sort_values(by="mean_comments", ascending=False).reset_index(drop=True)

    def get_comments_by_state(self) -> pd.DataFrame:
        """Compare comment metrics across open and closed issue lifecycles."""
        if "state" not in self.df.columns:
            return pd.DataFrame()

        rows = []
        for state, group in self.df.groupby("state"):
            comments = pd.to_numeric(group["comments_count"], errors="coerce").dropna()
            total_issues = len(group)
            zero_comments = int((comments == 0).sum())
            zero_rate = round(zero_comments / total_issues * 100, 2) if total_issues > 0 else 0.0

            rows.append({
                "state": state,
                "issue_count": total_issues,
                "total_comments": int(comments.sum()),
                "mean_comments": round(float(comments.mean()), 2) if not comments.empty else 0.0,
                "median_comments": round(float(comments.median()), 2) if not comments.empty else 0.0,
                "std_comments": round(float(comments.std()), 2) if len(comments) > 1 else 0.0,
                "p75_comments": round(float(comments.quantile(0.75)), 2) if not comments.empty else 0.0,
                "max_comments": int(comments.max()) if not comments.empty else 0,
                "zero_comment_count": zero_comments,
                "zero_comment_rate_pct": zero_rate,
            })

        return pd.DataFrame(rows).sort_values(by="issue_count", ascending=False).reset_index(drop=True)

    def get_comments_by_label(self) -> pd.DataFrame:
        """Compare comment metrics across canonical classification labels."""
        label_col = "canonical_label" if "canonical_label" in self.df.columns else "primary_label"
        if label_col not in self.df.columns:
            return pd.DataFrame()

        rows = []
        for label, group in self.df.groupby(label_col):
            comments = pd.to_numeric(group["comments_count"], errors="coerce").dropna()
            total_issues = len(group)
            zero_comments = int((comments == 0).sum())
            zero_rate = round(zero_comments / total_issues * 100, 2) if total_issues > 0 else 0.0

            rows.append({
                "label": label,
                "issue_count": total_issues,
                "total_comments": int(comments.sum()),
                "mean_comments": round(float(comments.mean()), 2) if not comments.empty else 0.0,
                "median_comments": round(float(comments.median()), 2) if not comments.empty else 0.0,
                "std_comments": round(float(comments.std()), 2) if len(comments) > 1 else 0.0,
                "p75_comments": round(float(comments.quantile(0.75)), 2) if not comments.empty else 0.0,
                "max_comments": int(comments.max()) if not comments.empty else 0,
                "zero_comment_count": zero_comments,
                "zero_comment_rate_pct": zero_rate,
            })

        return pd.DataFrame(rows).sort_values(by="mean_comments", ascending=False).reset_index(drop=True)

    def get_comments_vs_resolution(self) -> pd.DataFrame:
        """Analyze resolution time (in days) segmented by comment volume buckets for closed issues."""
        if "resolution_time_days" not in self.df.columns or "state" not in self.df.columns:
            return pd.DataFrame()

        closed = self.df[self.df["state"] == "closed"].copy()
        if closed.empty:
            return pd.DataFrame()

        closed["bucket"] = pd.cut(
            closed["comments_count"],
            bins=self.COMMENT_BUCKET_BINS,
            labels=self.COMMENT_BUCKET_LABELS,
            right=True,
        )

        rows = []
        for bucket in self.COMMENT_BUCKET_LABELS:
            group = closed[closed["bucket"] == bucket]
            res_days = pd.to_numeric(group["resolution_time_days"], errors="coerce").dropna()
            count = len(res_days)

            if count > 0:
                mean_res = round(float(res_days.mean()), 2)
                median_res = round(float(res_days.median()), 2)
                p25_res = round(float(res_days.quantile(0.25)), 2)
                p75_res = round(float(res_days.quantile(0.75)), 2)
                iqr_res = round(p75_res - p25_res, 2)
                max_res = round(float(res_days.max()), 2)
            else:
                mean_res = median_res = p25_res = p75_res = iqr_res = max_res = 0.0

            rows.append({
                "bucket": bucket,
                "closed_issue_count": count,
                "mean_resolution_days": mean_res,
                "median_resolution_days": median_res,
                "p25_resolution_days": p25_res,
                "p75_resolution_days": p75_res,
                "iqr_resolution_days": iqr_res,
                "max_resolution_days": max_res,
            })

        return pd.DataFrame(rows)

    def get_reaction_dynamics(self) -> Dict[str, Any]:
        """Compute summary statistics for issue reactions and engagement ratios."""
        reactions_stats = self._compute_numeric_stats(
            self.df["reactions_count"] if "reactions_count" in self.df.columns else pd.Series(dtype=float)
        )
        interaction_stats = self._compute_numeric_stats(
            self.df["interaction_total"] if "interaction_total" in self.df.columns else pd.Series(dtype=float)
        )
        ratio_stats = self._compute_numeric_stats(
            self.df["reactions_per_comment"] if "reactions_per_comment" in self.df.columns else pd.Series(dtype=float)
        )

        zero_reactions = 0
        zero_reaction_rate = 0.0
        if "reactions_count" in self.df.columns:
            r = pd.to_numeric(self.df["reactions_count"], errors="coerce").fillna(0)
            zero_reactions = int((r == 0).sum())
            zero_reaction_rate = round(zero_reactions / len(self.df) * 100, 2) if len(self.df) > 0 else 0.0

        return {
            "reactions_count": reactions_stats,
            "interaction_total": interaction_stats,
            "reactions_per_comment": ratio_stats,
            "zero_reaction_count": zero_reactions,
            "zero_reaction_rate_pct": zero_reaction_rate,
        }

    def get_sampled_comments_profile(self) -> Dict[str, Any]:
        """Profile sampled raw comment JSON payloads if directory is provided and contains files."""
        if not self.comments_dir or not self.comments_dir.exists():
            return {
                "available": False,
                "total_files": 0,
                "total_comments": 0,
                "author_associations": {},
                "avg_char_length": 0.0,
                "avg_word_length": 0.0,
                "status": "Comments directory not found or not specified.",
            }

        files = list(self.comments_dir.glob("*.json"))
        if not files:
            return {
                "available": False,
                "total_files": 0,
                "total_comments": 0,
                "author_associations": {},
                "avg_char_length": 0.0,
                "avg_word_length": 0.0,
                "status": "No JSON comment files in comments directory.",
            }

        parsed_comments: List[Dict[str, Any]] = []
        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        parsed_comments.extend(data)
                    elif isinstance(data, dict):
                        parsed_comments.append(data)
            except Exception:
                continue

        total_comments = len(parsed_comments)
        if total_comments == 0:
            return {
                "available": True,
                "total_files": len(files),
                "total_comments": 0,
                "author_associations": {},
                "avg_char_length": 0.0,
                "avg_word_length": 0.0,
                "status": "Parsed 0 valid comment items from files.",
            }

        associations: Dict[str, int] = {}
        char_lengths: List[int] = []
        word_lengths: List[int] = []

        for c in parsed_comments:
            role = c.get("author_association", "NONE")
            associations[role] = associations.get(role, 0) + 1

            body = c.get("body") or ""
            char_lengths.append(len(body))
            word_lengths.append(len(body.split()))

        avg_char = round(float(np.mean(char_lengths)), 2) if char_lengths else 0.0
        avg_word = round(float(np.mean(word_lengths)), 2) if word_lengths else 0.0

        return {
            "available": True,
            "total_files": len(files),
            "total_comments": total_comments,
            "author_associations": dict(sorted(associations.items(), key=lambda item: item[1], reverse=True)),
            "avg_char_length": avg_char,
            "avg_word_length": avg_word,
            "status": "Successfully profiled raw comment samples.",
        }

    def compute_full_comment_analysis(self) -> Dict[str, Any]:
        """Aggregate all comment analysis dimensions into a consolidated dictionary."""
        return {
            "comment_volume_distribution": self.get_comment_volume_distribution(),
            "comment_buckets": self.get_comment_buckets().to_dict(orient="records"),
            "comments_by_repository": self.get_comments_by_repository().to_dict(orient="records"),
            "comments_by_state": self.get_comments_by_state().to_dict(orient="records"),
            "comments_by_label": self.get_comments_by_label().to_dict(orient="records"),
            "comments_vs_resolution": self.get_comments_vs_resolution().to_dict(orient="records"),
            "reaction_dynamics": self.get_reaction_dynamics(),
            "sampled_comments_profile": self.get_sampled_comments_profile(),
        }

    def generate_summary_report(self) -> str:
        """Generate a concise, formatted text summary of the comment analysis findings."""
        dist = self.get_comment_volume_distribution()
        buckets_df = self.get_comment_buckets()
        repos_df = self.get_comments_by_repository()
        states_df = self.get_comments_by_state()
        labels_df = self.get_comments_by_label()
        res_df = self.get_comments_vs_resolution()
        reactions = self.get_reaction_dynamics()
        raw_profile = self.get_sampled_comments_profile()

        lines = [
            "# OpenSourceLens — Comment Analysis Summary",
            "",
            "## 1. Overall Comment Distribution",
            f"- **Total Issues Analyzed:** {dist['count']:,}",
            f"- **Mean Comments / Issue:** {dist['mean']} (Std: {dist['std']})",
            f"- **Median Comments:** {dist['median']} (IQR: {dist['iqr']}, Skewness: {dist['skewness']})",
            f"- **Percentiles:** p10={dist['p10']}, p25={dist['p25']}, p50={dist['median']}, p75={dist['p75']}, p90={dist['p90']}, p95={dist['p95']}, p99={dist['p99']}, Max={dist['max']}",
            "",
            "## 2. Comment Volume Buckets",
            "| Bucket | Issue Count | Percentage | Cumulative % |",
            "|---|---|---|---|",
        ]
        for _, r in buckets_df.iterrows():
            lines.append(f"| {r['bucket']} | {r['issue_count']:,} | {r['percentage']:.2f}% | {r['cumulative_percentage']:.2f}% |")

        lines.extend([
            "",
            "## 3. Comments by Repository",
            "| Repository | Issues | Mean Comments | Median | p75 | Max | Zero-Comment % |",
            "|---|---|---|---|---|---|---|",
        ])
        for _, r in repos_df.iterrows():
            lines.append(f"| `{r['repo_name']}` | {r['issue_count']:,} | {r['mean_comments']:.2f} | {r['median_comments']:.1f} | {r['p75_comments']:.1f} | {r['max_comments']} | {r['zero_comment_rate_pct']:.2f}% |")

        lines.extend([
            "",
            "## 4. Comments by State & Canonical Label",
            "### State Comparison",
            "| State | Issues | Mean Comments | Median | Zero-Comment % |",
            "|---|---|---|---|---|",
        ])
        for _, r in states_df.iterrows():
            lines.append(f"| {r['state']} | {r['issue_count']:,} | {r['mean_comments']:.2f} | {r['median_comments']:.1f} | {r['zero_comment_rate_pct']:.2f}% |")

        lines.extend([
            "",
            "### Label Comparison",
            "| Label | Issues | Mean Comments | Median | Zero-Comment % |",
            "|---|---|---|---|---|",
        ])
        for _, r in labels_df.iterrows():
            lines.append(f"| `{r['label']}` | {r['issue_count']:,} | {r['mean_comments']:.2f} | {r['median_comments']:.1f} | {r['zero_comment_rate_pct']:.2f}% |")

        lines.extend([
            "",
            "## 5. Comment Count vs. Resolution Time (Closed Issues)",
            "| Comment Bucket | Closed Issues | Mean Days | Median Days | p75 Days |",
            "|---|---|---|---|---|",
        ])
        for _, r in res_df.iterrows():
            lines.append(f"| {r['bucket']} | {r['closed_issue_count']:,} | {r['mean_resolution_days']:.2f} | {r['median_resolution_days']:.2f} | {r['p75_resolution_days']:.2f} |")

        lines.extend([
            "",
            "## 6. Reactions & Raw Comment Profiling",
            f"- **Zero-Reaction Issues:** {reactions['zero_reaction_count']:,} ({reactions['zero_reaction_rate_pct']}%)",
            f"- **Max Reactions on an Issue:** {reactions['reactions_count']['max']}",
            f"- **Raw Comment Samples Available:** {raw_profile['available']} ({raw_profile['total_comments']} parsed comments from {raw_profile['total_files']} files)",
            f"- **Author Associations in Sample:** {raw_profile['author_associations']}",
        ])

        return "\n".join(lines)
