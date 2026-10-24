"""Issue Correlation and Multicollinearity Analysis Module for OpenSourceLens EDA.

Provides comprehensive analytical engines for bivariate linear (Pearson) and monotonic rank
(Spearman) correlation matrices, target-specific feature associations (resolution time,
classification proxies), top correlated feature pair extractions, and multicollinearity
diagnostics to guide Phase 4 baseline modeling.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import pandas as pd
import numpy as np


class IssueCorrelationAnalyzer:
    """Analyzer for evaluating linear/rank correlations and detecting multicollinearity."""

    DEFAULT_NUMERICAL_FEATURES: List[str] = [
        "clean_title_word_count",
        "clean_body_word_count",
        "comments_count",
        "reactions_count",
        "interaction_total",
        "reactions_per_comment",
        "labels_count",
        "code_block_count",
        "url_count",
        "uppercase_ratio_title",
        "body_to_title_word_ratio",
        "avg_word_length_title",
        "avg_word_length_body",
        "has_code_block",
        "has_stack_trace",
        "has_url",
        "title_has_bracket_tag",
        "is_weekend",
        "is_internal_author",
        "resolution_time_days",
    ]

    def __init__(
        self,
        data: Union[pd.DataFrame, str, Path] = "data/processed/issues_master.csv",
        default_features: Optional[List[str]] = None,
    ):
        """Initialize correlation analyzer with DataFrame or path to CSV/Parquet file.

        Args:
            data: Loaded pd.DataFrame or path to processed dataset file.
            default_features: Optional custom list of feature names to analyze by default.

        Raises:
            FileNotFoundError: If input file path does not exist.
            ValueError: If dataset is empty.
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

        if default_features is not None:
            self.configured_features = default_features
        else:
            id_cols = {"issue_id", "issue_number", "id"}
            self.configured_features = [
                c for c in self.df.columns if c not in id_cols and pd.api.types.is_numeric_dtype(self.df[c])
            ]

        # Convert boolean columns to float (1.0/0.0) for numerical correlation
        for col in self.df.columns:
            if self.df[col].dtype == bool:
                self.df[col] = self.df[col].astype(float)
            elif self.df[col].dtype == object:
                # Handle possible string booleans from CSV
                val_sample = self.df[col].dropna().iloc[0] if not self.df[col].dropna().empty else None
                if val_sample in [True, False, "True", "False"]:
                    self.df[col] = self.df[col].map({"True": 1.0, "False": 0.0, True: 1.0, False: 0.0})

    def get_available_features(self, features: Optional[List[str]] = None) -> List[str]:
        """Return list of candidate features present in the dataset with non-zero variance.

        Args:
            features: Optional list of features to filter. Defaults to configured features.

        Returns:
            List of valid, non-constant numerical/boolean column names.
        """
        candidates = features if features is not None else self.configured_features
        available: List[str] = []

        for col in candidates:
            if col not in self.df.columns:
                continue
            numeric_col = pd.to_numeric(self.df[col], errors="coerce").dropna()
            # Must have at least 2 observations and non-zero standard deviation
            if len(numeric_col) > 1 and numeric_col.std() > 0:
                available.append(col)

        return available

    def get_correlation_matrix(
        self,
        method: str = "pearson",
        features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compute square correlation matrix for selected features.

        Args:
            method: Correlation method: 'pearson', 'spearman', or 'kendall'.
            features: Optional custom feature subset.

        Returns:
            Square symmetric correlation DataFrame.
        """
        valid_methods = ["pearson", "spearman", "kendall"]
        method_lower = method.lower()
        if method_lower not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Expected one of {valid_methods}.")

        avail_cols = self.get_available_features(features)
        if not avail_cols:
            return pd.DataFrame()

        num_df = self.df[avail_cols].apply(pd.to_numeric, errors="coerce")
        if method_lower == "spearman":
            corr_matrix = num_df.rank().corr(method="pearson").round(4)
        else:
            corr_matrix = num_df.corr(method=method_lower).round(4)
        return corr_matrix

    def get_target_correlations(
        self,
        target: str = "resolution_time_days",
        subset: Optional[str] = None,
        features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compute Pearson and Spearman correlations of all features against a target feature.

        Args:
            target: Name of the target column.
            subset: Optional subset filter, e.g. 'closed' for resolution time analysis.
            features: Optional custom candidate feature subset.

        Returns:
            DataFrame sorted by absolute Spearman correlation descending.
        """
        if target not in self.df.columns:
            return pd.DataFrame(columns=[
                "feature", "pearson_corr", "spearman_corr", "abs_pearson", "abs_spearman", "diff_spearman_pearson"
            ])

        data_subset = self.df
        if subset == "closed" and "state" in self.df.columns:
            data_subset = self.df[self.df["state"] == "closed"]

        target_series = pd.to_numeric(data_subset[target], errors="coerce")
        if target_series.dropna().empty or target_series.dropna().std() == 0:
            return pd.DataFrame(columns=[
                "feature", "pearson_corr", "spearman_corr", "abs_pearson", "abs_spearman", "diff_spearman_pearson"
            ])

        avail_cols = [c for c in self.get_available_features(features) if c != target]
        rows = []

        for col in avail_cols:
            col_series = pd.to_numeric(data_subset[col], errors="coerce")
            valid_mask = target_series.notna() & col_series.notna()
            t_clean = target_series[valid_mask]
            c_clean = col_series[valid_mask]

            if len(t_clean) < 3 or c_clean.std() == 0:
                continue

            pearson = float(c_clean.corr(t_clean, method="pearson"))
            spearman = float(c_clean.rank().corr(t_clean.rank(), method="pearson"))

            if np.isnan(pearson):
                pearson = 0.0
            if np.isnan(spearman):
                spearman = 0.0

            rows.append({
                "feature": col,
                "pearson_corr": round(pearson, 4),
                "spearman_corr": round(spearman, 4),
                "abs_pearson": round(abs(pearson), 4),
                "abs_spearman": round(abs(spearman), 4),
                "diff_spearman_pearson": round(spearman - pearson, 4),
            })

        if not rows:
            return pd.DataFrame(columns=[
                "feature", "pearson_corr", "spearman_corr", "abs_pearson", "abs_spearman", "diff_spearman_pearson"
            ])

        return pd.DataFrame(rows).sort_values(by="abs_spearman", ascending=False).reset_index(drop=True)

    def get_top_correlated_pairs(
        self,
        threshold: float = 0.50,
        method: str = "pearson",
        features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Extract unique off-diagonal feature pairs exceeding the absolute correlation threshold.

        Args:
            threshold: Minimum absolute correlation magnitude (0.0 to 1.0).
            method: 'pearson' or 'spearman'.
            features: Optional custom feature subset.

        Returns:
            DataFrame of top correlated pairs sorted by absolute correlation descending.
        """
        corr_matrix = self.get_correlation_matrix(method=method, features=features)
        if corr_matrix.empty:
            return pd.DataFrame(columns=[
                "feature_1", "feature_2", "correlation", "abs_correlation", "relationship"
            ])

        cols = corr_matrix.columns.tolist()
        pairs = []

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                col1, col2 = cols[i], cols[j]
                val = corr_matrix.loc[col1, col2]
                if pd.notna(val) and abs(val) >= threshold:
                    pairs.append({
                        "feature_1": col1,
                        "feature_2": col2,
                        "correlation": round(float(val), 4),
                        "abs_correlation": round(abs(float(val)), 4),
                        "relationship": "Positive" if val > 0 else "Negative",
                    })

        if not pairs:
            return pd.DataFrame(columns=[
                "feature_1", "feature_2", "correlation", "abs_correlation", "relationship"
            ])

        return pd.DataFrame(pairs).sort_values(by="abs_correlation", ascending=False).reset_index(drop=True)

    def get_multicollinearity_report(
        self,
        threshold: float = 0.70,
        features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Identify high-collinearity feature pairs that could induce variance inflation in ML models.

        Args:
            threshold: Absolute correlation cutoff for flagging severe collinearity (default: 0.70).
            features: Optional custom feature subset.

        Returns:
            Dictionary detailing high-risk pairs, affected features, and modeling recommendations.
        """
        pairs_df = self.get_top_correlated_pairs(threshold=threshold, method="pearson", features=features)
        spearman_pairs_df = self.get_top_correlated_pairs(threshold=threshold, method="spearman", features=features)

        affected_features = set()
        diagnostics = []

        for _, row in pairs_df.iterrows():
            f1, f2 = row["feature_1"], row["feature_2"]
            affected_features.add(f1)
            affected_features.add(f2)

            diag = {
                "feature_1": f1,
                "feature_2": f2,
                "pearson_corr": row["correlation"],
                "risk_level": "Severe Multicollinearity" if row["abs_correlation"] >= 0.85 else "High Multicollinearity",
                "recommendation": f"Consider dropping '{f2}' or applying Ridge regularization in Phase 4 linear models.",
            }
            diagnostics.append(diag)

        return {
            "threshold": threshold,
            "total_high_risk_pairs": len(diagnostics),
            "affected_features_count": len(affected_features),
            "affected_features": sorted(list(affected_features)),
            "high_risk_pairs": diagnostics,
            "rank_collinear_pairs_count": len(spearman_pairs_df),
        }

    def get_correlation_summary(self) -> Dict[str, Any]:
        """Compute high-level statistical summary across all analyzed features."""
        corr_matrix = self.get_correlation_matrix(method="pearson")
        if corr_matrix.empty:
            return {"features_analyzed": 0}

        top_pairs = self.get_top_correlated_pairs(threshold=0.0, method="pearson")
        res_corr = self.get_target_correlations(target="resolution_time_days", subset="closed")
        comments_corr = self.get_target_correlations(target="comments_count")

        highest_pos = top_pairs[top_pairs["correlation"] > 0].iloc[0].to_dict() if not top_pairs[top_pairs["correlation"] > 0].empty else {}
        highest_neg = top_pairs[top_pairs["correlation"] < 0].iloc[-1].to_dict() if not top_pairs[top_pairs["correlation"] < 0].empty else {}

        mean_abs_corr = round(float(top_pairs["abs_correlation"].mean()), 4) if not top_pairs.empty else 0.0

        top_res_features = res_corr.head(3)[["feature", "spearman_corr", "pearson_corr"]].to_dict(orient="records") if not res_corr.empty else []
        top_comm_features = comments_corr.head(3)[["feature", "spearman_corr", "pearson_corr"]].to_dict(orient="records") if not comments_corr.empty else []

        return {
            "features_analyzed": len(corr_matrix.columns),
            "total_feature_pairs": len(top_pairs),
            "mean_abs_correlation": mean_abs_corr,
            "highest_positive_pair": highest_pos,
            "highest_negative_pair": highest_neg,
            "top_resolution_predictors": top_res_features,
            "top_discussion_predictors": top_comm_features,
        }

    def compute_full_correlation_analysis(self) -> Dict[str, Any]:
        """Consolidate correlation matrices, target correlations, and collinearity alerts."""
        return {
            "correlation_summary": self.get_correlation_summary(),
            "target_correlations_resolution": self.get_target_correlations(target="resolution_time_days", subset="closed").to_dict(orient="records"),
            "target_correlations_comments": self.get_target_correlations(target="comments_count").to_dict(orient="records"),
            "top_correlated_pairs_pearson": self.get_top_correlated_pairs(threshold=0.40, method="pearson").to_dict(orient="records"),
            "top_correlated_pairs_spearman": self.get_top_correlated_pairs(threshold=0.40, method="spearman").to_dict(orient="records"),
            "multicollinearity_report": self.get_multicollinearity_report(threshold=0.70),
        }

    def generate_summary_report(self) -> str:
        """Generate a concise, formatted text summary of the correlation findings."""
        summary = self.get_correlation_summary()
        collinear = self.get_multicollinearity_report(threshold=0.70)
        res_df = self.get_target_correlations(target="resolution_time_days", subset="closed")
        top_pairs_df = self.get_top_correlated_pairs(threshold=0.40, method="pearson")

        lines = [
            "# OpenSourceLens — Correlation & Multicollinearity Analysis Summary",
            "",
            "## 1. Overview & Inter-Feature Dynamics",
            f"- **Features Analyzed:** {summary['features_analyzed']}",
            f"- **Total Feature Pairs Evaluated:** {summary['total_feature_pairs']:,}",
            f"- **Mean Absolute Correlation Across Features:** {summary['mean_abs_correlation']:.4f}",
        ]

        if summary.get("highest_positive_pair"):
            hp = summary["highest_positive_pair"]
            lines.append(f"- **Strongest Positive Pair:** `{hp.get('feature_1')}` & `{hp.get('feature_2')}` ($r = {hp.get('correlation')}$)")

        if summary.get("highest_negative_pair"):
            hn = summary["highest_negative_pair"]
            lines.append(f"- **Strongest Negative Pair:** `{hn.get('feature_1')}` & `{hn.get('feature_2')}` ($r = {hn.get('correlation')}$)")

        lines.extend([
            "",
            "## 2. Target Associations: Features vs. Resolution Time (Closed Issues)",
            "| Feature | Spearman Rank (ρ) | Pearson Linear (r) | Rank vs. Linear Diff |",
            "|---|---|---|---|",
        ])
        for _, r in res_df.head(10).iterrows():
            lines.append(f"| `{r['feature']}` | {r['spearman_corr']:+.4f} | {r['pearson_corr']:+.4f} | {r['diff_spearman_pearson']:+.4f} |")

        lines.extend([
            "",
            "## 3. High Correlation Feature Pairs (|r| >= 0.40)",
            "| Feature 1 | Feature 2 | Pearson (r) | Relationship |",
            "|---|---|---|---|",
        ])
        for _, r in top_pairs_df.head(10).iterrows():
            lines.append(f"| `{r['feature_1']}` | `{r['feature_2']}` | {r['correlation']:+.4f} | {r['relationship']} |")

        lines.extend([
            "",
            "## 4. Multicollinearity & ML Modeling Diagnostics",
            f"- **Collinearity Threshold:** |r| >= {collinear['threshold']}",
            f"- **Total High-Risk Pairs Flagged:** {collinear['total_high_risk_pairs']}",
            f"- **Features Involved:** {', '.join(f'`{f}`' for f in collinear['affected_features'])}",
            "",
            "### Recommended Actions for Phase 4 Baseline Modeling:",
        ])
        for diag in collinear["high_risk_pairs"]:
            lines.append(f"- `{diag['feature_1']}` & `{diag['feature_2']}` ($r = {diag['pearson_corr']:+.4f}$): {diag['recommendation']}")

        return "\n".join(lines)
