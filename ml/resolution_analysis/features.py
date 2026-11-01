"""
OpenSourceLens — Resolution Feature Engineering & Leakage Audit (Phase 8, Step 8.3).

Identifies and ranks predictive features available at issue creation time (t=0),
enforces strict data leakage boundaries to prevent post-creation contamination,
computes linear and non-linear feature rankings, and produces standardized feature matrices.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Canonical classification of eligible creation-time features vs. leaked features
ELIGIBLE_NUMERIC_FEATURES: List[str] = [
    "clean_title_word_count",
    "clean_body_word_count",
    "body_to_title_word_ratio",
    "avg_word_length_title",
    "avg_word_length_body",
    "has_code_block",
    "code_block_count",
    "has_stack_trace",
    "has_url",
    "url_count",
    "title_has_bracket_tag",
    "uppercase_ratio_title",
    "question_mark_count",
    "exclamation_mark_count",
    "is_internal_author",
    "is_bot_author",
    "created_hour",
    "created_day_of_week",
    "is_weekend",
    "labels_count",
]

ELIGIBLE_CATEGORICAL_FEATURES: List[str] = [
    "repo_name",
    "canonical_label",
    "author_association",
]

# Features that accumulate post-creation or directly define the target duration
LEAKED_FEATURES: List[str] = [
    "comments_count",
    "reactions_count",
    "interaction_total",
    "reactions_per_comment",
    "updated_at",
    "closed_at",
    "state",
    "state_reason",
    "speed_tier",
    "is_immediate_closure",
    "resolution_type",
    "is_censored",
    "event_observed",
    "duration_at_risk_days",
    "resolution_time_seconds",
    "resolution_time_days",
    "resolution_time_hours",
    "issue_age_days",
]

TARGET_COLUMN: str = "resolution_time_log_hours"


class ResolutionFeatureEngineer:
    """Extracts, audits, ranks, and transforms issue features for resolution prediction."""

    def __init__(
        self,
        data: Union[pd.DataFrame, str, Path] = "data/processed/issues_resolution.parquet",
    ):
        """Initialize feature engineer.

        Args:
            data: Loaded DataFrame or path to processed Parquet/CSV file.
        """
        if isinstance(data, (str, Path)):
            p = Path(data)
            if not p.exists():
                raise FileNotFoundError(f"Dataset not found at: {p}")
            if p.suffix == ".parquet":
                self.df = pd.read_parquet(p)
            elif p.suffix == ".csv":
                self.df = pd.read_csv(p)
            else:
                raise ValueError(f"Unsupported format: {p.suffix}")
            self.source_path = str(p)
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
            self.source_path = "DataFrame in-memory"
        else:
            raise TypeError(f"Invalid data type: {type(data)}")

        if self.df.empty:
            raise ValueError("Dataset is empty.")

        # Modeling cohort: closed issues with non-null target duration
        if "is_censored" in self.df.columns:
            self.modeling_df = self.df[self.df["is_censored"] == False].copy()
        elif "state" in self.df.columns:
            self.modeling_df = self.df[self.df["state"] == "closed"].copy()
        else:
            self.modeling_df = self.df.copy()

        # Ensure target column exists
        if TARGET_COLUMN not in self.modeling_df.columns:
            if "resolution_time_hours" in self.modeling_df.columns:
                self.modeling_df[TARGET_COLUMN] = np.log(self.modeling_df["resolution_time_hours"] + 1.0)
            else:
                raise KeyError(f"Missing required target column '{TARGET_COLUMN}'")

    @staticmethod
    def audit_leakage(df: pd.DataFrame) -> Dict[str, Any]:
        """Perform data leakage audit on input DataFrame columns.

        Returns:
            Dictionary identifying eligible creation features, excluded leaked features,
            and leakage compliance boolean.
        """
        cols = set(df.columns)
        present_eligible_num = [c for c in ELIGIBLE_NUMERIC_FEATURES if c in cols]
        present_eligible_cat = [c for c in ELIGIBLE_CATEGORICAL_FEATURES if c in cols]
        present_leaked = [c for c in LEAKED_FEATURES if c in cols]

        return {
            "eligible_numeric_features": present_eligible_num,
            "eligible_categorical_features": present_eligible_cat,
            "total_eligible_features": len(present_eligible_num) + len(present_eligible_cat),
            "excluded_leaked_features": present_leaked,
            "total_leaked_features": len(present_leaked),
            "leakage_boundary_enforced": True,
        }

    def compute_numeric_rankings(self) -> pd.DataFrame:
        """Compute Pearson r and Spearman rho correlations for numeric creation features."""
        target = self.modeling_df[TARGET_COLUMN].values
        records = []

        cols = [c for c in ELIGIBLE_NUMERIC_FEATURES if c in self.modeling_df.columns]
        for col in cols:
            series = pd.to_numeric(self.modeling_df[col], errors="coerce").fillna(0.0).values
            if len(series) < 2 or np.std(series) < 1e-12:
                p_corr, p_val = 0.0, 1.0
                s_corr, s_val = 0.0, 1.0
            else:
                p_corr, p_val = stats.pearsonr(series, target)
                s_corr, s_val = stats.spearmanr(series, target)

            records.append({
                "feature": col,
                "feature_type": "numeric",
                "pearson_r": round(float(p_corr), 4),
                "pearson_p_value": float(p_val),
                "spearman_rho": round(float(s_corr), 4),
                "spearman_p_value": float(s_val),
                "abs_spearman": round(abs(float(s_corr)), 4),
            })

        return pd.DataFrame(records).sort_values("abs_spearman", ascending=False).reset_index(drop=True)

    def compute_all_feature_rankings(self, random_state: int = 42) -> pd.DataFrame:
        """Compute comprehensive ranking across all eligible features using ANOVA and Mutual Information.

        Args:
            random_state: Seed for Mutual Information estimation.

        Returns:
            DataFrame with feature rankings sorted by Mutual Information.
        """
        # Prepare numeric and one-hot categorical data
        num_cols = [c for c in ELIGIBLE_NUMERIC_FEATURES if c in self.modeling_df.columns]
        cat_cols = [c for c in ELIGIBLE_CATEGORICAL_FEATURES if c in self.modeling_df.columns]

        X_num = self.modeling_df[num_cols].fillna(0.0).astype(float)

        # One-hot encode categoricals
        enc = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        encoded_cat = enc.fit_transform(self.modeling_df[cat_cols].fillna("unknown").astype(str))
        cat_feature_names = enc.get_feature_names_out(cat_cols).tolist()
        X_cat = pd.DataFrame(encoded_cat, columns=cat_feature_names, index=self.modeling_df.index)

        # Combined feature matrix
        X_all = pd.concat([X_num, X_cat], axis=1)
        y = self.modeling_df[TARGET_COLUMN].values

        # 1. ANOVA F-statistic
        f_stats, f_pvals = f_regression(X_all, y)

        # 2. Mutual Information
        mi_scores = mutual_info_regression(X_all, y, random_state=random_state)

        # 3. Correlations (Pearson and Spearman)
        records = []
        for idx, feat in enumerate(X_all.columns):
            s_val = X_all[feat].values
            if len(s_val) < 2 or np.std(s_val) < 1e-12:
                p_corr, s_corr = 0.0, 0.0
            else:
                p_corr, _ = stats.pearsonr(s_val, y)
                s_corr, _ = stats.spearmanr(s_val, y)
            is_cat = feat in cat_feature_names

            records.append({
                "feature": feat,
                "feature_type": "categorical_dummy" if is_cat else "numeric",
                "parent_group": feat.split("_")[0] if is_cat else "numeric_marker",
                "mutual_info": round(float(mi_scores[idx]), 4),
                "f_statistic": round(float(f_stats[idx]), 2),
                "f_p_value": float(f_pvals[idx]),
                "pearson_r": round(float(p_corr), 4),
                "spearman_rho": round(float(s_corr), 4),
            })

        df_rank = pd.DataFrame(records).sort_values("mutual_info", ascending=False).reset_index(drop=True)
        return df_rank

    def build_feature_matrix(
        self,
        selected_features: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Transform raw data into a normalized, leakage-free modeling matrix X and target y.

        Standardizes numeric features to mean=0, std=1; one-hot encodes categoricals.

        Args:
            selected_features: Optional subset of feature names to retain.

        Returns:
            Tuple of (X_transformed_df, y_target_series).
        """
        num_cols = [c for c in ELIGIBLE_NUMERIC_FEATURES if c in self.modeling_df.columns]
        cat_cols = [c for c in ELIGIBLE_CATEGORICAL_FEATURES if c in self.modeling_df.columns]

        # Standardize numeric columns
        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(self.modeling_df[num_cols].fillna(0.0).astype(float))
        df_num = pd.DataFrame(X_num_scaled, columns=num_cols, index=self.modeling_df.index)

        # One-hot encode categorical columns
        enc = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        encoded_cat = enc.fit_transform(self.modeling_df[cat_cols].fillna("unknown").astype(str))
        cat_names = enc.get_feature_names_out(cat_cols).tolist()
        df_cat = pd.DataFrame(encoded_cat, columns=cat_names, index=self.modeling_df.index)

        # Combined X
        X = pd.concat([df_num, df_cat], axis=1)

        # Add issue identification metadata for tracking
        if "issue_id" in self.modeling_df.columns:
            X.insert(0, "issue_id", self.modeling_df["issue_id"].values)
        if "issue_number" in self.modeling_df.columns:
            X.insert(1, "issue_number", self.modeling_df["issue_number"].values)

        if selected_features:
            keep_cols = [c for c in selected_features if c in X.columns]
            if "issue_id" in X.columns and "issue_id" not in keep_cols:
                keep_cols = ["issue_id"] + keep_cols
            if "issue_number" in X.columns and "issue_number" not in keep_cols:
                keep_cols = ["issue_number"] + keep_cols
            X = X[keep_cols]

        y = self.modeling_df[TARGET_COLUMN].copy()
        return X, y

    def check_multicollinearity(
        self,
        top_n: int = 12,
    ) -> pd.DataFrame:
        """Compute pairwise correlation matrix across top numeric features to audit collinearity."""
        ranks = self.compute_numeric_rankings()
        top_numeric = ranks["feature"].head(top_n).tolist()

        sub_df = self.modeling_df[top_numeric].fillna(0.0).astype(float)
        corr_matrix = sub_df.corr(method="spearman").round(3)
        return corr_matrix

    def generate_fig30_visualization(
        self,
        output_path: Union[str, Path] = "ml/resolution_analysis/figures/fig30_feature_correlations_and_importance.png",
    ) -> Path:
        """Generate publication Figure 30: Feature Correlations, Mutual Information, and Collinearity.

        Panel A: Mutual Information Ranking Bar Chart (Top 12 Features).
        Panel B: Spearman Rank Correlation Bar Chart (Eligible Numeric Features).
        Panel C: Pairwise Correlation Heatmap of Top Features.

        Args:
            output_path: Output PNG filepath.

        Returns:
            Resolved Path to saved figure.
        """
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(15, 11), dpi=300)
        gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

        ax_a = fig.add_subplot(gs[0, :])  # Panel A spans top row
        ax_b = fig.add_subplot(gs[1, 0])  # Panel B bottom left
        ax_c = fig.add_subplot(gs[1, 1])  # Panel C bottom right

        # Style colors
        c_blue = "#2563eb"
        c_red = "#dc2626"
        c_emerald = "#059669"
        c_gray = "#64748b"

        # ---------------- Panel A: Top Mutual Information Rankings ----------------
        rankings = self.compute_all_feature_rankings(random_state=42).head(14)
        feat_labels = [
            f.replace("repo_name_", "repo: ").replace("canonical_label_", "category: ").replace("author_association_", "author: ")
            for f in rankings["feature"]
        ]
        mi_vals = rankings["mutual_info"].values

        y_pos = np.arange(len(feat_labels))
        bars_a = ax_a.barh(y_pos, mi_vals, align="center", color=c_blue, alpha=0.85, edgecolor="white", height=0.68)
        
        # Color top 3 drivers with accent
        for i in range(min(3, len(bars_a))):
            bars_a[i].set_facecolor(c_emerald)

        ax_a.set_yticks(y_pos)
        ax_a.set_yticklabels(feat_labels, fontsize=9.5)
        ax_a.invert_yaxis()  # Highest score at top
        ax_a.set_xlabel("Mutual Information Score: I(X; y)", fontsize=10.5, fontweight="bold")
        ax_a.set_title("(A) Top Creation-Time Features by Mutual Information with Log Resolution Duration", fontsize=11, fontweight="bold")
        ax_a.grid(True, linestyle="--", alpha=0.35, axis="x")

        # Value labels on bars
        for bar in bars_a:
            w = bar.get_width()
            ax_a.text(w + 0.003, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", fontsize=8.5, color="#1e293b", fontweight="bold")
        ax_a.set_xlim(0, max(mi_vals) * 1.15)

        # ---------------- Panel B: Spearman Rho for Numeric Markers ----------------
        num_ranks = self.compute_numeric_rankings().head(10)
        num_names = [f.replace("_", " ").title() for f in num_ranks["feature"]]
        rho_vals = num_ranks["spearman_rho"].values

        y_pos_b = np.arange(len(num_names))
        colors_b = [c_emerald if r > 0 else c_red for r in rho_vals]
        bars_b = ax_b.barh(y_pos_b, rho_vals, align="center", color=colors_b, alpha=0.85, edgecolor="white", height=0.65)

        ax_b.set_yticks(y_pos_b)
        ax_b.set_yticklabels(num_names, fontsize=9)
        ax_b.invert_yaxis()
        ax_b.axvline(0, color="#1e293b", lw=1.2, linestyle="-")
        ax_b.set_xlabel("Spearman Rank Correlation: \u03c1", fontsize=10)
        ax_b.set_title("(B) Monotonic Correlation of Numeric Markers", fontsize=10.5, fontweight="bold")
        ax_b.grid(True, linestyle="--", alpha=0.35, axis="x")

        # ---------------- Panel C: Pairwise Correlation Heatmap ----------------
        corr_matrix = self.check_multicollinearity(top_n=8)
        cols_short = [c.replace("_count", "").replace("clean_", "") for c in corr_matrix.columns]

        im = ax_c.imshow(corr_matrix.values, cmap="coolwarm", vmin=-0.5, vmax=1.0)
        ax_c.set_xticks(range(len(cols_short)))
        ax_c.set_yticks(range(len(cols_short)))
        ax_c.set_xticklabels(cols_short, rotation=45, ha="right", fontsize=8.5)
        ax_c.set_yticklabels(cols_short, fontsize=8.5)
        ax_c.set_title("(C) Multicollinearity Audit (Spearman \u03c1 Matrix)", fontsize=10.5, fontweight="bold")

        # Annotate correlation numbers
        for i in range(len(cols_short)):
            for j in range(len(cols_short)):
                val = corr_matrix.iloc[i, j]
                text_col = "white" if abs(val) > 0.55 else "black"
                ax_c.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_col, fontsize=7.5)

        cbar = fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

        # Figure Suptitle
        fig.suptitle(
            "Figure 30: OpenSourceLens Feature Importance, Non-Linear Mutual Information & Leakage Audit",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )

        plt.savefig(out_p, bbox_inches="tight")
        plt.close(fig)
        return out_p

    def export_artifacts(
        self,
        output_parquet: Union[str, Path] = "data/processed/resolution_features.parquet",
        manifest_path: Union[str, Path] = "data/processed/evaluation/resolution_feature_analysis.json",
    ) -> Tuple[Path, Path]:
        """Export standardized feature matrix and evaluation manifest.

        Args:
            output_parquet: Destination parquet path.
            manifest_path: Destination JSON manifest path.

        Returns:
            Tuple of (output_parquet_path, manifest_path).
        """
        p_parquet = Path(output_parquet)
        p_man = Path(manifest_path)
        p_parquet.parent.mkdir(parents=True, exist_ok=True)
        p_man.parent.mkdir(parents=True, exist_ok=True)

        # 1. Build and export feature matrix
        X, y = self.build_feature_matrix()
        feature_df = X.copy()
        feature_df[TARGET_COLUMN] = y.values
        feature_df.to_parquet(p_parquet, index=False)

        # 2. Compute rankings and audit
        leakage_audit = self.audit_leakage(self.df)
        all_ranks = self.compute_all_feature_rankings(random_state=42)
        numeric_ranks = self.compute_numeric_rankings()

        manifest_content = {
            "phase": "Phase 8: Resolution Analysis & Prediction",
            "step": "Step 8.3: Identify Useful Features",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_dataset": self.source_path,
            "target_variable": TARGET_COLUMN,
            "sample_counts": {
                "total_issues": len(self.df),
                "modeling_cohort_closed": len(self.modeling_df),
                "feature_matrix_shape": list(feature_df.shape),
            },
            "leakage_audit": leakage_audit,
            "top_features_by_mutual_info": all_ranks.head(15).to_dict(orient="records"),
            "numeric_feature_correlations": numeric_ranks.to_dict(orient="records"),
            "artifacts": {
                "feature_matrix_parquet": str(p_parquet),
                "feature_matrix_bytes": p_parquet.stat().st_size,
                "figure_artifact": "ml/resolution_analysis/figures/fig30_feature_correlations_and_importance.png",
            },
        }

        with open(p_man, "w", encoding="utf-8") as f:
            json.dump(manifest_content, f, indent=2)

        return p_parquet, p_man
