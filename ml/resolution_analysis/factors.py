"""
OpenSourceLens — Resolution Time Factor Investigation & Econometric Modeling (Phase 8, Step 8.4).

Performs multivariate OLS regression on log-transformed resolution duration,
evaluates Variance Inflation Factors (VIF) for multicollinearity diagnostics,
partitions explained variance into 5 distinct Factor Families via ANOVA,
evaluates repo-by-category interaction effects, and generates publication Figure 31.
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
from sklearn.linear_model import LinearRegression


FACTOR_FAMILIES: Dict[str, List[str]] = {
    "Repository Governance": [
        "repo_name_kubernetes/kubernetes",
        "repo_name_microsoft/vscode",
        "repo_name_pytorch/pytorch",
    ],
    "Issue Typology": [
        "canonical_label_documentation",
        "canonical_label_feature",
        "canonical_label_other",
        "canonical_label_question",
        "canonical_label_unlabeled",
    ],
    "Structural Complexity": [
        "clean_title_word_count",
        "clean_body_word_count",
        "body_to_title_word_ratio",
        "avg_word_length_title",
        "avg_word_length_body",
        "title_has_bracket_tag",
        "uppercase_ratio_title",
        "question_mark_count",
        "exclamation_mark_count",
    ],
    "Technical Evidence": [
        "has_code_block",
        "code_block_count",
        "has_stack_trace",
        "has_url",
        "url_count",
    ],
    "Temporal & Author Context": [
        "created_hour",
        "created_day_of_week",
        "is_weekend",
        "is_internal_author",
        "is_bot_author",
        "labels_count",
    ],
}


class ResolutionFactorAnalyzer:
    """
    Multivariate statistical regression and econometric factor analysis
    for issue resolution duration.
    """

    def __init__(
        self,
        features_path: Optional[Union[str, Path]] = None,
        resolution_path: Optional[Union[str, Path]] = None,
    ):
        base_dir = Path(__file__).resolve().parent.parent.parent
        self.features_path = Path(features_path or base_dir / "data" / "processed" / "resolution_features.parquet")
        self.resolution_path = Path(resolution_path or base_dir / "data" / "processed" / "issues_resolution.parquet")

        self.df_features: Optional[pd.DataFrame] = None
        self.df_resolution: Optional[pd.DataFrame] = None
        self.X_matrix: Optional[np.ndarray] = None
        self.y_vector: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        self.ols_results: Optional[Dict[str, Any]] = None
        self.vif_results: Optional[Dict[str, float]] = None
        self.anova_results: Optional[Dict[str, Any]] = None
        self.interaction_results: Optional[Dict[str, Any]] = None

    def load_data(self) -> pd.DataFrame:
        """Loads and prepares features and resolution datasets."""
        if not self.features_path.exists():
            raise FileNotFoundError(f"Features file not found at: {self.features_path}")

        self.df_features = pd.read_parquet(self.features_path)
        if self.resolution_path.exists():
            self.df_resolution = pd.read_parquet(self.resolution_path)

        # Exclude IDs, target, and collinear author_association dummies (retaining is_internal_author)
        ignore_cols = {"issue_id", "issue_number", "resolution_time_log_hours"}
        self.feature_names = [
            c for c in self.df_features.columns
            if c not in ignore_cols and not c.startswith("author_association_")
        ]

        self.X_matrix = self.df_features[self.feature_names].values
        self.y_vector = self.df_features["resolution_time_log_hours"].values
        return self.df_features

    def fit_ols(self) -> Dict[str, Any]:
        """
        Fits multivariate OLS regression on log-transformed resolution duration.
        Computes coefficients, SEs, t-statistics, p-values, 95% CIs, and effect sizes.
        """
        if self.X_matrix is None or self.y_vector is None:
            self.load_data()

        X = self.X_matrix
        y = self.y_vector
        n, p = X.shape
        
        # Design matrix with intercept
        X_const = np.column_stack([np.ones(n), X])
        col_names = ["const"] + self.feature_names

        # Normal equations with pseudo-inverse for numerical stability
        XtX = X_const.T @ X_const
        XtX_inv = np.linalg.pinv(XtX)
        beta = XtX_inv @ (X_const.T @ y)

        # Residuals and variance
        y_pred = X_const @ beta
        residuals = y - y_pred
        rss = float(np.sum(residuals ** 2))
        tss = float(np.sum((y - np.mean(y)) ** 2))

        df_model = p
        df_error = n - p - 1
        r_squared = 1.0 - (rss / tss)
        adj_r_squared = 1.0 - (rss / df_error) / (tss / (n - 1))

        # Overall F-test
        f_stat = (r_squared / df_model) / ((1.0 - r_squared) / df_error)
        f_pval = float(1.0 - stats.f.cdf(f_stat, df_model, df_error))

        # Residual variance and covariance
        s_squared = rss / df_error
        cov_matrix = s_squared * XtX_inv
        std_errors = np.sqrt(np.maximum(np.diag(cov_matrix), 1e-15))

        # t-statistics and two-tailed p-values
        t_stats = beta / std_errors
        p_values = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stats), df=df_error))

        # 95% Confidence Intervals
        t_crit = float(stats.t.ppf(0.975, df=df_error))
        ci_lower = beta - t_crit * std_errors
        ci_upper = beta + t_crit * std_errors

        # Percentage effect size: (exp(beta) - 1) * 100
        pct_effects = (np.exp(beta) - 1.0) * 100.0

        # Log-Likelihood, AIC, BIC
        ll = float(-0.5 * n * (np.log(2.0 * np.pi) + np.log(rss / n) + 1.0))
        aic = float(2.0 * (p + 1) - 2.0 * ll)
        bic = float(np.log(n) * (p + 1) - 2.0 * ll)

        coefficients_list = []
        for i, name in enumerate(col_names):
            coefficients_list.append({
                "feature": name,
                "coefficient": float(beta[i]),
                "std_error": float(std_errors[i]),
                "t_statistic": float(t_stats[i]),
                "p_value": float(p_values[i]),
                "ci_lower_95": float(ci_lower[i]),
                "ci_upper_95": float(ci_upper[i]),
                "pct_effect": float(pct_effects[i]),
                "is_significant_05": bool(p_values[i] < 0.05),
                "is_significant_01": bool(p_values[i] < 0.01),
            })

        self.ols_results = {
            "n_samples": int(n),
            "n_predictors": int(p),
            "df_model": int(df_model),
            "df_error": int(df_error),
            "rss": rss,
            "tss": tss,
            "r_squared": float(r_squared),
            "adj_r_squared": float(adj_r_squared),
            "f_statistic": float(f_stat),
            "f_p_value": float(f_pval),
            "log_likelihood": ll,
            "aic": aic,
            "bic": bic,
            "residual_std": float(np.sqrt(s_squared)),
            "coefficients": coefficients_list,
        }
        return self.ols_results

    def compute_vif(self) -> Dict[str, float]:
        """
        Computes Variance Inflation Factor (VIF) for all predictors.
        VIF_j = 1 / (1 - R_j^2)
        """
        if self.X_matrix is None:
            self.load_data()

        X_df = self.df_features[self.feature_names]
        vifs: Dict[str, float] = {}

        for col in self.feature_names:
            y_col = X_df[col].values
            X_other = X_df.drop(columns=[col]).values

            if np.std(y_col) < 1e-12:
                vifs[col] = 1.0
                continue

            reg = LinearRegression().fit(X_other, y_col)
            r2 = float(reg.score(X_other, y_col))
            if r2 >= 0.99999:
                vif = 99999.0
            else:
                vif = float(1.0 / (1.0 - r2))
            vifs[col] = round(vif, 4)

        self.vif_results = vifs
        return self.vif_results

    def compute_anova_decomposition(self) -> Dict[str, Any]:
        """
        Partitions sum of squares into 5 Factor Families via partial ANOVA:
        Repository Governance, Issue Typology, Structural Complexity,
        Technical Evidence, and Temporal & Author Context.
        """
        if self.ols_results is None:
            self.fit_ols()

        y = self.y_vector
        n = len(y)
        full_rss = self.ols_results["rss"]
        df_error = self.ols_results["df_error"]

        family_results: List[Dict[str, Any]] = []

        for family_name, family_cols in FACTOR_FAMILIES.items():
            valid_cols = [c for c in family_cols if c in self.feature_names]
            if not valid_cols:
                continue

            reduced_cols = [c for c in self.feature_names if c not in valid_cols]
            X_red = self.df_features[reduced_cols].values
            X_red_const = np.column_stack([np.ones(n), X_red])

            beta_red = np.linalg.pinv(X_red_const.T @ X_red_const) @ (X_red_const.T @ y)
            rss_red = float(np.sum((y - X_red_const @ beta_red) ** 2))

            ss_family = max(0.0, rss_red - full_rss)
            df_family = len(valid_cols)

            f_stat = (ss_family / df_family) / (full_rss / df_error) if df_family > 0 else 0.0
            p_val = float(1.0 - stats.f.cdf(f_stat, df_family, df_error)) if df_family > 0 else 1.0
            partial_eta2 = float(ss_family / (ss_family + full_rss))

            family_results.append({
                "family": family_name,
                "n_features": df_family,
                "features": valid_cols,
                "sum_of_squares": ss_family,
                "df": df_family,
                "f_statistic": float(f_stat),
                "p_value": float(p_val),
                "partial_eta_squared": partial_eta2,
                "is_significant_05": bool(p_val < 0.05),
            })

        total_ss_explained = sum(f["sum_of_squares"] for f in family_results)
        for f in family_results:
            pct = (f["sum_of_squares"] / total_ss_explained * 100.0) if total_ss_explained > 0 else 0.0
            f["pct_explained_variance"] = float(pct)

        family_results.sort(key=lambda x: x["sum_of_squares"], reverse=True)

        self.anova_results = {
            "factor_families": family_results,
            "total_family_ss": float(total_ss_explained),
            "model_rss": float(full_rss),
            "model_tss": float(self.ols_results["tss"]),
        }
        return self.anova_results

    def compute_interaction_effects(self) -> Dict[str, Any]:
        """
        Tests interaction effects between repository and issue category:
        repo_name x canonical_label.
        """
        if self.ols_results is None:
            self.fit_ols()

        y = self.y_vector
        n = len(y)
        rss_base = self.ols_results["rss"]

        repo_cols = [c for c in self.feature_names if c.startswith("repo_name_")]
        cat_cols = [c for c in self.feature_names if c.startswith("canonical_label_")]

        inter_cols = []
        inter_names = []
        for r in repo_cols:
            for c in cat_cols:
                interaction = self.df_features[r].values * self.df_features[c].values
                # Retain interactions with at least 3 occurrences to prevent rank degeneracy
                if np.sum(interaction) >= 3:
                    inter_cols.append(interaction)
                    r_clean = r.replace("repo_name_", "")
                    c_clean = c.replace("canonical_label_", "")
                    inter_names.append(f"{r_clean} * {c_clean}")

        if not inter_cols:
            self.interaction_results = {
                "n_interaction_terms": 0,
                "f_statistic": 0.0,
                "p_value": 1.0,
                "is_significant_05": False,
            }
            return self.interaction_results

        X_inter = np.column_stack(inter_cols)
        X_full = np.column_stack([np.ones(n), self.X_matrix, X_inter])

        beta_full = np.linalg.pinv(X_full.T @ X_full) @ (X_full.T @ y)
        rss_inter = float(np.sum((y - X_full @ beta_full) ** 2))

        df_inter = len(inter_names)
        df_error_inter = n - X_full.shape[1]
        ss_inter = max(0.0, rss_base - rss_inter)

        f_inter = (ss_inter / df_inter) / (rss_inter / df_error_inter) if df_error_inter > 0 else 0.0
        p_inter = float(1.0 - stats.f.cdf(f_inter, df_inter, df_error_inter)) if df_error_inter > 0 else 1.0

        self.interaction_results = {
            "n_interaction_terms": int(df_inter),
            "interaction_terms": inter_names,
            "ss_interaction": float(ss_inter),
            "f_statistic": float(f_inter),
            "p_value": float(p_inter),
            "df_inter": int(df_inter),
            "df_error": int(df_error_inter),
            "is_significant_05": bool(p_inter < 0.05),
        }
        return self.interaction_results

    def generate_figures(self, output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Generates Figure 31: 4-panel comprehensive factor analysis plot.
        - Panel A: Forest plot of standardized OLS regression coefficients with 95% CIs.
        - Panel B: Factor family variance decomposition (ANOVA % explained variance).
        - Panel C: Repository marginal resolution durations across categories.
        - Panel D: Actionable structural factor effect sizes (percentage shifts).
        """
        if self.ols_results is None:
            self.fit_ols()
        if self.anova_results is None:
            self.compute_anova_decomposition()

        base_dir = Path(__file__).resolve().parent.parent.parent
        out_file = Path(output_path or base_dir / "ml" / "resolution_analysis" / "figures" / "fig31_resolution_time_factors.png")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=300)

        # -------------------------------------------------------------
        # Panel A: Forest plot of OLS coefficients
        # -------------------------------------------------------------
        ax_a = axes[0, 0]
        coef_data = [c for c in self.ols_results["coefficients"] if c["feature"] != "const"]
        coef_data.sort(key=lambda x: x["coefficient"])

        features_plot = [c["feature"] for c in coef_data]
        coefs = [c["coefficient"] for c in coef_data]
        err_low = [c["coefficient"] - c["ci_lower_95"] for c in coef_data]
        err_high = [c["ci_upper_95"] - c["coefficient"] for c in coef_data]
        colors = [
            "#2ca02c" if c["is_significant_05"] and c["coefficient"] < 0 else
            "#d62728" if c["is_significant_05"] and c["coefficient"] > 0 else
            "#7f7f7f"
            for c in coef_data
        ]

        y_pos = np.arange(len(features_plot))
        ax_a.errorbar(
            coefs, y_pos, xerr=[err_low, err_high],
            fmt="none", ecolor="#444444", elinewidth=1.2, capsize=3, zorder=2
        )
        ax_a.scatter(coefs, y_pos, c=colors, s=40, zorder=3)
        ax_a.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax_a.set_yticks(y_pos)
        ax_a.set_yticklabels(features_plot, fontsize=8)
        ax_a.set_xlabel("Standardized Regression Coefficient (β) ± 95% CI", fontsize=10, fontweight="bold")
        ax_a.set_title("A: Forest Plot of OLS Predictor Coefficients (Log Resolution Time)", fontsize=11, fontweight="bold")
        ax_a.grid(True, linestyle=":", alpha=0.5, axis="x")

        # Custom legend for Panel A
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c", markersize=8, label="Significant Accelerator (p<0.05)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", markersize=8, label="Significant Decelerator (p<0.05)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#7f7f7f", markersize=8, label="Non-Significant (p≥0.05)"),
        ]
        ax_a.legend(handles=legend_elements, loc="lower right", fontsize=8)

        # -------------------------------------------------------------
        # Panel B: Factor Family Variance Decomposition (ANOVA)
        # -------------------------------------------------------------
        ax_b = axes[0, 1]
        fam_data = self.anova_results["factor_families"]
        fam_names = [f["family"] for f in fam_data]
        pcts = [f["pct_explained_variance"] for f in fam_data]
        f_stats = [f["f_statistic"] for f in fam_data]

        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        y_fam = np.arange(len(fam_names))
        bars = ax_b.barh(y_fam, pcts, color=palette[:len(fam_names)], edgecolor="black", linewidth=0.8, alpha=0.85)
        ax_b.set_yticks(y_fam)
        ax_b.set_yticklabels(fam_names, fontsize=10, fontweight="bold")
        ax_b.set_xlabel("Proportion of Explained Sum of Squares (% R²)", fontsize=10, fontweight="bold")
        ax_b.set_title("B: Factor Family Variance Attribution (Partial ANOVA)", fontsize=11, fontweight="bold")
        ax_b.invert_yaxis()
        ax_b.grid(True, linestyle=":", alpha=0.5, axis="x")

        for i, bar in enumerate(bars):
            width = bar.get_width()
            sig_marker = "***" if fam_data[i]["p_value"] < 0.001 else ("*" if fam_data[i]["p_value"] < 0.05 else "ns")
            ax_b.text(
                width + 1.0, bar.get_y() + bar.get_height() / 2,
                f"{width:.1f}% (F={f_stats[i]:.1f}, {sig_marker})",
                va="center", fontsize=9, fontweight="bold"
            )
        ax_b.set_xlim(0, max(pcts) + 15)

        # -------------------------------------------------------------
        # Panel C: Repository Marginal Means across Issue Categories
        # -------------------------------------------------------------
        ax_c = axes[1, 0]
        if self.df_resolution is not None and "canonical_label" in self.df_resolution.columns:
            closed_df = self.df_resolution[self.df_resolution["state"] == "closed"]
            summary_pivot = closed_df.groupby(["repo_name", "canonical_label"])["resolution_time_log_hours"].agg(["mean", "sem"]).unstack("repo_name")

            major_cats = ["bug", "feature", "other", "documentation"]
            valid_cats = [c for c in major_cats if c in summary_pivot["mean"].index]
            
            repo_colors = {
                "microsoft/vscode": "#007acc",
                "pytorch/pytorch": "#ee4c2c",
                "kubernetes/kubernetes": "#326ce5",
                "huggingface/transformers": "#ffcc00",
            }

            for col in summary_pivot["mean"].columns:
                means = summary_pivot["mean"][col].loc[valid_cats]
                sems = summary_pivot["sem"][col].loc[valid_cats]
                c_val = repo_colors.get(col, "#333333")
                ax_c.plot(valid_cats, means, marker="o", linewidth=2.0, label=col, color=c_val)
                ax_c.fill_between(valid_cats, means - sems, means + sems, alpha=0.15, color=c_val)

            ax_c.set_ylabel("Mean Log Resolution Hours (ln[hrs + 1])", fontsize=10, fontweight="bold")
            ax_c.set_xlabel("Issue Canonical Category", fontsize=10, fontweight="bold")
            ax_c.set_title("C: Repository × Category Interaction Profile", fontsize=11, fontweight="bold")
            ax_c.legend(loc="upper right", fontsize=8)
            ax_c.grid(True, linestyle=":", alpha=0.5)
        else:
            ax_c.text(0.5, 0.5, "Resolution data not available for panel C", ha="center", va="center")

        # -------------------------------------------------------------
        # Panel D: Actionable Structural & Contextual Factors Effect
        # -------------------------------------------------------------
        ax_d = axes[1, 1]
        actionable_features = [
            "code_block_count",
            "has_url",
            "clean_title_word_count",
            "question_mark_count",
            "title_has_bracket_tag",
            "url_count",
            "created_hour",
            "canonical_label_other",
            "canonical_label_documentation",
            "repo_name_microsoft/vscode",
        ]
        actionable_dict = {c["feature"]: c for c in self.ols_results["coefficients"]}
        present_actionable = [f for f in actionable_features if f in actionable_dict]

        effects = [actionable_dict[f]["pct_effect"] for f in present_actionable]
        clean_labels = [
            f.replace("canonical_label_", "category: ")
             .replace("repo_name_", "repo: ")
             .replace("_", " ")
            for f in present_actionable
        ]

        d_colors = ["#2ca02c" if e < 0 else "#d62728" for e in effects]
        y_d = np.arange(len(clean_labels))

        bars_d = ax_d.barh(y_d, effects, color=d_colors, edgecolor="black", linewidth=0.8, alpha=0.85)
        ax_d.axvline(0, color="black", linestyle="--", linewidth=1.0)
        ax_d.set_yticks(y_d)
        ax_d.set_yticklabels(clean_labels, fontsize=9)
        ax_d.set_xlabel("Percentage Change in Resolution Time (Δ% = [e^β - 1] × 100%)", fontsize=10, fontweight="bold")
        ax_d.set_title("D: Actionable & Contextual Factor Effect Sizes", fontsize=11, fontweight="bold")
        ax_d.grid(True, linestyle=":", alpha=0.5, axis="x")

        for bar in bars_d:
            width = bar.get_width()
            align = "left" if width >= 0 else "right"
            offset = 2.0 if width >= 0 else -2.0
            ax_d.text(
                width + offset, bar.get_y() + bar.get_height() / 2,
                f"{width:+.1f}%",
                va="center", ha=align, fontsize=8, fontweight="bold"
            )

        plt.tight_layout()
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close()
        return out_file

    def export_analysis_manifest(self, output_path: Optional[Union[str, Path]] = None) -> Path:
        """Exports complete factor analysis results to JSON."""
        if self.ols_results is None:
            self.fit_ols()
        if self.vif_results is None:
            self.compute_vif()
        if self.anova_results is None:
            self.compute_anova_decomposition()
        if self.interaction_results is None:
            self.compute_interaction_effects()

        base_dir = Path(__file__).resolve().parent.parent.parent
        out_file = Path(output_path or base_dir / "data" / "processed" / "evaluation" / "resolution_factors_analysis.json")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        manifest = {
            "metadata": {
                "pipeline_step": "Phase 8, Step 8.4 (Investigate Resolution-Time Factors)",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "n_samples": self.ols_results["n_samples"],
                "n_predictors": self.ols_results["n_predictors"],
            },
            "ols_model_summary": {
                "r_squared": self.ols_results["r_squared"],
                "adj_r_squared": self.ols_results["adj_r_squared"],
                "f_statistic": self.ols_results["f_statistic"],
                "f_p_value": self.ols_results["f_p_value"],
                "rss": self.ols_results["rss"],
                "tss": self.ols_results["tss"],
                "residual_std": self.ols_results["residual_std"],
                "log_likelihood": self.ols_results["log_likelihood"],
                "aic": self.ols_results["aic"],
                "bic": self.ols_results["bic"],
            },
            "multicollinearity_vif": self.vif_results,
            "anova_variance_decomposition": self.anova_results,
            "interaction_effects": self.interaction_results,
            "coefficients": self.ols_results["coefficients"],
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return out_file

    def run_full_analysis(
        self,
        plot_path: Optional[Union[str, Path]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Runs the complete factor analysis pipeline end-to-end."""
        self.load_data()
        self.fit_ols()
        self.compute_vif()
        self.compute_anova_decomposition()
        self.compute_interaction_effects()
        manifest_file = self.export_analysis_manifest(manifest_path)
        plot_file = self.generate_figures(plot_path)

        return {
            "ols_summary": self.ols_results,
            "vif": self.vif_results,
            "anova": self.anova_results,
            "interaction": self.interaction_results,
            "manifest_path": str(manifest_file),
            "plot_path": str(plot_file),
        }


if __name__ == "__main__":
    analyzer = ResolutionFactorAnalyzer()
    res = analyzer.run_full_analysis()
    print("=" * 70)
    print("PHASE 8, STEP 8.4 FACTOR INVESTIGATION COMPLETED")
    print(f"R²: {res['ols_summary']['r_squared']:.4f} | Adj R²: {res['ols_summary']['adj_r_squared']:.4f}")
    print(f"Overall F-statistic: {res['ols_summary']['f_statistic']:.2f} (p = {res['ols_summary']['f_p_value']:.2e})")
    print(f"Manifest exported: {res['manifest_path']}")
    print(f"Figure 31 generated: {res['plot_path']}")
    print("=" * 70)
