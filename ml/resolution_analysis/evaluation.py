"""
OpenSourceLens — Resolution Time Detailed Model Evaluation & Diagnostics (Phase 8, Step 8.6).

Performs in-depth residual diagnostics, heteroskedasticity and normality testing,
subgroup fairness and generalizability evaluations across repositories and categories,
ordinal classification error analysis, and generates publication Figure 33.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

SPEED_TIER_ORDER: List[str] = ["< 1 Day", "1-7 Days", "7-30 Days", "30+ Days"]


class ResolutionEvaluator:
    """
    Detailed statistical diagnostic evaluation and subgroup performance analysis
    for champion resolution prediction models.
    """

    def __init__(
        self,
        features_path: Optional[Union[str, Path]] = None,
        resolution_path: Optional[Union[str, Path]] = None,
        models_dir: Optional[Union[str, Path]] = None,
    ):
        base_dir = Path(__file__).resolve().parent.parent.parent
        self.features_path = Path(features_path or base_dir / "data" / "processed" / "resolution_features.parquet")
        self.resolution_path = Path(resolution_path or base_dir / "data" / "processed" / "issues_resolution.parquet")
        self.models_dir = Path(models_dir or base_dir / "models")

        self.df_test: Optional[pd.DataFrame] = None
        self.feature_names: List[str] = []
        self.regressor_model: Optional[Any] = None
        self.classifier_model: Optional[Any] = None

        self.reg_preds: Optional[np.ndarray] = None
        self.clf_preds: Optional[np.ndarray] = None
        self.residuals: Optional[np.ndarray] = None

        self.residual_diagnostics: Dict[str, Any] = {}
        self.subgroup_results: Dict[str, Any] = {}
        self.ordinal_results: Dict[str, Any] = {}

    def load_test_data_and_models(self, test_size: float = 0.2, random_state: int = 42) -> pd.DataFrame:
        """Loads features, merges metadata, performs deterministic holdout split, and loads models."""
        if not self.features_path.exists():
            raise FileNotFoundError(f"Features file not found at: {self.features_path}")
        if not self.resolution_path.exists():
            raise FileNotFoundError(f"Resolution file not found at: {self.resolution_path}")

        f_df = pd.read_parquet(self.features_path)
        r_cols = ["issue_id", "title", "repo_name", "canonical_label", "speed_tier", "resolution_time_hours"]
        r_df = pd.read_parquet(self.resolution_path)[r_cols]

        merged = pd.merge(f_df, r_df, on="issue_id", how="inner")

        ignore_cols = {"issue_id", "issue_number", "resolution_time_log_hours", "speed_tier", "title", "repo_name", "canonical_label", "resolution_time_hours"}
        self.feature_names = [
            c for c in f_df.columns
            if c not in ignore_cols and not c.startswith("author_association_")
        ]

        # Deterministic 80/20 stratified split identical to ResolutionBaselineTrainer
        _, test_df = train_test_split(
            merged, test_size=test_size, random_state=random_state, stratify=merged["speed_tier"]
        )
        self.df_test = test_df.copy().reset_index(drop=True)

        # Load models
        reg_file = self.models_dir / "resolution_regressor_champion.joblib"
        clf_file = self.models_dir / "resolution_classifier_champion.joblib"

        if not reg_file.exists() or not clf_file.exists():
            raise FileNotFoundError("Champion models not found in models/ directory. Run Step 8.5 first.")

        reg_payload = joblib.load(reg_file)
        clf_payload = joblib.load(clf_file)

        self.regressor_model = reg_payload["model"]
        self.classifier_model = clf_payload["model"]

        # Generate test predictions
        X_test = self.df_test[self.feature_names].values
        self.reg_preds = self.regressor_model.predict(X_test)
        self.clf_preds = self.classifier_model.predict(X_test)

        self.df_test["pred_log_hours"] = self.reg_preds
        self.df_test["pred_hours"] = np.maximum(0.0, np.exp(self.reg_preds) - 1.0)
        self.df_test["pred_speed_tier"] = self.clf_preds

        y_actual = self.df_test["resolution_time_log_hours"].values
        self.residuals = y_actual - self.reg_preds
        self.df_test["residual"] = self.residuals
        return self.df_test

    def evaluate_residuals(self) -> Dict[str, Any]:
        """
        Conducts formal statistical residual diagnostics:
        - Normality: D'Agostino-Pearson omnibus test & Shapiro-Wilk test
        - Skewness and kurtosis
        - Heteroskedasticity: Breusch-Pagan test
        - Extreme error case studies (Top 5 under-predictions and Top 5 over-predictions)
        """
        if self.residuals is None:
            self.load_test_data_and_models()

        res = self.residuals
        n = len(res)
        res_mean = float(np.mean(res))
        res_std = float(np.std(res))
        skewness = float(stats.skew(res))
        excess_kurt = float(stats.kurtosis(res))

        # Normality tests
        omni_stat, omni_p = stats.normaltest(res)
        shapiro_stat, shapiro_p = stats.shapiro(res)

        # Breusch-Pagan heteroskedasticity test: regress e² on fitted values
        e_sq = res ** 2
        slope, intercept, r_val, bp_p_val, _ = stats.linregress(self.reg_preds, e_sq)
        bp_stat = float(n * (r_val ** 2))
        bp_pval = float(1.0 - stats.chi2.cdf(bp_stat, df=1))

        # Top 5 Under-predictions (actual >> predicted, lingered longer than expected)
        under_cases = self.df_test.sort_values("residual", ascending=False).head(5)
        under_list = []
        for _, row in under_cases.iterrows():
            under_list.append({
                "issue_number": int(row["issue_number"]),
                "repo_name": str(row["repo_name"]),
                "title": str(row["title"])[:80],
                "canonical_label": str(row["canonical_label"]),
                "actual_hours": float(row["resolution_time_hours"]),
                "predicted_hours": float(row["pred_hours"]),
                "residual_log": float(row["residual"]),
            })

        # Top 5 Over-predictions (actual << predicted, resolved much faster than expected)
        over_cases = self.df_test.sort_values("residual", ascending=True).head(5)
        over_list = []
        for _, row in over_cases.iterrows():
            over_list.append({
                "issue_number": int(row["issue_number"]),
                "repo_name": str(row["repo_name"]),
                "title": str(row["title"])[:80],
                "canonical_label": str(row["canonical_label"]),
                "actual_hours": float(row["resolution_time_hours"]),
                "predicted_hours": float(row["pred_hours"]),
                "residual_log": float(row["residual"]),
            })

        self.residual_diagnostics = {
            "n_samples": n,
            "mean": res_mean,
            "std": res_std,
            "skewness": skewness,
            "excess_kurtosis": excess_kurt,
            "normality_omnibus": {
                "statistic": float(omni_stat),
                "p_value": float(omni_p),
                "is_gaussian_05": bool(omni_p >= 0.05),
            },
            "normality_shapiro": {
                "statistic": float(shapiro_stat),
                "p_value": float(shapiro_p),
                "is_gaussian_05": bool(shapiro_p >= 0.05),
            },
            "breusch_pagan_heteroskedasticity": {
                "statistic": bp_stat,
                "p_value": bp_pval,
                "is_homoskedastic_05": bool(bp_pval >= 0.05),
            },
            "top_under_predictions": under_list,
            "top_over_predictions": over_list,
        }
        return self.residual_diagnostics

    def evaluate_subgroups(self) -> Dict[str, Any]:
        """
        Disaggregates model performance across repositories and issue categories
        to evaluate generalizability and detect cross-ecosystem bias.
        """
        if self.df_test is None:
            self.load_test_data_and_models()

        repo_metrics = {}
        for repo, grp in self.df_test.groupby("repo_name"):
            n_sub = len(grp)
            y_sub = grp["resolution_time_log_hours"].values
            preds_sub = grp["pred_log_hours"].values
            clf_actual = grp["speed_tier"].values
            clf_preds = grp["pred_speed_tier"].values

            r2 = float(r2_score(y_sub, preds_sub)) if n_sub > 1 else 0.0
            mae_log = float(mean_absolute_error(y_sub, preds_sub))
            rmse_log = float(np.sqrt(mean_squared_error(y_sub, preds_sub)))
            acc = float(accuracy_score(clf_actual, clf_preds))
            bal_acc = float(balanced_accuracy_score(clf_actual, clf_preds))
            macro_f1 = float(f1_score(clf_actual, clf_preds, average="macro", zero_division=0))

            if np.std(preds_sub) > 1e-12:
                rho, pval = stats.spearmanr(y_sub, preds_sub)
            else:
                rho, pval = 0.0, 1.0

            repo_metrics[repo] = {
                "n_samples": n_sub,
                "test_r2": r2,
                "test_mae_log_hours": mae_log,
                "test_rmse_log_hours": rmse_log,
                "spearman_rho": float(rho),
                "spearman_p_value": float(pval),
                "test_accuracy": acc,
                "test_balanced_accuracy": bal_acc,
                "test_macro_f1": macro_f1,
            }

        cat_metrics = {}
        for cat, grp in self.df_test.groupby("canonical_label"):
            n_sub = len(grp)
            if n_sub < 5:
                continue

            y_sub = grp["resolution_time_log_hours"].values
            preds_sub = grp["pred_log_hours"].values
            clf_actual = grp["speed_tier"].values
            clf_preds = grp["pred_speed_tier"].values

            r2 = float(r2_score(y_sub, preds_sub)) if n_sub > 1 else 0.0
            mae_log = float(mean_absolute_error(y_sub, preds_sub))
            acc = float(accuracy_score(clf_actual, clf_preds))
            macro_f1 = float(f1_score(clf_actual, clf_preds, average="macro", zero_division=0))

            if np.std(preds_sub) > 1e-12:
                rho, pval = stats.spearmanr(y_sub, preds_sub)
            else:
                rho, pval = 0.0, 1.0

            cat_metrics[cat] = {
                "n_samples": n_sub,
                "test_r2": r2,
                "test_mae_log_hours": mae_log,
                "spearman_rho": float(rho),
                "test_accuracy": acc,
                "test_macro_f1": macro_f1,
            }

        self.subgroup_results = {
            "by_repository": repo_metrics,
            "by_canonical_label": cat_metrics,
        }
        return self.subgroup_results

    def evaluate_ordinal_classification(self) -> Dict[str, Any]:
        """
        Evaluates ordinal classification accuracy, proximity bounds,
        and weighted Cohen's Kappa on velocity tiers.
        """
        if self.df_test is None:
            self.load_test_data_and_models()

        order_map = {name: i for i, name in enumerate(SPEED_TIER_ORDER)}
        y_actual_ranks = np.array([order_map[v] for v in self.df_test["speed_tier"]])
        pred_ranks = np.array([order_map[v] for v in self.df_test["pred_speed_tier"]])

        diff = np.abs(y_actual_ranks - pred_ranks)
        n = len(diff)

        exact_acc = float(np.mean(diff == 0))
        adjacent_acc = float(np.mean(diff <= 1))
        severe_rate = float(np.mean(diff >= 2))
        extreme_rate = float(np.mean(diff == 3))
        maoe = float(np.mean(diff))

        linear_kappa = float(cohen_kappa_score(y_actual_ranks, pred_ranks, weights="linear"))
        quad_kappa = float(cohen_kappa_score(y_actual_ranks, pred_ranks, weights="quadratic"))

        self.ordinal_results = {
            "n_samples": n,
            "exact_accuracy": exact_acc,
            "adjacent_accuracy_within_1_tier": adjacent_acc,
            "severe_error_rate_ge_2_tiers": severe_rate,
            "extreme_error_rate_3_tiers": extreme_rate,
            "mean_absolute_ordinal_error": maoe,
            "linear_weighted_cohen_kappa": linear_kappa,
            "quadratic_weighted_cohen_kappa": quad_kappa,
            "error_distance_distribution": {
                "distance_0_exact": int(np.sum(diff == 0)),
                "distance_1_off_by_one": int(np.sum(diff == 1)),
                "distance_2_off_by_two": int(np.sum(diff == 2)),
                "distance_3_off_by_three": int(np.sum(diff == 3)),
            },
        }
        return self.ordinal_results

    def generate_figures(self, output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Generates Figure 33: 4-panel comprehensive evaluation plot.
        - Panel A: Residuals vs. Fitted values with LOESS trend line.
        - Panel B: Normal Q-Q plot of standardized residuals.
        - Panel C: Subgroup disaggregation bar chart (MAE & Accuracy by Repo).
        - Panel D: Ordinal error proximity distribution.
        """
        if not self.residual_diagnostics:
            self.evaluate_residuals()
        if not self.subgroup_results:
            self.evaluate_subgroups()
        if not self.ordinal_results:
            self.evaluate_ordinal_classification()

        base_dir = Path(__file__).resolve().parent.parent.parent
        out_file = Path(output_path or base_dir / "ml" / "resolution_analysis" / "figures" / "fig33_resolution_model_evaluation.png")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

        # -------------------------------------------------------------
        # Panel A: Residuals vs. Fitted Values
        # -------------------------------------------------------------
        ax_a = axes[0, 0]
        ax_a.scatter(self.reg_preds, self.residuals, alpha=0.55, color="#1f77b4", edgecolors="none", s=35)
        ax_a.axhline(0, color="red", linestyle="--", linewidth=1.2, label="Zero Error Baseline")

        # Trend line
        slope, intercept, _, _, _ = stats.linregress(self.reg_preds, self.residuals)
        x_trend = np.linspace(float(np.min(self.reg_preds)), float(np.max(self.reg_preds)), 100)
        ax_a.plot(x_trend, intercept + slope * x_trend, "k-", linewidth=1.0, label="Residual Trend")

        ax_a.set_xlabel("Fitted Values (Predicted Log Resolution Hours)", fontsize=10, fontweight="bold")
        ax_a.set_ylabel("Residuals (Observed - Predicted)", fontsize=10, fontweight="bold")
        bp_p = self.residual_diagnostics["breusch_pagan_heteroskedasticity"]["p_value"]
        ax_a.set_title(f"A: Residuals vs. Fitted Values (Breusch-Pagan p={bp_p:.3f})", fontsize=11, fontweight="bold")
        ax_a.legend(loc="upper left", fontsize=9)
        ax_a.grid(True, linestyle=":", alpha=0.5)

        # -------------------------------------------------------------
        # Panel B: Normal Q-Q Plot of Standardized Residuals
        # -------------------------------------------------------------
        ax_b = axes[0, 1]
        std_residuals = (self.residuals - np.mean(self.residuals)) / np.std(self.residuals)
        (osm, osr), (slope_qq, intercept_qq, r_qq) = stats.probplot(std_residuals, dist="norm")
        ax_b.scatter(osm, osr, color="#2ca02c", alpha=0.6, s=35, label="Standardized Residuals")
        ax_b.plot(osm, slope_qq * np.array(osm) + intercept_qq, "r--", linewidth=1.2, label=f"Gaussian Line (r={r_qq:.3f})")

        shapiro_p = self.residual_diagnostics["normality_shapiro"]["p_value"]
        ax_b.set_xlabel("Theoretical Gaussian Quantiles", fontsize=10, fontweight="bold")
        ax_b.set_ylabel("Sample Quantiles", fontsize=10, fontweight="bold")
        ax_b.set_title(f"B: Normal Q-Q Plot of Residuals (Shapiro-Wilk p={shapiro_p:.4f})", fontsize=11, fontweight="bold")
        ax_b.legend(loc="upper left", fontsize=9)
        ax_b.grid(True, linestyle=":", alpha=0.5)

        # -------------------------------------------------------------
        # Panel C: Subgroup Performance Breakdown by Repository
        # -------------------------------------------------------------
        ax_c = axes[1, 0]
        repos = list(self.subgroup_results["by_repository"].keys())
        mae_sub = [self.subgroup_results["by_repository"][r]["test_mae_log_hours"] for r in repos]
        acc_sub = [self.subgroup_results["by_repository"][r]["test_accuracy"] * 100.0 for r in repos]

        clean_repos = [r.replace("microsoft/", "").replace("pytorch/", "").replace("kubernetes/", "").replace("huggingface/", "") for r in repos]
        x_sub = np.arange(len(repos))
        width = 0.35

        ax_c_twin = ax_c.twinx()
        b1 = ax_c.bar(x_sub - width / 2, mae_sub, width, label="Test MAE (log-hrs)", color="#ff7f0e", edgecolor="black", alpha=0.85)
        b2 = ax_c_twin.bar(x_sub + width / 2, acc_sub, width, label="Classification Acc (%)", color="#9467bd", edgecolor="black", alpha=0.85)

        ax_c.set_xticks(x_sub)
        ax_c.set_xticklabels(clean_repos, fontsize=9, fontweight="bold")
        ax_c.set_ylabel("MAE (log-hours)", color="#d95f02", fontsize=10, fontweight="bold")
        ax_c_twin.set_ylabel("4-Tier Accuracy (%)", color="#7570b3", fontsize=10, fontweight="bold")
        ax_c_twin.set_ylim(0, 100)
        ax_c.set_title("C: Subgroup Performance by Repository", fontsize=11, fontweight="bold")
        ax_c.grid(True, linestyle=":", alpha=0.5, axis="y")

        # -------------------------------------------------------------
        # Panel D: Ordinal Error Proximity Distribution
        # -------------------------------------------------------------
        ax_d = axes[1, 1]
        dist_data = self.ordinal_results["error_distance_distribution"]
        categories = ["Exact (0)", "Off-by-1 (±1)", "Off-by-2 (±2)", "Extreme (±3)"]
        counts = [dist_data["distance_0_exact"], dist_data["distance_1_off_by_one"], dist_data["distance_2_off_by_two"], dist_data["distance_3_off_by_three"]]
        pcts = [c / len(self.df_test) * 100.0 for c in counts]
        colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]

        y_pos = np.arange(len(categories))
        bars = ax_d.barh(y_pos, pcts, color=colors, edgecolor="black", alpha=0.85)
        ax_d.set_yticks(y_pos)
        ax_d.set_yticklabels(categories, fontsize=10, fontweight="bold")
        ax_d.invert_yaxis()
        ax_d.set_xlabel("Percentage of Test Set Issues (%)", fontsize=10, fontweight="bold")
        adj_pct = self.ordinal_results["adjacent_accuracy_within_1_tier"] * 100.0
        ax_d.set_title(f"D: Ordinal Classification Proximity (Within ±1: {adj_pct:.1f}%)", fontsize=11, fontweight="bold")
        ax_d.grid(True, linestyle=":", alpha=0.5, axis="x")

        for bar in bars:
            width = bar.get_width()
            ax_d.text(
                width + 1.0, bar.get_y() + bar.get_height() / 2,
                f"{width:.1f}%",
                va="center", fontsize=9, fontweight="bold"
            )
        ax_d.set_xlim(0, max(pcts) + 12)

        plt.tight_layout()
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close()
        return out_file

    def export_diagnostics_manifest(self, output_path: Optional[Union[str, Path]] = None) -> Path:
        """Exports complete evaluation diagnostics to JSON."""
        if not self.residual_diagnostics:
            self.evaluate_residuals()
        if not self.subgroup_results:
            self.evaluate_subgroups()
        if not self.ordinal_results:
            self.evaluate_ordinal_classification()

        base_dir = Path(__file__).resolve().parent.parent.parent
        out_file = Path(output_path or base_dir / "data" / "processed" / "evaluation" / "resolution_evaluation_diagnostics.json")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        manifest = {
            "metadata": {
                "pipeline_step": "Phase 8, Step 8.6 (Detailed Model Evaluation & Diagnostics)",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "n_test_samples": len(self.df_test),
            },
            "residual_diagnostics": self.residual_diagnostics,
            "subgroup_performance": self.subgroup_results,
            "ordinal_classification_analysis": self.ordinal_results,
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return out_file

    def run_full_evaluation(
        self,
        plot_path: Optional[Union[str, Path]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Runs the complete evaluation pipeline end-to-end."""
        self.load_test_data_and_models()
        self.evaluate_residuals()
        self.evaluate_subgroups()
        self.evaluate_ordinal_classification()
        manifest_file = self.export_diagnostics_manifest(manifest_path)
        plot_file = self.generate_figures(plot_path)

        return {
            "residuals": self.residual_diagnostics,
            "subgroups": self.subgroup_results,
            "ordinal": self.ordinal_results,
            "manifest_path": str(manifest_file),
            "plot_path": str(plot_file),
        }


if __name__ == "__main__":
    evaluator = ResolutionEvaluator()
    res = evaluator.run_full_evaluation()
    print("=" * 70)
    print("PHASE 8, STEP 8.6 MODEL EVALUATION & DIAGNOSTICS COMPLETED")
    print(f"Test Normality Omnibus p-val: {res['residuals']['normality_omnibus']['p_value']:.4f}")
    print(f"Breusch-Pagan p-val:         {res['residuals']['breusch_pagan_heteroskedasticity']['p_value']:.4f}")
    print(f"Exact Accuracy:              {res['ordinal']['exact_accuracy']*100:.1f}%")
    print(f"Adjacent Accuracy (±1 Tier): {res['ordinal']['adjacent_accuracy_within_1_tier']*100:.1f}%")
    print(f"Linear Weighted Kappa:       {res['ordinal']['linear_weighted_cohen_kappa']:.4f}")
    print(f"Manifest exported:           {res['manifest_path']}")
    print(f"Figure 33 generated:         {res['plot_path']}")
    print("=" * 70)
