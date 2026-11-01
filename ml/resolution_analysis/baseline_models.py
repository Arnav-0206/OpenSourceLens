"""
OpenSourceLens — Resolution Time Baseline Prediction Modeling (Phase 8, Step 8.5).

Trains, benchmarks, and serializes baseline machine learning models for:
1. Continuous duration regression (log resolution hours).
2. Discrete speed-tier classification (< 1 Day, 1-7 Days, 7-30 Days, 30+ Days).

Enforces strict t=0 feature isolation, 80/20 stratified split, 5-fold cross-validation,
and exports champion models and Figure 32 benchmark comparisons.
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
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split

SPEED_TIER_ORDER: List[str] = ["< 1 Day", "1-7 Days", "7-30 Days", "30+ Days"]


class ResolutionBaselineTrainer:
    """
    Trains and benchmarks baseline prediction models for issue resolution time
    under both regression and classification formulations.
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
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.df_merged: Optional[pd.DataFrame] = None
        self.feature_names: List[str] = []
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_reg_train: Optional[np.ndarray] = None
        self.y_reg_test: Optional[np.ndarray] = None
        self.y_clf_train: Optional[np.ndarray] = None
        self.y_clf_test: Optional[np.ndarray] = None

        self.regression_benchmarks: Dict[str, Dict[str, Any]] = {}
        self.classification_benchmarks: Dict[str, Dict[str, Any]] = {}
        self.trained_regressors: Dict[str, Any] = {}
        self.trained_classifiers: Dict[str, Any] = {}

        self.champion_regressor_name: Optional[str] = None
        self.champion_classifier_name: Optional[str] = None
        self.champion_regressor: Optional[Any] = None
        self.champion_classifier: Optional[Any] = None

    def load_and_split_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads features and speed tier labels, drops collinear dummies, and performs
        an 80/20 stratified train/test split.
        """
        if not self.features_path.exists():
            raise FileNotFoundError(f"Features file not found at: {self.features_path}")
        if not self.resolution_path.exists():
            raise FileNotFoundError(f"Resolution file not found at: {self.resolution_path}")

        f_df = pd.read_parquet(self.features_path)
        r_df = pd.read_parquet(self.resolution_path)[["issue_id", "speed_tier"]]

        self.df_merged = pd.merge(f_df, r_df, on="issue_id", how="inner")

        ignore_cols = {"issue_id", "issue_number", "resolution_time_log_hours", "speed_tier"}
        self.feature_names = [
            c for c in self.df_merged.columns
            if c not in ignore_cols and not c.startswith("author_association_")
        ]

        X = np.asarray(self.df_merged[self.feature_names].values, dtype=float)
        y_reg = np.asarray(self.df_merged["resolution_time_log_hours"].values, dtype=float)
        y_clf = np.asarray(self.df_merged["speed_tier"].to_numpy(), dtype=str)

        (
            self.X_train,
            self.X_test,
            self.y_reg_train,
            self.y_reg_test,
            self.y_clf_train,
            self.y_clf_test,
        ) = train_test_split(
            X, y_reg, y_clf, test_size=test_size, random_state=random_state, stratify=y_clf
        )
        return self.X_train, self.X_test

    def train_regressors(self) -> Dict[str, Dict[str, Any]]:
        """
        Trains and benchmarks 4 candidate regression models:
        1. Dummy (Mean)
        2. Ridge (alpha=1.0)
        3. Random Forest Regressor
        4. Gradient Boosting Regressor
        """
        if self.X_train is None:
            self.load_and_split_data()

        models = {
            "dummy_mean": DummyRegressor(strategy="mean"),
            "dummy_median": DummyRegressor(strategy="median"),
            "ridge": Ridge(alpha=1.0, random_state=42),
            "random_forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42),
        }

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in models.items():
            # 5-fold cross validation R² on training set
            cv_scores = cross_val_score(model, self.X_train, self.y_reg_train, cv=kf, scoring="r2")
            cv_mae = -cross_val_score(model, self.X_train, self.y_reg_train, cv=kf, scoring="neg_mean_absolute_error")

            # Fit on full training set
            model.fit(self.X_train, self.y_reg_train)
            test_preds = model.predict(self.X_test)

            # Test metrics
            r2 = float(r2_score(self.y_reg_test, test_preds))
            mae = float(mean_absolute_error(self.y_reg_test, test_preds))
            rmse = float(np.sqrt(mean_squared_error(self.y_reg_test, test_preds)))
            
            # Spearman rank correlation on test set
            if np.std(test_preds) < 1e-12:
                spearman_rho, spearman_pval = 0.0, 1.0
            else:
                spearman_rho, spearman_pval = stats.spearmanr(self.y_reg_test, test_preds)

            # Translation to real hours: MAE in original hours
            actual_hours = np.exp(self.y_reg_test) - 1.0
            pred_hours = np.maximum(0.0, np.exp(test_preds) - 1.0)
            mae_hours = float(mean_absolute_error(actual_hours, pred_hours))

            self.trained_regressors[name] = model
            self.regression_benchmarks[name] = {
                "model_name": name,
                "cv_r2_mean": float(np.mean(cv_scores)),
                "cv_r2_std": float(np.std(cv_scores)),
                "cv_mae_mean": float(np.mean(cv_mae)),
                "test_r2": r2,
                "test_mae_log_hours": mae,
                "test_rmse_log_hours": rmse,
                "test_mae_actual_hours": mae_hours,
                "spearman_rho": float(spearman_rho) if not np.isnan(spearman_rho) else 0.0,
                "spearman_p_value": float(spearman_pval) if not np.isnan(spearman_pval) else 1.0,
                "test_predictions": [float(p) for p in test_preds],
            }

        # Select champion by highest test R²
        valid_candidates = {k: v for k, v in self.regression_benchmarks.items() if not k.startswith("dummy")}
        self.champion_regressor_name = max(valid_candidates.keys(), key=lambda k: valid_candidates[k]["test_r2"])
        self.champion_regressor = self.trained_regressors[self.champion_regressor_name]
        return self.regression_benchmarks

    def train_classifiers(self) -> Dict[str, Dict[str, Any]]:
        """
        Trains and benchmarks 4 candidate classification models:
        1. Dummy (Prior / Most Frequent)
        2. Logistic Regression (L2)
        3. Random Forest Classifier
        4. Gradient Boosting Classifier
        """
        if self.X_train is None:
            self.load_and_split_data()

        models = {
            "dummy_prior": DummyClassifier(strategy="prior"),
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
            "random_forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            "gradient_boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42),
        }

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in models.items():
            # 5-fold cross validation Macro F1 on training set
            cv_f1 = cross_val_score(model, self.X_train, self.y_clf_train, cv=skf, scoring="f1_macro")
            cv_acc = cross_val_score(model, self.X_train, self.y_clf_train, cv=skf, scoring="accuracy")

            # Fit on full training set
            model.fit(self.X_train, self.y_clf_train)
            test_preds = model.predict(self.X_test)

            acc = float(accuracy_score(self.y_clf_test, test_preds))
            bal_acc = float(balanced_accuracy_score(self.y_clf_test, test_preds))
            macro_f1 = float(f1_score(self.y_clf_test, test_preds, average="macro"))
            weighted_f1 = float(f1_score(self.y_clf_test, test_preds, average="weighted"))

            cm = confusion_matrix(self.y_clf_test, test_preds, labels=SPEED_TIER_ORDER).tolist()

            self.trained_classifiers[name] = model
            self.classification_benchmarks[name] = {
                "model_name": name,
                "cv_macro_f1_mean": float(np.mean(cv_f1)),
                "cv_macro_f1_std": float(np.std(cv_f1)),
                "cv_accuracy_mean": float(np.mean(cv_acc)),
                "test_accuracy": acc,
                "test_balanced_accuracy": bal_acc,
                "test_macro_f1": macro_f1,
                "test_weighted_f1": weighted_f1,
                "confusion_matrix": cm,
                "test_predictions": [str(p) for p in test_preds],
            }

        # Select champion by highest test Macro F1
        valid_candidates = {k: v for k, v in self.classification_benchmarks.items() if not k.startswith("dummy")}
        self.champion_classifier_name = max(valid_candidates.keys(), key=lambda k: valid_candidates[k]["test_macro_f1"])
        self.champion_classifier = self.trained_classifiers[self.champion_classifier_name]
        return self.classification_benchmarks

    def save_models(self) -> Dict[str, Path]:
        """Serializes champion models and metadata to models/ directory."""
        if self.champion_regressor is None or self.champion_classifier is None:
            self.train_regressors()
            self.train_classifiers()

        reg_file = self.models_dir / "resolution_regressor_champion.joblib"
        clf_file = self.models_dir / "resolution_classifier_champion.joblib"

        reg_payload = {
            "model": self.champion_regressor,
            "model_name": self.champion_regressor_name,
            "feature_names": self.feature_names,
            "target": "resolution_time_log_hours",
            "metrics": self.regression_benchmarks[self.champion_regressor_name],
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        joblib.dump(reg_payload, reg_file)

        clf_payload = {
            "model": self.champion_classifier,
            "model_name": self.champion_classifier_name,
            "feature_names": self.feature_names,
            "target": "speed_tier",
            "classes": SPEED_TIER_ORDER,
            "metrics": self.classification_benchmarks[self.champion_classifier_name],
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        joblib.dump(clf_payload, clf_file)

        return {"regressor": reg_file, "classifier": clf_file}

    def export_benchmark_manifest(self, output_path: Optional[Union[str, Path]] = None) -> Path:
        """Exports complete benchmark results and split metadata to JSON."""
        if not self.regression_benchmarks or not self.classification_benchmarks:
            self.train_regressors()
            self.train_classifiers()

        base_dir = Path(__file__).resolve().parent.parent.parent
        out_file = Path(output_path or base_dir / "data" / "processed" / "evaluation" / "resolution_baseline_models.json")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # Sanitize predictions from summary view to keep manifest clean
        reg_summary = {}
        for k, v in self.regression_benchmarks.items():
            item = v.copy()
            item.pop("test_predictions", None)
            reg_summary[k] = item

        clf_summary = {}
        for k, v in self.classification_benchmarks.items():
            item = v.copy()
            item.pop("test_predictions", None)
            clf_summary[k] = item

        manifest = {
            "metadata": {
                "pipeline_step": "Phase 8, Step 8.5 (Build Baseline Prediction Model)",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "n_samples_total": len(self.df_merged),
                "n_samples_train": len(self.X_train),
                "n_samples_test": len(self.X_test),
                "n_features": len(self.feature_names),
                "feature_names": self.feature_names,
                "champion_regressor": self.champion_regressor_name,
                "champion_classifier": self.champion_classifier_name,
            },
            "regression_benchmarks": reg_summary,
            "classification_benchmarks": clf_summary,
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return out_file

    def generate_figures(self, output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Generates Figure 32: 4-panel comprehensive benchmark plot.
        - Panel A: Regressor comparison (Test R² and MAE).
        - Panel B: Champion regressor predicted vs. observed scatter plot.
        - Panel C: Classifier comparison (Test Accuracy and Macro F1).
        - Panel D: Champion classifier confusion matrix heatmap.
        """
        if not self.regression_benchmarks or not self.classification_benchmarks:
            self.train_regressors()
            self.train_classifiers()

        base_dir = Path(__file__).resolve().parent.parent.parent
        out_file = Path(output_path or base_dir / "ml" / "resolution_analysis" / "figures" / "fig32_baseline_model_benchmarks.png")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

        # -------------------------------------------------------------
        # Panel A: Regressor Benchmark Comparison (R² and MAE)
        # -------------------------------------------------------------
        ax_a = axes[0, 0]
        reg_names = list(self.regression_benchmarks.keys())
        r2_vals = [self.regression_benchmarks[m]["test_r2"] for m in reg_names]
        mae_vals = [self.regression_benchmarks[m]["test_mae_log_hours"] for m in reg_names]

        clean_reg_names = [m.replace("_", " ").title() for m in reg_names]
        x_reg = np.arange(len(reg_names))
        width = 0.35

        ax_a.bar(x_reg - width / 2, r2_vals, width, label="Test R²", color="#1f77b4", edgecolor="black", alpha=0.85)
        ax_a.bar(x_reg + width / 2, mae_vals, width, label="Test MAE (log-hrs)", color="#ff7f0e", edgecolor="black", alpha=0.85)
        ax_a.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax_a.set_xticks(x_reg)
        ax_a.set_xticklabels(clean_reg_names, rotation=15, ha="right", fontsize=9)
        ax_a.set_ylabel("Metric Value", fontsize=10, fontweight="bold")
        ax_a.set_title(f"A: Regressor Benchmarks (Champion: {self.champion_regressor_name.title()})", fontsize=11, fontweight="bold")
        ax_a.legend(loc="upper left", fontsize=9)
        ax_a.grid(True, linestyle=":", alpha=0.5, axis="y")

        # -------------------------------------------------------------
        # Panel B: Champion Regressor Predicted vs. Observed Scatter
        # -------------------------------------------------------------
        ax_b = axes[0, 1]
        champ_preds = self.regression_benchmarks[self.champion_regressor_name]["test_predictions"]
        y_test = self.y_reg_test

        ax_b.scatter(y_test, champ_preds, alpha=0.5, color="#2ca02c", edgecolors="none", s=35)
        min_val = min(float(np.min(y_test)), float(np.min(champ_preds)))
        max_val = max(float(np.max(y_test)), float(np.max(champ_preds)))
        ax_b.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Identity Line (y = x)")

        # Trend line
        slope, intercept, r_val, _, _ = stats.linregress(y_test, champ_preds)
        x_trend = np.linspace(min_val, max_val, 100)
        ax_b.plot(x_trend, intercept + slope * x_trend, "b-", linewidth=1.2, label=f"Fit (r = {r_val:.2f})")

        ax_b.set_xlabel("Observed Log Resolution Hours (ln[hrs + 1])", fontsize=10, fontweight="bold")
        ax_b.set_ylabel("Predicted Log Resolution Hours", fontsize=10, fontweight="bold")
        ax_b.set_title(f"B: {self.champion_regressor_name.title()} Predicted vs. Actual", fontsize=11, fontweight="bold")
        ax_b.legend(loc="upper left", fontsize=9)
        ax_b.grid(True, linestyle=":", alpha=0.5)

        # -------------------------------------------------------------
        # Panel C: Classifier Benchmark Comparison (Accuracy & Macro F1)
        # -------------------------------------------------------------
        ax_c = axes[1, 0]
        clf_names = list(self.classification_benchmarks.keys())
        acc_vals = [self.classification_benchmarks[m]["test_accuracy"] for m in clf_names]
        f1_vals = [self.classification_benchmarks[m]["test_macro_f1"] for m in clf_names]

        clean_clf_names = [m.replace("_", " ").title() for m in clf_names]
        x_clf = np.arange(len(clf_names))

        ax_c.bar(x_clf - width / 2, acc_vals, width, label="Test Accuracy", color="#2ca02c", edgecolor="black", alpha=0.85)
        ax_c.bar(x_clf + width / 2, f1_vals, width, label="Test Macro F1", color="#9467bd", edgecolor="black", alpha=0.85)
        ax_c.set_xticks(x_clf)
        ax_c.set_xticklabels(clean_clf_names, rotation=15, ha="right", fontsize=9)
        ax_c.set_ylabel("Score (0.0 – 1.0)", fontsize=10, fontweight="bold")
        ax_c.set_title(f"C: 4-Tier Classifier Benchmarks (Champion: {self.champion_classifier_name.title()})", fontsize=11, fontweight="bold")
        ax_c.set_ylim(0, 0.7)
        ax_c.legend(loc="upper left", fontsize=9)
        ax_c.grid(True, linestyle=":", alpha=0.5, axis="y")

        # -------------------------------------------------------------
        # Panel D: Champion Classifier Confusion Matrix Heatmap
        # -------------------------------------------------------------
        ax_d = axes[1, 1]
        cm = np.array(self.classification_benchmarks[self.champion_classifier_name]["confusion_matrix"])
        # Row-normalized percentages
        cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1.0)

        im = ax_d.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04, label="Row Normalized Recall")

        ax_d.set_xticks(np.arange(len(SPEED_TIER_ORDER)))
        ax_d.set_yticks(np.arange(len(SPEED_TIER_ORDER)))
        ax_d.set_xticklabels(SPEED_TIER_ORDER, fontsize=9)
        ax_d.set_yticklabels(SPEED_TIER_ORDER, fontsize=9)
        ax_d.set_xlabel("Predicted Velocity Bracket", fontsize=10, fontweight="bold")
        ax_d.set_ylabel("Actual Velocity Bracket", fontsize=10, fontweight="bold")
        ax_d.set_title(f"D: {self.champion_classifier_name.title()} Confusion Matrix", fontsize=11, fontweight="bold")

        # Text labels in each cell
        for i in range(len(SPEED_TIER_ORDER)):
            for j in range(len(SPEED_TIER_ORDER)):
                count = cm[i, j]
                pct = cm_norm[i, j] * 100.0
                text_color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax_d.text(
                    j, i, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center", color=text_color, fontsize=8, fontweight="bold"
                )

        plt.tight_layout()
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close()
        return out_file

    def run_full_pipeline(
        self,
        plot_path: Optional[Union[str, Path]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Executes entire baseline training and benchmarking pipeline end-to-end."""
        self.load_and_split_data()
        self.train_regressors()
        self.train_classifiers()
        saved_files = self.save_models()
        manifest_file = self.export_benchmark_manifest(manifest_path)
        plot_file = self.generate_figures(plot_path)

        return {
            "champion_regressor": self.champion_regressor_name,
            "champion_classifier": self.champion_classifier_name,
            "regression_benchmarks": self.regression_benchmarks,
            "classification_benchmarks": self.classification_benchmarks,
            "saved_models": {k: str(v) for k, v in saved_files.items()},
            "manifest_path": str(manifest_file),
            "plot_path": str(plot_file),
        }


if __name__ == "__main__":
    trainer = ResolutionBaselineTrainer()
    res = trainer.run_full_pipeline()
    print("=" * 70)
    print("PHASE 8, STEP 8.5 BASELINE MODEL TRAINING COMPLETED")
    print(f"Champion Regressor:  {res['champion_regressor']} (Test R²: {res['regression_benchmarks'][res['champion_regressor']]['test_r2']:.4f})")
    print(f"Champion Classifier: {res['champion_classifier']} (Test Macro F1: {res['classification_benchmarks'][res['champion_classifier']]['test_macro_f1']:.4f}, Acc: {res['classification_benchmarks'][res['champion_classifier']]['test_accuracy']:.4f})")
    print(f"Saved Models: {res['saved_models']}")
    print(f"Manifest exported: {res['manifest_path']}")
    print(f"Figure 32 generated: {res['plot_path']}")
    print("=" * 70)
