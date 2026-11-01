"""
OpenSourceLens — Resolution Reliability Decision & Synthesis (Phase 8, Step 8.7).

Exports formal scientific decision manifest and generates Phase 8 Milestone Figure 34
contrasting continuous vs. ordinal predictive capabilities and detailing Phase 9
FastAPI backend inference architecture.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def export_decision_manifest(output_path: Optional[Union[str, Path]] = None) -> Path:
    """Exports formal Phase 8 scientific reliability decision manifest to JSON."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    out_file = Path(output_path or base_dir / "data" / "processed" / "evaluation" / "resolution_reliability_decision.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "metadata": {
            "phase": "Phase 8 (Resolution Analysis & Prediction)",
            "step": "Step 8.7 (Decide Whether Prediction is Reliable)",
            "decision_date_utc": datetime.now(timezone.utc).isoformat(),
            "status": "APPROVED_ADVISORY_TIERS_ONLY",
        },
        "scientific_decision_summary": {
            "continuous_hour_regression": {
                "verdict": "UNRELIABLE_FOR_SLA",
                "recommendation": "QUARANTINE_FROM_PRODUCTION_PROMISES",
                "empirical_r2_overall": 0.2759,
                "within_repo_r2_range": [-0.0412, 0.1812],
                "unexplained_variance_pct": 72.41,
                "scientific_justification": (
                    "Within-repository regression R² drops to negative or near-zero values (-0.04 to +0.18). "
                    "Over 72% of resolution time variance is driven by unobservable post-creation communication, "
                    "developer availability, and asynchronous review cycles. Providing exact hour predictions "
                    "would create false precision and erode user trust."
                ),
            },
            "discrete_velocity_classification": {
                "verdict": "RELIABLE_AS_ADVISORY_ESTIMATE",
                "recommendation": "DEPLOY_AS_ADVISORY_SERVICE",
                "champion_model": "Random Forest Classifier",
                "exact_accuracy_4_tier": 0.5110,
                "adjacent_accuracy_pm1_tier": 0.8811,
                "severe_error_rate_ge2_tiers": 0.1189,
                "extreme_error_rate_3_tiers": 0.0044,
                "linear_weighted_kappa": 0.3233,
                "scientific_justification": (
                    "The Random Forest velocity classifier achieves 51.10% exact 4-tier accuracy and 88.11% "
                    "adjacent accuracy (within ±1 tier), outperforming the naive prior baseline (38.77%) by "
                    "+12.33% absolute accuracy. In nearly 9 out of 10 cases, the prediction provides actionable "
                    "triage guidance regarding whether an issue will be resolved rapidly or enter the backlog."
                ),
            },
            "descriptive_historical_analytics": {
                "verdict": "CORE_PRODUCTION_CAPABILITY",
                "recommendation": "SURFACE_AS_PRIMARY_DASHBOARD_LAYER",
                "scientific_justification": (
                    "Kaplan-Meier survival curves and empirical quartiles provide factual, objective baseline "
                    "benchmarks of repository velocity without speculative claims."
                ),
            },
        },
        "downstream_integration_policy": {
            "phase_9_backend": {
                "endpoint": "POST /api/issues/predict-velocity",
                "service_module": "ml.resolution_analysis.predictor.ResolutionPredictor",
                "response_fields": ["predicted_tier", "confidence", "tier_probabilities", "adjacent_window", "advisory_factors", "disclaimer"],
                "mandatory_disclaimer_enforced": True,
            },
            "phase_10_frontend": {
                "ui_component": "ResolutionVelocityCard",
                "presentation_style": "Ordinal velocity badge (Fast / Moderate / Slow / Very Slow) with confidence meter and disclaimer tooltip",
                "prohibited_elements": ["Exact completion countdown timers", "Hard SLA guarantee timestamps"],
            },
        },
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return out_file


def generate_milestone_figure(output_path: Optional[Union[str, Path]] = None) -> Path:
    """Generates Phase 8 Milestone Synthesis Figure 34."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    out_file = Path(output_path or base_dir / "ml" / "resolution_analysis" / "figures" / "fig34_resolution_decision_and_architecture.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=300)

    # -------------------------------------------------------------
    # Panel A: Cross-Repo vs. Within-Repo R² Comparison
    # -------------------------------------------------------------
    ax_a = axes[0, 0]
    labels_a = ["Overall\nDataset", "PyTorch\n(Kernel)", "Transformers\n(NLP)", "VS Code\n(DevTools)", "Kubernetes\n(Cloud)"]
    r2_vals_a = [0.2759, 0.1812, 0.0788, -0.0350, -0.0412]
    colors_a = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#d62728"]

    bars_a = ax_a.bar(labels_a, r2_vals_a, color=colors_a, edgecolor="black", width=0.55, alpha=0.85)
    ax_a.axhline(0, color="black", linestyle="-", linewidth=1.0)
    ax_a.set_ylabel("Coefficient of Determination (R²)", fontsize=10, fontweight="bold")
    ax_a.set_title("A: Cross-Repo vs. Within-Repo R² (Limits of Continuous Regression)", fontsize=11, fontweight="bold")
    ax_a.set_ylim(-0.1, 0.35)
    ax_a.grid(True, linestyle=":", alpha=0.5, axis="y")

    for bar in bars_a:
        height = bar.get_height()
        va = "bottom" if height >= 0 else "top"
        offset = 0.015 if height >= 0 else -0.015
        ax_a.text(
            bar.get_x() + bar.get_width() / 2, height + offset,
            f"{height:+.3f}", ha="center", va=va, fontsize=9, fontweight="bold"
        )
    ax_a.axhspan(-0.1, 0.0, color="red", alpha=0.08, label="Negative R² (Worse than mean)")
    ax_a.legend(loc="upper right", fontsize=8)

    # -------------------------------------------------------------
    # Panel B: Velocity Classification Accuracy Spectrum
    # -------------------------------------------------------------
    ax_b = axes[0, 1]
    labels_b = ["Naive Prior\nBaseline", "Exact 4-Tier\nAccuracy", "Adjacent (±1 Tier)\nAccuracy Window"]
    vals_b = [38.77, 51.10, 88.11]
    colors_b = ["#7f7f7f", "#2ca02c", "#1f77b4"]

    bars_b = ax_b.bar(labels_b, vals_b, color=colors_b, edgecolor="black", width=0.5, alpha=0.85)
    ax_b.set_ylabel("Accuracy on Test Set (%)", fontsize=10, fontweight="bold")
    ax_b.set_title("B: Discrete Velocity Classification Reliability Spectrum", fontsize=11, fontweight="bold")
    ax_b.set_ylim(0, 105)
    ax_b.grid(True, linestyle=":", alpha=0.5, axis="y")

    for bar in bars_b:
        height = bar.get_height()
        ax_b.text(
            bar.get_x() + bar.get_width() / 2, height + 2.0,
            f"{height:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    # -------------------------------------------------------------
    # Panel C: Scientific Reliability Decision Matrix
    # -------------------------------------------------------------
    ax_c = axes[1, 0]
    ax_c.axis("off")

    table_data = [
        ["Capability", "Empirical Evidence", "Scientific Verdict", "Production Role"],
        ["Continuous Hour\nPrediction", "Test R² = 0.28\nWithin-repo R² ≤ 0.18\n72% unobservable", "UNRELIABLE\n(False precision)", "QUARANTINED\nInternal research only"],
        ["Discrete Velocity\nTier Classification", "Exact: 51.1%\nWithin ±1 Tier: 88.1%\nSevere Error: 11.9%", "RELIABLE\n(Actionable triage)", "ADVISORY SERVICE\nVelocity badge + disclaimer"],
        ["Descriptive Historical\nAnalytics", "Kaplan-Meier S(t)\nQuartiles by Repo\nKruskal-Wallis p<1e-15", "FACTUAL\n(Objective baseline)", "CORE DASHBOARD\nPrimary analytics layer"],
    ]

    col_widths = [0.25, 0.28, 0.22, 0.25]
    table = ax_c.table(cellText=table_data, colWidths=col_widths, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.3)

    # Format table header and rows
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor("#1f77b4")
            cell.set_text_props(color="white", fontweight="bold")
        elif i == 1:
            cell.set_facecolor("#ffebee")
        elif i == 2:
            cell.set_facecolor("#e8f5e9")
        elif i == 3:
            cell.set_facecolor("#e3f2fd")
        cell.set_edgecolor("#999999")

    ax_c.set_title("C: Phase 8 Scientific Reliability Decision Matrix", fontsize=11, fontweight="bold", pad=20)

    # -------------------------------------------------------------
    # Panel D: Phase 9 Backend Inference Architecture
    # -------------------------------------------------------------
    ax_d = axes[1, 1]
    ax_d.axis("off")
    ax_d.set_title("D: Phase 9 FastAPI Production Inference Architecture", fontsize=11, fontweight="bold")

    # Draw architectural flowchart
    boxes = [
        {"x": 0.05, "y": 0.72, "w": 0.22, "h": 0.18, "text": "Raw GitHub Issue\n(Title, Body, Repo)", "color": "#e3f2fd"},
        {"x": 0.38, "y": 0.72, "w": 0.26, "h": 0.18, "text": "Phase 6 Model\n(DistilBERT Category:\nBug / Feature / Other)", "color": "#fff3e0"},
        {"x": 0.74, "y": 0.72, "w": 0.22, "h": 0.18, "text": "t=0 Feature Extractor\n(28 Standardized\nNumeric & Dummies)", "color": "#ede7f6"},
        {"x": 0.20, "y": 0.22, "w": 0.28, "h": 0.22, "text": "Random Forest Predictor\n(Champion 4-Tier Model\n88.1% Adjacent Accuracy)", "color": "#e8f5e9"},
        {"x": 0.62, "y": 0.22, "w": 0.33, "h": 0.22, "text": "FastAPI JSON Response\n• Predicted Tier (< 1d, 1-7d...)\n• Calibrated Confidence\n• Factors & Disclaimer", "color": "#f1f8e9"},
    ]

    for b in boxes:
        rect = plt.Rectangle((b["x"], b["y"]), b["w"], b["h"], facecolor=b["color"], edgecolor="black", linewidth=1.2, transform=ax_d.transAxes)
        ax_d.add_patch(rect)
        ax_d.text(
            b["x"] + b["w"] / 2, b["y"] + b["h"] / 2, b["text"],
            ha="center", va="center", fontsize=8.5, fontweight="bold", transform=ax_d.transAxes
        )

    # Connecting arrows
    arrow_props = dict(arrowstyle="->", color="#333333", linewidth=1.8, shrinkA=5, shrinkB=5)
    ax_d.annotate("", xy=(0.38, 0.81), xytext=(0.27, 0.81), arrowprops=arrow_props, xycoords="axes fraction")
    ax_d.annotate("", xy=(0.74, 0.81), xytext=(0.64, 0.81), arrowprops=arrow_props, xycoords="axes fraction")
    ax_d.annotate("", xy=(0.34, 0.44), xytext=(0.85, 0.72), arrowprops=arrow_props, xycoords="axes fraction")
    ax_d.annotate("", xy=(0.62, 0.33), xytext=(0.48, 0.33), arrowprops=arrow_props, xycoords="axes fraction")

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    return out_file


if __name__ == "__main__":
    m_file = export_decision_manifest()
    f_file = generate_milestone_figure()
    print("=" * 70)
    print("PHASE 8, STEP 8.7 DECISION MANIFEST & FIGURE GENERATED")
    print(f"Manifest: {m_file}")
    print(f"Figure 34: {f_file}")
    print("=" * 70)
