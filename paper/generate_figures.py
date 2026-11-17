"""Generate publication-grade IEEE conference paper figures directly from
verified evaluation artifacts in OpenSourceLens.

Outputs both vector PDF (for LaTeX/Overleaf) and 300-DPI PNG (for visual QA)
into paper/figures/.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "processed" / "evaluation"
MODELS_DIR = ROOT / "models"
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size": 8.2,
    "axes.titlesize": 9.0,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.2,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.2,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Colorblind-safe Okabe-Ito inspired publication palette
C_NAVY = "#1B4965"
C_TEAL = "#2A9D8F"
C_AMBER = "#D95F3B"
C_GOLD = "#E9C46A"
C_SLATE = "#64748B"
C_PURPLE = "#6D597A"


def save_fig(fig: plt.Figure, stem: str) -> None:
    pdf_path = OUT_DIR / f"{stem}.pdf"
    png_path = OUT_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.06, dpi=300)
    plt.close(fig)
    print(f"Saved {pdf_path.name} and {png_path.name}")


def generate_fig1_framework() -> None:
    """Figure 1: Methodological & Empirical Framework Architecture (IEEE 2-column)."""
    fig, ax = plt.subplots(figsize=(9.2, 3.1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 36)
    ax.axis("off")

    def draw_box(x, y, w, h, title, lines, fc="#FFFFFF", ec=C_NAVY, lw=1.15, title_color=C_NAVY):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.35,rounding_size=0.9",
            facecolor=fc, edgecolor=ec, linewidth=lw
        )
        ax.add_patch(box)
        # Header divider band
        ax.plot([x + 0.5, x + w - 0.5], [y + h - 4.2, y + h - 4.2], color=ec, lw=0.6, alpha=0.45)
        ax.text(
            x + w / 2, y + h - 2.1, title,
            ha="center", va="center", fontsize=6.9, fontweight="bold", color=title_color
        )
        body = "\n".join(lines)
        ax.text(
            x + 1.1, y + h - 5.4, body,
            ha="left", va="top", fontsize=6.4, color="#1E293B", linespacing=1.42
        )

    def draw_arrow(x1, y1, x2, y2, color="#334155"):
        arr = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>,head_length=5.0,head_width=2.8",
            color=color, linewidth=1.2, mutation_scale=1.0
        )
        ax.add_patch(arr)

    # Stage 1: Acquisition
    draw_box(
        0.8, 5.2, 22.0, 29.0,
        "1. Rate-Budgeted Ingestion",
        [
            "• Any Public GitHub Repo",
            "• Eligibility & Non-Empty Gate",
            "• Async REST Harvester",
            "  (10-req / 50-req Budgets)",
            "• Commits, PRs, Reviews,",
            "  Pure Issues, Releases/Tags",
            "• Lease-Fenced Job Queue"
        ],
        fc="#EEF5F9", ec=C_NAVY, title_color=C_NAVY
    )

    # Stage 2: Snapshot Isolation
    draw_box(
        25.8, 5.2, 19.5, 29.0,
        "2. Snapshot & Isolation",
        [
            "• Relational Persistence",
            "  (SQLite WAL / PG 16)",
            "• Monotonic Source",
            "  Revision Increment",
            "• Frozen UTC ref_time",
            "• Strict t = 0 Feature",
            "  Leakage Quarantine"
        ],
        fc="#F4F6F8", ec=C_SLATE, title_color="#2C3E50"
    )

    # Stage 3A: Guarded Statistical Engine (Top)
    draw_box(
        48.5, 20.5, 24.5, 13.7,
        "3A. Guarded Statistics",
        [
            "• Activity: MAD & Change-Points",
            "• Contributors: Gini, Lorenz, BF",
            "• PRs & Issues: KM Survival",
            "• Releases: CV_rel & SemVer"
        ],
        fc="#EBF7F5", ec=C_TEAL, title_color="#176158"
    )

    # Stage 3B: Generalizable ML / NLP (Bottom)
    draw_box(
        48.5, 5.2, 24.5, 13.7,
        "3B. Generalizable ML / NLP",
        [
            "• 5-Class Classifier + Abstention",
            "• Dense Retrieval (MiniLM-L6)",
            "• Cosine Threshold Calibration",
            "• Econometric SLA Feasibility"
        ],
        fc="#FDF2EE", ec=C_AMBER, title_color="#A83E21"
    )

    # Stage 4: Synthesis & Guardrails
    draw_box(
        76.2, 5.2, 23.0, 29.0,
        "4. Scientific Guardrails",
        [
            "• Strict NULL != 0.0 Policy",
            "• Confidence Penalty Matrix",
            "  (No Opaque 0–100 Scores)",
            "• Monthly Cross-Domain",
            "  Spearman (B = 10,000)",
            "• |r_AR1| > 0.40 Guard",
            "• Benjamini-Hochberg FDR"
        ],
        fc="#F5F0F7", ec=C_PURPLE, title_color="#4A3556"
    )

    # Arrows
    draw_arrow(23.1, 19.7, 25.5, 19.7)
    draw_arrow(45.6, 25.5, 48.2, 27.3)
    draw_arrow(45.6, 14.0, 48.2, 12.0)
    draw_arrow(73.3, 27.3, 75.9, 24.5)
    draw_arrow(73.3, 12.0, 75.9, 15.0)

    # Footer caption bar
    ax.text(
        50.0, 1.4,
        "OpenSourceLens Reference Methodology (v2026.2)  |  Zero Post-Creation Leakage  |  Validated on 4 Benchmark Ecosystems (N = 2,328) + 3 Unseen Live Repositories",
        ha="center", va="center", fontsize=6.9, style="italic", color="#334155"
    )

    save_fig(fig, "fig1_framework")


def generate_fig2_classification_loro() -> None:
    """Figure 2: (a) DistilBERT Length Truncation vs. Classical Baseline; (b) LORO Abstention."""
    with open(EVAL_DIR / "model_comparison_benchmark.json", encoding="utf-8") as f:
        comp = json.load(f)
    with open(EVAL_DIR / "transformer_error_analysis.json", encoding="utf-8") as f:
        err = json.load(f)
    with open(MODELS_DIR / "generalizable_issue_classifier_manifest.json", encoding="utf-8") as f:
        loro = json.load(f)

    classical_acc = comp["models"][1]["accuracy"] * 100.0  # 81.56%
    distilbert_acc = comp["models"][2]["accuracy"] * 100.0  # 75.32%

    quartiles = err["length_and_truncation_impact"]["length_quartile_analysis"]
    q_labels = ["Q1\n(17–127w)", "Q2\n(129–205w)", "Q3\n(208–388w)", "Q4\n(390–1708w)"]
    q_keys = ["Q1 (Short)", "Q2 (Medium-Low)", "Q3 (Medium-High)", "Q4 (Long)"]
    q_accs = [quartiles[k]["accuracy_pct"] for k in q_keys]

    loro_summary = loro["loro_validation_summary"]
    repo_order = [
        ("microsoft/vscode", "VS Code"),
        ("pytorch/pytorch", "PyTorch"),
        ("huggingface/transformers", "HF Transf."),
        ("kubernetes/kubernetes", "Kubernetes"),
    ]
    repo_names = [lbl for _, lbl in repo_order]
    raw_accs = [loro_summary[k]["accuracy"] * 100.0 for k, _ in repo_order]
    conf_accs = [loro_summary[k]["confident_accuracy"] * 100.0 for k, _ in repo_order]
    covs = [loro_summary[k]["confident_coverage"] * 100.0 for k, _ in repo_order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.0), gridspec_kw={"wspace": 0.30})

    # Panel (a)
    x = np.arange(len(q_labels))
    bars = ax1.bar(x, q_accs, width=0.50, color=C_NAVY, alpha=0.90, edgecolor="black", linewidth=0.5, label="DistilBERT (256-tok max)")
    ax1.axhline(classical_acc, color=C_TEAL, linestyle="--", linewidth=1.4, label=f"Platt LinearSVC ({classical_acc:.2f}%)")
    ax1.axhline(distilbert_acc, color=C_AMBER, linestyle=":", linewidth=1.4, label=f"DistilBERT Overall ({distilbert_acc:.2f}%)")
    for bar, val in zip(bars, q_accs):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 1.0, f"{val:.1f}%", ha="center", va="bottom", fontsize=7.4, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(q_labels, fontsize=7.2)
    ax1.set_ylim(52, 91)
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title("(a) DistilBERT Decay by Length Quartile", pad=8)
    ax1.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.94, edgecolor="#CBD5E1", fontsize=6.9)

    # Panel (b)
    x2 = np.arange(len(repo_names))
    w = 0.36
    b1 = ax2.bar(x2 - w / 2, raw_accs, width=w, color=C_SLATE, alpha=0.82, edgecolor="black", linewidth=0.5, label="Raw OOD Acc.")
    b2 = ax2.bar(x2 + w / 2, conf_accs, width=w, color=C_TEAL, alpha=0.92, edgecolor="black", linewidth=0.5, label="Confident Acc. (Abstention)")
    for bar, val in zip(b1, raw_accs):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 1.3, f"{val:.1f}%", ha="center", va="bottom", fontsize=6.8)
    for bar, val, cov in zip(b2, conf_accs, covs):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 1.3, f"{val:.1f}%\n({cov:.0f}% cov)", ha="center", va="bottom", fontsize=6.4, fontweight="bold", color="#124E45")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(repo_names, fontsize=7.4)
    ax2.set_ylim(0, 96)
    ax2.set_ylabel("Held-Out Repo Accuracy (%)")
    ax2.set_title("(b) Strict Leave-One-Repo-Out Transfer", pad=8)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.94, edgecolor="#CBD5E1", fontsize=6.9)

    save_fig(fig, "fig2_classification_loro")


def generate_fig3_retrieval_calibration() -> None:
    """Figure 3: (a) Dense vs. TF-IDF Retrieval Benchmarks; (b) Threshold Calibration."""
    with open(EVAL_DIR / "retrieval_evaluation_benchmark.json", encoding="utf-8") as f:
        ret = json.load(f)
    with open(EVAL_DIR / "similarity_threshold_benchmark.json", encoding="utf-8") as f:
        thr = json.load(f)

    dense = ret["sentence_transformers_all_MiniLM_L6_v2"]
    lex = ret["classical_lexical_tfidf_baseline"]

    metrics = ["MRR", "Hit@1", "Hit@3", "Hit@5", "Hit@10", "NDCG@5"]
    keys = ["mrr", "hit_rate_at_1", "hit_rate_at_3", "hit_rate_at_5", "hit_rate_at_10", "ndcg_at_5"]
    dense_vals = [dense[k] * 100.0 for k in keys]
    lex_vals = [lex[k] * 100.0 for k in keys]

    op = thr["operating_points"]
    rej_vals = [
        op["low_weak"]["noise_rejection_rate"] * 100.0,
        op["medium_related"]["noise_rejection_rate"] * 100.0,
        op["high_duplicate"]["noise_rejection_rate"] * 100.0,
    ]
    conc_vals = [
        op["low_weak"]["label_concordance"] * 100.0,
        op["medium_related"]["label_concordance"] * 100.0,
        op["high_duplicate"]["label_concordance"] * 100.0,
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 2.95), gridspec_kw={"wspace": 0.30})

    # Panel (a)
    x = np.arange(len(metrics))
    w = 0.37
    b1 = ax1.bar(x - w / 2, lex_vals, width=w, color=C_SLATE, alpha=0.80, edgecolor="black", linewidth=0.5, label="Lexical TF-IDF")
    b2 = ax1.bar(x + w / 2, dense_vals, width=w, color=C_NAVY, alpha=0.92, edgecolor="black", linewidth=0.5, label="all-MiniLM-L6-v2")
    for idx, (bar1, v1, bar2, v2) in enumerate(zip(b1, lex_vals, b2, dense_vals)):
        ax1.text(bar1.get_x() + bar1.get_width() / 2 - 0.03, v1 + 0.9, f"{v1:.1f}", ha="center", va="bottom", fontsize=6.2, color="#475569")
        ax1.text(bar2.get_x() + bar2.get_width() / 2 + 0.04, v2 + 0.9, f"{v2:.1f}", ha="center", va="bottom", fontsize=6.4, fontweight="bold", color=C_NAVY)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=7.0)
    ax1.set_ylim(48, 92)
    ax1.set_ylabel("Retrieval Score (%)")
    ax1.set_title("(a) Dense vs. Lexical Retrieval (N = 120)", pad=8)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.94, edgecolor="#CBD5E1", fontsize=7.0)

    # Panel (b)
    x_t = np.arange(3)
    t_labels = ["Low / Weak\n(cos ≥ 0.45)", "Medium Related\n(cos ≥ 0.55)", "High Duplicate\n(cos ≥ 0.70)"]
    ax2.plot(x_t, rej_vals, marker="s", linewidth=1.8, markersize=6, color=C_TEAL, label="Noise Rejection Rate (%)")
    ax2.plot(x_t, conc_vals, marker="o", linewidth=1.8, markersize=6, color=C_AMBER, label="Label Concordance (%)")
    for xi, r, c in zip(x_t, rej_vals, conc_vals):
        ax2.text(xi, r + 1.8, f"{r:.2f}%", ha="center", va="bottom", fontsize=7.0, fontweight="bold", color="#176158")
        ax2.text(xi, c - 4.2, f"{c:.2f}%", ha="center", va="top", fontsize=7.0, fontweight="bold", color="#A83E21")
    ax2.axvspan(1.65, 2.35, color=C_GOLD, alpha=0.20, label="Duplicate Alert Band")
    ax2.set_xticks(x_t)
    ax2.set_xticklabels(t_labels, fontsize=7.1)
    ax2.set_xlim(-0.35, 2.35)
    ax2.set_ylim(45, 108)
    ax2.set_ylabel("Operating Metric (%)")
    ax2.set_title("(b) Cosine Similarity Operating Points", pad=8)
    ax2.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.94, edgecolor="#CBD5E1", fontsize=6.8)

    save_fig(fig, "fig3_retrieval_calibration")


def generate_fig4_resolution_feasibility() -> None:
    """Figure 4: (a) ANOVA Type-III Attribution; (b) Pooled vs. Within-Repo R2 Collapse."""
    with open(EVAL_DIR / "resolution_factors_analysis.json", encoding="utf-8") as f:
        fac = json.load(f)
    with open(EVAL_DIR / "resolution_baseline_models.json", encoding="utf-8") as f:
        base = json.load(f)
    with open(EVAL_DIR / "resolution_evaluation_diagnostics.json", encoding="utf-8") as f:
        diag = json.load(f)

    pooled_r2 = base["regression_benchmarks"]["ridge"]["test_r2"]
    by_repo = diag["subgroup_performance"]["by_repository"]
    r2_labels = ["Pooled\n(4 Repos)", "PyTorch", "HF Transf.", "VS Code", "Kubernetes"]
    r2_vals = [
        pooled_r2,
        by_repo["pytorch/pytorch"]["test_r2"],
        by_repo["huggingface/transformers"]["test_r2"],
        by_repo["microsoft/vscode"]["test_r2"],
        by_repo["kubernetes/kubernetes"]["test_r2"],
    ]
    r2_colors = [C_NAVY, C_TEAL, C_TEAL, C_AMBER, C_AMBER]

    families = fac["anova_variance_decomposition"]["factor_families"]
    fam_labels = [
        "Repository\nGovernance***",
        "Structural\nComplexity*",
        "Technical\nEvidence**",
        "Issue\nTypology*",
        "Temporal &\nAuthor (n.s.)"
    ]
    fam_pcts = [f["pct_explained_variance"] for f in families]
    fam_colors = [C_AMBER, C_NAVY, C_TEAL, C_PURPLE, C_SLATE]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.0), gridspec_kw={"wspace": 0.34})

    # Panel (a)
    y_pos = np.arange(len(fam_labels))[::-1]
    bars1 = ax1.barh(y_pos, fam_pcts, height=0.58, color=fam_colors, alpha=0.90, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars1, fam_pcts):
        ax1.text(val + 1.3, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", ha="left", fontsize=7.4, fontweight="bold")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(fam_labels, fontsize=7.2)
    ax1.set_xlim(0, 88)
    ax1.set_xlabel("Share of Explained Variance (%)")
    ax1.set_title("(a) ANOVA Attribution of ln(Hours + 1)", pad=8)
    ax1.text(
        33.5, 0.25,
        "OLS R² = 0.252\nF = 13.33, p < 1e-16\n(*p<.05, **p<.01, ***p<.001)",
        fontsize=6.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="#CBD5E1")
    )

    # Panel (b)
    x2 = np.arange(len(r2_labels))
    bars2 = ax2.bar(x2, r2_vals, width=0.50, color=r2_colors, alpha=0.90, edgecolor="black", linewidth=0.5)
    ax2.axhline(0.0, color="black", linewidth=0.9)
    for bar, val in zip(bars2, r2_vals):
        va = "bottom" if val >= 0 else "top"
        offset = 0.012 if val >= 0 else -0.013
        ax2.text(bar.get_x() + bar.get_width() / 2, val + offset, f"{val:+.3f}", ha="center", va=va, fontsize=7.1, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(r2_labels, fontsize=7.1)
    ax2.set_ylim(-0.11, 0.35)
    ax2.set_ylabel("Holdout Test R² (Ridge, t = 0)")
    ax2.set_title("(b) Pooled vs. Within-Repo R² Collapse", pad=8)
    ax2.annotate(
        "Within-repo R² collapses\n(72.4% variance unexplained)",
        xy=(3.5, -0.042), xytext=(1.55, 0.25),
        fontsize=6.8, color="#8C2D19", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#8C2D19", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF5F0", edgecolor=C_AMBER, lw=0.7)
    )

    save_fig(fig, "fig4_resolution_feasibility")


if __name__ == "__main__":
    generate_fig1_framework()
    generate_fig2_classification_loro()
    generate_fig3_retrieval_calibration()
    generate_fig4_resolution_feasibility()
    print("All 4 IEEE figures regenerated successfully.")
