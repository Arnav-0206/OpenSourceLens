"""
OpenSourceLens — Similarity Threshold Calibrator (Phase 7, Step 7.5)
Establishes, calibrates, and validates empirical multi-tiered decision thresholds
for related-issue detection and duplicate identification.
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.related_issues.similarity import SimilarityEngine
from ml.related_issues.store import IssueEmbeddingStore


DEFAULT_THRESHOLDS = {
    "high": 0.70,     # Potential Duplicate / Near-identical
    "medium": 0.55,   # Semantically Related / Shared subsystem
    "low": 0.45,      # Weak Match / Topical overlap
}


class SimilarityThresholdCalibrator:
    """Calibrates, validates, and applies multi-tiered similarity thresholds for issue retrieval."""

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = thresholds.copy() if thresholds is not None else DEFAULT_THRESHOLDS.copy()

    def classify_score(
        self,
        score: float,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Classify a similarity score into a calibrated confidence tier.

        Args:
            score: Cosine similarity score in [-1.0, 1.0].
            thresholds: Optional override for threshold boundaries.

        Returns:
            Dictionary with tier, label, confidence, is_related flag, and action recommendation.
        """
        th = thresholds if thresholds is not None else self.thresholds
        t_high = th.get("high", 0.70)
        t_med = th.get("medium", 0.55)
        t_low = th.get("low", 0.45)

        if score >= t_high:
            return {
                "tier": "HIGH",
                "label": "Potential Duplicate",
                "confidence": "High",
                "is_related": True,
                "recommendation": "Strong duplicate candidate; consider linking or closing as duplicate.",
            }
        elif score >= t_med:
            return {
                "tier": "MEDIUM",
                "label": "Semantically Related",
                "confidence": "Medium",
                "is_related": True,
                "recommendation": "Shares core components or error context; reference in issue discussion.",
            }
        elif score >= t_low:
            return {
                "tier": "LOW",
                "label": "Weak Match",
                "confidence": "Low",
                "is_related": True,
                "recommendation": "Incidental topical overlap; display with caution as a weak lead.",
            }
        else:
            return {
                "tier": "UNRELATED",
                "label": "Unrelated",
                "confidence": "None",
                "is_related": False,
                "recommendation": "Below noise threshold; discard to prevent alert fatigue.",
            }

    def filter_and_enrich_candidates(
        self,
        candidates: List[Dict[str, Any]],
        min_threshold: Optional[float] = None,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Enrich candidates with confidence tiers and filter out scores below min_threshold.

        Args:
            candidates: List of candidate dictionaries from RelatedIssueRetriever.
            min_threshold: Minimum score required to keep candidate (defaults to low threshold).
            thresholds: Optional threshold boundary dict.

        Returns:
            Filtered and enriched candidate list.
        """
        th = thresholds if thresholds is not None else self.thresholds
        cutoff = min_threshold if min_threshold is not None else th.get("low", 0.45)

        enriched: List[Dict[str, Any]] = []
        for cand in candidates:
            score = cand.get("similarity_score", 0.0)
            classification = self.classify_score(score, thresholds=th)

            if score >= cutoff:
                cand_copy = cand.copy()
                cand_copy.update({
                    "confidence_tier": classification["tier"],
                    "tier_label": classification["label"],
                    "confidence": classification["confidence"],
                    "is_related": classification["is_related"],
                    "action_recommendation": classification["recommendation"],
                })
                enriched.append(cand_copy)

        return enriched

    def sweep_threshold_tradeoffs(
        self,
        store: IssueEmbeddingStore,
        sample_size: int = 500,
        threshold_steps: Optional[np.ndarray] = None,
        random_seed: int = 42,
    ) -> pd.DataFrame:
        """
        Sweep candidate similarity thresholds from 0.30 to 0.85 to profile
        inter-repo noise rejection, intra-repo candidate retention, and label concordance.

        Args:
            store: Loaded IssueEmbeddingStore.
            sample_size: Number of issues to sample for pairwise sweep.
            threshold_steps: Array of thresholds to evaluate.
            random_seed: Random seed for sampling.

        Returns:
            DataFrame containing trade-off metrics at each threshold.
        """
        if not store.is_loaded or store.embeddings is None or store.metadata is None:
            raise RuntimeError("IssueEmbeddingStore is not loaded.")

        steps = (
            threshold_steps
            if threshold_steps is not None
            else np.arange(0.30, 0.86, 0.025)
        )

        n_total = len(store.embeddings)
        n = min(sample_size, n_total)

        np.random.seed(random_seed)
        sample_idx = np.random.choice(n_total, size=n, replace=False)

        sub_embs = store.embeddings[sample_idx]
        sub_meta = store.metadata.iloc[sample_idx].reset_index(drop=True)

        engine = SimilarityEngine()
        sim_matrix = engine.batch_cosine_similarity(sub_embs, sub_embs, assume_normalized=True)

        i_idx, j_idx = np.triu_indices(n, k=1)
        sim_values = sim_matrix[i_idx, j_idx]

        repos_i = sub_meta["repo_name"].iloc[i_idx].to_numpy()
        repos_j = sub_meta["repo_name"].iloc[j_idx].to_numpy()
        labels_i = sub_meta["canonical_label"].iloc[i_idx].to_numpy()
        labels_j = sub_meta["canonical_label"].iloc[j_idx].to_numpy()

        intra_mask = (repos_i == repos_j)
        inter_mask = ~intra_mask

        intra_sims = sim_values[intra_mask]
        inter_sims = sim_values[inter_mask]

        intra_labels_i = labels_i[intra_mask]
        intra_labels_j = labels_j[intra_mask]

        records: List[Dict[str, Any]] = []

        for tau in steps:
            tau_val = round(float(tau), 4)

            # Noise Rejection: fraction of cross-repo pairs correctly rejected (sim < tau)
            inter_rejected = np.sum(inter_sims < tau_val)
            noise_rejection_rate = inter_rejected / len(inter_sims) if len(inter_sims) > 0 else 1.0

            # Candidate Retention: fraction of intra-repo pairs meeting threshold (sim >= tau)
            intra_accepted = np.sum(intra_sims >= tau_val)
            intra_retention_rate = intra_accepted / len(intra_sims) if len(intra_sims) > 0 else 0.0

            # Label Concordance Proxy: among retained intra-repo pairs, what % share the same label?
            retained_mask = (intra_sims >= tau_val)
            if np.sum(retained_mask) > 0:
                retained_matches = np.sum(intra_labels_i[retained_mask] == intra_labels_j[retained_mask])
                label_concordance = retained_matches / np.sum(retained_mask)
            else:
                label_concordance = 1.0

            records.append({
                "threshold": tau_val,
                "noise_rejection_rate": round(float(noise_rejection_rate), 4),
                "intra_retention_rate": round(float(intra_retention_rate), 4),
                "retained_intra_count": int(intra_accepted),
                "label_concordance": round(float(label_concordance), 4),
            })

        return pd.DataFrame(records)

    def plot_threshold_calibration(
        self,
        tradeoff_df: pd.DataFrame,
        tier_counts: Dict[str, int],
        save_path: Union[str, Path] = "ml/related_issues/figures/fig27_similarity_threshold_calibration.png",
    ) -> Path:
        """
        Generate dual-panel visualization of threshold trade-offs and confidence tier distributions.
        """
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

        # Panel A: Trade-Off Curves
        taus = tradeoff_df["threshold"].to_numpy()
        rejection = tradeoff_df["noise_rejection_rate"].to_numpy() * 100.0
        retention = tradeoff_df["intra_retention_rate"].to_numpy() * 100.0
        concordance = tradeoff_df["label_concordance"].to_numpy() * 100.0

        ax1.plot(taus, rejection, color="#1f77b4", linewidth=2.2, label="Inter-Repo Noise Rejection (%)")
        ax1.plot(taus, retention, color="#ff7f0e", linewidth=2.2, label="Intra-Repo Candidate Retention (%)")
        ax1.plot(taus, concordance, color="#2ca02c", linestyle="--", linewidth=1.8, label="Label Concordance Proxy (%)")

        # Vertical band markers
        t_low = self.thresholds.get("low", 0.45)
        t_med = self.thresholds.get("medium", 0.55)
        t_high = self.thresholds.get("high", 0.70)

        ax1.axvline(t_low, color="#9467bd", linestyle=":", linewidth=1.5, label=f"Low Match ($\\tau={t_low}$)")
        ax1.axvline(t_med, color="#e377c2", linestyle=":", linewidth=1.5, label=f"Medium Related ($\\tau={t_med}$)")
        ax1.axvline(t_high, color="#d62728", linestyle=":", linewidth=1.5, label=f"High Duplicate ($\\tau={t_high}$)")

        ax1.set_title("(A) Threshold Trade-Offs: Noise Rejection vs. Retention", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Similarity Threshold ($\\tau$)", fontsize=11)
        ax1.set_ylabel("Rate (%)", fontsize=11)
        ax1.set_xlim(0.30, 0.85)
        ax1.set_ylim(0, 105)
        ax1.grid(True, linestyle=":", alpha=0.5)
        ax1.legend(loc="center left", framealpha=0.92, fontsize=8.5)

        # Panel B: Confidence Tier Distribution Breakdown
        labels = ["High\n(Potential Duplicate)", "Medium\n(Related)", "Low\n(Weak Match)", "Unrelated\n(< Cutoff)"]
        keys = ["HIGH", "MEDIUM", "LOW", "UNRELATED"]
        counts = [tier_counts.get(k, 0) for k in keys]
        total = sum(counts)
        pcts = [c / total * 100.0 if total > 0 else 0.0 for c in counts]

        colors = ["#d62728", "#ff7f0e", "#1f77b4", "#7f7f7f"]
        bars = ax2.bar(labels, pcts, color=colors, edgecolor="black", alpha=0.85, width=0.55)

        for bar, count, pct in zip(bars, counts, pcts):
            yval = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + 1.2,
                f"{pct:.1f}%\n({count})",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        ax2.set_title("(B) Top-1 Candidate Confidence Tier Breakdown", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Proportion of Test Queries (%)", fontsize=11)
        ax2.set_ylim(0, max(pcts) * 1.35 if pcts else 100)
        ax2.grid(True, linestyle=":", alpha=0.5, axis="y")

        plt.subplots_adjust(top=0.90, bottom=0.14, left=0.07, right=0.95, wspace=0.25)
        plt.savefig(out_path, dpi=300)
        plt.close(fig)

        print(f"Threshold calibration figure saved to: {out_path}")
        return out_path


def run_threshold_calibration_pipeline(
    store_dir: Union[str, Path] = "models/related_issues",
    figure_path: Union[str, Path] = "ml/related_issues/figures/fig27_similarity_threshold_calibration.png",
    benchmark_out_path: Union[str, Path] = "data/processed/evaluation/similarity_threshold_benchmark.json",
    sample_size: int = 500,
) -> Tuple[SimilarityThresholdCalibrator, Dict[str, Any]]:
    """Execute end-to-end threshold calibration workflow for Step 7.5."""
    from ml.related_issues.retriever import RelatedIssueRetriever

    print(f"\n=======================================================")
    print(f"Starting Similarity Threshold Calibration (Phase 7, Step 7.5)")
    print(f"Store: {store_dir}")
    print(f"=======================================================\n")

    store = IssueEmbeddingStore(store_dir=store_dir)
    store.load(verify_checksums=True)

    calibrator = SimilarityThresholdCalibrator(thresholds=DEFAULT_THRESHOLDS)

    print(f"Sweeping threshold trade-offs across 0.30 to 0.85 (sample={sample_size})...")
    tradeoff_df = calibrator.sweep_threshold_tradeoffs(store, sample_size=sample_size)

    # Evaluate metrics at calibrated points
    def get_metrics_at(tau: float) -> Dict[str, float]:
        idx = (tradeoff_df["threshold"] - tau).abs().idxmin()
        row = tradeoff_df.iloc[idx]
        return {
            "threshold": float(row["threshold"]),
            "noise_rejection_rate": float(row["noise_rejection_rate"]),
            "intra_retention_rate": float(row["intra_retention_rate"]),
            "label_concordance": float(row["label_concordance"]),
        }

    high_metrics = get_metrics_at(DEFAULT_THRESHOLDS["high"])
    med_metrics = get_metrics_at(DEFAULT_THRESHOLDS["medium"])
    low_metrics = get_metrics_at(DEFAULT_THRESHOLDS["low"])

    # Evaluate tier breakdown across sample retrieval queries
    print("Evaluating confidence tier distributions on historical top-1 candidates...")
    retriever = RelatedIssueRetriever(store=store)
    np.random.seed(42)
    sample_test_ids = np.random.choice(store.metadata["issue_id"], size=100, replace=False)

    tier_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNRELATED": 0}
    for q_id in sample_test_ids:
        res = retriever.retrieve_by_id(issue_id=q_id, top_k=1)
        if res:
            cls_res = calibrator.classify_score(res[0]["similarity_score"])
            tier_counts[cls_res["tier"]] += 1
        else:
            tier_counts["UNRELATED"] += 1

    # Plot diagnostic figure
    calibrator.plot_threshold_calibration(
        tradeoff_df=tradeoff_df,
        tier_counts=tier_counts,
        save_path=figure_path,
    )

    # Save benchmark JSON
    benchmark_payload = {
        "title": "OpenSourceLens Similarity Threshold Calibration Benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "calibrated_thresholds": DEFAULT_THRESHOLDS,
        "operating_points": {
            "high_duplicate": high_metrics,
            "medium_related": med_metrics,
            "low_weak": low_metrics,
        },
        "top1_tier_distribution_100_queries": tier_counts,
        "top1_tier_percentages": {
            k: round(v / len(sample_test_ids) * 100.0, 1) for k, v in tier_counts.items()
        },
        "sweep_samples": sample_size,
    }

    bench_path = Path(benchmark_out_path)
    bench_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, indent=2)

    print(f"Similarity threshold benchmark JSON saved to: {bench_path}")

    print("\n--- Similarity Threshold Calibration Summary ---")
    print(f"HIGH Duplicate (tau >= {high_metrics['threshold']:.2f}):  Noise Rejection = {high_metrics['noise_rejection_rate']*100:.2f}%, Label Concordance = {high_metrics['label_concordance']*100:.1f}%")
    print(f"MED Related    (tau >= {med_metrics['threshold']:.2f}):  Noise Rejection = {med_metrics['noise_rejection_rate']*100:.2f}%, Label Concordance = {med_metrics['label_concordance']*100:.1f}%")
    print(f"LOW Weak       (tau >= {low_metrics['threshold']:.2f}):  Noise Rejection = {low_metrics['noise_rejection_rate']*100:.2f}%, Label Concordance = {low_metrics['label_concordance']*100:.1f}%")
    print(f"Top-1 Tiers on Test Sample: HIGH: {tier_counts['HIGH']}%, MED: {tier_counts['MEDIUM']}%, LOW: {tier_counts['LOW']}%, UNRELATED: {tier_counts['UNRELATED']}%")
    print("------------------------------------------------\n")

    return calibrator, benchmark_payload


if __name__ == "__main__":
    run_threshold_calibration_pipeline()
