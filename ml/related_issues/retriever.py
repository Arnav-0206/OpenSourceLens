"""
OpenSourceLens — Related Issue Retriever (Phase 7, Step 7.4)
Provides top-K nearest-neighbor candidate extraction by text or issue ID,
self-match exclusion, repo and state filtering, and retrieval benchmarking.
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.related_issues.embedder import IssueEmbedder
from ml.related_issues.similarity import SimilarityEngine
from ml.related_issues.store import IssueEmbeddingStore


class RelatedIssueRetriever:
    """High-performance top-K semantic retriever for GitHub issues."""

    def __init__(
        self,
        store: Optional[IssueEmbeddingStore] = None,
        embedder: Optional[IssueEmbedder] = None,
        store_dir: Union[str, Path] = "models/related_issues",
    ):
        self.store_dir = Path(store_dir)
        self.store = store if store is not None else IssueEmbeddingStore(self.store_dir)
        self.embedder = embedder
        self.engine = SimilarityEngine()

    def _ensure_store_loaded(self) -> None:
        """Ensure the underlying IssueEmbeddingStore is populated."""
        if not self.store.is_loaded:
            self.store.load(verify_checksums=True)

    def _ensure_embedder_loaded(self) -> None:
        """Ensure the Sentence-Transformer embedder is initialized."""
        if self.embedder is None:
            self.embedder = IssueEmbedder(model_name="all-MiniLM-L6-v2", normalize_embeddings=True)

    def _extract_top_k(
        self,
        scores: np.ndarray,
        metadata: pd.DataFrame,
        top_k: int,
        exclude_issue_ids: Optional[Set[Any]] = None,
        state_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract and rank top-K candidates using O(N) argpartition.

        Args:
            scores: Array of similarity scores matching metadata rows.
            metadata: Aligned DataFrame.
            top_k: Number of candidates to return.
            exclude_issue_ids: Set of issue IDs to mask out (e.g. self-match).
            state_filter: Optional filter on issue state ('open', 'closed').

        Returns:
            Ranked list of candidate dictionaries.
        """
        n_candidates = len(scores)
        if n_candidates == 0:
            return []

        # Masking
        valid_mask = np.ones(n_candidates, dtype=bool)

        if exclude_issue_ids and "issue_id" in metadata.columns:
            # Handle int and str representations
            raw_ids = metadata["issue_id"].to_numpy()
            is_excluded = np.isin(raw_ids, list(exclude_issue_ids))
            if raw_ids.dtype.kind in ("i", "u", "f"):
                # Also check string representations
                str_ids = raw_ids.astype(str)
                str_excluded = np.isin(str_ids, [str(x) for x in exclude_issue_ids])
                is_excluded = is_excluded | str_excluded
            valid_mask &= ~is_excluded

        if state_filter and "state" in metadata.columns:
            valid_mask &= (metadata["state"].str.lower() == state_filter.lower()).to_numpy()

        n_valid = int(valid_mask.sum())
        if n_valid == 0:
            return []

        k = min(top_k, n_valid)

        # Invalidate masked candidates with -infinity
        masked_scores = scores.copy()
        masked_scores[~valid_mask] = -np.inf

        # O(N) selection via argpartition
        if k < n_candidates:
            partition_idx = np.argpartition(-masked_scores, k)[:k]
        else:
            partition_idx = np.arange(n_candidates)

        # Sort only the top-K subset in descending order
        sorted_order = np.argsort(-masked_scores[partition_idx])
        final_indices = partition_idx[sorted_order]

        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(final_indices, 1):
            score = float(masked_scores[idx])
            if score <= -1e8:
                break  # Reached masked out items

            row = metadata.iloc[idx]
            issue_id = row.get("issue_id", None)
            issue_num = int(row.get("issue_number", 0)) if "issue_number" in row and pd.notna(row["issue_number"]) else None

            results.append({
                "rank": rank,
                "issue_id": int(issue_id) if pd.notna(issue_id) and str(issue_id).isdigit() else issue_id,
                "issue_number": issue_num,
                "repo_name": str(row.get("repo_name", "")),
                "title": str(row.get("title", "")),
                "clean_title": str(row.get("clean_title", "")),
                "canonical_label": str(row.get("canonical_label", "")),
                "state": str(row.get("state", "")),
                "created_at": str(row.get("created_at", "")),
                "similarity_score": round(score, 4),
            })

        return results

    def retrieve_by_vector(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        repo_filter: Optional[str] = None,
        state_filter: Optional[str] = None,
        exclude_issue_ids: Optional[Set[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-K candidates given a precomputed query vector.

        Args:
            query_vector: 1D dense vector of shape (384,).
            top_k: Number of nearest neighbors to retrieve.
            repo_filter: Restrict search to specific repository.
            state_filter: Restrict to 'open' or 'closed' issues.
            exclude_issue_ids: Set of issue IDs to omit.

        Returns:
            Ranked list of top-K candidate dictionaries.
        """
        self._ensure_store_loaded()

        if repo_filter:
            sub_embs, sub_meta, _ = self.store.filter_by_repo(repo_filter)
            scores = self.engine.cosine_similarity(query_vector, sub_embs, assume_normalized=True)
            meta = sub_meta
        else:
            scores = self.engine.cosine_similarity(query_vector, self.store.embeddings, assume_normalized=True)
            meta = self.store.metadata

        return self._extract_top_k(
            scores=scores,
            metadata=meta,
            top_k=top_k,
            exclude_issue_ids=exclude_issue_ids,
            state_filter=state_filter,
        )

    def retrieve_by_text(
        self,
        title: str,
        body: str = "",
        top_k: int = 5,
        repo_filter: Optional[str] = None,
        state_filter: Optional[str] = None,
        exclude_issue_ids: Optional[Set[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-K candidates for arbitrary new issue title and body.

        Args:
            title: Issue title.
            body: Issue description body.
            top_k: Number of candidates to return.
            repo_filter: Optional repository scope.
            state_filter: Optional issue state filter.
            exclude_issue_ids: Optional IDs to exclude.

        Returns:
            Ranked list of top-K candidate dictionaries.
        """
        self._ensure_embedder_loaded()
        formatted_text = self.embedder.format_issue_text(title, body)
        query_vector = self.embedder.encode_text(formatted_text)

        return self.retrieve_by_vector(
            query_vector=query_vector,
            top_k=top_k,
            repo_filter=repo_filter,
            state_filter=state_filter,
            exclude_issue_ids=exclude_issue_ids,
        )

    def retrieve_by_id(
        self,
        issue_id: Union[int, str],
        top_k: int = 5,
        repo_filter: Optional[str] = None,
        state_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-K related issues for an existing historical issue, strictly excluding self.

        Args:
            issue_id: Historical issue ID to search for.
            top_k: Number of peers to retrieve.
            repo_filter: Optional repo restriction.
            state_filter: Optional state filter.

        Returns:
            Ranked list of top-K related issue dictionaries.
        """
        self._ensure_store_loaded()
        query_vector, _ = self.store.get_issue_by_id(issue_id)

        # Exclude self from matches
        exclude_set = {issue_id, str(issue_id)}
        if str(issue_id).isdigit():
            exclude_set.add(int(issue_id))

        return self.retrieve_by_vector(
            query_vector=query_vector,
            top_k=top_k,
            repo_filter=repo_filter,
            state_filter=state_filter,
            exclude_issue_ids=exclude_set,
        )

    def benchmark_retrieval(
        self,
        sample_queries: int = 50,
        top_k_values: List[int] = [1, 3, 5, 10, 20],
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        """Profile text vs vector retrieval latency across varying K and evaluate score distributions."""
        self._ensure_store_loaded()
        self._ensure_embedder_loaded()

        n_total = len(self.store.embeddings)
        np.random.seed(random_seed)
        sample_indices = np.random.choice(n_total, size=min(sample_queries, n_total), replace=False)

        top1_scores: List[float] = []
        top3_scores: List[float] = []
        top5_scores: List[float] = []

        vector_latencies_ms: Dict[int, float] = {k: 0.0 for k in top_k_values}
        text_latencies_ms: Dict[int, float] = {k: 0.0 for k in top_k_values}

        # 1. Profile score distributions (using query by ID with self-exclusion)
        for idx in sample_indices:
            issue_id = self.store.metadata.iloc[idx]["issue_id"]
            results = self.retrieve_by_id(issue_id=issue_id, top_k=5)
            if len(results) >= 1:
                top1_scores.append(results[0]["similarity_score"])
            if len(results) >= 3:
                top3_scores.append(results[2]["similarity_score"])
            if len(results) >= 5:
                top5_scores.append(results[4]["similarity_score"])

        # 2. Profile latency vs K
        sample_vectors = self.store.embeddings[sample_indices[:20]]
        sample_titles = self.store.metadata.iloc[sample_indices[:10]]["clean_title"].tolist()
        sample_bodies = self.store.metadata.iloc[sample_indices[:10]]["clean_body"].tolist()

        for k in top_k_values:
            # Vector-only latency (search + rank)
            t0_v = time.perf_counter()
            for vec in sample_vectors:
                _ = self.retrieve_by_vector(query_vector=vec, top_k=k)
            vec_time_per_q = ((time.perf_counter() - t0_v) / len(sample_vectors)) * 1000.0
            vector_latencies_ms[k] = round(vec_time_per_q, 3)

            # End-to-end Text latency (encode + search + rank)
            t0_t = time.perf_counter()
            for title, body in zip(sample_titles, sample_bodies):
                _ = self.retrieve_by_text(title=title, body=body, top_k=k)
            text_time_per_q = ((time.perf_counter() - t0_t) / len(sample_titles)) * 1000.0
            text_latencies_ms[k] = round(text_time_per_q, 3)

        return {
            "sample_queries": len(sample_indices),
            "top1_scores": top1_scores,
            "top3_scores": top3_scores,
            "top5_scores": top5_scores,
            "top1_mean": round(float(np.mean(top1_scores)), 4) if top1_scores else 0.0,
            "top1_median": round(float(np.median(top1_scores)), 4) if top1_scores else 0.0,
            "top3_mean": round(float(np.mean(top3_scores)), 4) if top3_scores else 0.0,
            "top5_mean": round(float(np.mean(top5_scores)), 4) if top5_scores else 0.0,
            "vector_latencies_ms": vector_latencies_ms,
            "text_latencies_ms": text_latencies_ms,
            "vector_throughput_qps": round(1000.0 / vector_latencies_ms[5], 1) if vector_latencies_ms[5] > 0 else 0.0,
            "text_throughput_qps": round(1000.0 / text_latencies_ms[5], 1) if text_latencies_ms[5] > 0 else 0.0,
        }

    def plot_topk_diagnostics(
        self,
        benchmark: Dict[str, Any],
        save_path: Union[str, Path] = "ml/related_issues/figures/fig26_topk_retrieval_analysis.png",
    ) -> Path:
        """Generate dual-panel visualization of top-K score distributions and latency breakdown."""
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

        # Panel A: Top-1, Top-3, Top-5 Score Distributions
        box_data = [
            benchmark["top1_scores"],
            benchmark["top3_scores"],
            benchmark["top5_scores"],
        ]
        labels = [
            f"Top-1 Candidate\n($\\mu={benchmark['top1_mean']:.3f}$)",
            f"Top-3 Candidate\n($\\mu={benchmark['top3_mean']:.3f}$)",
            f"Top-5 Candidate\n($\\mu={benchmark['top5_mean']:.3f}$)",
        ]

        bp = ax1.boxplot(box_data, tick_labels=labels, patch_artist=True, widths=0.45)
        colors = ["#2ca02c", "#1f77b4", "#9467bd"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor("black")

        for median in bp["medians"]:
            median.set(color="red", linewidth=2.0)

        ax1.set_title("(A) Semantic Similarity Scores by Top-K Rank", fontsize=12, fontweight="bold")
        ax1.set_ylabel(r"Cosine Similarity Score ($\cos \theta \in [-1, 1]$)", fontsize=11)
        ax1.set_ylim(-0.1, 1.05)
        ax1.grid(True, linestyle=":", alpha=0.5)

        # Panel B: Latency vs K (Text vs Vector Query)
        k_values = list(benchmark["vector_latencies_ms"].keys())
        v_lat = [benchmark["vector_latencies_ms"][k] for k in k_values]
        t_lat = [benchmark["text_latencies_ms"][k] for k in k_values]

        ax2.plot(k_values, t_lat, marker="s", color="#d62728", linewidth=2.0, label="Text Query (Encode + Search + Rank)")
        ax2.plot(k_values, v_lat, marker="o", color="#1f77b4", linewidth=2.0, label="Vector Query (Search + Rank Only)")

        for k, v in zip(k_values, v_lat):
            ax2.annotate(f"{v:.2f} ms", xy=(k, v), xytext=(0, 6), textcoords="offset points", fontsize=8)
        for k, t in zip(k_values, t_lat):
            ax2.annotate(f"{t:.1f} ms", xy=(k, t), xytext=(0, 6), textcoords="offset points", fontsize=8)

        ax2.set_title("(B) End-to-End Retrieval Latency vs. K", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Number of Retrieved Issues ($K$)", fontsize=11)
        ax2.set_ylabel("Query Latency (Milliseconds, ms)", fontsize=11)
        ax2.set_ylim(0, max(t_lat) * 1.3)
        ax2.grid(True, linestyle=":", alpha=0.5)
        ax2.legend(loc="upper left", framealpha=0.92)

        plt.subplots_adjust(top=0.90, bottom=0.12, left=0.07, right=0.95, wspace=0.25)
        plt.savefig(out_path, dpi=300)
        plt.close(fig)

        print(f"Top-K retrieval diagnostics figure saved to: {out_path}")
        return out_path


def run_retrieval_pipeline(
    store_dir: Union[str, Path] = "models/related_issues",
    figure_path: Union[str, Path] = "ml/related_issues/figures/fig26_topk_retrieval_analysis.png",
    benchmark_out_path: Union[str, Path] = "data/processed/evaluation/topk_retrieval_benchmark.json",
    sample_queries: int = 50,
) -> Tuple[RelatedIssueRetriever, Dict[str, Any]]:
    """Execute end-to-end top-K retrieval benchmarking and qualitative evaluation for Step 7.4."""
    print(f"\n=======================================================")
    print(f"Starting Top-K Issue Retrieval Pipeline (Phase 7, Step 7.4)")
    print(f"Loading store from: {store_dir}")
    print(f"=======================================================\n")

    retriever = RelatedIssueRetriever(store_dir=store_dir)
    retriever._ensure_store_loaded()
    retriever._ensure_embedder_loaded()

    # 1. Qualitative Evaluation: Text query demonstration
    test_title = "CUDA memory out of bounds during backpropagation with multi-GPU training"
    test_body = "Encountered CUDA runtime error: device-side assert triggered when calling loss.backward() with torch.distributed."
    print(f"--- Demonstration: Query by Natural Language Text ---")
    print(f"Query Title: '{test_title}'")
    print(f"Repository Scope: pytorch/pytorch")
    text_results = retriever.retrieve_by_text(
        title=test_title,
        body=test_body,
        top_k=3,
        repo_filter="pytorch/pytorch",
    )
    for res in text_results:
        print(f"  [Rank {res['rank']}] Sim: {res['similarity_score']:.4f} | #{res['issue_number']} ({res['canonical_label']}) - {res['title'][:65]}")

    # 2. Qualitative Evaluation: Query by existing issue ID with self-exclusion
    sample_id = retriever.store.metadata.iloc[10]["issue_id"]
    sample_title = retriever.store.metadata.iloc[10]["title"]
    sample_repo = retriever.store.metadata.iloc[10]["repo_name"]
    print(f"\n--- Demonstration: Query by Existing Issue ID (Self-Excluded) ---")
    print(f"Query Issue ID: {sample_id} ({sample_repo}) | '{sample_title[:65]}'")
    id_results = retriever.retrieve_by_id(issue_id=sample_id, top_k=3)
    for res in id_results:
        print(f"  [Rank {res['rank']}] Sim: {res['similarity_score']:.4f} | ID: {res['issue_id']} #{res['issue_number']} ({res['repo_name']}) - {res['title'][:65]}")

    # 3. Empirical Benchmarking
    print(f"\nBenchmarking retrieval latency and score distributions over {sample_queries} sample queries...")
    bench = retriever.benchmark_retrieval(sample_queries=sample_queries)

    # 4. Generate Diagnostic Figure
    retriever.plot_topk_diagnostics(benchmark=bench, save_path=figure_path)

    # 5. Save Benchmark JSON
    benchmark_payload = {
        "title": "OpenSourceLens Top-K Semantic Retrieval Benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus_size": len(retriever.store.embeddings),
        "embedding_dimension": int(retriever.store.embeddings.shape[1]),
        "sample_queries": bench["sample_queries"],
        "top1_similarity_mean": bench["top1_mean"],
        "top1_similarity_median": bench["top1_median"],
        "top3_similarity_mean": bench["top3_mean"],
        "top5_similarity_mean": bench["top5_mean"],
        "vector_query_latency_ms": bench["vector_latencies_ms"],
        "text_query_latency_ms": bench["text_latencies_ms"],
        "vector_throughput_qps": bench["vector_throughput_qps"],
        "text_throughput_qps": bench["text_throughput_qps"],
        "sample_text_query_results": text_results,
        "sample_id_query_results": id_results,
    }

    bench_path = Path(benchmark_out_path)
    bench_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, indent=2)

    print(f"Top-K retrieval benchmark JSON saved to: {bench_path}")

    print("\n--- Top-K Retrieval Summary ---")
    print(f"Top-1 Candidate Mean Sim:  {bench['top1_mean']:.4f} (median={bench['top1_median']:.4f})")
    print(f"Top-3 Candidate Mean Sim:  {bench['top3_mean']:.4f}")
    print(f"Top-5 Candidate Mean Sim:  {bench['top5_mean']:.4f}")
    print(f"Vector Query Latency:      {bench['vector_latencies_ms'][5]:.3f} ms ({bench['vector_throughput_qps']:.1f} QPS)")
    print(f"Text Query Latency:        {bench['text_latencies_ms'][5]:.2f} ms ({bench['text_throughput_qps']:.1f} QPS)")
    print("-------------------------------\n")

    return retriever, benchmark_payload


if __name__ == "__main__":
    run_retrieval_pipeline()
