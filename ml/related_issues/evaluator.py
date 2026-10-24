"""
OpenSourceLens — Retrieval Quality Evaluator (Phase 7, Step 7.6)
Evaluates Information Retrieval (IR) metrics (MRR, HitRate@K, Precision@K, NDCG@K)
benchmarking Sentence-Transformers (all-MiniLM-L6-v2) against a classical Lexical TF-IDF baseline.
"""

import json
from pathlib import Path
import re
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ml.related_issues.retriever import RelatedIssueRetriever
from ml.related_issues.store import IssueEmbeddingStore


# -------------------------------------------------------------------------
# Classical Lexical Baseline (TF-IDF Cosine Similarity)
# -------------------------------------------------------------------------

class LexicalRetrievalBaseline:
    """Classical keyword-based TF-IDF retrieval baseline for comparative benchmarking."""

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            sublinear_tf=True,
        )
        self.tfidf_matrix = None
        self.metadata: Optional[pd.DataFrame] = None
        self._id_to_idx: Dict[Any, int] = {}

    def fit(self, df: pd.DataFrame) -> "LexicalRetrievalBaseline":
        """Fit TF-IDF matrix on issue titles and descriptions."""
        self.metadata = df.reset_index(drop=True)

        titles = self.metadata["clean_title"].fillna("").astype(str)
        bodies = (
            self.metadata["clean_body"].fillna("").astype(str)
            if "clean_body" in self.metadata.columns
            else pd.Series([""] * len(df))
        )
        corpus_texts = (titles + " " + bodies).tolist()

        self.tfidf_matrix = self.vectorizer.fit_transform(corpus_texts)

        # Build ID index
        self._id_to_idx = {}
        for idx, iid in enumerate(self.metadata["issue_id"]):
            self._id_to_idx[iid] = idx
            self._id_to_idx[str(iid)] = idx

        return self

    def retrieve(
        self,
        query_text: str,
        top_k: int = 10,
        repo_filter: Optional[str] = None,
        exclude_issue_ids: Optional[Set[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-K issues using TF-IDF sparse cosine similarity."""
        if self.tfidf_matrix is None or self.metadata is None:
            raise RuntimeError("Baseline has not been fitted.")

        q_vec = self.vectorizer.transform([query_text])
        # Sparse dot product
        raw_sims = self.tfidf_matrix.dot(q_vec.T).toarray().ravel()

        n_candidates = len(raw_sims)
        valid_mask = np.ones(n_candidates, dtype=bool)

        if repo_filter and "repo_name" in self.metadata.columns:
            valid_mask &= (self.metadata["repo_name"] == repo_filter).to_numpy()

        if exclude_issue_ids and "issue_id" in self.metadata.columns:
            raw_ids = self.metadata["issue_id"].to_numpy()
            is_ex = np.isin(raw_ids, list(exclude_issue_ids))
            if raw_ids.dtype.kind in ("i", "u", "f"):
                is_ex |= np.isin(raw_ids.astype(str), [str(x) for x in exclude_issue_ids])
            valid_mask &= ~is_ex

        n_valid = int(valid_mask.sum())
        if n_valid == 0:
            return []

        k = min(top_k, n_valid)
        scores = raw_sims.copy()
        scores[~valid_mask] = -1.0

        top_indices = np.argpartition(-scores, k)[:k]
        sorted_order = np.argsort(-scores[top_indices])
        final_idx = top_indices[sorted_order]

        results = []
        for rank, idx in enumerate(final_idx, 1):
            score = float(scores[idx])
            if score < 0.0:
                break
            row = self.metadata.iloc[idx]
            results.append({
                "rank": rank,
                "issue_id": row["issue_id"],
                "issue_number": row.get("issue_number"),
                "repo_name": row.get("repo_name"),
                "title": row.get("title"),
                "similarity_score": round(score, 4),
            })
        return results


# -------------------------------------------------------------------------
# Information Retrieval (IR) Ranking Metrics
# -------------------------------------------------------------------------

def compute_mrr(ranks: List[Optional[int]]) -> float:
    """Compute Mean Reciprocal Rank (MRR)."""
    if not ranks:
        return 0.0
    reciprocals = [1.0 / r if (r is not None and r > 0) else 0.0 for r in ranks]
    return float(np.mean(reciprocals))


def compute_hit_rate_at_k(ranks: List[Optional[int]], k: int) -> float:
    """Compute Hit Rate @ K (Recall @ K) — proportion of queries with rank <= k."""
    if not ranks:
        return 0.0
    hits = [1.0 if (r is not None and 1 <= r <= k) else 0.0 for r in ranks]
    return float(np.mean(hits))


def compute_ndcg_at_k(ranks: List[Optional[int]], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain @ K for binary relevance."""
    if not ranks:
        return 0.0
    gains = []
    for r in ranks:
        if r is not None and 1 <= r <= k:
            dcg = 1.0 / np.log2(r + 1)
            idcg = 1.0 / np.log2(1 + 1)  # Ideal rank = 1
            gains.append(dcg / idcg)
        else:
            gains.append(0.0)
    return float(np.mean(gains))


def compute_precision_at_k(ranks: List[Optional[int]], k: int) -> float:
    """Compute Precision @ K in single-target retrieval (1/k if hit else 0)."""
    if not ranks or k <= 0:
        return 0.0
    precisions = [1.0 / k if (r is not None and 1 <= r <= k) else 0.0 for r in ranks]
    return float(np.mean(precisions))


# -------------------------------------------------------------------------
# Retrieval Evaluator
# -------------------------------------------------------------------------

class RetrievalEvaluator:
    """Evaluates and compares neural vs. lexical issue retrieval quality."""

    def __init__(self, store: Optional[IssueEmbeddingStore] = None):
        self.store = store

    def mine_linked_reference_pairs(
        self,
        df: pd.DataFrame,
        max_pairs: int = 150,
        random_seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """
        Extract ground-truth linked issue pairs from intra-repo '#number' cross-references.
        """
        repo_map: Dict[str, Dict[int, Any]] = {}
        for repo, group in df.groupby("repo_name"):
            repo_map[repo] = dict(zip(group["issue_number"].dropna().astype(int), group["issue_id"]))

        pairs: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            repo = row["repo_name"]
            curr_num = int(row["issue_number"]) if pd.notna(row["issue_number"]) else -1
            text = f"{row.get('title', '')} {row.get('body', '')}"

            matches = set(int(m) for m in re.findall(r"#(\d+)", text) if m.isdigit())
            matches.discard(curr_num)

            valid_targets = matches.intersection(repo_map.get(repo, {}).keys())
            for target_num in valid_targets:
                target_id = repo_map[repo][target_num]
                pairs.append({
                    "query_id": row["issue_id"],
                    "query_number": curr_num,
                    "query_title": str(row.get("clean_title", row.get("title", ""))),
                    "query_body": str(row.get("clean_body", row.get("body", ""))),
                    "repo_name": repo,
                    "target_id": target_id,
                    "target_number": target_num,
                    "evaluation_type": "cross_reference",
                })

        # Deterministic sample if exceeding max_pairs
        if len(pairs) > max_pairs:
            np.random.seed(random_seed)
            sample_idx = np.random.choice(len(pairs), size=max_pairs, replace=False)
            pairs = [pairs[i] for i in sample_idx]

        return pairs

    def create_title_to_corpus_evaluation_set(
        self,
        df: pd.DataFrame,
        n_queries: int = 100,
        random_seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """
        Create query perturbation evaluation set: querying with title to retrieve target issue.
        """
        np.random.seed(random_seed)
        sample_df = df.sample(n=min(n_queries, len(df)), random_state=random_seed)

        eval_set: List[Dict[str, Any]] = []
        for _, row in sample_df.iterrows():
            eval_set.append({
                "query_id": row["issue_id"],
                "query_number": int(row.get("issue_number", 0)) if pd.notna(row.get("issue_number")) else 0,
                "query_title": str(row.get("clean_title", row.get("title", ""))),
                "query_body": "",  # Title-only query
                "repo_name": row["repo_name"],
                "target_id": row["issue_id"],
                "target_number": int(row.get("issue_number", 0)) if pd.notna(row.get("issue_number")) else 0,
                "evaluation_type": "title_perturbation",
            })
        return eval_set

    def evaluate_neural_retriever(
        self,
        retriever: RelatedIssueRetriever,
        queries: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Optional[int]]:
        """Run queries through neural retriever and extract target ranks."""
        ranks: List[Optional[int]] = []
        for q in queries:
            target_id = q["target_id"]
            query_id = q["query_id"]

            # If cross-reference: exclude self so we don't return query issue
            # If perturbation: target IS query_id, so do NOT exclude self
            is_perturbation = (q.get("evaluation_type") == "title_perturbation")
            exclude_ids = {query_id} if not is_perturbation else None

            results = retriever.retrieve_by_text(
                title=q["query_title"],
                body=q.get("query_body", ""),
                top_k=top_k,
                repo_filter=q.get("repo_name"),
                exclude_issue_ids=exclude_ids,
            )

            # Find rank of target_id
            found_rank = None
            for item in results:
                if item["issue_id"] == target_id or str(item["issue_id"]) == str(target_id):
                    found_rank = item["rank"]
                    break
            ranks.append(found_rank)

        return ranks

    def evaluate_lexical_baseline(
        self,
        baseline: LexicalRetrievalBaseline,
        queries: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Optional[int]]:
        """Run queries through classical TF-IDF baseline and extract target ranks."""
        ranks: List[Optional[int]] = []
        for q in queries:
            target_id = q["target_id"]
            query_id = q["query_id"]

            is_perturbation = (q.get("evaluation_type") == "title_perturbation")
            exclude_ids = {query_id} if not is_perturbation else None

            query_text = f"{q['query_title']} {q.get('query_body', '')}".strip()
            results = baseline.retrieve(
                query_text=query_text,
                top_k=top_k,
                repo_filter=q.get("repo_name"),
                exclude_issue_ids=exclude_ids,
            )

            found_rank = None
            for item in results:
                if item["issue_id"] == target_id or str(item["issue_id"]) == str(target_id):
                    found_rank = item["rank"]
                    break
            ranks.append(found_rank)

        return ranks

    def summarize_metrics(
        self,
        ranks: List[Optional[int]],
        k_values: List[int] = [1, 3, 5, 10],
    ) -> Dict[str, Any]:
        """Aggregate all IR metrics from target rank list."""
        valid_ranks = [r for r in ranks if r is not None]
        summary: Dict[str, Any] = {
            "total_queries": len(ranks),
            "hits_found": len(valid_ranks),
            "mrr": round(compute_mrr(ranks), 4),
            "median_rank": round(float(np.median(valid_ranks)), 1) if valid_ranks else None,
        }

        for k in k_values:
            summary[f"hit_rate_at_{k}"] = round(compute_hit_rate_at_k(ranks, k), 4)
            summary[f"precision_at_{k}"] = round(compute_precision_at_k(ranks, k), 4)
            summary[f"ndcg_at_{k}"] = round(compute_ndcg_at_k(ranks, k), 4)

        return summary

    def plot_comparative_benchmark(
        self,
        neural_summary: Dict[str, Any],
        lexical_summary: Dict[str, Any],
        save_path: Union[str, Path] = "ml/related_issues/figures/fig28_retrieval_evaluation_benchmark.png",
    ) -> Path:
        """Generate dual-panel comparative visualization (Neural vs. Lexical IR metrics)."""
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

        # Panel A: Hit Rate @ K Curves
        k_vals = [1, 3, 5, 10]
        neural_hits = [neural_summary[f"hit_rate_at_{k}"] * 100.0 for k in k_vals]
        lexical_hits = [lexical_summary[f"hit_rate_at_{k}"] * 100.0 for k in k_vals]

        ax1.plot(k_vals, neural_hits, marker="o", color="#1f77b4", linewidth=2.5, markersize=8, label="Neural (Sentence-Transformers MiniLM)")
        ax1.plot(k_vals, lexical_hits, marker="s", color="#d62728", linestyle="--", linewidth=2.2, markersize=7, label="Classical Lexical Baseline (TF-IDF)")

        for k, n_val, l_val in zip(k_vals, neural_hits, lexical_hits):
            ax1.annotate(f"{n_val:.1f}%", xy=(k, n_val), xytext=(0, 7), textcoords="offset points", fontsize=8.5, fontweight="bold", color="#1f77b4")
            ax1.annotate(f"{l_val:.1f}%", xy=(k, l_val), xytext=(0, -12), textcoords="offset points", fontsize=8.5, color="#d62728")

        ax1.set_title("(A) Hit Rate @ K Comparison (Recall)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Top-K Candidates ($K$)", fontsize=11)
        ax1.set_ylabel("Hit Rate (% Queries Successfully Finding Target)", fontsize=11)
        ax1.set_xticks(k_vals)
        ax1.set_ylim(0, 108)
        ax1.grid(True, linestyle=":", alpha=0.5)
        ax1.legend(loc="lower right", framealpha=0.92)

        # Panel B: Core Ranking Metrics Bar Chart
        metric_keys = ["mrr", "hit_rate_at_1", "hit_rate_at_5", "ndcg_at_5"]
        metric_labels = ["MRR", "HitRate@1", "HitRate@5", "NDCG@5"]

        neural_scores = [neural_summary[k] for k in metric_keys]
        lexical_scores = [lexical_summary[k] for k in metric_keys]

        x = np.arange(len(metric_labels))
        width = 0.35

        b1 = ax2.bar(x - width/2, neural_scores, width, label="Sentence-Transformers", color="#1f77b4", edgecolor="black", alpha=0.85)
        b2 = ax2.bar(x + width/2, lexical_scores, width, label="TF-IDF Baseline", color="#d62728", edgecolor="black", alpha=0.85)

        for bar in b1:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        for bar in b2:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=8.5)

        ax2.set_title("(B) Information Retrieval Ranking Metrics", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Score (0.0 to 1.0)", fontsize=11)
        ax2.set_xticks(x)
        ax2.set_xticklabels(metric_labels, fontsize=10, fontweight="medium")
        ax2.set_ylim(0, 1.15)
        ax2.grid(True, linestyle=":", alpha=0.5, axis="y")
        ax2.legend(loc="upper right", framealpha=0.92)

        plt.subplots_adjust(top=0.90, bottom=0.12, left=0.07, right=0.95, wspace=0.25)
        plt.savefig(out_path, dpi=300)
        plt.close(fig)

        print(f"Retrieval evaluation figure saved to: {out_path}")
        return out_path


def run_retrieval_evaluation_pipeline(
    store_dir: Union[str, Path] = "models/related_issues",
    figure_path: Union[str, Path] = "ml/related_issues/figures/fig28_retrieval_evaluation_benchmark.png",
    benchmark_out_path: Union[str, Path] = "data/processed/evaluation/retrieval_evaluation_benchmark.json",
    n_queries: int = 100,
) -> Tuple[RetrievalEvaluator, Dict[str, Any]]:
    """Execute end-to-end quantitative retrieval quality benchmark for Step 7.6."""
    print(f"\n=======================================================")
    print(f"Starting Retrieval Quality Evaluation Pipeline (Phase 7, Step 7.6)")
    print(f"Store: {store_dir}")
    print(f"=======================================================\n")

    store = IssueEmbeddingStore(store_dir=store_dir)
    store.load(verify_checksums=True)
    df = store.metadata

    evaluator = RetrievalEvaluator(store=store)

    # 1. Build Ground-Truth Evaluation Dataset
    print("Mining ground-truth evaluation pairs...")
    linked_pairs = evaluator.mine_linked_reference_pairs(df, max_pairs=60)
    perturb_queries = evaluator.create_title_to_corpus_evaluation_set(df, n_queries=60)
    eval_queries = linked_pairs + perturb_queries

    print(f"Evaluation dataset constructed: {len(eval_queries)} total queries "
          f"({len(linked_pairs)} cross-references, {len(perturb_queries)} title queries).")

    # 2. Fit Classical TF-IDF Baseline
    print("Fitting classical Lexical TF-IDF baseline...")
    baseline = LexicalRetrievalBaseline(max_features=10000, ngram_range=(1, 2))
    baseline.fit(df)

    # 3. Evaluate Neural Retriever
    print("Evaluating Sentence-Transformers (all-MiniLM-L6-v2) retriever...")
    retriever = RelatedIssueRetriever(store=store)
    t0_neural = time.perf_counter()
    neural_ranks = evaluator.evaluate_neural_retriever(retriever, eval_queries, top_k=10)
    neural_time = time.perf_counter() - t0_neural
    neural_summary = evaluator.summarize_metrics(neural_ranks)
    neural_summary["total_eval_time_sec"] = round(neural_time, 2)
    neural_summary["latency_ms_per_query"] = round((neural_time / len(eval_queries)) * 1000.0, 2)

    # 4. Evaluate Lexical Baseline
    print("Evaluating Lexical TF-IDF baseline...")
    t0_lex = time.perf_counter()
    lexical_ranks = evaluator.evaluate_lexical_baseline(baseline, eval_queries, top_k=10)
    lex_time = time.perf_counter() - t0_lex
    lexical_summary = evaluator.summarize_metrics(lexical_ranks)
    lexical_summary["total_eval_time_sec"] = round(lex_time, 2)
    lexical_summary["latency_ms_per_query"] = round((lex_time / len(eval_queries)) * 1000.0, 2)

    # 5. Diagnostic Plot
    evaluator.plot_comparative_benchmark(
        neural_summary=neural_summary,
        lexical_summary=lexical_summary,
        save_path=figure_path,
    )

    # 6. Save Benchmark JSON
    benchmark_payload = {
        "title": "OpenSourceLens Related Issue Retrieval Quality Benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evaluation_queries_count": len(eval_queries),
        "cross_reference_count": len(linked_pairs),
        "perturbation_count": len(perturb_queries),
        "sentence_transformers_all_MiniLM_L6_v2": neural_summary,
        "classical_lexical_tfidf_baseline": lexical_summary,
        "relative_improvements_pct": {
            "mrr_delta": round((neural_summary["mrr"] - lexical_summary["mrr"]), 4),
            "hit_rate_at_1_delta": round((neural_summary["hit_rate_at_1"] - lexical_summary["hit_rate_at_1"]), 4),
            "hit_rate_at_5_delta": round((neural_summary["hit_rate_at_5"] - lexical_summary["hit_rate_at_5"]), 4),
            "ndcg_at_5_delta": round((neural_summary["ndcg_at_5"] - lexical_summary["ndcg_at_5"]), 4),
        },
    }

    bench_path = Path(benchmark_out_path)
    bench_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, indent=2)

    print(f"Retrieval evaluation benchmark JSON saved to: {bench_path}")

    print("\n--- Retrieval Quality Benchmark Summary ---")
    print(f"Metric         | Sentence-Transformers | TF-IDF Baseline | Relative Delta")
    print(f"---------------+-----------------------+-----------------+---------------")
    print(f"MRR            | {neural_summary['mrr']:.4f}                | {lexical_summary['mrr']:.4f}          | {benchmark_payload['relative_improvements_pct']['mrr_delta']:+.4f}")
    print(f"HitRate@1      | {neural_summary['hit_rate_at_1']*100:.1f}%                 | {lexical_summary['hit_rate_at_1']*100:.1f}%           | {benchmark_payload['relative_improvements_pct']['hit_rate_at_1_delta']*100:+.1f}%")
    print(f"HitRate@3      | {neural_summary['hit_rate_at_3']*100:.1f}%                 | {lexical_summary['hit_rate_at_3']*100:.1f}%           | {(neural_summary['hit_rate_at_3'] - lexical_summary['hit_rate_at_3'])*100:+.1f}%")
    print(f"HitRate@5      | {neural_summary['hit_rate_at_5']*100:.1f}%                 | {lexical_summary['hit_rate_at_5']*100:.1f}%           | {benchmark_payload['relative_improvements_pct']['hit_rate_at_5_delta']*100:+.1f}%")
    print(f"HitRate@10     | {neural_summary['hit_rate_at_10']*100:.1f}%                 | {lexical_summary['hit_rate_at_10']*100:.1f}%           | {(neural_summary['hit_rate_at_10'] - lexical_summary['hit_rate_at_10'])*100:+.1f}%")
    print(f"NDCG@5         | {neural_summary['ndcg_at_5']:.4f}                | {lexical_summary['ndcg_at_5']:.4f}          | {benchmark_payload['relative_improvements_pct']['ndcg_at_5_delta']:+.4f}")
    print(f"Latency/Query  | {neural_summary['latency_ms_per_query']:.2f} ms              | {lexical_summary['latency_ms_per_query']:.2f} ms           | -")
    print("-------------------------------------------------------------------------\n")

    return evaluator, benchmark_payload


if __name__ == "__main__":
    run_retrieval_evaluation_pipeline()
