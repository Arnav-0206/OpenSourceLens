"""Issue Embedding Pipeline for Semantic Related-Issue Detection (Phase 7, Step 7.1).

Implements the sentence embedding generator using SentenceTransformers (all-MiniLM-L6-v2):
- Encodes natural language GitHub issues into 384-dimensional dense vectors
- Applies strict unit L2 normalization (||v||_2 = 1.0) for ultra-fast dot product cosine similarity
- Batched inference with multithreaded CPU execution
- Diagnostic visualization (fig24_embedding_norm_and_latency.png)
- Quantitative benchmark serialization (embedding_generation_benchmark.json)
"""

import datetime
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch

DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int = 384


class IssueEmbedder:
    """Manages sentence embedding generation, L2 normalization, and vector profiling."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: Optional[Union[str, torch.device]] = None,
        normalize_embeddings: bool = True,
    ):
        """Initialize the sentence transformer embedding pipeline."""
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings

        if device is not None:
            self.device = str(device)
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(self.model_name, device=self.device)
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension = self.model.get_embedding_dimension() or EMBEDDING_DIMENSION
        elif hasattr(self.model, "get_sentence_embedding_dimension"):
            self.dimension = self.model.get_sentence_embedding_dimension() or EMBEDDING_DIMENSION
        else:
            self.dimension = EMBEDDING_DIMENSION

    def format_issue_text(self, title: str, body: Optional[str] = "") -> str:
        """Format issue title and body into structured representation."""
        clean_title = (title or "").strip()
        clean_body = (body or "").strip()
        if clean_body:
            return f"Title: {clean_title} | Description: {clean_body}"
        return f"Title: {clean_title}"

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single issue text string into an L2-normalized 1D vector of shape (D,)."""
        vec = self.model.encode(
            text,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vec.astype(np.float32)

    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode a list of text strings into an L2-normalized 2D matrix of shape (N, D)."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_issues(
        self,
        df: pd.DataFrame,
        title_col: str = "clean_title",
        body_col: str = "clean_body",
        batch_size: int = 64,
        show_progress_bar: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Encode issues from a DataFrame and record execution benchmarks."""
        if title_col not in df.columns:
            # Fall back to 'title' if clean_title not present
            title_col = "title" if "title" in df.columns else title_col

        titles = df[title_col].fillna("").astype(str).tolist()
        bodies = (
            df[body_col].fillna("").astype(str).tolist()
            if body_col in df.columns
            else [""] * len(df)
        )

        formatted_texts = [
            self.format_issue_text(t, b) for t, b in zip(titles, bodies)
        ]

        t0 = time.perf_counter()
        embeddings = self.encode_texts(
            formatted_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )
        total_time = time.perf_counter() - t0

        n_samples = len(df)
        throughput = n_samples / total_time if total_time > 0 else 0.0
        latency_ms = (total_time / n_samples) * 1000.0 if n_samples > 0 else 0.0

        # Memory calculations
        memory_bytes = embeddings.nbytes
        memory_mb = memory_bytes / (1024 * 1024)

        norms = np.linalg.norm(embeddings, axis=1)

        benchmark = {
            "model_name": self.model_name,
            "embedding_dimension": self.dimension,
            "sample_count": n_samples,
            "batch_size": batch_size,
            "device": self.device,
            "total_encoding_seconds": round(total_time, 3),
            "throughput_samples_per_sec": round(throughput, 2),
            "latency_ms_per_sample": round(latency_ms, 2),
            "memory_footprint_mb": round(memory_mb, 3),
            "l2_norm_mean": round(float(np.mean(norms)), 5),
            "l2_norm_std": round(float(np.std(norms)), 6),
            "l2_norm_min": round(float(np.min(norms)), 5),
            "l2_norm_max": round(float(np.max(norms)), 5),
        }

        return embeddings, benchmark

    def plot_embedding_diagnostics(
        self,
        embeddings: np.ndarray,
        df: Optional[pd.DataFrame] = None,
        save_path: Union[str, Path] = "ml/related_issues/figures/fig24_embedding_norm_and_latency.png",
    ) -> Path:
        """Generate dual-panel diagnostic visualization for vector norms and similarity distribution."""
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        norms = np.linalg.norm(embeddings, axis=1)
        n_samples = len(embeddings)

        # Sample pairwise cosine similarities
        np.random.seed(42)
        sample_indices = np.random.choice(n_samples, size=min(300, n_samples), replace=False)
        sub_matrix = embeddings[sample_indices]
        # Since vectors are unit normalized, cosine sim is dot product
        pairwise_sims = np.dot(sub_matrix, sub_matrix.T)
        # Extract upper triangle without self-similarity diagonal
        triu_indices = np.triu_indices(len(sample_indices), k=1)
        sim_values = pairwise_sims[triu_indices]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

        # Panel A: L2 Norm Distribution
        norm_range = (0.98, 1.02) if np.isclose(norms.min(), norms.max(), atol=1e-5) else None
        ax1.hist(norms, bins=30, range=norm_range, color="#1f77b4", edgecolor="black", alpha=0.85)
        ax1.axvline(1.0, color="red", linestyle="--", linewidth=1.5, label="Unit Norm Target (||v||_2 = 1.0)")
        ax1.set_title("A. Vector L2 Norm Distribution (Unit Sphere)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("L2 Norm ||v||_2", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Frequency", fontsize=10, fontweight="bold")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right", frameon=True)

        # Panel B: Pairwise Cosine Similarity Distribution
        sim_range = (-0.2, 1.0) if len(sim_values) == 0 or np.isclose(sim_values.min(), sim_values.max(), atol=1e-5) else None
        ax2.hist(sim_values, bins=40, range=sim_range, color="#2ca02c", edgecolor="black", alpha=0.85)
        mean_sim = float(np.mean(sim_values))
        median_sim = float(np.median(sim_values))
        ax2.axvline(mean_sim, color="blue", linestyle="-", linewidth=1.5, label=f"Mean Sim: {mean_sim:.3f}")
        ax2.axvline(median_sim, color="orange", linestyle=":", linewidth=1.8, label=f"Median Sim: {median_sim:.3f}")
        ax2.set_title("B. Pairwise Semantic Cosine Similarities (Sampled)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Cosine Similarity", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Pair Count", fontsize=10, fontweight="bold")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper right", frameon=True)

        plt.suptitle(
            f"OpenSourceLens: Sentence-Transformer Vector Diagnostics ({self.model_name}, N={n_samples})",
            fontsize=12,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

        print(f"Embedding diagnostics figure saved to: {out_path}")
        return out_path

    def save_benchmark_summary(
        self,
        benchmark_data: Dict[str, Any],
        save_path: Union[str, Path] = "data/processed/evaluation/embedding_generation_benchmark.json",
    ) -> Path:
        """Serialize embedding generation benchmark to JSON."""
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "title": "OpenSourceLens Sentence Transformer Embedding Benchmark",
            "timestamp": datetime.datetime.now().isoformat(),
            **benchmark_data,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"Embedding benchmark JSON saved to: {out_path}")
        return out_path


def run_embedding_generation_pipeline(
    issues_csv_path: Union[str, Path] = "data/processed/issues_master.csv",
    figure_path: Union[str, Path] = "ml/related_issues/figures/fig24_embedding_norm_and_latency.png",
    benchmark_json_path: Union[str, Path] = "data/processed/evaluation/embedding_generation_benchmark.json",
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Execute end-to-end embedding generation workflow for Step 7.1."""
    df = pd.read_csv(issues_csv_path)

    print(f"\n=======================================================")
    print(f"Starting Issue Embedding Generation (Phase 7, Step 7.1)")
    print(f"Model: {model_name} | Corpus: {len(df)} issues | Batch Size: {batch_size}")
    print(f"=======================================================\n")

    embedder = IssueEmbedder(model_name=model_name, normalize_embeddings=True)
    embeddings, benchmark = embedder.encode_issues(
        df=df,
        title_col="clean_title",
        body_col="clean_body",
        batch_size=batch_size,
        show_progress_bar=True,
    )

    embedder.plot_embedding_diagnostics(embeddings, df=df, save_path=figure_path)
    embedder.save_benchmark_summary(benchmark, save_path=benchmark_json_path)

    print("\n--- Embedding Generation Summary ---")
    print(f"Issues Encoded:        {benchmark['sample_count']}")
    print(f"Vector Dimensions:     {benchmark['embedding_dimension']}")
    print(f"Throughput:            {benchmark['throughput_samples_per_sec']:.1f} issues/sec")
    print(f"Per-Issue Latency:     {benchmark['latency_ms_per_sample']:.2f} ms")
    print(f"Total Time:            {benchmark['total_encoding_seconds']:.2f} s")
    print(f"RAM Footprint:         {benchmark['memory_footprint_mb']:.2f} MB")
    print(f"L2 Norm (Mean +/- Std): {benchmark['l2_norm_mean']:.5f} +/- {benchmark['l2_norm_std']:.6f}")
    print("------------------------------------\n")

    return embeddings, benchmark


if __name__ == "__main__":
    run_embedding_generation_pipeline()
