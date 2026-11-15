"""Issue Embedding Pipeline for Semantic Representation & Similarity Search.

Implements the sentence embedding generator using SentenceTransformers (all-MiniLM-L6-v2):
- Encodes natural language GitHub issues into 384-dimensional dense vectors
- Applies strict unit L2 normalization (||v||_2 = 1.0) for ultra-fast dot product cosine similarity
- Batched inference with multithreaded CPU/GPU execution
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

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
        from sentence_transformers import SentenceTransformer

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
        show_progress_bar: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Encode issues from a DataFrame and record execution benchmarks."""
        if title_col not in df.columns:
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

        memory_bytes = embeddings.nbytes
        memory_mb = memory_bytes / (1024 * 1024)

        norms = np.linalg.norm(embeddings, axis=1) if len(embeddings) > 0 else np.array([])

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
            "l2_norm_mean": round(float(np.mean(norms)), 5) if len(norms) > 0 else 0.0,
            "l2_norm_std": round(float(np.std(norms)), 6) if len(norms) > 0 else 0.0,
            "l2_norm_min": round(float(np.min(norms)), 5) if len(norms) > 0 else 0.0,
            "l2_norm_max": round(float(np.max(norms)), 5) if len(norms) > 0 else 0.0,
        }

        return embeddings, benchmark
