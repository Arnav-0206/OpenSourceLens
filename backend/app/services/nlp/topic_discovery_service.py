"""Unsupervised Topic & Theme Discovery Service for OpenSourceLens.

Applies Spherical KMeans clustering over SentenceTransformer dense embeddings and
extracts interpretable topic clusters, representative key phrases, and cluster
health metrics via Class-based TF-IDF (c-TF-IDF).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy.orm import Session

from backend.app.schemas.analytics_v2 import (
    DiscoveredTopicItem,
    TopicDiscoveryResponse,
)
from backend.app.services.nlp.semantic_index_service import semantic_index_service
from backend.app.services.nlp.text_cleaner import IssueTextCleaner

logger = logging.getLogger(__name__)

CUSTOM_STOP_WORDS = {
    "code_block",
    "inline_code",
    "url_ref",
    "user_ref",
    "issue_ref",
    "github",
    "issue",
    "issues",
    "error",
    "problem",
    "using",
    "use",
    "need",
    "like",
    "want",
    "step",
    "steps",
    "reproduce",
    "reproduction",
    "expected",
    "actual",
    "behavior",
    "version",
    "run",
    "running",
    "test",
    "tests",
    "file",
    "line",
}


def spherical_kmeans(
    X: np.ndarray,
    k: int,
    max_iter: int = 50,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure-NumPy Spherical KMeans optimizing cosine similarity on unit sphere."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]

    # KMeans++ cosine initialization
    centroids = np.empty((k, X.shape[1]), dtype=np.float32)
    first_idx = rng.integers(0, n)
    centroids[0] = X[first_idx]
    for c in range(1, k):
        sims = np.max(np.dot(X, centroids[:c].T), axis=1)
        probs = np.clip(1.0 - sims, 0, None)
        total_p = float(np.sum(probs))
        if total_p > 0:
            probs = probs / total_p
            centroids[c] = X[rng.choice(n, p=probs)]
        else:
            centroids[c] = X[rng.integers(0, n)]

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max_iter):
        sim_matrix = np.dot(X, centroids.T)
        new_labels = np.argmax(sim_matrix, axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if np.any(mask):
                new_c = np.mean(X[mask], axis=0)
                norm = float(np.linalg.norm(new_c))
                centroids[c] = new_c / (norm + 1e-12)

    return labels, centroids


class TopicDiscoveryService:
    """Discovers recurring themes and topics across a repository's backlog."""

    _instance: Optional[TopicDiscoveryService] = None

    def __new__(cls) -> TopicDiscoveryService:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self.cleaner = IssueTextCleaner()
        self._initialized = True

    def discover_repository_topics(
        self,
        repository_id: int,
        repo_name: str,
        db: Session,
        max_clusters: int = 8,
    ) -> TopicDiscoveryResponse:
        """Run unsupervised clustering and keyword extraction on repository items."""
        t0 = time.perf_counter()

        # 1. Fetch normalized dense vectors and item metadata
        matrix, metadata = semantic_index_service.get_repository_vectors(
            repository_id=repository_id,
            db=db,
            include_discussions=True,
        )

        # Auto-index repository if vectors are not yet computed
        if matrix.shape[0] == 0:
            semantic_index_service.index_repository(repository_id, db)
            matrix, metadata = semantic_index_service.get_repository_vectors(
                repository_id=repository_id,
                db=db,
                include_discussions=True,
            )

        n_samples = matrix.shape[0]
        if n_samples < 3:
            return TopicDiscoveryResponse(
                repo_name=repo_name,
                total_topics_discovered=0,
                total_items_clustered=n_samples,
                methodology="Spherical KMeans + c-TF-IDF",
                topics=[],
                execution_time_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )

        # 2. Determine cluster count K
        k = min(max_clusters, max(2, n_samples // 20))
        if k > n_samples:
            k = max(2, n_samples)

        # 3. Run Spherical KMeans clustering on unit sphere
        labels, centroids = spherical_kmeans(matrix, k=k)

        # 4. Group documents by cluster for c-TF-IDF keyword extraction
        cluster_texts: List[str] = []
        cluster_items_map: Dict[int, List[int]] = {c: [] for c in range(k)}

        for idx, cluster_id in enumerate(labels):
            cluster_items_map[cluster_id].append(idx)

        for c in range(k):
            item_indices = cluster_items_map[c]
            cluster_doc_parts = []
            for idx in item_indices:
                meta = metadata[idx]
                clean_title = self.cleaner.clean_text(meta.get("title", ""))
                cluster_doc_parts.append(clean_title)
            cluster_texts.append(" ".join(cluster_doc_parts))

        # 5. Extract top c-TF-IDF keywords per cluster
        vectorizer = TfidfVectorizer(
            max_features=1500,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(cluster_texts)
            feature_names = np.array(vectorizer.get_feature_names_out())
        except ValueError:
            feature_names = np.array([])
            tfidf_matrix = None

        # 6. Build DiscoveredTopicItem objects
        topics: List[DiscoveredTopicItem] = []

        for c in range(k):
            item_indices = cluster_items_map[c]
            total_count = len(item_indices)
            if total_count == 0:
                continue

            # Keywords extraction
            cluster_keywords: List[str] = []
            if tfidf_matrix is not None and len(feature_names) > 0:
                row_scores = tfidf_matrix[c].toarray()[0]
                sorted_feature_indices = np.argsort(-row_scores)
                for f_idx in sorted_feature_indices:
                    token = str(feature_names[f_idx]).strip()
                    if token and token not in CUSTOM_STOP_WORDS and len(token) > 2:
                        cluster_keywords.append(token)
                    if len(cluster_keywords) >= 5:
                        break

            # Synthesize human-readable title
            if cluster_keywords:
                top_tokens = [w.capitalize() for w in cluster_keywords[:3]]
                topic_title = " & ".join(top_tokens)
            else:
                topic_title = f"Topic Cluster #{c + 1}"

            # Calculate open vs. closed / answered counts
            open_count = 0
            closed_count = 0
            centroid = centroids[c]

            # Find closest representative items to centroid
            item_scores = []
            for idx in item_indices:
                meta = metadata[idx]
                state = meta.get("state", "open").lower()
                if state in ("open", "reopened"):
                    open_count += 1
                else:
                    closed_count += 1

                vec = matrix[idx]
                sim = float(np.dot(vec, centroid))
                item_scores.append((sim, meta))

            item_scores.sort(key=lambda x: x[0], reverse=True)
            representative = [
                {
                    "number": item["number"],
                    "title": item["title"],
                    "entity_type": item["entity_type"],
                    "state": item["state"],
                    "similarity_to_centroid": round(float(sim_score), 4),
                }
                for sim_score, item in item_scores[:3]
            ]

            topics.append(
                DiscoveredTopicItem(
                    cluster_id=c,
                    title=topic_title,
                    keywords=cluster_keywords,
                    total_count=total_count,
                    open_count=open_count,
                    closed_count=closed_count,
                    representative_items=representative,
                )
            )

        # Sort topics by total count descending
        topics.sort(key=lambda t: t.total_count, reverse=True)

        execution_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return TopicDiscoveryResponse(
            repo_name=repo_name,
            total_topics_discovered=len(topics),
            total_items_clustered=n_samples,
            methodology="Spherical KMeans + c-TF-IDF",
            topics=topics,
            execution_time_ms=execution_ms,
        )


topic_discovery_service = TopicDiscoveryService()
