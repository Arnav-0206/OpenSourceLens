"""Dynamic Per-Repository Semantic Indexing & Vector Search Service.

Uses SentenceTransformers (all-MiniLM-L6-v2) to generate 384-dimensional dense
L2-normalized vectors for any repository's issues and discussions dynamically,
persisting them in issue_embeddings and discussion_embeddings binary tables, and
executing sub-millisecond dot product cosine similarity searches.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.models import (
    Discussion,
    DiscussionEmbedding,
    Issue,
    IssueEmbedding,
    Repository,
)
from backend.app.services.nlp.embedder import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    IssueEmbedder,
)

logger = logging.getLogger(__name__)

# Calibrated operating thresholds from benchmarks
HIGH_DUPLICATE_THRESHOLD: float = 0.70
MEDIUM_RELATED_THRESHOLD: float = 0.55
LOW_WEAK_THRESHOLD: float = 0.45


class SemanticIndexService:
    """Manages per-repository vector stores and similarity retrieval."""

    _instance: Optional[SemanticIndexService] = None

    def __new__(cls) -> SemanticIndexService:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self._embedder: Optional[IssueEmbedder] = None
        self._initialized = True

    @property
    def embedder(self) -> IssueEmbedder:
        """Lazy load sentence transformer embedder."""
        if self._embedder is None:
            logger.info("Initializing IssueEmbedder (%s)...", DEFAULT_EMBEDDING_MODEL)
            self._embedder = IssueEmbedder(
                model_name=DEFAULT_EMBEDDING_MODEL,
                normalize_embeddings=True,
            )
        return self._embedder

    def index_repository_issues(
        self,
        repository_id: int,
        db: Session,
        batch_size: int = 64,
    ) -> int:
        """Compute and persist embeddings for unindexed issues in a repository.

        Args:
            repository_id: Target repository primary key.
            db: Active database session.
            batch_size: Encoding batch size.

        Returns:
            Count of newly indexed issues.
        """
        # Select issues without an existing embedding
        stmt = (
            select(Issue)
            .outerjoin(IssueEmbedding, Issue.id == IssueEmbedding.issue_id)
            .where(
                Issue.repository_id == repository_id,
                IssueEmbedding.id.is_(None),
            )
            .order_by(Issue.created_at.desc())
        )
        unindexed_issues = list(db.scalars(stmt).all())
        if not unindexed_issues:
            return 0

        logger.info(
            "Indexing %d unindexed issues for repository %d...",
            len(unindexed_issues),
            repository_id,
        )

        total_indexed = 0
        for i in range(0, len(unindexed_issues), batch_size):
            batch = unindexed_issues[i : i + batch_size]
            texts = [
                self.embedder.format_issue_text(
                    title=issue.title,
                    body=issue.clean_body or issue.body or "",
                )
                for issue in batch
            ]
            vectors = self.embedder.encode_texts(texts, batch_size=batch_size)

            for issue, vec in zip(batch, vectors):
                blob = vec.astype(np.float32).tobytes()
                record = IssueEmbedding(
                    repository_id=repository_id,
                    issue_id=issue.id,
                    embedding_blob=blob,
                    model_version=DEFAULT_EMBEDDING_MODEL,
                )
                db.add(record)
                total_indexed += 1

            db.commit()

        logger.info(
            "Successfully indexed %d issues for repository %d",
            total_indexed,
            repository_id,
        )
        return total_indexed

    def index_repository_discussions(
        self,
        repository_id: int,
        db: Session,
        batch_size: int = 64,
    ) -> int:
        """Compute and persist embeddings for unindexed discussions in a repository."""
        stmt = (
            select(Discussion)
            .outerjoin(
                DiscussionEmbedding,
                Discussion.id == DiscussionEmbedding.discussion_id,
            )
            .where(
                Discussion.repository_id == repository_id,
                DiscussionEmbedding.id.is_(None),
            )
            .order_by(Discussion.created_at.desc())
        )
        unindexed = list(db.scalars(stmt).all())
        if not unindexed:
            return 0

        logger.info(
            "Indexing %d unindexed discussions for repository %d...",
            len(unindexed),
            repository_id,
        )

        total_indexed = 0
        for i in range(0, len(unindexed), batch_size):
            batch = unindexed[i : i + batch_size]
            texts = [
                f"Title: {d.title} | Category: {d.category_name or 'General'} | Content: {d.body or ''}"
                for d in batch
            ]
            vectors = self.embedder.encode_texts(texts, batch_size=batch_size)

            for disc, vec in zip(batch, vectors):
                blob = vec.astype(np.float32).tobytes()
                record = DiscussionEmbedding(
                    repository_id=repository_id,
                    discussion_id=disc.id,
                    embedding_blob=blob,
                    model_version=DEFAULT_EMBEDDING_MODEL,
                )
                db.add(record)
                total_indexed += 1

            db.commit()

        logger.info(
            "Successfully indexed %d discussions for repository %d",
            total_indexed,
            repository_id,
        )
        return total_indexed

    def index_repository(self, repository_id: int, db: Session) -> Dict[str, int]:
        """Index both issues and discussions for a repository."""
        issues_count = self.index_repository_issues(repository_id, db)
        disc_count = self.index_repository_discussions(repository_id, db)
        return {
            "indexed_issues": issues_count,
            "indexed_discussions": disc_count,
        }

    def get_repository_vectors(
        self,
        repository_id: int,
        db: Session,
        include_discussions: bool = True,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Load and deserialize normalized vectors and metadata for a repository.

        Returns:
            Tuple of (embeddings_matrix, metadata_list).
            embeddings_matrix has shape (M, 384).
        """
        # 1. Fetch issue vectors with joined issue details
        stmt_issues = (
            select(IssueEmbedding, Issue)
            .join(Issue, IssueEmbedding.issue_id == Issue.id)
            .where(IssueEmbedding.repository_id == repository_id)
        )
        issue_rows = db.execute(stmt_issues).all()

        vectors_list: List[np.ndarray] = []
        metadata_list: List[Dict[str, Any]] = []

        for emb, issue in issue_rows:
            vec = np.frombuffer(emb.embedding_blob, dtype=np.float32).copy()
            vectors_list.append(vec)
            metadata_list.append(
                {
                    "entity_type": "issue",
                    "id": issue.id,
                    "number": issue.issue_number,
                    "title": issue.title,
                    "state": issue.state,
                    "canonical_label": issue.canonical_label,
                    "author": issue.author,
                    "comments_count": issue.comments_count,
                    "created_at": issue.created_at.isoformat() if issue.created_at else None,
                }
            )

        # 2. Optionally fetch discussion vectors
        if include_discussions:
            stmt_disc = (
                select(DiscussionEmbedding, Discussion)
                .join(Discussion, DiscussionEmbedding.discussion_id == Discussion.id)
                .where(DiscussionEmbedding.repository_id == repository_id)
            )
            disc_rows = db.execute(stmt_disc).all()
            for emb, disc in disc_rows:
                vec = np.frombuffer(emb.embedding_blob, dtype=np.float32).copy()
                vectors_list.append(vec)
                metadata_list.append(
                    {
                        "entity_type": "discussion",
                        "id": disc.id,
                        "number": disc.discussion_number,
                        "title": disc.title,
                        "state": "answered" if disc.is_answered else "open",
                        "category": disc.category_name,
                        "author": disc.author,
                        "comments_count": disc.comments_count,
                        "upvote_count": disc.upvote_count,
                        "created_at": disc.created_at.isoformat() if disc.created_at else None,
                    }
                )

        if not vectors_list:
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32), []

        matrix = np.vstack(vectors_list)
        return matrix, metadata_list

    def query_similar(
        self,
        repository_id: int,
        query_text: str,
        db: Session,
        top_k: int = 5,
        threshold: float = LOW_WEAK_THRESHOLD,
        include_discussions: bool = True,
    ) -> Dict[str, Any]:
        """Perform sub-millisecond semantic similarity search across repository items.

        Args:
            repository_id: Repository to search within.
            query_text: Natural language query text or issue content.
            db: Active database session.
            top_k: Maximum candidates to return.
            threshold: Minimum cosine similarity score.
            include_discussions: Whether to include discussion threads.

        Returns:
            Dictionary with query context, duplicate candidate alerts, and ranked results.
        """
        t0 = time.perf_counter()

        # Load repository vector matrix
        matrix, metadata = self.get_repository_vectors(
            repository_id=repository_id,
            db=db,
            include_discussions=include_discussions,
        )

        # If index is empty, auto-index and retry
        if matrix.shape[0] == 0:
            self.index_repository(repository_id, db)
            matrix, metadata = self.get_repository_vectors(
                repository_id=repository_id,
                db=db,
                include_discussions=include_discussions,
            )

        if matrix.shape[0] == 0:
            return {
                "query": query_text,
                "total_candidates_found": 0,
                "has_potential_duplicate": False,
                "highest_similarity_score": 0.0,
                "primary_recommendation": "Repository has no indexed issues or discussions.",
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "candidates": [],
            }

        # Encode query text
        query_vec = self.embedder.encode_text(query_text)

        # Exact cosine similarity via dot product of unit-normalized vectors
        scores = np.dot(matrix, query_vec)

        # Filter by threshold and rank
        qualifying_indices = np.where(scores >= threshold)[0]
        sorted_indices = qualifying_indices[np.argsort(-scores[qualifying_indices])]
        top_indices = sorted_indices[:top_k]

        candidates = []
        for rank, idx in enumerate(top_indices, start=1):
            score = round(float(scores[idx]), 4)
            item = dict(metadata[idx])
            item["rank"] = rank
            item["similarity_score"] = score

            if score >= HIGH_DUPLICATE_THRESHOLD:
                item["confidence_tier"] = "HIGH"
                item["duplicate_candidate"] = True
            elif score >= MEDIUM_RELATED_THRESHOLD:
                item["confidence_tier"] = "MEDIUM"
                item["duplicate_candidate"] = False
            else:
                item["confidence_tier"] = "LOW"
                item["duplicate_candidate"] = False

            candidates.append(item)

        max_score = round(float(np.max(scores)), 4) if len(scores) > 0 else 0.0
        has_duplicate = len(candidates) > 0 and candidates[0]["similarity_score"] >= HIGH_DUPLICATE_THRESHOLD

        if has_duplicate:
            top_match = candidates[0]
            rec = (
                f"Potential duplicate detected: {top_match['entity_type']} #{top_match['number']} "
                f"('{top_match['title']}') shares {round(top_match['similarity_score'] * 100, 1)}% "
                f"semantic similarity."
            )
        elif len(candidates) > 0:
            rec = (
                f"Found {len(candidates)} related { 'threads' if len(candidates) > 1 else 'thread' }. "
                f"Top context match: #{candidates[0]['number']} ({round(candidates[0]['similarity_score'] * 100, 1)}% similarity)."
            )
        else:
            rec = f"No semantically overlapping items found above threshold {threshold}."

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            "query": query_text,
            "total_candidates_found": len(candidates),
            "has_potential_duplicate": has_duplicate,
            "highest_similarity_score": max_score,
            "primary_recommendation": rec,
            "latency_ms": latency_ms,
            "candidates": candidates,
        }


semantic_index_service = SemanticIndexService()
