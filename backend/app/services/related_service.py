"""Semantic Related Issues Retrieval Service for OpenSourceLens.

Supports dual-mode retrieval:
1. Fast indexed cache retrieval from related_issues_cache table (O(1) lookup).
2. Dynamic dense embedding similarity search using all-MiniLM-L6-v2 for ad-hoc text.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException, status
import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.models import Issue, RelatedIssuesCache
from backend.app.schemas.related import (
    RelatedIssueMatch,
    RelatedIssuesResponse,
    RelatedSearchRequest,
)
from ml.preprocessing.text_cleaner import IssueTextCleaner

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EMBEDDINGS_NPZ_PATH = PROJECT_ROOT / "models" / "related_issues" / "embeddings.npz"


class RelatedIssueService:
    """Singleton service for cached and dynamic semantic related-issue retrieval."""

    _instance: Optional[RelatedIssueService] = None

    def __new__(cls) -> RelatedIssueService:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self._cleaner = IssueTextCleaner()
        self._embedder = None
        self._embeddings: Optional[np.ndarray] = None
        self._emb_issue_ids: Optional[np.ndarray] = None
        self._initialized = True

    def _get_embedder(self) -> Any:
        """Lazy load SentenceTransformer embedder on CPU."""
        if self._embedder is None:
            from ml.related_issues.embedder import IssueEmbedder

            logger.info("Initializing IssueEmbedder (all-MiniLM-L6-v2)...")
            self._embedder = IssueEmbedder(
                model_name="all-MiniLM-L6-v2", normalize_embeddings=True
            )
        return self._embedder

    def _get_embeddings_store(self) -> Tuple[np.ndarray, np.ndarray]:
        """Lazy load precomputed candidate embeddings array from disk."""
        if self._embeddings is None or self._emb_issue_ids is None:
            logger.info("Loading precomputed embeddings from %s...", EMBEDDINGS_NPZ_PATH)
            data = np.load(EMBEDDINGS_NPZ_PATH)
            self._embeddings = data["embeddings"]  # Shape: (2328, 384)
            self._emb_issue_ids = data["issue_ids"]  # Shape: (2328,)
        return self._embeddings, self._emb_issue_ids

    def get_cached_related(
        self,
        issue_id: int,
        top_k: int = 5,
        state: Optional[str] = None,
        db: Session = None,
    ) -> RelatedIssuesResponse:
        """Retrieve precomputed top-K related issues from the relational cache.

        Args:
            issue_id: GitHub issue ID of the source query issue.
            top_k: Max candidate issues to return.
            state: Optional issue state filter ('open' or 'closed').
            db: Active SQLAlchemy database session.

        Returns:
            RelatedIssuesResponse with ranked matches and similarity scores.
        """
        t0 = time.perf_counter()

        # Verify source issue exists
        source_issue = db.scalar(select(Issue).where(Issue.id == issue_id))
        if not source_issue:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Issue ID {issue_id} not found in database.",
            )

        # Query cached relationships joined with target issue
        stmt = (
            select(RelatedIssuesCache, Issue)
            .join(Issue, RelatedIssuesCache.target_issue_id == Issue.id)
            .where(RelatedIssuesCache.source_issue_id == issue_id)
            .order_by(RelatedIssuesCache.rank.asc())
        )

        if state:
            stmt = stmt.where(Issue.state == state.lower())

        results = db.execute(stmt).all()

        matches: List[RelatedIssueMatch] = []
        for rank_idx, (edge, target) in enumerate(results[:top_k], start=1):
            matches.append(
                RelatedIssueMatch(
                    issue_id=target.id,
                    repo_name=target.repo_name,
                    issue_number=target.issue_number,
                    title=target.title,
                    state=target.state,
                    canonical_label=target.canonical_label,
                    similarity_score=edge.similarity_score,
                    rank=rank_idx,
                    created_at=target.created_at,
                    closed_at=target.closed_at,
                )
            )

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return RelatedIssuesResponse(
            query_issue_id=issue_id,
            query_text=source_issue.title,
            model_version="all-MiniLM-L6-v2",
            total_matches=len(matches),
            matches=matches,
            retrieval_time_ms=latency_ms,
        )

    def search_dynamic_related(
        self,
        request: RelatedSearchRequest,
        db: Session = None,
    ) -> RelatedIssuesResponse:
        """Embed user input text in real-time and retrieve nearest neighbor issues.

        Args:
            request: Validated RelatedSearchRequest with title, body, and filters.
            db: Active SQLAlchemy database session.

        Returns:
            RelatedIssuesResponse with ranked semantic matches.
        """
        t0 = time.perf_counter()

        clean_title = self._cleaner.clean_text(request.title)
        clean_body = self._cleaner.clean_text(request.body) if request.body else ""
        query_text = f"{clean_title} {clean_body}".strip()

        # 1. Encode query into 384-d normalized embedding
        embedder = self._get_embedder()
        query_vector = embedder.encode_text(query_text)  # Shape: (384,)

        # 2. Get candidate embeddings
        candidate_embeddings, emb_ids = self._get_embeddings_store()

        # 3. Filter candidate embeddings based on repository and state
        if request.repo_name or request.state:
            stmt = select(Issue.id)
            if request.repo_name:
                stmt = stmt.where(Issue.repo_name == request.repo_name)
            if request.state:
                stmt = stmt.where(Issue.state == request.state.lower())

            valid_ids_set = set(db.scalars(stmt).all())
            if not valid_ids_set:
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                return RelatedIssuesResponse(
                    query_issue_id=None,
                    query_text=request.title,
                    model_version="all-MiniLM-L6-v2",
                    total_matches=0,
                    matches=[],
                    retrieval_time_ms=latency_ms,
                )

            candidate_indices = [
                idx for idx, iid in enumerate(emb_ids) if iid in valid_ids_set
            ]
            if not candidate_indices:
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                return RelatedIssuesResponse(
                    query_issue_id=None,
                    query_text=request.title,
                    model_version="all-MiniLM-L6-v2",
                    total_matches=0,
                    matches=[],
                    retrieval_time_ms=latency_ms,
                )

            sub_matrix = candidate_embeddings[candidate_indices]  # Shape: (M, 384)
            sub_ids = emb_ids[candidate_indices]  # Shape: (M,)
        else:
            sub_matrix = candidate_embeddings
            sub_ids = emb_ids

        # 4. Cosine similarity via dot product (both vectors are L2-normalized)
        sim_scores = np.dot(sub_matrix, query_vector)  # Shape: (M,)

        effective_k = min(request.top_k, len(sim_scores))
        top_k_indices = np.argpartition(sim_scores, -effective_k)[-effective_k:]
        sorted_top_k = top_k_indices[np.argsort(-sim_scores[top_k_indices])]

        target_ids = [int(sub_ids[cand_idx]) for cand_idx in sorted_top_k]
        top_issues = db.scalars(select(Issue).where(Issue.id.in_(target_ids))).all()
        issue_dict = {iss.id: iss for iss in top_issues}

        matches: List[RelatedIssueMatch] = []
        for rank_0, cand_idx in enumerate(sorted_top_k):
            target_id = int(sub_ids[cand_idx])
            target_iss = issue_dict.get(target_id)
            if not target_iss:
                continue
            score = float(sim_scores[cand_idx])

            matches.append(
                RelatedIssueMatch(
                    issue_id=target_iss.id,
                    repo_name=target_iss.repo_name,
                    issue_number=target_iss.issue_number,
                    title=target_iss.title,
                    state=target_iss.state,
                    canonical_label=target_iss.canonical_label,
                    similarity_score=round(score, 4),
                    rank=rank_0 + 1,
                    created_at=target_iss.created_at,
                    closed_at=target_iss.closed_at,
                )
            )

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return RelatedIssuesResponse(
            query_issue_id=None,
            query_text=request.title,
            model_version="all-MiniLM-L6-v2",
            total_matches=len(matches),
            matches=matches,
            retrieval_time_ms=latency_ms,
        )


related_service = RelatedIssueService()
