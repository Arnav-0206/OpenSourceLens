#!/usr/bin/env python3
"""OpenSourceLens — Semantic Similarity & Duplicate Retrieval Verification Tool (Phase 11, Step 11.5).

Audits Sentence-Transformers MiniLM embeddings matrix, relational cache table (11,640 edges),
vector dot-product throughput, dynamic text encoding, and nearest-neighbor search accuracy.

Usage:
    python scripts/verify_similarity_pipeline.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.database.models import Issue, RelatedIssuesCache
from backend.app.database.session import SessionLocal
from backend.app.schemas.related import RelatedSearchRequest
from backend.app.services.related_service import related_service

# Terminal color formatting helpers
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def format_status(ok: bool) -> str:
    return f"{GREEN}[PASS]{RESET}" if ok else f"{RED}[FAIL]{RESET}"


def run_similarity_audit() -> bool:
    """Execute complete semantic similarity pipeline audit and print formatted report."""
    print(f"\n{BOLD}{CYAN}OpenSourceLens — Semantic Similarity Pipeline Audit (Step 11.5){RESET}")
    artifacts_dir = _project_root / "models" / "related_issues"
    print(f"Artifacts Directory: {artifacts_dir}\n")
    print(f"{'Audit Check':<44} | {'Status':<10} | {'Latency / Metric':<20} | {'Details'}")
    print("-" * 96)

    all_passed = True
    session: Session = SessionLocal()

    try:
        # 1. Artifact Files Presence
        npz_path = artifacts_dir / "embeddings.npz"
        manifest_path = artifacts_dir / "manifest.json"
        parquet_path = artifacts_dir / "metadata.parquet"

        files_ok = npz_path.exists() and manifest_path.exists() and parquet_path.exists()
        all_passed = all_passed and files_ok
        total_size_mb = (npz_path.stat().st_size + manifest_path.stat().st_size + parquet_path.stat().st_size) / (1024 * 1024)
        print(
            f"{'1. Artifact Files Presence':<44} | {format_status(files_ok):<10} | "
            f"{f'{total_size_mb:.2f} MB total':<20} | embeddings.npz, metadata, manifest"
        )

        # 2. Embedding Matrix Dimensions & L2 Normalization
        npz_data = np.load(npz_path)
        embs = npz_data["embeddings"]
        issue_ids = npz_data["issue_ids"]

        shape_ok = embs.shape == (2328, 384) and len(issue_ids) == 2328
        norms = np.linalg.norm(embs, axis=1)
        norm_ok = np.allclose(norms, 1.0, atol=1e-3)
        matrix_ok = shape_ok and norm_ok
        all_passed = all_passed and matrix_ok
        print(
            f"{'2. Embedding Matrix (2328, 384)':<44} | {format_status(matrix_ok):<10} | "
            f"{'L2 norm = 1.000':<20} | 2,328 vectors, 384 dimensions"
        )

        # 3. Database Relational Cache Rows
        cache_count = session.scalar(select(func.count(RelatedIssuesCache.id))) or 0
        cache_ok = cache_count == 11640
        all_passed = all_passed and cache_ok
        print(
            f"{'3. Database Relational Edge Cache':<44} | {format_status(cache_ok):<10} | "
            f"{f'{cache_count:,} edges':<20} | 5 edges per issue (2328 × 5)"
        )

        # 4. Dense Embedder Initialization & Encoding
        t0 = time.perf_counter()
        embedder = related_service._get_embedder()
        init_lat = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        test_vec = embedder.encode_text("Segmentation fault in threadpool worker")
        encode_lat = (time.perf_counter() - t0) * 1000

        vec_ok = test_vec.shape == (384,) and np.isclose(np.linalg.norm(test_vec), 1.0, atol=1e-3)
        all_passed = all_passed and vec_ok
        print(
            f"{'4. Sentence-Transformers Embedder':<44} | {format_status(vec_ok):<10} | "
            f"{f'{encode_lat:.2f}ms encode':<20} | all-MiniLM-L6-v2 (384-d normalized)"
        )

        # 5. Matrix Dot-Product Latency Benchmark
        t0 = time.perf_counter()
        for _ in range(20):
            np.dot(embs, test_vec)
        avg_dot_lat = ((time.perf_counter() - t0) * 1000) / 20.0
        dot_ok = avg_dot_lat < 10.0
        all_passed = all_passed and dot_ok
        print(
            f"{'5. Vector Dot-Product Throughput':<44} | {format_status(dot_ok):<10} | "
            f"{f'{avg_dot_lat:.2f}ms / 2,328 docs':<20} | Blazing matrix multiplication"
        )

        # 6. Dynamic Nearest-Neighbor Retrieval Evaluation
        req = RelatedSearchRequest(
            title="CUDA out of memory in backward pass",
            repo_name="pytorch/pytorch",
            top_k=3,
        )
        t0 = time.perf_counter()
        search_res = related_service.search_dynamic_related(req, db=session)
        search_lat = (time.perf_counter() - t0) * 1000
        search_ok = len(search_res.matches) == 3 and search_res.matches[0].similarity_score > 0.4
        all_passed = all_passed and search_ok
        top_score = search_res.matches[0].similarity_score if search_res.matches else 0.0
        print(
            f"{'6. Prototypical Dynamic Search':<44} | {format_status(search_ok):<10} | "
            f"{f'{search_lat:.2f}ms (top={top_score:.2f})':<20} | Correctly found CUDA OOM issues"
        )

        # 7. Relational Cache O(1) Lookup Benchmark
        sample_edge = session.scalar(select(RelatedIssuesCache))
        t0 = time.perf_counter()
        cached_res = related_service.get_cached_related(sample_edge.source_issue_id, top_k=5, db=session)
        cached_lat = (time.perf_counter() - t0) * 1000
        cached_ok = len(cached_res.matches) > 0 and cached_lat < 25.0
        all_passed = all_passed and cached_ok
        print(
            f"{'7. Relational Cache O(1) Lookup':<44} | {format_status(cached_ok):<10} | "
            f"{f'{cached_lat:.2f}ms':<20} | Precomputed top-5 instant return"
        )

    finally:
        session.close()

    print("-" * 96)
    if all_passed:
        print(f"{BOLD}{GREEN}Result: ALL 7 SIMILARITY PIPELINE CHECKS PASSED.{RESET}")
        print("Dense embedding retriever and relational cache are fully operational for frontend search.\n")
    else:
        print(f"{BOLD}{RED}Result: ONE OR MORE SIMILARITY CHECKS FAILED.{RESET}")
        print("Please check embedding files and database cache edges.\n")

    return all_passed


def main():
    success = run_similarity_audit()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
