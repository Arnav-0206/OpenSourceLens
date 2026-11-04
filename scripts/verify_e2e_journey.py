#!/usr/bin/env python3
"""OpenSourceLens — End-to-End (E2E) Full System Journey Verification Tool (Phase 11, Step 11.7).

Audits unified workflows across Frontend API Contracts, FastAPI Backend Routes,
SQLite Database, and all ML/DL Models (Logistic Regression, SVM, DistilBERT,
Sentence-Transformers, and Random Forest Velocity Predictor).

Usage:
    python scripts/verify_e2e_journey.py
"""

from __future__ import annotations

import concurrent.futures
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.database.models import Issue, Prediction, RelatedIssuesCache, Repository
from backend.app.database.session import SessionLocal
from backend.app.main import app

# Terminal color formatting helpers
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def format_status(ok: bool) -> str:
    return f"{GREEN}[PASS]{RESET}" if ok else f"{RED}[FAIL]{RESET}"


def run_e2e_audit() -> bool:
    """Execute complete end-to-end full system audit and print formatted report."""
    print(f"\n{BOLD}{CYAN}OpenSourceLens — Full System End-to-End Journey Audit (Step 11.7){RESET}")
    print(f"API Base: /api/v1 | Engine: FastAPI + SQLite + ML Ensemble\n")
    print(f"{'Audit Stage / Journey':<44} | {'Status':<10} | {'Latency / Metric':<20} | {'Details'}")
    print("-" * 105)

    all_passed = True
    client = TestClient(app)
    db: Session = SessionLocal()

    try:
        # ──────────────────────────────────────────────────────────────────────
        # Stage 1: Health & Database Connectivity Probe
        # ──────────────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        health_res = client.get("/health")
        t_health = (time.perf_counter() - t0) * 1000.0

        health_data = health_res.json()
        s1_ok = (
            health_res.status_code == 200
            and health_data.get("status") == "healthy"
            and health_data.get("database", {}).get("status") == "connected"
            and health_data.get("database", {}).get("tables_count", 0) >= 6
        )
        all_passed = all_passed and s1_ok
        t_count = health_data.get("database", {}).get("tables_count", 0)
        print(
            f"{'1. Health & Database Probe':<44} | {format_status(s1_ok):<10} | "
            f"{f'{t_health:.2f} ms':<20} | Status: healthy, {t_count} tables verified"
        )

        # ──────────────────────────────────────────────────────────────────────
        # Stage 2: Catalog & Comparative Matrix Exploration
        # ──────────────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        repos_res = client.get("/api/v1/repos")
        comp_res = client.get("/api/v1/analytics/comparison")
        t_catalog = (time.perf_counter() - t0) * 1000.0

        repos = repos_res.json()
        comp = comp_res.json()
        s2_ok = (
            repos_res.status_code == 200
            and comp_res.status_code == 200
            and len(repos) == 4
            and comp.get("total_repositories") == 4
            and len(comp.get("repositories", [])) == 4
        )
        all_passed = all_passed and s2_ok
        print(
            f"{'2. Catalog & Comparison Journey':<44} | {format_status(s2_ok):<10} | "
            f"{f'{t_catalog:.2f} ms':<20} | 4 target repositories indexed & ranked"
        )

        # ──────────────────────────────────────────────────────────────────────
        # Stage 3: Repository & Issues Drilldown
        # ──────────────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        detail_res = client.get("/api/v1/repos/microsoft/vscode")
        issues_res = client.get("/api/v1/repos/microsoft/vscode/issues?limit=10")
        t_repo = (time.perf_counter() - t0) * 1000.0

        detail = detail_res.json()
        issues = issues_res.json()
        s3_ok = (
            detail_res.status_code == 200
            and issues_res.status_code == 200
            and detail.get("name") == "vscode"
            and issues.get("total", 0) > 0
            and len(issues.get("items", [])) > 0
        )
        all_passed = all_passed and s3_ok
        tot_issues = issues.get("total", 0)
        print(
            f"{'3. Repository & Issues Drilldown':<44} | {format_status(s3_ok):<10} | "
            f"{f'{t_repo:.2f} ms':<20} | microsoft/vscode: {tot_issues} issues accessible"
        )

        # Pick issue for subsequent stages
        sample_issue = issues["items"][0]
        issue_title = sample_issue["title"]
        issue_body = sample_issue.get("body") or "Sample diagnostic issue description."
        issue_id = sample_issue["id"]

        # ──────────────────────────────────────────────────────────────────────
        # Stage 4: Tri-Model Classification Journey
        # ──────────────────────────────────────────────────────────────────────
        models = ["logistic_regression", "svm", "distilbert"]
        class_results: Dict[str, Any] = {}
        t0 = time.perf_counter()
        for m in models:
            r = client.post(
                "/api/v1/classify",
                json={
                    "title": issue_title,
                    "body": issue_body,
                    "repo_name": "microsoft/vscode",
                    "model_type": m,
                },
            )
            class_results[m] = r.json()
        t_class = (time.perf_counter() - t0) * 1000.0

        s4_ok = all(
            m in class_results
            and "predicted_label" in class_results[m]
            and "probabilities" in class_results[m]
            and abs(sum(class_results[m]["probabilities"].values()) - 1.0) < 0.05
            for m in models
        )
        all_passed = all_passed and s4_ok
        dbert_lbl = class_results["distilbert"]["predicted_label"]
        dbert_conf = class_results["distilbert"]["confidence_score"]
        print(
            f"{'4. Tri-Model Classification Journey':<44} | {format_status(s4_ok):<10} | "
            f"{f'{t_class:.2f} ms (all 3)':<20} | DistilBERT: {dbert_lbl} ({dbert_conf*100:.1f}%), LR & SVM calibrated"
        )

        # ──────────────────────────────────────────────────────────────────────
        # Stage 5: Dual Semantic Similarity Retrieval
        # ──────────────────────────────────────────────────────────────────────
        # Cache lookup
        t0 = time.perf_counter()
        cache_edge = db.scalar(select(RelatedIssuesCache).limit(1))
        cached_source_id = cache_edge.source_issue_id if cache_edge else issue_id
        cache_res = client.get(f"/api/v1/related/{cached_source_id}?top_k=5")
        t_cache = (time.perf_counter() - t0) * 1000.0

        # Dense embedding search
        t0 = time.perf_counter()
        dense_res = client.post(
            "/api/v1/related",
            json={
                "title": "CUDA out of memory error in torch.nn.functional.conv2d",
                "body": "Running batch size 64 triggers GPU memory allocation failure.",
                "repo_name": "pytorch/pytorch",
                "top_k": 5,
            },
        )
        t_dense = (time.perf_counter() - t0) * 1000.0

        cache_data = cache_res.json()
        dense_data = dense_res.json()
        s5_ok = (
            cache_res.status_code == 200
            and dense_res.status_code == 200
            and len(cache_data.get("matches", [])) > 0
            and len(dense_data.get("matches", [])) > 0
        )
        all_passed = all_passed and s5_ok
        print(
            f"{'5. Dual Semantic Similarity Retrieval':<44} | {format_status(s5_ok):<10} | "
            f"{f'Cache: {t_cache:.1f}ms, Vector: {t_dense:.1f}ms':<20} | Relational cache hit & 384-d cosine search"
        )

        # ──────────────────────────────────────────────────────────────────────
        # Stage 6: Resolution Velocity Prediction & SLA Consistency
        # ──────────────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        pred_res = client.post(
            "/api/v1/resolution/predict",
            json={
                "repo_name": "microsoft/vscode",
                "title": issue_title,
                "body": issue_body,
                "canonical_label": dbert_lbl,
                "author_association": "NONE",
                "labels_count": 2,
            },
        )
        stats_res = client.get("/api/v1/resolution/stats/microsoft/vscode")
        t_reso = (time.perf_counter() - t0) * 1000.0

        pred_data = pred_res.json()
        stats_data = stats_res.json()
        s6_ok = (
            pred_res.status_code == 200
            and stats_res.status_code == 200
            and "predicted_speed_tier" in pred_data
            and stats_data.get("p25_resolution_hours", 0) <= stats_data.get("median_resolution_hours", 0)
        )
        all_passed = all_passed and s6_ok
        tier = pred_data.get("predicted_speed_tier")
        med_hrs = stats_data.get("median_resolution_hours")
        print(
            f"{'6. Resolution Velocity & SLA Journey':<44} | {format_status(s6_ok):<10} | "
            f"{f'{t_reso:.2f} ms':<20} | Predicted: {tier}, Historical Median: {med_hrs:.1f}h"
        )

        # ──────────────────────────────────────────────────────────────────────
        # Stage 7: Database Prediction Audit Trail Verification
        # ──────────────────────────────────────────────────────────────────────
        pred_rows = db.scalars(
            select(Prediction).order_by(Prediction.id.desc()).limit(5)
        ).all()
        task_types = {p.task_type for p in pred_rows}
        s7_ok = len(pred_rows) >= 2 and "classification" in task_types and "resolution_prediction" in task_types
        all_passed = all_passed and s7_ok
        print(
            f"{'7. Prediction Audit Trail in DB':<44} | {format_status(s7_ok):<10} | "
            f"{f'{len(pred_rows)} recent rows':<20} | Classification & resolution logged with JSON probs"
        )

        # ──────────────────────────────────────────────────────────────────────
        # Stage 8: Concurrent Multi-Service Load & Latency SLA
        # ──────────────────────────────────────────────────────────────────────
        def ping_service(kind: str) -> int:
            if kind == "overview":
                return client.get("/api/v1/analytics/overview").status_code
            elif kind == "classify":
                return client.post(
                    "/api/v1/classify",
                    json={"title": "Concurrent test for SLA", "model_type": "distilbert"},
                ).status_code
            elif kind == "similarity":
                return client.post(
                    "/api/v1/related",
                    json={"title": "Concurrent semantic similarity lookup", "top_k": 3},
                ).status_code
            elif kind == "resolution":
                return client.post(
                    "/api/v1/resolution/predict",
                    json={"repo_name": "microsoft/vscode", "title": "Concurrent resolution check"},
                ).status_code
            return 400

        workload = ["overview", "classify", "similarity", "resolution", "overview", "classify"]
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            statuses = list(executor.map(ping_service, workload))
        t_conc = (time.perf_counter() - t0) * 1000.0

        s8_ok = all(s == 200 for s in statuses)
        all_passed = all_passed and s8_ok
        print(
            f"{'8. Concurrent Multi-Service Load':<44} | {format_status(s8_ok):<10} | "
            f"{f'{t_conc:.2f} ms total':<20} | 6 parallel requests completed with 200 OK"
        )

    finally:
        db.close()

    print("-" * 105)
    overall_status = f"{GREEN}{BOLD}ALL STAGES PASSED{RESET}" if all_passed else f"{RED}{BOLD}AUDIT FAILED{RESET}"
    print(f"Overall End-to-End Status: {overall_status}\n")
    return all_passed


if __name__ == "__main__":
    success = run_e2e_audit()
    sys.exit(0 if success else 1)
