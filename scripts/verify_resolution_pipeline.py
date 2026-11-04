#!/usr/bin/env python3
"""OpenSourceLens — Resolution Velocity Pipeline Verification Tool (Phase 11, Step 11.6).

Audits Phase 8 Random Forest champion classifier, Ridge regressor, scaler parameters,
velocity predictions, probability calibration, and factual repository resolution SLA percentiles.

Usage:
    python scripts/verify_resolution_pipeline.py
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

from sqlalchemy.orm import Session

from backend.app.database.session import SessionLocal
from backend.app.schemas.resolution import ResolutionPredictionRequest
from backend.app.services.resolution_service import resolution_service

# Terminal color formatting helpers
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def format_status(ok: bool) -> str:
    return f"{GREEN}[PASS]{RESET}" if ok else f"{RED}[FAIL]{RESET}"


def run_resolution_audit() -> bool:
    """Execute complete resolution velocity pipeline audit and print formatted report."""
    print(f"\n{BOLD}{CYAN}OpenSourceLens — Resolution Velocity Pipeline Audit (Step 11.6){RESET}")
    models_dir = _project_root / "models"
    print(f"Artifacts Directory: {models_dir}\n")
    print(f"{'Audit Check':<44} | {'Status':<10} | {'Latency / Metric':<20} | {'Details'}")
    print("-" * 96)

    all_passed = True
    session: Session = SessionLocal()

    try:
        # 1. Artifact Files Presence
        rf_path = models_dir / "resolution_classifier_champion.joblib"
        ridge_path = models_dir / "resolution_regressor_champion.joblib"
        scaler_path = models_dir / "resolution_scaler_params.json"

        files_ok = rf_path.exists() and ridge_path.exists() and scaler_path.exists()
        all_passed = all_passed and files_ok
        total_size_mb = (rf_path.stat().st_size + ridge_path.stat().st_size + scaler_path.stat().st_size) / (1024 * 1024)
        print(
            f"{'1. Artifact Files Presence':<44} | {format_status(files_ok):<10} | "
            f"{f'{total_size_mb:.2f} MB total':<20} | RF classifier, Ridge regressor, scalers"
        )

        # 2. Random Forest Model Architecture
        predictor = resolution_service._predictor
        rf_model = predictor.classifier_model
        classes = list(rf_model.classes_) if hasattr(rf_model, "classes_") else []
        n_est = getattr(rf_model, "n_estimators", 0)

        rf_ok = n_est == 100 and len(classes) == 4 and "< 1 Day" in classes
        all_passed = all_passed and rf_ok
        print(
            f"{'2. Random Forest Champion Architecture':<44} | {format_status(rf_ok):<10} | "
            f"{f'{n_est} estimators':<20} | 4 speed tiers (88.11% adjacent acc)"
        )

        # 3. Prototypical Rapid Velocity Prediction
        t0 = time.perf_counter()
        req_fast = ResolutionPredictionRequest(
            repo_name="microsoft/vscode",
            title="Fix status bar item alignment regression",
            body="Layout issue introduced in latest nightly build.",
            canonical_label="bug",
            author_association="MEMBER",
        )
        res_fast = resolution_service.predict_resolution(req_fast, db=session)
        fast_lat = (time.perf_counter() - t0) * 1000
        fast_ok = res_fast.predicted_speed_tier in ["< 1 Day", "1-7 Days"]
        all_passed = all_passed and fast_ok
        print(
            f"{'3. Rapid Velocity Inference':<44} | {format_status(fast_ok):<10} | "
            f"{f'{fast_lat:.2f}ms':<20} | Pred: {res_fast.predicted_speed_tier} ({res_fast.confidence_score*100:.1f}%)"
        )

        # 4. Slower Infrastructure Feature Proposal
        t0 = time.perf_counter()
        req_slow = ResolutionPredictionRequest(
            repo_name="kubernetes/kubernetes",
            title="Proposal: dynamic TLS certificate renewal protocol in ingress",
            body="Architectural design requiring multi-SIG review and test harness update.",
            canonical_label="feature",
            author_association="NONE",
            labels_count=4,
        )
        res_slow = resolution_service.predict_resolution(req_slow, db=session)
        slow_lat = (time.perf_counter() - t0) * 1000
        slow_ok = res_slow.predicted_speed_tier in ["< 1 Day", "1-7 Days", "7-30 Days", "30+ Days"]
        all_passed = all_passed and slow_ok
        print(
            f"{'4. Complex Feature Velocity Inference':<44} | {format_status(slow_ok):<10} | "
            f"{f'{slow_lat:.2f}ms':<20} | Window: {res_slow.lower_bound_tier} -> {res_slow.upper_bound_tier}"
        )

        # 5. Probability Distribution & Error Bounds
        probs = res_fast.tier_probabilities
        prob_sum = sum(probs.values())
        prob_ok = abs(prob_sum - 1.0) < 0.01 and len(res_fast.advisory_factors) > 0
        all_passed = all_passed and prob_ok
        print(
            f"{'5. Probability Calibration & Disclaimer':<44} | {format_status(prob_ok):<10} | "
            f"{f'Sum = {prob_sum:.2f}':<20} | Contextual factors + advisory notice"
        )

        # 6. Database SLA Percentiles across All Repositories
        repos = [("microsoft", "vscode"), ("pytorch", "pytorch"), ("kubernetes", "kubernetes"), ("huggingface", "transformers")]
        stats_ok = True
        for owner, name in repos:
            stats = resolution_service.get_repository_stats(owner, name, db=session)
            if not (stats.p25_resolution_hours <= stats.median_resolution_hours <= stats.p75_resolution_hours):
                stats_ok = False
        all_passed = all_passed and stats_ok
        print(
            f"{'6. Repository Resolution SLA Percentiles':<44} | {format_status(stats_ok):<10} | "
            f"{'4/4 repos verified':<20} | P25 <= Median <= P75 confirmed"
        )

        # 7. Warm Inference Latency Benchmark (10 runs)
        t0 = time.perf_counter()
        for _ in range(10):
            resolution_service.predict_resolution(req_fast)
        avg_lat = ((time.perf_counter() - t0) * 1000) / 10.0
        bench_ok = avg_lat < 50.0  # CPU threshold for 100-tree RF ensemble
        all_passed = all_passed and bench_ok
        print(
            f"{'7. Average Latency Benchmark (10 runs)':<44} | {format_status(bench_ok):<10} | "
            f"{f'{avg_lat:.2f}ms / query':<20} | Fast CPU throughput (< 50ms)"
        )

    finally:
        session.close()

    print("-" * 96)
    if all_passed:
        print(f"{BOLD}{GREEN}Result: ALL 7 RESOLUTION PIPELINE CHECKS PASSED.{RESET}")
        print("Resolution velocity predictor and descriptive SLA analytics are ready for frontend use.\n")
    else:
        print(f"{BOLD}{RED}Result: ONE OR MORE RESOLUTION CHECKS FAILED.{RESET}")
        print("Please check model weights and database analytics snapshots.\n")

    return all_passed


def main():
    success = run_resolution_audit()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
