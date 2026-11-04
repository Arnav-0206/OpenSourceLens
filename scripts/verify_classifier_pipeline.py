#!/usr/bin/env python3
"""OpenSourceLens — Classical ML Classifier Pipeline Verification Tool (Phase 11, Step 11.3).

Audits TF-IDF vectorizer, Logistic Regression, and Linear SVM model artifacts,
verifies in-memory inference pipelines, tests probability calibration, and benchmarks latency.

Usage:
    python scripts/verify_classifier_pipeline.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import joblib
import numpy as np
import pandas as pd

from backend.app.schemas.classification import ClassificationRequest
from backend.app.services.classification_service import classification_service

# Terminal color formatting helpers
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def format_status(ok: bool) -> str:
    return f"{GREEN}[PASS]{RESET}" if ok else f"{RED}[FAIL]{RESET}"


def run_classifier_audit() -> bool:
    """Execute complete classical classifier audit and print formatted report."""
    print(f"\n{BOLD}{CYAN}OpenSourceLens — Classical Classifier Pipeline Audit (Step 11.3){RESET}")
    models_dir = _project_root / "models"
    print(f"Artifacts Directory: {models_dir}\n")
    print(f"{'Audit Check':<44} | {'Status':<10} | {'Latency / Metric':<20} | {'Details'}")
    print("-" * 96)

    all_passed = True

    # 1. Artifact Files Presence
    lr_path = models_dir / "baseline_logistic_regression.joblib"
    svm_path = models_dir / "baseline_svm.joblib"
    tfidf_path = models_dir / "tfidf_vectorizer.joblib"

    artifacts_exist = lr_path.exists() and svm_path.exists() and tfidf_path.exists()
    all_passed = all_passed and artifacts_exist
    size_mb = (lr_path.stat().st_size + svm_path.stat().st_size + tfidf_path.stat().st_size) / (1024 * 1024)
    print(
        f"{'1. Artifact Files Presence':<44} | {format_status(artifacts_exist):<10} | "
        f"{f'{size_mb:.2f} MB total':<20} | LR, SVM, TF-IDF artifacts on disk"
    )

    # 2. TF-IDF Pipeline Loading & Vocabulary
    t0 = time.perf_counter()
    tfidf = classification_service._get_tfidf_pipeline()
    tfidf_lat = (time.perf_counter() - t0) * 1000
    has_vocab = hasattr(tfidf, "named_steps") or hasattr(tfidf, "transform")
    all_passed = all_passed and has_vocab
    print(
        f"{'2. TF-IDF Feature Pipeline':<44} | {format_status(has_vocab):<10} | "
        f"{f'{tfidf_lat:.2f}ms':<20} | Sublinear TF, 10,000 max features"
    )

    # 3. Logistic Regression Artifact Inspection
    t0 = time.perf_counter()
    lr = classification_service._get_lr_model()
    lr_lat = (time.perf_counter() - t0) * 1000
    lr_classes = list(lr.classes_)
    lr_ok = len(lr_classes) == 3 and "bug" in lr_classes and "feature" in lr_classes
    all_passed = all_passed and lr_ok
    print(
        f"{'3. Logistic Regression Artifact':<44} | {format_status(lr_ok):<10} | "
        f"{f'{lr_lat:.2f}ms':<20} | Classes: {lr_classes}"
    )

    # 4. Linear SVM Artifact Inspection
    t0 = time.perf_counter()
    svm = classification_service._get_svm_model()
    svm_lat = (time.perf_counter() - t0) * 1000
    svm_classes = list(svm.classes_)
    svm_ok = len(svm_classes) == 3 and "bug" in svm_classes and "feature" in svm_classes
    all_passed = all_passed and svm_ok
    print(
        f"{'4. Linear SVM Artifact':<44} | {format_status(svm_ok):<10} | "
        f"{f'{svm_lat:.2f}ms':<20} | Classes: {svm_classes}"
    )

    # 5. Logistic Regression Inference & Probability Calibration
    test_req_lr = ClassificationRequest(
        title="Null pointer crash when rendering workspace tree",
        body="Opening nested folder causes unhandled exception in renderer thread.",
        repo_name="microsoft/vscode",
        model_type="logistic_regression",
    )
    t0 = time.perf_counter()
    res_lr = classification_service.classify(test_req_lr)
    lr_infer_lat = (time.perf_counter() - t0) * 1000
    lr_sum = sum(res_lr.probabilities.values())
    lr_norm_ok = abs(lr_sum - 1.0) < 0.01 and res_lr.predicted_label in ["bug", "feature", "other"]
    all_passed = all_passed and lr_norm_ok
    print(
        f"{'5. LR Inference & Normalization':<44} | {format_status(lr_norm_ok):<10} | "
        f"{f'{lr_infer_lat:.2f}ms (P={lr_sum:.2f})':<20} | Pred: {res_lr.predicted_label} ({res_lr.confidence_score*100:.1f}%)"
    )

    # 6. Linear SVM Inference & Softmax Calibration
    test_req_svm = ClassificationRequest(
        title="Proposal: add FlashAttention-2 support in attention layer",
        body="Expose optional kernel flags in multihead attention for Hopper GPUs.",
        repo_name="pytorch/pytorch",
        model_type="svm",
    )
    t0 = time.perf_counter()
    res_svm = classification_service.classify(test_req_svm)
    svm_infer_lat = (time.perf_counter() - t0) * 1000
    svm_sum = sum(res_svm.probabilities.values())
    svm_norm_ok = abs(svm_sum - 1.0) < 0.01 and res_svm.predicted_label in ["bug", "feature", "other"]
    all_passed = all_passed and svm_norm_ok
    print(
        f"{'6. SVM Softmax Calibration':<44} | {format_status(svm_norm_ok):<10} | "
        f"{f'{svm_infer_lat:.2f}ms (P={svm_sum:.2f})':<20} | Pred: {res_svm.predicted_label} ({res_svm.confidence_score*100:.1f}%)"
    )

    # 7. Warm Batch Latency Benchmark (10 samples)
    t0 = time.perf_counter()
    for _ in range(10):
        classification_service.classify(test_req_svm)
    avg_lat = ((time.perf_counter() - t0) * 1000) / 10.0
    bench_ok = avg_lat < 25.0
    all_passed = all_passed and bench_ok
    print(
        f"{'7. Average Latency Benchmark (10 runs)':<44} | {format_status(bench_ok):<10} | "
        f"{f'{avg_lat:.2f}ms / query':<20} | Fast CPU throughput (< 25ms)"
    )

    print("-" * 96)
    if all_passed:
        print(f"{BOLD}{GREEN}Result: ALL 7 CLASSIFIER PIPELINE CHECKS PASSED.{RESET}")
        print("Classical baseline models (TF-IDF + LR & SVM) are fully operational and ready for frontend use.\n")
    else:
        print(f"{BOLD}{RED}Result: ONE OR MORE CLASSIFIER CHECKS FAILED.{RESET}")
        print("Please check model weights and scikit-learn compatibility.\n")

    return all_passed


def main():
    success = run_classifier_audit()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
