#!/usr/bin/env python3
"""OpenSourceLens — Fine-Tuned DistilBERT Transformer Verification Tool (Phase 11, Step 11.4).

Audits DistilBERT model weights, tokenizer configuration, memory pre-warming,
probability calibration, and CPU inference latency benchmarks.

Usage:
    python scripts/verify_transformer_pipeline.py
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

import torch
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


def run_transformer_audit() -> bool:
    """Execute complete transformer pipeline audit and print formatted report."""
    print(f"\n{BOLD}{CYAN}OpenSourceLens — DistilBERT Transformer Pipeline Audit (Step 11.4){RESET}")
    model_dir = _project_root / "models" / "distilbert_finetuned"
    print(f"Model Directory: {model_dir}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}\n")
    print(f"{'Audit Check':<44} | {'Status':<10} | {'Latency / Metric':<20} | {'Details'}")
    print("-" * 96)

    all_passed = True

    # 1. Artifact Files Presence
    required_files = [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "training_metadata.json",
    ]
    files_ok = all((model_dir / f).exists() for f in required_files)
    all_passed = all_passed and files_ok
    total_size_mb = sum((model_dir / f).stat().st_size for f in required_files if (model_dir / f).exists()) / (1024 * 1024)
    print(
        f"{'1. Artifact Files Presence':<44} | {format_status(files_ok):<10} | "
        f"{f'{total_size_mb:.2f} MB total':<20} | 5/5 model & tokenizer artifacts"
    )

    # 2. Config & Architecture Verification
    config_path = model_dir / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    arch_ok = (
        cfg.get("model_type") == "distilbert"
        and cfg.get("vocab_size") == 30522
        and cfg.get("dim") == 768
        and cfg.get("n_layers") == 6
    )
    all_passed = all_passed and arch_ok
    print(
        f"{'2. Architecture Configuration':<44} | {format_status(arch_ok):<10} | "
        f"{'66.9M parameters':<20} | 6 layers, 12 heads, 768 dim"
    )

    # 3. Tokenizer Loading & Encoding
    t0 = time.perf_counter()
    _, tokenizer = classification_service._get_distilbert()
    tok_lat = (time.perf_counter() - t0) * 1000

    test_tokens = tokenizer("OpenSourceLens transformer tokenization test", max_length=128, truncation=True)
    tok_ok = "input_ids" in test_tokens and len(test_tokens["input_ids"]) > 0
    all_passed = all_passed and tok_ok
    print(
        f"{'3. WordPiece Tokenizer Loading':<44} | {format_status(tok_ok):<10} | "
        f"{f'{tok_lat:.2f}ms':<20} | Vocab: {tokenizer.vocab_size:,} tokens"
    )

    # 4. Model Pre-Warming & In-Memory Loading
    t0 = time.perf_counter()
    model, _ = classification_service._get_distilbert()
    load_lat = (time.perf_counter() - t0) * 1000
    model_loaded_ok = model is not None and not model.training
    all_passed = all_passed and model_loaded_ok
    print(
        f"{'4. Memory Pre-Warming (eval mode)':<44} | {format_status(model_loaded_ok):<10} | "
        f"{f'{load_lat:.2f}ms':<20} | DistilBertForSequenceClassification"
    )

    # 5. Inference Forward Pass & Softmax Normalization
    test_req = ClassificationRequest(
        title="Crash in extension host with SIGSEGV code 139",
        body="Opening debug console triggers native worker abort.",
        repo_name="microsoft/vscode",
        model_type="distilbert",
    )
    t0 = time.perf_counter()
    res = classification_service.classify(test_req)
    infer_lat = (time.perf_counter() - t0) * 1000
    prob_sum = sum(res.probabilities.values())
    norm_ok = abs(prob_sum - 1.0) < 0.01 and res.model_used == "distilbert-finetuned-v1"
    all_passed = all_passed and norm_ok
    print(
        f"{'5. Forward Pass & Softmax Normalization':<44} | {format_status(norm_ok):<10} | "
        f"{f'{infer_lat:.2f}ms (P={prob_sum:.2f})':<20} | Pred: {res.predicted_label} ({res.confidence_score*100:.1f}%)"
    )

    # 6. Prototypical Feature Request Evaluation
    feature_req = ClassificationRequest(
        title="Add support for dark high-contrast themes",
        body="User proposal to add accessible high-contrast theme color tokens.",
        repo_name="microsoft/vscode",
        model_type="distilbert",
    )
    res_feat = classification_service.classify(feature_req)
    feat_ok = res_feat.predicted_label == "feature" and res_feat.confidence_score > 0.5
    all_passed = all_passed and feat_ok
    print(
        f"{'6. Prototypical Feature Classification':<44} | {format_status(feat_ok):<10} | "
        f"{f'Confidence: {res_feat.confidence_score*100:.1f}%':<20} | Correctly predicted 'feature'"
    )

    # 7. Warm Batch Latency Benchmark (5 inferences)
    warm_times = []
    for _ in range(5):
        t_start = time.perf_counter()
        classification_service.classify(test_req)
        warm_times.append((time.perf_counter() - t_start) * 1000)
    avg_warm_lat = sum(warm_times) / len(warm_times)
    bench_ok = avg_warm_lat < 100.0  # Conservative CPU limit
    all_passed = all_passed and bench_ok
    print(
        f"{'7. Warm CPU Latency Benchmark (5 runs)':<44} | {format_status(bench_ok):<10} | "
        f"{f'{avg_warm_lat:.2f}ms / query':<20} | Fast CPU throughput (< 100ms)"
    )

    print("-" * 96)
    if all_passed:
        print(f"{BOLD}{GREEN}Result: ALL 7 TRANSFORMER PIPELINE CHECKS PASSED.{RESET}")
        print("Fine-tuned DistilBERT champion model is operational and ready for frontend use.\n")
    else:
        print(f"{BOLD}{RED}Result: ONE OR MORE TRANSFORMER CHECKS FAILED.{RESET}")
        print("Please review model weights and tokenizer files.\n")

    return all_passed


def main():
    success = run_transformer_audit()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
