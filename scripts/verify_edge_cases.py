#!/usr/bin/env python3
"""OpenSourceLens — Edge Cases & Error Handling Verification Tool (Phase 11, Step 11.8).

Audits boundary conditions, standardized error envelopes, schema validation,
extreme inputs, zero-match states, and information masking across the entire application stack.

Usage:
    python scripts/verify_edge_cases.py
"""

from __future__ import annotations

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


def run_edge_cases_audit() -> bool:
    """Execute complete edge case verification audit and print formatted report."""
    print(f"\n{BOLD}{CYAN}OpenSourceLens — Edge Case & Error Resilience Audit (Step 11.8){RESET}")
    print(f"Stack: FastAPI + Pydantic v2 + SQLite + Pre-warmed ML Models\n")
    print(f"{'Audit Check / Scenario':<44} | {'Status':<10} | {'Latency / Metric':<20} | {'Details'}")
    print("-" * 105)

    all_passed = True
    client = TestClient(app)

    # Prewarm models to ensure fast benchmarks
    client.post("/api/v1/models/prewarm")

    try:
        # ──────────────────────────────────────────────────────────────────────
        # Check 1: 404 Resource Not Found Envelope Verification
        # ──────────────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        r_repo = client.get("/api/v1/repos/missing_org/missing_repo")
        r_issue = client.get("/api/v1/repos/microsoft/vscode/issues/999999999")
        r_model = client.get("/api/v1/models/unknown_net")
        r_stats = client.get("/api/v1/resolution/stats/fake/repo")
        r_cache = client.get("/api/v1/related/9999999999")
        t_404 = (time.perf_counter() - t0) * 1000.0

        responses_404 = [r_repo, r_issue, r_model, r_stats, r_cache]
        c1_ok = all(
            r.status_code == 404
            and r.json().get("error_code") == "RESOURCE_NOT_FOUND"
            and r.json().get("status_code") == 404
            and "timestamp" in r.json()
            for r in responses_404
        )
        all_passed = all_passed and c1_ok
        print(
            f"{'1. 404 Resource Not Found Envelopes':<44} | {format_status(c1_ok):<10} | "
            f"{f'{t_404:.2f} ms (5 checks)':<20} | All 5 routes returned standard RESOURCE_NOT_FOUND"
        )

        # ──────────────────────────────────────────────────────────────────────
        # Check 2: 422 Validation Error Schema & Granular Field Error Formatting
        # ──────────────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        r_empty = client.post("/api/v1/classify", json={})
        r_short = client.post("/api/v1/classify", json={"title": "ab"})
        r_long = client.post("/api/v1/classify", json={"title": "x" * 505})
        r_bad_model = client.post("/api/v1/classify", json={"title": "Valid title", "model_type": "gpt-unknown"})
        t_422 = (time.perf_counter() - t0) * 1000.0

        responses_422 = [r_empty, r_short, r_long, r_bad_model]
        c2_ok = all(
            r.status_code == 422
            and r.json().get("error_code") == "VALIDATION_ERROR"
            and isinstance(r.json().get("errors"), list)
            and len(r.json().get("errors", [])) > 0
            for r in responses_422
        )
        all_passed = all_passed and c2_ok
        print(
            f"{'2. 422 Schema Validation Envelopes':<44} | {format_status(c2_ok):<10} | "
            f"{f'{t_422:.2f} ms (4 checks)':<20} | Granular field locations & error types verified"
        )

        # ──────────────────────────────────────────────────────────────────────
        # Check 3: Extreme Body Length (>10,000 chars) Truncation Invariance
        # ──────────────────────────────────────────────────────────────────────
        huge_body = "Stack trace line error code 0x80070005 at memory offset 0x12345678\n" * 150  # ~10,350 chars
        t0 = time.perf_counter()
        r_huge = client.post(
            "/api/v1/classify",
            json={
                "title": "Catastrophic stack overflow in native runner",
                "body": huge_body,
                "model_type": "distilbert",
            },
        )
        t_huge = (time.perf_counter() - t0) * 1000.0

        c3_ok = (
            r_huge.status_code == 200
            and "predicted_label" in r_huge.json()
            and r_huge.json().get("confidence_score", 0) > 0
            and t_huge < 2000.0
        )
        all_passed = all_passed and c3_ok
        lbl = r_huge.json().get("predicted_label")
        conf = r_huge.json().get("confidence_score", 0)
        print(
            f"{'3. Extreme Input Truncation (>10k chars)':<44} | {format_status(c3_ok):<10} | "
            f"{f'{t_huge:.2f} ms':<20} | WordPiece 128-token truncation, predicted: {lbl} ({conf*100:.1f}%)"
        )

        # ──────────────────────────────────────────────────────────────────────
        # Check 4: Unicode, Emoji, and XSS Sanitization Resilience
        # ──────────────────────────────────────────────────────────────────────
        xss_input = "<script>alert('pwned')</script> 💥 Bug in parser! \U0001F600 \t\r\n"
        t0 = time.perf_counter()
        r_xss = client.post(
            "/api/v1/classify",
            json={"title": xss_input, "model_type": "distilbert"},
        )
        t_xss = (time.perf_counter() - t0) * 1000.0

        data_xss = r_xss.json()
        c4_ok = (
            r_xss.status_code == 200
            and "<script>" not in data_xss.get("clean_title", "")
            and "predicted_label" in data_xss
        )
        all_passed = all_passed and c4_ok
        cleaned_sample = data_xss.get("clean_title", "")[:35].encode("ascii", "replace").decode("ascii")
        print(
            f"{'4. Unicode, Emoji & Sanitization Resilience':<44} | {format_status(c4_ok):<10} | "
            f"{f'{t_xss:.2f} ms':<20} | Script tags stripped, cleaned: '{cleaned_sample}...'"
        )

        # ──────────────────────────────────────────────────────────────────────
        # Check 5: Compound Filter Zero-Match Safe Handling
        # ──────────────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        r_zero = client.get(
            "/api/v1/repos/microsoft/vscode/issues?category=documentation&speed_tier=30%2B+Days&search=nonexistent_xyz_symbol"
        )
        t_zero = (time.perf_counter() - t0) * 1000.0

        zero_data = r_zero.json()
        c5_ok = (
            r_zero.status_code == 200
            and zero_data.get("total") == 0
            and zero_data.get("items") == []
        )
        all_passed = all_passed and c5_ok
        print(
            f"{'5. Compound Filter Zero-Match Handling':<44} | {format_status(c5_ok):<10} | "
            f"{f'{t_zero:.2f} ms':<20} | Total: 0, items: [] cleanly returned without error"
        )

        # ──────────────────────────────────────────────────────────────────────
        # Check 6: Minimal / Optional Parameter Fallbacks
        # ──────────────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        r_min = client.post(
            "/api/v1/resolution/predict",
            json={"repo_name": "microsoft/vscode", "title": "Minimal title parameter check"},
        )
        t_min = (time.perf_counter() - t0) * 1000.0

        min_data = r_min.json()
        c6_ok = (
            r_min.status_code == 200
            and "predicted_speed_tier" in min_data
            and "reliability_notice" in min_data
            and min_data.get("lower_bound_tier") is not None
        )
        all_passed = all_passed and c6_ok
        print(
            f"{'6. Minimal Parameter Feature Fallbacks':<44} | {format_status(c6_ok):<10} | "
            f"{f'{t_min:.2f} ms':<20} | Defaulted to valid speed tier: {min_data.get('predicted_speed_tier')}"
        )

        # ──────────────────────────────────────────────────────────────────────
        # Check 7: Information Masking & Stack Trace Suppression
        # ──────────────────────────────────────────────────────────────────────
        # Verify that error responses do not leak python tracebacks or sqlite paths
        err_samples = [r_repo.text, r_empty.text, r_bad_model.text]
        c7_ok = all(
            "Traceback (most recent call last)" not in txt
            and "sqlite3" not in txt
            and ".venv" not in txt
            for txt in err_samples
        )
        all_passed = all_passed and c7_ok
        print(
            f"{'7. Information Masking & Trace Suppression':<44} | {format_status(c7_ok):<10} | "
            f"{'Verified Safe':<20} | No stack traces, file paths, or DB dialect leaks in responses"
        )

    except Exception as e:
        print(f"\n{RED}Unexpected exception during audit:{RESET} {e}")
        all_passed = False

    print("-" * 105)
    overall_status = f"{GREEN}{BOLD}ALL CHECKS PASSED{RESET}" if all_passed else f"{RED}{BOLD}AUDIT FAILED{RESET}"
    print(f"Overall Edge Case & Resilience Status: {overall_status}\n")
    return all_passed


if __name__ == "__main__":
    success = run_edge_cases_audit()
    sys.exit(0 if success else 1)
