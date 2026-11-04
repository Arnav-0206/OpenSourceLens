#!/usr/bin/env python3
"""OpenSourceLens — Frontend & Backend Connectivity Diagnostic Tool (Phase 11, Step 11.1).

Verifies network reachability, CORS headers, latency benchmarks, and schema compatibility
between the Next.js frontend (port 3000) and the FastAPI backend (port 8000).

Usage:
    # Test against running FastAPI server (http://localhost:8000):
    python scripts/verify_connection.py

    # Test in programmatic / CI test mode (uses TestClient without requiring active server):
    python scripts/verify_connection.py --test-mode

    # Custom backend URL:
    python scripts/verify_connection.py --backend-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path for direct script execution
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Terminal color formatting helpers
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def format_status(ok: bool) -> str:
    return f"{GREEN}[PASS]{RESET}" if ok else f"{RED}[FAIL]{RESET}"


class DiagnosticClient:
    """Wrapper that abstracts over requests (for live server) and TestClient (for test-mode)."""

    def __init__(self, base_url: str, test_mode: bool = False):
        self.base_url = base_url.rstrip("/")
        self.test_mode = test_mode
        self._test_client = None

        if self.test_mode:
            from fastapi.testclient import TestClient
            from backend.app.main import app
            self._test_client = TestClient(app)

    def request(
        self,
        method: str,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Dict[str, str], Any, float]:
        """Perform request and measure round-trip latency in ms."""
        url = f"{self.base_url}{path}" if not self.test_mode else path
        req_headers = headers or {}
        start_time = time.perf_counter()

        if self.test_mode:
            assert self._test_client is not None
            resp = self._test_client.request(
                method=method,
                url=url,
                headers=req_headers,
                json=json_data,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            try:
                data = resp.json()
            except Exception:
                data = resp.text
            return resp.status_code, dict(resp.headers), data, elapsed_ms
        else:
            import urllib.request
            import urllib.error

            req = urllib.request.Request(
                url=url,
                headers=req_headers,
                method=method,
            )
            if json_data is not None:
                body_bytes = json.dumps(json_data).encode("utf-8")
                req.data = body_bytes
                req.add_header("Content-Type", "application/json")

            try:
                with urllib.request.urlopen(req, timeout=5) as response:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    status_code = response.status
                    resp_headers = dict(response.headers)
                    raw_body = response.read().decode("utf-8")
                    try:
                        data = json.loads(raw_body)
                    except Exception:
                        data = raw_body
                    return status_code, resp_headers, data, elapsed_ms
            except urllib.error.HTTPError as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                raw_body = e.read().decode("utf-8")
                try:
                    data = json.loads(raw_body)
                except Exception:
                    data = raw_body
                return e.code, dict(e.headers), data, elapsed_ms
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                return 0, {}, {"error": str(e)}, elapsed_ms


def run_diagnostics(client: DiagnosticClient, frontend_origin: str) -> bool:
    """Run full diagnostic test suite and print results table."""
    print(f"\n{BOLD}{CYAN}OpenSourceLens — Frontend/Backend Connectivity & CORS Diagnostics{RESET}")
    print(f"Target: {client.base_url} (Test Mode: {client.test_mode})")
    print(f"Testing Origin: {frontend_origin}\n")
    print(f"{'Check':<36} | {'Status':<10} | {'Latency':<10} | {'Details'}")
    print("-" * 80)

    all_passed = True

    # 1. Root Discovery Check
    status, _, data, lat = client.request("GET", "/", headers={"Origin": frontend_origin})
    ok = status == 200 and isinstance(data, dict) and "version" in data
    all_passed = all_passed and ok
    print(
        f"{'1. Root Discovery (GET /)':<36} | {format_status(ok):<10} | {f'{lat:.1f}ms':<10} | "
        f"{'v' + data.get('version', '?') if ok else 'Unreachable'}"
    )

    # 2. Health & DB Check
    status, _, data, lat = client.request("GET", "/health", headers={"Origin": frontend_origin})
    db_info = data.get("database", {}) if isinstance(data, dict) else {}
    ok = status == 200 and data.get("status") == "healthy" and db_info.get("status") == "connected"
    all_passed = all_passed and ok
    db_str = f"DB: {db_info.get('dialect', '?')} ({db_info.get('tables_count', 0)} tables)" if ok else "DB error"
    print(
        f"{'2. Health Probe (GET /health)':<36} | {format_status(ok):<10} | {f'{lat:.1f}ms':<10} | {db_str}"
    )

    # 3. CORS Preflight Check
    status, headers, _, lat = client.request(
        "OPTIONS",
        "/api/v1/classify",
        headers={
            "Origin": frontend_origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    cors_origin = headers.get("access-control-allow-origin")
    ok = status == 200 and (cors_origin == frontend_origin or cors_origin == "*")
    all_passed = all_passed and ok
    print(
        f"{'3. CORS Preflight (OPTIONS)':<36} | {format_status(ok):<10} | {f'{lat:.1f}ms':<10} | "
        f"Allow-Origin: {cors_origin or 'Missing'}"
    )

    # 4. Repositories Endpoint Check
    status, _, data, lat = client.request("GET", "/api/v1/repos", headers={"Origin": frontend_origin})
    ok = status == 200 and isinstance(data, list)
    all_passed = all_passed and ok
    repo_cnt = len(data) if ok else 0
    print(
        f"{'4. Repositories (GET /api/v1/repos)':<36} | {format_status(ok):<10} | {f'{lat:.1f}ms':<10} | "
        f"{repo_cnt} repositories indexed"
    )

    # 5. Global Analytics Overview
    status, _, data, lat = client.request("GET", "/api/v1/analytics/overview", headers={"Origin": frontend_origin})
    ok = status == 200 and isinstance(data, dict) and "total_issues" in data
    all_passed = all_passed and ok
    total_issues = data.get("total_issues", 0) if ok else 0
    print(
        f"{'5. Analytics (GET /overview)':<36} | {format_status(ok):<10} | {f'{lat:.1f}ms':<10} | "
        f"{total_issues:,} total issues"
    )

    # 6. ML Classification Inference
    status, _, data, lat = client.request(
        "POST",
        "/api/v1/classify",
        headers={"Origin": frontend_origin},
        json_data={
            "title": "Syntax error in type definition",
            "body": "Compiler throws unexpected token on line 42",
            "model_type": "distilbert",
        },
    )
    ok = status == 200 and isinstance(data, dict) and "predicted_label" in data
    all_passed = all_passed and ok
    cat = data.get("predicted_label", "?") if ok else "failed"
    conf = data.get("confidence_score", 0.0) if ok else 0.0
    print(
        f"{'6. ML Classify (POST /classify)':<36} | {format_status(ok):<10} | {f'{lat:.1f}ms':<10} | "
        f"Pred: {cat} ({conf*100:.1f}%)"
    )

    # 7. Model Registry Check
    status, _, data, lat = client.request("GET", "/api/v1/models", headers={"Origin": frontend_origin})
    ok = status == 200 and isinstance(data, dict) and "models" in data
    all_passed = all_passed and ok
    model_count = data.get("total_models", 0) if ok else 0
    print(
        f"{'7. Model Registry (GET /models)':<36} | {format_status(ok):<10} | {f'{lat:.1f}ms':<10} | "
        f"{model_count} models registered"
    )

    print("-" * 80)
    if all_passed:
        print(f"{BOLD}{GREEN}Result: ALL 7 DIAGNOSTIC CHECKS PASSED.{RESET}")
        print("FastAPI backend is ready to serve Next.js frontend at http://localhost:3000.\n")
    else:
        print(f"{BOLD}{RED}Result: ONE OR MORE CHECKS FAILED.{RESET}")
        print("Please check that FastAPI is running and CORS origins include http://localhost:3000.\n")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Verify Frontend-Backend Connection for OpenSourceLens")
    parser.add_argument(
        "--backend-url",
        default=os.getenv("NEXT_PUBLIC_API_URL", "http://localhost:8000").replace("/api/v1", ""),
        help="FastAPI Backend Root URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--origin",
        default="http://localhost:3000",
        help="Frontend Origin URL to test CORS for (default: http://localhost:3000)",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in-process via FastAPI TestClient (no running server needed)",
    )

    args = parser.parse_args()
    client = DiagnosticClient(base_url=args.backend_url, test_mode=args.test_mode)
    success = run_diagnostics(client, frontend_origin=args.origin)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
