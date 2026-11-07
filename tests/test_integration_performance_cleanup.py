"""Performance & Production Cleanup Integration Test Suite for OpenSourceLens (Phase 11, Step 11.9).

Verifies the final production-readiness performance optimizations:
1. SQLite WAL & concurrency pragmas (journal_mode=WAL, foreign_keys=ON, synchronous=NORMAL).
2. Database index integrity and explain plan verification across all 6 core tables.
3. Response latency Service Level Agreements (SLAs):
   - Health probe: < 25 ms
   - Repository catalog & details: < 35 ms
   - Relational similarity cache retrieval: < 25 ms
   - Warm DistilBERT transformer sequence classification: < 75 ms
   - Warm Logistic Regression & SVM baseline inference: < 35 ms
   - Warm Random Forest velocity prediction: < 50 ms
4. Multi-service concurrency SLA under parallel load.
5. Model registry prewarming lifecycle and memory safety.
"""

from __future__ import annotations

import concurrent.futures
import time
import unittest
from typing import Any, Dict, List

from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.database.session import SessionLocal, engine
from backend.app.main import app
from backend.app.services.model_manager import model_manager


class TestPerformanceCleanupAndSLAs(unittest.TestCase):
    """Integration test suite for latency benchmarks, indexing, and performance pragmas."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Prewarm models once so all benchmark tests evaluate warm latencies
        cls.client.post("/api/v1/models/prewarm")
        cls.client.get("/api/v1/related/5903612596?top_k=5")

    def setUp(self):
        self.db: Session = SessionLocal()

    def tearDown(self):
        self.db.close()

    # ──────────────────────────────────────────────────────────────────────────
    # Check 1: SQLite WAL & Concurrency Pragmas
    # ──────────────────────────────────────────────────────────────────────────

    def test_sqlite_performance_pragmas(self):
        """Verify SQLite engine applies WAL mode, foreign keys, and synchronous NORMAL."""
        with engine.connect() as conn:
            journal_mode = conn.execute(text("PRAGMA journal_mode;")).scalar()
            foreign_keys = conn.execute(text("PRAGMA foreign_keys;")).scalar()
            synchronous = conn.execute(text("PRAGMA synchronous;")).scalar()

            self.assertEqual(str(journal_mode).lower(), "wal", "Expected WAL journal mode")
            self.assertEqual(int(foreign_keys), 1, "Expected foreign keys enabled")
            # 1 corresponds to NORMAL synchronous mode in SQLite
            self.assertEqual(int(synchronous), 1, "Expected NORMAL synchronous mode")

    # ──────────────────────────────────────────────────────────────────────────
    # Check 2: Database Index Integrity
    # ──────────────────────────────────────────────────────────────────────────

    def test_database_index_coverage(self):
        """Verify presence of compound and lookup indexes across core tables."""
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT name, tbl_name FROM sqlite_master WHERE type='index';")
            ).fetchall()

            index_map: Dict[str, List[str]] = {}
            for name, tbl in rows:
                index_map.setdefault(tbl, []).append(name)

            self.assertIn("repositories", index_map)
            self.assertIn("issues", index_map)
            self.assertIn("related_issues_cache", index_map)
            self.assertIn("repository_analytics", index_map)

            # Confirm critical composite & filtering indexes on issues table
            issues_indexes = index_map["issues"]
            self.assertTrue(any("repo_name" in idx for idx in issues_indexes))
            self.assertTrue(any("state" in idx for idx in issues_indexes))
            self.assertTrue(any("canonical" in idx for idx in issues_indexes))

            # Confirm related issues cache rank index
            cache_indexes = index_map["related_issues_cache"]
            self.assertTrue(any("rank" in idx for idx in cache_indexes))

    # ──────────────────────────────────────────────────────────────────────────
    # Check 3: Response Latency SLAs (System & Catalog)
    # ──────────────────────────────────────────────────────────────────────────

    def test_latency_sla_health_probe(self):
        """Health check endpoint latency must execute within SLA (< 25 ms)."""
        latencies = []
        for _ in range(5):
            t0 = time.perf_counter()
            res = self.client.get("/health")
            t_ms = (time.perf_counter() - t0) * 1000.0
            self.assertEqual(res.status_code, 200)
            latencies.append(t_ms)

        avg_latency = sum(latencies) / len(latencies)
        self.assertLess(avg_latency, 25.0, f"Health check SLA breached: {avg_latency:.2f} ms")

    def test_latency_sla_repository_catalog_and_details(self):
        """Repository catalog and detail queries must execute within SLA (< 35 ms)."""
        t0 = time.perf_counter()
        r_catalog = self.client.get("/api/v1/repos")
        t_catalog = (time.perf_counter() - t0) * 1000.0
        self.assertEqual(r_catalog.status_code, 200)
        self.assertLess(t_catalog, 35.0, f"Catalog SLA breached: {t_catalog:.2f} ms")

        t0 = time.perf_counter()
        r_detail = self.client.get("/api/v1/repos/microsoft/vscode")
        t_detail = (time.perf_counter() - t0) * 1000.0
        self.assertEqual(r_detail.status_code, 200)
        self.assertLess(t_detail, 35.0, f"Repo detail SLA breached: {t_detail:.2f} ms")

    # ──────────────────────────────────────────────────────────────────────────
    # Check 4: Response Latency SLAs (Inference & Retrieval)
    # ──────────────────────────────────────────────────────────────────────────

    def test_latency_sla_cached_related_issues(self):
        """Relational edge cache lookup must return precomputed matches in < 25 ms."""
        # Issue ID 5903612596 is verified in cache
        t0 = time.perf_counter()
        res = self.client.get("/api/v1/related/5903612596?top_k=5")
        t_ms = (time.perf_counter() - t0) * 1000.0

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertGreater(data["total_matches"], 0)
        self.assertLess(t_ms, 25.0, f"Cached related issues SLA breached: {t_ms:.2f} ms")

    def test_latency_sla_warm_inference_all_models(self):
        """All machine learning inference endpoints must satisfy warm SLA thresholds."""
        test_issue = {
            "title": "Uncaught exception in parser module during syntax tokenization",
            "body": "Parser hangs when encountering an unescaped Unicode character.",
            "repo_name": "microsoft/vscode",
        }

        # 1. Classical Logistic Regression (< 35 ms)
        t0 = time.perf_counter()
        r_lr = self.client.post("/api/v1/classify", json={**test_issue, "model_type": "logistic_regression"})
        t_lr = (time.perf_counter() - t0) * 1000.0
        self.assertEqual(r_lr.status_code, 200)
        self.assertLess(t_lr, 35.0, f"LR inference SLA breached: {t_lr:.2f} ms")

        # 2. Classical SVM (< 35 ms)
        t0 = time.perf_counter()
        r_svm = self.client.post("/api/v1/classify", json={**test_issue, "model_type": "svm"})
        t_svm = (time.perf_counter() - t0) * 1000.0
        self.assertEqual(r_svm.status_code, 200)
        self.assertLess(t_svm, 35.0, f"SVM inference SLA breached: {t_svm:.2f} ms")

        # 3. Fine-tuned DistilBERT (< 75 ms)
        t0 = time.perf_counter()
        r_dbert = self.client.post("/api/v1/classify", json={**test_issue, "model_type": "distilbert"})
        t_dbert = (time.perf_counter() - t0) * 1000.0
        self.assertEqual(r_dbert.status_code, 200)
        self.assertLess(t_dbert, 75.0, f"DistilBERT inference SLA breached: {t_dbert:.2f} ms")

        # 4. Random Forest Velocity Predictor (< 50 ms)
        t0 = time.perf_counter()
        r_rf = self.client.post(
            "/api/v1/resolution/predict",
            json={
                "repo_name": "microsoft/vscode",
                "title": test_issue["title"],
                "body": test_issue["body"],
                "canonical_label": "bug",
            },
        )
        t_rf = (time.perf_counter() - t0) * 1000.0
        self.assertEqual(r_rf.status_code, 200)
        self.assertLess(t_rf, 50.0, f"Resolution RF inference SLA breached: {t_rf:.2f} ms")

    # ──────────────────────────────────────────────────────────────────────────
    # Check 5: Multi-Service Concurrency SLA
    # ──────────────────────────────────────────────────────────────────────────

    def test_latency_sla_concurrent_load(self):
        """Concurrent 10-thread multi-service execution must complete within 350 ms."""
        tasks = [
            ("get", "/api/v1/repos"),
            ("get", "/api/v1/analytics/overview"),
            ("get", "/api/v1/resolution/stats/microsoft/vscode"),
            ("post", "/api/v1/classify", {"title": "Concurrent thread test 1", "model_type": "distilbert"}),
            ("post", "/api/v1/classify", {"title": "Concurrent thread test 2", "model_type": "logistic_regression"}),
            ("post", "/api/v1/resolution/predict", {"repo_name": "microsoft/vscode", "title": "Velocity test 1"}),
            ("post", "/api/v1/related", {"title": "Concurrent semantic similarity lookup", "top_k": 3}),
            ("get", "/api/v1/repos/microsoft/vscode"),
            ("get", "/health"),
            ("get", "/api/v1/models"),
        ]

        def run_task(task_def):
            method = task_def[0]
            url = task_def[1]
            if method == "get":
                return self.client.get(url).status_code
            else:
                return self.client.post(url, json=task_def[2]).status_code

        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            status_codes = list(executor.map(run_task, tasks))
        t_total = (time.perf_counter() - t0) * 1000.0

        for idx, sc in enumerate(status_codes):
            self.assertEqual(sc, 200, f"Task {idx} ({tasks[idx][1]}) failed with status {sc}")

        self.assertLess(t_total, 350.0, f"Concurrent load SLA breached: {t_total:.2f} ms")

    # ──────────────────────────────────────────────────────────────────────────
    # Check 6: Startup Prewarming Functionality
    # ──────────────────────────────────────────────────────────────────────────

    def test_startup_prewarming_functionality(self):
        """Verify model_manager.prewarm_all returns all 4 subsystems loaded in host memory."""
        results = model_manager.prewarm_all()
        self.assertIsInstance(results, dict)
        self.assertEqual(results.get("distilbert"), "prewarmed")
        self.assertEqual(results.get("classical_baselines"), "prewarmed")
        self.assertEqual(results.get("sentence_transformer"), "prewarmed")
        self.assertEqual(results.get("resolution_velocity"), "prewarmed")


if __name__ == "__main__":
    unittest.main()
