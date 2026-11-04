"""End-to-End (E2E) Full-System Integration Test Suite for OpenSourceLens (Phase 11, Step 11.7).

Verifies the unified, coupled execution pathways connecting:
Frontend API Contracts <-> FastAPI Backend Routes <-> SQLite Database <-> ML & Deep Learning Models:
1. Complete Exploration-to-AI Diagnostic User Journey:
   - Repos catalog -> comparison -> repository deep dive -> issues list -> multi-model classification,
     semantic similarity retrieval, resolution velocity prediction, and database audit persistence.
2. Tri-Model Classification Invariance & Contract Uniformity:
   - Logistic Regression vs SVM vs Fine-tuned DistilBERT on identical inputs with probability calibration.
3. Dual-Mode Semantic Retrieval:
   - Sub-millisecond relational edge cache lookup vs cold dense embedding vector search (all-MiniLM-L6-v2).
4. Descriptive & Predictive Resolution Consistency:
   - Resolution percentiles (P25 <= Median <= P75) across all 4 target repos and Random Forest velocity tiering.
5. Model Registry & Prewarming Lifecycle:
   - Metadata introspection across all 4 registered model systems and warm-up execution.
6. Database Audit Trail Integrity:
   - JSON serialization, timing metrics, and referential integrity in the 'predictions' table.
7. Concurrent Multi-Service Load & Thread-Safety:
   - Parallel multi-threaded execution across classification, similarity, resolution, and database queries.
"""

from __future__ import annotations

import concurrent.futures
import time
import unittest
from typing import Any, Dict, List

from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.database.models import Issue, Prediction, RelatedIssuesCache, Repository
from backend.app.database.session import SessionLocal
from backend.app.main import app


class TestFullSystemEndToEndJourneys(unittest.TestCase):
    """End-to-end full system integration test suite."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.target_repos = [
            ("microsoft", "vscode"),
            ("pytorch", "pytorch"),
            ("kubernetes", "kubernetes"),
            ("huggingface", "transformers"),
        ]
        cls.canonical_classes = {"bug", "feature", "documentation", "question", "other"}
        cls.valid_speed_tiers = ["< 1 Day", "1-7 Days", "7-30 Days", "30+ Days"]

    def setUp(self):
        self.db: Session = SessionLocal()

    def tearDown(self):
        self.db.close()

    # ──────────────────────────────────────────────────────────────────────────
    # Journey 1: Repository Exploration -> Issue List -> AI Multi-Diagnostic
    # ──────────────────────────────────────────────────────────────────────────

    def test_e2e_journey_repository_to_issue_to_ai_diagnosis(self):
        """User journey: Repos catalog -> comparison -> repo details -> issue selection -> AI diagnosis."""
        # Step 1: Browse repository catalog
        repos_res = self.client.get("/api/v1/repos")
        self.assertEqual(repos_res.status_code, 200)
        repos = repos_res.json()
        self.assertGreaterEqual(len(repos), 4)
        repo_names = [r["repo_name"] for r in repos]
        self.assertIn("microsoft/vscode", repo_names)

        # Step 2: Global comparative analytics
        comp_res = self.client.get("/api/v1/analytics/comparison")
        self.assertEqual(comp_res.status_code, 200)
        comp_data = comp_res.json()
        self.assertEqual(comp_data["total_repositories"], 4)
        self.assertIn("repositories", comp_data)
        self.assertEqual(len(comp_data["repositories"]), 4)

        # Step 3: Deep dive into repository details
        detail_res = self.client.get("/api/v1/repos/microsoft/vscode")
        self.assertEqual(detail_res.status_code, 200)
        detail_data = detail_res.json()
        self.assertEqual(detail_data["name"], "vscode")
        self.assertGreater(detail_data["total_analyzed_issues"], 0)

        # Step 4: Repository resolution analytics
        res_stats = self.client.get("/api/v1/resolution/stats/microsoft/vscode")
        self.assertEqual(res_stats.status_code, 200)
        stats_data = res_stats.json()
        self.assertGreater(stats_data["median_resolution_hours"], 0)

        # Step 5: Query issues list for vscode (unfiltered state to get full issues)
        issues_res = self.client.get("/api/v1/repos/microsoft/vscode/issues?limit=5")
        self.assertEqual(issues_res.status_code, 200)
        issues_page = issues_res.json()
        self.assertGreaterEqual(issues_page["total"], 1)
        self.assertGreaterEqual(len(issues_page["items"]), 1)

        selected_issue = issues_page["items"][0]
        issue_title = selected_issue["title"]
        issue_body = selected_issue.get("body") or "Reproduction steps and debugging traces."
        issue_id = selected_issue["id"]

        # Step 6: Dispatch AI Classification using champion DistilBERT
        initial_pred_count = self.db.scalar(select(func.count()).select_from(Prediction))

        class_res = self.client.post(
            "/api/v1/classify",
            json={
                "title": issue_title,
                "body": issue_body,
                "repo_name": "microsoft/vscode",
                "model_type": "distilbert",
                "issue_id": issue_id,
            },
        )
        self.assertEqual(class_res.status_code, 200)
        class_data = class_res.json()
        self.assertIn(class_data["predicted_label"], self.canonical_classes)
        self.assertGreater(class_data["confidence_score"], 0.0)
        self.assertAlmostEqual(sum(class_data["probabilities"].values()), 1.0, places=2)

        # Step 7: Dispatch Semantic Duplicate / Related Issues Search
        rel_res = self.client.post(
            "/api/v1/related",
            json={
                "title": issue_title,
                "body": issue_body,
                "repo_name": "microsoft/vscode",
                "top_k": 5,
            },
        )
        self.assertEqual(rel_res.status_code, 200)
        rel_data = rel_res.json()
        self.assertGreaterEqual(len(rel_data["matches"]), 1)
        for match in rel_data["matches"]:
            self.assertEqual(match["repo_name"], "microsoft/vscode")
            self.assertGreaterEqual(match["similarity_score"], -1.0)
            self.assertLessEqual(match["similarity_score"], 1.0)

        # Step 8: Dispatch Resolution Velocity Prediction
        velo_res = self.client.post(
            "/api/v1/resolution/predict",
            json={
                "repo_name": "microsoft/vscode",
                "title": issue_title,
                "body": issue_body,
                "canonical_label": class_data["predicted_label"],
                "author_association": "NONE",
                "labels_count": 2,
                "issue_id": issue_id,
            },
        )
        self.assertEqual(velo_res.status_code, 200)
        velo_data = velo_res.json()
        self.assertIn(velo_data["predicted_speed_tier"], self.valid_speed_tiers)
        self.assertAlmostEqual(sum(velo_data["tier_probabilities"].values()), 1.0, places=2)

        # Step 9: Verify audit persistence in predictions table
        final_pred_count = self.db.scalar(select(func.count()).select_from(Prediction))
        # At least classification and resolution prediction were logged
        self.assertGreaterEqual(final_pred_count, initial_pred_count + 2)

    # ──────────────────────────────────────────────────────────────────────────
    # Journey 2: Multi-Model Classification Invariance & Calibration
    # ──────────────────────────────────────────────────────────────────────────

    def test_e2e_multi_model_classification_comparison(self):
        """Cross-model verification: Logistic Regression, SVM, and DistilBERT on identical input."""
        test_payload = {
            "title": "Syntax highlight crash in WebAssembly parser module",
            "body": "Opening a .wasm text file throws an unhandled exception in tokenizer.cpp: line 42",
            "repo_name": "microsoft/vscode",
        }

        models = ["logistic_regression", "svm", "distilbert"]
        responses: Dict[str, Any] = {}

        for model_name in models:
            res = self.client.post(
                "/api/v1/classify",
                json={**test_payload, "model_type": model_name},
            )
            self.assertEqual(res.status_code, 200, f"Model {model_name} failed inference")
            data = res.json()
            responses[model_name] = data

            # Validate common schema invariants
            self.assertIn(data["predicted_label"], self.canonical_classes)
            self.assertGreater(data["confidence_score"], 0.0)
            self.assertLessEqual(data["confidence_score"], 1.0)
            self.assertGreater(data["inference_time_ms"], 0.0)

            # Probabilities dictionary completeness and normalization
            probs = data["probabilities"]
            for cls_key, prob_val in probs.items():
                self.assertIn(cls_key, self.canonical_classes)
                self.assertGreaterEqual(prob_val, 0.0)
                self.assertLessEqual(prob_val, 1.0)
            self.assertAlmostEqual(sum(probs.values()), 1.0, places=2)
            self.assertAlmostEqual(probs[data["predicted_label"]], data["confidence_score"], places=4)

        # All models successfully output a valid canonical category
        for model_name, data in responses.items():
            self.assertIn(data["predicted_label"], self.canonical_classes)

    # ──────────────────────────────────────────────────────────────────────────
    # Journey 3: Dual-Mode Semantic Retrieval (Cache vs Dense Embedding)
    # ──────────────────────────────────────────────────────────────────────────

    def test_e2e_semantic_similarity_cached_vs_dense(self):
        """Verify both relational cache edge retrieval and novel query dense embedding search."""
        # Mode A: Cache lookup on known issue
        cached_edge = self.db.scalar(
            select(RelatedIssuesCache)
            .where(RelatedIssuesCache.rank == 1)
            .limit(1)
        )
        self.assertIsNotNone(cached_edge, "RelatedIssuesCache has no seed edges")
        source_id = cached_edge.source_issue_id

        t0 = time.perf_counter()
        cache_res = self.client.get(f"/api/v1/related/{source_id}?top_k=5")
        t_cache = (time.perf_counter() - t0) * 1000.0

        self.assertEqual(cache_res.status_code, 200)
        cache_data = cache_res.json()
        self.assertEqual(cache_data["query_issue_id"], source_id)
        self.assertGreaterEqual(len(cache_data["matches"]), 1)
        # Cache lookup should be exceptionally fast
        self.assertLess(t_cache, 250.0)

        # Mode B: Dynamic embedding vector search for novel query
        t0 = time.perf_counter()
        dense_res = self.client.post(
            "/api/v1/related",
            json={
                "title": "CUDA out of memory error when executing multi-GPU DDP training",
                "body": "torch.distributed.all_reduce hangs and throws OOM on rank 1 with V100 GPU",
                "repo_name": "pytorch/pytorch",
                "top_k": 5,
            },
        )
        t_dense = (time.perf_counter() - t0) * 1000.0

        self.assertEqual(dense_res.status_code, 200)
        dense_data = dense_res.json()
        self.assertGreaterEqual(len(dense_data["matches"]), 1)
        for match in dense_data["matches"]:
            self.assertEqual(match["repo_name"], "pytorch/pytorch")
            self.assertGreater(match["similarity_score"], 0.20)

        # Verify rank ordering
        scores = [m["similarity_score"] for m in dense_data["matches"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    # ──────────────────────────────────────────────────────────────────────────
    # Journey 4: Resolution Analytics & Velocity Predictor Consistency
    # ──────────────────────────────────────────────────────────────────────────

    def test_e2e_resolution_prediction_and_analytics_consistency(self):
        """Verify SLA distributions and Random Forest velocity predictions across all 4 repos."""
        for owner, name in self.target_repos:
            repo_slug = f"{owner}/{name}"
            stats_res = self.client.get(f"/api/v1/resolution/stats/{owner}/{name}")
            self.assertEqual(stats_res.status_code, 200)
            stats = stats_res.json()

            # Monotonic percentile invariants: P25 <= Median <= P75
            p25 = stats["p25_resolution_hours"]
            med = stats["median_resolution_hours"]
            p75 = stats["p75_resolution_hours"]
            self.assertLessEqual(p25, med, f"P25 > Median for {repo_slug}")
            self.assertLessEqual(med, p75, f"Median > P75 for {repo_slug}")

            # Predict velocity for a typical issue in this repo
            pred_res = self.client.post(
                "/api/v1/resolution/predict",
                json={
                    "repo_name": repo_slug,
                    "title": f"Documentation typo in {name} getting started guide",
                    "body": "Fix link pointing to outdated tutorial page.",
                    "canonical_label": "documentation",
                    "author_association": "CONTRIBUTOR",
                    "labels_count": 1,
                },
            )
            self.assertEqual(pred_res.status_code, 200)
            pred = pred_res.json()
            self.assertEqual(pred["repo_name"], repo_slug)
            self.assertIn(pred["predicted_speed_tier"], self.valid_speed_tiers)
            self.assertIn("reliability_notice", pred)

    # ──────────────────────────────────────────────────────────────────────────
    # Journey 5: Model Registry & Prewarming Lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    def test_e2e_model_registry_and_prewarming_lifecycle(self):
        """Inspect all registered model systems and verify pre-warming endpoint."""
        # 1. Fetch registry
        registry_res = self.client.get("/api/v1/models")
        self.assertEqual(registry_res.status_code, 200)
        registry = registry_res.json()
        self.assertEqual(registry["total_models"], 4)

        model_ids = [m["model_id"] for m in registry["models"]]
        expected_models = [
            "distilbert",
            "classical_baselines",
            "sentence_transformer",
            "resolution_velocity",
        ]
        for em in expected_models:
            self.assertIn(em, model_ids)

        # 2. Inspect individual model metadata
        for em in expected_models:
            m_res = self.client.get(f"/api/v1/models/{em}")
            self.assertEqual(m_res.status_code, 200)
            m_data = m_res.json()
            self.assertEqual(m_data["model_id"], em)
            self.assertIn("task_type", m_data)

        # 3. Prewarm models
        warm_res = self.client.post("/api/v1/models/prewarm")
        self.assertEqual(warm_res.status_code, 200)
        warm_data = warm_res.json()
        self.assertIn("distilbert", warm_data)
        self.assertIn("sentence_transformer", warm_data)
        self.assertEqual(warm_data["distilbert"], "prewarmed")

    # ──────────────────────────────────────────────────────────────────────────
    # Journey 6: Database Audit Trail Integrity
    # ──────────────────────────────────────────────────────────────────────────

    def test_e2e_audit_trail_database_persistence(self):
        """Verify that classification and velocity prediction invocations persist in SQLite."""
        unique_title = f"E2E Audit Test Unique Issue {time.time()}"
        unique_body = "Detailed system diagnostic body for database verification."

        # Make classification request
        class_res = self.client.post(
            "/api/v1/classify",
            json={
                "title": unique_title,
                "body": unique_body,
                "repo_name": "kubernetes/kubernetes",
                "model_type": "distilbert",
            },
        )
        self.assertEqual(class_res.status_code, 200)

        # Make resolution prediction request
        reso_res = self.client.post(
            "/api/v1/resolution/predict",
            json={
                "repo_name": "kubernetes/kubernetes",
                "title": unique_title,
                "body": unique_body,
                "canonical_label": "bug",
            },
        )
        self.assertEqual(reso_res.status_code, 200)

        # Query database to assert audit record
        predictions = self.db.scalars(
            select(Prediction)
            .where(Prediction.input_title == unique_title)
            .order_by(Prediction.id.asc())
        ).all()

        self.assertEqual(len(predictions), 2, "Expected 2 prediction audit rows (classification + resolution)")

        class_pred = predictions[0]
        self.assertEqual(class_pred.task_type, "classification")
        self.assertEqual(class_pred.model_name, "distilbert-finetuned-v1")
        self.assertIsNotNone(class_pred.probabilities)
        self.assertIsInstance(class_pred.probabilities, dict)
        self.assertGreater(class_pred.inference_time_ms, 0.0)

        reso_pred = predictions[1]
        self.assertEqual(reso_pred.task_type, "resolution_prediction")
        self.assertEqual(reso_pred.model_name, "resolution-rf-classifier-champion")
        self.assertIsNotNone(reso_pred.probabilities)
        self.assertIsInstance(reso_pred.probabilities, dict)

    # ──────────────────────────────────────────────────────────────────────────
    # Journey 7: Concurrent Multi-Service Request Thread Safety
    # ──────────────────────────────────────────────────────────────────────────

    def test_e2e_concurrent_multi_service_load(self):
        """Verify thread-safe multi-service concurrency under parallel client requests."""
        def execute_task(task_type: str) -> int:
            if task_type == "repos":
                r = self.client.get("/api/v1/repos")
                return r.status_code
            elif task_type == "classify_baseline":
                r = self.client.post(
                    "/api/v1/classify",
                    json={"title": "Concurrent baseline classification check", "model_type": "logistic_regression"},
                )
                return r.status_code
            elif task_type == "classify_transformer":
                r = self.client.post(
                    "/api/v1/classify",
                    json={"title": "Concurrent transformer classification check", "model_type": "distilbert"},
                )
                return r.status_code
            elif task_type == "similarity":
                r = self.client.post(
                    "/api/v1/related",
                    json={"title": "Concurrent semantic similarity vector lookup", "top_k": 3},
                )
                return r.status_code
            elif task_type == "resolution":
                r = self.client.post(
                    "/api/v1/resolution/predict",
                    json={"repo_name": "pytorch/pytorch", "title": "Concurrent velocity prediction test"},
                )
                return r.status_code
            elif task_type == "overview":
                r = self.client.get("/api/v1/analytics/overview")
                return r.status_code
            return 400

        task_types = [
            "repos",
            "classify_baseline",
            "classify_transformer",
            "similarity",
            "resolution",
            "overview",
            "classify_transformer",
            "resolution",
            "similarity",
            "repos",
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_results = list(executor.map(execute_task, task_types))

        for idx, status_code in enumerate(future_results):
            self.assertEqual(
                status_code,
                200,
                f"Concurrent request {idx} ({task_types[idx]}) returned status {status_code}",
            )


if __name__ == "__main__":
    unittest.main()
