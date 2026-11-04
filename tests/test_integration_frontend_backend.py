"""Integration Tests for Frontend & Backend Connectivity, CORS Protocol, and API Contracts.

Step 11.1 Verification Suite:
- Validates CORS preflight (OPTIONS) negotiation for Next.js frontend origins.
- Validates Access-Control-Allow-* headers on typical REST methods (GET, POST).
- Validates origin filtering (allowed localhost:3000 vs disallowed origins).
- Verifies that FastAPI JSON payloads strictly conform to frontend TypeScript definitions:
    * HealthResponse (/health)
    * RootResponse (/)
    * RepositorySummary[] (/api/v1/repos)
    * GlobalAnalyticsOverview (/api/v1/analytics/overview)
    * ClassificationResponse (/api/v1/classify)
    * RelatedIssuesResponse (/api/v1/related)
    * ResolutionPredictionResponse (/api/v1/resolution/predict)
    * ModelRegistryResponse (/api/v1/models)
"""

from __future__ import annotations

import unittest
from fastapi.testclient import TestClient

from backend.app.core.config import settings
from backend.app.main import app


class TestFrontendBackendIntegration(unittest.TestCase):
    """Test suite ensuring Next.js frontend can seamlessly communicate with FastAPI backend."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.allowed_origin = "http://localhost:3000"
        cls.disallowed_origin = "http://unauthorized-domain.com"

    # ──────────────────────────────────────────────────────────────────────────
    # 1. CORS Negotiations & Header Verification
    # ──────────────────────────────────────────────────────────────────────────

    def test_cors_preflight_options_request(self):
        """Verify preflight OPTIONS request from Next.js origin succeeds with expected headers."""
        headers = {
            "Origin": self.allowed_origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type,Accept",
        }
        response = self.client.options("/api/v1/classify", headers=headers)
        self.assertEqual(response.status_code, 200)

        self.assertEqual(
            response.headers.get("access-control-allow-origin"),
            self.allowed_origin,
        )
        self.assertIn("POST", response.headers.get("access-control-allow-methods", ""))
        self.assertEqual(
            response.headers.get("access-control-allow-credentials"),
            "true",
        )

    def test_cors_get_headers_allowed_origin(self):
        """Verify GET /health includes proper CORS headers when requested by Next.js origin."""
        response = self.client.get("/health", headers={"Origin": self.allowed_origin})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers.get("access-control-allow-origin"),
            self.allowed_origin,
        )
        self.assertEqual(
            response.headers.get("access-control-allow-credentials"),
            "true",
        )

    def test_cors_rejection_for_disallowed_origin(self):
        """Verify untrusted origins do not receive Access-Control-Allow-Origin header."""
        response = self.client.get("/health", headers={"Origin": self.disallowed_origin})
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("access-control-allow-origin", response.headers)

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Schema Contract Validations Matching Frontend TypeScript Types
    # ──────────────────────────────────────────────────────────────────────────

    def test_health_response_contract(self):
        """Verify /health returns payload matching frontend HealthResponse interface."""
        response = self.client.get("/health", headers={"Origin": self.allowed_origin})
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("status", data)
        self.assertIn("version", data)
        self.assertIn("timestamp", data)
        self.assertIn("database", data)

        db = data["database"]
        self.assertIn("status", db)
        self.assertIn("dialect", db)
        self.assertIn("tables_count", db)
        self.assertIn("latency_ms", db)
        self.assertIn("error", db)
        self.assertIsInstance(db["latency_ms"], (int, float))

    def test_root_response_contract(self):
        """Verify / returns payload matching frontend RootResponse interface."""
        response = self.client.get("/", headers={"Origin": self.allowed_origin})
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("title", data)
        self.assertIn("version", data)
        self.assertIn("description", data)
        self.assertIn("docs_url", data)
        self.assertIn("api_v1_url", data)
        self.assertIn("supported_repositories", data)
        self.assertIsInstance(data["supported_repositories"], list)

    def test_repositories_summary_contract(self):
        """Verify /api/v1/repos returns list matching frontend RepositorySummary interface."""
        response = self.client.get("/api/v1/repos", headers={"Origin": self.allowed_origin})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)

        if len(data) > 0:
            repo = data[0]
            for key in [
                "id",
                "repo_name",
                "owner",
                "name",
                "description",
                "primary_language",
                "stars_count",
                "forks_count",
                "open_issues_count",
                "collected_at",
            ]:
                self.assertIn(key, repo)

    def test_analytics_overview_contract(self):
        """Verify /api/v1/analytics/overview returns data matching GlobalAnalyticsOverview."""
        response = self.client.get("/api/v1/analytics/overview", headers={"Origin": self.allowed_origin})
        self.assertEqual(response.status_code, 200)
        data = response.json()

        expected_fields = [
            "total_repositories",
            "total_issues",
            "total_open_issues",
            "total_closed_issues",
            "global_closure_rate",
            "category_distribution",
            "speed_tier_distribution",
        ]
        for f in expected_fields:
            self.assertIn(f, data)
        self.assertIn("fastest_repository", data)
        self.assertIn("slowest_repository", data)

    def test_classification_contract(self):
        """Verify /api/v1/classify returns data matching frontend ClassificationResponse."""
        payload = {
            "title": "Fix memory leak in buffer allocation",
            "body": "When running under heavy concurrency, buffer pool leaks descriptors.",
            "model_type": "distilbert",
        }
        response = self.client.post("/api/v1/classify", json=payload, headers={"Origin": self.allowed_origin})
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("predicted_label", data)
        self.assertIn("canonical_label_id", data)
        self.assertIn("confidence_score", data)
        self.assertIn("model_used", data)
        self.assertIn("probabilities", data)
        self.assertIn("inference_time_ms", data)
        self.assertIn("clean_title", data)
        self.assertIn("is_classified_category", data)
        self.assertIsInstance(data["probabilities"], dict)

    def test_related_issues_contract(self):
        """Verify /api/v1/related returns data matching frontend RelatedIssuesResponse."""
        payload = {
            "title": "Segmentation fault in tensor multiplication",
            "body": "Running matrix multiplication on CUDA device causes crash",
            "repo_name": "pytorch/pytorch",
            "top_k": 3,
        }
        response = self.client.post("/api/v1/related", json=payload, headers={"Origin": self.allowed_origin})
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("model_version", data)
        self.assertIn("total_matches", data)
        self.assertIn("matches", data)
        self.assertIn("retrieval_time_ms", data)
        self.assertIsInstance(data["matches"], list)

        if len(data["matches"]) > 0:
            match = data["matches"][0]
            for field in [
                "issue_id",
                "repo_name",
                "issue_number",
                "title",
                "state",
                "canonical_label",
                "similarity_score",
                "rank",
                "created_at",
            ]:
                self.assertIn(field, match)

    def test_resolution_prediction_contract(self):
        """Verify /api/v1/resolution/predict returns data matching ResolutionPredictionResponse."""
        payload = {
            "repo_name": "microsoft/vscode",
            "title": "Documentation typo in quickstart tutorial",
            "body": "Page 2 line 15 has an outdated URL link to repository.",
        }
        response = self.client.post("/api/v1/resolution/predict", json=payload, headers={"Origin": self.allowed_origin})
        self.assertEqual(response.status_code, 200)
        data = response.json()

        expected_fields = [
            "repo_name",
            "predicted_speed_tier",
            "confidence_score",
            "tier_probabilities",
            "lower_bound_tier",
            "upper_bound_tier",
            "advisory_factors",
            "reliability_notice",
            "model_used",
            "inference_time_ms",
        ]
        for f in expected_fields:
            self.assertIn(f, data)

    def test_model_registry_contract(self):
        """Verify /api/v1/models returns data matching frontend ModelRegistryResponse."""
        response = self.client.get("/api/v1/models", headers={"Origin": self.allowed_origin})
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("models", data)
        self.assertIn("total_models", data)
        self.assertIsInstance(data["models"], list)

        if len(data["models"]) > 0:
            model = data["models"][0]
            for field in ["model_id", "name", "task_type", "version", "artifact_path", "is_loaded", "device", "status"]:
                self.assertIn(field, model)


if __name__ == "__main__":
    unittest.main()
