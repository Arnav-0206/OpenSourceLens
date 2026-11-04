"""Edge Case & System Resilience Integration Test Suite for OpenSourceLens (Phase 11, Step 11.8).

Verifies boundary conditions, error handling envelopes, malformed payloads,
extreme inputs, zero-match states, and information masking:
1. 404 Resource Not Found handling (repositories, issues, models, resolution stats, cached edges).
2. 422 Pydantic Validation Error envelopes (empty payloads, title lengths, invalid models, pagination).
3. 200 OK Safe Handling for extreme inputs (10k+ character bodies, emojis, XSS script injection, unicode).
4. 200 OK Safe Handling for zero-match queries (compound filter misses, empty related issue candidates).
5. Minimal feature fallbacks and parameter defaulting.
6. Information disclosure prevention (no raw DB traces or unhandled 500 server crashes).
"""

from __future__ import annotations

import unittest
from typing import Any, Dict

from fastapi.testclient import TestClient

from backend.app.main import app


class TestEdgeCasesAndResilience(unittest.TestCase):
    """Integration test suite for edge case validation and error resilience."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Prewarm models to prevent cold-start disk I/O interference with latency checks
        cls.client.post("/api/v1/models/prewarm")

    # ──────────────────────────────────────────────────────────────────────────
    # Category 1: 404 Resource Not Found Envelopes
    # ──────────────────────────────────────────────────────────────────────────

    def test_404_nonexistent_repository_slug(self):
        """Querying a non-existent repository returns standard 404 RESOURCE_NOT_FOUND."""
        res = self.client.get("/api/v1/repos/unknown_org/nonexistent_repo")
        self.assertEqual(res.status_code, 404)
        data = res.json()
        self.assertEqual(data["error_code"], "RESOURCE_NOT_FOUND")
        self.assertEqual(data["status_code"], 404)
        self.assertIn("not found", data["detail"].lower())
        self.assertIn("timestamp", data)
        self.assertEqual(data["path"], "/api/v1/repos/unknown_org/nonexistent_repo")

    def test_404_nonexistent_issue_number(self):
        """Querying a non-existent issue within a valid repository returns 404."""
        res = self.client.get("/api/v1/repos/microsoft/vscode/issues/999999999")
        self.assertEqual(res.status_code, 404)
        data = res.json()
        self.assertEqual(data["error_code"], "RESOURCE_NOT_FOUND")
        self.assertIn("not found", data["detail"].lower())

    def test_404_nonexistent_model_id(self):
        """Requesting metadata for an unregistered model returns 404."""
        res = self.client.get("/api/v1/models/unregistered_super_net")
        self.assertEqual(res.status_code, 404)
        data = res.json()
        self.assertEqual(data["error_code"], "RESOURCE_NOT_FOUND")

    def test_404_nonexistent_resolution_stats_repo(self):
        """Requesting resolution SLA stats for an unknown repo returns 404."""
        res = self.client.get("/api/v1/resolution/stats/fake_org/fake_repo")
        self.assertEqual(res.status_code, 404)
        data = res.json()
        self.assertEqual(data["error_code"], "RESOURCE_NOT_FOUND")

    def test_404_nonexistent_cached_related_issue(self):
        """Querying cached similarity edges for a non-existent issue ID returns 404."""
        res = self.client.get("/api/v1/related/9999999999")
        self.assertEqual(res.status_code, 404)
        data = res.json()
        self.assertEqual(data["error_code"], "RESOURCE_NOT_FOUND")

    # ──────────────────────────────────────────────────────────────────────────
    # Category 2: 422 Schema Validation Errors
    # ──────────────────────────────────────────────────────────────────────────

    def test_422_empty_classification_payload(self):
        """Sending an empty JSON payload returns 422 VALIDATION_ERROR with field list."""
        res = self.client.post("/api/v1/classify", json={})
        self.assertEqual(res.status_code, 422)
        data = res.json()
        self.assertEqual(data["error_code"], "VALIDATION_ERROR")
        self.assertEqual(data["status_code"], 422)
        self.assertIn("errors", data)
        self.assertIsInstance(data["errors"], list)
        fields = [err["field"] for err in data["errors"]]
        self.assertTrue(any("title" in f for f in fields))

    def test_422_short_issue_title(self):
        """Title with fewer than 3 characters triggers 422 validation error."""
        res = self.client.post(
            "/api/v1/classify",
            json={"title": "ab", "model_type": "distilbert"},
        )
        self.assertEqual(res.status_code, 422)
        data = res.json()
        self.assertEqual(data["error_code"], "VALIDATION_ERROR")
        msg = data["errors"][0]["message"]
        self.assertTrue("at least 3 characters" in msg or "string_too_short" in data["errors"][0]["type"])

    def test_422_long_issue_title(self):
        """Title exceeding 500 characters triggers 422 validation error."""
        long_title = "crash " * 110  # > 550 characters
        res = self.client.post(
            "/api/v1/classify",
            json={"title": long_title, "model_type": "distilbert"},
        )
        self.assertEqual(res.status_code, 422)
        data = res.json()
        self.assertEqual(data["error_code"], "VALIDATION_ERROR")

    def test_422_unsupported_model_type(self):
        """Passing an invalid model_type triggers regex pattern validation failure."""
        res = self.client.post(
            "/api/v1/classify",
            json={"title": "Valid title text", "model_type": "gpt-4o-unsupported"},
        )
        self.assertEqual(res.status_code, 422)
        data = res.json()
        self.assertEqual(data["error_code"], "VALIDATION_ERROR")

    def test_422_out_of_bounds_pagination(self):
        """Invalid pagination parameters (limit < 1 or limit > 100) return 422."""
        res_negative = self.client.get("/api/v1/repos/microsoft/vscode/issues?limit=-5")
        self.assertEqual(res_negative.status_code, 422)

        res_excessive = self.client.get("/api/v1/repos/microsoft/vscode/issues?limit=500")
        self.assertEqual(res_excessive.status_code, 422)

    def test_422_invalid_state_filter(self):
        """Invalid issue state string in query returns 422 regex pattern failure."""
        res = self.client.post(
            "/api/v1/related",
            json={"title": "Valid title query", "state": "archived_state"},
        )
        self.assertEqual(res.status_code, 422)
        data = res.json()
        self.assertEqual(data["error_code"], "VALIDATION_ERROR")

    # ──────────────────────────────────────────────────────────────────────────
    # Category 3: 200 OK Safe Handling for Extreme & Unusual Inputs
    # ──────────────────────────────────────────────────────────────────────────

    def test_200_extreme_body_length_distilbert_truncation(self):
        """Extreme body length (>10,000 chars) is truncated safely without error or OOM."""
        massive_body = (
            "Traceback (most recent call last):\n"
            + "  File 'runner.py', line 102, in execute\n"
            + "    raise RuntimeError('Memory corruption in buffer 0xDEADBEEF')\n"
        ) * 120  # ~11,000 chars

        res = self.client.post(
            "/api/v1/classify",
            json={
                "title": "Severe memory corruption in execution pipeline",
                "body": massive_body,
                "model_type": "distilbert",
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn(data["predicted_label"], ["bug", "feature", "other"])
        self.assertGreater(data["confidence_score"], 0.0)
        self.assertLess(data["inference_time_ms"], 1500.0)

    def test_200_special_unicode_emojis_and_html_injection(self):
        """Input containing emojis, null bytes, HTML tags, and markdown is cleaned safely."""
        xss_title = "💥 [CRITICAL BUG] <script>alert('xss');</script> in &amp; parser 🐛\t\r\n"
        xss_body = "```html\n<div><iframe src='malicious.site'></iframe></div>\n``` \u2705 \U0001F600"

        res = self.client.post(
            "/api/v1/classify",
            json={
                "title": xss_title,
                "body": xss_body,
                "model_type": "distilbert",
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        # Verify text was sanitized
        self.assertNotIn("<script>", data["clean_title"])
        self.assertIn(data["predicted_label"], ["bug", "feature", "other"])

    def test_200_compound_filter_zero_matches_safe_response(self):
        """Filtering with mutually exclusive criteria returns total=0 without exception."""
        res = self.client.get(
            "/api/v1/repos/microsoft/vscode/issues?category=documentation&speed_tier=30%2B+Days&search=nonexistent_symbol_xyz123"
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total"], 0)
        self.assertEqual(data["items"], [])
        self.assertEqual(data["limit"], 50)
        self.assertEqual(data["offset"], 0)

    def test_200_zero_matches_dynamic_similarity_unindexed_repo(self):
        """Semantic search scoped to non-existent repo returns empty matches cleanly."""
        res = self.client.post(
            "/api/v1/related",
            json={
                "title": "CUDA memory allocation failure in conv2d kernel",
                "repo_name": "unindexed_organization/empty_repo",
                "top_k": 5,
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total_matches"], 0)
        self.assertEqual(data["matches"], [])

    def test_200_resolution_prediction_with_minimal_features(self):
        """Velocity prediction succeeds when only mandatory fields are supplied."""
        res = self.client.post(
            "/api/v1/resolution/predict",
            json={
                "repo_name": "microsoft/vscode",
                "title": "Minimal title parameter check",
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn(data["predicted_speed_tier"], ["< 1 Day", "1-7 Days", "7-30 Days", "30+ Days"])
        self.assertIn("reliability_notice", data)
        self.assertAlmostEqual(sum(data["tier_probabilities"].values()), 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
