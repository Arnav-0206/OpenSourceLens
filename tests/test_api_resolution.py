"""Integration tests for resolution analysis and advisory prediction API endpoints."""

import unittest

from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.app.database.models import Prediction
from backend.app.database.session import SessionLocal
from backend.app.main import app


class TestAPIResolution(unittest.TestCase):
    """Test suite for /api/v1/resolution endpoints."""

    @classmethod
    def setUpClass(cls):
        """Initialize TestClient."""
        cls.client = TestClient(app)

    def test_predict_resolution_vscode_success(self):
        """Verify POST /api/v1/resolution/predict yields valid velocity tier and error bounds."""
        payload = {
            "repo_name": "microsoft/vscode",
            "title": "[UI] Crash when opening debug console on Windows",
            "body": "Steps to reproduce:\n```python\nimport sys\nprint('hello')\n```",
            "canonical_label": "bug",
            "author_association": "MEMBER",
            "labels_count": 2,
        }
        response = self.client.post("/api/v1/resolution/predict", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["repo_name"], "microsoft/vscode")
        self.assertIn(data["predicted_speed_tier"], ["< 1 Day", "1-7 Days", "7-30 Days", "30+ Days"])
        self.assertGreater(data["confidence_score"], 0.0)
        self.assertLessEqual(data["confidence_score"], 1.0)

        # Probabilities
        probs = data["tier_probabilities"]
        self.assertEqual(len(probs), 4)
        self.assertIn("< 1 Day", probs)
        self.assertIn("1-7 Days", probs)
        self.assertIn("7-30 Days", probs)
        self.assertIn("30+ Days", probs)
        self.assertAlmostEqual(sum(probs.values()), 1.0, delta=0.02)

        # Adjacent error bounds
        self.assertIn(data["lower_bound_tier"], ["< 1 Day", "1-7 Days", "7-30 Days", "30+ Days"])
        self.assertIn(data["upper_bound_tier"], ["< 1 Day", "1-7 Days", "7-30 Days", "30+ Days"])

        # Mandatory advisory disclaimer
        self.assertIn("advisory", data["reliability_notice"].lower())
        self.assertIn("sla", data["reliability_notice"].lower())

        # Verify audit persistence
        session = SessionLocal()
        try:
            audit = session.scalar(
                select(Prediction)
                .where(Prediction.input_title == payload["title"])
                .order_by(Prediction.created_at.desc())
            )
            self.assertIsNotNone(audit)
            self.assertEqual(audit.task_type, "resolution_prediction")
            self.assertEqual(audit.predicted_label, data["predicted_speed_tier"])
        finally:
            session.close()

    def test_predict_resolution_pytorch_success(self):
        """Verify POST /api/v1/resolution/predict works for PyTorch repository."""
        payload = {
            "repo_name": "pytorch/pytorch",
            "title": "CUDA out of memory error during distributed training backward pass",
            "body": "Reproduction script:\n```python\nimport torch\n```",
            "canonical_label": "bug",
            "author_association": "NONE",
            "labels_count": 1,
        }
        response = self.client.post("/api/v1/resolution/predict", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["repo_name"], "pytorch/pytorch")
        self.assertIn(data["predicted_speed_tier"], ["< 1 Day", "1-7 Days", "7-30 Days", "30+ Days"])

    def test_predict_resolution_validation_error(self):
        """Verify validation error when title is too short."""
        response = self.client.post(
            "/api/v1/resolution/predict",
            json={"repo_name": "microsoft/vscode", "title": "ab"},
        )
        self.assertEqual(response.status_code, 422)

    def test_get_resolution_stats_success(self):
        """Verify GET /api/v1/resolution/stats/{owner}/{name} returns descriptive percentiles."""
        response = self.client.get("/api/v1/resolution/stats/microsoft/vscode")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["repo_name"], "microsoft/vscode")
        self.assertEqual(data["n_closed_issues"], 242)
        self.assertIsNotNone(data["median_resolution_hours"])
        self.assertAlmostEqual(data["median_resolution_hours"], 7.71, delta=0.5)
        self.assertIsNotNone(data["p25_resolution_hours"])
        self.assertIsNotNone(data["p75_resolution_hours"])
        self.assertGreater(data["pct_resolved_under_24h"], 0.0)
        self.assertGreater(data["survival_rate_30d"], 0.0)

    def test_get_resolution_stats_not_found(self):
        """Verify 404 response for nonexistent repository."""
        response = self.client.get("/api/v1/resolution/stats/nonexistent_org/nonexistent_repo")
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found", response.json()["detail"].lower())


if __name__ == "__main__":
    unittest.main()
