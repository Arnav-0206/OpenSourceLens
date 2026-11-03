"""Integration tests for model registry and health introspection endpoints."""

import unittest

from fastapi.testclient import TestClient

from backend.app.main import app


class TestAPIModels(unittest.TestCase):
    """Test suite for /api/v1/models endpoints."""

    @classmethod
    def setUpClass(cls):
        """Initialize TestClient."""
        cls.client = TestClient(app)

    def test_list_registered_models(self):
        """Verify GET /api/v1/models lists all 4 project model subsystems."""
        response = self.client.get("/api/v1/models")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["total_models"], 4)
        self.assertIn(data["system_device"], ["cpu", "cuda"])

        model_ids = [m["model_id"] for m in data["models"]]
        self.assertIn("distilbert", model_ids)
        self.assertIn("classical_baselines", model_ids)
        self.assertIn("sentence_transformer", model_ids)
        self.assertIn("resolution_velocity", model_ids)

        for m in data["models"]:
            self.assertEqual(m["status"], "ready")
            self.assertIsNotNone(m["artifact_path"])

    def test_get_distilbert_details(self):
        """Verify GET /api/v1/models/distilbert returns transformer architecture info."""
        response = self.client.get("/api/v1/models/distilbert")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["model_id"], "distilbert")
        self.assertEqual(data["task_type"], "classification")
        self.assertIn("66955011", str(data["details"]["num_parameters"]))
        self.assertEqual(data["details"]["classes"], ["bug", "feature", "other"])

    def test_get_resolution_velocity_details(self):
        """Verify GET /api/v1/models/resolution_velocity returns 4-tier model metadata."""
        response = self.client.get("/api/v1/models/resolution_velocity")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["model_id"], "resolution_velocity")
        self.assertEqual(data["task_type"], "resolution_prediction")
        self.assertIn("RandomForestClassifier", data["details"]["classifier"])
        self.assertEqual(data["details"]["classes"], ["< 1 Day", "1-7 Days", "7-30 Days", "30+ Days"])

    def test_get_sentence_transformer_details(self):
        """Verify GET /api/v1/models/sentence_transformer returns embedding dimension."""
        response = self.client.get("/api/v1/models/sentence_transformer")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["model_id"], "sentence_transformer")
        self.assertEqual(data["task_type"], "semantic_retrieval")
        self.assertEqual(data["details"]["embedding_dimension"], 384)

    def test_get_model_not_found(self):
        """Verify 404 response for unknown model slug."""
        response = self.client.get("/api/v1/models/nonexistent_model_id")
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found", response.json()["detail"].lower())

    def test_post_prewarm_models(self):
        """Verify POST /api/v1/models/prewarm loads models without error."""
        response = self.client.post("/api/v1/models/prewarm")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("distilbert", data)
        self.assertIn("sentence_transformer", data)
        self.assertIn("resolution_velocity", data)


if __name__ == "__main__":
    unittest.main()
