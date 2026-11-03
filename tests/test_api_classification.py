"""Integration tests for issue classification inference endpoint."""

import unittest

from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.app.database.models import Prediction
from backend.app.database.session import SessionLocal
from backend.app.main import app


class TestAPIClassification(unittest.TestCase):
    """Test suite for POST /api/v1/classify inference endpoint."""

    @classmethod
    def setUpClass(cls):
        """Initialize TestClient."""
        cls.client = TestClient(app)

    def test_classify_distilbert_success(self):
        """Verify fine-tuned DistilBERT inference produces valid prediction and probabilities."""
        payload = {
            "title": "Segmentation fault in tensor operation",
            "body": "Running matmul with large dimensions crashes with SIGSEGV in libtorch.so",
            "repo_name": "pytorch/pytorch",
            "model_type": "distilbert",
        }
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn(data["predicted_label"], ["bug", "feature", "other"])
        self.assertGreater(data["confidence_score"], 0.0)
        self.assertLessEqual(data["confidence_score"], 1.0)
        self.assertIn("distilbert", data["model_used"].lower())
        self.assertGreater(data["inference_time_ms"], 0.0)

        # Probabilities validation
        probs = data["probabilities"]
        self.assertIn("bug", probs)
        self.assertIn("feature", probs)
        self.assertIn("other", probs)
        self.assertAlmostEqual(sum(probs.values()), 1.0, delta=0.02)

        # Verify audit persistence
        session = SessionLocal()
        try:
            audit = session.scalar(
                select(Prediction)
                .where(Prediction.input_title == payload["title"])
                .order_by(Prediction.created_at.desc())
            )
            self.assertIsNotNone(audit)
            self.assertEqual(audit.predicted_label, data["predicted_label"])
            self.assertEqual(audit.task_type, "classification")
            self.assertEqual(audit.repo_name, "pytorch/pytorch")
        finally:
            session.close()

    def test_classify_logistic_regression_baseline(self):
        """Verify TF-IDF Logistic Regression baseline produces valid predictions."""
        payload = {
            "title": "Add support for dark high-contrast mode",
            "body": "Accessibility feature request for dark high-contrast themes",
            "model_type": "logistic_regression",
        }
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn(data["predicted_label"], ["bug", "feature", "other"])
        self.assertIn("logistic-regression", data["model_used"])
        probs = data["probabilities"]
        self.assertAlmostEqual(sum(probs.values()), 1.0, delta=0.02)

    def test_classify_svm_baseline(self):
        """Verify TF-IDF SVM baseline produces calibrated probabilities."""
        payload = {
            "title": "Fix memory leak during model checkpoint save",
            "body": "CUDA memory is not freed after torch.save()",
            "model_type": "svm",
        }
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn(data["predicted_label"], ["bug", "feature", "other"])
        self.assertIn("svm", data["model_used"])
        probs = data["probabilities"]
        self.assertAlmostEqual(sum(probs.values()), 1.0, delta=0.02)

    def test_validation_errors(self):
        """Verify validation errors for title length and invalid model types."""
        # Title too short (< 3 chars)
        res_short = self.client.post("/api/v1/classify", json={"title": "ab"})
        self.assertEqual(res_short.status_code, 422)

        # Invalid model type
        res_invalid_model = self.client.post(
            "/api/v1/classify",
            json={"title": "Valid title text", "model_type": "unsupported_model_xyz"},
        )
        self.assertEqual(res_invalid_model.status_code, 422)


if __name__ == "__main__":
    unittest.main()
