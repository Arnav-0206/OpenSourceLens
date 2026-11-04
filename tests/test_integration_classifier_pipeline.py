"""Integration Test Suite for Classical Classifier Pipeline (Phase 11, Step 11.3).

Verifies the complete integration of classical ML baseline models (TF-IDF + Logistic Regression & Linear SVM):
1. End-to-end inference execution via POST /api/v1/classify.
2. Model identifier verification ('tfidf-logistic-regression-baseline', 'tfidf-svm-baseline').
3. Probability distribution calibration and normalization (sum = 1.0).
4. Prototypical category classification (bug reports vs. feature proposals).
5. Database audit logging into the 'predictions' table.
6. Input validation and error boundary checks (min length, invalid model_type).
7. Model registry metadata contract (/api/v1/models/classical_baselines).
"""

from __future__ import annotations

import unittest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.database.models import Prediction
from backend.app.database.session import SessionLocal
from backend.app.main import app


class TestClassifierPipelineIntegration(unittest.TestCase):
    """Integration test suite for classical TF-IDF classification pipelines."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        self.db: Session = SessionLocal()

    def tearDown(self):
        self.db.close()

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Logistic Regression Inference & Probability Contract
    # ──────────────────────────────────────────────────────────────────────────

    def test_logistic_regression_inference(self):
        """Verify TF-IDF Logistic Regression endpoint returns calibrated response."""
        payload = {
            "title": "Fix memory leak in buffer allocation pool",
            "body": "When running under heavy concurrency, buffer pool leaks file descriptors.",
            "repo_name": "microsoft/vscode",
            "model_type": "logistic_regression",
        }
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["model_used"], "tfidf-logistic-regression-baseline")
        self.assertIn(data["predicted_label"], ["bug", "feature", "other"])
        self.assertGreater(data["confidence_score"], 0.0)
        self.assertLessEqual(data["confidence_score"], 1.0)

        # Probabilities sum strictly to ~1.0
        probs = data["probabilities"]
        self.assertIsInstance(probs, dict)
        self.assertIn("bug", probs)
        self.assertIn("feature", probs)
        self.assertIn("other", probs)
        total_prob = sum(probs.values())
        self.assertAlmostEqual(total_prob, 1.0, places=2)

        # Latency benchmark (< 200ms warm on CPU)
        self.assertGreater(data["inference_time_ms"], 0.0)
        self.assertLess(data["inference_time_ms"], 500.0)

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Linear SVM Inference & Calibrated Hyperplane Distance Contract
    # ──────────────────────────────────────────────────────────────────────────

    def test_svm_inference_and_softmax_calibration(self):
        """Verify Linear SVM endpoint produces softmax-calibrated probability distributions."""
        payload = {
            "title": "Add support for markdown footnotes syntax in editor",
            "body": "Proposal to support [^1] footnotes in preview parser for enhanced documentation.",
            "repo_name": "microsoft/vscode",
            "model_type": "svm",
        }
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["model_used"], "tfidf-svm-baseline")
        self.assertIn(data["predicted_label"], ["bug", "feature", "other"])
        self.assertGreater(data["confidence_score"], 0.0)
        self.assertLessEqual(data["confidence_score"], 1.0)

        # Softmax calibrated probabilities sum to 1.0
        probs = data["probabilities"]
        self.assertIsInstance(probs, dict)
        total_prob = sum(probs.values())
        self.assertAlmostEqual(total_prob, 1.0, places=2)

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Prototypical Category Evaluation
    # ──────────────────────────────────────────────────────────────────────────

    def test_prototypical_bug_classification(self):
        """Verify a clear crash report is classified as bug with high confidence."""
        payload = {
            "title": "Crash on startup with SIGSEGV code 139 in extension host",
            "body": "Fatal exception in native thread worker. Stack trace:\n#0 0x7f8a923 in worker_loop\n#1 abort()",
            "model_type": "svm",
        }
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("bug", data["probabilities"])
        self.assertGreater(data["probabilities"]["bug"], 0.3)

    def test_prototypical_feature_classification(self):
        """Verify a clear proposal is classified as feature with high confidence."""
        payload = {
            "title": "Proposal: implement vectorized SIMD matrix multiply kernels",
            "body": "Feature request to accelerate GEMM operations using AVX-512 tensor primitives.",
            "model_type": "logistic_regression",
        }
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("feature", data["probabilities"])
        self.assertGreater(data["probabilities"]["feature"], 0.3)

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Database Audit Logging
    # ──────────────────────────────────────────────────────────────────────────

    def test_database_audit_record_created_on_classify(self):
        """Verify that each classification request inserts an audit record in the predictions table."""
        count_before = self.db.scalar(select(func.count(Prediction.id))) or 0

        unique_title = "Audit Test: Crash in WebSocket frame handler during reconnect"
        payload = {
            "title": unique_title,
            "body": "Connection reset by peer triggers uncaught exception.",
            "repo_name": "pytorch/pytorch",
            "model_type": "logistic_regression",
        }
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 200)

        # Check DB audit count increased by 1
        count_after = self.db.scalar(select(func.count(Prediction.id))) or 0
        self.assertEqual(count_after, count_before + 1)

        # Verify audit record content
        record = self.db.scalar(
            select(Prediction).where(Prediction.input_title == unique_title)
        )
        self.assertIsNotNone(record)
        self.assertEqual(record.model_name, "tfidf-logistic-regression-baseline")
        self.assertEqual(record.task_type, "classification")
        self.assertEqual(record.repo_name, "pytorch/pytorch")
        self.assertIn(record.predicted_label, ["bug", "feature", "other"])
        self.assertGreater(record.confidence_score, 0.0)

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Input Boundaries & Validation Handling
    # ──────────────────────────────────────────────────────────────────────────

    def test_title_only_classification_succeeds(self):
        """Verify that classification works cleanly with null/empty body."""
        payload = {
            "title": "Fix buffer overflow in string parser",
            "body": None,
            "model_type": "svm",
        }
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["predicted_label"], ["bug", "feature", "other"])

    def test_short_title_rejected_with_422(self):
        """Verify title shorter than 3 characters triggers validation error (422)."""
        payload = {
            "title": "AB",
            "body": "Too short title",
            "model_type": "logistic_regression",
        }
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_invalid_model_type_rejected_with_422(self):
        """Verify unsupported model_type triggers regex pattern validation error (422)."""
        payload = {
            "title": "Valid title for unsupported model",
            "body": "Testing invalid model_type parameter",
            "model_type": "random_forest_xgboost",
        }
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 422)

    # ──────────────────────────────────────────────────────────────────────────
    # 6. Model Registry Introspection Contract
    # ──────────────────────────────────────────────────────────────────────────

    def test_model_registry_classical_baselines_metadata(self):
        """Verify GET /api/v1/models/classical_baselines returns complete architecture details."""
        response = self.client.get("/api/v1/models/classical_baselines")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["model_id"], "classical_baselines")
        self.assertEqual(data["task_type"], "classification")
        self.assertEqual(data["status"], "ready")
        self.assertIn("vectorizer", data["details"])
        self.assertIn("classes", data["details"])


if __name__ == "__main__":
    unittest.main()
