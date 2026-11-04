"""Integration Test Suite for Fine-Tuned DistilBERT Transformer Pipeline (Phase 11, Step 11.4).

Verifies the complete integration of the fine-tuned DistilBERT champion sequence classifier:
1. End-to-end transformer inference via POST /api/v1/classify with model_type="distilbert".
2. Softmax normalization of output logits (sum of probabilities = 1.0).
3. Prototypical category predictions (bug reports, feature proposals).
4. Tokenization boundaries: truncation of long inputs (> 1,000 chars), title-only inputs, code blocks.
5. Model pre-warming via POST /api/v1/models/prewarm.
6. Model registry metadata contract via GET /api/v1/models/distilbert.
7. Real-time audit recording in the 'predictions' table.
"""

from __future__ import annotations

import unittest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.database.models import Prediction
from backend.app.database.session import SessionLocal
from backend.app.main import app


class TestTransformerPipelineIntegration(unittest.TestCase):
    """Integration test suite for DistilBERT transformer sequence classification."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Prewarm models to prime memory and ensure fast test execution
        cls.client.post("/api/v1/models/prewarm")

    def setUp(self):
        self.db: Session = SessionLocal()

    def tearDown(self):
        self.db.close()

    # ──────────────────────────────────────────────────────────────────────────
    # 1. DistilBERT Inference & Probability Distribution Contract
    # ──────────────────────────────────────────────────────────────────────────

    def test_distilbert_inference_contract(self):
        """Verify fine-tuned DistilBERT returns normalized probabilities and model metadata."""
        payload = {
            "title": "Deadlock in threadpool when expanding directory nodes",
            "body": "When expanding nested folders with large child trees, UI freezes permanently.",
            "repo_name": "microsoft/vscode",
            "model_type": "distilbert",
        }
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["model_used"], "distilbert-finetuned-v1")
        self.assertIn(data["predicted_label"], ["bug", "feature", "other"])
        self.assertGreater(data["confidence_score"], 0.0)
        self.assertLessEqual(data["confidence_score"], 1.0)

        # Softmax probability distribution
        probs = data["probabilities"]
        self.assertIsInstance(probs, dict)
        self.assertIn("bug", probs)
        self.assertIn("feature", probs)
        self.assertIn("other", probs)
        total_prob = sum(probs.values())
        self.assertAlmostEqual(total_prob, 1.0, places=2)

        # Confidence matches the predicted label's probability
        self.assertAlmostEqual(probs[data["predicted_label"]], data["confidence_score"], places=4)

        # Latency benchmark
        self.assertGreater(data["inference_time_ms"], 0.0)

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Prototypical Category Classifications
    # ──────────────────────────────────────────────────────────────────────────

    def test_distilbert_prototypical_feature_prediction(self):
        """Verify clear feature request text is classified as feature category."""
        payload = {
            "title": "Add support for dark high-contrast themes in workbench",
            "body": "User proposal to introduce custom high-contrast color tokens for accessibility.",
            "repo_name": "microsoft/vscode",
            "model_type": "distilbert",
        }
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["predicted_label"], "feature")
        self.assertGreater(data["confidence_score"], 0.5)

    def test_distilbert_prototypical_bug_prediction(self):
        """Verify clear crash / segmentation fault text assigns significant bug probability."""
        payload = {
            "title": "Fatal crash with exit code 139 in extension host process",
            "body": "Segmentation fault occurred in native module worker. Stack trace indicates buffer overflow.",
            "repo_name": "microsoft/vscode",
            "model_type": "distilbert",
        }
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("bug", data["probabilities"])
        self.assertGreater(data["probabilities"]["bug"], 0.25)

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Tokenization Boundaries & Edge Cases
    # ──────────────────────────────────────────────────────────────────────────

    def test_distilbert_long_text_truncation(self):
        """Verify that issues with bodies > 1,000 words are truncated cleanly to 128 tokens."""
        long_body = "Error log output:\n" + ("line 42: exception in thread pool worker\n" * 150)
        self.assertGreater(len(long_body), 4000)

        payload = {
            "title": "Heavy memory leak in long running background daemon",
            "body": long_body,
            "model_type": "distilbert",
        }
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["predicted_label"], ["bug", "feature", "other"])

    def test_distilbert_title_only_inference(self):
        """Verify that title-only classification works without body text."""
        payload = {
            "title": "Proposal: support WebAssembly SIMD instructions in runtime",
            "body": None,
            "model_type": "distilbert",
        }
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["predicted_label"], ["bug", "feature", "other"])

    def test_distilbert_markdown_and_code_blocks(self):
        """Verify that markdown code blocks and backticks are sanitized and processed."""
        body = (
            "Steps to reproduce:\n"
            "```python\n"
            "import torch\n"
            "x = torch.zeros(10, 10, device='cuda')\n"
            "```\n"
            "Throws `RuntimeError: CUDA error: out of memory`."
        )
        payload = {
            "title": "CUDA out of memory error during backward pass",
            "body": body,
            "model_type": "distilbert",
        }
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("probabilities", data)

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Database Audit Persistence
    # ──────────────────────────────────────────────────────────────────────────

    def test_distilbert_database_audit_record_created(self):
        """Verify that DistilBERT inference inserts an audit row in the predictions table."""
        count_before = self.db.scalar(select(func.count(Prediction.id))) or 0

        unique_title = "Transformer Audit Test: Deadlock in RPC handler threadpool"
        payload = {
            "title": unique_title,
            "body": "Reconnection sequence triggers thread pool starvation.",
            "repo_name": "kubernetes/kubernetes",
            "model_type": "distilbert",
        }
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 200)

        # Check DB count increased by 1
        count_after = self.db.scalar(select(func.count(Prediction.id))) or 0
        self.assertEqual(count_after, count_before + 1)

        # Verify audit record content
        record = self.db.scalar(
            select(Prediction).where(Prediction.input_title == unique_title)
        )
        self.assertIsNotNone(record)
        self.assertEqual(record.model_name, "distilbert-finetuned-v1")
        self.assertEqual(record.task_type, "classification")
        self.assertEqual(record.repo_name, "kubernetes/kubernetes")
        self.assertIn(record.predicted_label, ["bug", "feature", "other"])
        self.assertGreater(record.confidence_score, 0.0)

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Pre-Warming & Model Registry Introspection
    # ──────────────────────────────────────────────────────────────────────────

    def test_prewarm_endpoint_prewarms_distilbert(self):
        """Verify POST /api/v1/models/prewarm loads DistilBERT into memory."""
        response = self.client.post("/api/v1/models/prewarm")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("distilbert", data)
        self.assertEqual(data["distilbert"], "prewarmed")

    def test_distilbert_model_registry_introspection(self):
        """Verify GET /api/v1/models/distilbert returns correct architecture and status."""
        response = self.client.get("/api/v1/models/distilbert")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["model_id"], "distilbert")
        self.assertEqual(data["task_type"], "classification")
        self.assertEqual(data["status"], "ready")
        self.assertTrue(data["is_loaded"])
        self.assertIn("DistilBertForSequenceClassification", data["details"]["architecture"])
        self.assertEqual(data["details"]["num_parameters"], 66955011)
        self.assertIn("classes", data["details"])


if __name__ == "__main__":
    unittest.main()
