"""Integration Test Suite for Resolution Velocity Advisory Pipeline (Phase 11, Step 11.6).

Verifies the complete integration of the resolution analysis and velocity prediction subsystem:
1. End-to-end velocity prediction via POST /api/v1/resolution/predict with Random Forest champion.
2. 4-tier probability distribution calibration and normalization (sum = 1.0).
3. Conservative adjacent accuracy boundary windows (lower_bound_tier to upper_bound_tier).
4. Domain contextual advisory factors and mandatory scientific reliability notice.
5. Database audit logging into the 'predictions' table (task_type='resolution_prediction').
6. Repository descriptive resolution SLA statistics via GET /api/v1/resolution/stats/{owner}/{name}.
7. Percentile consistency (P25 <= Median <= P75) across all 4 target repositories.
8. Model registry introspection contract for resolution_velocity subsystem.
"""

from __future__ import annotations

import unittest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.database.models import Prediction
from backend.app.database.session import SessionLocal
from backend.app.main import app


class TestResolutionPipelineIntegration(unittest.TestCase):
    """Integration test suite for resolution velocity prediction and historical SLA analytics."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.target_repos = [
            ("microsoft", "vscode"),
            ("pytorch", "pytorch"),
            ("kubernetes", "kubernetes"),
            ("huggingface", "transformers"),
        ]
        cls.valid_tiers = ["< 1 Day", "1-7 Days", "7-30 Days", "30+ Days"]

    def setUp(self):
        self.db: Session = SessionLocal()

    def tearDown(self):
        self.db.close()

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Resolution Velocity Prediction Contract
    # ──────────────────────────────────────────────────────────────────────────

    def test_resolution_velocity_prediction_contract(self):
        """Verify POST /api/v1/resolution/predict returns 4-tier prediction with error bounds."""
        payload = {
            "repo_name": "microsoft/vscode",
            "title": "Intermittent crash in terminal process worker",
            "body": "When reloading window with active terminal, native process exits with code 139.",
            "canonical_label": "bug",
            "author_association": "MEMBER",
            "labels_count": 2,
        }
        response = self.client.post("/api/v1/resolution/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["repo_name"], "microsoft/vscode")
        self.assertEqual(data["model_used"], "resolution-rf-classifier-champion")
        self.assertIn(data["predicted_speed_tier"], self.valid_tiers)
        self.assertGreater(data["confidence_score"], 0.0)
        self.assertLessEqual(data["confidence_score"], 1.0)

        # Probabilities across all 4 tiers
        probs = data["tier_probabilities"]
        self.assertIsInstance(probs, dict)
        for tier in self.valid_tiers:
            self.assertIn(tier, probs)
        total_prob = sum(probs.values())
        self.assertAlmostEqual(total_prob, 1.0, places=2)

        # Adjacent bounds
        self.assertIn(data["lower_bound_tier"], self.valid_tiers)
        self.assertIn(data["upper_bound_tier"], self.valid_tiers)

        # Advisory context factors & reliability disclaimer
        self.assertIsInstance(data["advisory_factors"], list)
        self.assertGreater(len(data["advisory_factors"]), 0)
        self.assertIn("reliability_notice", data)
        self.assertGreater(len(data["reliability_notice"]), 20)

        # Latency benchmark
        self.assertGreater(data["inference_time_ms"], 0.0)

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Fast vs. Slow Velocity Prototypical Predictions
    # ──────────────────────────────────────────────────────────────────────────

    def test_fast_tier_velocity_prediction(self):
        """Verify member-filed bug in rapid-triage repo assigns high probability to fast tiers."""
        payload = {
            "repo_name": "microsoft/vscode",
            "title": "Fix regression in status bar item display alignment",
            "body": "Commit abc1234 regressed status bar layout. Quick patch needed.",
            "canonical_label": "bug",
            "author_association": "MEMBER",
            "labels_count": 1,
        }
        response = self.client.post("/api/v1/resolution/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # In VS Code with MEMBER author, fast tiers (< 1 Day or 1-7 Days) should dominate
        probs = data["tier_probabilities"]
        fast_prob = probs.get("< 1 Day", 0.0) + probs.get("1-7 Days", 0.0)
        self.assertGreater(fast_prob, 0.5)

    def test_slower_tier_velocity_prediction(self):
        """Verify community feature proposal in complex infrastructure predicts slower velocity."""
        payload = {
            "repo_name": "kubernetes/kubernetes",
            "title": "Proposal: overhaul ingress controller dynamic TLS cert renewal protocol",
            "body": "Architectural proposal requiring design doc review, multiple API groups, and cluster tests.",
            "canonical_label": "feature",
            "author_association": "NONE",
            "labels_count": 5,
        }
        response = self.client.post("/api/v1/resolution/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        probs = data["tier_probabilities"]
        # In Kubernetes feature proposals by non-members, slower tiers should have substantial weight
        slow_prob = probs.get("7-30 Days", 0.0) + probs.get("30+ Days", 0.0)
        self.assertGreater(slow_prob, 0.05)

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Adjacent Accuracy Boundary Window
    # ──────────────────────────────────────────────────────────────────────────

    def test_adjacent_accuracy_window_structure(self):
        """Verify lower_bound_tier and upper_bound_tier enclose the predicted speed tier."""
        tier_indices = {tier: idx for idx, tier in enumerate(self.valid_tiers)}

        payload = {
            "repo_name": "pytorch/pytorch",
            "title": "Investigate CUDA memory fragmentation during long training",
            "canonical_label": "bug",
        }
        response = self.client.post("/api/v1/resolution/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        pred_idx = tier_indices[data["predicted_speed_tier"]]
        low_idx = tier_indices[data["lower_bound_tier"]]
        high_idx = tier_indices[data["upper_bound_tier"]]

        self.assertLessEqual(low_idx, pred_idx)
        self.assertGreaterEqual(high_idx, pred_idx)
        # Window spans at most 1 tier in either direction
        self.assertLessEqual(pred_idx - low_idx, 1)
        self.assertLessEqual(high_idx - pred_idx, 1)

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Database Audit Logging
    # ──────────────────────────────────────────────────────────────────────────

    def test_resolution_database_audit_record_created(self):
        """Verify that velocity prediction inserts an audit row into the predictions table."""
        count_before = self.db.scalar(select(func.count(Prediction.id))) or 0

        unique_title = "Resolution Audit Test: High memory utilization in daemon worker"
        payload = {
            "repo_name": "kubernetes/kubernetes",
            "title": unique_title,
            "body": "Memory spikes after 48 hours of pod churn.",
            "canonical_label": "bug",
            "author_association": "CONTRIBUTOR",
        }
        response = self.client.post("/api/v1/resolution/predict", json=payload)
        self.assertEqual(response.status_code, 200)

        # Check DB audit count increased by 1
        count_after = self.db.scalar(select(func.count(Prediction.id))) or 0
        self.assertEqual(count_after, count_before + 1)

        # Verify audit record content
        record = self.db.scalar(
            select(Prediction).where(Prediction.input_title == unique_title)
        )
        self.assertIsNotNone(record)
        self.assertEqual(record.model_name, "resolution-rf-classifier-champion")
        self.assertEqual(record.task_type, "resolution_prediction")
        self.assertEqual(record.repo_name, "kubernetes/kubernetes")
        self.assertIn(record.predicted_label, self.valid_tiers)
        self.assertGreater(record.confidence_score, 0.0)

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Descriptive Repository Resolution Statistics
    # ──────────────────────────────────────────────────────────────────────────

    def test_repository_resolution_stats_all_target_repos(self):
        """Verify GET /api/v1/resolution/stats returns factual SLA percentiles for all 4 repos."""
        for owner, name in self.target_repos:
            response = self.client.get(f"/api/v1/resolution/stats/{owner}/{name}")
            self.assertEqual(response.status_code, 200)
            data = response.json()

            self.assertEqual(data["repo_name"], f"{owner}/{name}")
            self.assertGreater(data["n_closed_issues"], 100)
            self.assertGreater(data["median_resolution_hours"], 0.0)
            self.assertGreater(data["p25_resolution_hours"], 0.0)
            self.assertGreater(data["p75_resolution_hours"], 0.0)

            # Percentile ordering: P25 <= Median <= P75
            self.assertLessEqual(data["p25_resolution_hours"], data["median_resolution_hours"])
            self.assertLessEqual(data["median_resolution_hours"], data["p75_resolution_hours"])

            # Closure rates and survival
            self.assertGreaterEqual(data["pct_resolved_under_24h"], 0.0)
            self.assertLessEqual(data["pct_resolved_under_24h"], 100.0)
            self.assertGreaterEqual(data["pct_resolved_under_7d"], data["pct_resolved_under_24h"])
            self.assertIsNotNone(data["survival_rate_30d"])
            self.assertGreaterEqual(data["survival_rate_30d"], 0.0)
            self.assertLessEqual(data["survival_rate_30d"], 1.0)

            # Tier distribution percentages
            self.assertIsInstance(data["tier_distribution"], dict)
            self.assertGreater(len(data["tier_distribution"]), 0)

    def test_repository_stats_nonexistent_repo_returns_404(self):
        """Verify requesting resolution stats for non-existent repository returns 404."""
        response = self.client.get("/api/v1/resolution/stats/fake_owner_123/nonexistent_repo")
        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertIn("detail", data)

    # ──────────────────────────────────────────────────────────────────────────
    # 6. Model Registry Introspection Contract
    # ──────────────────────────────────────────────────────────────────────────

    def test_model_registry_resolution_velocity_metadata(self):
        """Verify GET /api/v1/models/resolution_velocity returns champion classifier metadata."""
        response = self.client.get("/api/v1/models/resolution_velocity")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["model_id"], "resolution_velocity")
        self.assertEqual(data["task_type"], "resolution_prediction")
        self.assertEqual(data["status"], "ready")
        self.assertIn("RandomForestClassifier", data["details"]["classifier"])
        self.assertEqual(data["details"]["classes"], self.valid_tiers)
        self.assertIn("adjacent_accuracy", data["details"])


if __name__ == "__main__":
    unittest.main()
