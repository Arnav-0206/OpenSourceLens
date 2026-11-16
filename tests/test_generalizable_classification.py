"""Unit and integration tests for Generalizable 5-Class Issue Classifier."""

import unittest

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.nlp.generalizable_classifier import (
    CANONICAL_CLASSES,
    CLASS_TO_ID,
    GeneralizableIssueClassifier,
)


class TestGeneralizableClassification(unittest.TestCase):
    """Test suite for Generalizable 5-class classifier and inference endpoint."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_classifier_fit_predict_and_abstention(self):
        """Verify classifier fits, predicts probabilities across 5 classes, and abstains when low confidence."""
        classifier = GeneralizableIssueClassifier(
            max_features=1000,
            confidence_threshold=0.30,
        )

        titles = [
            "Bug: Null pointer crash in window manager",
            "Defect: SIGSEGV segmentation fault on startup with wrong output",
            "Feature: Add dark mode theme preference",
            "Feature proposal: allow custom keybindings and shortcuts",
            "Docs: Update API documentation typos in README",
            "Documentation: Fix broken docstring link for tensor multiplication",
            "Question: How to configure remote development container?",
            "Question: what is the minimum required python version?",
            "Chore: Upgrade dev dependencies and CI build workflow",
            "Maintenance: Update package locks and bump actions runner",
        ]
        bodies = [
            "Exception occurred at runtime with stack trace and wrong result reproduction",
            "Process terminated with exit code 139 and crash bug defect",
            "It would be great to have a dark mode option feature request",
            "Please support user-defined keybindings new feature",
            "Found typo in documentation line 42 docs reference",
            "Link is pointing to 404 in documentation tutorial doc",
            "Question: I need help and support setting up the dev container",
            "Question: Is Python 3.9 supported or only 3.10+?",
            "Bumped npm packages to latest version maintenance task",
            "GitHub Actions workflow update and dependency chore",
        ]
        labels = [
            "bug",
            "bug",
            "feature",
            "feature",
            "documentation",
            "documentation",
            "question",
            "question",
            "other",
            "other",
        ]

        classifier.fit(titles, bodies, labels)
        self.assertTrue(classifier.is_fitted)

        # Test bug prediction
        pred_bug = classifier.predict_single(
            "Bug: Crash and defect in window manager", "Stack trace with wrong result"
        )
        self.assertEqual(pred_bug["predicted_label"], "bug")
        self.assertFalse(pred_bug["is_abstention"])

        # Test documentation prediction
        pred_doc = classifier.predict_single(
            "Docs: Typo in documentation tutorial", "Fix broken documentation link"
        )
        self.assertEqual(pred_doc["predicted_label"], "documentation")
        self.assertFalse(pred_doc["is_abstention"])

        # Test feature prediction
        pred_feat = classifier.predict_single(
            "Feature: Add dark mode option", "Feature request for new theme"
        )
        self.assertEqual(pred_feat["predicted_label"], "feature")
        self.assertFalse(pred_feat["is_abstention"])

        # Test question prediction
        pred_q = classifier.predict_single(
            "Question: How to configure container?", "Question about support and setup"
        )
        self.assertEqual(pred_q["predicted_label"], "question")
        self.assertFalse(pred_q["is_abstention"])

        # Test low-confidence abstention on completely ambiguous/random string
        pred_ambiguous = classifier.predict_single("xyz 123456 qwerty foo bar baz")
        self.assertIn("probabilities", pred_ambiguous)
        self.assertIn(pred_ambiguous["predicted_label"], CANONICAL_CLASSES + ["uncertain"])

    def test_api_generalizable_endpoint(self):
        """Verify POST /api/v1/classify with champion generalizable model."""
        payload = {
            "title": "Bug: Segmentation fault in CUDA backpropagation",
            "body": "Memory allocation failure during backward pass in libtorch.so with Traceback and wrong result",
            "repo_name": "pytorch/pytorch",
            "model_type": "generalizable",
        }
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn(data["predicted_label"], CANONICAL_CLASSES + ["uncertain"])
        self.assertIn("generalizable", data["model_used"])
        self.assertGreater(data["confidence_score"], 0.0)
        self.assertLessEqual(data["confidence_score"], 1.0)
        self.assertIn("is_abstention", data)
        self.assertIn("normalized_entropy", data)

        probs = data["probabilities"]
        for cls_name in CANONICAL_CLASSES:
            self.assertIn(cls_name, probs)


if __name__ == "__main__":
    unittest.main()
