"""
Unit and integration tests for ml/repository_analyzer.py.
Verifies repository input normalization, batch classification, category distribution,
and pairwise intra-repository duplicate detection.
"""

from pathlib import Path
import unittest

import numpy as np

from ml.deep_learning.pipeline import TransformerIssueClassifier
from ml.repository_analyzer import (
    calculate_category_breakdown,
    classify_issues_batch,
    find_intra_repo_duplicates,
    normalize_repo_input,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class MockEmbedder:
    """Mock embedder returning deterministic orthogonal or duplicate vectors for testing."""

    def __init__(self, dim: int = 16):
        self.dim = dim

    def encode(self, texts, normalize_embeddings=True):
        vectors = []
        for idx, text in enumerate(texts):
            # Create vectors: 0 and 1 are almost identical, 2 is orthogonal
            vec = np.zeros(self.dim, dtype=np.float32)
            if idx == 0:
                vec[0] = 1.0
            elif idx == 1:
                vec[0] = 0.95
                vec[1] = 0.05
            else:
                vec[2] = 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vectors.append(vec)
        return np.array(vectors)


class TestRepositoryAnalyzer(unittest.TestCase):
    """Test suite for repository analyzer components."""

    def test_normalize_repo_input_valid(self):
        """Verify URL and slug normalization handles diverse formats."""
        cases = [
            ("psf/requests", ("psf", "requests")),
            ("pallets/flask/", ("pallets", "flask")),
            ("https://github.com/scikit-learn/scikit-learn", ("scikit-learn", "scikit-learn")),
            ("http://github.com/astral-sh/uv/", ("astral-sh", "uv")),
            ("github.com/fastapi/fastapi", ("fastapi", "fastapi")),
        ]
        for inp, expected in cases:
            self.assertEqual(normalize_repo_input(inp), expected)

    def test_normalize_repo_input_invalid(self):
        """Verify malformed repository inputs raise ValueError."""
        invalid_cases = ["", "   ", "singleword", "too/many/parts/here", "https://otherhost.com/a/b/c"]
        for inp in invalid_cases:
            with self.assertRaises(ValueError):
                normalize_repo_input(inp)

    def test_calculate_category_breakdown(self):
        """Verify distribution calculation handles empty and standard lists."""
        empty_res = calculate_category_breakdown([])
        self.assertEqual(empty_res["total"], 0)
        self.assertEqual(empty_res["percentages"]["bug"], 0.0)

        classified = [
            {"predicted_category": "bug"},
            {"predicted_category": "bug"},
            {"predicted_category": "feature"},
            {"predicted_category": "other"},
        ]
        res = calculate_category_breakdown(classified)
        self.assertEqual(res["total"], 4)
        self.assertEqual(res["counts"]["bug"], 2)
        self.assertEqual(res["counts"]["feature"], 1)
        self.assertEqual(res["counts"]["other"], 1)
        self.assertEqual(res["percentages"]["bug"], 50.0)
        self.assertEqual(res["percentages"]["feature"], 25.0)
        self.assertEqual(res["percentages"]["other"], 25.0)

    def test_find_intra_repo_duplicates_with_mock(self):
        """Verify duplicate detection identifies high-similarity pairs."""
        mock_embedder = MockEmbedder()
        issues = [
            {"issue_number": 101, "title": "Memory leak on socket close", "body": "Heap out of memory"},
            {"issue_number": 102, "title": "Memory leak on socket close", "body": "Heap out of memory crash"},
            {"issue_number": 103, "title": "Add dark mode theme", "body": "Feature request for dark theme"},
        ]

        result = find_intra_repo_duplicates(mock_embedder, issues, duplicate_threshold=0.80)
        self.assertTrue(result["has_duplicates"])
        self.assertEqual(len(result["duplicate_pairs"]), 1)

        dup = result["duplicate_pairs"][0]
        self.assertEqual(dup["issue_1"]["issue_number"], 101)
        self.assertEqual(dup["issue_2"]["issue_number"], 102)
        self.assertGreaterEqual(dup["similarity_score"], 0.80)
        self.assertIn("recommendation", dup)

    def test_batch_classification(self):
        """Verify batch classification runs DistilBERT on sample issues."""
        model_path = PROJECT_ROOT / "models" / "distilbert_finetuned"
        if not model_path.exists():
            self.skipTest("DistilBERT model not found on disk")

        classifier = TransformerIssueClassifier(model_dir=model_path)
        sample_issues = [
            {"issue_number": 1, "title": "Crash when opening large workspace", "body": "Heap error"},
            {"issue_number": 2, "title": "Add support for Python 3.14", "body": "Feature request"},
        ]

        enriched = classify_issues_batch(classifier, sample_issues)
        self.assertEqual(len(enriched), 2)
        for iss in enriched:
            self.assertIn("predicted_category", iss)
            self.assertIn("confidence", iss)
            self.assertIn("probabilities", iss)
            self.assertIn(iss["predicted_category"], ["bug", "feature", "other"])


if __name__ == "__main__":
    unittest.main()
