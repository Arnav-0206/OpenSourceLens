"""Unit tests for LabelNormalizer in ml.preprocessing.label_normalizer."""

import unittest
import pandas as pd

from ml.preprocessing.label_normalizer import LabelNormalizer


class TestLabelNormalizer(unittest.TestCase):
    """Test suite for LabelNormalizer."""

    def setUp(self):
        self.normalizer = LabelNormalizer()

    def test_map_labels_bug(self):
        self.assertEqual(self.normalizer.map_labels(["bug"]), "bug")
        self.assertEqual(self.normalizer.map_labels(["kind/bug", "sig/node"]), "bug")
        self.assertEqual(self.normalizer.map_labels(["module: correctness (silent)"]), "bug")
        self.assertEqual(self.normalizer.map_labels(["CUDA Crash on A100"]), "bug")
        self.assertEqual(self.normalizer.map_labels(["Kind/Flake"]), "bug")

    def test_map_labels_feature(self):
        self.assertEqual(self.normalizer.map_labels(["Feature Request"]), "feature")
        self.assertEqual(self.normalizer.map_labels(["kind/feature"]), "feature")
        self.assertEqual(self.normalizer.map_labels(["enhancement", "ui"]), "feature")
        self.assertEqual(self.normalizer.map_labels(["new model"]), "feature")

    def test_map_labels_documentation(self):
        self.assertEqual(self.normalizer.map_labels(["documentation"]), "documentation")
        self.assertEqual(self.normalizer.map_labels(["kind/documentation"]), "documentation")
        self.assertEqual(self.normalizer.map_labels(["docs"]), "documentation")
        self.assertEqual(self.normalizer.map_labels(["module: docs"]), "documentation")

    def test_map_labels_question(self):
        self.assertEqual(self.normalizer.map_labels(["question"]), "question")
        self.assertEqual(self.normalizer.map_labels(["kind/support"]), "question")
        self.assertEqual(self.normalizer.map_labels(["support", "triage"]), "question")

    def test_map_labels_other_and_unlabeled(self):
        # Workflow / metadata only
        self.assertEqual(self.normalizer.map_labels(["sig/node", "bot-triaged", "triaged"]), "other")
        self.assertEqual(self.normalizer.map_labels(["new release", "insiders-released"]), "other")

        # Unlabeled / Empty / None
        self.assertEqual(self.normalizer.map_labels([]), "unlabeled")
        self.assertEqual(self.normalizer.map_labels(None), "unlabeled")
        self.assertEqual(self.normalizer.map_labels([""]), "unlabeled")

    def test_deterministic_priority(self):
        # Bug + Feature -> Bug
        self.assertEqual(self.normalizer.map_labels(["kind/bug", "enhancement"]), "bug")
        # Feature + Docs -> Feature
        self.assertEqual(self.normalizer.map_labels(["feature", "docs"]), "feature")
        # Docs + Question -> Docs
        self.assertEqual(self.normalizer.map_labels(["documentation", "question"]), "documentation")

    def test_normalize_dataframe(self):
        df = pd.DataFrame([
            {"issue_id": 1, "raw_labels": ["bug"], "repo_name": "microsoft/vscode"},
            {"issue_id": 2, "raw_labels": ["feature request"], "repo_name": "microsoft/vscode"},
            {"issue_id": 3, "raw_labels": ["sig/node"], "repo_name": "kubernetes/kubernetes"},
            {"issue_id": 4, "raw_labels": [], "repo_name": "pytorch/pytorch"},
        ])

        norm_df = self.normalizer.normalize_dataframe(df)

        self.assertIn("canonical_label", norm_df.columns)
        self.assertIn("canonical_label_id", norm_df.columns)
        self.assertIn("is_classified_category", norm_df.columns)

        # Row 0: bug
        self.assertEqual(norm_df.loc[0, "canonical_label"], "bug")
        self.assertEqual(norm_df.loc[0, "canonical_label_id"], 0)
        self.assertTrue(norm_df.loc[0, "is_classified_category"])

        # Row 1: feature
        self.assertEqual(norm_df.loc[1, "canonical_label"], "feature")
        self.assertEqual(norm_df.loc[1, "canonical_label_id"], 1)
        self.assertTrue(norm_df.loc[1, "is_classified_category"])

        # Row 2: other
        self.assertEqual(norm_df.loc[2, "canonical_label"], "other")
        self.assertEqual(norm_df.loc[2, "canonical_label_id"], 4)
        self.assertFalse(norm_df.loc[2, "is_classified_category"])

        # Row 3: unlabeled
        self.assertEqual(norm_df.loc[3, "canonical_label"], "unlabeled")
        self.assertEqual(norm_df.loc[3, "canonical_label_id"], -1)
        self.assertFalse(norm_df.loc[3, "is_classified_category"])

    def test_summarize_and_format_report(self):
        df = pd.DataFrame([
            {"issue_id": 1, "canonical_label": "bug", "is_classified_category": True, "repo_name": "repoA"},
            {"issue_id": 2, "canonical_label": "feature", "is_classified_category": True, "repo_name": "repoA"},
            {"issue_id": 3, "canonical_label": "other", "is_classified_category": False, "repo_name": "repoB"},
        ])

        summary = LabelNormalizer.summarize_distribution(df)
        self.assertEqual(summary["total_records"], 3)
        self.assertEqual(summary["core_candidates_count"], 2)
        self.assertEqual(summary["core_candidates_percent"], 66.67)

        report = LabelNormalizer.format_distribution_report(summary)
        self.assertIn("### Label Normalization & Taxonomy Summary", report)
        self.assertIn("repoA", report)
        self.assertIn("repoB", report)


if __name__ == "__main__":
    unittest.main()
