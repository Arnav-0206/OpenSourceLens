"""
Unit and regression tests for OpenSourceLens Milestone Demo (demo_app.py).
Verifies preset catalogs, figure gallery paths, inference routing, explainability,
and semantic retrieval formatting without requiring a live web browser.
"""

from pathlib import Path
import unittest

from demo_app import (
    FIGURE_GALLERY,
    PRESETS,
    classify_with_classical,
    classify_with_transformer,
    format_badge,
    get_preset_issues,
    load_classical_pipeline,
    load_retrieval_pipeline,
    load_transformer_classifier,
    retrieve_related_issues,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestMilestoneDemoApp(unittest.TestCase):
    """Test suite for milestone demonstration app components."""

    def test_presets_validity(self):
        """Verify preset scenarios have valid titles, bodies, and descriptions."""
        presets = get_preset_issues()
        self.assertIsInstance(presets, dict)
        self.assertGreaterEqual(len(presets), 4)

        # Skip default placeholder preset
        for key, data in presets.items():
            self.assertIn("title", data)
            self.assertIn("body", data)
            self.assertIn("description", data)
            if key != "Select a preset or custom input...":
                self.assertGreater(len(data["title"]), 10, f"Preset {key} title too short")
                self.assertGreater(len(data["body"]), 20, f"Preset {key} body too short")

    def test_figure_gallery_paths_exist(self):
        """Verify that every single publication figure referenced in the gallery exists on disk."""
        total_figures = 0
        missing_figures = []

        for category, figures in FIGURE_GALLERY.items():
            self.assertGreater(len(figures), 0, f"Category {category} has no figures")
            for fig_info in figures:
                total_figures += 1
                fig_path = PROJECT_ROOT / fig_info["path"]
                if not fig_path.exists():
                    missing_figures.append(str(fig_info["path"]))

        self.assertEqual(
            len(missing_figures),
            0,
            f"The following figures are missing from disk: {missing_figures}",
        )
        self.assertGreaterEqual(total_figures, 10, "Expected at least 10 featured figures in gallery")

    def test_format_badge_html(self):
        """Verify badge formatting helper generates correct HTML and colors."""
        bug_badge = format_badge("bug")
        self.assertIn("#d73a49", bug_badge)
        self.assertIn("BUG", bug_badge)

        feat_badge = format_badge("feature")
        self.assertIn("#0366d6", feat_badge)
        self.assertIn("FEATURE", feat_badge)

        high_badge = format_badge("HIGH")
        self.assertIn("#28a745", high_badge)
        self.assertIn("HIGH", high_badge)

    def test_classical_pipeline_and_explainability(self):
        """Verify loading classical pipeline and extracting top contributing features."""
        pipeline = load_classical_pipeline()
        self.assertIsNotNone(pipeline, "Failed to load classical pipeline from disk")

        title = "Memory leak and crash in typescript extension host"
        body = "Allocation failed - JavaScript heap out of memory error"
        res = classify_with_classical(pipeline, title, body, top_k_features=5)

        self.assertIn("predicted_class", res)
        self.assertIn("confidence", res)
        self.assertIn("class_probabilities", res)
        self.assertIn("top_contributing_features", res)
        self.assertIn(res["predicted_class"], ["bug", "feature", "other"])
        self.assertGreater(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)

        # Check top features
        top_terms = res["top_contributing_features"]
        self.assertIsInstance(top_terms, list)
        self.assertGreater(len(top_terms), 0)
        self.assertIn("feature", top_terms[0])
        self.assertIn("contribution_score", top_terms[0])

    def test_transformer_classification(self):
        """Verify DistilBERT inference returns proper structure and probabilities."""
        classifier = load_transformer_classifier()
        self.assertIsNotNone(classifier, "Failed to load DistilBERT classifier from disk")

        title = "VS Code editor crashes when opening large workspace"
        body = "Heap out of memory error in renderer process"
        res = classify_with_transformer(classifier, title, body)

        self.assertIn("label", res)
        self.assertIn("confidence", res)
        self.assertIn("probabilities", res)
        self.assertIn("is_uncertain", res)
        self.assertIn(res["label"], ["bug", "feature", "other"])

        probs = res["probabilities"]
        self.assertEqual(set(probs.keys()), {"bug", "feature", "other"})
        total_prob = sum(probs.values())
        self.assertAlmostEqual(total_prob, 1.0, places=2)

    def test_semantic_retrieval_and_filtering(self):
        """Verify semantic retrieval pipeline retrieves related issues and applies filters."""
        retriever = load_retrieval_pipeline()
        self.assertIsNotNone(retriever, "Failed to load RelatedIssuePipeline from disk")

        title = "CUDA error: out of memory when allocating tensor"
        body = "RuntimeError: CUDA out of memory. Tried to allocate 2 GiB"

        res = retrieve_related_issues(
            pipeline=retriever,
            title=title,
            body=body,
            repo_filter="All Repositories",
            top_k=5,
            min_confidence="ALL",
        )

        self.assertIn("summary", res)
        self.assertIn("related_issues", res)
        self.assertIn("execution_metadata", res)

        issues = res["related_issues"]
        self.assertGreater(len(issues), 0)
        self.assertLessEqual(len(issues), 5)

        # Check first candidate fields
        top_cand = issues[0]
        self.assertIn("title", top_cand)
        self.assertIn("similarity_score", top_cand)
        self.assertIn("confidence_tier", top_cand)
        self.assertIn("html_url", top_cand)
        self.assertTrue(top_cand["html_url"].startswith("https://github.com/"))
        self.assertGreater(top_cand["similarity_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
