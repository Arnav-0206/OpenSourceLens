"""
Unit and regression tests for OpenSourceLens Milestone Demo (demo_app.py).
Verifies repository presets, single-issue presets, figure gallery paths,
category badge rendering, and model resource loaders without requiring a live web browser.
"""

from pathlib import Path
import unittest

from demo_app import (
    FIGURE_GALLERY,
    REPO_PRESETS,
    SINGLE_ISSUE_PRESETS,
    get_classical_pipeline,
    get_retrieval_pipeline,
    get_sentence_embedder,
    get_transformer_classifier,
    render_category_badge,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestMilestoneDemoApp(unittest.TestCase):
    """Test suite for milestone demonstration app components."""

    def test_repo_presets_validity(self):
        """Verify repository presets have valid names, labels, and descriptions."""
        self.assertIsInstance(REPO_PRESETS, list)
        self.assertGreaterEqual(len(REPO_PRESETS), 3)

        for p in REPO_PRESETS:
            self.assertIn("name", p)
            self.assertIn("label", p)
            self.assertIn("desc", p)
            self.assertIn("/", p["name"])

    def test_single_issue_presets_validity(self):
        """Verify single-issue scenarios for Model Lab have valid titles and bodies."""
        self.assertIsInstance(SINGLE_ISSUE_PRESETS, dict)
        self.assertGreaterEqual(len(SINGLE_ISSUE_PRESETS), 3)

        for key, data in SINGLE_ISSUE_PRESETS.items():
            self.assertIn("title", data)
            self.assertIn("body", data)
            if key != "Select an issue scenario...":
                self.assertGreater(len(data["title"]), 5)

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

    def test_render_category_badge(self):
        """Verify category badge formatting helper generates correct HTML and colors."""
        bug_badge = render_category_badge("bug")
        self.assertIn("#ef4444", bug_badge)
        self.assertIn("BUG", bug_badge)

        feat_badge = render_category_badge("feature")
        self.assertIn("#3b82f6", feat_badge)
        self.assertIn("FEATURE", feat_badge)

        other_badge = render_category_badge("other")
        self.assertIn("#6b7280", other_badge)
        self.assertIn("OTHER", other_badge)

    def test_model_resource_loaders(self):
        """Verify cached loaders load production models from disk."""
        transformer = get_transformer_classifier()
        self.assertIsNotNone(transformer, "Failed to load Transformer classifier")
        self.assertIn("bug", transformer.class_names)

        classical = get_classical_pipeline()
        self.assertIsNotNone(classical, "Failed to load Classical pipeline")
        self.assertIn("bug", classical.classes_)

        embedder = get_sentence_embedder()
        self.assertIsNotNone(embedder, "Failed to load Sentence embedder")

        retriever = get_retrieval_pipeline()
        self.assertIsNotNone(retriever, "Failed to load Retrieval pipeline")
        self.assertTrue(retriever.store.is_loaded)


if __name__ == "__main__":
    unittest.main()
