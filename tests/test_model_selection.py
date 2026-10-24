"""Unit Tests for Transformer Model Selection (Step 6.1)."""

from pathlib import Path
import tempfile
import unittest
import json

from ml.deep_learning.model_selection import TransformerModelSelector


class TestTransformerModelSelection(unittest.TestCase):
    """Test suite for TransformerModelSelector."""

    def test_initialization(self):
        """Verify default initialization and invalid model handling."""
        selector = TransformerModelSelector()
        self.assertEqual(selector.primary_model_name, "distilbert-base-uncased")

        with self.assertRaises(ValueError):
            TransformerModelSelector(primary_model_name="nonexistent-model-xyz")

    def test_detect_hardware(self):
        """Verify hardware detection schema and environment discovery."""
        selector = TransformerModelSelector()
        hw = selector.detect_hardware()

        self.assertIn("target_device", hw)
        self.assertIn(hw["target_device"], ["cpu", "cuda"])
        self.assertIn("torch_version", hw)
        self.assertIn("transformers_version", hw)
        self.assertIn("accelerate_version", hw)
        self.assertIn("cpu_threads", hw)
        self.assertGreater(hw["cpu_threads"], 0)
        self.assertIn("device_details", hw)

    def test_get_candidate_models(self):
        """Verify candidate models registry completeness."""
        selector = TransformerModelSelector()
        candidates = selector.get_candidate_models()

        self.assertEqual(len(candidates), 4)
        model_names = [m["model_name"] for m in candidates]
        self.assertIn("distilbert-base-uncased", model_names)
        self.assertIn("bert-base-uncased", model_names)
        self.assertIn("roberta-base", model_names)
        self.assertIn("microsoft/codebert-base", model_names)

        for m in candidates:
            self.assertIn("parameters", m)
            self.assertIn("layers", m)
            self.assertIn("hidden_dimension", m)
            self.assertIn("pros", m)
            self.assertIn("cons", m)

    def test_select_model(self):
        """Verify model selection configuration and label mappings."""
        selector = TransformerModelSelector()
        sel = selector.select_model("distilbert-base-uncased", max_seq_length=256, batch_size=16)

        self.assertEqual(sel["selected_model_name"], "distilbert-base-uncased")
        self.assertEqual(sel["num_labels"], 3)
        self.assertEqual(sel["target_classes"], ["bug", "feature", "other"])

        # Label encoding verification
        self.assertEqual(sel["label2id"], {"bug": 0, "feature": 1, "other": 2})
        self.assertEqual(sel["id2label"], {0: "bug", 1: "feature", 2: "other"})

        # Resource verification
        self.assertEqual(sel["max_sequence_length"], 256)
        self.assertEqual(sel["batch_size"], 16)
        self.assertGreater(sel["batches_per_epoch"], 0)

    def test_save_and_run_all(self):
        """Verify end-to-end execution and report persistence."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = Path(tmp_dir) / "benchmark.json"

            selector = TransformerModelSelector()
            res = selector.run_all(output_path=report_path)

            self.assertTrue(report_path.exists())
            self.assertGreater(report_path.stat().st_size, 500)

            with open(report_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data["metadata"]["phase"], "6")
            self.assertEqual(data["metadata"]["step"], "6.1")
            self.assertIn("hardware_profile", data)
            self.assertIn("model_selection", data)
            self.assertIn("candidate_architectures_benchmarked", data)


if __name__ == "__main__":
    unittest.main()
