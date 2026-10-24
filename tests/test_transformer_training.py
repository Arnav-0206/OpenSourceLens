"""Unit Tests for Transformer Fine-Tuning Pipeline (Step 6.4)."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.deep_learning.train import (
    TransformerFineTuner,
    compute_class_weights,
    set_seed,
)


class TestTransformerTraining(unittest.TestCase):
    """Test suite for TransformerFineTuner and associated training utilities."""

    @classmethod
    def setUpClass(cls):
        """Set up small dataset and mock trainer for fast unit tests."""
        set_seed(42)

        # Tiny synthetic DataFrame for test verification
        cls.mock_train_df = pd.DataFrame({
            "issue_id": ["iss_1", "iss_2", "iss_3", "iss_4", "iss_5", "iss_6"],
            "text": [
                "Title: Fix null pointer crash | Description: App terminates unexpectedly",
                "Title: Add dark mode toggle | Description: Feature request for user interface",
                "Title: Update documentation readme | Description: Docs typo fixes",
                "Title: Memory leak in buffer | Description: High heap consumption observed",
                "Title: Support ARM64 architecture | Description: Build fails on macOS arm",
                "Title: Refactor build script | Description: Cleanup unused variables",
            ],
            "label": [0, 1, 2, 0, 1, 2],
        })

        cls.mock_val_df = pd.DataFrame({
            "issue_id": ["iss_7", "iss_8", "iss_9"],
            "text": [
                "Title: Crash on launch | Description: Segmentation fault in core",
                "Title: Export to CSV feature | Description: Allow users to download reports",
                "Title: Update dependencies | Description: Upgrade numpy version",
            ],
            "label": [0, 1, 2],
        })

    def test_compute_class_weights(self):
        """Verify balanced inverse frequency weight computation."""
        labels = [0, 0, 0, 1, 2, 2]
        # N = 6, C = 3
        # Class 0: 6 / (3 * 3) = 2/3 = 0.6667
        # Class 1: 6 / (3 * 1) = 2.0
        # Class 2: 6 / (3 * 2) = 1.0
        weights = compute_class_weights(labels, num_classes=3)
        self.assertEqual(weights.shape, (3,))
        self.assertAlmostEqual(weights[0].item(), 6.0 / (3 * 3), places=4)
        self.assertAlmostEqual(weights[1].item(), 6.0 / (3 * 1), places=4)
        self.assertAlmostEqual(weights[2].item(), 6.0 / (3 * 2), places=4)

        # Empty labels list raises error
        with self.assertRaises(ValueError):
            compute_class_weights([], num_classes=3)

    def test_optimizer_parameter_groups(self):
        """Verify weight decay parameter grouping excludes bias and LayerNorm."""
        trainer = TransformerFineTuner(
            model_name="distilbert-base-uncased",
            max_length=32,
            batch_size=2,
            epochs=1,
        )
        optimizer, scheduler = trainer.get_optimizer_and_scheduler(total_steps=10)

        param_groups = optimizer.param_groups
        self.assertEqual(len(param_groups), 2)
        # Group 0: weight decay > 0
        self.assertEqual(param_groups[0]["weight_decay"], 0.01)
        # Group 1: weight decay == 0
        self.assertEqual(param_groups[1]["weight_decay"], 0.0)

    def test_build_dataloaders(self):
        """Verify DataLoader batching and tensor key output."""
        trainer = TransformerFineTuner(
            model_name="distilbert-base-uncased",
            max_length=32,
            batch_size=2,
        )
        train_loader, val_loader = trainer.build_dataloaders(
            self.mock_train_df, self.mock_val_df, pre_tokenize=True
        )

        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(val_loader, DataLoader)
        self.assertEqual(len(train_loader), 3)  # 6 samples / batch 2 = 3 batches
        self.assertEqual(len(val_loader), 2)    # 3 samples / batch 2 = 2 batches

        batch = next(iter(train_loader))
        self.assertIn("input_ids", batch)
        self.assertIn("attention_mask", batch)
        self.assertIn("label", batch)
        self.assertEqual(batch["input_ids"].shape, (2, 32))
        self.assertEqual(batch["attention_mask"].shape, (2, 32))
        self.assertEqual(batch["label"].shape, (2,))

    def test_train_and_evaluate_step(self):
        """Verify train_epoch and evaluate execution without exceptions on mock data."""
        trainer = TransformerFineTuner(
            model_name="distilbert-base-uncased",
            max_length=32,
            batch_size=2,
            epochs=1,
        )
        train_loader, val_loader = trainer.build_dataloaders(
            self.mock_train_df, self.mock_val_df, pre_tokenize=True
        )
        optimizer, scheduler = trainer.get_optimizer_and_scheduler(total_steps=len(train_loader))
        criterion = nn.CrossEntropyLoss()

        # Run 1 epoch
        avg_loss = trainer.train_epoch(
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            epoch_idx=1,
            log_interval=1,
        )
        self.assertIsInstance(avg_loss, float)
        self.assertGreater(avg_loss, 0.0)

        # Run evaluate
        val_metrics = trainer.evaluate(val_loader, criterion=criterion)
        self.assertIn("loss", val_metrics)
        self.assertIn("accuracy", val_metrics)
        self.assertIn("macro_f1", val_metrics)
        self.assertIn("per_class", val_metrics)
        self.assertIn("confusion_matrix", val_metrics)
        self.assertEqual(val_metrics["sample_count"], 3)
        self.assertIn("bug", val_metrics["per_class"])
        self.assertIn("feature", val_metrics["per_class"])
        self.assertIn("other", val_metrics["per_class"])

    def test_checkpoint_saving_and_loading(self):
        """Verify checkpoint saving creates model files and training metadata."""
        trainer = TransformerFineTuner(
            model_name="distilbert-base-uncased",
            max_length=32,
            batch_size=2,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = trainer.save_checkpoint(tmp_dir, metadata={"epoch": 1, "macro_f1": 0.85})
            self.assertTrue((Path(save_path) / "config.json").exists())
            self.assertTrue((Path(save_path) / "training_metadata.json").exists())

            # Load checkpoint into another instance
            new_trainer = TransformerFineTuner(
                model_name="distilbert-base-uncased",
                max_length=32,
            )
            new_trainer.load_checkpoint(save_path)
            self.assertIsNotNone(new_trainer.model)

    def test_plot_curves_and_save_history(self):
        """Verify plot_training_curves generates image and save_training_history creates valid JSON."""
        trainer = TransformerFineTuner(
            model_name="distilbert-base-uncased",
            max_length=32,
        )
        mock_history = {
            "model_name": "distilbert-base-uncased",
            "epochs": [
                {
                    "epoch": 1,
                    "train_loss": 0.95,
                    "val_loss": 0.82,
                    "val_accuracy": 0.70,
                    "val_macro_f1": 0.68,
                    "val_per_class": {
                        "bug": {"f1_score": 0.75},
                        "feature": {"f1_score": 0.60},
                        "other": {"f1_score": 0.70},
                    },
                },
                {
                    "epoch": 2,
                    "train_loss": 0.65,
                    "val_loss": 0.60,
                    "val_accuracy": 0.81,
                    "val_macro_f1": 0.80,
                    "val_per_class": {
                        "bug": {"f1_score": 0.82},
                        "feature": {"f1_score": 0.76},
                        "other": {"f1_score": 0.82},
                    },
                },
            ],
            "best_epoch": 2,
            "best_val_macro_f1": 0.80,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            fig_path = Path(tmp_dir) / "test_fig.png"
            trainer.plot_training_curves(history=mock_history, save_path=fig_path)
            self.assertTrue(fig_path.exists())
            self.assertGreater(fig_path.stat().st_size, 1000)

            hist_path = Path(tmp_dir) / "test_history.json"
            trainer.save_training_history(history=mock_history, save_path=hist_path)
            self.assertTrue(hist_path.exists())
            with open(hist_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["best_epoch"], 2)


if __name__ == "__main__":
    unittest.main()
