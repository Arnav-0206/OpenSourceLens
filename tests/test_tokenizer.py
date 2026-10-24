"""Unit Tests for Transformer Tokenization and Subword Analysis (Step 6.3)."""

from pathlib import Path
import tempfile
import unittest
import json

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.deep_learning.tokenizer import (
    IssueTokenizer,
    TokenizedIssueDataset,
    SubwordTokenAnalyzer,
)


class TestTransformerTokenizer(unittest.TestCase):
    """Test suite for IssueTokenizer, TokenizedIssueDataset, and SubwordTokenAnalyzer."""

    @classmethod
    def setUpClass(cls):
        """Initialize a shared tokenizer instance."""
        cls.tokenizer = IssueTokenizer(model_name="distilbert-base-uncased", max_length=64)

    def test_tokenizer_properties_and_shapes(self):
        """Verify vocabulary size, tensor shapes, and attention masks."""
        self.assertEqual(self.tokenizer.vocab_size, 30522)
        self.assertEqual(self.tokenizer.max_length, 64)

        # Single string tokenization
        single_enc = self.tokenizer.tokenize_text("Title: Fix memory leak | Description: OOM crash")
        self.assertIn("input_ids", single_enc)
        self.assertIn("attention_mask", single_enc)
        self.assertEqual(single_enc["input_ids"].shape, (1, 64))
        self.assertEqual(single_enc["attention_mask"].shape, (1, 64))

        # Check special tokens: first token is [CLS] (id 101), second token in mask is 1
        self.assertEqual(single_enc["input_ids"][0][0].item(), 101)
        self.assertEqual(single_enc["attention_mask"][0][0].item(), 1)

    def test_batch_tokenization(self):
        """Verify batch tokenization dimensions."""
        texts = [
            "Title: Bug 1 | Description: Short text",
            "Title: Feature 2 | Description: Longer text with more detailed description",
        ]
        batch_enc = self.tokenizer.batch_tokenize(texts, max_length=32)
        self.assertEqual(batch_enc["input_ids"].shape, (2, 32))
        self.assertEqual(batch_enc["attention_mask"].shape, (2, 32))

    def test_token_counting(self):
        """Verify raw subword counting without truncation."""
        text = "Hello world from open source lens"
        count = self.tokenizer.count_tokens(text)
        self.assertGreaterEqual(count, 6)

    def test_tokenized_issue_dataset_and_dataloader(self):
        """Verify TokenizedIssueDataset tensor outputs and DataLoader iteration."""
        texts = [
            "Title: Fix bug #1 | Description: NullPointer",
            "Title: Add button | Description: UI enhancement",
            "Title: Chore | Description: Bump deps",
        ]
        labels = [0, 1, 2]
        ids = ["issue_1", "issue_2", "issue_3"]

        # Test on-the-fly mode
        dataset = TokenizedIssueDataset(texts, labels, self.tokenizer, max_length=32, issue_ids=ids)
        self.assertEqual(len(dataset), 3)

        sample = dataset[0]
        self.assertEqual(sample["input_ids"].shape, torch.Size([32]))
        self.assertEqual(sample["attention_mask"].shape, torch.Size([32]))
        self.assertEqual(sample["label"].item(), 0)
        self.assertEqual(sample["label"].dtype, torch.long)
        self.assertEqual(sample["issue_id"], "issue_1")

        # Test DataLoader batching
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        self.assertEqual(batch["input_ids"].shape, (2, 32))
        self.assertEqual(batch["attention_mask"].shape, (2, 32))
        self.assertEqual(batch["label"].shape, (2,))

        # Test pre_tokenize=True mode
        pre_ds = TokenizedIssueDataset(texts, labels, self.tokenizer, max_length=32, issue_ids=ids, pre_tokenize=True)
        self.assertEqual(len(pre_ds), 3)
        pre_sample = pre_ds[1]
        self.assertEqual(pre_sample["label"].item(), 1)

    def test_save_and_load_local_tokenizer(self):
        """Verify local saving and reloading of tokenizer artifacts."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "offline_tokenizer"
            self.tokenizer.save_pretrained(save_path)

            self.assertTrue(save_path.exists())
            self.assertTrue(
                (save_path / "vocab.txt").exists() or (save_path / "tokenizer.json").exists()
            )
            self.assertTrue((save_path / "tokenizer_config.json").exists())

            # Load from local directory
            loaded_tok = IssueTokenizer(max_length=48, local_dir=save_path)
            self.assertEqual(loaded_tok.vocab_size, self.tokenizer.vocab_size)
            enc = loaded_tok.tokenize_text("Test issue")
            self.assertEqual(enc["input_ids"].shape, (1, 48))

    def test_subword_token_analyzer(self):
        """Verify subword token length analysis, plotting, and JSON serialization."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fig_path = Path(tmp_dir) / "fig18.png"
            json_path = Path(tmp_dir) / "analysis.json"

            df = pd.DataFrame(
                {
                    "text": [
                        "Title: Short issue | Description: Simple error.",
                        "Title: Long issue | Description: " + "word " * 100,
                    ],
                    "target_label": ["bug", "feature"],
                    "label": [0, 1],
                    "issue_id": ["id_1", "id_2"],
                }
            )

            analyzer = SubwordTokenAnalyzer(self.tokenizer)
            split_res = analyzer.analyze_split(df, "test_split")

            self.assertIn("truncation_rate_pct", split_res)
            self.assertIn("token_retention_pct", split_res)
            self.assertIn("token_length_stats", split_res)
            self.assertEqual(split_res["sample_count"], 2)

            domain_audit = analyzer.audit_domain_entities()
            self.assertGreater(len(domain_audit), 0)
            self.assertIn("subwords", domain_audit[0])

            # Test plot
            analysis_dict = {"test_split": split_res}
            saved_fig = analyzer.plot_token_distributions(analysis_dict, fig_path)
            self.assertTrue(saved_fig.exists())
            self.assertGreater(saved_fig.stat().st_size, 1000)

            # Test JSON save
            saved_json = analyzer.save_analysis(analysis_dict, domain_audit, json_path)
            self.assertTrue(saved_json.exists())

            with open(saved_json, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data["metadata"]["step"], "6.3")
            self.assertIn("split_token_statistics", data)
            self.assertIn("domain_token_segmentation_audit", data)


if __name__ == "__main__":
    unittest.main()
