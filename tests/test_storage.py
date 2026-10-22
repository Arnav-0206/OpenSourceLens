"""Unit tests for RawDataStorage module."""

import unittest
import tempfile
import shutil
from pathlib import Path

from ml.data_collection.storage import RawDataStorage


class TestRawDataStorage(unittest.TestCase):
    """Test suite for atomic storage, directory layout, Unicode safety, and loading."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.storage = RawDataStorage(base_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_ensure_directories_created(self):
        """Test that all required category directories are created."""
        base = Path(self.temp_dir)
        for cat in ["repositories", "issues", "comments", "events"]:
            self.assertTrue((base / cat).is_dir())

    def test_sanitize_repo_identifier(self):
        """Test conversion of various repo string formats into clean file prefixes."""
        cases = [
            ("microsoft/vscode", "microsoft__vscode"),
            ("pytorch/pytorch/", "pytorch__pytorch"),
            ("https://github.com/kubernetes/kubernetes", "kubernetes__kubernetes"),
            ("https://github.com/huggingface/transformers/", "huggingface__transformers"),
        ]
        for input_val, expected in cases:
            with self.subTest(input_val=input_val):
                self.assertEqual(RawDataStorage.sanitize_repo_identifier(input_val), expected)

    def test_save_and_load_repository(self):
        """Test atomic saving and loading of repository raw data."""
        mock_data = {
            "name": "vscode",
            "full_name": "microsoft/vscode",
            "stargazers_count": 190000,
            "description": "Visual Studio Code editor",
        }

        saved_path = self.storage.save_repository("microsoft/vscode", mock_data)
        self.assertTrue(Path(saved_path).is_file())
        self.assertTrue(saved_path.endswith("microsoft__vscode.json"))

        loaded = self.storage.load_raw(saved_path)
        self.assertEqual(loaded, mock_data)

    def test_save_issues_with_unicode_and_emojis(self):
        """Test atomic saving of issues with Unicode symbols and emojis."""
        mock_issues = [
            {
                "id": 1,
                "number": 101,
                "title": "Bug in unicode parser ¯\\_(ツ)_/¯",
                "body": "Special characters: 日本語, é, ñ, 🚀, 👍",
                "labels": [{"name": "¯\\_(ツ)_/¯"}, {"name": "kind/bug"}],
            }
        ]

        saved_path = self.storage.save_issues("kubernetes/kubernetes", mock_issues, batch_id="batch_01")
        self.assertTrue(Path(saved_path).is_file())
        self.assertIn("kubernetes__kubernetes__issues_batch_01.json", saved_path)

        loaded = self.storage.load_raw(saved_path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["title"], "Bug in unicode parser ¯\\_(ツ)_/¯")
        self.assertIn("🚀", loaded[0]["body"])

    def test_save_comments_and_events(self):
        """Test saving raw comment and event payloads."""
        mock_comments = [{"id": 1, "body": "Comment 1"}]
        mock_events = [{"id": 2, "event": "closed"}]

        c_path = self.storage.save_comments("pytorch/pytorch", 123, mock_comments)
        e_path = self.storage.save_events("pytorch/pytorch", 123, mock_events)

        self.assertTrue(Path(c_path).is_file())
        self.assertTrue(Path(e_path).is_file())
        self.assertIn("pytorch__pytorch__issue_123__comments.json", c_path)
        self.assertIn("pytorch__pytorch__issue_123__events.json", e_path)

    def test_list_files_and_get_storage_summary(self):
        """Test listing files and inventory summary reporting."""
        self.storage.save_repository("microsoft/vscode", {"test": 1})
        self.storage.save_repository("pytorch/pytorch", {"test": 2})
        self.storage.save_issues("microsoft/vscode", [{"test": 3}], batch_id="b1")

        repo_files = self.storage.list_files("repositories")
        self.assertEqual(len(repo_files), 2)

        vscode_repos = self.storage.list_files("repositories", repo_identifier="microsoft/vscode")
        self.assertEqual(len(vscode_repos), 1)

        summary = self.storage.get_storage_summary()
        self.assertEqual(summary["total_files"], 3)
        self.assertGreater(summary["total_bytes"], 0)
        self.assertEqual(summary["categories"]["repositories"]["file_count"], 2)
        self.assertEqual(summary["categories"]["issues"]["file_count"], 1)


if __name__ == "__main__":
    unittest.main()
