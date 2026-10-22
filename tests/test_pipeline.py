"""Unit tests for CollectionPipeline orchestrator."""

import unittest
from unittest.mock import MagicMock, patch
from ml.data_collection.pipeline import CollectionPipeline, DEFAULT_CANDIDATE_REPOSITORIES


class TestCollectionPipeline(unittest.TestCase):
    """Test suite for CollectionPipeline."""

    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_storage = MagicMock()
        self.pipeline = CollectionPipeline(
            client=self.mock_client,
            storage=self.mock_storage,
            issues_per_repo=20,
            sample_details_count=2,
        )

    def test_init_defaults(self):
        pipeline = CollectionPipeline(
            client=self.mock_client,
            storage=self.mock_storage,
        )
        self.assertEqual(pipeline.issues_per_repo, 600)
        self.assertEqual(pipeline.sample_details_count, 10)
        self.assertEqual(DEFAULT_CANDIDATE_REPOSITORIES, [
            "microsoft/vscode",
            "pytorch/pytorch",
            "kubernetes/kubernetes",
            "huggingface/transformers",
        ])

    def test_collect_repository_metadata(self):
        fake_meta = {"name": "vscode", "stars": 150000, "language": "TypeScript"}
        self.pipeline.repo_collector.collect = MagicMock(return_value={"metadata": fake_meta, "raw": {"id": 123}})
        self.mock_storage.save_repository.return_value = "data/raw/repositories/microsoft_vscode.json"

        result = self.pipeline.collect_repository_metadata("microsoft/vscode")
        self.assertEqual(result, fake_meta)
        self.pipeline.repo_collector.collect.assert_called_once_with("microsoft/vscode")
        self.mock_storage.save_repository.assert_called_once_with("microsoft/vscode", {"id": 123})

    def test_collect_issues(self):
        # 20 issues requested -> 10 closed, 10 open
        fake_closed = [
            {"raw": {"id": 1, "number": 101}, "issue": {"issue_number": 101}},
            {"raw": {"id": 2, "number": 102}, "issue": {"issue_number": 102}},
        ]
        fake_open = [
            {"raw": {"id": 3, "number": 103}, "issue": {"issue_number": 103}},
            {"raw": {"id": 4, "number": 104}, "issue": {"issue_number": 104}},
        ]

        def fake_stream(repo, state, **kwargs):
            if state == "closed":
                return iter(fake_closed)
            return iter(fake_open)

        self.pipeline.issue_collector.collect_issues_stream = MagicMock(side_effect=fake_stream)
        self.mock_storage.save_issues.return_value = "data/raw/issues/mock_issues.json"

        numbers = self.pipeline.collect_issues("microsoft/vscode", target_count=20)
        self.assertEqual(numbers["closed"], [101, 102])
        self.assertEqual(numbers["open"], [103, 104])
        self.assertEqual(self.mock_storage.save_issues.call_count, 2)

    def test_collect_discussions_and_events(self):
        self.pipeline.comment_collector.collect_comments_page = MagicMock(return_value=[
            {"raw": {"id": 999, "body": "test comment"}}
        ])
        self.pipeline.event_collector.collect_events_page = MagicMock(return_value=[
            {"raw": {"id": 888, "event": "closed"}}
        ])

        counts = self.pipeline.collect_discussions_and_events("microsoft/vscode", [101, 102])
        self.assertEqual(counts["comments"], 2)
        self.assertEqual(counts["events"], 2)
        self.assertEqual(self.mock_storage.save_comments.call_count, 2)
        self.assertEqual(self.mock_storage.save_events.call_count, 2)

    def test_run_success(self):
        self.mock_client.get_quota_status.side_effect = [
            {"remaining": 5000, "limit": 5000, "reset_epoch": 1234567890},
            {"remaining": 4980, "limit": 5000, "reset_epoch": 1234567890},
        ]
        self.mock_storage.get_storage_summary.return_value = {
            "total_files": 12,
            "total_size_mb": 4.5,
            "by_category": {},
        }

        self.pipeline.collect_repository_metadata = MagicMock(return_value={"stars": 100, "language": "Python"})
        self.pipeline.collect_issues = MagicMock(return_value={"closed": [1, 2], "open": [3, 4]})
        self.pipeline.collect_discussions_and_events = MagicMock(return_value={"comments": 2, "events": 2})

        summary = self.pipeline.run(repositories=["test/repo1", "test/repo2"])

        self.assertEqual(summary["total_issues_collected"], 8)  # 4 issues per repo * 2 repos
        self.assertEqual(summary["total_comments_collected"], 4)
        self.assertEqual(summary["total_events_collected"], 4)
        self.assertEqual(summary["quota_used"], 20)
        self.assertIn("test/repo1", summary["repositories"])
        self.assertIn("test/repo2", summary["repositories"])


if __name__ == "__main__":
    unittest.main()
