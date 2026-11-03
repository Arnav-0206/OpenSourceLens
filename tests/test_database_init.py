"""Unit tests for database initialization, healthcheck, and schema provisioning."""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from sqlalchemy import create_engine

from backend.app.database.init_db import (
    check_connection,
    create_database_tables,
    drop_database_tables,
    get_table_status,
    main,
)


class TestDatabaseInit(unittest.TestCase):
    """Test suite for database setup and provisioning utilities."""

    def setUp(self):
        """Create a temporary SQLite database for isolated test execution."""
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = os.path.join(self.temp_dir.name, "test_init.db")
        self.db_url = f"sqlite:///{self.db_path}"
        self.engine = create_engine(self.db_url)

    def tearDown(self):
        """Clean up temporary files and engine."""
        self.engine.dispose()
        self.temp_dir.cleanup()

    def test_check_connection_success(self):
        """Verify check_connection returns successful status for a valid engine."""
        res = check_connection(self.engine)
        self.assertEqual(res["status"], "connected")
        self.assertEqual(res["dialect"], "sqlite")
        self.assertIsNone(res["error"])
        self.assertGreater(res["latency_ms"], 0.0)

    def test_check_connection_failure(self):
        """Verify check_connection gracefully catches and reports connection errors."""
        unreachable_engine = create_engine(
            "sqlite:////invalid_dir_path_xyz_123/database.db"
        )
        res = check_connection(unreachable_engine)
        self.assertEqual(res["status"], "unreachable")
        self.assertIsNotNone(res["error"])

    def test_create_and_inspect_tables(self):
        """Verify create_database_tables creates all 6 models and get_table_status validates completeness."""
        # Prior to creation, status should report incomplete
        status_before = get_table_status(self.engine)
        self.assertFalse(status_before["is_complete"])
        self.assertEqual(len(status_before["existing_tables"]), 0)

        # Create tables
        create_res = create_database_tables(self.engine)
        self.assertEqual(create_res["status"], "success")
        self.assertEqual(create_res["table_count"], 6)

        # Status after creation
        status_after = get_table_status(self.engine)
        self.assertTrue(status_after["is_complete"])
        self.assertEqual(len(status_after["existing_tables"]), 6)

        expected_tables = {
            "repositories",
            "issues",
            "issue_labels",
            "repository_analytics",
            "predictions",
            "related_issues_cache",
        }
        self.assertEqual(set(status_after["existing_tables"]), expected_tables)

        # Verify column and index inspection
        issue_meta = status_after["table_details"]["issues"]
        self.assertEqual(issue_meta["columns_count"], 39)
        self.assertGreaterEqual(issue_meta["indexes_count"], 4)

    def test_drop_database_tables(self):
        """Verify safety guard on drop_database_tables and successful deletion when confirmed."""
        create_database_tables(self.engine)
        self.assertTrue(get_table_status(self.engine)["is_complete"])

        # Attempt drop without confirmation must raise ValueError
        with self.assertRaises(ValueError):
            drop_database_tables(self.engine, confirm=False)

        # Attempt drop with confirmation
        drop_res = drop_database_tables(self.engine, confirm=True)
        self.assertEqual(drop_res["status"], "dropped")
        self.assertEqual(len(drop_res["remaining_tables"]), 0)

    def test_cli_main_check_and_status(self):
        """Verify CLI main entrypoint executes check, create, and status successfully."""
        with patch.object(
            sys,
            "argv",
            ["init_db", "--url", self.db_url, "--create", "--status", "--check"],
        ):
            exit_code = main()
            self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
