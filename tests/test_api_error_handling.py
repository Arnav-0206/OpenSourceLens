"""Unit and integration tests for OpenSourceLens API Error Handling (Phase 9, Step 9.11).

Tests standardized error envelopes, custom domain exceptions, validation error
formatting, database error masking, and unhandled exception catching.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from backend.app.core.error_handlers import register_error_handlers
from backend.app.core.exceptions import (
    DatabaseServiceError,
    EntityNotFoundException,
    InvalidParameterException,
    IssueNotFoundException,
    ModelInferenceError,
    ModelNotFoundException,
    OpenSourceLensException,
    RepositoryNotFoundException,
)
from backend.app.database.session import get_db
from backend.app.main import app


class TestApiErrorHandling(unittest.TestCase):
    """Test suite for structured error responses and exception handlers."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_404_unmapped_route(self):
        """Verify 404 on non-existent route returns standard ErrorResponse envelope."""
        response = self.client.get("/api/v1/nonexistent/endpoint/route")
        self.assertEqual(response.status_code, 404)
        data = response.json()

        self.assertIn("detail", data)
        self.assertEqual(data["error_code"], "RESOURCE_NOT_FOUND")
        self.assertEqual(data["status_code"], 404)
        self.assertEqual(data["path"], "/api/v1/nonexistent/endpoint/route")
        self.assertIn("timestamp", data)

    def test_404_repository_not_found(self):
        """Verify 404 for unknown repository returns structured envelope."""
        response = self.client.get("/api/v1/repos/fake_org/fake_repo")
        self.assertEqual(response.status_code, 404)
        data = response.json()

        self.assertIn("detail", data)
        self.assertIn("not found", data["detail"].lower())
        self.assertEqual(data["error_code"], "RESOURCE_NOT_FOUND")
        self.assertEqual(data["status_code"], 404)
        self.assertEqual(data["path"], "/api/v1/repos/fake_org/fake_repo")

    def test_404_issue_not_found(self):
        """Verify 404 for unknown issue returns structured envelope."""
        response = self.client.get("/api/v1/repos/microsoft/vscode/issues/999999999")
        self.assertEqual(response.status_code, 404)
        data = response.json()

        self.assertIn("detail", data)
        self.assertIn("not found", data["detail"].lower())
        self.assertEqual(data["error_code"], "RESOURCE_NOT_FOUND")
        self.assertEqual(data["status_code"], 404)

    def test_404_model_not_found(self):
        """Verify 404 for unknown model returns structured envelope."""
        response = self.client.get("/api/v1/models/unsupported_deep_network")
        self.assertEqual(response.status_code, 404)
        data = response.json()

        self.assertIn("detail", data)
        self.assertIn("not found", data["detail"].lower())
        self.assertEqual(data["error_code"], "RESOURCE_NOT_FOUND")
        self.assertEqual(data["status_code"], 404)

    def test_422_validation_error_missing_body(self):
        """Verify 422 for missing required body field formats ValidationErrorResponse."""
        response = self.client.post("/api/v1/classify", json={})
        self.assertEqual(response.status_code, 422)
        data = response.json()

        self.assertIn("detail", data)
        self.assertEqual(data["error_code"], "VALIDATION_ERROR")
        self.assertEqual(data["status_code"], 422)
        self.assertEqual(data["path"], "/api/v1/classify")
        self.assertIn("timestamp", data)
        self.assertIn("errors", data)
        self.assertIsInstance(data["errors"], list)
        self.assertGreaterEqual(len(data["errors"]), 1)

        first_err = data["errors"][0]
        self.assertIn("field", first_err)
        self.assertIn("message", first_err)
        self.assertIn("type", first_err)
        self.assertIn("body.title", first_err["field"])

    def test_422_validation_error_invalid_query_parameter(self):
        """Verify 422 for negative or invalid integer query parameter."""
        response = self.client.get("/api/v1/repos/microsoft/vscode/issues?limit=-1")
        self.assertEqual(response.status_code, 422)
        data = response.json()

        self.assertEqual(data["error_code"], "VALIDATION_ERROR")
        self.assertEqual(data["status_code"], 422)
        self.assertIn("errors", data)
        self.assertTrue(any("limit" in err["field"] for err in data["errors"]))

    def test_domain_exceptions_handling(self):
        """Verify domain exceptions trigger proper status codes and error payloads."""
        test_app = FastAPI()
        register_error_handlers(test_app)

        @test_app.get("/test/domain/repo-not-found")
        def route_repo_not_found():
            raise RepositoryNotFoundException("test/nonexistent")

        @test_app.get("/test/domain/issue-not-found")
        def route_issue_not_found():
            raise IssueNotFoundException("test/nonexistent", 42)

        @test_app.get("/test/domain/model-not-found")
        def route_model_not_found():
            raise ModelNotFoundException("bert_unknown")

        @test_app.get("/test/domain/invalid-param")
        def route_invalid_param():
            raise InvalidParameterException("Page index cannot be negative.", parameter="page")

        @test_app.get("/test/domain/model-inference-error")
        def route_model_error():
            raise ModelInferenceError("Embedding vector generation timed out.", model_id="distilbert")

        @test_app.get("/test/domain/db-error")
        def route_db_service_error():
            raise DatabaseServiceError("Connection pool exhausted.")

        client = TestClient(test_app)

        # 1. RepositoryNotFoundException
        res1 = client.get("/test/domain/repo-not-found")
        self.assertEqual(res1.status_code, 404)
        self.assertEqual(res1.json()["error_code"], "RESOURCE_NOT_FOUND")
        self.assertIn("test/nonexistent", res1.json()["detail"])
        self.assertEqual(res1.json()["details"]["entity_id"], "test/nonexistent")

        # 2. IssueNotFoundException
        res2 = client.get("/test/domain/issue-not-found")
        self.assertEqual(res2.status_code, 404)
        self.assertEqual(res2.json()["error_code"], "RESOURCE_NOT_FOUND")
        self.assertIn("#42", res2.json()["detail"])

        # 3. ModelNotFoundException
        res3 = client.get("/test/domain/model-not-found")
        self.assertEqual(res3.status_code, 404)
        self.assertEqual(res3.json()["error_code"], "RESOURCE_NOT_FOUND")

        # 4. InvalidParameterException
        res4 = client.get("/test/domain/invalid-param")
        self.assertEqual(res4.status_code, 400)
        self.assertEqual(res4.json()["error_code"], "INVALID_PARAMETER")
        self.assertEqual(res4.json()["details"]["parameter"], "page")

        # 5. ModelInferenceError
        res5 = client.get("/test/domain/model-inference-error")
        self.assertEqual(res5.status_code, 500)
        self.assertEqual(res5.json()["error_code"], "MODEL_INFERENCE_ERROR")
        self.assertEqual(res5.json()["details"]["model_id"], "distilbert")

        # 6. DatabaseServiceError
        res6 = client.get("/test/domain/db-error")
        self.assertEqual(res6.status_code, 503)
        self.assertEqual(res6.json()["error_code"], "DATABASE_ERROR")

    def test_sqlalchemy_error_handler_masks_details(self):
        """Verify SQLAlchemy database exceptions return clean 503 without leaking credentials or SQL."""
        test_app = FastAPI()
        register_error_handlers(test_app)

        @test_app.get("/test/trigger-sql-error")
        def route_sql_error():
            # Raise simulated OperationalError with SQL statement
            raise OperationalError("SELECT * FROM secret_credentials WHERE user='admin'", {}, Exception("DB down"))

        client = TestClient(test_app, raise_server_exceptions=False)
        response = client.get("/test/trigger-sql-error")

        self.assertEqual(response.status_code, 503)
        data = response.json()
        self.assertEqual(data["error_code"], "DATABASE_ERROR")
        self.assertEqual(data["status_code"], 503)
        self.assertEqual(data["path"], "/test/trigger-sql-error")
        # Ensure raw query is NOT leaked to caller
        self.assertNotIn("secret_credentials", data["detail"])
        self.assertIn("database operation failed", data["detail"].lower())

    def test_unhandled_500_exception_handler(self):
        """Verify unhandled general Exception returns sanitized 500 JSON without traceback."""
        test_app = FastAPI()
        register_error_handlers(test_app)

        @test_app.get("/test/trigger-panic")
        def route_panic():
            raise ZeroDivisionError("division by zero in complex calculation")

        client = TestClient(test_app, raise_server_exceptions=False)
        response = client.get("/test/trigger-panic")

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertEqual(data["error_code"], "INTERNAL_SERVER_ERROR")
        self.assertEqual(data["status_code"], 500)
        self.assertEqual(data["path"], "/test/trigger-panic")
        self.assertIn("internal server error", data["detail"].lower())
        # Ensure stack trace / specific division error is sanitized
        self.assertNotIn("ZeroDivisionError", data["detail"])


if __name__ == "__main__":
    unittest.main()
