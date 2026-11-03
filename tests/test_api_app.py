"""Integration tests for FastAPI application, healthchecks, and discovery endpoints."""

import unittest

from fastapi.testclient import TestClient

from backend.app.core.config import settings
from backend.app.main import app


class TestFastAPIApp(unittest.TestCase):
    """Test suite for FastAPI core application and discovery routes."""

    @classmethod
    def setUpClass(cls):
        """Initialize FastAPI TestClient."""
        cls.client = TestClient(app)

    def test_root_endpoint(self):
        """Verify GET / returns project metadata and supported repositories."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["title"], settings.PROJECT_NAME)
        self.assertEqual(data["version"], settings.VERSION)
        self.assertEqual(data["docs_url"], "/docs")
        self.assertEqual(data["api_v1_url"], "/api/v1")
        self.assertIsInstance(data["supported_repositories"], list)
        self.assertGreaterEqual(len(data["supported_repositories"]), 4)
        self.assertIn("microsoft/vscode", data["supported_repositories"])

    def test_health_check_endpoint(self):
        """Verify GET /health reports healthy status and active database dialect."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["version"], settings.VERSION)
        self.assertIn("timestamp", data)

        db_health = data["database"]
        self.assertEqual(db_health["status"], "connected")
        self.assertIn(db_health["dialect"], ["sqlite", "postgresql"])
        self.assertGreaterEqual(db_health["tables_count"], 6)
        self.assertGreater(db_health["latency_ms"], 0.0)
        self.assertIsNone(db_health["error"])

    def test_api_v1_ping(self):
        """Verify GET /api/v1/ping is mounted and responsive."""
        response = self.client.get("/api/v1/ping")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("active", data["message"])

    def test_openapi_documentation_endpoints(self):
        """Verify OpenAPI spec JSON and Swagger UI endpoints are operational."""
        docs_res = self.client.get("/docs")
        self.assertEqual(docs_res.status_code, 200)
        self.assertIn("text/html", docs_res.headers["content-type"])

        openapi_res = self.client.get("/openapi.json")
        self.assertEqual(openapi_res.status_code, 200)
        spec = openapi_res.json()
        self.assertEqual(spec["info"]["title"], settings.PROJECT_NAME)
        self.assertIn("/health", spec["paths"])
        self.assertIn("/api/v1/ping", spec["paths"])

    def test_cors_middleware_headers(self):
        """Verify CORS preflight and origin headers are properly handled."""
        origin = "http://localhost:3000"
        response = self.client.get("/", headers={"Origin": origin})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), origin)


if __name__ == "__main__":
    unittest.main()
