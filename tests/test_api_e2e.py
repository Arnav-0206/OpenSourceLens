"""End-to-End (E2E) Integration & Performance Test Suite for OpenSourceLens (Phase 9, Step 9.12).

Simulates complete user journeys through the FastAPI backend:
1. System Discovery (root, health, openapi docs)
2. Repository & Issue Exploration (rankings, filters, single issue details)
3. Analytical Drilldowns (global overview, comparative matrix, monthly trends)
4. Interactive ML Workflows (prewarming, DistilBERT classification, semantic search, resolution prediction)
5. Performance Latency Benchmarks (< 200ms DB responses, < 50ms cached similarity lookups)
6. Concurrent Request Thread-Safety Verification
"""

from __future__ import annotations

import concurrent.futures
import time
import unittest

from fastapi.testclient import TestClient

from backend.app.main import app


class TestApiEndToEndJourney(unittest.TestCase):
    """Full End-to-End user workflow and benchmark test suite."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_e2e_system_discovery_flow(self):
        """User journey: verify root discovery, system health probe, and OpenAPI contract."""
        # 1. Discover root
        root_res = self.client.get("/")
        self.assertEqual(root_res.status_code, 200)
        root_data = root_res.json()
        self.assertEqual(root_data["title"], "OpenSourceLens API")
        self.assertIn("microsoft/vscode", root_data["supported_repositories"])
        self.assertIn("pytorch/pytorch", root_data["supported_repositories"])
        self.assertIn("kubernetes/kubernetes", root_data["supported_repositories"])
        self.assertIn("huggingface/transformers", root_data["supported_repositories"])

        # 2. Health check
        health_res = self.client.get("/health")
        self.assertEqual(health_res.status_code, 200)
        health_data = health_res.json()
        self.assertEqual(health_data["status"], "healthy")
        self.assertEqual(health_data["database"]["status"], "connected")
        self.assertGreaterEqual(health_data["database"]["tables_count"], 6)

        # 3. OpenAPI docs contract
        docs_res = self.client.get("/docs")
        self.assertEqual(docs_res.status_code, 200)
        self.assertIn("text/html", docs_res.headers["content-type"])

        openapi_res = self.client.get("/openapi.json")
        self.assertEqual(openapi_res.status_code, 200)
        openapi_data = openapi_res.json()
        paths = openapi_data["paths"]
        self.assertIn("/api/v1/repos", paths)
        self.assertIn("/api/v1/analytics/overview", paths)
        self.assertIn("/api/v1/classify", paths)
        self.assertIn("/api/v1/related", paths)
        self.assertIn("/api/v1/resolution/predict", paths)
        self.assertIn("/api/v1/models", paths)

    def test_e2e_repository_and_issue_exploration_flow(self):
        """User journey: explore repositories, filter issues, inspect issue detail with labels."""
        # 1. Fetch repositories ordered by popularity
        repos_res = self.client.get("/api/v1/repos")
        self.assertEqual(repos_res.status_code, 200)
        repos = repos_res.json()
        self.assertEqual(len(repos), 4)

        # Star counts should be sorted descending
        stars = [r["stars_count"] for r in repos]
        self.assertEqual(stars, sorted(stars, reverse=True))

        top_repo = repos[0]
        self.assertEqual(top_repo["repo_name"], "microsoft/vscode")

        # 2. Drill down into top repository details
        detail_res = self.client.get("/api/v1/repos/microsoft/vscode")
        self.assertEqual(detail_res.status_code, 200)
        detail = detail_res.json()
        self.assertEqual(detail["name"], "vscode")
        self.assertEqual(detail["owner"], "microsoft")
        self.assertGreater(detail["total_analyzed_issues"], 500)
        self.assertGreater(detail["closed_issues_count"], 200)
        self.assertGreater(detail["median_resolution_hours"], 0)
        self.assertIn("< 1 Day", detail["speed_tier_distribution"])

        # 3. Search filtered issues: closed bugs
        issues_res = self.client.get("/api/v1/repos/microsoft/vscode/issues?state=closed&category=bug&limit=10")
        self.assertEqual(issues_res.status_code, 200)
        issues_page = issues_res.json()
        self.assertGreaterEqual(issues_page["total"], 1)
        self.assertLessEqual(len(issues_page["items"]), 10)

        for item in issues_page["items"]:
            self.assertEqual(item["state"], "closed")
            self.assertEqual(item["canonical_label"], "bug")

        # 4. Fetch full single issue detail
        sample_number = issues_page["items"][0]["issue_number"]
        single_issue_res = self.client.get(f"/api/v1/repos/microsoft/vscode/issues/{sample_number}")
        self.assertEqual(single_issue_res.status_code, 200)
        single_issue = single_issue_res.json()
        self.assertEqual(single_issue["issue_number"], sample_number)
        self.assertIn("body", single_issue)
        self.assertIn("clean_title", single_issue)
        self.assertIn("labels", single_issue)
        self.assertIsInstance(single_issue["labels"], list)

    def test_e2e_analytics_and_comparisons_flow(self):
        """User journey: view global macro analytics, cross-repository benchmark, and monthly trends."""
        # 1. Global overview
        overview_res = self.client.get("/api/v1/analytics/overview")
        self.assertEqual(overview_res.status_code, 200)
        overview = overview_res.json()
        self.assertEqual(overview["total_issues"], 2328)
        self.assertEqual(overview["total_repositories"], 4)
        self.assertEqual(overview["fastest_repository"], "microsoft/vscode")
        self.assertEqual(overview["slowest_repository"], "kubernetes/kubernetes")

        # 2. Cross-repository comparative matrix
        comp_res = self.client.get("/api/v1/analytics/comparison")
        self.assertEqual(comp_res.status_code, 200)
        matrix = comp_res.json()
        self.assertEqual(matrix["total_repositories"], 4)
        self.assertEqual(len(matrix["repositories"]), 4)

        for item in matrix["repositories"]:
            self.assertGreater(item["total_issues"], 0)
            self.assertGreater(item["median_resolution_hours"], 0)
            self.assertGreaterEqual(item["rapid_closure_percentage"], 0.0)

        # 3. Monthly velocity trends
        trends_res = self.client.get("/api/v1/analytics/microsoft/vscode/trends")
        self.assertEqual(trends_res.status_code, 200)
        trends = trends_res.json()
        self.assertEqual(trends["repo_name"], "microsoft/vscode")
        self.assertGreater(len(trends["trends"]), 0)

        # Verify chronological ordering
        periods = [p["period"] for p in trends["trends"]]
        self.assertEqual(periods, sorted(periods))

    def test_e2e_interactive_ml_workflows(self):
        """User journey: model introspection, issue classification, semantic search, and velocity prediction."""
        # 1. Inspect model registry & prewarm
        models_res = self.client.get("/api/v1/models")
        self.assertEqual(models_res.status_code, 200)
        models_list = models_res.json()["models"]
        model_ids = [m["model_id"] for m in models_list]
        self.assertIn("distilbert", model_ids)
        self.assertIn("sentence_transformer", model_ids)
        self.assertIn("resolution_velocity", model_ids)

        prewarm_res = self.client.post("/api/v1/models/prewarm")
        self.assertEqual(prewarm_res.status_code, 200)
        prewarm_data = prewarm_res.json()
        self.assertEqual(prewarm_data["distilbert"], "prewarmed")
        self.assertEqual(prewarm_data["sentence_transformer"], "prewarmed")
        self.assertEqual(prewarm_data["resolution_velocity"], "prewarmed")

        # 2. User writes a new bug issue
        test_issue = {
            "title": "Crash on window resize when using multi-monitor setup",
            "body": "Whenever dragging the editor window from the primary 4K display to the secondary 1080p monitor, the renderer crashes with code 0xC0000005.",
            "repo_name": "microsoft/vscode",
            "model_type": "distilbert",
        }

        # Step A: Classify category
        classify_res = self.client.post("/api/v1/classify", json=test_issue)
        self.assertEqual(classify_res.status_code, 200)
        classify_data = classify_res.json()
        self.assertIn(classify_data["predicted_label"], ["bug", "feature", "other"])
        self.assertGreater(classify_data["confidence_score"], 0.3)
        self.assertIn("bug", classify_data["probabilities"])
        self.assertIn("feature", classify_data["probabilities"])
        self.assertIn("other", classify_data["probabilities"])
        self.assertAlmostEqual(sum(classify_data["probabilities"].values()), 1.0, delta=0.02)
        self.assertIn("distilbert", classify_data["model_used"].lower())

        # Step B: Semantic search for duplicates / related issues
        related_search_res = self.client.post(
            "/api/v1/related",
            json={
                "title": test_issue["title"],
                "body": test_issue["body"],
                "top_k": 5,
            },
        )
        self.assertEqual(related_search_res.status_code, 200)
        search_data = related_search_res.json()
        self.assertEqual(len(search_data["matches"]), 5)

        # Similarities must be descending and bounded in [0, 1]
        sims = [m["similarity_score"] for m in search_data["matches"]]
        self.assertEqual(sims, sorted(sims, reverse=True))
        self.assertTrue(all(0.0 <= s <= 1.0 for s in sims))

        # Step C: Resolution velocity prediction
        resolution_res = self.client.post(
            "/api/v1/resolution/predict",
            json={
                "title": test_issue["title"],
                "body": test_issue["body"],
                "repo_name": "microsoft/vscode",
            },
        )
        self.assertEqual(resolution_res.status_code, 200)
        res_data = resolution_res.json()
        self.assertIn(res_data["predicted_speed_tier"], ["< 1 Day", "1-7 Days", "7-30 Days", "30+ Days"])
        self.assertIn("lower_bound_tier", res_data)
        self.assertIn("upper_bound_tier", res_data)
        self.assertIn("advisory", res_data["reliability_notice"].lower())

        # Step D: Lookup cached precomputed related issues for an existing issue
        first_match_id = search_data["matches"][0]["issue_id"]
        cached_res = self.client.get(f"/api/v1/related/{first_match_id}?top_k=3")
        self.assertEqual(cached_res.status_code, 200)
        cached_data = cached_res.json()
        self.assertGreaterEqual(len(cached_data["matches"]), 1)

    def test_e2e_performance_and_latency_benchmarks(self):
        """Benchmark: assert response latencies conform to interactive web requirements."""
        # 1. Fast database queries: GET /api/v1/repos should be < 200ms
        t0 = time.perf_counter()
        res_repos = self.client.get("/api/v1/repos")
        elapsed_repos = (time.perf_counter() - t0) * 1000
        self.assertEqual(res_repos.status_code, 200)
        self.assertLess(elapsed_repos, 200.0, f"GET /repos took {elapsed_repos:.2f}ms (expected < 200ms)")

        # 2. Analytical overview computation should be < 250ms
        t0 = time.perf_counter()
        res_overview = self.client.get("/api/v1/analytics/overview")
        elapsed_overview = (time.perf_counter() - t0) * 1000
        self.assertEqual(res_overview.status_code, 200)
        self.assertLess(elapsed_overview, 250.0, f"GET /analytics/overview took {elapsed_overview:.2f}ms (expected < 250ms)")

        # 3. Cached related issue lookup should be sub-50ms
        t0 = time.perf_counter()
        res_cached = self.client.get("/api/v1/related/5903612596?top_k=5")
        elapsed_cached = (time.perf_counter() - t0) * 1000
        self.assertEqual(res_cached.status_code, 200)
        self.assertLess(elapsed_cached, 50.0, f"Cached related lookup took {elapsed_cached:.2f}ms (expected < 50ms)")

    def test_e2e_concurrency_stability(self):
        """Thread-safety: verify concurrent requests do not corrupt DB sessions or model state."""
        urls = [
            "/api/v1/repos",
            "/api/v1/analytics/overview",
            "/api/v1/repos/microsoft/vscode",
            "/api/v1/analytics/comparison",
            "/api/v1/models",
            "/health",
            "/api/v1/repos/pytorch/pytorch",
            "/api/v1/analytics/pytorch/pytorch/trends",
            "/api/v1/repos/kubernetes/kubernetes",
            "/api/v1/repos/huggingface/transformers",
        ]

        def fetch_url(url: str):
            client = TestClient(app)
            return client.get(url)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(fetch_url, u) for u in urls]
            results = [f.result() for f in futures]

        for res in results:
            self.assertEqual(res.status_code, 200)


if __name__ == "__main__":
    unittest.main()
