# OpenSourceLens — Complete HTTP API Reference

OpenSourceLens serves REST JSON endpoints via FastAPI ([`backend/app/main.py`](../backend/app/main.py), [`backend/app/api/v1/api.py`](../backend/app/api/v1/api.py)). Interactive Swagger and ReDoc specs are automatically available at `/docs` and `/redoc` (OpenAPI schema at `/openapi.json`).

> **Methodology Version:** The current analytical engine operates on `methodology_version = "2026.2"`, covering 7 analytical domains, PR depth metrics, git tags fallback, and GitHub Discussions (GraphQL v4).

---

## 1. Error Envelope Standard

All HTTP, domain, database, and validation errors are normalized by [`backend/app/core/error_handlers.py`](../backend/app/core/error_handlers.py) into structured JSON responses:

- **Standard Error (`ErrorResponse`):**
  ```json
  {
    "detail": "Repository pallets/flask was not found.",
    "error_code": "RESOURCE_NOT_FOUND",
    "status_code": 404,
    "path": "/api/v1/analytics/pallets/flask/status",
    "timestamp": "2026-11-17T10:00:00Z",
    "details": null
  }
  ```
- **Validation Error (`ValidationErrorResponse`, HTTP `422`):**
  ```json
  {
    "detail": "Request body validation failed.",
    "error_code": "VALIDATION_ERROR",
    "status_code": 422,
    "path": "/api/v1/classify",
    "timestamp": "2026-11-17T10:00:00Z",
    "errors": [
      {
        "field": "body.title",
        "message": "Field required",
        "type": "missing"
      }
    ]
  }
  ```

Common error codes: `RESOURCE_NOT_FOUND` (`404`), `INVALID_REPOSITORY_URL` (`400`), `REPOSITORY_INELIGIBLE` (`422` / `400`), `INVALID_PARAMETER` (`400`), `GITHUB_RATE_LIMIT` (`429`), `GITHUB_API_ERROR` (`502`), `DATABASE_ERROR` (`500`), `INTERNAL_SERVER_ERROR` (`500`). All V2 public error messages run through `_sanitize_public_error_message()` to strip filesystem paths, connection strings, and stack frames.

---

## 2. System & Health Telemetry Endpoints

### `GET /`
- **Summary:** Root discovery endpoint.
- **Response (`200 OK`, `RootResponse`):**
  ```json
  {
    "title": "OpenSourceLens API",
    "version": "1.0.0",
    "description": "Intelligent GitHub Repository Analytics...",
    "docs_url": "/docs",
    "api_v1_url": "/api/v1",
    "supported_repositories": ["RocketChat/Rocket.Chat", "pallets/click"]
  }
  ```

### `GET /health` and `GET /api/v1/health`
- **Summary:** Production health and system telemetry probe.
- **Response (`200 OK`, `HealthResponse`):**
  ```json
  {
    "status": "healthy",
    "version": "1.0.0",
    "timestamp": "2026-11-17T10:15:00Z",
    "database": {
      "status": "connected",
      "dialect": "sqlite",
      "latency_ms": 1.42,
      "tables_count": 21,
      "error": null
    },
    "queue_depths": {
      "ingestion": { "pending": 0, "running": 0 },
      "analytics": { "pending": 0, "running": 0 }
    },
    "active_workers": {
      "active_workers_count": 1,
      "active_worker_ids": ["worker-compose-1"]
    },
    "model_status": {
      "classifier_generalizable": "ready",
      "semantic_embeddings": "ready",
      "topic_discovery": "ready"
    }
  }
  ```

### `GET /api/v1/ping`
- **Summary:** Lightweight health ping probe.
- **Response (`200 OK`):** `{ "message": "OpenSourceLens API v1 is active", "status": "ok" }`.

---

## 3. Dynamic Ingestion Endpoints (`/api/ingest/*` and `/api/v1/ingest/*`)

Implemented in [`backend/app/api/v1/endpoints/ingest.py`](../backend/app/api/v1/endpoints/ingest.py) and dual-mounted at `/api/ingest` and `/api/v1/ingest`:

### `POST /api/v1/ingest/preview`
- **Summary:** Validate repository eligibility and preview rate-budget consumption and data sufficiency without downloading entity streams.
- **Request Body (`IngestionPreviewRequest`):**
  ```json
  {
    "repo": "https://github.com/pallets/flask"
  }
  ```
- **Response (`200 OK`, `IngestionPreviewResponse`):**
  ```json
  {
    "repo_name": "pallets/flask",
    "owner": "pallets",
    "name": "flask",
    "canonical_url": "https://github.com/pallets/flask",
    "is_eligible": true,
    "rejection_reason": null,
    "metadata": {
      "stars_count": 69000,
      "forks_count": 16000,
      "open_issues_count": 45,
      "primary_language": "Python"
    },
    "estimated_budgets": {
      "fast_snapshot": { "max_requests": 10, "estimated_duration_seconds": 12 },
      "standard_diagnostic": { "max_requests": 50, "estimated_duration_seconds": 45 }
    },
    "confidence_preview": {
      "expected_band": "high",
      "predicted_sufficient_domains": 6
    },
    "badges": []
  }
  ```

### `POST /api/v1/ingest/submit`
- **Summary:** Enqueue an asynchronous repository ingestion job.
- **Request Body (`IngestionSubmitRequest`):**
  ```json
  {
    "repo": "pallets/flask",
    "mode": "standard_diagnostic",
    "priority": 0
  }
  ```
- **Response (`IngestionSubmitResponse`):**
  - **Status `201 Created`:** When a new `CollectionRun` is submitted.
  - **Status `200 OK`:** When an active (`pending`/`running`) collection run already exists for this repository.
  ```json
  {
    "job_id": 14,
    "repo_name": "pallets/flask",
    "mode": "standard_diagnostic",
    "status": "pending",
    "created_at": "2026-11-17T10:20:00Z",
    "message": "Ingestion job submitted successfully."
  }
  ```

### `GET /api/v1/ingest/status/{job_id}`
- **Summary:** Poll live ingestion progress, entity stream completion, and rate-limit consumption.
- **Response (`200 OK`, `IngestionStatusResponse`):**
  ```json
  {
    "job_id": 14,
    "repo_name": "pallets/flask",
    "mode": "standard_diagnostic",
    "status": "completed",
    "worker_id": "worker-embedded-1",
    "leased_until": "2026-11-17T10:30:00Z",
    "lease_version": 1,
    "retry_count": 0,
    "max_retries": 2,
    "is_partial": false,
    "partial_reason": null,
    "resumes_at": null,
    "rate_limit_consumed": 28,
    "items_collected": {
      "commits": 300,
      "contributors": 120,
      "pull_requests": 100,
      "issues": 100,
      "discussions": 25,
      "releases": 15
    },
    "entity_status": {
      "commits": "complete",
      "pull_requests": "complete",
      "issues": "complete",
      "discussions": "complete",
      "releases": "complete"
    },
    "coverage_summary": { ... },
    "error_message": null,
    "started_at": "2026-11-17T10:20:02Z",
    "completed_at": "2026-11-17T10:20:38Z",
    "duration_seconds": 36.2
  }
  ```

---

## 4. Analytics V2 Endpoints (`/api/v1/analytics/...`)

Implemented in [`backend/app/api/v1/endpoints/analytics.py`](../backend/app/api/v1/endpoints/analytics.py):

### `POST /api/v1/analytics/{owner}/{name}/recompute`
- **Summary:** Request calculation or recalculation of repository analytical metrics.
- **Request Body (optional `AnalysisRecomputeRequest`):**
  ```json
  {
    "force": false,
    "requested_methodology_version": "2026.2"
  }
  ```
- **Response (`AnalysisRecomputeResponse`):**
  - **Status `202 Accepted`:** Enqueued new calculation (`enqueued=true`) or run is already in progress (`already_active=true`).
  - **Status `200 OK`:** Reused up-to-date completed run for current source revision (`cached=true`).

### `GET /api/v1/analytics/{owner}/{name}/status?methodology_version=2026.2`
- **Summary:** Check current analysis readiness, active job status, and staleness relative to source data revision.
- **Response (`200 OK`, `AnalysisStatusResponse`):**
  ```json
  {
    "repository": {
      "id": 1,
      "repo_name": "pallets/flask",
      "owner": "pallets",
      "name": "flask",
      "analytics_source_revision": 2,
      "last_synced_at": "2026-11-17T10:20:38Z",
      "last_analyzed_at": "2026-11-17T10:21:05Z"
    },
    "has_analysis": true,
    "status": "completed",
    "is_stale": false,
    "retry_count": 0,
    "max_retries": 3,
    "failure_category": null,
    "current_run": { ... },
    "attempts": [ ... ],
    "latest_completed_run_id": 7
  }
  ```

### `GET /api/v1/analytics/{owner}/{name}/latest?methodology_version=2026.2`
- **Summary:** Retrieve the complete synthesized 7-domain analytical metrics payload for the latest completed run.
- **Response (`200 OK`, `AnalysisLatestResponse`):**
  ```json
  {
    "repository": { ... },
    "is_stale": false,
    "active_run_id": null,
    "active_run_status": null,
    "run": {
      "id": 7,
      "repository_id": 1,
      "status": "completed",
      "methodology_version": "2026.2",
      "analytics_source_revision": 2,
      "analysis_reference_time": "2026-11-17T10:20:45Z",
      "metrics_payload": {
        "development_activity": { ... },
        "contributor_ecosystem": { ... },
        "pull_request_intelligence": { ... },
        "issue_intelligence": { ... },
        "repository_synthesis": {
          "diagnostic_scorecard": { ... },
          "cross_domain_correlations": [ ... ],
          "release_evolution": { ... },
          "discussions_intelligence": { ... }
        }
      },
      "coverage_summary": { ... },
      "confidence_summary": {
        "overall_score": 0.88,
        "confidence_band": "high",
        "domain_scores": { ... }
      },
      "rule_interpretations": [ ... ]
    }
  }
  ```

### `GET /api/v1/analytics/{owner}/{name}/domains/{domain_key}?methodology_version=2026.2`
- **Summary:** Retrieve a targeted slice of a single analytical domain.
- **Path Parameters:**
  - `domain_key`: `development_activity`, `contributor_ecosystem`, `pull_request_intelligence`, `issue_intelligence`, `release_evolution`, `discussions_intelligence`, or `repository_synthesis`.
- **Response (`200 OK`, `AnalysisDomainSliceResponse`):**
  Returns `{ repository, run_id, domain, analytics_source_revision, methodology_version, analysis_reference_time, is_stale, data, coverage, confidence_breakdown, rules }`.

### `GET /api/v1/analytics/{owner}/{name}/history?limit=20&offset=0`
- **Summary:** Paginated historical list of all analysis runs for a repository.

### `GET /api/v1/analytics/{owner}/{name}/runs/{run_id}`
- **Summary:** Retrieve the full detail of a specific historical analysis run by ID.

---

## 5. Applied Intelligence & Discovery Endpoints (Stage 5 & Stage 6)

### `POST /api/v1/analytics/{owner}/{name}/similar`
- **Summary:** Dynamic in-repository semantic similarity and duplicate issue retrieval.
- **Request Body (`SimilarQueryRequest`):**
  ```json
  {
    "title": "ModuleNotFoundError when importing flask.cli",
    "body": "Running under Python 3.12 fails to find Click sub-dependencies.",
    "issue_id": null,
    "top_k": 5,
    "threshold": 0.65
  }
  ```
- **Response (`200 OK`, `SimilarQueryResponse`):**
  ```json
  {
    "repository_id": 1,
    "repo_name": "pallets/flask",
    "query_text": "ModuleNotFoundError when importing flask.cli ...",
    "total_indexed_issues": 100,
    "threshold": 0.65,
    "matches": [
      {
        "issue_id": 4821,
        "issue_number": 4821,
        "title": "Import error on Python 3.12 with CLI commands",
        "state": "closed",
        "similarity_score": 0.892,
        "is_potential_duplicate": true,
        "html_url": "https://github.com/pallets/flask/issues/4821",
        "labels": ["bug", "cli"]
      }
    ]
  }
  ```

### `GET /api/v1/analytics/{owner}/{name}/issues/classified?state=open&limit=60`
- **Summary:** Automatically fetch stored repository issues and classify each into `bug`, `feature`, `documentation`, `question`, or `other` using the prior-calibrated 5-class ML classifier (`GeneralizableIssueClassifier`) combined with canonical GitHub label normalization.
- **Query Parameters:**
  - `state`: `"open"` (default), `"closed"`, or `"all"`.
  - `limit`: Maximum number of recent issues to fetch and classify (`1`–`200`, default `60`).
- **Response (`200 OK`):**
  ```json
  {
    "repository": "microsoft/vscode",
    "state_filter": "open",
    "total_open": 294,
    "total_closed": 242,
    "total_matching": 294,
    "classified_count": 60,
    "model_used": "Generalizable 5-Class ML Classifier (TF-IDF + Calibrated Linear)",
    "category_counts": { "bug": 38, "feature": 14, "documentation": 3, "question": 2, "other": 3 },
    "items": [
      {
        "issue_number": 342520,
        "title": "Cannot update while running on a read-only volume.",
        "state": "open",
        "author": "octocat",
        "created_at": "2026-11-15T10:12:00+00:00",
        "comments_count": 4,
        "html_url": "https://github.com/microsoft/vscode/issues/342520",
        "github_labels": ["install-update"],
        "predicted_category": "bug",
        "confidence": 0.80,
        "is_uncertain": false
      }
    ]
  }
  ```

### `GET /api/v1/analytics/{owner}/{name}/topics?k=5&min_cluster_size=3`
- **Summary:** Unsupervised backlog theme discovery using Spherical KMeans clustering and c-TF-IDF keyword extraction.
- **Query Parameters:**
  - `k`: Number of topical clusters (optional, automatically estimated if omitted).
  - `min_cluster_size`: Minimum issues required per cluster (default: 3).
- **Response (`200 OK`, `TopicDiscoveryResponse`):**
  ```json
  {
    "repo_name": "pallets/flask",
    "total_issues_analyzed": 100,
    "clusters_count": 4,
    "topics": [
      {
        "cluster_id": 0,
        "topic_name": "CLI & Script Execution",
        "keywords": ["cli", "click", "command", "argv", "runner"],
        "issue_count": 28,
        "open_count": 8,
        "closed_count": 20,
        "representative_issues": [
          { "issue_number": 4821, "title": "Import error with CLI", "similarity_to_centroid": 0.884 }
        ]
      }
    ]
  }
  ```

### `GET /api/v1/analytics/{owner}/{name}/report?format=markdown`
- **Summary:** Generate a comprehensive Executive Intelligence Audit Report.
- **Query Parameters:**
  - `format`: `"markdown"` (returns raw `text/markdown`) or `"json"` (returns structured `application/json`).
- **Response (`200 OK`):**
  - In `markdown` format: Complete formatted multi-page audit report featuring executive summary, methodology version, confidence rating, domain scorecards, cross-domain risk findings, and actionable maintainer recommendations.
  - In `json` format: Structured object containing `{ repository, methodology_version, generated_at, executive_summary, domain_scorecards, cross_domain_findings, recommendations }`.

---

## 6. Machine Learning Inference & Model Registry Endpoints

Implemented in [`backend/app/api/v1/endpoints/classify.py`](../backend/app/api/v1/endpoints/classify.py) and [`backend/app/api/v1/endpoints/models.py`](../backend/app/api/v1/endpoints/models.py):

### `POST /api/v1/classify`
- **Summary:** Classify issue text into canonical engineering categories (`bug`, `feature`, `documentation`, `question`, `maintenance`) using the generalizable champion classifier.
- **Request Body (`ClassificationRequest`):**
  ```json
  {
    "title": "Documentation missing for custom CLI argument parsing",
    "body": "The quickstart tutorial does not mention how to register Click group options.",
    "repo_name": "pallets/flask",
    "model_type": "distilbert"
  }
  ```
- **Response (`200 OK`, `ClassificationResponse`):**
  ```json
  {
    "predicted_category": "documentation",
    "confidence": 0.942,
    "is_uncertain": false,
    "probabilities": {
      "documentation": 0.942,
      "question": 0.031,
      "feature": 0.015,
      "bug": 0.008,
      "maintenance": 0.004
    },
    "model_used": "distilbert-finetuned",
    "prediction_id": 128
  }
  ```
  *(Note: If max confidence < 0.55, `is_uncertain=True` and `predicted_category="unclassified"` per calibrated abstention rules).*

### `GET /api/v1/models`
- **Summary:** List all registered ML/NLP models and current host memory load status.
- **Response (`200 OK`):** Returns registry status for `distilbert`, `generalizable_classifier`, and `sentence_transformer`.

### `POST /api/v1/models/prewarm`
- **Summary:** Pre-load model weights into host memory to ensure low-latency subsequent inference.
- **Response (`200 OK`):** `{ "distilbert": "prewarmed", "sentence_transformer": "prewarmed", "generalizable_classifier": "prewarmed" }`.
