# OpenSourceLens — Complete HTTP API Reference

OpenSourceLens serves REST JSON endpoints via FastAPI ([`backend/app/main.py`](../backend/app/main.py), [`backend/app/api/v1/api.py`](../backend/app/api/v1/api.py)). Interactive Swagger and ReDoc specs are automatically available at `/docs` and `/redoc` (OpenAPI schema at `/openapi.json`).

> **Scope & Legacy Endpoint Note:** For product context, see [product-vision.md](./product-vision.md). Sections **3.2** (Dynamic Ingestion) and **3.3** (Modern Analytics V2) power the universal any-repository architecture. Sections **3.4** (Legacy Phase A Repositories/Analytics) and **3.5** (benchmark-coupled `RelatedIssueService` and `ResolutionService` routes) are legacy interfaces retained temporarily until code cleanup and per-repository ML decoupling.

---

## 1. Error Envelope Standard

All HTTP, domain, database, and validation errors are normalized by [`backend/app/core/error_handlers.py`](../backend/app/core/error_handlers.py) into:

- **`ErrorResponse`:** `{ detail, error_code, status_code, path, timestamp, details? }`
- **HTTP `422` (`ValidationErrorResponse`):** `{ detail, error_code: "VALIDATION_ERROR", status_code: 422, path, timestamp, errors: [{ field, message, type }] }`

Common error codes: `RESOURCE_NOT_FOUND` (`404`), `INVALID_REPOSITORY_URL` (`400`), `REPOSITORY_INELIGIBLE` (`422` / `400` in submit handlers), `INVALID_PARAMETER` (`400`), `GITHUB_RATE_LIMIT` (`429`), `GITHUB_API_ERROR` (`502` / upstream status), `DATABASE_ERROR` (`500`), `INTERNAL_SERVER_ERROR` (`500`). Note that `GET /api/v1/analytics/*` V2 error responses run failure messages through `_sanitize_public_error_message()` to strip local paths, URIs, and secrets.

---

## 2. Legacy Analytics vs. Modern Analytics V2

| Dimension | Legacy Analytics (`/api/v1/analytics/...`) | Modern Analytics V2 (`/api/v1/analytics/{owner}/{name}/...`) |
|---|---|---|
| **Backing Tables** | `repository_analytics` + `issues` (Phase A) | `analysis_runs` + `analysis_run_attempts` + `repositories` (Phase C) |
| **Analytical Scope** | Issue counts, categories, basic resolution stats | Full 6-domain intelligence (`development_activity`, `contributor_ecosystem`, `pull_request_intelligence`, `issue_intelligence`, `release_evolution` embedded in `repository_synthesis`) |
| **Execution Model** | Synchronous read of pre-populated rows | Asynchronous durable job enqueue (`POST recompute`), status polling, staleness vs. `analytics_source_revision`, methodology filtering, retry & attempt audit |
| **Repo Resolution** | Case-sensitive `repo_name == f"{owner}/{name}"` | Case-insensitive `_resolve_repository_ci()` returning canonical `RepositoryFreshnessEnvelope` |
| **Frontend Usage** | **Currently consumed by `frontend/`** | **Not yet wired into `frontend/`** |

---

## 3. Endpoint Inventory

### 3.1 System & Health Endpoints
- **`GET /`** → `RootResponse`: `{ title, version, description, docs_url, api_v1_url, supported_repositories }`. Reads tracked repos from DB or defaults to the 4 benchmark slugs. Status: `200`.
- **`GET /health`** → `HealthResponse`: Evaluates DB connectivity (`check_connection`, `get_table_status`), returning `{ status: "healthy"|"degraded", version, timestamp, database: { status, dialect, latency_ms, tables_count, error } }`. Status: `200`.
- **`GET /api/v1/ping`** → `{ ping: "pong", api_version: "v1" }`. Status: `200`.

### 3.2 Dynamic Ingestion Endpoints (`/api/ingest/*` and `/api/v1/ingest/*`)
Implemented in [`backend/app/api/v1/endpoints/ingest.py`](../backend/app/api/v1/endpoints/ingest.py) and dual-mounted at both `/api/ingest` and `/api/v1/ingest`:

- **`POST /api/v1/ingest/preview`** / **`POST /api/ingest/preview`**
  - **Body (`IngestionPreviewRequest`):** `{ "repo": "pallets/flask" }` (or GitHub URL).
  - **Response (`200`, `IngestionPreviewResponse`):** `{ repo_name, owner, name, canonical_url, is_eligible, rejection_reason, metadata, estimated_budgets, confidence_preview, badges }`.
  - **Errors:** `400` invalid URL, `404` repo not found on GitHub, `429` GitHub rate limit, `502` GitHub API error.
- **`POST /api/v1/ingest/submit`** / **`POST /api/ingest/submit`**
  - **Body (`IngestionSubmitRequest`):** `{ "repo": "pallets/flask", "mode": "standard_diagnostic" | "fast_snapshot", "priority": 0 }`.
  - **Response (`IngestionSubmitResponse`):** `{ job_id, repo_name, mode, status, created_at, message }`.
  - **Status Codes:** `201 Created` when a new `CollectionRun` job is created; `200 OK` if an active (`pending`/`running`) `CollectionRun` already exists for the repository.
  - **Errors:** `400` invalid URL or ineligible repo (private / empty stub), `404` repo not found, `429` rate limit, `422` invalid mode.
- **`GET /api/v1/ingest/status/{job_id}`** / **`GET /api/ingest/status/{job_id}`**
  - **Response (`200`, `IngestionStatusResponse`):** `{ job_id, repo_name, mode, status, worker_id, leased_until, lease_version, retry_count, max_retries, is_partial, partial_reason, resumes_at, rate_limit_consumed, items_collected, entity_status, coverage_summary, error_message, started_at, completed_at, duration_seconds }`.
  - **Errors:** `404` job ID not found.

Example preview/submit flow:
```bash
curl -X POST http://127.0.0.1:8000/api/v1/ingest/preview \
  -H "Content-Type: application/json" \
  -d '{"repo": "https://github.com/pallets/flask"}'

curl -X POST http://127.0.0.1:8000/api/v1/ingest/submit \
  -H "Content-Type: application/json" \
  -d '{"repo": "pallets/flask", "mode": "fast_snapshot"}'
```

### 3.3 Modern Analytics V2 Endpoints (`/api/v1/analytics/...`)
Implemented in [`backend/app/api/v1/endpoints/analytics.py`](../backend/app/api/v1/endpoints/analytics.py):

- **`POST /api/v1/analytics/{owner}/{name}/recompute`**
  - **Body (optional `AnalysisRecomputeRequest`):** `{ "force": false, "requested_methodology_version": "2026.1" }` (defaults when body omitted).
  - **Response (`AnalysisRecomputeResponse`):** `{ repository: RepositoryFreshnessEnvelope, run: AnalysisRunRead, enqueued, already_active, cached, message }`.
  - **Status Codes:** `202 Accepted` for new (`enqueued=true`) or already-active (`already_active=true`) jobs; `200 OK` when reusing an up-to-date completed run (`cached=true`). Non-blocking (does not run calculation inside the request).
  - **Errors:** `404` unknown repository, `400` invalid methodology version.
- **`GET /api/v1/analytics/{owner}/{name}/status?methodology_version=2026.1`**
  - **Response (`200`, `AnalysisStatusResponse`):** `{ repository, has_analysis, status, is_stale, retry_count, max_retries, failure_category, current_run, attempts, latest_completed_run_id }`. (`status` is `"not_analyzed"` if no runs exist yet.)
- **`GET /api/v1/analytics/{owner}/{name}/latest?methodology_version=2026.1`**
  - **Response (`200`, `AnalysisLatestResponse`):** `{ repository, is_stale, active_run_id, active_run_status, run }`.
  - **Errors:** `404` if repository unknown or has no completed run for that methodology.
- **`GET /api/v1/analytics/{owner}/{name}/history?limit=20&offset=0&status=...`**
  - **Query Params:** `limit` (`1..100`, default `20`), `offset` (`>= 0`, default `0`), optional `status` filter (`cancelled`, `completed`, `failed`, `pending`, `running` in `_VALID_RUN_STATUSES`; staleness is computed dynamically via `is_stale`, and superseded mid-run aborts are persisted as `cancelled`).
  - **Response (`200`, `AnalysisHistoryResponse`):** `{ repository, total, limit, offset, items: [AnalysisRunRead] }`.
- **`GET /api/v1/analytics/{owner}/{name}/runs/{run_id}`** and **`GET /api/v1/analytics/runs/{run_id}`**
  - **Response (`200`, `AnalysisRunDetailResponse`):** `{ repository, is_stale, run }`. Repository-scoped route enforces `run.repository_id == repo.id`; global route looks up any `run_id`.
- **`GET /api/v1/analytics/{owner}/{name}/domains/{domain_key}?methodology_version=2026.1`**
  - **Path Params:** `domain_key` in `{"development_activity", "contributor_ecosystem", "pull_request_intelligence", "issue_intelligence", "release_evolution", "repository_synthesis"}`. Projects `release_evolution` transparently from `repository_synthesis.release_evolution` along with its domain coverage, confidence breakdown, and filtered rules.
  - **Response (`200`, `AnalysisDomainSliceResponse`):** `{ repository, run_id, domain, analytics_source_revision, methodology_version, analysis_reference_time, is_stale, data, coverage, confidence_breakdown, rules }`.

Example V2 trigger + status poll:
```bash
curl -X POST http://127.0.0.1:8000/api/v1/analytics/pallets/flask/recompute \
  -H "Content-Type: application/json" \
  -d '{"force": false, "requested_methodology_version": "2026.1"}'

curl http://127.0.0.1:8000/api/v1/analytics/pallets/flask/status
```

### 3.4 Legacy Phase A Repository & Analytics Endpoints
- **`GET /api/v1/repos`** → `List[RepositorySummary]`: Lists all tracked repos ordered by `stars_count DESC`.
- **`GET /api/v1/repos/{owner}/{name}`** → `RepositoryDetail`: Case-sensitive repo lookup + latest `RepositoryAnalytics` row + issue counts.
- **`GET /api/v1/repos/{owner}/{name}/issues`** → `IssueListResponse`: Paginated (`limit=50`, `offset=0`), filterable by `state`, `category`, `speed_tier`, `search` (case-sensitive repo match).
- **`GET /api/v1/repos/{owner}/{name}/issues/{number}`** → `IssueDetail`: Full issue text + labels + resolution metadata.
- **`GET /api/v1/analytics/overview`** → `GlobalAnalyticsOverview`: Global issue aggregates across legacy `RepositoryAnalytics`.
- **`GET /api/v1/analytics/comparison`** → `RepositoryComparisonResponse`: Multi-repo comparison over legacy snapshots.
- **`GET /api/v1/analytics/{owner}/{name}`** → `RepositoryAnalyticsResponse`: Single-repo Phase A resolution/category snapshot (`404` if missing).
- **`GET /api/v1/analytics/{owner}/{name}/trends`** → `RepositoryTrendsResponse`: Legacy monthly issue creation vs. closure counts derived from `issues` table (`404` if repo missing). Note route ordering in `analytics.py` places static/V2 subpaths before `/{owner}/{name}` so no FastAPI path collision occurs.

### 3.5 Phase A ML Inference & Model Registry Endpoints
- **`POST /api/v1/classify`** (`ClassificationRequest` -> `ClassificationResponse`): Runs `distilbert` (default), `svm`, or `logistic_regression` on `{ title, body?, repo_name?, model_type, issue_id? }` and logs an audit row in `predictions`.
- **`GET /api/v1/related/{issue_id}?top_k=5&state=...`** → `RelatedIssuesResponse`: Queries precomputed `related_issues_cache` (`top_k` `1..20`, optional `state`).
- **`POST /api/v1/related`** (`RelatedSearchRequest` -> `RelatedIssuesResponse`): Encodes query with `all-MiniLM-L6-v2` against static `models/related_issues/embeddings.npz` (`repo_name` and `state` filters apply SQL filter over `issues` table before numpy masking).
- **`POST /api/v1/resolution/predict`** (`ResolutionPredictionRequest` -> `ResolutionPredictionResponse`): Predicts advisory velocity tier (`< 1 Day`, `1-7 Days`, `7-30 Days`, `30+ Days`), probabilities, adjacent `±1` bounds, advisory factors, and mandatory disclaimer. Logs to `predictions`.
- **`GET /api/v1/resolution/stats/{owner}/{name}`** → `ResolutionStatsResponse`: Reads legacy `RepositoryAnalytics` percentiles/survival (`404` if repo or legacy analytics row absent).
- **`GET /api/v1/models`** → `ModelRegistryResponse`: Lists 4 registered models (`distilbert`, `classical_baselines`, `sentence_transformer`, `resolution_velocity`) and memory load state.
- **`GET /api/v1/models/{model_id}`** → `ModelInfoItem`: Supports aliases (`transformer`, `svm`, `logistic_regression`, `related`, `embedder`, `resolution`, `velocity`, etc.).
- **`POST /api/v1/models/prewarm`** → `Dict[str, str]`: Synchronously loads all model weights into host memory.
