# OpenSourceLens — Workers & Deployment Guide

This guide documents background worker execution, environment configuration, lease recovery, and dialect deployment differences. It explicitly separates **locally tested behavior** from **unverified production behavior**. For product scope, see [product-vision.md](./product-vision.md).

---

## 1. Environment Variables

All environment variables currently read by the Python/Node codebase (and their comparison with `.env.example`):

| Variable | Read Location | Default | In `.env.example`? | Purpose |
|---|---|---|---|---|
| `DATABASE_URL` | `backend/app/database/session.py:31`, `backend/alembic/env.py:51` | Local SQLite (`data/opensource_lens.db`) | Yes | SQLAlchemy database connection URL (`postgres://` normalized to `postgresql://`). |
| `GITHUB_TOKEN` | `backend/app/services/github_client.py:67`, `backend/app/services/repository_eligibility.py:38`, `ml/...` | `None` (unauthenticated `60 req/hr`) | Yes | GitHub REST API v3 PAT (`5,000 req/hr` authenticated quota). |
| `BACKEND_CORS_ORIGINS` | `backend/app/core/config.py:28` | `["http://localhost:3000", "http://127.0.0.1:3000"]` | Yes | Comma-separated or JSON CORS origins. |
| `PREWARM_ON_STARTUP` | `backend/app/core/config.py:43` | `False` | Yes | Pre-load DistilBERT, MiniLM, TF-IDF, and RF models on API boot. |
| `ENABLE_ANALYTICS_WORKER` | `backend/app/core/config.py:47` | `False` | **No (Doc drift)** | Run an embedded asyncio `AnalyticsWorker` inside FastAPI `lifespan()`. |
| `ANALYTICS_WORKER_POLL_INTERVAL_SECONDS` | `backend/app/core/config.py:51` | `1.0` | **No (Doc drift)** | Poll interval (`>= 0.1s`) for embedded `AnalyticsWorker`. |
| `ANALYTICS_WORKER_RECONCILE_INTERVAL_SECONDS` | `backend/app/core/config.py:54` | `60.0` | **No (Doc drift)** | Lease/stale reconciliation cadence (`>= 1.0s`) in embedded `AnalyticsWorker`. |
| `NEXT_PUBLIC_API_URL` | `scripts/verify_connection.py:235`, `frontend/` | `http://localhost:8000/api/v1` | Yes | Public FastAPI URL for frontend/diagnostics. |
| `RUN_LIVE_GITHUB_TESTS` | `tests/test_async_repository_collector.py`, `tests/test_e2e_ingestion_lifecycle.py` | Unset (skipped) | No (test-only) | Opt-in flag (`=1`) to run live network tests against `api.github.com`. |

> **Configuration Drift Note:** `.env.example` also lists `FASTAPI_HOST` and `FASTAPI_PORT`, which are **not** read by [`backend/app/core/config.py`](../backend/app/core/config.py) or [`backend/app/main.py`](../backend/app/main.py) (`uvicorn.run` hardcodes `0.0.0.0:8000`). Use uvicorn CLI flags (`--host`, `--port`) instead.

---

## 2. Workers & Execution Modes

### 2.1 `AnalyticsWorker`
Implemented in [`backend/app/services/analytics_worker.py`](../backend/app/services/analytics_worker.py):
- **Loop & Reconciliation:** `run_worker_loop()` calls `reconcile_jobs(db)` periodically (`reconcile_interval=60.0s`), then executes `process_one_job()` (`claim_job` → `AnalyticsOrchestrator.process_claimed_run()` → publish/failure transition). Includes `drain_queue()` helper for batch draining until queue empty.
- **Standalone CLI Entrypoint (Implemented):**
  ```powershell
  .\.venv\Scripts\python.exe -m backend.app.services.analytics_worker
  ```
- **Embedded FastAPI Lifespan Mode (Opt-In):**
  Set `ENABLE_ANALYTICS_WORKER=true` in `.env` before starting uvicorn. FastAPI starts `AnalyticsWorker.run_worker_loop()` as an `asyncio.Task` on startup and signals cancellation on shutdown. Suitable for single-instance local execution; avoid enabling in every multi-process uvicorn worker when using SQLite.

### 2.2 `IngestionWorker`
Implemented in [`backend/app/services/ingestion_worker.py`](../backend/app/services/ingestion_worker.py):
- **Loop & Heartbeat:** `process_one_job()` first runs `DurableJobQueue.recover_expired_jobs(db)`, claims a job (`lease_duration_seconds=600`), launches an asyncio heartbeat renewing the lease every `15.0s`, runs `RepositoryCollector.collect_repository()`, enforces lease fencing on `complete_job()`/`fail_job()`, and invokes `_enqueue_post_ingestion_analytics()` after successful commits.
- **Standalone / Lifespan Gap:** Unlike `AnalyticsWorker`, **`IngestionWorker` currently has NO `if __name__ == "__main__"` CLI block and NO FastAPI `lifespan()` integration**. To run it continuously outside tests, invoke its loop via Python:
  ```powershell
  .\.venv\Scripts\python.exe -c "import asyncio; from backend.app.database.session import SessionLocal; from backend.app.services.ingestion_worker import IngestionWorker; asyncio.run(IngestionWorker().run_worker_loop(SessionLocal))"
  ```

---

## 3. Lease Recovery, Retries & Reconciliation

| Mechanism | Ingestion (`collection_runs`) | Analytics (`analysis_runs`) |
|---|---|---|
| **Default Lease / Max Retries** | `600s` lease, `max_retries = 2` (up to 3 total claims) | `600s` lease, `max_retries = 3` (up to 4 total attempts) |
| **Active Heartbeat Renewal** | Yes — `15.0s` async loop inside `IngestionWorker` (`_heartbeat_loop`) | **No active background heartbeat** during synchronous orchestrator computation; relies on static 600s lease |
| **Non-Retryable Handling** | Client errors recorded in `CollectionRun`; no separate error category | Immediate terminal `status="failed"` when `_classify_exception()` returns `"non_retryable"` |
| **Lease Expiration Recovery** | `recover_expired_jobs()`: re-queues as `pending` (`retry_count += 1`) or marks `failed` when exhausted | `reconcile_analytics_jobs()`: records `interruption_reason="lease_expired"`, re-queues or marks `failed`; also repairs missing post-ingestion analytics runs for recent `completed`/`completed_partial` `CollectionRun`s |

---

## 4. SQLite Local vs. PostgreSQL Deployment Considerations

- **Tested Locally (Verified in Unit & Integration Tests):**
  - SQLite WAL mode (`journal_mode=WAL`, `foreign_keys=ON`), single-process FastAPI + `AnalyticsWorker` tests (`TestClient` / mocked DB), SQLite `BEGIN IMMEDIATE` queue locking, and isolated WAL snapshot reads (`tests/test_analytics_queue.py`, `tests/test_analytics_worker.py`).
- **Configured but UNVERIFIED in Production / Multi-Worker Deployments:**
  - **PostgreSQL Docker Compose (`docker-compose.yml`):** Only defines the `postgres:16-alpine` `db` container; there are no Dockerfiles or Compose services for FastAPI, `IngestionWorker`, `AnalyticsWorker`, or Next.js.
  - **Driver Dependency:** `psycopg2-binary>=2.9.9` is commented out in [`backend/requirements.txt`](../backend/requirements.txt) and not installed in `.venv`.
  - **Multi-Worker PostgreSQL Concurrency:** `FOR UPDATE SKIP LOCKED` branches in `DurableJobQueue` and `AnalyticsJobQueue` have been tested via unit mocks / SQL compilation inspection only — **never verified against a live multi-worker PostgreSQL cluster**.
  - **Migration 001 Fresh-DB Warning:** Do not provision an empty production PostgreSQL instance using `alembic upgrade head` alone without first checking [database.md §4](./database.md#4-alembic-migrations--known-schema-drift-warning) due to Phase A column drift in Migration `001`.
  - **Authentication / Rate Limiting:** FastAPI exposes no API authentication, authorization, or per-client rate limiting; public deployment requires an external reverse proxy / auth gateway.
