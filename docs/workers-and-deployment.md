# OpenSourceLens — Workers & Production Deployment Guide

This guide documents background worker execution, configuration, lease recovery, and containerized multi-service deployment.

---

## 1. Environment Variables Reference

All environment variables read by OpenSourceLens and their default values (automatically loaded from `.env` via `python-dotenv`):

| Variable | Default Value | Purpose |
|---|---|---|
| `DATABASE_URL` | `sqlite:///data/opensource_lens.db` | Database connection URL. In Docker Compose, set to PostgreSQL: `postgresql://user:pass@db:5432/opensourcelens`. |
| `GITHUB_TOKEN` | `None` | Optional GitHub Personal Access Token granting 5,000 req/hr quota (unauthenticated IP limit is 60 req/hr). |
| `FASTAPI_HOST` | `0.0.0.0` | Bind IP address for FastAPI server. |
| `FASTAPI_PORT` | `8000` | Bind port for FastAPI server. |
| `BACKEND_CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Allowed CORS origins for browser clients. |
| `ENABLE_INGESTION_WORKER` | `True` | Automatically runs an embedded asyncio `IngestionWorker` in FastAPI `lifespan()`. |
| `ENABLE_ANALYTICS_WORKER` | `True` | Automatically runs an embedded asyncio `AnalyticsWorker` in FastAPI `lifespan()`. |
| `INGESTION_WORKER_POLL_INTERVAL_SECONDS` | `1.0` | Queue polling interval for the ingestion worker. |
| `ANALYTICS_WORKER_POLL_INTERVAL_SECONDS` | `1.0` | Queue polling interval for the analytics worker. |
| `ANALYTICS_WORKER_RECONCILE_INTERVAL_SECONDS` | `60.0` | Cadence for reconciling expired leases and orphaned jobs. |
| `PREWARM_ON_STARTUP` | `False` | Pre-load transformer weights and embeddings into host RAM on application boot. |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000/api/v1` | Public API base URL used by the Next.js frontend. |
| `BACKEND_INTERNAL_URL` | `http://127.0.0.1:8000` | Server-side internal API URL for Docker container routing. |

---

## 2. Worker Execution Architectures

OpenSourceLens provides two worker runtime models:

### 2.1 Embedded In-Process Mode (Default for Local Development)
When `ENABLE_INGESTION_WORKER=true` and `ENABLE_ANALYTICS_WORKER=true` (the defaults), starting FastAPI automatically boots both background worker loops inside the asyncio event loop:
```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
A single terminal command serves the HTTP API, ingests GitHub streams, computes 7-domain analytics, and indexes semantic embeddings.

### 2.2 Decoupled Standalone Daemon Mode (Production & Distributed Scale)
For high-volume production deployments, worker daemons run in dedicated processes or separate containers:
```powershell
# Start standalone IngestionWorker daemon
.\.venv\Scripts\python.exe -m backend.app.services.ingestion_worker

# Start standalone AnalyticsWorker daemon
.\.venv\Scripts\python.exe -m backend.app.services.analytics_worker
```
When running standalone workers, set `ENABLE_INGESTION_WORKER=false` and `ENABLE_ANALYTICS_WORKER=false` in the API container to prevent redundant polling loops.

---

## 3. Lease Recovery, Retries & Reconciliation

Both queues enforce optimistic monotonic `lease_version` fencing to prevent double execution or split-brain commits:

| Mechanism | Ingestion Queue (`collection_runs`) | Analytics Queue (`analysis_runs`) |
|---|---|---|
| **Default Lease Duration** | `600s` (10 minutes) | `600s` (10 minutes) |
| **Max Retries** | `2` (up to 3 total attempts) | `3` (up to 4 total attempts) |
| **Active Heartbeat** | Async loop renewing lease every `15.0s` (`_heartbeat_loop`) | Monotonic attempt tracking per claimed run |
| **Lease Expiration Recovery** | `DurableJobQueue.recover_expired_jobs()` re-queues expired jobs as `pending` or marks `failed` once max retries are exceeded. | `AnalyticsJobQueue.reconcile_analytics_jobs()` marks expired runs `failed` or re-queues them with attempt logs. |
| **Rate-Limit Auto-Resumption** | `reopen_resumed_rate_limited_jobs()` checks ISO `resumes_at` and automatically re-queues `rate_limited` jobs when reset time passes. | N/A (Analytics calculations execute purely against local database snapshots). |

---

## 4. Multi-Container Docker Compose Stack (Stage 6)

The repository provides a production multi-container stack orchestrated via [`docker-compose.yml`](../docker-compose.yml):

```bash
# 1. Provide an optional GitHub token for 5,000 req/hr rate limits
export GITHUB_TOKEN="ghp_your_token_here"

# 2. Build and launch all services
docker compose up --build
```

### 4.1 Orchestrated Services
1. **`db` (`postgres:16-alpine`):**
   - High-performance relational database with `osl_pgdata` volume persistence.
   - Built-in healthcheck: `pg_isready -U opensourcelens -d opensourcelens` interval 5s.
2. **`backend` ([`Dockerfile.backend`](../Dockerfile.backend)):**
   - Python 3.11+ slim container serving FastAPI on port 8000 via Uvicorn.
   - Depends on `db` achieving `healthy` state before initializing migrations and starting.
   - Mounts volumes `osl_models` and `osl_data` to cache ML weights and exports.
3. **`analytics_worker` ([`Dockerfile.worker`](../Dockerfile.worker)):**
   - Dedicated standalone `AnalyticsWorker` daemon container processing background analytical queues.
   - Communicates with PostgreSQL via `DATABASE_URL`.
4. **`frontend` ([`Dockerfile.frontend`](../Dockerfile.frontend)):**
   - Next.js 15 standalone Node.js production server on port 3000.
   - Proxies API requests to `backend:8000`.

### 4.2 Health & Telemetry Probes
Production health telemetry is accessible at:
- `http://localhost:8000/health`
- `http://localhost:8000/api/v1/health`

The probe returns:
- Database connectivity status, driver dialect, and ping latency in milliseconds.
- Queue depths: pending and running job counts for both `ingestion` and `analytics`.
- Active worker leases and worker IDs.
- ML model readiness: `classifier_generalizable`, `semantic_embeddings`, `topic_discovery`.
