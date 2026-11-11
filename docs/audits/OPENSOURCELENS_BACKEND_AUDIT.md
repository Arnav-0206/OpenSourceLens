# OpenSourceLens — Comprehensive Backend Technical Audit

**Audit Baseline Commit:** `9f394e6` (`feat(api): implement C.8.3 modern analytics v2 endpoints and worker integration`)  
**Previous Remote Checkpoint (`origin/main`):** `6d88338` (Phase C.7)  
**Audit Scope:** Read-only code, database, schema, configuration, test suite, and documentation inspection  
**Live Database Inspected:** `data/opensource_lens.db` (SQLite, Alembic revision `003_add_phase_c_analytics_infrastructure`)  

---

## 1. Executive Summary

OpenSourceLens has evolved across two major phases since its Phase 11 benchmark prototype (`7fb1c67`):
1. **Phase B (Dynamic GitHub Ingestion):** Implemented strict URL validation, public/non-empty repository eligibility checking, rate-budgeted async GitHub REST collection (`fast_snapshot` 10 requests, `standard_diagnostic` 50 requests), multi-entity relational normalization, atomic mutation/revision tracking, and a leased durable queue (`DurableJobQueue`, `IngestionWorker`).
2. **Phase C (C.1–C.8.3 Six-Domain Analytics Engine & V2 API):** Implemented six analytical modules (`development_activity`, `contributor_ecosystem`, `pull_request_intelligence`, `issue_intelligence`, `release_evolution`, and `repository_synthesis`), unified snapshot orchestration (`AnalyticsOrchestrator`), lease-and-revision fenced durable publication (`AnalyticsJobQueue`), background execution (`AnalyticsWorker`), post-ingestion automatic analytics enqueueing (`IngestionWorker._enqueue_post_ingestion_analytics`), and versioned V2 REST endpoints under `/api/v1/analytics`.

**Audit Verdict:** The backend core logic is architecturally cohesive, scientifically defensive against common open-source analytics pitfalls (NULL-vs-zero coercion, partial-week bias, right-censoring, AR(1) serial autocorrelation, bounded-history retention leakage, and ghost-worker races), and verified across **388 passing offline unit/integration backend tests** (with 3 opt-in live GitHub tests skipped). However, the system is **NOT yet production-ready as a complete end-to-end platform** due to seven concrete blockers documented below: Migration `001` vs. ORM Phase A column drift on fresh DBs, missing standalone CLI/lifespan runtime wiring for `IngestionWorker`, legacy-only Next.js frontend integration, unverified PostgreSQL multi-worker concurrency, un-indexed dynamic ML artifacts, missing API authentication/rate-limiting, and zero live multi-repo production verification.

### Component Status Matrix

| Component | Implemented | Tested | Production verified | Limitations |
|---|---|---|---|---|
| **FastAPI App & Error Envelopes** (`main.py`, `core/error_handlers.py`) | Yes | Yes (SQLite `TestClient`) | No | No API auth, authz, or per-client rate limiting; `FASTAPI_HOST`/`FASTAPI_PORT` env vars ignored (`main.py:190`). |
| **URL Validator & Eligibility Checker** (`validator.py`, `repository_eligibility.py`) | Yes | Yes (unit & mocked HTTPX) | Partially (live GitHub tests opt-in/skipped) | Requires `GITHUB_TOKEN` to avoid `60 req/hr` IP exhaustion; preview does not harvest streams. |
| **GitHub Client & Rate Budgeter** (`github_client.py`, `rate_budgeter.py`) | Yes | Yes (mocked HTTPX) | Partially (live skipped) | `paginate()` follows `Link` `rel="next"` without re-validating target host (`github_client.py:206,464`); no auto-scheduler for `rate_limited` resumption. |
| **Repository Collector & Revision Bump** (`repository_collector.py`) | Yes | Yes (SQLite) | No | Bounded modes cap commits (`<=300`), commit diffs (`<=5`), PR reviews (`<=20`); direct calls bypassing `complete_job` write flat `coverage_summary` (`:376`). |
| **Durable Ingestion Queue & `IngestionWorker`** (`job_queue.py`, `ingestion_worker.py`) | Yes | Yes (SQLite) | No | **No `__main__` CLI entrypoint** and not started in `lifespan()`; heartbeat uses asyncio loop sharing `Session` if sync drivers block; Postgres `FOR UPDATE SKIP LOCKED` unverified live. |
| **Database Schema & Alembic Migrations** (`models.py`, `alembic/versions/`) | Yes (17 tables) | Partially (SQLite migrations) | No | **Migration `001` mismatches Phase A ORM columns** on `issue_labels`, `repository_analytics`, `predictions`; `psycopg2-binary` not installed/commented out. |
| **Analytics Engine (C.2–C.6)** (`services/analytics/*.py`, `rule_engine.py`) | Yes (6 domains) | Yes (offline synthetic/DB tests) | No | Bounded ingestion leaves churn diff coverage and PR formal/inline review samples sparse; retention cohorts require unimplemented `full_history` collection mode; block-bootstrap CIs deferred. |
| **Analytics Orchestrator & Queue (C.1, C.7)** (`orchestrator.py`, `analytics_queue.py`) | Yes | Yes (SQLite WAL) | No | `_abort_superseded_stale_run` / `_VALID_RUN_STATUSES` mismatch (`/history?status=stale` rejected at `analytics.py:581`); `retry_count` vs non-retryable termination covered on SQLite only. |
| **`AnalyticsWorker` & Post-Ingestion Wiring (C.8.1–C.8.2)** (`analytics_worker.py`) | Yes | Yes (SQLite) | No | Static 600s lease with no background heartbeat during long orchestrations; opt-in `ENABLE_ANALYTICS_WORKER` shares process with uvicorn workers. |
| **Analytics V2 API (C.8.3)** (`api/v1/endpoints/analytics.py`) | Yes (7 V2 routes) | Yes (`test_api_analytics_v2.py`) | No | Coexists awkwardly with Phase A case-sensitive legacy endpoints (`repos.py`, legacy `analytics.py`); `_VALID_RUN_STATUSES` omits `"stale"`. |
| **Phase A ML/NLP Services** (`classification_service.py`, `related_service.py`, `resolution_service.py`) | Yes (benchmark artifacts) | Yes (pipeline tests) | Offline benchmark only | `related_service` searches static 2,328-issue `embeddings.npz` (`:169`), ignoring newly ingested issues; `resolution_service` still encodes benchmark `repo_name`. |
| **Next.js Frontend** (`frontend/src/`) | Phase 10 only | Not run in audit | No | Wired strictly to Phase A legacy APIs (`frontend/src/lib/api.ts`); no dynamic ingestion UI or Phase C multi-domain dashboard. |

---

## 2. Project Scope & Current Completion Status

OpenSourceLens's intended scope (`AGENTS.md`, `docs/0.1-project-definition.md`) is a **Single-Repository Intelligence Workspace** supporting **any eligible public GitHub repository**, using four benchmark repos (`microsoft/vscode`, `pytorch/pytorch`, `kubernetes/kubernetes`, `huggingface/transformers`) strictly for offline research/ML training rather than production scope limits.

Current milestone completion verified from repository state (`git log`):
- **Phases 0–11 (Track A Prototype):** Complete on offline 4-repo dataset (`7fb1c67`).
- **Phase A (Frontend Hardcoding Cleanup):** Complete (`ed93489`, see `docs/phase_a_completion_report.md`).
- **Phase B (Dynamic GitHub Ingestion & Durable Queue):** Complete (`validator`, `repository_eligibility`, `rate_budgeter`, `github_client`, `repository_collector`, `job_queue`, `ingestion_worker`, `ingest.py`, Migration `002`).
- **Phase C.1–C.7 (Six-Domain Analytics Engine & Orchestrator):** Complete (`6d88338` pushed to `origin/main`).
- **Phase C.8.1–C.8.3 (`AnalyticsWorker`, Post-Ingestion Wiring, V2 API):** Complete locally (`277b20b`, `ab92d20`, `9f394e6`; authorized for push with this audit commit).
- **Remaining Unfinished Track B Work:** Fresh-DB schema reconciliation (Migration 001), `IngestionWorker` CLI/daemon packaging, dynamic ML vectorization & decoupling, Phase D Frontend V2 integration, and live multi-worker PostgreSQL verification.

---

## 3. Actual Architecture & Component Map

1. **API Layer (`backend/app/main.py`, `backend/app/api/v1/`):**
   - `main.py:100-129` creates `FastAPI`, configures CORS (`BACKEND_CORS_ORIGINS`), mounts `api_router` under `/api/v1` and dual-mounts `ingest_router` under `/api/ingest`, and registers exception handlers (`backend/app/core/error_handlers.py`).
   - Route groups: `system` (`/`, `/health`, `/api/v1/ping`), `ingest` (`/api/ingest/*`, `/api/v1/ingest/*`), `repositories` (`/api/v1/repos/*`), `analytics` (`/api/v1/analytics/overview|comparison|{owner}/{name}|{owner}/{name}/trends`), `analytics-v2` (`/api/v1/analytics/runs/{run_id}`, `/{owner}/{name}/recompute|status|latest|history|runs/{run_id}|domains/{domain_key}`), `classification` (`/api/v1/classify`), `related-issues` (`/api/v1/related*`), `resolution` (`/api/v1/resolution/*`), `models` (`/api/v1/models*`).
2. **Ingestion Domain (`backend/app/services/`):**
   - `validator.py` (`GitHubURLValidator`), `repository_eligibility.py` (`RepositoryEligibilityChecker`), `confidence_matrix.py` (`ConfidenceMatrixEvaluator`), `rate_budgeter.py` (`EntityRateBudgeter`), `github_client.py` (`AsyncGitHubClient`), `repository_collector.py` (`RepositoryCollector`), `job_queue.py` (`DurableJobQueue`), `ingestion_worker.py` (`IngestionWorker`).
3. **Analytics Domain (`backend/app/services/analytics/`, `analytics_queue.py`, `analytics_worker.py`):**
   - `base.py` (`BaseMetricCalculator`, `CalculationResult`), `development_activity.py` (`DevelopmentActivityCalculator`), `contributors.py` (`ContributorIntelligenceCalculator`), `pull_requests.py` (`PullRequestIntelligenceCalculator`), `issues.py` (`IssueIntelligenceCalculator`), `releases.py` (`ReleaseEvolutionCalculator`), `cross_domain.py` (`CrossDomainSynthesisCalculator`), `rule_engine.py` (`build_release_rules`, `build_cross_domain_rules`, `consolidate_rules`), `orchestrator.py` (`AnalyticsOrchestrator`), `analytics_queue.py` (`AnalyticsJobQueue`), `analytics_worker.py` (`AnalyticsWorker`).
4. **Phase A ML/NLP Services (`backend/app/services/`, `ml/`):**
   - `model_manager.py` (`ModelManager`), `classification_service.py` (`ClassificationService`), `related_service.py` (`RelatedIssueService`), `resolution_service.py` (`ResolutionService`), reading pretrained artifacts under `models/`.
5. **Frontend (`frontend/src/`):** Legacy Phase 10 Next.js 15 UI (`api.ts` hits `/health`, `/api/v1/repos`, `/api/v1/analytics/overview|comparison|{o}/{n}|trends`, `/api/v1/classify`, `/api/v1/related`, `/api/v1/resolution`, `/api/v1/models`).

---

## 4. Repository Structure

Top-level and key module structure verified on disk:
- `backend/app/main.py`, `backend/app/core/{config.py, error_handlers.py, exceptions.py}`
- `backend/app/database/{base.py, models.py, session.py, init_db.py, ingest.py, schema_generator.py}`
- `backend/app/api/v1/{api.py, endpoints/{analytics.py, classify.py, ingest.py, models.py, related.py, repos.py, resolution.py}}`
- `backend/app/schemas/{analytics.py, analytics_v2.py, classification.py, errors.py, health.py, ingestion.py, issue.py, models.py, related.py, repository.py, resolution.py}`
- `backend/app/services/` + `backend/app/services/analytics/`
- `backend/alembic/versions/{001_initial_benchmark_schema.py, 002_add_dynamic_ingestion_and_multi_entity_tables.py, 003_add_phase_c_analytics_infrastructure.py}`
- `data/opensource_lens.db` (24.5 MB SQLite file), `data/{raw, processed, backups}/`
- `models/` (`baseline_logistic_regression.joblib`, `baseline_svm.joblib`, `final_classical_nlp_pipeline.joblib`, `tfidf_vectorizer.joblib`, `distilbert_finetuned/`, `related_issues/`, `resolution_*`, `tokenizer/`)
- `ml/` (`classification`, `data_collection`, `deep_learning`, `eda`, `nlp`, `preprocessing`, `related_issues`, `repository_analyzer.py`, `resolution_analysis`)
- `scripts/` (9 `verify_*.py` Phase 11 diagnostics), `tests/` (94 test files, 1,079 collected tests), `docs/`, `frontend/`, `docker-compose.yml`, `alembic.ini`.

---

## 5. Database Schema & Data Flow

Inspection of `Base.metadata.tables` and live `sqlite3` introspection on `data/opensource_lens.db` verified **17 ORM data tables** (+ `alembic_version`):
- **Phase A Tables:** `repositories` (19 cols), `issues` (39 cols), `issue_labels` (5 cols), `repository_analytics` (15 cols), `predictions` (12 cols), `related_issues_cache` (7 cols).
- **Phase B Tables:** `collection_runs` (16 cols), `commits` (13 cols incl. `committer_date` from Migration 003), `contributors` (19 cols), `contributor_activities` (7 cols), `pull_requests` (17 cols), `pull_request_reviews` (8 cols), `pull_request_review_comments` (12 cols), `releases` (8 cols), `issue_comments` (12 cols).
- **Phase C Tables:** `analysis_runs` (23 cols), `analysis_run_attempts` (10 cols).

### Live Benchmark Database State & Severe Limitation
Direct query against `data/opensource_lens.db` shows:
- `repositories`: `4` rows (`vscode`, `pytorch`, `kubernetes`, `transformers`)
- `issues`: `2,328` rows
- `issue_labels`: `6,361` rows
- `repository_analytics`: `4` rows (legacy summaries only)
- `predictions`: `1,623` rows
- `related_issues_cache`: `11,640` rows
- **`collection_runs`: `0`, `commits`: `0`, `contributors`: `0`, `contributor_activities`: `0`, `pull_requests`: `0`, `pull_request_reviews`: `0`, `pull_request_review_comments`: `0`, `releases`: `0`, `issue_comments`: `0`, `analysis_runs`: `0`, `analysis_run_attempts`: `0`.**

**Implication:** Out of the box, the local database cannot produce meaningful C.2 (`commits`), C.3 (`contributors`), C.4 (`pull_requests`), or C.6 (`releases`, `PAIR-01..03`) analytics for any of the 4 benchmark repositories without performing fresh live ingestion first. Only C.5 (`issue_intelligence`) has benchmark rows, and even those lack `CollectionRun` lineage (`representativeness = "unverified_no_collection_run"`).

### Critical Schema Migration Bug (`001_initial_benchmark_schema.py` vs. `models.py`)
Migration `001` creates different columns than `backend/app/database/models.py` on three tables:
- `issue_labels`: `001` defines `raw_label, normalized_label, color, description`, whereas `models.py` defines `label_name, is_canonical, canonical_category`.
- `repository_analytics`: `001` defines `closure_rate, category_bug_count, category_feature_count, ...`, whereas `models.py` defines `category_distribution, speed_tier_distribution, created_at`.
- `predictions`: `001` defines `model_version` and omits `input_title, input_body`, whereas `models.py` expects `input_title, input_body` and omits `model_version`.
Thus `alembic upgrade head` on a **fresh empty database** creates a schema incompatible with Phase A ORM models unless `init_db()` (`Base.metadata.create_all`) ran first and `001` was stamped. Also `pull_requests` in `models.py` omits `UniqueConstraint("repository_id", "pr_number")` even though Migration `002:195` declares `uq_pr_repo_number`.

---

## 6. GitHub Ingestion Pipeline & 12-Step Product Workflow Trace

Tracing the complete intended 12-step user workflow through actual files and line numbers:

1. **Step 1 — User submits a GitHub repository URL:**
   - Endpoint: `POST /api/v1/ingest/submit` (or `/api/ingest/submit`) in `backend/app/api/v1/endpoints/ingest.py`.
   - Gap: **Frontend does not call this endpoint yet** (`frontend/src/lib/api.ts` lacks any `/ingest` call). Only API/curl callers can initiate ingestion.
2. **Step 2 — Repository validation and eligibility:**
   - `GitHubURLValidator.validate_and_canonicalize()` (`validator.py:60-206`) normalizes URL/slug and filters hostnames/reserved paths.
   - `RepositoryEligibilityChecker.check_eligibility()` (`repository_eligibility.py:248-279`) calls GitHub `/repos/{owner}/{repo}`, enforces public & non-empty (`size > 0` KB or `commits_count >= 1`) gates, allows forks/archived repos with badges, and calls `ConfidenceMatrixEvaluator` (`confidence_matrix.py:251-318`).
   - Caveat: Unauthenticated calls without `GITHUB_TOKEN` hit GitHub's `60 req/hr` limit quickly; `RepositoryEligibilityResult` on submit checks eligibility with `open_issues_count` rather than actual commit/PR stream depth.
3. **Step 3 — Ingestion preview and submission:**
   - `POST /preview` returns metadata, both budget profiles (`10` vs `50` max requests), and preview confidence (`ingest.py:42-104`).
   - `POST /submit` (`ingest.py:107-213`) creates `Repository` row if absent using `result.github_id or (abs(hash(result.repo_name)) % 1000000000)` (`ingest.py:174`, hash salted per process if `github_id` missing in mocked/incomplete metadata!) and calls `DurableJobQueue.submit_job()` returning `201 Created` (new) or `200 OK` (active job reused).
4. **Step 4 — Durable ingestion worker execution:**
   - `IngestionWorker.process_one_job()` (`ingestion_worker.py:58-248`) recovers expired leases (`recover_expired_jobs`), claims next `pending` job (`claim_next_job`, `lease_version += 1`, `lease_duration=600s`), and starts `_heartbeat_loop()` (`15s` `renew_lease`).
   - **Critical Operational Gap:** `IngestionWorker` has **no `__main__` CLI entrypoint** and is **not launched in FastAPI `lifespan()`**, so submitted `CollectionRun` jobs stay `pending` forever unless a custom script runs `IngestionWorker.run_worker_loop(SessionLocal)`. Also `_heartbeat_loop` shares the same `db: Session` in asyncio while `collect_repository()` uses blocking SQLAlchemy calls on that `Session` (`ingestion_worker.py:141,164`).
5. **Step 5 — Entity normalization and database persistence:**
   - `RepositoryCollector.collect_repository()` (`repository_collector.py:132-396`) fetches metadata, commits + up to `commit_details` diffs (`0` in `fast_snapshot`, `5` in `standard_diagnostic`), contributors (`github_id` anchored), PRs, sampled PR reviews (`<=2` or `<=20`), inline PR comments (`0` in `fast_snapshot`), pure issues (filtering `"pull_request" in item`), issue/PR conversation comments (checked against `chk_issue_comment_parent`), releases, and calls `reconcile_contributor_metrics()`.
   - Tracks `mutated_records`; bumps `repositories.analytics_source_revision += 1` only when data changed (`mutated_records` non-empty) or historical rows trigger update.
   - Limitations: `collect_issues` never deletes removed raw `IssueLabel` rows; `collect_commits` diffs are severely sampled (`<= 5` commits in standard mode, `0` in fast mode); `AsyncGitHubClient.paginate()` blindly follows RFC 5988 `Link` `next` URLs without re-verifying host matches `api.github.com` (`github_client.py:206,464`).
6. **Step 6 — `CollectionRun` completion:**
   - `DurableJobQueue.complete_job()` (`job_queue.py:343-430`) enforces `lease_version` fencing and writes terminal status (`failed` if `summary.errors`, `rate_limited` if `resumes_at` set, `completed_partial` if `is_partial`, else `completed`) and writes dual-compatible top-level + `entity_status` keys inside `coverage_summary`.
   - Minor inconsistency: calling `RepositoryCollector.collect_repository(..., collection_run_id=id)` directly (`repository_collector.py:367-377`) writes only flat `entity_status` and checks `resumes_at` before `errors`.
   - Gap: `rate_limited` jobs record `resumes_at`, but **no daemon/scheduler automatically re-enqueues them** after `resumes_at` expires.
7. **Step 7 — Automatic analytics enqueue:**
   - `IngestionWorker._enqueue_post_ingestion_analytics()` (`ingestion_worker.py:249-347`) runs **after** `complete_job()` commits, strictly gates out failed (`summary.errors`), `rate_limited` (`resumes_at is not None`), uncompleted runs, and revoked-lease workers, and calls `AnalyticsJobQueue.enqueue_analysis()`. Failures roll back cleanly and can be recovered by `AnalyticsJobQueue.reconcile_analytics_jobs()`.
8. **Step 8 — `AnalyticsWorker` execution:**
   - `AnalyticsWorker` (`analytics_worker.py:31-232`) has both standalone CLI (`python -m backend.app.services.analytics_worker`) and FastAPI opt-in lifespan integration (`ENABLE_ANALYTICS_WORKER=true` in `main.py:62-77`). Periodically calls `reconcile_jobs()` (`60s` default) and `process_one_job()`.
   - Limitation: Unlike `IngestionWorker`, `AnalyticsWorker` does **not** run a lease-renewal heartbeat during `process_claimed_run()`, relying on the static `600s` lease window (`analytics_queue.py:181, orchestrator.py:928`).
9. **Step 9 — Six-domain orchestration:**
   - `AnalyticsOrchestrator.process_claimed_run()` (`orchestrator.py:844-1059`) opens an isolated read snapshot (`open_analysis_snapshot`, separate SQLite WAL connection in SQLite mode), pins `analytics_source_revision` and `analysis_reference_time` in UTC, executes C.2–C.5 + Release Evolution (`release_evolution`), passes precomputed results to `CrossDomainSynthesisCalculator` (`repository_synthesis`), and builds a single `OrchestratedAnalysisBundle`.
10. **Step 10 — `AnalysisPayload` validation:**
    - `build_publication_bundle()` (`orchestrator.py:460-723`) validates the 5-key root payload via Pydantic v2 `AnalysisPayload`, ensures `release_evolution` remains intact inside `repository_synthesis`, aggregates 6-domain `coverage_summary` and `confidence_summary`, and runs `consolidate_rules()`. Validation errors raise non-retryable failure (`NonRetryableAnalyticsError` -> immediate terminal `failed` state since commit `6d88338`).
11. **Step 11 — Durable publication:**
    - `process_claimed_run()` re-checks `Repository.analytics_source_revision`; if newer data committed mid-run, aborts as `status="stale"` (`_abort_superseded_stale_run`) rather than publishing stale results (`retry_count`/replacement handled). Otherwise `publish_and_finalize()` (`analytics_queue.py:438-559`) atomically persists `metrics_payload`, `coverage_summary`, `confidence_summary`, `rule_interpretations`, `source_content_digest`, updates `last_analyzed_at`, and closes the `AnalysisRunAttempt` row (`attempt_number`, `outcome`, immutable once completed).
12. **Step 12 — Modern API retrieval:**
    - Clients call `GET /api/v1/analytics/{owner}/{name}/status`, `/latest`, `/history`, `/runs/{run_id}`, or `/domains/{domain_key}` (`analytics.py:331-709`), receiving sanitized errors, `is_stale` freshness comparison against live `analytics_source_revision`, and full or single-domain metrics (with `release_evolution` seamlessly projectable from `domains/release_evolution`).
    - Gap: Case-sensitive legacy `/api/v1/repos/{owner}/{name}` and Phase 10 `frontend/` remain unintegrated with V2 analytics.

---

## 7. Durable Queue & Worker Architecture

- **Lease Mechanics:** Both queues use `worker_id`, `leased_until`, and `lease_version` monotonic counters (`collection_runs`, `analysis_runs`). SQL `UPDATE ... WHERE id = :id AND worker_id = :worker_id AND lease_version = :lease_version AND status = 'running'` enforces fencing (`rowcount == 0` -> lease lost / ghost rejected).
- **Retry Policies:**
  - `DurableJobQueue`: `max_retries = 2` (3 total attempts on lease expiration recovery); explicit `fail_job()` marks `failed` immediately.
  - `AnalyticsJobQueue`: `max_retries = 3` (4 total attempts). Orchestrator `_classify_exception()` (`orchestrator.py:277-292`) maps `ValidationError`, `InvalidParameterException`, `EntityNotFoundException`, and `NonRetryableAnalyticsError` to `failure_category="non_retryable"`, which since `6d88338` bypasses retries in `record_attempt_failure(retryable=False)` (`analytics_queue.py:561-665`). `attempt` rows store `error_category` and `interruption_reason`.
- **Reconciliation:** `reconcile_analytics_jobs()` sweeps expired leases (`interruption_reason="lease_expired"`), and recovers completed/partial `CollectionRun`s missing a corresponding `AnalysisRun`.

---

## 8. All Six Analytics Domains & Methodologies (Summary & Citations)

Detailed formulas, sample gates, confidence penalties, and rule IDs are documented in [docs/analytics.md](../analytics.md). Key audited highlights by module:
1. **`DevelopmentActivityCalculator` (`development_activity.py`, 1,891L):** Day-normalized rolling velocities (`28d/84d/364d`), completed-week consistency/CV (`>=4w, >=5 commits`), trailing 4w/12w rolling + EWMA (`alpha=0.4`), rolling MAD/z-score/Poisson anomaly detector (`>=8w, >=15 commits`, sparsity guard), `max_ssb_permutation_test` (`B=999`, `p<=0.01`, practical shift gates), 7x24 UTC rhythms, diff-aware churn excluding NULL-diff weeks from volatility, landing latency strictly requiring non-NULL `committer_date`.
2. **`ContributorIntelligenceCalculator` (`contributors.py`, 1,211L):** Author identity resolution anchored on immutable `github_id` and unlinked `git_email` without display-name guessing; bot exclusion & unattributed isolation (`>25%` penalty); Gini, 11-point Lorenz deciles, Shannon entropy, BF50/BF80; monthly retention cohorts gated strictly on verified `full_history`.
3. **`PullRequestIntelligenceCalculator` (`pull_requests.py`, 1,574L):** Separated `TTFResp` (all channels), `TTFR_formal`, `TTFR_inline`; self-interaction & bot filtering; merge & abandonment outcomes bounded by `ref_time`; merge latency percentiles preserving `0.0h`; Kaplan-Meier `S(7d)/S(30d)` (`>=15 PRs`); PR size bins distinguishing NULL diffs from `0` lines; `review_latency_share` and reviewer load concentration; stream-coverage flags via `CollectionRun`.
4. **`IssueIntelligenceCalculator` (`issues.py`, 1,286L):** Monthly creation vs closure flow (`scope_note` distinguishing sample flow from total backlog stock); GitHub `open_issues_count` stock disambiguation (`pure_issue_sample_coverage_percentage = None`); timestamp-recomputed SLAs, skewness & speed tiers; KM `S(30d/60d/90d)`; 5-bin backlog aging (`<7d, 7..30d, >30..90d, >90..365d, >365d`); comments correlation; `"unlabeled"` vs `"other"` separation.
5. **`ReleaseEvolutionCalculator` (`releases.py`, 596L):** Inter-release interval stats & `CV_release` (`>=3` releases); 14d/60d pre-release commit surges requiring `>=74d` commit baseline depth and non-zero baseline (`dormant_baseline_activity` isolated); SemVer 2.0.0 adoption (`Major/Minor/Patch/Prerelease`); rules `RULE-REL-EMPTY`, `RULE-REL-001..004` in `rule_engine.py`.
6. **`CrossDomainSynthesisCalculator` (`cross_domain.py`, 1,106L):** Non-composite diagnostic scorecard (`chi_deferred=True`), embedded `release_evolution`, `repo_context` (`last_synced_at` omitted), aligned monthly pairs (`PAIR-01..03`), rules `RULE-SYN-001..004`.

---

## 9. Statistical Reliability & Data Coverage Safeguards

The backend implements strong statistical hygiene:
- **Zero Non-Null Coercion:** NULL `additions`/`deletions`, NULL `committer_date`, and uncollected review streams never degrade silently to `0.0`. PAIR-01 only emits monthly churn when 100% of eligible commits in that month have non-NULL diff stats (`cross_domain.py:788-795`).
- **Temporal & Follow-up Consistency:** Ongoing calendar week is excluded from completed-week consistency/anomaly/churn-volatility baselines (`development_activity.py:289`); all queries clamp events at `<= analysis_reference_time`.
- **Stream-Coverage Guards:** C.4 & PAIR-02 inspect `CollectionRun.mode`, `is_partial`, and per-stream `coverage_summary` flags so skipped/sampled reviews (`fast_snapshot`, `<50%` coverage, missing reviews) suppress inferential correlation p-values (`status = "incomplete_review_coverage"`, `cross_domain.py:241,533-685`).
- **Bounded Collection Reality Check (Crucial Audit Note):** Under default `fast_snapshot` (10 reqs) and `standard_diagnostic` (50 reqs), `commit_details` is capped at `0` or `5` commits, `pr_reviews` at `2` or `20` PRs, and `pr_inline_comments` is skipped in `fast_snapshot`. Consequently, in real-world bounded ingestion on medium/large repos, `code_churn`, `PAIR-01`, `ttfr_formal`, `ttfr_inline`, `PAIR-02`, and `retention` cohorts will routinely report `insufficient`, `diff_metrics_not_collected`, `incomplete_review_coverage`, or `unavailable_requires_full_history_ingestion` unless `full_history` collection or higher budgets are implemented.

---

## 10. Cross-Domain Synthesis

`cross_domain.py` tests 3 calendar-month pairs with deterministic Monte Carlo rank permutations (`B = 10,000`, SHA-256 seed bound to repo/revision/metrics/methodology/B), consecutive-month lag-1 autocorrelation guard (`|r_AR1| > 0.40` -> `serial_dependence_detected`), minimum overlapping months (`K >= 12`), and Benjamini-Hochberg FDR adjustment (`alpha = 0.05`) across `tested` pairs. Findings (`RULE-SYN-002`) explicitly prohibit causal claims. Notice block-bootstrap confidence intervals remain deferred (`confidence_intervals_deferred=True`).

---

## 11. Analytics Orchestration & Publication

`AnalyticsOrchestrator` (`orchestrator.py`) guarantees intra-run snapshot consistency (shared `snapshot_revision`, shared UTC `analysis_reference_time`, reused C.2–C.6 outputs without double computation), strict Pydantic v2 `AnalysisPayload` validation, domain-weighted confidence scoring (`score = mean(1 - penalty_i)` across 6 domains, bands `high >= 0.80`, `medium >= 0.60`, `low >= 0.40`, `unreliable < 0.40` or all core empty/unavailable), deterministic rule consolidation, and race-safe stale-revision & lease-version fencing on commit (`publish_and_finalize`). Note: `methodology_version` currently only accepts `"2026.1"` (`analytics.py:131`, `orchestrator.py:265`).

---

## 12. Complete API Inventory

All 29 unique REST operations (31 OpenAPI operations due to `/api/ingest` + `/api/v1/ingest` dual mounting, plus `/docs`, `/redoc`, `/openapi.json`) are documented in [docs/api.md](../api.md). Highlights verified against `app.openapi()`:
- **System:** `GET /`, `GET /health`, `GET /api/v1/ping`
- **Ingestion (dual-mounted):** `POST /{prefix}/preview`, `POST /{prefix}/submit` (`201` new / `200` active), `GET /{prefix}/status/{job_id}`
- **Analytics V2:** `GET /api/v1/analytics/runs/{run_id}`, `POST /api/v1/analytics/{owner}/{name}/recompute` (`202` new or active / `200` cached), `GET /api/v1/analytics/{owner}/{name}/status|latest|history|runs/{run_id}|domains/{domain_key}`
- **Legacy Repo & Analytics:** `GET /api/v1/repos`, `/repos/{owner}/{name}`, `/repos/{owner}/{name}/issues`, `/repos/{owner}/{name}/issues/{number}`, `GET /api/v1/analytics/overview|comparison|{owner}/{name}|{owner}/{name}/trends`
- **Legacy ML:** `POST /api/v1/classify`, `POST /api/v1/related`, `GET /api/v1/related/{issue_id}`, `POST /api/v1/resolution/predict`, `GET /api/v1/resolution/stats/{owner}/{name}`, `GET /api/v1/models`, `GET /api/v1/models/{model_id}`, `POST /api/v1/models/prewarm`.

---

## 13. Configuration & Environment Variables

Inspected via AST/regex across `backend/`, `scripts/`, `ml/`, `.env.example`, and `frontend/.env.example`:
- **Active Backend Vars:** `DATABASE_URL`, `GITHUB_TOKEN`, `BACKEND_CORS_ORIGINS`, `PREWARM_ON_STARTUP`, `ENABLE_ANALYTICS_WORKER`, `ANALYTICS_WORKER_POLL_INTERVAL_SECONDS`, `ANALYTICS_WORKER_RECONCILE_INTERVAL_SECONDS`.
- **Config Drift Identified:**
  1. `.env.example` omits all 3 C.8.3 `ENABLE_ANALYTICS_WORKER*` / `ANALYTICS_WORKER_*` variables.
  2. `.env.example` defines `FASTAPI_HOST` and `FASTAPI_PORT`, neither of which is read by `backend/app/core/config.py` or `main.py` (`uvicorn.run` hardcodes `0.0.0.0:8000`).
  3. `pytest` has an undocumented opt-in flag `RUN_LIVE_GITHUB_TESTS=1`.

---

## 14. Local Development Setup

Verified on Windows (`.\.venv\Scripts\python.exe`, Python 3.14):
- **Important Packaging Gap:** Root `requirements.txt` and `backend/requirements.txt` together do **not** declare ML/scientific runtime dependencies (`torch`, `transformers`, `sentence-transformers`, `scikit-learn`, `scipy`, `numpy`, `httpx`, `joblib`, `pytest`), even though `backend/app/main.py` / services / analytics import `numpy`, `scipy`, `torch`, `transformers`, `sentence_transformers`, `scikit-learn`, `joblib`, and `httpx` (`analytics/orchestrator.py` and `cross_domain.py` import `numpy` & `scipy` directly). Installing only the two `requirements.txt` files in a clean environment will fail on startup/analytics unless ML packages are installed separately.
- **Windows Alembic Constraint:** `.\.venv\Scripts\alembic.exe` can be blocked by Windows Application Control policy; run `.\.venv\Scripts\python.exe -m alembic -c alembic.ini ...` cleanly.

---

## 15. Testing Strategy & Verification History

Verified directly in this audit session:
- **Total Collected Tests:** `1,079` tests across 93 test modules (`pytest --collect-only -q`).
- **Executed Backend Suite Run (This Session):** Running all 29 Phase B, Phase C, Alembic, Queue, Worker, Orchestrator, and FastAPI endpoint test files yielded **`388 passed, 3 skipped in 64.65s`** on SQLite.
- **Skipped Tests Verified (`-rs`):** The 3 skipped tests are live-network GitHub API integration tests (`tests/test_async_repository_collector.py:1320`, `tests/test_e2e_ingestion_lifecycle.py:283, 328`) gated behind `RUN_LIVE_GITHUB_TESTS=1`.
- **Remaining Suite Scope:** ~688 additional tests cover Phase 1–8 ML training, EDA, preprocessing, evaluation, and Phase 11 integration checks (`test_integration_*.py`, etc.), which were not re-executed in this backend audit run.
- **Critical Limitation:** Passing offline unit/SQLite tests with mocked `AsyncGitHubClient` responses does **NOT** prove production readiness under real GitHub rate-limiting, large-repository memory loads, or concurrent PostgreSQL workers.

---

## 16. Security & Privacy Review

- **No Authentication / Authorization:** All API endpoints (including `POST /ingest/submit`, `POST /analytics/{owner}/{name}/recompute`, `POST /models/prewarm`) are completely unauthenticated. Without a reverse proxy / auth middleware, any caller can enqueue GitHub ingestion jobs, trigger heavy ML model loading, or exhaust the server's `GITHUB_TOKEN` quota.
- **No Request Rate Limiting:** API endpoints lack per-IP or per-user throttling.
- **URL Validation & SSRF Risk:** `GitHubURLValidator` strictly enforces `github.com`/`www.github.com` hostnames (`validator.py:112`). However, `AsyncGitHubClient.paginate()` (`github_client.py:461-464`) passes `Link` header `next` URLs directly into `request()`, where `endpoint_or_url.startswith("https://")` (`github_client.py:206`) accepts arbitrary hosts without re-asserting `api.github.com` (currently trusting upstream GitHub responses, though mocked/MITM headers wouldn't be host-checked).
- **Error Sanitization:** `analytics.py` applies `_sanitize_public_error_message()` (`analytics.py:70-89`) to strip filesystem paths (`/home/...`, `C:\...`) and URIs (`*://...`) from V2 error fields. However, legacy handlers and unhandled detail strings elsewhere do not use this sanitizer, and `RepositoryIneligibleException` / transport exception messages can surface verbatim HTTP details.
- **Secrets & Privacy:** `.gitignore` excludes `.env`; audit verified no secrets/tokens appear in code. Git commit author emails (`author_email`, `git_email` in `commits` and `contributors`) are persisted in the database for unlinked identity reconciliation, though V2 contributor outputs only expose `label` / sanitized summaries when unlinked. Note Domain 1 UTC activity rhythms explicitly avoid local-timezone inference.

---

## 17. Error Handling & Recovery Mechanisms

- **API Exception Normalization:** `core/error_handlers.py` catches `OpenSourceLensException`, `HTTPException`, `RequestValidationError`, `SQLAlchemyError`, and generic `Exception`, mapping to consistent JSON envelopes.
- **Queue Recovery:**
  - `DurableJobQueue`: Expired running leases recover via `recover_expired_jobs()` up to `max_retries=2`; worker lease revocations abort execution via `JobLeaseLostException` before `complete_job` writes.
  - `AnalyticsJobQueue`: Distinguishes retryable runtime errors (`retry_count += 1` back to `pending` up to `max_retries=3`) from `non_retryable` schema/entity/parameter failures (immediate terminal `failed` state); `reconcile_analytics_jobs()` sweeps orphaned leases (`lease_expired`) and backfills missing post-ingestion analytics enqueues.
- **Recovery Gaps:**
  1. `rate_limited` `CollectionRun` jobs (`resumes_at` recorded) have **no automated delayed resumption**.
  2. `IngestionWorker` does not run periodically unless manually wrapped and executed externally.
  3. Because `AnalyticsWorker` lacks a background heartbeat, any run exceeding `600s` (e.g., very large synthetic/future dataset) would expire its lease cleanly even while actively running, risking duplicate claim followed by ghost fencing on commit.

---

## 18. Performance & Scalability Considerations

- **Full Snapshot In-Memory Loading:** All 6 analytics calculators load repository rows for the pinned `repository_id` into Python memory (`fetchall()`) per `AnalysisRun` (`development_activity.py`, `contributors.py`, `pull_requests.py`, `issues.py`, `cross_domain.py`). This is fast for budgeted `50`-request snapshots (`~100–500` items per stream), but **does not scale linearly to true `full_history` repositories with `50,000–500,000+` commits/issues/comments** without SQL aggregation or streaming chunking.
- **Monte Carlo Permutation Cost (`CrossDomainSynthesisCalculator` & `DevelopmentActivityCalculator`):** `B = 10,000` NumPy vectorized permutations (`CrossDomainSynthesisCalculator`, `cross_domain.py:287-289`) over `12–120` months is fast (~milliseconds per pair). `_run_max_ssb_permutation_test` (`B = 999`, Python loops or vectorized — `development_activity.py:1152`) across hundreds of weeks can be noticeably slower.
- **Per-Item N+1 Queries in Collector:** `RepositoryCollector` queries existence (`db.scalar(select(...))`) per commit, per contributor, per PR, per review, per inline comment, per issue, per label, and per issue comment inside Python loops, causing thousands of round-trips if scaled to `full_history` without bulk prefetch/upsert.
- **Static Embedding Load (`RelatedIssueService`):** Loads all `embeddings.npz` vectors into process RAM; concurrent requests remain synchronous in FastAPI worker threads.
- **ML Inference Blocking:** `classify` (`DistilBERT` PyTorch inference), `related`, and `resolution` endpoints run synchronous CPU/GPU compute inside sync FastAPI threadpool handlers; large `PREWARM_ON_STARTUP` increases baseline RAM footprint significantly.

---

## 19. SQLite vs. PostgreSQL Limitations

| Aspect | SQLite WAL (Verified Locally) | PostgreSQL 16 (Configured / Unverified Live) |
|---|---|---|
| **Queue Concurrency** | Single-writer file lock limits throughput; `sqlite3` `Session` cannot safely be shared across true OS threads. | `FOR UPDATE SKIP LOCKED` supports true horizontal multi-worker scaling, **but untested against live Postgres**. |
| **Driver Dependency** | Stdlib `sqlite3`. | `psycopg2-binary` is commented out in `backend/requirements.txt` and missing from `.venv`. |
| **Migration Reliability** | `data/opensource_lens.db` works via prior `create_all` + stamp; fresh `alembic upgrade head` fails Phase A ORM alignment. | **Fresh `alembic upgrade head` fails Phase A ORM alignment due to Migration `001` drift.** |
| **Docker Deployment** | Local file only. | `docker-compose.yml` only spins up Postgres; no FastAPI/worker Dockerfiles or Compose services exist. |

---

## 20. Production-Readiness Assessment

**Overall Production Readiness: Not Production Ready (Strong Research & Backend Core; Operational & UI Gaps Remain).**
- **Safe Today For:** Local single-user/developer execution, API testing with `SQLite WAL`, offline academic analytics research, and controlled single-process ingestion demos with a valid `GITHUB_TOKEN`.
- **Not Safe/Verified Today For:** Multi-tenant internet deployment, unattended horizontal multi-worker PostgreSQL clusters, `full_history` ingestion on mega-repositories, or browser UI workflows without completing Next.js V2 integration and security hardening.

---

## 21. Known Bugs, Gaps & Technical Debt Ledger

### Critical / High Priority
1. **Migration `001` Schema Drift vs. ORM (`backend/alembic/versions/001_initial_benchmark_schema.py` vs. `models.py`):** Fresh DB initialized solely by `alembic upgrade head` mismatches `issue_labels`, `repository_analytics`, and `predictions` ORM columns.
2. **Missing `IngestionWorker` CLI & Process Supervision (`backend/app/services/ingestion_worker.py`):** No `__main__` entrypoint or `lifespan()` integration; submitted ingestion jobs sit `pending` indefinitely without an external custom runner.
3. **Frontend Disconnected from Phase B & C.8.3 (`frontend/src/`):** Phase 10 UI calls legacy Phase A endpoints only; no dynamic URL ingestion or V2 6-domain visualization exists.
4. **Live PostgreSQL Multi-Worker Concurrency & Driver Missing:** `psycopg2-binary` absent from runtime deps; `FOR UPDATE SKIP LOCKED` & Docker full-stack never run live.
5. **No API Authentication or Rate Limiting:** Unprotected `POST /submit`, `/recompute`, and `/prewarm`.

### Medium / Low Priority
6. **`_VALID_RUN_STATUSES` Rejects `"stale"` (`backend/app/api/v1/endpoints/analytics.py:581`):** `AnalyticsOrchestrator` writes `AnalysisRun.status = "stale"` on mid-run revision superseding, yet `GET /history?status=stale` fails validation.
7. **No Scheduler for `rate_limited` Ingestion Jobs:** `CollectionRun(status="rate_limited", resumes_at=...)` stays paused indefinitely without manual/cron intervention.
8. **`requirements.txt` Incomplete for Analytics/ML:** `numpy`, `scipy`, `torch`, `transformers`, `sentence-transformers`, `scikit-learn`, `joblib`, and `httpx` are missing from root + `backend/requirements.txt`.
9. **Legacy Case-Sensitive Lookups (`repos.py`, legacy `analytics.py`) vs. Case-Insensitive V2 (`_resolve_repository_ci`)** and **`Repository.id` Fallback Hash (`ingest.py:174`)**: Non-deterministic across Python restarts if `github_id` is absent.
10. **Collector `coverage_summary` Path Divergence (`repository_collector.py:367-377`)**: Direct calls without `DurableJobQueue.complete_job()` omit nested `"entity_status"` and prioritize `resumes_at` ahead of `errors`.
11. **Missing `UniqueConstraint("repository_id", "pr_number")` in `PullRequest` ORM (`models.py`)** despite existence in Migration `002:195`.
12. **Phase A ML Benchmark Coupling:** `RelatedIssueService` only searches static 2,328-issue `embeddings.npz`; `ResolutionService` expects benchmark `repo_name`; `.env.example` omits C.8.3 worker vars and includes unused `FASTAPI_HOST`/`FASTAPI_PORT`; `AnalyticsWorker` lacks active heartbeat renewal during execution; `full_history` budget mode is unimplemented in `BUDGET_CONFIGS`.

---

## 22. Unverified Behavior & Required Experiments

1. **Live Unseen-Repo End-to-End Ingestion + Analytics:** Enable `RUN_LIVE_GITHUB_TESTS=1` with `GITHUB_TOKEN` and execute full 12-step ingestion -> auto-enqueue -> `AnalyticsWorker` -> `/latest` retrieval across diverse small, medium, and large unseen public repositories, inspecting actual subdomain coverage statuses under `fast_snapshot` and `standard_diagnostic`.
2. **Live PostgreSQL Multi-Worker Concurrency Benchmark:** Provision Postgres 16 (`docker-compose.yml` + `psycopg2-binary`), test fresh DB migration reconciliation, spawn `N >= 3` concurrent `IngestionWorker` and `AnalyticsWorker` processes, and verify `FOR UPDATE SKIP LOCKED`, partial unique indexes, lease recovery, and concurrent revision-bump stale abort behavior under stress.
3. **Out-of-Distribution ML Generalization Verification (Phase 14-R):** Measure DistilBERT category precision/recall and decoupling strategies on newly ingested repositories outside the 4 benchmark repos, and design incremental dynamic embeddings beyond `embeddings.npz`.

---

## 23. Recommended Next Steps

1. **Immediate Correctness Fixes:** Reconcile `001_initial_benchmark_schema.py` (or add a clean reconciliation migration / test fresh Postgres `alembic upgrade head` end-to-end), add `__main__` CLI + optional lifespan wiring to `IngestionWorker`, add `"stale"` to `_VALID_RUN_STATUSES`, add `rate_limited` auto-resumption polling, and unify `requirements.txt` / `.env.example`.
2. **Live PostgreSQL & GitHub Verification:** Install PostgreSQL driver, run `RUN_LIVE_GITHUB_TESTS=1`, and stress-test multi-worker PostgreSQL queue claims & stale-revision superseding.
3. **Dynamic ML Decoupling:** Generate embeddings for newly ingested issues cleanly and remove benchmark repository leakage from resolution advisory models.
4. **Frontend V2 Workspace Integration:** Build repository submission/progress views (`POST /api/v1/ingest/preview|submit`, `GET /status/{job_id}`) and 6-domain analytics tabs backed by `/api/v1/analytics/{owner}/{name}/*`.
5. **Production Hardening:** Add reverse-proxy auth / per-client rate limiting, `AnalyticsWorker` lease heartbeat or chunked snapshot processing, and Dockerfiles/Compose services for full stack.

---

## 24. Final System Assessment

**OpenSourceLens possesses a high-rigor, scientifically principled Data Science and durable job orchestration backend (Phases B & C.1–C.8.3), verified across 388 offline SQLite tests.** Its statistical guardrails—explicit unavailable states, strict NULL-vs-zero separation, snapshot freezing, full-history gating on retention, AR(1) and review-coverage guards on cross-domain tests, and optimistic `lease_version` + `analytics_source_revision` fencing—exceed typical CRUD dashboard quality. However, bridging the final mile to a production platform requires resolving fresh-DB Migration `001` drift, adding standalone CLI/process supervision for `IngestionWorker`, upgrading the Phase 10 Next.js frontend to consume V2 analytics and ingestion APIs, decoupling Phase A ML models from static benchmark artifacts, and running live validation against real GitHub repositories and PostgreSQL concurrency.

---

## Appendix: Outdated & Superseded Documentation Audit

Prior to this refresh, documentation suffered from severe pre-implementation drift:
- **`README.md` (Replaced):** Listed `11-R` as "Next", referenced non-existent `LICENSE` and 9 missing `scripts/verify_phase11_*.py` filenames, and omitted workers and V2 routes.
- **`.env.example` (Untouched by Read-Only Constraint — Documented as Drift):** Missing `ENABLE_ANALYTICS_WORKER*` / `ANALYTICS_WORKER_*` vars; lists unread `FASTAPI_HOST` / `FASTAPI_PORT`.
- **`docs/0.7-architecture.md` & `docs/roadmap.md`:** Referenced nonexistent planned paths (`/api/repos/analyze`, `activity_service.py`, `RepositorySearchBar.tsx`, etc.). **Action taken:** Preserved original planning content with non-destructive status banners in `0.7-architecture.md` and updated `roadmap.md` with actual Track B/Phase C status.
- **`docs/database_schema_specification.md` & `docs/0.3-data-definition.md`:** Omitted 11 Phase B/C tables and listed draft column names. **Action taken:** Added historical status banners linking to [docs/database.md](../database.md).
- **`docs/0.2-project-features.md`, `0.4-repository-selection.md`, `0.5-ml-problem-definition.md`:** Contained superseded class counts (5 vs 3 classes), early eligibility gates (`>= 20` commits vs decoupled `>= 1`), and early z-score threshold (`|z| > 2.5` vs implemented rolling robust MAD + max-SSB). **Action taken:** Added clear alignment banners linking to [docs/analytics.md](../analytics.md) and [docs/ingestion.md](../ingestion.md).
- **`docs/0.1-project-definition.md`, `0.6-methodology.md`, `0.8-stage-1-proposal.md`, `docs/phase_a_completion_report.md`, `docs/resolution_prediction_feasibility_report.md`, `data/README.md`:** Accurately describe project vision, Track A research, and completed Phase A / Phase 8 milestones; retained as valid contextual/academic documentation.
