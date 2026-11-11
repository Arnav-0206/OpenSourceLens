# OpenSourceLens — System Architecture

OpenSourceLens is a Data Science-powered GitHub Repository Intelligence Platform. The backend is a multi-stage asynchronous pipeline built on **FastAPI**, **SQLAlchemy 2.0**, **Alembic**, **scikit-learn / PyTorch / Sentence-Transformers**, and a six-domain statistical analytics engine (`methodology_version = "2026.1"`).

> **Note on current system state:** The **Phase B** ingestion subsystem and **Phase C (C.1–C.8.3)** six-domain analytics subsystem are implemented in the backend (`backend/`). The Next.js frontend in [`frontend/`](../frontend/) remains at its Phase 10 baseline and still targets legacy Phase A repository/ML endpoints rather than the ingestion or Analytics V2 APIs. See [OPENSOURCELENS_BACKEND_AUDIT.md](./audits/OPENSOURCELENS_BACKEND_AUDIT.md) for the complete gap analysis.

---

## 1. End-to-End Architecture Diagram

```mermaid
flowchart TD
    Client["API Consumer / Future UI"] -->|POST preview / submit| FastAPI["FastAPI Application (backend/app/main.py)"]
    Client -->|POST recompute / GET status, latest, history, domains| FastAPI
    Client -->|Legacy GET repos, analytics, POST classify, related, resolution| FastAPI

    subgraph IngestionFlow["Phase B: Dynamic GitHub Ingestion"]
        FastAPI --> Validator["GitHubURLValidator"]
        Validator --> Eligibility["RepositoryEligibilityChecker + ConfidenceMatrixEvaluator"]
        Eligibility -->|enqueue CollectionRun| IngQueue["DurableJobQueue (collection_runs)"]
        IngQueue --> IngWorker["IngestionWorker (heartbeat + lease_version fencing)"]
        IngWorker --> GHClient["AsyncGitHubClient + EntityRateBudgeter"]
        GHClient <-->|REST API v3| GitHub["api.github.com"]
        GHClient --> Collector["RepositoryCollector (normalization + mutation tracking)"]
    end

    Collector -->|atomic source_revision increment| DB[(SQLite WAL / PostgreSQL)]
    IngWorker -->|complete CollectionRun + auto-enqueue| AnaQueue["AnalyticsJobQueue (analysis_runs / attempts)"]

    subgraph AnalyticsFlow["Phase C: Multi-Domain Analytics & Publication"]
        FastAPI -->|POST recompute| AnaQueue
        FastAPI -.->|opt-in ENABLE_ANALYTICS_WORKER| AnaWorker["AnalyticsWorker (poll + reconcile)"]
        AnaQueue --> AnaWorker
        AnaWorker --> Orchestrator["AnalyticsOrchestrator"]
        Orchestrator -->|pinned snapshot + frozen ref_time| D1["C.2 Development Activity"]
        Orchestrator --> D2["C.3 Contributors"]
        Orchestrator --> D3["C.4 Pull Requests"]
        Orchestrator --> D4["C.5 Issues & Maintenance"]
        Orchestrator --> D5["C.6 Release Evolution"]
        D1 & D2 & D3 & D4 & D5 --> D6["C.6 Cross-Domain Synthesis (Scorecard + Spearman/BH FDR)"]
        D6 --> PayloadVal["AnalysisPayload (Pydantic v2) + ConfidenceSummary + consolidate_rules"]
        PayloadVal -->|lease + source_revision fenced commit| DB
    end

    subgraph PhaseAML["Phase A ML Services (Offline Benchmark Artifacts)"]
        MLMgr["ModelManager"] --> Cls["ClassificationService (DistilBERT, TF-IDF LR/SVM)"]
        MLMgr --> Rel["RelatedIssueService (all-MiniLM-L6-v2 + static embeddings.npz)"]
        MLMgr --> Res["ResolutionService (RF Champion Classifier)"]
    end

    FastAPI --> DB
    FastAPI --> MLMgr
```

---

## 2. FastAPI Application Structure

The FastAPI application is constructed in [`backend/app/main.py`](../backend/app/main.py) (`create_application()`, `lifespan()`):

- **Configuration & Error Envelopes:** [`backend/app/core/config.py`](../backend/app/core/config.py) reads CORS and optional worker settings; [`backend/app/core/error_handlers.py`](../backend/app/core/error_handlers.py) and [`backend/app/core/exceptions.py`](../backend/app/core/exceptions.py) normalize domain errors, SQLAlchemy failures, HTTP exceptions, and validation errors into structured `ErrorResponse` / `ValidationErrorResponse` JSON envelopes.
- **Router Mounting:**
  - `GET /` and `GET /health` live directly on `app`.
  - [`backend/app/api/v1/api.py`](../backend/app/api/v1/api.py) mounts `/api/v1/ping`, `/api/v1/repos`, `/api/v1/analytics`, `/api/v1/classify`, `/api/v1/related`, `/api/v1/resolution`, `/api/v1/models`, and `/api/v1/ingest`.
  - [`backend/app/main.py`](../backend/app/main.py#L124) also mounts `ingest_router` directly at `/api/ingest`.
- **Startup Lifespan:** Verifies DB readiness, runs `Base.metadata.create_all(bind=engine)` via `init_db()`, optionally prewarms Phase A ML artifacts if `PREWARM_ON_STARTUP=true`, and optionally starts an in-process background `AnalyticsWorker` task when `ENABLE_ANALYTICS_WORKER=true`. Note that `IngestionWorker` is **not** started inside `lifespan()`.

---

## 3. Repository Ingestion Pipeline

1. **URL Canonicalization:** [`GitHubURLValidator.validate_and_canonicalize()`](../backend/app/services/validator.py#L60-L206) parses full URLs, scheme-less URLs (`github.com/owner/repo`), or slugs (`owner/repo`), rejecting reserved GitHub paths, scoped npm-style names (`@org/pkg`), non-GitHub hosts, and malformed owner/repo names.
2. **Pre-Flight Eligibility & Confidence Preview:** [`RepositoryEligibilityChecker`](../backend/app/services/repository_eligibility.py#L30-L279) fetches `/repos/{owner}/{repo}`, gates out private and empty (`size == 0` / `< 1` commit) repositories while allowing forks and archived repos with informational badges, and computes an initial sample sufficiency preview via [`ConfidenceMatrixEvaluator`](../backend/app/services/confidence_matrix.py#L25-L318).
3. **GitHub Client & Budgeting:** [`AsyncGitHubClient`](../backend/app/services/github_client.py#L44-L491) enforces a per-instance concurrency semaphore (`max_concurrency=2`), `100 ms` inter-request pacing, primary rate-limit threshold halting (`X-RateLimit-Remaining < 10`), `Retry-After` handling (`<= 300s` async sleep, `> 300s` structured abort with `resumes_at`), RFC 5988 `Link` pagination, and strict quota checks against [`EntityRateBudgeter`](../backend/app/services/rate_budgeter.py#L53-L183) (`10` requests in `fast_snapshot`, `50` in `standard_diagnostic`).
4. **Normalization & Persistence:** [`RepositoryCollector`](../backend/app/services/repository_collector.py#L118-L1703) harvests metadata, commits (+ optional diff stats within budget), contributors, pull requests, sampled PR reviews, inline PR comments, pure issues (filtering `"pull_request" in item`), issue/PR conversation comments (enforcing mutual parent exclusion), releases, and derived contributor aggregates. It tracks exact `mutated_records` and atomically increments `repositories.analytics_source_revision` whenever analytical source rows change.

See [ingestion.md](./ingestion.md) for the full 12-stage lifecycle, collection modes, and partial-collection semantics.

---

## 4. Durable Queues & Workers

OpenSourceLens uses two SQL-backed durable job queues with monotonic `lease_version` fencing:

- **Ingestion Queue (`DurableJobQueue` + `IngestionWorker`):** Manages `collection_runs` rows (`pending`, `running`, `completed`, `completed_partial`, `rate_limited`, `failed`). `IngestionWorker` renews leases every `15s` inside `process_one_job()` and, upon clean completion (`completed` or `completed_partial` with empty `summary.errors` and `resumes_at is None`), triggers `_enqueue_post_ingestion_analytics()`.
- **Analytics Queue (`AnalyticsJobQueue` + `AnalyticsWorker`):** Manages `analysis_runs` and append-only `analysis_run_attempts` rows (`pending`, `running`, `completed`, `failed`, `stale`). Supports active-run partial unique index deduplication, stale-revision superseding, lease expiration recovery via `reconcile_analytics_jobs()`, and non-retryable terminal failure handling.

See [workers-and-deployment.md](./workers-and-deployment.md) and [database.md](./database.md) for lease rules, SQLite vs. PostgreSQL concurrency, and execution modes.

---

## 5. Analytics Orchestration, Six Domains & Synthesis

[`AnalyticsOrchestrator`](../backend/app/services/analytics/orchestrator.py#L256-L1140) coordinates execution for a claimed `AnalysisRun`:

1. Opens a snapshot transaction (`AnalyticsJobQueue.open_analysis_snapshot()`) pinning `analytics_source_revision` and freezing `analysis_reference_time` in UTC.
2. Executes five base calculators implementing [`BaseMetricCalculator`](../backend/app/services/analytics/base.py#L32-L66):
   - `DevelopmentActivityCalculator` (C.2)
   - `ContributorIntelligenceCalculator` (C.3)
   - `PullRequestIntelligenceCalculator` (C.4)
   - `IssueIntelligenceCalculator` (C.5)
   - `ReleaseEvolutionCalculator` (C.6)
3. Passes the precomputed domain results into `CrossDomainSynthesisCalculator` (C.6), which avoids recomputation while computing the Multi-Domain Diagnostic Scorecard, paired calendar-month Spearman Monte Carlo permutation tests (`B=10,000`, `K>=12`, `|r_AR1| > 0.40` suppression, Benjamini-Hochberg FDR `alpha=0.05`), and cross-domain rules.
4. Embeds `release_evolution` inside `repository_synthesis` to preserve the 5-key [`AnalysisPayload`](../backend/app/schemas/analytics_v2.py#L59-L66) Pydantic v2 contract, computes aggregated `coverage_summary` and `confidence_summary`, consolidates rules via [`consolidate_rules()`](../backend/app/services/analytics/rule_engine.py#L249-L276), and commits atomically via `publish_and_finalize()`.

See [analytics.md](./analytics.md) for exact formulas, thresholds, statistical guards, and rule IDs.

---

## 6. Phase A Pre-Trained Machine Learning Services

Separate from the Phase C statistical calculators, the API exposes offline-trained ML/NLP artifacts built on the 4-repository benchmark corpus (2,328 issues) via [`ModelManager`](../backend/app/services/model_manager.py#L27-L210):

- **Issue Categorization (`ClassificationService`):** Fine-tuned DistilBERT (`models/distilbert_finetuned`, 3 classes: `bug`, `feature`, `other`), TF-IDF Logistic Regression, and TF-IDF Linear SVM.
- **Semantic Related-Issue Search (`RelatedIssueService`):** Queries precomputed relational cache edges (`related_issues_cache`, 11,640 edges) or runs `all-MiniLM-L6-v2` cosine similarity against the static `models/related_issues/embeddings.npz` benchmark store (`2,328` issues — does **not** index newly ingested repositories yet).
- **Resolution Velocity Advisory (`ResolutionService`):** Random Forest 4-tier classifier (`< 1 Day`, `1-7 Days`, `7-30 Days`, `30+ Days`) using `t=0` features and benchmark-trained encodings, returning adjacent `±1` tier error bounds and mandatory advisory disclaimers.

---

## 7. API Response & Publication Flow

FastAPI serves both **Phase A legacy analytics endpoints** (reading the legacy `repository_analytics` summary table) and **Phase C Analytics V2 endpoints** under `/api/v1/analytics/{owner}/{name}/...` (`recompute`, `status`, `latest`, `history`, `runs/{run_id}`, `domains/{domain_key}`) backed directly by `AnalysisRun` and `AnalysisRunAttempt`. See [api.md](./api.md) for complete endpoint contracts and status codes.
