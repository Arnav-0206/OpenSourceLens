# OpenSourceLens — Master Execution Roadmap

This roadmap defines the complete project execution plan, structured into two core evolutionary tracks:
1. **Phases 0–11: Benchmark Lab Foundation (Completed)** — Exhaustive offline data collection, exploratory data analysis, classical ML baseline development, transformer fine-tuning, dense vector semantic retrieval, database architecture, and initial web UI integration across four diverse benchmark repositories (`microsoft/vscode`, `pytorch/pytorch`, `kubernetes/kubernetes`, `huggingface/transformers`).
2. **Phases 11-R–16-R: Universal Repository Intelligence Platform (Realigned Execution Plan)** — Architectural extension to dynamically ingest, analyze, and generate multi-domain intelligence for **any eligible public GitHub repository** entered by a user.

---

## 🏛️ Track A: Benchmark Lab Foundation (Completed)

### Phase 0 — Project Definition & Planning ✅
- **0.1** Freeze the problem & Universal Vision (`docs/0.1-project-definition.md`)
- **0.2** Define Core Features & Analytical Domains (`docs/0.2-project-features.md`)
- **0.3** Multi-Entity Data Schema Definition (`docs/0.3-data-definition.md`)
- **0.4** Benchmark Corpus Selection & Generalization Strategy (`docs/0.4-repository-selection.md`)
- **0.5** Data Science & ML Problem Formalization (`docs/0.5-ml-problem-definition.md`)
- **0.6** Dual-Track Methodology Specification (`docs/0.6-methodology.md`)
- **0.7** System Architecture Design (`docs/0.7-architecture.md`)
- **0.8** Project Proposal Specification (`docs/0.8-stage-1-proposal.md`)

### Phase 1 — Benchmark Data Source & Collection Pipeline ✅
- **1.1–1.10** Built modular collectors (`repository_collector.py`, `issue_collector.py`, `comment_collector.py`, `event_collector.py`), pagination engine (`paginator.py`), resilient GitHub client with exponential backoff (`github_client.py`), and raw data storage (`data/raw/`).

### Phase 2 — Data Cleaning & Dataset Construction ✅
- **2.1–2.10** Built raw data loaders, markdown/code/URL text strippers, timestamp processors, canonical label normalizers (`bug`, `feature`, `other`), feature extractors, and processed dataset storage (`data/processed/`).

### Phase 3 — Exploratory Data Analysis (EDA) ✅
- **3.1–3.11** Performed distribution analysis, label balance inspection, time-series creation/closure analysis, cross-repository velocity comparisons, comment patterns, and Kaplan-Meier survival curves (`ml/eda/FINDINGS.md`).

### Phase 4 — Classical Machine Learning Baseline ✅
- **4.1–4.7** Established TF-IDF feature extraction (10,000 max features, 1-2 n-grams), trained Logistic Regression and Linear SVM baselines, achieving 75.6%–77.1% accuracy.

### Phase 5 — NLP Enhancement ✅
- **5.1–5.5** Implemented sublinear TF scaling, noise regex filtering, and combined title+body representations, establishing an enhanced classical pipeline.

### Phase 6 — Deep Learning & Transformer Fine-Tuning ✅
- **6.1–6.8** Fine-tuned `distilbert-base-uncased` (66.9M parameters) on curated issue sequences, achieving **81.3% test accuracy** and **0.79 macro-F1**, outperforming classical baselines across all minority classes. Saved model artifacts in `models/distilbert_finetuned/`.

### Phase 7 — Related-Issue Semantic Retrieval ✅
- **7.1–7.7** Implemented dense vector embeddings via Sentence Transformers (`all-MiniLM-L6-v2`), generated 384-dimensional unit-normalized embeddings, calibrated cosine similarity thresholds ($\ge 0.70$ related, $\ge 0.85$ duplicate), and built retrieval pipeline.

### Phase 8 — Resolution Analysis & Velocity Prediction ✅
- **8.1–8.7** Analyzed continuous turnaround distributions, implemented Random Forest velocity classification across SLA tiers, evaluated feature importance, and produced the formal feasibility report (`docs/resolution_prediction_feasibility_report.md`).

### Phase 9 — Relational Database & Backend API ✅
- **9.1–9.12** Designed dual-dialect schema (PostgreSQL + SQLite WAL), built SQLAlchemy 2.0 ORM models, created FastAPI REST endpoints, and verified with 73 unit tests.

### Phase 10 — Frontend UI (Next.js 15) ✅
- **10.1–10.12** Created Next.js 15 App Router interface with Tailwind CSS, Recharts visualizations, issue explorer with multi-facet filters, and interactive AI workspace.

### Phase 11 — Full Integration & Verification ✅
- **11.1–11.9** Connected frontend ↔ backend ↔ database ↔ ML models. Verified with 145 backend integration tests and zero-error Next.js production compilation.

---

## 🚀 Track B: Realigned Universal Platform Evolution (Actual Implementation Status)

> [!NOTE]
> The early pre-implementation draft sections (11-R–16-R below) were superseded in execution by **Phase A**, **Phase B (Dynamic GitHub Ingestion)**, and **Phase C (C.1–C.8.3 Six-Domain Analytics Engine, Worker & V2 API)**. See the completed ledger below and [OPENSOURCELENS_BACKEND_AUDIT.md](./audits/OPENSOURCELENS_BACKEND_AUDIT.md).

### Implemented Track B Milestones

| Milestone | Scope Implemented | Status | Documentation |
|---|---|---|---|
| **Phase A** | Obsolete 4-repo frontend hardcoding cleanup (`ed93489`); baseline preserved at `checkpoint/pre-cleanup-baseline` (`7fb1c67`) | ✅ Complete | [phase_a_completion_report.md](./phase_a_completion_report.md) |
| **Phase B** | URL validation, eligibility checker, rate-budgeted `AsyncGitHubClient`, multi-entity `RepositoryCollector` + revision tracking, `DurableJobQueue`, `IngestionWorker`, `/api/ingest/*` & `/api/v1/ingest/*`, Migration `002` | ✅ Complete | [ingestion.md](./ingestion.md), [database.md](./database.md) |
| **Phase C.1** | `analysis_runs`, `analysis_run_attempts`, `AnalyticsJobQueue`, snapshot isolation protocol, Migration `003` | ✅ Complete | [database.md](./database.md) |
| **Phase C.2** | Development Activity Intelligence (`velocity`, `consistency`, `rolling_statistics`, `anomalies`, `max_ssb` change-points, UTC rhythms, diff-aware churn, commit landing latency) | ✅ Complete | [analytics.md §2](./analytics.md#2-domain-1-development-activity-development_activity) |
| **Phase C.3** | Contributor Intelligence (`github_id` actor hygiene, Gini, 11-point Lorenz deciles, Shannon entropy, BF50/BF80, full-history retention gate) | ✅ Complete | [analytics.md §3](./analytics.md#3-domain-2-contributor-ecosystem-contributor_ecosystem) |
| **Phase C.4** | Pull Request Intelligence (`TTFResp`, `TTFR_formal`, `TTFR_inline`, outcomes, merge latency, Kaplan-Meier survival, diff size bins, bottlenecks) | ✅ Complete | [analytics.md §4](./analytics.md#4-domain-3-pull-request-intelligence-pull_request_intelligence) |
| **Phase C.5** | Issue & Maintenance Intelligence (monthly flow, backlog stock separation, SLA percentiles, KM survival, 5-bin backlog aging, discussion Spearman, category hygiene) | ✅ Complete | [analytics.md §5](./analytics.md#5-domain-4-issue--maintenance-intelligence-issue_intelligence) |
| **Phase C.6** | Release Evolution (cadence/CV, 14d/60d pre-release surges, SemVer) & Cross-Domain Synthesis (diagnostic scorecard, paired Spearman + `B=10,000` permutation, AR(1) & review coverage guards, BH FDR) | ✅ Complete | [analytics.md §6–7](./analytics.md#6-domain-5-release-evolution-release_evolution) |
| **Phase C.7** | `AnalyticsOrchestrator` snapshot consistency, `AnalysisPayload` validation, confidence aggregation, deterministic `consolidate_rules`, non-retryable terminal queue enforcement (`6d88338`) | ✅ Complete | [analytics.md §1](./analytics.md#1-orchestration-validation--publication-flow) |
| **Phase C.8.1–C.8.3** | `AnalyticsWorker` + CLI & lifespan integration, post-ingestion analytics enqueue wiring, V2 endpoints (`/recompute`, `/status`, `/latest`, `/history`, `/runs/{id}`, `/domains/{key}`) (`9f394e6`) | ✅ Complete | [api.md](./api.md), [workers-and-deployment.md](./workers-and-deployment.md) |
| **Doc Refresh & Audit** | Architecture, DB, Ingestion, Analytics, API, Workers docs + comprehensive backend audit | ✅ Complete | [OPENSOURCELENS_BACKEND_AUDIT.md](./audits/OPENSOURCELENS_BACKEND_AUDIT.md) |

### Remaining Work & Planned Phases (Not Yet Implemented)

1. **Schema & Operational Fixes:** Reconcile Migration `001` column drift against Phase A ORM models, add `IngestionWorker` CLI entrypoint and opt-in lifespan/runner wiring, fix `_VALID_RUN_STATUSES` `"stale"` filter & legacy route case-insensitive resolution, and add `rate_limited` job resumption scheduling.
2. **Dynamic ML Decoupling:** Vectorize newly ingested issues into queryable embeddings (instead of static `models/related_issues/embeddings.npz`) and decouple resolution advisory prediction from benchmark repo encodings.
3. **Frontend V2 Integration (Track B UI):** Connect [`frontend/`](../frontend/) (currently Phase 10 legacy) to `/api/v1/ingest/*` and `/api/v1/analytics/{owner}/{name}/*` V2 views with interactive multi-domain charts and diagnostics.
4. **Live Verification & Deployment:** Execute live multi-worker PostgreSQL concurrency tests, run `RUN_LIVE_GITHUB_TESTS=1` end-to-end tests against real unseen public GitHub repos, and containerize FastAPI, workers, PostgreSQL, and Next.js.

---

<details>
<summary>Original Track B Pre-Implementation Planning Reference (Historical Phases 11-R–16-R)</summary>

### Phase 11-R — Universal Ingestion Engine & Schema Migration (Historical Draft)
**Goal:** Enable OpenSourceLens to ingest commits, contributors, pull requests, and issues dynamically for any valid public repository entered by a user.

| Step | Title | Description | Original Planned Deliverable |
|---|---|---|---|
| **11-R.1** | Repository URL Validator | Parse GitHub URLs, extract `{owner}/{repo}`, and verify public existence via API | `backend/app/services/validator.py` *(implemented)* |
| **11-R.2** | Multi-Entity Collection Pipeline | Collect commits, contributors, PRs, reviews, issues, comments, and releases | Planned `ml/data_collection/dynamic_collector.py` *(implemented in `backend/app/services/repository_collector.py`)* |
| **11-R.3** | Tiered Rate-Limit Budgeter | Quota allocation (`fast_snapshot` vs `standard_diagnostic`) | Planned `budgeter.py` *(implemented in `rate_budgeter.py`)* |
| **11-R.4** | Relational Schema Migration | Add `commits`, `contributors`, `pull_requests`, queues | `backend/alembic/versions/002_*`, `003_*` *(implemented)* |
| **11-R.5** | Dynamic Ingestion API Endpoint | Preview, submit, and status tracking | Planned `POST /api/repos/analyze` *(implemented at `/api/ingest/*` & `/api/v1/ingest/*`)* |

### Phase 12-R–16-R Summary (Original Draft Names vs. Reality)
- **12-R Engines:** Implemented in `backend/app/services/analytics/` (`development_activity.py`, `contributors.py`, `pull_requests.py`, `issues.py`, `releases.py`, `cross_domain.py`, `orchestrator.py`) instead of `activity_service.py` / `pr_service.py` / etc. Dynamic semantic vector caching and resolution advisory decoupling remain pending.
- **13-R Frontend Workspace:** Pending migration from Phase 10 legacy pages.
- **14-R Live Generalization & E2E:** Unit/mocked integration suites pass offline; live unseen-repo GitHub generalization runs remain pending.
- **15-R Deployment Packaging:** Only `postgres:16-alpine` exists in `docker-compose.yml`; service containers and live PostgreSQL worker tests remain pending.
- **16-R Academic Evaluation & Defense:** Pending live multi-repo benchmark expansion.

</details>
