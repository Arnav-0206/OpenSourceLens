# OpenSourceLens — Master Execution Roadmap

**Product Vision:** See [docs/product-vision.md](./product-vision.md). OpenSourceLens is a universal **GitHub Repository Intelligence and Analytics Platform** that harvests rich public GitHub API data for **any eligible public repository** and turns it into statistical, ML/NLP, and visual intelligence—excluding static source-code/AST analysis, CI/CD logs, dependency/security scanning, and chatbots.

This roadmap is organized **cleanup-first, then feature-by-feature in demoable stages**, so each completed milestone delivers a concrete, presentable capability.

---

## 0. Current Baseline (Already Implemented — Retain & Build On)

The backend core from recent phases is complete on SQLite (PostgreSQL execution unverified live):

| Layer | Implemented Foundation | Key Locations |
|---|---|---|
| **Dynamic Ingestion (Phase B)** | URL validation, public/non-empty eligibility gating, preview & submit APIs (`/api/v1/ingest/*`), rate-budgeted `AsyncGitHubClient` (`fast_snapshot` / `standard_diagnostic`), multi-entity `RepositoryCollector`, atomic `analytics_source_revision` bumping, `DurableJobQueue`, `IngestionWorker` with `15s` lease heartbeat | `backend/app/services/{validator,repository_eligibility,confidence_matrix,github_client,rate_budgeter,repository_collector,job_queue,ingestion_worker}.py`, `backend/app/api/v1/endpoints/ingest.py` |
| **Six-Domain Statistical Engine (Phase C.1–C.7)** | Snapshot-isolated calculators (`methodology_version = "2026.1"`): Development Activity, Contributor Ecosystem, Pull Request Intelligence, Issue Intelligence, Release Evolution, Cross-Domain Synthesis (`B=10,000` Spearman permutations, `\|r_AR1\| > 0.40` guard, BH FDR), deterministic rule consolidation, strict `NULL` vs. `0.0` honesty | `backend/app/services/analytics/*.py`, `backend/app/schemas/analytics_v2.py` |
| **Orchestration, Worker & V2 API (C.7–C.8.3)** | `AnalyticsJobQueue`, `AnalyticsOrchestrator`, `AnalyticsWorker` (`__main__` + opt-in lifespan), post-ingestion auto-enqueue, and Analytics V2 endpoints (`/api/v1/analytics/{owner}/{name}/{recompute,status,latest,history,runs/{id},domains/{key}}`) | `backend/app/services/{analytics_queue,analytics_worker}.py`, `backend/app/api/v1/endpoints/analytics.py` |
| **Documentation Refresh** | Product vision, system architecture, ingestion, analytics specs, DB/migrations, API reference, workers guide, and backend audit | [README.md](../README.md), [product-vision.md](./product-vision.md), [audits/OPENSOURCELENS_BACKEND_AUDIT.md](./audits/OPENSOURCELENS_BACKEND_AUDIT.md) |

---

## Stage 1 — Repository & Source-Code Cleanup (COMPLETED)

**Goal:** Remove dead prototypes, benchmark-coupled legacy code, duplicated collectors, and obsolete Phase 10/11 scaffolding *before* adding new features, while safely preserving shared runtime utilities (`IssueTextCleaner`, `all-MiniLM-L6-v2` embedder, and active DB migrations).

**Status:** **COMPLETED** (Commits: `5cc3dc0`, `64dbc33`, `e4a24ca`, `181997e`, `5ce3dd7`)

### 1.1 Dependency Extraction (Completed)
- Extracted `IssueTextCleaner` and `IssueEmbedder` into `backend/app/services/nlp/`.
- Verified zero broken imports across live collectors and services.

### 1.2 Prototype, Duplicate Collector & Legacy Script Removal (Completed)
- Removed `demo_app.py`, `scripts/` (all 9 scripts), `ml/data_collection/`, `ml/repository_analyzer.py`, `backend/app/database/ingest.py`, and obsolete schema generators.
- Removed benchmark-coupled resolution predictor (`ml/resolution_analysis/`, `resolution_service.py`), static embeddings store dependency (`related_service.py`), and unmounted obsolete routers.
- Removed obsolete lab directories (`ml/eda/`, `ml/preprocessing/`, `ml/nlp/`, `ml/classification/`, `ml/deep_learning/`, `ml/related_issues/`).
- Removed legacy Phase A routes in `analytics.py` (`/overview`, `/comparison`, `/{owner}/{name}`, `/{owner}/{name}/trends`), `repos.py`, and legacy schemas.
- Removed paired obsolete test files (now 379 passing tests, 0 failures).

### 1.3 Frontend Scrub (Completed)
- Removed static `REPO_REGISTRY` and `FALLBACK_ISSUE_CORPUS`.
- Retired dead routes (`/repos`, `/analytics`, `/analyzer`) and obsolete prototype components.
- Verified Next.js build succeeds with 0 TypeScript or lint errors.

### 1.4 What You Can Present After Stage 1
- **Clean Architecture & Zero Dead Code:** Lean, single-source-of-truth codebase with no conflicting collectors, no 4-repo dummy leakage, clean FastAPI Swagger `/docs` showing only universal endpoints, and an all-green test suite.

---

## Stage 2 — Backend Stabilization & First Live Any-Repo E2E Verification (COMPLETED)

**Goal:** Eliminate operational blockers, reconcile fresh-database schema migrations, add full CLI and lifespan support for background workers, handle rate-limit resumption automatically, align runtime scientific dependencies, and verify live ingestion and analytics on real unseen public GitHub repositories.

**Status:** **COMPLETED** (Commits: `89efeb0`, `85747f3`, `34562a2`)

### 2.1 Fresh-DB Migration Reconciliation (Completed)
- Reconciled Alembic Migration `001_initial_benchmark_schema.py` table definitions (`issue_labels`, `repository_analytics`, `predictions`) with `backend/app/database/models.py`.
- Verified clean database provisioning from empty SQLite/PostgreSQL states via `alembic upgrade head`.
- Added `test_fresh_database_upgrade_head_and_orm_queries()` to `tests/test_alembic_migrations.py`.

### 2.2 IngestionWorker CLI & Rate-Limit Resumption (Completed)
- Added CLI entrypoint (`python -m backend.app.services.ingestion_worker`) with graceful SIGINT/SIGTERM handling.
- Integrated optional `ENABLE_INGESTION_WORKER` into FastAPI `lifespan()`.
- Implemented `reopen_resumed_rate_limited_jobs()` in `DurableJobQueue`, automatically re-queuing rate-limited jobs once `resumes_at` has passed.
- Added 3 automated tests in `tests/test_durable_job_queue.py` verifying rate-limit reopening and standalone worker behavior.

### 2.3 Environment & Runtime Dependencies Synchronization (Completed)
- Synchronized `.env.example` with all configuration settings (`ENABLE_INGESTION_WORKER`, `ENABLE_ANALYTICS_WORKER`, polling intervals, `FASTAPI_HOST`, `FASTAPI_PORT`, `RUN_LIVE_GITHUB_TESTS`).
- Synchronized `requirements.txt` and `backend/requirements.txt` declaring all required Data Science, ML/NLP, and HTTP dependencies (`scipy`, `numpy`, `scikit-learn`, `torch`, `transformers`, `sentence-transformers`, `httpx`, `pytest`).

### 2.4 Live Unseen-Repository E2E Verification (Completed)
- Verified live GitHub API ingestion against real-world unseen repositories (`pallets/click`, `pallets/flask`, `psf/requests`) with `RUN_LIVE_GITHUB_TESTS=1`.
- Verified rate-budget adherence (`<= 10` requests for fast snapshot), multi-entity ingestion, and database normalization.
- Verified 100% green status across all 386 project tests (383 passed offline, 3 live tests passing opt-in, 0 failures).

### 2.5 What You Can Present After Stage 2
- **Fully Stabilized Any-Repo Engine:** Submit **any live public GitHub repo URL**, watch rate-budgeted multi-entity collection execute via durable background workers, automatically resume across rate-limits, and provision cleanly from scratch on any new environment.

---

## Stage 3 — Richer GitHub Entity Depth & GitHub Discussions Integration (COMPLETED)

**Goal:** Close the GitHub API collection gaps identified in [product-vision.md](./product-vision.md) so the platform truly extracts deep repository context rather than shallow samples.

**Status:** **COMPLETED** (Commits: `44afbec`, `f8a5616`, `7463ebb`, `f48f8ca`, `a660fcb`)

### 3.1 Schema & Migration 004 (Completed)
- Created Alembic Migration `004_add_discussions_and_pr_depth.py` adding PR depth columns (`body`, `head_branch`, `base_branch`, `is_draft`, `labels`, `requested_reviewers`) and release tag fallback columns (`is_tag_fallback`, `commit_sha`).
- Created new relational tables `discussions` and `discussion_comments` with foreign keys, indexes, and unique constraints.
- Updated `backend/app/database/models.py` with corresponding SQLAlchemy 2.0 ORM mappings.
- Verified live SQLite schema upgrade and Alembic tests in `tests/test_alembic_migrations.py`.

### 3.2 PR Depth & Diff Details Ingestion (Completed)
- Enriched `RepositoryCollector.collect_pull_requests()` to extract `body`, `draft`, `head_branch`, `base_branch`, `labels`, and `requested_reviewers`.
- Added `pr_details` stream budget to `EntityRateBudgeter` (5 in `standard_diagnostic`, 0 in `fast_snapshot`) to sample individual PR diff metrics (`additions`, `deletions`, `changed_files`).
- Verified diff-size turnaround classification (`small..xl`) in `PullRequestIntelligenceCalculator`.

### 3.3 Release Tags Fallback Harvesting (Completed)
- Implemented `/tags` fallback in `RepositoryCollector.collect_releases()` when formal `/releases` returns zero items, querying tag commit dates and marking `is_tag_fallback=True`.
- Updated `ReleaseEvolutionCalculator` to incorporate tag fallback releases and track `tag_fallback_releases_count` in coverage telemetry.

### 3.4 GitHub Discussions GraphQL Client & Ingestion (Completed)
- Added `execute_graphql()` method to `AsyncGitHubClient` with query validation, error handling, rate budget accounting, and inter-request pacing.
- Implemented `RepositoryCollector.collect_discussions()` querying up to 10 discussions per page, extracting categories, question answerability, marked answers, comment threads, and upvotes.

### 3.5 Discussions Intelligence Analytics Engine & Synthesis (Completed)
- Created `DiscussionsIntelligenceCalculator` (Module 7 / Stage 3) implementing `BaseMetricCalculator`.
- Computes discussions volume, category breakdown, Q&A resolution dynamics (`qna_answered_rate`, `time_to_answer_hours` quantiles p50/p75/p90/mean), community engagement (median comments & upvotes), and contributor overlap bridging.
- Generates deterministic rules: `RULE-DISC-EMPTY`, `RULE-DISC-001`, `RULE-DISC-002`, `RULE-DISC-003`, `RULE-DISC-004`.
- Integrated `community_discussions` card into `CrossDomainSynthesisCalculator` diagnostic scorecard.
- Bumped `CURRENT_METHODOLOGY_VERSION = "2026.2"` and wired Discussions into `AnalyticsOrchestrator` and V2 domain endpoints.
- Added comprehensive unit tests in `tests/test_analytics_discussions.py` (6/6 passed).

### 3.6 What You Can Present After Stage 3
- **Deep GitHub Data Showcase:** Show that for any submitted repository, OpenSourceLens captures PR diff sizes, branches, drafts, reviewers, git tag release history, **and live GitHub Discussions insights** (Q&A resolution rates, answer latency percentiles, and community engagement)—far richer than standard GitHub counters. All 394 project tests pass with 0 failures.

---

## Stage 4 — Interactive Single-Repository Workspace UI *(Feature Slice 2: Visual Core)* — [COMPLETED]

**Goal:** Deliver the intuitive, interactive Next.js web application where any contributor, maintainer, or visitor can analyze a repository visually without touching API tools.

1. **Universal Entry & Live Job Flow (`/`, `/explore`):** [DONE]
   - Clean GitHub URL / `owner/repo` submission bar -> instant pre-flight eligibility & budget preview (`fast_snapshot` vs. `standard_diagnostic`) -> live step-by-step ingestion & analytics progress bar (`status` polling + transparent coverage/truncation banners) implemented in `frontend/src/components/ingestion/RepoSearchFlow.tsx` and embedded in `frontend/src/app/page.tsx`.
2. **Repository Intelligence Workspace (`/repo/[owner]/[name]`):** [DONE]
   - Fully interactive Single-Repository Intelligence Workspace dynamic route implemented in `frontend/src/app/repo/[owner]/[name]/page.tsx` with `WorkspaceHeader.tsx` (real-time revision tracking, staleness warning, and one-click snapshot recompute).
   - **Overview & Consolidated Report Tab (`TabOverview.tsx`):** Multi-Domain Diagnostic Scorecard across 6 domains, Community Discussions health summary, transparent data-coverage badges, and plain-English rule interpretations sorted by severity (`critical`, `warning`, `info`).
   - **Development Activity & Releases Tab (`TabActivity.tsx`):** Weekly commit velocity with 4w/12w SMA & EWMA curves, anomaly & change-point markers, `7x24` UTC activity rhythm heatmap, commit landing latency quantiles, SemVer 2.0.0 adoption, and cadence regularity.
   - **Contributors & Collaboration Tab (`TabContributors.tsx`):** Identity-clean contributor directory, role breakdown, Gini inequality gauge, interactive Lorenz curve line chart, and `BF50`/`BF80` bus factor concentration alerts.
   - **Pull Requests & Code Review Tab (`TabPullRequests.tsx`):** `TTFResp` vs. `TTFR_formal` vs. `TTFR_inline` latency comparison, merge/abandonment outcome donut chart, turnaround quantiles (`p50`, `p75`, `p90`, `mean`), diff-size turnaround bins, and top reviewer bottlenecks.
   - **Issues & Backlog Tab (`TabIssues.tsx`):** Monthly flow chart (opened vs closed), SLA resolution quantiles, 5-bin open backlog aging distribution, and discussion intensity.
   - **Community Discussions Tab (`TabDiscussions.tsx`):** Q&A resolution rate meter, answer latency quantiles (`p50`, `p75`, `p90`, `mean`), category distribution bar chart, top engaged discussion threads, and contributor bridge overlap rate.

### 4.6 What You Can Present After Stage 4
- **Full End-to-End Product Demo:** Live browser walkthrough from pasting any GitHub URL or `owner/repo` into the universal search flow, watching the live pre-flight eligibility check and ingestion progress, to exploring rich interactive charts and diagnostic cards across Development Activity, Contributors, PR Reviews, Issues, Releases, and Discussions—with zero terminal commands or API clients needed.
- **Verification:** `npm --prefix frontend run type-check` passed with 0 errors; `npm --prefix frontend run build` succeeded creating optimized production bundles for all static and dynamic workspace routes.

---

## Stage 5 — Generalizable ML/NLP & Issue/Discussion AI Explorer *(Feature Slice 3: Applied Intelligence)*

**Goal:** Connect true repo-agnostic ML & Deep Learning directly into post-ingestion processing and surface actionable backlog/community discovery in the UI.

1. **Generalizable 5-Class Text Categorization:**
   - Upgrade/re-evaluate transformer + calibrated linear baseline for 5 actionable categories (`bug`, `feature`, `documentation`, `question`, `other`) using **cross-repository (Leave-One-Repo-Out) validation** and low-confidence abstention (`uncertain`), automatically scoring unlabeled issues/discussions post-ingestion without altering raw GitHub labels.
2. **Per-Repository Semantic Duplicate & Related-Thread Retrieval:**
   - Compute `all-MiniLM-L6-v2` embeddings dynamically for every newly ingested repository's issues & discussions (`issue_embeddings` / per-repo vector store), replacing benchmark `embeddings.npz` coupling and powering `GET/POST /api/v1/analytics/{owner}/{name}/similar`.
3. **Unsupervised Topic & Theme Discovery:**
   - Cluster per-repository dense embeddings and generate interpretable topic clusters + representative keywords across the repository's issues and discussions automatically (no pre-defined repo labels required).
4. **Issues, Backlog & AI Explorer UI Tab:**
   - Add Issue SLA quantiles, Kaplan-Meier issue survival (`S(30d)`/`S(60d)`/`S(90d)`), 5-bin open backlog aging (`fresh`..`dormant`), AI category filter, **Interactive Similar/Duplicate Issue Finder**, and **Recurring Topics Explorer** with drill-down to actual issues/discussions.

### What You Can Present After Stage 5
- **AI-Powered Backlog Intelligence Demo:** Paste any unseen repo URL and demonstrate: (1) auto-categorized unlabeled issues with confidence scores, (2) live semantic duplicate discovery scoped purely to that repo, and (3) automatically uncovered top recurring bug/feature/discussion themes.

---

## Stage 6 — Cross-Domain Discovery Explorer, Reports & Production Deployment *(Feature Slice 4: Complete Release)*

**Goal:** Complete the analytical storytelling layer, validate multi-worker scale on PostgreSQL, and package the entire platform for reliable deployment.

1. **Cross-Domain Explorer & Exportable Reports:**
   - Interactive monthly cross-domain chart view (`PAIR-01..03` churn vs. issues, review response vs. PR abandonment, commit pace vs. merges) with clear non-causal interpretation, autocorrelation (`\|r_AR1\|`) & FDR guards, and one-click exportable repository intelligence report (Markdown / JSON).
2. **Production Hardening & PostgreSQL Validation:**
   - Full Docker Compose stack (`db` PostgreSQL 16, FastAPI API, `IngestionWorker`, `AnalyticsWorker`, Next.js frontend).
   - Multi-worker PostgreSQL stress verification (`FOR UPDATE SKIP LOCKED`, concurrent claims, lease expiration recovery, mid-run revision superseding), `AnalyticsWorker` long-run heartbeat, and public API rate-limiting/protection.
3. **Cross-Repo Validation & Final Documentation:**
   - End-to-end validation benchmark across diverse unseen public GitHub repositories (small library, active framework, large ecosystem) and final evaluation summary.

### What You Can Present After Stage 6
- **Production-Ready Platform Delivery:** Complete Dockerized live deployment, statistically guarded cross-domain discovery charts, downloadable repo audit reports, and verified unseen-repository performance across diverse project sizes.
