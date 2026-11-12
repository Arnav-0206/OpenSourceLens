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

## Stage 1 — Repository & Source-Code Cleanup (Execute First)

**Goal:** Remove dead prototypes, benchmark-coupled legacy code, duplicated collectors, and obsolete Phase 10/11 scaffolding *before* adding new features, while safely preserving shared runtime utilities (`IssueTextCleaner`, `all-MiniLM-L6-v2` embedder, and active DB migrations).

> [!IMPORTANT]
> Per project rules, **no files or folders listed below are deleted until reviewed and explicitly approved** in a code-cleanup plan step.

### 1.1 Dependency Extraction (Safety Prerequisite Before Deleting `ml/`)
- Extract `IssueTextCleaner` (currently imported by `backend/app/services/repository_collector.py` and `classification_service.py` from `ml/preprocessing/text_cleaner.py`) into a clean backend module (`backend/app/services/nlp/text_cleaner.py`).
- Extract/retain the clean `all-MiniLM-L6-v2` embedding encoder helper cleanly for per-repo indexing in Stage 5, independent of `models/related_issues/embeddings.npz`.

### 1.2 Exact Targets Slated for Removal or Retirement

| Category | Specific Paths & Components to Remove | Reason |
|---|---|---|
| **Prototype App & Ad-Hoc Scripts** | `demo_app.py` (root Streamlit prototype); `scripts/verify_*.py` (`9` Phase 10/11 verification scripts); `scripts/` folder once emptied | Superseded by FastAPI + Next.js + `pytest` suite. |
| **Duplicate Legacy Ingestion & Schema Gen** | `ml/data_collection/*`; `ml/repository_analyzer.py`; `backend/app/database/ingest.py`; `backend/app/database/schema.sql`; `backend/app/database/schema_generator.py` | Replaced by Phase B `RepositoryCollector` + SQLAlchemy ORM + Alembic migrations. |
| **Invalid / Benchmark-Locked ML (`ResolutionPredictor`)** | `ml/resolution_analysis/*`; `backend/app/services/resolution_service.py`; `backend/app/api/v1/endpoints/resolution.py`; `backend/app/schemas/resolution.py`; `models/resolution_classifier_champion.joblib`; `models/resolution_regressor_champion.joblib`; `models/resolution_scaler_params.json` | Phase 8 proved continuous within-repo prediction fails (`R^2 = -0.04..+0.18`), and the RF model uses hardcoded 4-repo dummy variables invalid on unseen repos. Replaced by factual Kaplan-Meier survival & quantiles. |
| **Static 4-Repo Similarity Coupling** | `related_issues_cache` lookup path + static `models/related_issues/embeddings.npz` / `metadata.*` dependency in `backend/app/services/related_service.py` (keep `/api/v1/related*` route shells only if immediately replaced in Stage 5, or remove cleanly now and re-introduce clean V2 endpoints in Stage 5) | Locked to `2,328` static benchmark issues; cannot index newly submitted repositories. |
| **Obsolete Offline Track A Scripts & Legacy Tests** | `ml/eda/*`, `ml/preprocessing/*` (after extracting `IssueTextCleaner`), `ml/nlp/*`, `ml/classification/*`, `ml/deep_learning/*`, `ml/related_issues/*`; redundant pre-Phase B `tests/test_*` suites targeting deleted `demo_app.py`, `ml.data_collection`, `ml.eda.*`, `ml.resolution_*`, `database.ingest`, `schema_generator` | Track A lab scripts already fulfilled their purpose; keeping them creates false dual pipelines. *(Note: Confirm explicitly whether to retain `data/raw/`, `data/processed/`, and `models/distilbert_finetuned/` as offline training baselines for Stage 5 retraining or archive them outside active runtime.)* |
| **Legacy Phase A APIs & DB Summary Paths** | `backend/app/api/v1/endpoints/repos.py` (`/api/v1/repos*`), `backend/app/schemas/repository.py` (legacy summary schemas), and Phase A routes in `backend/app/api/v1/endpoints/analytics.py` (`/overview`, `/comparison`, legacy `/{owner}/{name}`, `/trends`) backed by `repository_analytics` | Case-sensitive legacy APIs serving stale 4-repo Phase A snapshots alongside Analytics V2. |
| **Phase 10 Hardcoded Frontend Scaffolding** | Hardcoded `REPO_REGISTRY`, `FALLBACK_ISSUE_CORPUS`, legacy `/analytics` (4-repo comparison page), `/analyzer` (Phase A static ML playground), and legacy `/repos` mock components under `frontend/src/` | Built around 4 static repos and Phase A APIs; clear slate for the Stage 4 universal workspace. |

### 1.3 What You Can Present After Stage 1
- **Clean Architecture & Zero Dead Code:** Lean, single-source-of-truth codebase with no conflicting collectors, no 4-repo dummy leakage, clean FastAPI Swagger `/docs` showing only universal endpoints, and a fast, all-green `pytest` run on active backend components.

---

## Stage 2 — Backend Stabilization & First Live Any-Repo E2E Verification

**Goal:** Eliminate the operational blockers identified in [OPENSOURCELENS_BACKEND_AUDIT.md](./audits/OPENSOURCELENS_BACKEND_AUDIT.md) and prove the backend pipeline works end-to-end on real unseen public GitHub repositories.

1. **Fresh-DB Migration Fix:** Reconcile Migration `001` vs. ORM drift (`issue_labels`, `repository_analytics`, `predictions`, plus `pull_requests` unique constraint parity) via a clean Alembic fix/migration so `alembic upgrade head` works on an empty SQLite or PostgreSQL database.
2. **Worker CLI & Config Hygiene:** Add `if __name__ == "__main__":` CLI + optional lifespan hook to `IngestionWorker`, add automated wake/re-enqueue handling for `rate_limited` jobs after `resumes_at`, and align `.env.example` + `requirements.txt` (including runtime analytics/ML dependencies and optional PostgreSQL driver guidance).
3. **Live Unseen-Repo Validation:** Execute `RUN_LIVE_GITHUB_TESTS=1` end-to-end across small, medium, and large unseen public repositories (`POST preview` -> `POST submit` -> `IngestionWorker` -> auto-enqueue -> `AnalyticsWorker` -> `GET /api/v1/analytics/{owner}/{name}/latest`).

### What You Can Present After Stage 2
- **Working Backend Engine Demo:** Submit **any live GitHub repo URL** via Swagger UI or API, watch live rate-budgeted ingestion and automatic 6-domain analytics complete cleanly on a freshly initialized database, and show real diagnostic output on unseen repositories.

---

## Stage 3 — Richer GitHub Entity Depth & GitHub Discussions Integration *(Feature Slice 1)*

**Goal:** Close the GitHub API collection gaps identified in [product-vision.md](./product-vision.md) so the platform truly extracts deep repository context rather than shallow samples.

1. **Pull Request & Commit Depth Expansion:**
   - Enrich `PullRequest` model & collector (`Migration 004`) to capture `body`, `head_branch`, `base_branch`, `is_draft`, `labels`, `requested_reviewers`, and fetch per-PR detail (`additions`, `deletions`, `changed_files`) so size-vs-turnaround analysis (`small..xl`) populates on live runs.
   - Increase/make configurable `commit_details` diff budgets and enrich issue fields (`assignees`, `reactions_count`, configurable pagination depth).
2. **Release Tags & Notes Support:**
   - Harvest GitHub `/tags` as fallback when formal `/releases` are unused, and persist release/tag metadata for richer evolution analysis.
3. **GitHub Discussions & Community Context:**
   - Add GitHub GraphQL (or Discussions REST/GraphQL client) collection for repository **Discussions** (`discussions`, `discussion_comments` tables: category, title, body, author, is_answered, upvotes/reactions, timestamps, comments).
   - Add Community/Discussions profile metrics (volume, category mix, Q&A answered rate, engagement, recurring themes hook) to the analytics payload (`methodology_version` bump, e.g. `"2026.2"`).

### What You Can Present After Stage 3
- **Deep GitHub Data Showcase:** Show that for any submitted repository, OpenSourceLens now captures PR diff sizes, branches, drafts, reviewers, issue reactions/assignees, git tag history, **and live GitHub Discussions insights**—far richer than basic GitHub repo counters.

---

## Stage 4 — Interactive Single-Repository Workspace UI *(Feature Slice 2: Visual Core)*

**Goal:** Deliver the intuitive, interactive Next.js web application where any contributor, maintainer, or visitor can analyze a repository visually without touching API tools.

1. **Universal Entry & Live Job Flow (`/`, `/explore`):**
   - Clean GitHub URL / `owner/repo` submission bar -> instant pre-flight eligibility & budget preview (`fast_snapshot` vs. `standard_diagnostic`) -> live step-by-step ingestion & analytics progress bar (`status` polling + transparent coverage/truncation banners).
2. **Repository Intelligence Workspace (`/repo/[owner]/[name]`):**
   - **Overview & Consolidated Report Tab:** Repo & Community Discussions summary, Multi-Domain Diagnostic Scorecard, transparent data-coverage badges (never hiding `NULL` or bounded windows), and plain-English rule findings sorted by severity.
   - **Development Activity & Releases Tab:** Weekly commit velocity with 4w/12w SMA & EWMA curves, anomaly & change-point markers, `7x24` UTC activity heatmap, churn coverage, commit landing latency quantiles, SemVer adoption, and pre-release surge timelines.
   - **Contributors & Collaboration Tab:** Identity-clean contributor leaderboard, role filter (commits/PRs/reviews/issues/comments), Gini inequality + interactive Lorenz curve, `BF50`/`BF80` concentration alerts, and bot/unattributed isolation metrics.
   - **Pull Requests & Code Review Tab:** `TTFResp` vs. `TTFR_formal` vs. `TTFR_inline` latency comparison, merge/abandonment outcome donut, PR Kaplan-Meier survival curve, diff-size turnaround chart, and reviewer bottleneck concentration.

### What You Can Present After Stage 4
- **Full End-to-End Product Demo:** Live browser walkthrough from pasting a GitHub URL to exploring interactive charts for Development Activity, Contributors, PR Reviews, Releases, and Discussions—fully understandable to any contributor or maintainer.

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
