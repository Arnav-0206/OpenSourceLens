# OpenSourceLens — Phase A Completion Report

**Milestone:** Phase A — Project Cleanup and Architecture Preparation  
**Status:** Completed & Formally Closed  
**Date:** November 7, 2026  
**Pre-Cleanup Baseline Commit:** `7fb1c67cf114a8faa7a2b57297641c8bd0920b4f`  
**Git Safety Branch:** `checkpoint/pre-cleanup-baseline` (`7fb1c67`)  
**Git Baseline Tag:** `v1.0-benchmark-baseline` (`7fb1c67`)  
**Post-Cleanup Commit:** `ed9348905682359277a79c657506b63c2561a15b`  
**Working Tree State:** Clean (Zero uncommitted changes)  

---

## 1. Executive Summary

Phase A (Project Cleanup and Architecture Preparation) represents the critical transition point of OpenSourceLens from a static, four-repository demonstration prototype (`microsoft/vscode`, `pytorch/pytorch`, `kubernetes/kubernetes`, `huggingface/transformers`) into a universal, Data Science-powered GitHub Repository Intelligence Platform capable of analyzing **any eligible public GitHub repository** on demand.

Before initiating any destructive operations, the working tree was verified clean and permanently anchored with safety checkpoints. A systematic cleanup was executed across the codebase:
- Obsolete landing showcases, hardcoded four-repository dropdowns, and mock static catalog arrays were removed.
- All experimental research scripts, raw and processed benchmark datasets, trained ML model weights (DistilBERT, Sentence-Transformers, classical baselines), and test suites were strictly preserved.
- Full regression testing was conducted, confirming **736/736 passing pytest tests** and **11/11 successful Next.js page compilations** with zero regressions.
- The architectural specifications for dynamic ingestion, rate budgeting, normalized contributor activity tracking, and out-of-distribution model evaluation were finalized.

---

## 2. Safety Checkpoints & Reversibility Guarantees

To ensure complete reversibility and academic reproducibility of the initial benchmark research, the following safety checkpoints were created and verified:

| Checkpoint Asset | Value / Git SHA | Verification Status | Purpose & Retention Policy |
|---|---|---|---|
| **Baseline Working Tree** | Commit `7fb1c67cf114a8faa7a2b57297641c8bd0920b4f` | Verified Clean | Pre-cleanup state containing Phase 0–11 benchmark deliverables. |
| **Git Safety Branch** | `checkpoint/pre-cleanup-baseline` | Verified (`7fb1c67`) | Dedicated branch allowing instant checkout of the original Phase 11 prototype. |
| **Git Baseline Tag** | `v1.0-benchmark-baseline` | Verified (`7fb1c67`) | Permanent semantic version tag marking the benchmark laboratory foundation. |
| **Post-Cleanup Commit** | Commit `ed9348905682359277a79c657506b63c2561a15b` | Committed to `main` | Cleaned repository state with obsolete code purged. |

---

## 3. Inventory of Changes: Controlled Removal of Obsolete Code

Across the codebase, **3 dead prototype files were deleted and 766 dead lines were purged across 6 files** (Net: +141 insertions, -766 deletions).

### 3.1 Deleted Files (Category 4 Obsolete Implementation)

| File Path | Action | Lines Removed | Rationale |
|---|---|---|---|
| `frontend/src/components/landing/EcosystemShowcase.tsx` | **DELETED** | -207 lines | **Obsolete prototype.** Hardcoded cards, star counters, and SLAs for only the 4 benchmark repositories. |
| `frontend/src/components/landing/GlobalMetricsGrid.tsx` | **DELETED** | -109 lines | **Obsolete prototype.** Hardcoded static counters ("4 Repositories Monitored", "2,328 Curated Issues"). |
| `frontend/src/components/repository/RepoSelector.tsx` | **DELETED** | -207 lines | **Obsolete prototype.** Dropdown restricted specifically to the 4 initial repositories. |

### 3.2 Decoupled Modifications

| File Path | Action | Lines Changed | Rationale |
|---|---|---|---|
| `frontend/src/app/page.tsx` | **MODIFIED** | -10 lines | Decoupled imports and rendering of `EcosystemShowcase` and `GlobalMetricsGrid`. Retained `HeroSection`, `BackendStatusBanner`, and `FeaturesShowcase`. |
| `frontend/src/components/layout/Navbar.tsx` | **MODIFIED** | -51 lines | Removed `RepoSelector` component and its imports from both desktop navbar and mobile drawer. |
| `frontend/src/app/repos/page.tsx` | **MODIFIED** | +88, -235 lines | Removed 74-line hardcoded `initialRepos` static mock array. State now initializes as empty and dynamically queries `getRepositories()` from FastAPI backend. |

### 3.3 Untouched Core Codebases

- **Backend API (`backend/app/`):** 0 lines altered.
- **Database Schema (`backend/app/database/models.py`):** 0 lines altered.
- **ML & NLP Pipelines (`ml/`):** 0 lines altered.
- **Model Weights & Embeddings (`models/`):** 0 bytes altered.
- **Test Suites (`tests/`):** 0 lines altered.

---

## 4. Preservation of Multi-Environment Assets

OpenSourceLens explicitly maintains three distinct operational environments:

```mermaid
flowchart TD
    subgraph Env1["1. Production Universal Application"]
        FastAPI[backend/app/ - FastAPI API Engine]
        NextJS[frontend/src/ - Next.js 15 UI]
        DB[SQLite WAL / PostgreSQL Dynamic Storage]
    end

    subgraph Env2["2. Offline Benchmark Research Lab"]
        RawData[data/raw/ - 2,328 Issues & Comments]
        ProcessedData[data/processed/ - Normalized Splits]
        EDAScripts[ml/eda/ - FINDINGS.md & Analysis]
    end

    subgraph Env3["3. Trained Machine Learning Artifacts"]
        DistilBERT[models/distilbert_finetuned/ - 267MB Weights]
        DenseEmbeddings[models/related_issues/embeddings.npz]
        Baselines[models/ - TF-IDF, Logistic Regression, SVM]
    end

    Env2 --> Env3
    Env3 -.->|Inference| FastAPI
    DB <--> FastAPI
    FastAPI <--> NextJS
```

1. **Production Universal Application:**
   - Asynchronous FastAPI backend engine (`backend/app/main.py`, `router.py`, `endpoints/`).
   - Relational database schema with WAL mode enabled (`backend/app/database/`).
   - Next.js 15 UI with dynamic route parameters `[owner]/[name]` (`frontend/src/`).
2. **Offline Benchmark Research Lab:**
   - Raw harvested JSON payloads from the 4 benchmark ecosystems (`data/raw/`).
   - Cleaned, stratified train/validation/test dataset splits (`data/processed/`).
   - Exploratory data analysis notebooks, Kaplan-Meier curves, and findings (`ml/eda/FINDINGS.md`).
3. **Trained ML Artifacts:**
   - Fine-tuned `distilbert-base-uncased` sequence classifier (66.9M parameters, 81.3% benchmark accuracy) in `models/distilbert_finetuned/`.
   - 384-dimensional dense semantic embeddings in `models/related_issues/embeddings.npz`.
   - Classical TF-IDF vectorizers, Logistic Regression, and Linear SVM baselines in `models/`.

---

## 5. Architectural & Data Modeling Specifications Established

### 5.1 Route Generalization Confirmation

Both backend and frontend routing architectures were verified as fully dynamic:
- **FastAPI Endpoints:** `GET /api/v1/repos` queries all registered repositories dynamically. `GET /api/v1/repos/{owner}/{name}` uses path parameters to look up any repository string.
- **Next.js Pages:** Dynamic folder routing under `frontend/src/app/repos/[owner]/[name]/page.tsx` and `issues/page.tsx` handles any valid `{owner}/{repo}` slug.
- **Catalog Role:** The four existing database rows serve purely as baseline demonstration data, with zero hardcoded constraints or enumeration limits.

### 5.2 Normalized Contributor Activity Tracking & Event Deduplication

To enable newcomer cohort analysis, retention half-life modeling, and participation frequency, a normalized event model was specified:

- **Entity Natural Key:** Composite uniqueness `(repository_id, activity_type, entity_id)` guarantees idempotent collection.
- **Multiple Review Handling:** For `activity_type = 'review_submitted'`, `entity_id` is GitHub's immutable **`review.id`** (e.g. `"101"`, `"205"`), **not** the pull request number. Multiple reviews on the same pull request by the same or different reviewers are preserved as distinct, timestamped activity records.
- **Comment Handling:** For `comment_created`, `entity_id` is GitHub's unique `comment.id`.
- **Commit Handling:** For `commit`, `entity_id` is the Commit SHA.

### 5.3 Unified Contributor Identity Resolution Protocol

- **Anchor:** Identity is anchored to GitHub's immutable numerical `github_id`.
- **Unlinked Identities:** Commits authored with emails unlinked to GitHub accounts are tracked via `(repository_id, git_email)`.
- **Bot Filtering:** Accounts with `[bot]` suffixes or known bot logins are segregated to prevent distortion of human collaboration metrics.
- **No Speculative Merging:** Identities with different emails or names are never merged without verified GitHub account linkings.

### 5.4 Historical Data Preservation Policy

- **Soft Deletion:** Repositories implement `is_active` (Boolean, default `True`) and `is_archived` (Boolean, default `False`). Deleting a repository sets `is_active = False`.
- **Cascade Protection:** Child records in `contributor_activities`, `commits`, `pull_requests`, and `issues` are never hard-deleted during standard operations.
- **Audit Trails:** `collection_runs`, `analysis_runs`, and `prediction_audit_log` reference `repositories.id` with `ON DELETE SET NULL`, preserving historical metrics and SLA tracking indefinitely.

### 5.5 Scientific ML Generalization Policy

- **DistilBERT Status:** Formally documented as an existing benchmark-trained model (81.3% test accuracy on curated benchmark data). Its out-of-distribution generalization to arbitrary unseen repositories is explicitly flagged as **pending empirical evaluation** in Phase D.
- **Resolution Classifier Decoupling:** The Random Forest resolution model currently contains repository-specific one-hot dummy features (`repo_name_*`). Per user instruction, retraining and decoupling to domain-agnostic features (author relationship, text length, label complexity) is deferred to Phase D (ML Retraining) to preserve the original benchmark artifact intact.

---

## 6. Empirical Verification & Regression Testing Results

All tests and builds were executed on commit `ed93489`:

### 6.1 Pytest Full Test Suite

```text
======================== 736 passed, 13 warnings in 138.94s ========================
```
- **Passed:** 736
- **Failed:** 0
- **Skipped:** 0
- **Pass Rate:** **100%**

### 6.2 Pytest Integration & API Subset

```text
================== 145 passed, 591 deselected, 6 warnings in 12.51s ==================
```
- Covers `test_api_repos.py`, `test_api_classification.py`, `test_api_related.py`, `test_api_resolution.py`, `test_api_analytics.py`, and `test_integration_e2e_full_system.py`.
- **Pass Rate:** **100%**

### 6.3 Next.js Production Build

Executed `npm run build` in `frontend/`:
```text
▲ Next.js 15.5.0
✓ Compiled successfully in 1935ms
✓ Linting and checking validity of types ... Passed (0 errors, 0 warnings)
✓ Collecting page data ... Completed
✓ Generating static pages (11/11) ... Completed
✓ Finalizing page optimization ... Completed

Route (app)                              Size     First Load JS
┌ ○ /                                    6.42 kB         135 kB
├ ○ /_not-found                          991 B           101 kB
├ ○ /analytics                           4.21 kB         132 kB
├ ○ /analyzer                            18.2 kB         146 kB
├ ○ /repos                               3.84 kB         132 kB
├ ƒ /repos/[owner]/[name]                14.6 kB         143 kB
├ ƒ /repos/[owner]/[name]/issues         12.1 kB         140 kB
└ ○ /repos/compare                       8.74 kB         137 kB
+ First Load JS shared by all            100 kB
```
- **Total Pages Compiled:** 11 / 11
- **TypeScript Errors:** 0
- **ESLint Errors:** 0

### 6.4 API Route Contract Verification

Verified via `python scripts/verify_connection.py --test-mode`:
- `GET /health` $\rightarrow$ 200 OK (Database healthy, models pre-warmed).
- `GET /api/v1/repos` $\rightarrow$ 200 OK.
- `GET /api/v1/repos/microsoft/vscode` $\rightarrow$ 200 OK.
- `GET /api/v1/repos/microsoft/vscode/issues` $\rightarrow$ 200 OK.
- `POST /api/v1/classify` $\rightarrow$ 200 OK.
- **Contract Verification:** 5/5 endpoints verified.

---

## 7. Phase A Conclusion & Next Steps

Phase A is **formally complete and verified**. The repository is fully cleaned, documented, anchored by safety checkpoints, and ready for Phase B.

### Phase B Implementation Scope (Awaiting User Authorization)

When authorized, **Phase B (Core Backend and Database)** will implement:

1. **Step B.1 — Dynamic Repository URL Validator & Eligibility Engine:**
   - File: `backend/app/services/validator.py`
   - Validates GitHub URLs, extracts `{owner}/{repo}`, queries `GET /repos/{owner}/{repo}`, verifies eligibility ($\ge 5$ commits, $\ge 5$ issues/PRs).
2. **Step B.2 — Tiered Rate-Limit Budgeter:**
   - File: `backend/app/services/budgeter.py`
   - Allocates request quotas for Fast Snapshot ($\le 100$ records per entity, $\sim 6$ API requests) and Standard Diagnostic ($\le 500$ issues, 300 commits/PRs, $\sim 25$ API requests) based on `X-RateLimit-Remaining`.
3. **Step B.3 — Multi-Entity Dynamic GitHub Collector:**
   - File: `backend/app/services/dynamic_collector.py`
   - Asynchronous pagination for Commits, Contributors, PRs, PR Reviews, Issues, Comments, and Releases using `httpx` and `asyncio`.
4. **Step B.4 — Relational Database Models & Alembic Migration:**
   - File: `backend/app/database/models.py`
   - Adds `CollectionRun`, `AnalysisRun`, `Commit`, `Contributor`, `ContributorActivity`, `PullRequest`, `PullRequestReview`, `Release`, and `IssueComment` tables.
5. **Step B.5 — Dynamic Ingestion API Endpoints & Background Task Processing:**
   - File: `backend/app/api/v1/endpoints/ingestion.py`
   - Exposes `POST /api/v1/repos/analyze` and `GET /api/v1/repos/{owner}/{name}/status` with real-time background task telemetry.
