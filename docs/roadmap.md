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

## 🚀 Track B: Realigned Universal Platform Evolution (Phases 11-R–16-R)

### Phase 11-R — Universal Ingestion Engine & Schema Migration 🔜
**Goal:** Enable OpenSourceLens to ingest commits, contributors, pull requests, and issues dynamically for any valid public repository entered by a user.

| Step | Title | Description | Target Deliverable |
|---|---|---|---|
| **11-R.1** | Repository URL Validator | Parse GitHub URLs, extract `{owner}/{repo}`, and verify public existence via API | `backend/app/services/validator.py` |
| **11-R.2** | Multi-Entity Collection Pipeline | Implement lightweight collectors for commits, contributors, PRs, and releases | `ml/data_collection/dynamic_collector.py` |
| **11-R.3** | Tiered Rate-Limit Budgeter | Implement quota allocation (Fast Snapshot vs Standard Diagnostic) to respect rate limits | `backend/app/services/budgeter.py` |
| **11-R.4** | Relational Schema Migration | Add `commits`, `contributors`, and `pull_requests` tables via SQLAlchemy migration | `backend/app/database/models.py` |
| **11-R.5** | Dynamic Ingestion API Endpoint | Expose `POST /api/repos/analyze` with async task tracking and progress reporting | `backend/app/api/ingestion_routes.py` |

---

### Phase 12-R — Multi-Domain Intelligence Engines
**Goal:** Implement the statistical, analytical, and NLP engines across the six core intelligence domains.

| Step | Title | Description | Target Deliverable |
|---|---|---|---|
| **12-R.1** | Development Activity Engine | Time-series commit/release aggregation, rolling trends, and anomaly detection | `backend/app/services/activity_service.py` |
| **12-R.2** | Contributor Ecosystem Engine | Gini coefficient, Lorenz curve calculation, Shannon entropy, core vs peripheral | `backend/app/services/contributor_service.py` |
| **12-R.3** | Pull Request Intelligence Engine | Review latency, time-to-merge distributions, and review bottleneck detection | `backend/app/services/pr_service.py` |
| **12-R.4** | Dynamic Semantic Vector Caching | Vectorize newly ingested issues on-demand with `all-MiniLM-L6-v2` and cache arrays | `backend/app/services/embedding_service.py` |
| **12-R.5** | Domain-Agnostic Resolution Advisory | Decouple resolution advisory from repo-specific one-hot dummies to universal signals | `backend/app/services/resolution_service.py` |
| **12-R.6** | Cross-Domain Intelligence Synthesizer | Correlate multi-domain signals into automated, evidence-backed findings report | `backend/app/services/intelligence_service.py` |

---

### Phase 13-R — Dynamic Frontend Workspace
**Goal:** Deliver an interactive, general-purpose repository intelligence workspace in Next.js.

| Step | Title | Description | Target Deliverable |
|---|---|---|---|
| **13-R.1** | Universal Repository Search Bar | Homepage search bar with URL validation, ingestion progress modal, and error states | `frontend/src/components/RepositorySearchBar.tsx` |
| **13-R.2** | Multi-Domain Tabbed Workspace | Redesign `/repos/[owner]/[name]` into 6 tabbed views (Overview, Activity, Contributors, PRs, Issues, Synthesis) | `frontend/src/app/repos/[owner]/[name]/page.tsx` |
| **13-R.3** | Advanced Analytical Visualizations | Recharts components for Lorenz curve, activity trendlines, and velocity gauges | `frontend/src/components/charts/` |
| **13-R.4** | Evidence-Backed Findings Card | Diagnostic summary card presenting automated intelligence observations | `frontend/src/components/FindingsReportCard.tsx` |
| **13-R.5** | Benchmark Explorer Mode | Preserved comparison view for the four initial baseline repositories | `frontend/src/app/benchmarks/page.tsx` |

---

### Phase 14-R — Generalization Testing & Performance Validation
**Goal:** Rigorously validate model inference and pipeline resilience on unseen public repositories.

| Step | Title | Description | Target Deliverable |
|---|---|---|---|
| **14-R.1** | Cross-Repository Generalization Suite | Evaluate DistilBERT and semantic retrieval on novel external repositories | `tests/test_generalization.py` |
| **14-R.2** | Rate-Limit Resilience Verification | Test API behavior under exhausted rate limits and backoff recovery | `tests/test_rate_limits.py` |
| **14-R.3** | End-to-End Workflow Integration Test | End-to-end integration test: URL submission → Ingestion → Analytics → Rendering | `tests/test_e2e_workflow.py` |

---

### Phase 15-R — Deployment & Production Packaging
**Goal:** Package and deploy the complete platform.

| Step | Title | Description | Target Deliverable |
|---|---|---|---|
| **15-R.1** | Docker Containerization | Multi-stage Dockerfiles for FastAPI backend and Next.js frontend | `Dockerfile.backend`, `Dockerfile.frontend` |
| **15-R.2** | PostgreSQL Deployment Configuration | Production database configuration and connection pooling setup | `backend/app/database/connection.py` |
| **15-R.3** | Model Artifact Serving Strategy | Optimization of DistilBERT and MiniLM caching for production hosting | `docker-compose.yml` |
| **15-R.4** | Production Smoke Testing | Verification of production deployment on cloud environment | Diagnostic report |

---

### Phase 16-R — Academic Research Paper & Final Defense
**Goal:** Complete academic documentation, research paper, and presentation.

| Step | Title | Description | Target Deliverable |
|---|---|---|---|
| **16-R.1** | Final Empirical Findings | Comprehensive comparative metrics across benchmark and dynamic repositories | `docs/final_evaluation_report.md` |
| **16-R.2** | Conference Research Paper | Full academic paper on repository intelligence and cross-repo NLP | `docs/conference_paper.pdf` |
| **16-R.3** | Project Defense Presentation | Final presentation slides and live demonstration walkthrough | `docs/presentation_slides.pptx` |
