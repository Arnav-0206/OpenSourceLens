# OpenSourceLens

**Deep, evidence-backed intelligence for any public GitHub repository.**

Submit a public GitHub URL or `owner/repo` slug, and OpenSourceLens collects rich underlying engineering and community data exposed by GitHub's public APIs—commits, contributors, issues, comments, pull requests, code reviews, releases, and (planned) discussions—and analyzes it using **statistical Data Science, Machine Learning, and Deep Learning**.

Instead of repeating surface-level GitHub counters (stars, forks, raw totals), OpenSourceLens helps **contributors**, **maintainers**, and **anyone evaluating a repository** understand how a project actually behaves beneath the surface.

> **Start here:** Read [docs/product-vision.md](./docs/product-vision.md) for the complete, jargon-free explanation of what OpenSourceLens covers, what is out of scope, what already runs today, and the stepwise execution roadmap.

---

## What OpenSourceLens Helps You Understand

- **If you want to contribute:** *How quickly do human maintainers actually respond to pull requests? Which issues stay open for months versus closing within days? Have other contributors already filed duplicate or closely related issues?*
- **If you maintain a repository:** *Is development momentum speeding up or slowing down? Does the project depend too heavily on one or two contributors (`BF50` / Gini concentration)? Which topics keep recurring across open issues?*
- **If you are evaluating a project:** *Are releases predictable (`CV_release`, SemVer compliance) or sporadic? Do spikes in churn or commit velocity align with incoming bug reports or PR review delays?*

---

## Scope Boundaries (In Scope vs. Out of Scope)

| In Scope | Explicitly Out of Scope |
|---|---|
| Any eligible **public** GitHub repository (`owner/repo` or URL) | Static source-code inspection, AST parsing, or code-complexity auditing |
| Rich public **GitHub API** telemetry: issues, comments, labels, assignees, reactions, PRs, PR reviews, inline review comments, branches/diffs, commits, contributors, releases, and **GitHub Discussions** (planned) | GitHub Actions / CI-CD workflow log analytics |
| Statistical analytics, survival curves, inequality/bus factor, generalizable issue/PR classification, per-repo duplicate similarity, and topic discovery | Dependency graph / SBOM / security vulnerability scanning |
| Interactive analytical dashboards, visual charts, and consolidated reports | Conversational AI / RAG chatbot interfaces |

The four repositories bundled in `data/` (`microsoft/vscode`, `pytorch/pytorch`, `kubernetes/kubernetes`, `huggingface/transformers`) are strictly **offline benchmark datasets** used for research and model validation—never product boundaries.

---

## Core Intelligence Areas & How DS / ML / DL Fit In

OpenSourceLens separates **objective statistical measurement** from **applied Machine Learning / Deep Learning** so insights stay honest, reproducible, and free of inflated scores:

1. **Repository Context & Community Profile** *(Planned Expansion)* — Public repo metadata, tags fallback, and **GitHub Discussions** (volume, categories, answered Q&A rates, recurring community topics).
2. **Development Activity & Commit History** — Rolling 4w/12w/52w commit velocities, completed-week consistency & CV, rolling anomalies (robust MAD / z-score / Poisson), `max_ssb` permutation change-points, `7x24` UTC rhythms, diff-aware code churn, and commit landing latency.
3. **Contributors & Collaboration Ecosystem** — Identity-clean contributor mapping (`github_id` + unlinked email fallback, bots separated), multi-role counts (commits, PRs, reviews, issues, comments), Gini inequality, 11-point Lorenz deciles, Shannon entropy, bus factors (`BF50` / `BF80`), and history-gated cohort retention.
4. **Pull Request & Code-Review Lifecycle** — Separated response metrics (`TTFResp`, formal `TTFR_formal`, inline `TTFR_inline`), merge vs. abandonment rates, merge latency quantiles, Kaplan-Meier PR survival, diff-size turnaround buckets, reviewer load concentration, plus planned PR descriptions/branches/draft/diff depth.
5. **Issue & Backlog Intelligence** — Monthly inflow/outflow, pure issue backlog separation, timestamp-derived SLA percentiles, right-censored Kaplan-Meier survival (`S(30d)`/`S(60d)`/`S(90d)`), 5-bin backlog aging, discussion intensity, canonical categorization (NLP), plus planned per-repo semantic duplicate retrieval and unsupervised topic discovery.
6. **Release & Project Evolution** — Inter-release cadence predictability (`CV_release`), SemVer 2.0.0 adoption, pre-release commit surges (`14d` vs. prior `60d`), and release dormancy (+ planned tags/release notes).
7. **Cross-Domain Discovery & Consolidated Report** — Multi-domain diagnostic scorecard, deterministic consolidated findings, and monthly cross-domain Spearman permutation tests (`B = 10,000`, `|r_AR1| > 0.40` autocorrelation guard, Benjamini-Hochberg FDR `alpha = 0.05`).

---

## Honest Project State: Implemented Foundation & Roadmap Status

- **Fully Implemented Architecture (Stages 1 through 6 Completed):**
  - **Phase B Dynamic Ingestion (Stage 3):** Full GitHub REST API v3 entity harvesting (commits, contributors, issues, comments, labels, assignees, reactions, pull requests, reviews, review comments, diff churn stats, branches, releases) and **GitHub Discussions** with pagination, rate-budgeting (`fast_snapshot` 10 reqs / `standard_diagnostic` 50 reqs), and automatic post-ingestion analytics enqueueing.
  - **Phase C Analytics Engine (Stage 2):** Six-domain statistical calculators (`DevelopmentActivity`, `ContributorIntelligence`, `PullRequestIntelligence`, `IssueIntelligence`, `ReleaseEvolution`, `CrossDomainSynthesis`), snapshot-isolated `AnalyticsOrchestrator`, `AnalyticsWorker`, and Analytics V2 API endpoints (`/api/v1/analytics/{owner}/{name}/...`).
  - **Interactive Workspace UI (Stage 4):** Next.js 15 App Router interface featuring domain tabs (Overview, Activity, Contributors, Pull Requests, Issues, Releases, Discussions), live ingestion tracker, and confidence penalty audit logs.
  - **Applied Intelligence & Backlog Discovery (Stage 5):** Generalizable 5-class NLP issue classifier with calibrated abstention, per-repo semantic duplicate retrieval via dense vector embeddings (`all-MiniLM-L6-v2`), and unsupervised Spherical KMeans topic discovery with c-TF-IDF keyword extraction.
  - **Cross-Domain Explorer & Deployment (Stage 6):** Cross-Domain Discovery Explorer (`PAIR-01..03` Spearman $r_s$, Monte Carlo permutation $p$, BH FDR $q$, AR(1) guardrails), one-click Executive Intelligence Report export (`.md` and `.json`), and a multi-container Docker Compose production stack.
- **Verification Status:** Complete suite of **414 passed, 3 skipped** automated tests spanning all migrations, models, collectors, calculators, and API routes.

---

## Quickstart: Production Multi-Container Docker Stack

OpenSourceLens provides an automated multi-container Docker Compose stack orchestrating PostgreSQL 16 Alpine, FastAPI Backend, Background AnalyticsWorker, and Next.js 15 Standalone Frontend.

```bash
# 1. Clone repository and set optional GitHub token (for 5,000 req/hr API budget)
export GITHUB_TOKEN="ghp_your_personal_access_token"

# 2. Start full production stack
docker compose up --build
```

Access the platform:
- **Interactive Workspace UI:** [http://localhost:3000](http://localhost:3000)
- **FastAPI Backend & Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **System Health Probe:** [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

---

## Local Development Setup

### 1. Python Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r backend\requirements.txt
```

### 2. Frontend Development Server
```powershell
cd frontend
npm install
npm run dev
# Running on http://localhost:3000
```

### 3. Running Backend Services
```powershell
# Start FastAPI backend server (http://127.0.0.1:8000)
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload

# Start AnalyticsWorker daemon (processes background analytical pipelines)
.\.venv\Scripts\python.exe -m backend.app.services.analytics_worker
```

### 4. Running Tests
```powershell
# Run the complete test suite (414 unit and integration tests)
.\.venv\Scripts\python.exe -m pytest tests/ -v
```
