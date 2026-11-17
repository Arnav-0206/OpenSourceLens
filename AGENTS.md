# OpenSourceLens — Agent Instructions & Source of Truth

## Project Identity

**OpenSourceLens** is a universal **GitHub Repository Intelligence and Analytics Platform**.
It collects rich publicly accessible telemetry for **any eligible public GitHub repository** (`owner/repo` or GitHub URL) from GitHub's APIs and applies Data Science, statistical analysis, Machine Learning, and Deep Learning to uncover meaningful patterns, trends, bottlenecks, and insights that GitHub's standard interface does not surface.

See [docs/product-vision.md](./docs/product-vision.md) for the canonical product definition and scope boundaries. Briefly:
- **In Scope:** Repository & community context (including **GitHub Discussions** GraphQL v4), commit & development activity, contributor ecosystem, pull requests & code reviews (including diff stats and branches), issues & backlog intelligence (including generalizable 5-class NLP categorization with calibrated abstention, per-repo semantic duplicate discovery, and topic discovery), release evolution (including git tags fallback), cross-domain synthesis, interactive workspace dashboard, and executive reports.
- **Explicitly Out of Scope:** Source-code/AST inspection or complexity auditing, GitHub Actions CI/CD pipeline analytics, dependency/SBOM vulnerability scanning, and conversational AI/chatbot interfaces.
- **Presentation Layer:** The web dashboard is the interactive presentation layer; the core is the statistical Data Science and generalizable ML/NLP workflow.

---

## Benchmark Repositories vs. Product Scope

### Crucial Distinction
- **Product Scope:** The application must accept and analyze **ANY eligible public GitHub repository** specified by a user (via URL or `owner/repo`). The primary product experience is an interactive **Single-Repository Intelligence Workspace**.
- **Benchmark / Sample Datasets:** The four initial repositories (`microsoft/vscode`, `pytorch/pytorch`, `kubernetes/kubernetes`, `huggingface/transformers`) serve strictly as **offline benchmark datasets**, experimentation baselines, and cross-domain validation sets. They do **NOT** define the operational boundary of the platform. All production pipelines, analytical models, and user interfaces generalize to unseen repositories (all legacy prototype files, 4-repo dummy encoders, and static lookups were cleanly removed in Stage 1).

---

## Source-of-Truth Hierarchy

When interpreting project decisions, use this priority order:

1. **Explicit user instructions** in the current interaction
2. **Canonical documentation** in `docs/` (including the revised roadmap and architecture)
3. **Existing implementation**, configuration, tests, and repository conventions
4. **Root-level contextual documents** in the parent `DS-PROJECT/` directory (Word documents from planning sessions)
5. **General technical knowledge** and recommendations

If sources disagree, identify the conflict, explain why it matters, and ask the user to resolve it before implementing.

---

## Development Protocol

### Strict Non-Destructive Working Protocol
- **No Coding without Approval:** Do not write code or alter implementation files without an approved implementation plan.
- **Preserve Existing Research & Work:** Do not delete existing datasets, ML models, exploratory scripts, or working components without explicit justification and user consent.
- **Phase-by-Phase Execution:** Implement only the step explicitly requested by the user.
- **Plan Before Every Change:** Before modifying any project file:
  1. Inspect relevant documentation and current code.
  2. Produce a downloadable implementation-plan artifact.
  3. **Wait for user approval** before executing.
- **After Implementation:** Provide a structured final report covering step number, status, files changed, summary, verification results, limitations, user actions needed, and next step.

---

## Anti-Hallucination & Data Science Integrity Rules

1. Never claim to have read, run, tested, or deployed something unless you actually did.
2. Never invent statistics, metrics, API responses, or file paths.
3. Inspect existing code before modifying it.
4. Distinguish facts from assumptions, estimates, and recommendations.
5. If a dependency, credential, or dataset is missing, report it.
6. Never fabricate placeholder metrics as real analytical results.
7. Do not silently change project requirements or scope.
8. If you cannot verify something, say so.
9. When implementation fails, investigate the actual error before applying fixes.
10. Prevent data leakage — labels used as targets must not be model input features.
11. Models must be evaluated for **out-of-distribution generalization** on unseen repositories, not just holdout sets from the benchmark repositories.

---

## Architecture & Technology Stack

### Technology Stack
- **Python 3.10+** — Primary language for data collection, processing, statistical modeling, and ML/NLP
- **Pandas & NumPy** — Data manipulation, feature engineering, and matrix operations
- **scikit-learn** — TF-IDF vectorization, Logistic Regression, Linear SVM, Random Forest, Gini calculation
- **PyTorch & Hugging Face Transformers** — Fine-tuned DistilBERT transformer sequence classification
- **Sentence Transformers** — Dense semantic embeddings (`all-MiniLM-L6-v2`) for issue similarity retrieval
- **FastAPI** — High-performance backend API serving dynamic data collection and real-time inference
- **SQLAlchemy 2.0 & SQLite (WAL mode) / PostgreSQL** — Structured data persistence, indexing, and prediction audit logging
- **Next.js 15.5 & TypeScript 5.7** — React 19 App Router frontend with Tailwind CSS 3.4 and Recharts 3.10
- **GitHub REST API v3 & GraphQL API v4** — Dynamic data sources
- **Docker & Docker Compose** — Multi-container production deployment (PostgreSQL 16, Backend, Worker, Frontend)

---

## Core Analytical Domains (Target Vision & Implemented Engine)

| # | Domain / Area | Core Techniques & Data | Backend Implementation Status |
|---|---|---|---|
| 1 | **Dynamic Ingestion & Repo Context** | URL validation, eligibility gating, rate-limit budgeting, pagination, rich GitHub entity harvesting, git tags fallback, and **GitHub Discussions** via GraphQL API v4 | **Fully Implemented** (Phases B & Stage 3: `AsyncGitHubClient`, `RepositoryCollector`, `DurableJobQueue`, `IngestionWorker`) |
| 2 | **Development Activity** | Velocity trends, completed-week consistency, rolling anomalies, permutation change-points, UTC rhythms, diff churn, landing latency | **Fully Implemented** (`DevelopmentActivityCalculator`) |
| 3 | **Contributor Ecosystem** | Identity resolution, bot isolation, Gini inequality, Lorenz deciles, Shannon entropy, `BF50`/`BF80`, multi-role counts, gated cohorts | **Fully Implemented** (`ContributorIntelligenceCalculator`) |
| 4 | **Pull Request Intelligence** | `TTFResp` / `TTFR_formal` / `TTFR_inline`, outcomes, merge latency quantiles, Kaplan-Meier survival, diff-size bins (`small..xl`), reviewer bottlenecks, branches, drafts, requested reviewers | **Fully Implemented** (`PullRequestIntelligenceCalculator`) |
| 5 | **Issue & Backlog Intelligence** | Monthly flow, pure stock isolation, SLA quantiles, Kaplan-Meier survival, aging, discussion intensity, generalizable 5-class NLP classification, per-repo dynamic semantic duplicate retrieval, and unsupervised Spherical KMeans topic discovery | **Fully Implemented** (`IssueIntelligenceCalculator`, `GeneralizableIssueClassifier`, `SemanticSimilarityService`, `TopicDiscoveryService`) |
| 6 | **Release Evolution** | SemVer 2.0.0 adoption, cadence regularity (`CV_release`), pre-release commit surges, dormancy, and git tags fallback (`/tags`) | **Fully Implemented** (`ReleaseEvolutionCalculator`) |
| 7 | **Repository Synthesis & Reports** | Multi-domain diagnostic scorecard, community discussions card, consolidated rule findings, calendar-month Spearman permutation tests with `\|r_AR1\|` & BH FDR guards, Cross-Domain Discovery Explorer, and Executive Report Generator (`.md`/`.json`) | **Fully Implemented** (`CrossDomainSynthesisCalculator`, `report_generator.py`) |

---

## Current Status & Roadmap Reference

- **Completed Platform Delivery (Stages 1 through 6):** All six stages from the master specification are 100% delivered, tested (414 passing tests), and operational end-to-end:
  - **Stage 1:** Strict prototype cleanup and benchmark decoupling.
  - **Stage 2:** Backend stabilization, migration reconciliation, and live unseen repository E2E verification.
  - **Stage 3:** Rich GitHub entity depth (PR diff stats, branches, bodies, draft status), git tags fallback, and GitHub Discussions (GraphQL v4).
  - **Stage 4:** Next.js 15.5 React 19 App Router interactive Single-Repository Intelligence Workspace UI (7 domain tabs, live progress tracker).
  - **Stage 5:** Generalizable 5-class NLP issue classifier with calibrated abstention, per-repo dynamic semantic duplicate retrieval (`all-MiniLM-L6-v2`), and Spherical KMeans topic discovery with c-TF-IDF keyword extraction.
  - **Stage 6:** Cross-Domain Discovery Explorer (`PAIR-01..03`), Executive Report Generator (`.md` and `.json`), and multi-container Docker Compose production stack.
  - **Ingestion Hardening:** Automatic `.env` loading with `python-dotenv`, embedded workers enabled by default in FastAPI `lifespan()`, GraphQL `answerChosenAt` schema alignment, and benchmark test isolation for dynamic repository ingestion.
- **Reference Documentation:** Anchored in [docs/product-vision.md](./docs/product-vision.md) and [docs/roadmap.md](./docs/roadmap.md).
