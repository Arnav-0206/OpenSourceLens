# OpenSourceLens — Agent Instructions & Source of Truth

## Project Identity

**OpenSourceLens** is a universal **GitHub Repository Intelligence and Analytics Platform**.
It collects rich publicly accessible telemetry for **any eligible public GitHub repository** (`owner/repo` or GitHub URL) from GitHub's APIs and applies Data Science, statistical analysis, Machine Learning, and Deep Learning to uncover meaningful patterns, trends, bottlenecks, and insights that GitHub's standard interface does not surface.

See [docs/product-vision.md](./docs/product-vision.md) for the canonical product definition and scope boundaries. Briefly:
- **In Scope:** Repository & community context (including planned **GitHub Discussions**), commit & development activity, contributor ecosystem, pull requests & code reviews, issues & backlog intelligence (including generalizable NLP categorization, per-repo semantic duplicate discovery, and topic discovery), release evolution, cross-domain synthesis, and an interactive analytical dashboard.
- **Explicitly Out of Scope:** Source-code/AST inspection or complexity auditing, GitHub Actions CI/CD pipeline analytics, dependency/SBOM vulnerability scanning, and conversational AI/chatbot interfaces.
- **Presentation Layer:** The web dashboard is the interactive presentation layer; the core is the statistical Data Science and generalizable ML/NLP workflow.

---

## Benchmark Repositories vs. Product Scope

### Crucial Distinction
- **Product Scope:** The application must accept and analyze **ANY eligible public GitHub repository** specified by a user (via URL or `owner/repo`). The primary product experience is an interactive **Single-Repository Intelligence Workspace**.
- **Benchmark / Sample Datasets:** The four initial repositories (`microsoft/vscode`, `pytorch/pytorch`, `kubernetes/kubernetes`, `huggingface/transformers`) serve strictly as **offline benchmark datasets**, experimentation baselines, and cross-domain validation sets. They do **NOT** define the operational boundary of the platform. All production pipelines, analytical models, and user interfaces must generalize to unseen repositories (benchmark-coupled legacy prototype files—such as `demo_app.py`, static `embeddings.npz` lookup, 4-repo dummy encoders, legacy Phase A endpoints, and Phase 10 hardcoded UI registries—are slated for cleanup).

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
- **GitHub REST API v3** — Dynamic data source

---

## Core Analytical Domains (Target Vision & Implemented Engine)

| # | Domain / Area | Core Techniques & Data | Backend Implementation Status |
|---|---|---|---|
| 1 | **Dynamic Ingestion & Repo Context** | URL validation, eligibility gating, rate-limit budgeting, pagination, rich GitHub entity harvesting (+ planned **GitHub Discussions**) | Phase B REST ingestion implemented; Discussions, PR diffs/body/branches, and issue assignees/reactions pending |
| 2 | **Development Activity** | Velocity trends, completed-week consistency, rolling anomalies, permutation change-points, UTC rhythms, diff churn, landing latency | Phase C.2 implemented (`DevelopmentActivityCalculator`) |
| 3 | **Contributor Ecosystem** | Identity resolution, bot isolation, Gini inequality, Lorenz deciles, Shannon entropy, `BF50`/`BF80`, multi-role counts, gated cohorts | Phase C.3 implemented (`ContributorIntelligenceCalculator`) |
| 4 | **Pull Request Intelligence** | `TTFResp` / `TTFR_formal` / `TTFR_inline`, outcomes, merge latency quantiles, Kaplan-Meier survival, diff-size bins, reviewer bottlenecks | Phase C.4 implemented (`PullRequestIntelligenceCalculator`) |
| 5 | **Issue & Backlog Intelligence** | Monthly flow, pure stock isolation, SLA quantiles, Kaplan-Meier survival, aging, discussion intensity + planned **per-repo duplicate retrieval, topic discovery, OOD classification** | Phase C.5 statistical engine implemented; Phase A ML services not yet generalized/wired |
| 6 | **Release Evolution** | SemVer 2.0.0 adoption, cadence regularity (`CV_release`), pre-release commit surges, dormancy (+ planned git tags fallback & notes) | Phase C.6 implemented (`ReleaseEvolutionCalculator`) |
| 7 | **Repository Synthesis** | Multi-domain diagnostic scorecard, consolidated rule findings, calendar-month Spearman permutation tests with `\|r_AR1\|` & BH FDR guards | Phase C.6–C.7 implemented (`CrossDomainSynthesisCalculator`) |

---

## Current Status & Roadmap Reference

- **Implemented Backend Foundation:** Phase B (Dynamic GitHub Ingestion) and Phase C.1–C.8.3 (6-Domain Statistical Engine, Orchestration, `AnalyticsWorker`, and Analytics V2 API) are implemented in `backend/`.
- **Active Alignment Track:** Documentation and product scope are anchored in [docs/product-vision.md](./docs/product-vision.md). Legacy prototype cleanup, ingestion richness, generalizable ML/NLP, and the interactive dashboard follow next. For milestone history and upcoming revisions, see [docs/roadmap.md](./docs/roadmap.md) (to be revised in a dedicated step).
