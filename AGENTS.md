# OpenSourceLens — Agent Instructions & Source of Truth

## Project Identity

**OpenSourceLens** is a **Data Science-powered GitHub Repository Intelligence Platform**. 
It collects publicly available data for **any eligible public GitHub repository** and applies data engineering, statistical analysis, machine learning, and natural language processing to discover meaningful patterns, trends, bottlenecks, and insights about open-source software projects.

The platform goes beyond raw surface-level statistics (such as total commit counts or contributor counts) to derive actionable intelligence:
- **Development Activity:** Activity consistency, velocity changes, anomaly detection, and trend analysis.
- **Contributor Ecosystem:** Contributor retention, cohort analysis, contribution concentration (Gini coefficient/entropy), and participation patterns.
- **Pull Request Intelligence:** Review turnaround, merge latency, PR lifecycle bottlenecks, and review dynamics.
- **Issue Intelligence:** Multi-category NLP classification, semantic duplicate discovery, backlog aging, and resolution velocity.
- **Repository-Level Synthesis:** Cross-domain metrics synthesis, health diagnostics, and evidence-backed findings.

The web dashboard is the interactive presentation layer; the core of the project is the **Data Science and Machine Learning workflow**.

---

## Benchmark Repositories vs. Product Scope

### Crucial Distinction
- **Product Scope:** The application must accept and analyze **ANY eligible public GitHub repository** specified by a user (via URL or `owner/repo`). The primary product experience is a **Single-Repository Intelligence Workspace**.
- **Benchmark / Sample Datasets:** The four initial repositories:
  1. `microsoft/vscode` (Developer Tools & IDE)
  2. `pytorch/pytorch` (Machine Learning Framework)
  3. `kubernetes/kubernetes` (Cloud Infrastructure)
  4. `huggingface/transformers` (NLP & Generative AI)
  
  These repositories serve as **offline benchmark datasets**, experimentation baselines, and cross-domain validation sets. They do **NOT** define the operational boundary of the platform. All production pipelines, analytical models, and user interfaces must be designed for **generalization to unseen repositories**.

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

## Core Analytical Domains

| # | Domain | Core Techniques | Key Deliverables |
|---|---|---|---|
| 1 | **Dynamic Ingestion** | GitHub REST API, rate-limit budgeting, pagination | On-demand repository ingestion, raw payload storage, schema extraction |
| 2 | **Development Activity** | Time-series decomposition, rolling averages, change-point detection | Commit velocity trends, activity consistency score, release frequency |
| 3 | **Contributor Ecosystem** | Gini inequality, cohort analysis, churn modeling | Contributor retention curves, bus factor / concentration metric, newcomer activity |
| 4 | **Pull Request Intelligence** | Survival analysis (Kaplan-Meier), IQR horizons | Time-to-first-review, merge duration percentiles, PR bottleneck detection |
| 5 | **Issue Intelligence** | Fine-tuned DistilBERT, Sentence-Transformers, RF velocity | Transferrable issue categorization, semantic duplicate detection, resolution horizons |
| 6 | **Repository Synthesis** | Multi-metric aggregation, heuristic synthesis | Automated evidence-backed repository intelligence report |

---

## Roadmap Status

| Phase | Title | Focus | Status |
|---|---|---|---|
| 0 | Project Definition & Planning | Architecture, problem formulation, benchmark definition | ✅ Complete |
| 1 | Benchmark Data Collection Pipeline | Harvest initial 4-repository sample dataset (2,328 issues) | ✅ Complete |
| 2 | Data Cleaning & Dataset Construction | Label normalization, text sanitization, stratified splits | ✅ Complete |
| 3 | Exploratory Data Analysis | Vocabulary, class balance, Kaplan-Meier resolution curves | ✅ Complete |
| 4 | Classical Machine Learning Baseline | TF-IDF + Logistic Regression & Linear SVM baselines | ✅ Complete |
| 5 | NLP Enhancement | Sublinear TF-IDF, n-grams, regex cleaning | ✅ Complete |
| 6 | Deep Learning & Transformer Model | Fine-tuned DistilBERT champion (81.3% test accuracy) | ✅ Complete |
| 7 | Related Issue Detection | Dense sentence embeddings & cosine similarity retrieval | ✅ Complete |
| 8 | Resolution Analysis & Prediction | Empirical SLA distributions & Random Forest velocity | ✅ Complete |
| 9 | Database & Backend | SQLite (WAL) / PostgreSQL schema, FastAPI endpoints | ✅ Complete |
| 10 | Frontend UI Prototype | Initial Next.js dashboard & AI workspace components | ✅ Complete |
| 11 | Initial Benchmark Integration | Connected local UI $\leftrightarrow$ API $\leftrightarrow$ DB $\leftrightarrow$ Models on sample data | ✅ Complete |
| **11-R** | **Product Scope Realignment** | **Architecture & documentation correction for dynamic any-repo platform** | 🔄 **Current Focus** |
| 12-R | Dynamic Ingestion & Eligibility Engine | URL validation, on-demand GitHub ingestion, rate budgeting | Pending |
| 13-R | Expanded Analytics Pipelines | Development activity, contributor ecosystem, PR lifecycle | Pending |
| 14-R | Model Generalization & Decoupling | Decouple models from 4-repo dummies; test on unseen repos | Pending |
| 15-R | Dynamic Single-Repo Workspace UI | URL input bar, progress telemetry, multi-domain intelligence views | Pending |
| 16-R | Deployment & Final Evaluation | Production hosting, research paper, and project defense | Pending |
