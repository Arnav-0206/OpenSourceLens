# OpenSourceLens — Product Vision & Analytical Scope

**OpenSourceLens** is a universal **GitHub Repository Intelligence and Analytics Platform**.

Anyone—an open-source contributor, a repository maintainer, an engineering evaluator, or a curious visitor—can submit the URL or slug (`owner/repo`) of **any eligible public GitHub repository**. OpenSourceLens harvests the rich activity, engineering, and community data exposed by GitHub's public APIs and transforms it with **statistical Data Science, Machine Learning, and Deep Learning** into clear, evidence-backed intelligence that GitHub's standard interface does not surface.

---

## 1. Why OpenSourceLens Exists

GitHub shows every repository's surface counters cleanly enough: stars, forks, open issue count, recent commits. Those vanity numbers rarely answer the practical questions people actually have before trusting, contributing to, or maintaining a project:

- **For Contributors:** *If I invest time opening a bug report or pull request here, will anyone respond? How long do PRs typically sit before a human reviews or merges them? Are there existing issues covering the same problem?*
- **For Maintainers:** *Where are our review bottlenecks? Is development sustained by an active community or dependent on one or two overloaded people? Which topics dominate our issue backlog, and what percentage of open issues have gone stale?*
- **For General Visitors & Evaluators:** *Is this project actively evolving, slowing down, or shipping sporadically? Do bursts of churn correlate with incoming bugs or PR backlogs?*

OpenSourceLens answers these questions empirically using GitHub's underlying entity streams—**not raw star/fork popularity metrics**—and presents findings honestly, complete with data coverage limits and sample-size safeguards.

---

## 2. Explicit Scope Boundaries (What We Are & What We Are Not)

To keep the platform focused and reliable, OpenSourceLens explicitly separates what belongs in the product from what does not:

### In Scope
- **Any Eligible Public GitHub Repository:** Arbitrary `github.com/{owner}/{repo}` or `owner/repo` submissions (public, non-empty repositories).
- **Deep Public GitHub API Data:** Issues, issue comments, labels, assignees, reactions, pull requests, PR review decisions, inline & general review comments, branch/diff metadata, commits, contributors, releases, repository tags fallback, and **GitHub Discussions** via GraphQL API v4.
- **Statistical DS, Generalizable ML & DL:** Time-series velocity, consistency, anomalies, change-points, contribution inequality & bus factor, Kaplan-Meier survival curves, PR/issue latency distributions, cross-domain statistical associations, generalizable 5-class text classification with calibrated abstention, per-repo semantic duplicate retrieval, and recurring topic discovery.
- **Interactive Workspace & Visual Reports:** Single-Repository Intelligence Workspace UI, interactive charts, entity drill-downs, Cross-Domain Discovery Explorer, and one-click Executive Intelligence Report export.

### Out of Scope
- **Source-Code Inspection:** No AST parsing, file/directory architecture mapping, cyclomatic complexity, or static code analysis.
- **CI/CD Pipeline Auditing:** No GitHub Actions workflow runs, job logs, or build-flake analytics.
- **Dependency & Security Auditing:** No SBOM extraction, lockfile parsing, Dependabot alert harvesting, or vulnerability scanning.
- **Chatbot / Conversational Assistant:** No RAG chat interface or conversational chatbot; the experience centers on structured interactive dashboards, visual analytics, and drill-down reports.
- **Benchmark-Locked Features:** No product workflow may restrict users to a fixed repository list or emit opaque `0–100` composite "health scores."

---

## 3. Role of the 4 Benchmark Repositories

The repository contains historical baseline data and models from four large open-source projects:
1. `microsoft/vscode`
2. `pytorch/pytorch`
3. `kubernetes/kubernetes`
4. `huggingface/transformers`

These four repositories serve **strictly as offline benchmark datasets** for research, model training, calibration, and cross-domain validation. They **do not** define which repositories OpenSourceLens can analyze. Production ingestion, statistics, ML/DL models, and frontend views generalize to **any unseen public repository**.

---

## 4. Analytical Areas (7-Area Scope & Implemented Engine)

OpenSourceLens organizes repository intelligence around **7 connected domains**, all fully implemented in the backend engine ([analytics.md](./analytics.md)):

| # | Area | Questions Answered & Rich GitHub Data Used | Backend Implementation Status |
|---|---|---|---|
| **1** | **Repository Context & Community Profile** | Language, license, topics, archived/fork status, git tags fallback, and **GitHub Discussions** (volume, categories, Q&A answered rate, reactions, recurring discussion themes). | **Fully Implemented** (Phases B & Stage 3: `AsyncGitHubClient` GraphQL v4, `DiscussionsIntelligenceCalculator`). |
| **2** | **Development Activity & Commit History** | Rolling velocities (4w/12w/52w), completed-week consistency & CV, SMA/EWMA trajectories, rolling modified-Z anomalies, permutation change-points, `7x24` UTC rhythms, diff churn (`additions`/`deletions`), commit landing latency. | **Fully Implemented** (`DevelopmentActivityCalculator`). |
| **3** | **Contributors & Collaboration Ecosystem** | Identity-clean roster (`github_id` + unlinked `git_email` fallback, bots isolated), multi-role participation (commits, PRs, formal reviews, issues, comments), Gini inequality, 11-point Lorenz curves, Shannon entropy, bus factors (`BF50`/`BF80`), newcomers, full-history retention cohorts. | **Fully Implemented** (`ContributorIntelligenceCalculator`). |
| **4** | **Issue & Backlog Intelligence** | Monthly inflow/outflow, pure issue stock separated from GitHub's combined counter, resolution SLA quantiles, right-censored Kaplan-Meier survival (`S(30d)`/`S(60d)`/`S(90d)`), 5-bin open backlog aging, discussion intensity, assignees, reactions, **generalizable 5-class NLP categorization with calibrated abstention, per-repo semantic duplicate retrieval, and topic discovery**. | **Fully Implemented** (`IssueIntelligenceCalculator`, `GeneralizableIssueClassifier`, `SemanticSimilarityService`, `TopicDiscoveryService`). |
| **5** | **Pull Request & Code-Review Lifecycle** | Separated response times (`TTFResp`, formal `TTFR_formal`, inline `TTFR_inline`), merge vs. abandonment outcomes, merge latency quantiles, Kaplan-Meier PR survival, diff-size turnaround buckets (`small`..`xl`), reviewer concentration, unreviewed backlog age, plus **PR descriptions, branches, drafts, requested reviewers, and diff sizes**. | **Fully Implemented** (`PullRequestIntelligenceCalculator`). |
| **6** | **Release & Project Evolution** | Inter-release cadence regularity (`CV_release`), SemVer 2.0.0 adoption (`major`/`minor`/`patch`/`prerelease`), pre-release commit surges (`14d` vs. prior `60d` baseline), dormancy (`days_since_last_release`), and **git tags fallback (`/tags`)**. | **Fully Implemented** (`ReleaseEvolutionCalculator`). |
| **7** | **Cross-Domain Discovery & Reports** | Multi-domain diagnostic scorecard, objective rule interpretations consolidated across domains, calendar-month cross-domain associations (`PAIR-01..03`) guarded by autocorrelation (`\|r_AR1\| > 0.40`), permutation (`B = 10,000`), and Benjamini-Hochberg FDR (`alpha = 0.05`), plus **Cross-Domain Discovery Explorer** and **Executive Intelligence Reports**. | **Fully Implemented** (`CrossDomainSynthesisCalculator`, `report_generator.py`). |

---

## 5. How Data Science, Machine Learning & Deep Learning Fit In

OpenSourceLens uses **statistical analysis** for objective measurements and reserves **ML/DL** strictly for unstructured text and pattern discovery:

1. **Statistical Data Science (Foundation):**
   - Non-parametric distributions (`p25`/`p50`/`p75`/`p90`, IQR, MAD, skewness), right-censored survival curves (Kaplan-Meier), permutation change-point & correlation tests, inequality coefficients (Gini, Shannon, Lorenz), and strict non-zero coercion (`NULL` stays `NULL` with explicit coverage penalties).
2. **Generalizable 5-Class Text Categorization (DL / NLP):**
   - Automatically classifies unlabeled issues into actionable engineering categories: `bug`, `feature`, `documentation`, `question`, and `maintenance`.
   - Implements calibrated confidence abstention (`threshold = 0.55`); low-confidence items are cleanly flagged as `unclassified` rather than guessing. Validated via Leave-One-Repo-Out (LORO) evaluation.
3. **Per-Repository Semantic Similarity & Duplicate Discovery (Sentence Transformers):**
   - Generates dense 384-dimensional sentence embeddings using `sentence-transformers/all-MiniLM-L6-v2` dynamically on demand for each newly ingested repository.
   - Embeddings are persisted in `issue_embeddings` and `discussion_embeddings` database tables as raw IEEE-754 binary BLOBs, enabling fast in-repository cosine similarity search and duplicate detection.
4. **Unsupervised Topic & Theme Discovery (Spherical KMeans & c-TF-IDF):**
   - Clusters repository dense issue embeddings on the unit sphere using Spherical KMeans and extracts characteristic keywords with c-TF-IDF, surfacing recurring bug topics and feature themes without requiring repo-specific training labels.
5. **Scientific Honesty Regarding Resolution-Time Prediction:**
   - Empirical research across benchmark issues demonstrated that exact hour-level prediction at creation time explains little within-repo variance ($71.1\%$ of explained variance came from repository identity). OpenSourceLens rejects deceptive exact resolution predictions in favor of empirical **Kaplan-Meier survival curves** and historical percentiles from the repository itself.

---

## 6. Completed Architecture Delivery

All 6 stages of the OpenSourceLens master specification are fully delivered:
- **Stage 1 (Cleanup):** Retired standalone Streamlit prototype `demo_app.py`, 4-repo dummy feature resolution predictors, static `embeddings.npz` lookup tables, and legacy Phase A endpoints.
- **Stage 2 (Backend Stabilization):** Reconciled fresh-database Alembic migrations, implemented durable worker CLIs, rate-limit auto-resumption, and verified live unseen repo ingestion.
- **Stage 3 (Entity Depth & Discussions):** Harvested PR diff metrics, branches, draft flags, reviewers, git tags fallback, and **GitHub Discussions** via GraphQL API v4 with `DiscussionsIntelligenceCalculator`.
- **Stage 4 (Workspace UI):** Delivered the Next.js 15.5 React 19 interactive Single-Repository Intelligence Workspace UI across 7 domain tabs with live ingestion tracking.
- **Stage 5 (Applied Intelligence):** Delivered generalizable 5-class NLP classification, per-repo semantic duplicate retrieval (`all-MiniLM-L6-v2`), and Spherical KMeans topic discovery.
- **Stage 6 (Production Hardening):** Delivered the Cross-Domain Discovery Explorer, one-click Executive Intelligence Report export (`.md`/`.json`), and multi-container Docker Compose production stack.
