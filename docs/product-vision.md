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
- **Deep Public GitHub API Data:** Issues, issue comments, labels, assignees, reactions, pull requests, PR review decisions, inline & general review comments, branch/diff metadata, commits, contributors, releases, repository tags/context, and **GitHub Discussions** (target expansion).
- **Statistical DS, Generalizable ML & DL:** Time-series velocity, consistency, anomalies, change-points, contribution inequality & bus factor, Kaplan-Meier survival curves, PR/issue latency distributions, cross-domain statistical associations, generalizable text classification, per-repo semantic duplicate retrieval, and recurring topic discovery.
- **Interactive Dashboard & Visual Reports (Initial Product Experience):** Multi-section repository intelligence views, interactive charts, entity drill-downs, and consolidated diagnostic findings.

### Out of Scope
- **Source-Code Inspection:** No AST parsing, file/directory architecture mapping, cyclomatic complexity, or static code analysis.
- **CI/CD Pipeline Auditing:** No GitHub Actions workflow runs, job logs, or build-flake analytics.
- **Dependency & Security Auditing:** No SBOM extraction, lockfile parsing, Dependabot alert harvesting, or vulnerability scanning.
- **Chatbot / Conversational Assistant:** No RAG chat interface or Q&A agent in the current product scope; the initial experience centers on structured interactive dashboards and drill-down reports.
- **Benchmark-Locked Features:** No product workflow may restrict users to a fixed repository list or emit opaque `0–100` composite "health scores."

---

## 3. Role of the 4 Benchmark Repositories

The repository contains historical data and models from four large open-source projects:
1. `microsoft/vscode`
2. `pytorch/pytorch`
3. `kubernetes/kubernetes`
4. `huggingface/transformers`

These four repositories serve **strictly as offline benchmark datasets** for research, model training, calibration, and cross-domain validation. They **do not** define which repositories OpenSourceLens can analyze. Production ingestion, statistics, ML/DL models, and frontend views must generalize to **unseen repositories**.

---

## 4. Target Analytical Areas (7-Area Scope)

OpenSourceLens organizes GitHub intelligence around **7 connected areas**, extending the six implemented backend statistical domains ([analytics.md](./analytics.md)) with repository/community context and deeper per-entity data:

| # | Area | Questions Answered & Rich GitHub Data Used | Current Backend Coverage |
|---|---|---|---|
| **1** | **Repository Context & Community Profile** | Language, license, topics, archived/fork status, repo tags fallback, plus **GitHub Discussions** (volume, categories, Q&A answered rate, reactions, recurring discussion themes). | Basic repo metadata collected; **GitHub Discussions & tags fallback not yet implemented**. |
| **2** | **Development Activity & Commit History** | Rolling velocities (4w/12w/52w), completed-week consistency & CV, SMA/EWMA trajectories, rolling modified-Z anomalies, permutation change-points, `7x24` UTC rhythms, diff churn (`additions`/`deletions`), commit landing latency (`committer_date - author_date`), commit-type signals. | **Implemented (`DevelopmentActivityCalculator`)**; current standard budget samples at most `5` commit diffs, often limiting churn coverage. |
| **3** | **Contributors & Collaboration Ecosystem** | Identity-clean roster (`github_id` + unlinked `git_email` fallback, bots isolated), multi-role participation (commits, PRs, formal reviews, issues, comments), Gini inequality, 11-point Lorenz curves, Shannon entropy, bus factors (`BF50`/`BF80`), newcomers, full-history retention cohorts. | **Implemented (`ContributorIntelligenceCalculator`)**; retention cohorts require full-history ingestion. |
| **4** | **Issue & Backlog Intelligence** | Monthly inflow/outflow, pure issue stock separated from GitHub's combined PR+issue counter, resolution SLA quartiles, right-censored Kaplan-Meier survival (`S(30d)`/`S(60d)`/`S(90d)`), 5-bin open backlog aging, discussion intensity, label breakdown (`"unlabeled"` preserved), plus **assignees, reactions, NLP categorization, per-repo duplicate detection, and topic discovery**. | Statistical engine **implemented (`IssueIntelligenceCalculator`)**; assignees/reactions, per-repo embeddings, topic discovery, and 5-class OOD classification are **not yet wired**. |
| **5** | **Pull Request & Code-Review Lifecycle** | Separated response times (`TTFResp`, formal `TTFR_formal`, inline `TTFR_inline`), merge vs. abandonment outcomes, merge latency quantiles, Kaplan-Meier PR survival, diff-size turnaround buckets (`small`..`xl`), reviewer concentration, open unreviewed backlog age, plus **PR descriptions, `head`/`base` branches, draft state, labels, requested vs. actual reviewers, and diff sizes**. | Statistical engine **implemented (`PullRequestIntelligenceCalculator`)**; list collection (`/pulls`) currently leaves `additions`/`deletions`/`changed_files` `NULL` and omits body/branches/draft/requested reviewers. |
| **6** | **Release & Project Evolution** | Inter-release cadence regularity (`CV_release`), SemVer 2.0.0 adoption (`major`/`minor`/`patch`/`prerelease`), pre-release commit surges (`14d` vs. prior `60d` baseline), dormancy (`days_since_last_release`), plus release notes/description insights and git tags fallback when formal Releases are unused. | **Implemented (`ReleaseEvolutionCalculator`)**; tags fallback and release-note text analysis not yet added. |
| **7** | **Cross-Domain Discovery & Consolidated Report** | Multi-domain diagnostic scorecard, objective rule interpretations consolidated across domains, calendar-month cross-domain associations (churn vs. issues, review response vs. PR abandonment, commit velocity vs. PR merges) guarded by autocorrelation (`\|r_AR1\| > 0.40`), permutation (`B = 10,000`), and Benjamini-Hochberg FDR (`alpha = 0.05`). | **Implemented (`CrossDomainSynthesisCalculator` + `AnalyticsOrchestrator`)**; interactive frontend report not yet connected. |

---

## 5. How Data Science, Machine Learning & Deep Learning Fit In

OpenSourceLens uses **statistical analysis** for objective measurements and reserves **ML/DL** strictly for unstructured text and cross-repo pattern discovery where heuristics fall short:

1. **Statistical Data Science (Implemented Foundation):**
   - Non-parametric distributions (`p25`/`p50`/`p75`/`p90`, IQR, MAD, skewness), censored survival curves (Kaplan-Meier), permutation change-point & correlation tests, inequality coefficients (Gini, Shannon, Lorenz), and strict non-zero coercion (`NULL` stays `NULL` with explicit coverage penalties).
2. **Automated Text Categorization (DL / NLP):**
   - Purpose: Classify **unlabeled or inconsistently labeled issues** (and optionally PRs/discussions) into actionable categories (`bug`, `feature`, `documentation`, `question`, `other`) without overwriting maintainer labels.
   - Generalization requirement: Must be calibrated for **unseen repositories** and abstain cleanly on low confidence. *(Current Phase A DistilBERT/TF-IDF services only classify 3 classes `bug`/`feature`/`other` on manual API calls and are not yet integrated into post-ingestion batch scoring.)*
3. **Per-Repository Semantic Similarity & Duplicate Discovery (DL Embeddings):**
   - Purpose: Compute dense sentence embeddings (`all-MiniLM-L6-v2` or successor) dynamically **for each newly ingested repository's issues/discussions**, enabling in-repo duplicate detection and related-thread exploration.
   - *(Current Phase A `RelatedIssueService` is bound to a static 2,328-issue benchmark file `models/related_issues/embeddings.npz` and must be replaced with per-repo dynamic indexing.)*
4. **Unsupervised Topic & Theme Discovery (NLP):**
   - Purpose: Cluster a repository's own dense issue/discussion embeddings and extract representative keywords so users immediately see recurring bug areas or feature demands—requiring zero repo-specific training labels. *(Planned addition.)*
5. **Scientific Verdict on Resolution-Time Prediction:**
   - Offline Phase 8 evaluation across `2,328` benchmark issues established that exact hour-level regression at creation time explains little within-repo variance (overall holdout $R^2 = 0.276$, within-repo $R^2 = -0.04$ to $+0.18$; $71.1\%$ of explained variance came from repository governance identity). Moreover, the Phase A `ResolutionPredictor` depends on hardcoded 4-repository dummy variables (`kubernetes`, `vscode`, `pytorch`), making it invalid on unseen repositories. OpenSourceLens therefore rejects exact resolution-hour promises in favor of factual historical **Kaplan-Meier survival curves** and empirical quantiles from the submitted repository itself (retaining predictive speed tiers only if re-designed without benchmark dummies and validated via Leave-One-Repo-Out evaluation).

---

## 6. Honest System Inventory & Planned Code Cleanup

Documentation reflects both what runs in code today and what legacy prototype parts remain scheduled for deletion before building new features:

### Implemented Today (Backend)
- **URL Validation, Eligibility & Preview:** [`validator.py`](../backend/app/services/validator.py), [`repository_eligibility.py`](../backend/app/services/repository_eligibility.py), [`confidence_matrix.py`](../backend/app/services/confidence_matrix.py).
- **Durable Async Ingestion Pipeline:** [`job_queue.py`](../backend/app/services/job_queue.py), [`ingestion_worker.py`](../backend/app/services/ingestion_worker.py), [`repository_collector.py`](../backend/app/services/repository_collector.py), [`github_client.py`](../backend/app/services/github_client.py), [`rate_budgeter.py`](../backend/app/services/rate_budgeter.py).
- **6-Domain Statistical Engine, Orchestration & V2 API:** [`backend/app/services/analytics/`](../backend/app/services/analytics/), [`analytics_queue.py`](../backend/app/services/analytics_queue.py), [`analytics_worker.py`](../backend/app/services/analytics_worker.py), [`analytics.py`](../backend/app/api/v1/endpoints/analytics.py).

### Legacy / Benchmark-Coupled Components Scheduled for Later Cleanup or Replacement
- **Prototype Entrypoints & Scripts:** [`demo_app.py`](../demo_app.py) (standalone Streamlit prototype) and legacy `scripts/verify_*.py` scripts built around the initial Phase 10/11 static flow.
- **Legacy Database & Static Ingest Paths:** [`backend/app/database/ingest.py`](../backend/app/database/ingest.py), [`schema.sql`](../backend/app/database/schema.sql), [`schema_generator.py`](../backend/app/database/schema_generator.py), and the Phase A `repository_analytics` / `related_issues_cache` summary workflow.
- **Legacy Phase A API Endpoints:** `GET /api/v1/repos*`, Phase A `GET /api/v1/analytics/{overview,comparison,{owner}/{name},trends}`, and the benchmark-coupled `RelatedIssueService` / `ResolutionService` endpoints until decoupled from `embeddings.npz` and 4-repo dummy encodings.
- **Frontend Phase 10 Baseline (`frontend/`):** Currently hardcodes benchmark fallbacks (`REPO_REGISTRY`, `FALLBACK_ISSUE_CORPUS`) and targets legacy endpoints instead of the dynamic ingestion + Analytics V2 workflow.
