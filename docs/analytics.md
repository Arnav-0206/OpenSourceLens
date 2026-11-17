# OpenSourceLens — Analytics Engine & Applied Intelligence Specification

OpenSourceLens implements a seven-domain statistical and applied machine learning analytics engine (`methodology_version = "2026.2"`) built on [`BaseMetricCalculator`](../backend/app/services/analytics/base.py) and coordinated by [`AnalyticsOrchestrator`](../backend/app/services/analytics/orchestrator.py).

> **Core Invariants:**
> 1. **Snapshot & Reference-Time Freezing:** Every domain calculator reads inside a single snapshot transaction pinned to the worker's claimed `analytics_source_revision` and frozen UTC `analysis_reference_time` (`ref_time`). Any rows timestamped after `ref_time` are strictly excluded.
> 2. **Honesty & No Fabricated Zeros:** Missing diffs, uncollected review streams, and small samples report explicit analytical statuses (`insufficient`, `empty`, `diff_metrics_not_collected`, etc.) rather than coercing `None` to `0.0`.

---

## 1. Orchestration, Validation & Publication Flow

[`AnalyticsOrchestrator`](../backend/app/services/analytics/orchestrator.py) executes the full pipeline in fixed sequence:

1. **Snapshot Open (`AnalyticsJobQueue.open_analysis_snapshot`):** Opens a dedicated read snapshot transaction pinned to `repository_id` and locks `analysis_reference_time` in UTC.
2. **Six Base Calculators:** Runs `development_activity`, `contributor_ecosystem`, `pull_request_intelligence`, `issue_intelligence`, `release_evolution`, and `discussions_intelligence`. Missing domains produce explicit `unavailable` calculation results (`penalty = 1.0`, `RULE-ORCH-UNAVAILABLE`).
3. **Cross-Domain Synthesis (`CrossDomainSynthesisCalculator`):** Receives precomputed domain results to construct the Multi-Domain Diagnostic Scorecard and executes calendar-month cross-domain correlation tests (`PAIR-01..03`) without re-querying domain data.
4. **Publication Bundle Assembly & Pydantic Validation:**
   - Embeds `release_evolution` and `discussions_intelligence` inside `repository_synthesis` to strictly preserve the top-level 5-key [`AnalysisPayload`](../backend/app/schemas/analytics_v2.py) contract: `development_activity`, `contributor_ecosystem`, `pull_request_intelligence`, `issue_intelligence`, and `repository_synthesis`.
   - Computes domain scores: `domain_score = round(max(0.0, 1.0 - confidence_penalty), 3)`.
   - Aggregates overall confidence as the unweighted mean across all 7 domains: `round(sum(domain_scores) / 7, 3)`.
   - Classifies confidence into bands: `high` (>= 0.80), `medium` (0.60–0.79), `low` (0.40–0.59), and `unreliable` (< 0.40 or all core empty).
5. **Rule Consolidation:** Deduplicates on `(rule_id, domain, finding)` across all domains and sorts deterministically by severity (`critical` < `warning` < `positive` < `info`), `domain`, `rule_id`, and `finding`.
6. **Optimistic Lease-Fenced Commit:** Verifies the worker lease and ensures the repository's `analytics_source_revision` was not superseded before atomically publishing to `analysis_runs`.

---

## 2. Domain 1: Development Activity (`development_activity`)

Source: [`backend/app/services/analytics/development_activity.py`](../backend/app/services/analytics/development_activity.py)

- **Velocity Trajectories:** Rolling day-window commit velocities: `trailing_4w_velocity = commits(> ref_time - 28d) / 4.0`, `trailing_12w_velocity = commits(> ref_time - 84d) / 12.0`, `trailing_52w_velocity = commits(> ref_time - 364d) / min(52.0, days / 7.0)`.
- **Completed-Week Consistency:** Excludes the ongoing partial week. Computes mean, median, sample stdev, IQR, MAD, and `CV = stdev / mean`, classifying regularity into `consistent_steady`, `consistently_low`, `moderately_variable`, `highly_volatile`, or `sparse_intermittent`.
- **Rolling Weekly Trajectories:** 4-week and 12-week SMA, rolling median, and 4-week EWMA (`alpha = 0.40`).
- **Anomaly Detection:** Modified Z-score against a trailing 6–8 completed-week baseline using `scale = max(1.4826 * MAD, 1.0)` with Poisson floor guardrails. Flags anomalous activity surges (`score >= 3.0`) and activity drops (`score <= -2.5`).
- **Permutation Change-Point Detection:** Single-split `max_ssb_permutation_test` (`B = 999` deterministic permutations, `min_segment = 5`, `p_perm <= 0.01`, `abs_mean_shift >= 3.0`).
- **Activity Rhythms:** 7x24 weekday-hour commit matrix in UTC.
- **Diff-Aware Code Churn:** Non-null `additions + deletions` with explicit `diff_coverage_ratio` tracking.
- **Landing Latency:** Hours between `committer_date` and `author_date` (isolating clock-skew negatives).
- **Rule IDs:** `RULE-ACT-EMPTY`, `RULE-ACT-001` (recency), `RULE-ACT-002` (velocity trend), `RULE-ACT-003` (release cadence), `RULE-ACT-004` (consistency), `RULE-ACT-005` (anomalies), `RULE-ACT-006` (change points), `RULE-ACT-007` (rhythms), `RULE-ACT-008` (churn), `RULE-ACT-009` (landing latency).

---

## 3. Domain 2: Contributor Ecosystem (`contributor_ecosystem`)

Source: [`backend/app/services/analytics/contributors.py`](../backend/app/services/analytics/contributors.py)

- **Actor Identity Resolution:** Anchors on immutable `github_id` (confidence 1.0) -> verified `Contributor` login (0.9/1.0) -> unlinked `git_email` (0.7). Isolates bot accounts (`is_bot` flag or `[bot]` suffix) under `identity_hygiene`.
- **Inequality & Concentration Metrics:**
  - Gini coefficient: $G = \frac{2 \sum_{i=1}^n i \cdot x_{(i)}}{n \sum x_i} - \frac{n + 1}{n}$ clamped to `[0.0, 1.0]`.
  - Standardized 11-point Lorenz deciles (`u in {0.0, 0.1, ..., 1.0}`).
  - Shannon entropy $H = -\sum p_i \log_2 p_i$ and normalized entropy $H / \log_2(n)$.
  - Bus factors `BF50` and `BF80` (minimum contributors covering >= 50% and >= 80% of commits).
- **Multi-Role Participation:** Tracks commits, authored PRs, formal reviews, opened issues, and comments per actor.
- **Retention Cohorts:** Monthly return rates gated behind verified full-history ingestion (`mode == "full_history"`).
- **Rule IDs:** `RULE-CON-EMPTY`, `RULE-CON-001` (bus factor risk), `RULE-CON-002` (high Gini inequality >= 0.75), `RULE-CON-003` (bot filtering transparency), `RULE-CON-004` (cohort status).

---

## 4. Domain 3: Pull Request Intelligence (`pull_request_intelligence`)

Source: [`backend/app/services/analytics/pull_requests.py`](../backend/app/services/analytics/pull_requests.py)

- **Separated Responsiveness Channels:** Excludes self-interactions and bot actions:
  - `TTFResp` (Time-to-First-Response): Minimum elapsed hours across formal reviews, inline review comments, and conversation comments.
  - `TTFR_formal`: Minimum hours across formal reviews (`APPROVED`, `CHANGES_REQUESTED`, `COMMENTED`).
  - `TTFR_inline`: Minimum hours across inline diff comments.
- **Outcome Distribution:** Merge rate, abandonment rate, and open ratio.
- **Merge Latency:** Percentiles (`p25`, `p50`, `p75`, `p90`, `mean`) for merged pull requests.
- **Kaplan-Meier Survival:** Product-limit estimator `compute_kaplan_meier_curve` with open PRs right-censored at `ref_time - created_at`, reporting `s7d`, `s30d`, and median survival days.
- **Diff Size vs. Turnaround:** Buckets resolved PRs into `small` (0–50 lines), `medium` (51–250), `large` (251–1000), and `xl` (>1000).
- **Bottlenecks:** Review latency share (`median(TTFResp) / median(Merge Latency)`), top reviewer concentration, and unreviewed PR backlog age.
- **Rule IDs:** `RULE-PR-EMPTY`, `RULE-PR-001` (slow response / review bottleneck), `RULE-PR-002` (abandonment rate), `RULE-PR-003` (reviewer concentration), `RULE-PR-004` (diff size turnaround ratio), `RULE-PR-005` (stream coverage transparency).

---

## 5. Domain 4: Issue & Maintenance Intelligence (`issue_intelligence`)

Source: [`backend/app/services/analytics/issues.py`](../backend/app/services/analytics/issues.py)

- **Observed Monthly Flow:** Monthly created count, closed count, net flow, and flow ratio (`closed / created`).
- **Pure Stock Separation:** Distinguishes pure open issues from GitHub's combined issue+PR counter.
- **Resolution Distribution:** Exact resolution hours from timestamps (`p25`, `p50`, `p75`, `p90`, `mean`, skewness) and speed tier categorization (`< 1 Day`, `1-7 Days`, `7-30 Days`, `30+ Days`).
- **Kaplan-Meier Survival:** Right-censored survival at `ref_time`, reporting `s30d`, `s60d`, `s90d`, and median survival days.
- **Backlog Aging:** 5-bin open issue aging: `fresh` (<7d), `active` (7–30d), `aging` (31–90d), `stale` (91–365d), and `dormant` (>365d).
- **Discussion Intensity:** Median and IQR comment counts per issue, plus guarded non-causal Spearman correlation between comment volume and resolution latency.
- **Rule IDs:** `RULE-ISS-EMPTY`, `RULE-ISS-001` (flow ratio deficit/surplus), `RULE-ISS-002` (stale backlog > 50%), `RULE-ISS-003` (heavy-tailed resolution skewness), `RULE-ISS-004` (unlabeled issue share), `RULE-ISS-005` (timestamp coverage).

---

## 6. Domain 5: Release Evolution (`release_evolution`)

Source: [`backend/app/services/analytics/releases.py`](../backend/app/services/analytics/releases.py)

- **Cadence Predictability:** Inter-release interval days, annualized release frequency, and coefficient of variation (`CV_release = stdev / mean`):
  - `CV_release <= 0.35`: `highly_scheduled`
  - `0.35 < CV_release <= 0.75`: `moderately_predictable`
  - `CV_release > 0.75`: `event_driven_sporadic`
- **Pre-Release Commit Surges:** Compares 14-day pre-release commit rate against the prior 60-day baseline (`mean_surge_ratio`, `median_surge_ratio`).
- **SemVer 2.0.0 Adoption:** Regex classification of release tags into Major (`minor=0, patch=0`), Minor (`patch=0`), Patch, and Prerelease.
- **Git Tags Fallback:** Evaluates `/tags` commit history when formal Releases are absent.
- **Rule IDs:** `RULE-REL-EMPTY`, `RULE-REL-001` (scheduled cadence), `RULE-REL-002` (dormancy > 365d), `RULE-REL-003` (pre-release surge >= 1.50x), `RULE-REL-004` (SemVer compliance).

---

## 7. Domain 6: Discussions Intelligence (`discussions_intelligence`)

Source: [`backend/app/services/analytics/discussions.py`](../backend/app/services/analytics/discussions.py)

- **Volume & Participation:** Total discussion threads, active community authors, author Gini coefficient, and author bus factor `BF50`.
- **Q&A Resolution Dynamics:** Answerable categories count, answered discussions count, `qa_resolution_rate = answered / answerable`, and `time_to_first_answer_hours` quantiles (`p50`, `p75`, `p90`, `mean`).
- **Community Engagement:** Category distribution breakdown, median and mean comments per discussion, median and mean upvotes per discussion.
- **Contributor Bridge Overlap:** Measures the proportion of active discussion participants who also contribute commits, pull requests, or code reviews to the codebase.
- **Rule IDs:**
  - `RULE-DISC-EMPTY`: Zero discussion threads found or Discussions disabled.
  - `RULE-DISC-001`: Q&A resolution rate assessment (warning if `< 0.50`, positive if `>= 0.75`).
  - `RULE-DISC-002`: Answer turnaround latency assessment (positive if `p50 <= 24.0h`, warning if `p50 > 168.0h`).
  - `RULE-DISC-003`: Discussion category concentration skew.
  - `RULE-DISC-004`: Contributor bridge overlap assessment.

---

## 8. Domain 7: Cross-Domain Synthesis & Discovery (`repository_synthesis`)

Source: [`backend/app/services/analytics/cross_domain.py`](../backend/app/services/analytics/cross_domain.py)

- **Multi-Domain Diagnostic Scorecard:** High-level diagnostic matrix summarizing activity, contributor resilience, PR responsiveness, issue health, release cadence, and community discussions.
- **Cross-Domain Monthly Spearman Correlations:** Evaluates 3 fixed calendar-month aligned pairs:
  1. `PAIR-01`: Monthly code churn vs. monthly issue creations (complete diff months strictly required).
  2. `PAIR-02`: Monthly median `TTFResp` vs. monthly PR abandonment rate (suppressed if review sampling < 50%).
  3. `PAIR-03`: Monthly commit velocity vs. monthly PR merge volume.
- **Statistical Safeguards:**
  - Minimum sample: $K \ge 12$ non-null overlapping calendar months.
  - Autocorrelation guard: Lag-1 serial correlation $|r_{\text{AR1}}| > 0.40$ on either series trips `serial_dependence_detected` and suppresses permutation $p$-values.
  - Monte Carlo permutation: Exact rank-vector two-tailed permutation ($B = 10,000$) with deterministic seeded permutations.
  - Multiplicity correction: Benjamini-Hochberg False Discovery Rate ($q \le 0.05$) across tested pairs.
- **Rule IDs:** `RULE-SYN-001` (archived repository context), `RULE-SYN-002` (statistically significant association passing BH FDR), `RULE-SYN-003` (autocorrelation suppression transparency), `RULE-SYN-004` (partial synthesis coverage).

---

## 9. Applied Machine Learning & NLP Services (Stage 5)

### 9.1 Generalizable 5-Class Text Classifier
Source: [`backend/app/services/nlp/generalizable_classifier.py`](../backend/app/services/nlp/generalizable_classifier.py)
- **Target Classes:** `bug`, `feature`, `documentation`, `question`, and `maintenance`.
- **Confidence Abstention:** If $\max(P(c)) < 0.55$, the model abstains from guessing, predicting `unclassified` with `is_uncertain = True`.
- **Evaluation:** Validated via Leave-One-Repo-Out (LORO) cross-validation across diverse repositories, achieving 69.7% confident accuracy on out-of-distribution repositories.

### 9.2 Dynamic Per-Repository Semantic Duplicate Retrieval
Source: [`backend/app/services/nlp/semantic_index_service.py`](../backend/app/services/nlp/semantic_index_service.py)
- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
- **Storage:** Persisted per repository as raw float32 IEEE-754 binary BLOBs in `issue_embeddings` and `discussion_embeddings`.
- **Search:** Dynamic cosine similarity computation over normalized vectors, returning ranked matches above threshold (default: 0.65) with potential duplicate flags ($\ge 0.70$).

### 9.3 Unsupervised Backlog Topic Discovery
Source: [`backend/app/services/nlp/topic_discovery_service.py`](../backend/app/services/nlp/topic_discovery_service.py)
- **Clustering:** Pure-NumPy Spherical KMeans on the unit sphere ($L_2$-normalized vectors, cosine distance). Automatically estimates optimal cluster count $K \in [3, 8]$.
- **Keyword Extraction:** Class-based TF-IDF (c-TF-IDF) identifying the most informative and distinguishing tokens per cluster.
- **Output:** Cluster names, keywords, issue counts (open vs. closed), and centroid-closest representative issues.

---

## 10. Executive Intelligence Report Generator (Stage 6)

Source: [`backend/app/services/analytics/report_generator.py`](../backend/app/services/analytics/report_generator.py)

- **Synthesis Engine:** Aggregates metrics from all 7 domains, diagnostic scorecards, confidence scores, and consolidated rules.
- **Formats:**
  - **Markdown (`format="markdown"`):** Complete executive briefing document with structured sections, tables, severity-ranked risk findings, and maintainer recommendations.
  - **JSON (`format="json"`):** Structured machine-readable schema for external dashboards or automated CI/CD gating.
- **Endpoint:** `GET /api/v1/analytics/{owner}/{name}/report`.
