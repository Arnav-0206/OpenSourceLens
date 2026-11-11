# OpenSourceLens — Analytics Engine & Orchestration Specification

OpenSourceLens implements a six-domain statistical analytics engine (`methodology_version = "2026.1"`) built on [`BaseMetricCalculator`](../backend/app/services/analytics/base.py#L32-L66) and coordinated by [`AnalyticsOrchestrator`](../backend/app/services/analytics/orchestrator.py#L256-L1140).

All domain calculators obey two core invariants:
1. **Snapshot & Reference-Time Freezing:** Every domain reads inside a single snapshot transaction against the worker's pinned `analytics_source_revision` and frozen UTC `analysis_reference_time` (`ref_time`). Any rows timestamped after `ref_time` are excluded.
2. **No Fabricated Zeros / No Silent Coercion:** Missing commit diffs, missing `committer_date` timestamps, uncollected review streams, and small samples report explicit statuses (`insufficient`, `empty`, `diff_metrics_not_collected`, `unavailable_requires_full_history_ingestion`, etc.) rather than coercing `None` to `0.0`.

---

## 1. Orchestration, Validation & Publication Flow

[`AnalyticsOrchestrator`](../backend/app/services/analytics/orchestrator.py) executes the full pipeline in fixed sequence:

1. **Snapshot Open (`AnalyticsJobQueue.open_analysis_snapshot`):** Opens a read snapshot (dedicated read session in SQLite WAL mode) pinned to `repository_id` and locks `analysis_reference_time` in UTC.
2. **Five Base Calculators:** Runs `development_activity`, `contributor_ecosystem`, `pull_request_intelligence`, `issue_intelligence`, and `release_evolution`. A missing/None result is converted to an explicit `unavailable` `CalculationResult` (`penalty = 1.0`, `RULE-ORCH-UNAVAILABLE`); unexpected domain runtime exceptions raise `DomainCalculationError(..., retryable=True)`.
3. **Module 6 Reuse (`CrossDomainSynthesisCalculator`):** Receives the 5 precomputed domain results (`precomputed_domains`) to construct the Multi-Domain Diagnostic Scorecard and executes monthly aligned cross-domain correlation tests without duplicating C.2–C.5 queries.
4. **Payload Assembly & Pydantic Validation (`build_publication_bundle`):**
   - Embeds `release_evolution` (with its `coverage` and `confidence` penalties) inside `repository_synthesis` and attaches `orchestration_metadata` (including `collection_lineage`, `unavailable_domains`, `methodology_version`, `snapshot_revision`, `analysis_reference_time`; `last_synced_at` is intentionally excluded from synthesis `repo_context` and metadata).
   - Validates the strict 5-key top-level structure through [`AnalysisPayload`](../backend/app/schemas/analytics_v2.py#L59-L66): `development_activity`, `contributor_ecosystem`, `pull_request_intelligence`, `issue_intelligence`, `repository_synthesis`.
5. **Coverage & Confidence Aggregation:**
   - Each domain produces `domain_score = round(max(0.0, 1.0 - confidence_penalty), 3)`.
   - Overall score is the unweighted mean across all 6 domains: `round(sum(domain_scores) / 6, 3)`.
   - Band classification (`_confidence_band`):
     - `unreliable`: `all_core_empty_or_unavailable` (all 5 base domains `empty` or `unavailable`) OR `score < 0.40`
     - `low`: `0.40 <= score < 0.60`
     - `medium`: `0.60 <= score < 0.80`
     - `high`: `score >= 0.80`
6. **Rule Consolidation (`consolidate_rules`):** Deduplicates on `(rule_id, domain, finding)` across Domains 1–6 and sorts deterministically by severity (`critical < warning < positive < info`), `domain`, `rule_id`, `finding`.
7. **Revision & Lease-Fenced Publication:** Before committing, `process_claimed_run()` checks if `Repository.analytics_source_revision` advanced beyond the pinned snapshot revision. If stale, it marks the run `status="stale"` (`_abort_superseded_stale_run`) and re-queues current revision work instead of publishing obsolete metrics. Otherwise `publish_and_finalize()` atomically persists `metrics_payload`, `coverage_summary`, `confidence_summary`, `rule_interpretations`, `source_content_digest`, updates `repositories.last_analyzed_at`, and records an immutable attempt.

---

## 2. Domain 1: Development Activity (`development_activity`)

Source: [`backend/app/services/analytics/development_activity.py`](../backend/app/services/analytics/development_activity.py)

### Metrics & Methodology
- **Velocity & Recency:**
  - Rolling day-window velocities (avoiding partial-week denominator distortion): `trailing_4w_velocity = commits(> ref_time - 28d) / 4.0`; `trailing_12w_velocity = commits(> ref_time - 84d) / 12.0`; `trailing_52w_velocity = commits(> ref_time - 364d) / min(52.0, max(1.0, observed_52w_days / 7.0))`.
  - `overall_weekly_velocity = total_commits / max(1.0, total_weeks_span)`.
  - `velocity_trend_ratio = trailing_4w_velocity / max(0.1, trailing_12w_velocity)`.
- **Completed-Week Consistency (`_calculate_consistency`):** Excludes the ongoing calendar week `[last_week, ref_time]`. Computes mean, median, sample stdev, linear-interpolated IQR, MAD, `CV = stdev / mean`, `active_week_ratio`, `zero_week_ratio`, and classifies regularity (`consistent_steady`, `consistently_low`, `moderately_variable`, `highly_volatile`, `sparse_intermittent`, or `insufficient_history`).
- **Rolling Weekly Statistics (`_calculate_rolling_statistics`):** Over completed weeks (capped in output to trailing 104 weeks while preserving full-history EWMA warmup): 4-week and 12-week SMA/stdev/median plus 4-week EWMA (`alpha = 2 / (4 + 1) = 0.4`).
- **Anomaly Detection (`_detect_anomalies`):** Evaluates weeks from index `>= 6` against a trailing `6..8` completed-week baseline using primary rolling modified Z-score (`scale = max(1.4826 * MAD, 1.0)`), falling back to sample z-score (`scale = max(stdev, 1.0)`) and Poisson floor (`max(1.0, sqrt(mean))`). Flags highs only when `observed >= 6`, `observed - b_median >= 4`, `observed >= 2.0 * max(1.0, b_median)`, and `score >= 3.0`; flags drops only on completed weeks when `nonzero_frac >= 0.75`, `b_median >= 4`, `b_mean >= 4`, `observed <= 0.25 * b_median`, and `score <= -2.5` (or `observed == 0` with `b_median >= 5` and `score <= -2.0`).
- **Change-Point Detection (`_detect_change_points`):** Single-split `max_ssb_permutation_test` (`B = 999` deterministic permutations, `min_segment = 5`, `p_perm <= 0.01`, plus practical magnitude gates `abs_mean_shift >= 3.0`, `abs_med_shift >= 2.0`, `rel_ratio >= 1.75`). Secondary Mann-Whitney U is recorded only diagnostically.
- **Activity Rhythms (`_calculate_activity_rhythms`):** `7x24` weekday-hour matrix in **UTC** (explicitly noting UTC timestamps do not capture contributor local hours).
- **Diff-Aware Code Churn (`_calculate_code_churn`):** Counts `additions + deletions` strictly on commits where both are non-NULL; tracks `diff_coverage_ratio`; excludes completed weeks containing missing diffs from `weekly_churn_volatility`.
- **Commit Landing Latency (`_calculate_commit_landing_latency`):** `committer_date - author_date` (strictly without `author_date` substitution when `committer_date IS NULL`), isolating negative clock-skew intervals and reporting `p25/p50/p75/p90` on valid non-negative hours.

### Gates, Penalties, Rules & Limitations
- **Required Data:** `Commit` rows (`<= ref_time`); `Release` rows for summary release cadence.
- **Sample Gates:**
  - Overall domain: `min_sample = 5` commits (`< 5` → `insufficient`), span `< 14d` → `degraded`.
  - Consistency: `>= 4` completed weeks and `>= 5` commits (`sufficient` at `>= 8` weeks).
  - Anomalies: `>= 8` completed weeks and `>= 15` commits; skipped if sparse (`alpha < 0.40` and `active_weeks < 6`).
  - Change points: `>= 12` completed weeks, `>= 30` commits, `>= 6` active weeks.
  - Rhythms: `>= 5` commits (`sufficient` at `>= 20`); Churn: `>= 5` diff commits (`sufficient` if `diff_coverage_ratio >= 0.50`, volatility requires `>= 4` clean completed weeks); Landing latency: `>= 5` valid non-negative pairs (`sufficient` if `>= 20` and coverage `>= 0.50`).
- **Confidence Penalties:** `PENALTY-EMPTY-COMMITS` (`0.5`), `PENALTY-SAMPLE-SIZE` (`<5`: `0.3`), `PENALTY-SHORT-SPAN` (`<14d`: `0.15`), `PENALTY-MODERATE-SAMPLE` (`5..19`: `0.1`).
- **Rule IDs:** `RULE-ACT-EMPTY`, `RULE-ACT-001` (recency), `RULE-ACT-002` (velocity trajectory), `RULE-ACT-003` (release cadence), `RULE-ACT-004` (consistency), `RULE-ACT-005` (anomalies), `RULE-ACT-006` (change points), `RULE-ACT-007` (UTC rhythms), `RULE-ACT-008` (churn), `RULE-ACT-009` (landing latency).
- **Limitations:** Standard diagnostic budgets collect at most 3 commit pages (300 commits) and 5 detailed commit diffs, often leaving churn coverage `< 50%` or `diff_metrics_not_collected`. Permutation tests assume exchangeable observations under the null.

---

## 3. Domain 2: Contributor Ecosystem (`contributor_ecosystem`)

Source: [`backend/app/services/analytics/contributors.py`](../backend/app/services/analytics/contributors.py)

### Metrics & Methodology
- **Actor Identity Resolution (`_resolve_actor`):** Resolves authors without speculative display-name merging: anchors on immutable `github_id` (`identity_confidence = 1.0`) → verified `Contributor` login (`0.9` / `1.0`) → unlinked `git_email` (`0.7`); display-name-only commits remain `is_unattributed = True` and are isolated from human concentration metrics. Bots (`is_bot` ORM flag or `[bot]` suffix) are tracked separately under `identity_hygiene` (`bot_commits_excluded_count`, `unattributed_commits_count`).
- **Concentration Metrics (`_calculate_concentration`):** Computed over attributed active human commit authors (`n`):
  - Gini coefficient: $G = \frac{2 \sum_{i=1}^n i \cdot x_{(i)}}{n \sum x_i} - \frac{n + 1}{n}$ clamped to `[0.0, 1.0]`.
  - Standardized 11-point Lorenz deciles (`u in {0.0, 0.1, ..., 1.0}`).
  - Shannon entropy $H = -\sum p_i \log_2 p_i$ and normalized entropy $H / \log_2(n)$.
  - Bus factors `bf50` and `bf80` (minimum authors covering `>= 50%` and `>= 80%` of commits), `top1_share`, `top5_share`.
- **Multi-Role Participation (`_aggregate_participation`):** Counts commits, authored PRs, formal reviews, opened issues, and conversation/inline comments per resolved actor; tracks `directory_contributors_count` separately.
- **Retention Cohorts (`_evaluate_retention`):** Strictly **gated behind verified `full_history` collection** (`mode == "full_history"`, `status == "completed"`, non-partial, non-truncated commits/PRs). When satisfied, groups actors by cohort month of first commit/PR authored and measures M+1, M+3, etc. return rates. Bounded snapshots return `status = "unavailable_requires_full_history_ingestion"`.

### Gates, Penalties, Rules & Limitations
- **Sample Gates:** Minimum `n >= 2` human commit authors for concentration (`n == 1` marks `degenerate_single_contributor=True`, `Gini=0.0`, Lorenz deferred); `sufficient` at `n >= 5`.
- **Confidence Penalties:** `PENALTY-EMPTY-CONTRIBUTORS` (`0.5`), `PENALTY-INSUFFICIENT-COMMIT-AUTHORS` (`<2`: `0.4`), `PENALTY-SMALL-CONTRIBUTOR-POOL` (`2..4`: `0.2`), `PENALTY-BOUNDED-SAMPLE-SCOPE` (`0.1` when bounded/partial), `PENALTY-UNATTRIBUTED-COMMITS` (`0.1` if unattributed share `> 0.25`).
- **Rule IDs:** `RULE-CON-EMPTY`, `RULE-CON-001` (bus factor / top1 risk vs distributed ownership), `RULE-CON-002` (Gini `>= 0.75`), `RULE-CON-003` (bot/unattributed filtering transparency when share `>= 0.20`), `RULE-CON-004` (retention cohort deferral or M+1 summary).
- **Limitations:** Under `fast_snapshot` / `standard_diagnostic`, retention cohorts are always disabled (`unavailable_requires_full_history_ingestion`), and concentration reflects only recent collected commits rather than lifetime history.

---

## 4. Domain 3: Pull Request Intelligence (`pull_request_intelligence`)

Source: [`backend/app/services/analytics/pull_requests.py`](../backend/app/services/analytics/pull_requests.py)

### Metrics & Methodology
- **Separated Responsiveness Channels (`_evaluate_responsiveness`):** Excludes PR-author self-interactions (`_is_self_interaction`), bots (`_is_bot`), unattributed interactions, and negative or future (`> ref_time`) timestamps:
  - `ttf_resp` (Time-to-First-Response): Minimum elapsed hours across formal reviews (`APPROVED`, `CHANGES_REQUESTED`, `COMMENTED`, `DISMISSED`), inline review comments, and PR conversation comments.
  - `ttfr_formal`: Minimum hours across formal reviews (`APPROVED`, `CHANGES_REQUESTED`, `COMMENTED`).
  - `ttfr_inline`: Minimum hours across `PullRequestReviewComment` inline diff comments (tracks `skipped_by_collection_mode` vs `inline_comments_not_collected` when missing).
- **Outcome Distribution (`_evaluate_outcomes`):** Splits snapshot-bounded PRs (`created_at <= ref_time`) using `merged_at <= ref_time` → `merged`, `closed_at <= ref_time` (unmerged) → `closed_unmerged`, else `open`; reports `merge_rate = merged / resolved`, `abandonment_rate = closed_unmerged / resolved`, and `open_ratio = open / total`.
- **Merge Latency (`_evaluate_merge_latency`):** Non-parametric percentiles (`p25`, `p50`, `p75`, `p90`, `iqr`, `mean`) over `(merged_at - created_at) / 3600`, preserving `0.0h` instant merges and discarding negative clock skews.
- **Kaplan-Meier Survival (`_evaluate_kaplan_meier_survival`):** Product-limit survival estimator `compute_kaplan_meier_curve` with open PRs right-censored at `ref_time - created_at`, computing `s7d`, `s30d`, `median_survival_days`, and step curve (capped at 200 steps).
- **Size vs Turnaround (`_evaluate_size_vs_turnaround`):** Buckets resolved PRs with non-NULL `additions` and `deletions` into `small` (`0..50`), `medium` (`51..250`), `large` (`251..1000`), `xl` (`>1000`). Preserves `0`-line diffs while marking NULL diffs `unavailable_no_diff_metrics`.
- **Bottlenecks (`_evaluate_bottlenecks`):** Computes `review_latency_share = median(TTFResp) / median(Merge Latency)`, maintainer formal review distribution (`top1_reviewer_share`, `top_reviewers`), and unreviewed open PR age.

### Gates, Penalties, Rules & Limitations
- **Sample Gates:** `ttf_resp` (`>= 5` limited, `>= 10` sufficient); `ttfr_formal` (`>= 10`); `ttfr_inline` (`>= 5`); `merge_latency` (`>= 5` valid merges); Kaplan-Meier (`>= 15` PRs); Size bins (`>= 10` resolved PRs with non-NULL diff stats). Overall domain requires `>= 10` PRs (`< 5` → `insufficient`).
- **Confidence Penalties:** Empty (`0.5`), `< 5` PRs (`0.4`), `5..9` PRs (`0.2`), `ttf_resp` insufficient (`0.1`), bounded/partial scope or truncated PR streams (`0.1`), truncated review stream (`0.05`).
- **Rule IDs:** `RULE-PR-EMPTY`, `RULE-PR-001` (review bottleneck / slow TTFResp), `RULE-PR-002` (abandonment rate), `RULE-PR-003` (review load concentration), `RULE-PR-004` (XL vs small latency ratio — requires `>= 2` merged PRs in both `small` and `xl` bins), `RULE-PR-005` (bounded/truncated stream transparency).
- **Limitations:** `RepositoryCollector` samples formal reviews on only up to 2 PRs (`fast_snapshot`) or 20 PRs (`standard_diagnostic`) and skips inline comments in `fast_snapshot` (`0` budget), so formal/inline responsiveness is often limited or mode-skipped.

---

## 5. Domain 4: Issue & Maintenance Intelligence (`issue_intelligence`)

Source: [`backend/app/services/analytics/issues.py`](../backend/app/services/analytics/issues.py)

### Metrics & Methodology
- **Observed Monthly Flow (`_evaluate_monthly_flow`):** Monthly issue `created_count`, `closed_count`, `net_flow_differential`, `cumulative_flow_differential`, and aggregate `flow_ratio = total_closed_observed / total_created_observed`. Explicitly warns this measures **observed sample flow, not repository backlog stock**.
- **Backlog Stock Separation (`_evaluate_backlog_stock`):** Compares local `open issues + open PRs` against GitHub's `repositories.open_issues_count` (which mixes both in REST API metadata), keeping `pure_issue_sample_coverage_percentage = None`.
- **Resolution Distribution (`_evaluate_resolution_distribution`):** Recomputes `(closed_at - created_at) / 3600` directly from timestamps (preserving `0.0h`, excluding negatives/missing timestamps), yielding `p25/p50/p75/p90`, `iqr`, `mean`, finite `skewness`, `high_skew_warning` (`skewness > 1.0`), and speed tiers (`< 1 Day`, `1-7 Days`, `7-30 Days`, `30+ Days`).
- **Kaplan-Meier Survival (`_evaluate_kaplan_meier_survival`):** Right-censored at `ref_time`; reports `s30d`, `s60d`, `s90d`, `median_survival_days`, and follow-up horizon indicators.
- **Backlog Aging (`_evaluate_backlog_aging`):** Bins open issues as of `ref_time` into `fresh` (`<7d`), `active` (`7..30d`), `aging` (`>30..90d`), `stale` (`>90..365d`), `dormant` (`>365d`) and computes `stale_fraction` (`>90d`).
- **Discussion Intensity (`_evaluate_discussion_intensity`):** Primary stats from `Issue.comments_count`, supplemental counts from `IssueComment` (`<= ref_time`), and guarded Spearman correlation between comments and resolution hours (`>= 5` closed issues, zero-variance guarded, non-causal).
- **Canonical Categories (`_evaluate_canonical_categories`):** Keeps `"unlabeled"` strictly separate from `"other"`, plus top 15 raw labels from `issue_labels`.

### Gates, Penalties, Rules & Limitations
- **Sample Gates:** Overall domain & monthly flow (`>= 10` issues, `< 5` → `insufficient`); Resolution distribution & Spearman (`>= 5` closed issues, sufficient at `>= 20`); Kaplan-Meier (`>= 15` issues).
- **Confidence Penalties:** Empty (`0.5`), `< 5` issues (`0.4`), `5..9` issues (`0.2`), `< 5` valid resolved issues (`0.1`), bounded/partial collection mode (`0.1`).
- **Rule IDs:** `RULE-ISS-EMPTY`, `RULE-ISS-001` (flow ratio `< 0.70` warning or `>= 1.0` surplus), `RULE-ISS-002` (stale backlog `> 50%` at `>= 10` open issues), `RULE-ISS-003` (heavy-tailed resolution skewness `> 1.0`), `RULE-ISS-004` (`>= 25%` unlabeled share), `RULE-ISS-005` (timestamp/coverage transparency).
- **Limitations:** On GitHub repos with thousands of issues, bounded collection harvests at most 100–500 recently updated issues (`sort=updated`), so long-untouched dormant issues or older closed history can be underrepresented.

---

## 6. Domain 5: Release Evolution (`release_evolution`)

Source: [`backend/app/services/analytics/releases.py`](../backend/app/services/analytics/releases.py), rules in [`rule_engine.py`](../backend/app/services/analytics/rule_engine.py#L25-L151)

### Metrics & Methodology
- **Cadence & Predictability (`_evaluate_cadence`):** Computes sorted inter-release intervals (excluding `published_at IS NULL` and `published_at > ref_time`), `days_since_last_release`, `mean/median/min/max/stdev_interval_days`, `annualized_release_frequency = 365.25 / mean`, and `cv_release = stdev / mean` (`R >= 3` and `mean > 0`):
  - `cv_release <= 0.35` → `highly_scheduled`
  - `0.35 < cv_release <= 0.75` → `moderately_predictable`
  - `cv_release > 0.75` → `event_driven_sporadic`
- **Pre-Release Surges (`_evaluate_prerelease_surges`):** Compares 14-day pre-release commit rate (`[T_k - 14d, T_k] / 14.0`) against the preceding 60-day baseline (`[T_k - 74d, T_k - 14d) / 60.0`). Releases without `>= 74d` prior commit history are tagged `insufficient_baseline_depth`; zero-baseline releases are flagged `dormant_baseline_activity` (avoiding division by zero). Aggregate `mean_surge_ratio` and `median_surge_ratio` require `>= 2` valid surge releases.
- **SemVer Adoption (`_evaluate_semver`):** Evaluates `tag_name` against strict SemVer 2.0.0 regex (`SEMVER_PATTERN`), classifying Major (`minor=0, patch=0`), Minor (`patch=0`), Patch, and Prerelease releases.

### Gates, Penalties, Rules & Limitations
- **Sample Gates:** Overall domain `sufficient` at `>= 5` dated releases, `degraded` at `2..4`, `insufficient` at `1`, `empty` at `0`. `cv_release` requires `>= 3` releases; aggregate surges require `>= 2` valid releases with `>= 74d` commit history.
- **Confidence Penalties:** `PENALTY-EMPTY-RELEASES` (`0.5`), `PENALTY-SINGLE-RELEASE` (`0.4`), `PENALTY-SMALL-RELEASE-SAMPLE` (`2..4`: `0.2`), `PENALTY-BOUNDED-RELEASE-SCOPE` (`0.1`).
- **Rule IDs:** `RULE-REL-EMPTY`, `RULE-REL-001` (`cv_release <= 0.35 & R >= 4`), `RULE-REL-002` (`days_since_last_release > 365d`), `RULE-REL-003` (mean surge ratio `>= 1.50`), `RULE-REL-004` (SemVer rate `>= 0.80` positive or `< 0.30` info).

---

## 7. Domain 6: Cross-Domain Synthesis (`repository_synthesis`)

Source: [`backend/app/services/analytics/cross_domain.py`](../backend/app/services/analytics/cross_domain.py), rules in [`rule_engine.py`](../backend/app/services/analytics/rule_engine.py#L154-L246)

### Metrics & Methodology
- **Multi-Domain Diagnostic Scorecard (`_build_diagnostic_scorecard`):** Summarizes key objective metrics across `activity_profile`, `contributor_resilience`, `pr_responsiveness`, `issue_health`, and `release_cadence`. Composite `0–100` CHI scores are **explicitly deferred** (`chi_deferred=True`).
- **Embedded `release_evolution`:** Stores full Release Evolution metrics + coverage + confidence inside `repository_synthesis["release_evolution"]` so top-level `AnalysisPayload` retains its 5-key contract.
- **Cross-Domain Calendar-Month Spearman Correlations (`evaluate_paired_spearman_permutation`):** Evaluates 3 fixed monthly aligned pairs:
  1. `PAIR-01`: `monthly_code_churn` vs. `monthly_issue_creations` — **Strict monthly diff completeness:** monthly churn is populated only when **all** eligible commits in that month have valid `additions` and `deletions` (or `0.0` for zero-commit months only when repo diff stats exist elsewhere); partial diff months stay `None` and `monthly_diff_coverage_ratio` is preserved separately.
  2. `PAIR-02`: `monthly_median_ttf_resp_hours` vs. `monthly_pr_abandonment_rate` — Applies explicit `_assess_pair02_collection_coverage()`: suppresses inferential p-values (`status = "incomplete_review_coverage"`) whenever reviews were not collected, both review & issue-comment streams are missing despite PRs existing, review sampling covers `< 50%` of PRs, or `CollectionRun` is partial/truncated on review/comment streams.
  3. `PAIR-03`: `monthly_commit_velocity` vs. `monthly_pr_merge_volume`.
- **Statistical Safeguards:**
  - Minimum `K >= 12` non-null overlapping calendar months (`insufficient_temporal_periods` below 12).
  - Consecutive-month lag-1 autocorrelation guard (`|r_AR1| > 0.40` on consecutive `(m, m+1)` calendar pairs trips `serial_dependence_detected` and suppresses permutation p-values).
  - Exact rank-vector two-tailed Monte Carlo permutation (`B = 10,000`, SHA-256 32-bit seed bound to `repository_id:snapshot_revision:metric_x:metric_y:methodology_version:B`) with plus-one correction `(exceed + 1) / (B + 1)`.
  - Benjamini-Hochberg FDR adjustment (`apply_benjamini_hochberg_fdr`, `alpha = 0.05`) across pairs that pass all guards and enter status `"tested"`.
  - Block-bootstrap confidence intervals are explicitly marked deferred (`confidence_intervals_deferred=True`).

### Penalties, Rules & Limitations
- **Confidence Penalties:** `PENALTY-NO-SUFFICIENT-DOMAINS` (`0.5`), `PENALTY-PARTIAL-DOMAIN-SYNTHESIS` (`0.3` insufficient / `0.2` if 1 sufficient), `PENALTY-INCOMPLETE-CROSS-DOMAIN-MATRIX` (`0.1` if `2..4` sufficient), `PENALTY-NO-PERMUTATION-TESTED-PAIRS` (`0.05` when `0` pairs pass for permutation testing).
- **Rule IDs:** `RULE-SYN-001` (archived repo context), `RULE-SYN-002` (BH FDR `<= 0.05` significant association — strictly non-causal wording), `RULE-SYN-003` (AR(1) autocorrelation suppression transparency), `RULE-SYN-004` (partial multi-domain coverage summary).
