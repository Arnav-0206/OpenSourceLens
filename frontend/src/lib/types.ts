/**
 * OpenSourceLens — Core TypeScript Data Contracts
 *
 * Direct typed mappings matching backend FastAPI schemas and analytical domains.
 */

// ─── Health & Root Discovery ────────────────────────────────────────────────

export interface DatabaseHealth {
  status: string;
  dialect: string;
  latency_ms: number;
  tables_count: number;
  error: string | null;
}

export interface HealthResponse {
  status: string;
  version: string;
  timestamp: string;
  database: DatabaseHealth;
}

export interface RootResponse {
  title: string;
  version: string;
  description: string;
  docs_url: string;
  api_v1_url: string;
  supported_repositories: string[];
}

// ─── Ingestion Contracts ────────────────────────────────────────────────────

export interface DomainConfidence {
  domain: string;
  domain_name: string;
  level: "insufficient_data" | "limited_sample" | "confident";
  sample_count: number;
  metric_name: string;
  thresholds: Record<string, any>;
  is_sufficient: boolean;
  message: string;
  warning?: string | null;
}

export interface RepositoryConfidenceReport {
  overall_confidence: "insufficient_data" | "limited_sample" | "confident";
  overall_score: number;
  domains: Record<string, DomainConfidence>;
  summary: string;
  sufficient_domains_count: number;
  total_domains_count: number;
}

export interface IngestPreviewRequest {
  repo: string;
}

export interface IngestPreviewResponse {
  repo_name: string;
  owner: string;
  name: string;
  canonical_url: string;
  is_eligible: boolean;
  rejection_reason?: string | null;
  metadata: {
    stars_count?: number;
    forks_count?: number;
    open_issues_count?: number;
    commits_count?: number | null;
    primary_language?: string | null;
    default_branch?: string | null;
    is_archived?: boolean;
    is_fork?: boolean;
    is_template?: boolean;
    is_empty?: boolean;
  };
  estimated_budgets: Record<string, { request_limit: number; allocations: Record<string, number> }>;
  confidence_preview?: RepositoryConfidenceReport | null;
  badges: string[];
}

export interface IngestSubmitRequest {
  repo: string;
  mode?: "fast_snapshot" | "standard_diagnostic";
  priority?: number;
}

export interface IngestSubmitResponse {
  job_id: number;
  repo_name: string;
  mode: string;
  status: string;
  created_at: string;
  message: string;
}

export interface IngestJobStatusResponse {
  job_id: number;
  repo_name: string;
  mode: string;
  status: "pending" | "in_progress" | "completed" | "completed_partial" | "rate_limited" | "failed";
  worker_id?: string | null;
  leased_until?: string | null;
  lease_version: number;
  retry_count: number;
  max_retries: number;
  is_partial: boolean;
  partial_reason?: string | null;
  resumes_at?: string | null;
  rate_limit_consumed: number;
  items_collected: Record<string, number>;
  entity_status: Record<string, string>;
  coverage_summary?: Record<string, any> | null;
  error_message?: string | null;
  started_at: string;
  completed_at?: string | null;
  duration_seconds?: number | null;
}

// ─── Analytics V2 Contracts ─────────────────────────────────────────────────

export interface RepositoryFreshnessEnvelope {
  repository_id: number;
  repo_name: string;
  analytics_source_revision: number;
  last_synced_at?: string | null;
  last_analyzed_at?: string | null;
  primary_language?: string | null;
  stars_count: number;
  forks_count: number;
  is_archived: boolean;
}

export interface MetricCoverageItem {
  domain: string;
  status: "sufficient" | "degraded" | "insufficient" | "empty" | "unavailable";
  sample_size: number;
  min_sample_required: number;
  details?: Record<string, any> | null;
}

export interface ConfidencePenalty {
  rule: string;
  penalty: number;
  reason: string;
  domain?: string;
}

export interface ConfidenceSummary {
  score: number;
  band: "high" | "medium" | "low" | "unreliable";
  penalties: ConfidencePenalty[];
  total_domains_evaluated?: number;
  unavailable_domains?: string[];
  is_bounded_or_partial?: boolean;
  domain_breakdown?: Record<string, {
    score: number;
    confidence_penalty: number;
    coverage_status: string;
    penalties_count: number;
    penalties: ConfidencePenalty[];
  }>;
}

export interface RuleInterpretation {
  rule_id: string;
  domain: string;
  severity: "critical" | "high" | "medium" | "low" | "info" | "warning" | "positive";
  finding: string;
  evidence?: Record<string, any>;
  recommendation?: string | null;
}

export interface AnalysisRunAttemptRead {
  id: number;
  analysis_run_id: number;
  attempt_number: number;
  worker_id: string;
  outcome: "running" | "completed" | "failed" | "lease_expired" | "interrupted";
  error_category?: string | null;
  error_message?: string | null;
  interruption_reason?: string | null;
  started_at: string;
  completed_at?: string | null;
}

// ─── Domain Analytical Payloads ─────────────────────────────────────────────

export interface WeeklyActivityPoint {
  week_start: string;
  commit_count?: number;
  count?: number;
  additions?: number;
  deletions?: number;
  total_changes?: number;
  active_contributors_count?: number;
  sma_4w?: number | null;
  sma_12w?: number | null;
  ewma_12w?: number | null;
  is_anomaly?: boolean;
}

export interface DailyActivityPoint {
  date: string;
  commit_count: number;
}

export interface DevelopmentActivityData {
  status: string;
  summary: {
    total_commits: number;
    active_weeks_count: number;
    active_days_count?: number;
    completed_weeks_evaluated?: number;
    completed_weeks_count?: number;
    velocity_trend?: string;
    current_cadence_days?: number | null;
  };
  velocity?: {
    trailing_4w_velocity?: number | null;
    trailing_12w_velocity?: number | null;
    trailing_52w_velocity?: number | null;
    velocity_trend_ratio?: number | null;
  };
  velocity_metrics?: {
    trailing_4w_velocity?: number | null;
    trailing_12w_velocity?: number | null;
    trailing_52w_velocity?: number | null;
    velocity_trend_ratio?: number | null;
  };
  weekly_time_series?: WeeklyActivityPoint[];
  weekly_series?: WeeklyActivityPoint[];
  daily_time_series_90d?: DailyActivityPoint[];
  rolling_statistics?: {
    status?: string;
    series?: Array<{
      week_start: string;
      commit_count: number;
      rolling_4w_mean?: number | null;
      rolling_12w_mean?: number | null;
      ewma_4w?: number | null;
      window_4w_complete?: boolean;
      window_12w_complete?: boolean;
    }>;
  };
  anomalies?: {
    status?: string;
    total_anomalies_detected?: number;
    high_activity_anomalies_count?: number;
    low_activity_anomalies_count?: number;
    items?: any[];
  };
  activity_rhythms?: {
    status?: string;
    total_commits?: number;
    weekday_names?: string[];
    weekday_hour_matrix: number[][]; // 7 days x 24 hours
    weekday_totals?: number[];
    hour_totals_utc?: number[];
    weekday_commits?: number;
    weekend_commits?: number;
    weekday_ratio?: number;
    weekend_ratio?: number;
    peak_weekday?: { index: number; name: string; commit_count: number };
    peak_utc_hour?: { hour_utc: number; commit_count: number };
  };
  utc_rhythm?: {
    matrix: number[][];
    peak_day?: number;
    peak_hour?: number;
    concentration_ratio?: number;
  };
  commit_landing_latency?: {
    status?: string;
    p25_hours?: number | null;
    p50_hours?: number | null;
    p75_hours?: number | null;
    p90_hours?: number | null;
    mean_hours?: number | null;
    valid_merged_count?: number;
    sample_size?: number;
  };
  landing_latency?: {
    p50?: number | null;
    p75?: number | null;
    p90?: number | null;
    sample_size?: number;
  };
  code_churn?: {
    status?: string;
    total_additions?: number;
    total_deletions?: number;
    net_churn?: number;
    diff_coverage_ratio?: number;
  };
  churn_metrics?: {
    total_additions?: number;
    total_deletions?: number;
    net_churn?: number;
  };
}

export interface LorenzPoint {
  population_share?: number;
  contribution_share?: number;
  contributor_percentile?: number;
  contribution_percentile?: number;
}

export interface ContributorLeaderboardItem {
  contributor?: string;
  login?: string;
  avatar_url?: string | null;
  commits?: number;
  total_commits?: number;
  commit_share?: number;
  prs_authored?: number;
  total_prs_authored?: number;
  prs_reviewed?: number;
  total_prs_reviewed?: number;
  issues_opened?: number;
  primary_role?: string;
  is_core?: boolean;
  is_newcomer?: boolean;
  is_bot?: boolean;
  is_unlinked?: boolean;
  tenure_days?: number | null;
}

export interface ContributorEcosystemData {
  status: string;
  scope?: {
    mode?: string;
    directory_contributors_count?: number;
  };
  concentration?: {
    status?: string;
    active_human_commit_authors_count?: number;
    total_attributed_human_commits?: number;
    gini_coefficient?: number | null;
    bf50?: number | null;
    bf80?: number | null;
    top1_share?: number | null;
    top5_share?: number | null;
    shannon_entropy_bits?: number | null;
    normalized_entropy?: number | null;
    lorenz_curve?: LorenzPoint[];
  };
  participation?: {
    directory_contributors_count?: number;
    directory_human_contributors_count?: number;
    directory_bot_contributors_count?: number;
    active_human_commit_authors_count?: number;
    active_human_multi_role_contributors_count?: number;
  };
  tenure?: {
    newcomers_30d_count?: number;
    core_contributors_count?: number;
    established_gt_30d_count?: number;
    mean_tenure_days?: number | null;
    median_tenure_days?: number | null;
  };
  bot_isolation?: {
    bot_accounts_detected?: number;
    bot_commits_excluded?: number;
  };
  top_contributors?: ContributorLeaderboardItem[];
  // Legacy aliases
  total_contributors?: number;
  gini_coefficient?: number | null;
  lorenz_curve?: LorenzPoint[];
  bus_factors?: {
    bf50?: number | null;
    bf80?: number | null;
    total_evaluated?: number;
  };
  cohorts?: {
    newcomers_30d?: number;
    core_contributors?: number;
    bot_count?: number;
    unlinked_count?: number;
  };
  leaderboard?: ContributorLeaderboardItem[];
}

export interface PullRequestIntelligenceData {
  status: string;
  scope?: {
    mode?: string;
    is_partial?: boolean;
  };
  outcome_distribution?: {
    status?: string;
    total_eligible_prs: number;
    merged_count: number;
    closed_unmerged_count: number;
    open_count: number;
    resolved_count: number;
    merge_rate?: number | null;
    abandonment_rate?: number | null;
    open_ratio?: number | null;
  };
  responsiveness?: {
    ttf_resp?: {
      sample_count?: number;
      p25_hours?: number | null;
      p50_hours?: number | null;
      p75_hours?: number | null;
      p90_hours?: number | null;
      mean_hours?: number | null;
    };
    ttfr_formal?: {
      sample_count?: number;
      p50_hours?: number | null;
    };
    ttfr_inline?: {
      sample_count?: number;
      p50_hours?: number | null;
    };
  };
  merge_latency?: {
    status?: string;
    sample_count?: number;
    p25_hours?: number | null;
    p50_hours?: number | null;
    p75_hours?: number | null;
    p90_hours?: number | null;
    mean_hours?: number | null;
    valid_merged_count?: number;
  };
  survival_km?: {
    status?: string;
    s7d?: number | null;
    s30d?: number | null;
    median_survival_days?: number | null;
    survival_curve?: Array<{ time_days: number; survival_probability: number }>;
  };
  size_vs_turnaround?: {
    status?: string;
    diff_coverage_ratio?: number;
    bins?: Record<string, { count: number; p50_hours?: number | null; mean_hours?: number | null }>;
  };
  review_bottlenecks?: {
    status?: string;
    review_latency_share?: number | null;
    maintainer_review_load?: {
      active_human_reviewers_count?: number;
      total_human_formal_reviews?: number;
      top1_reviewer_share?: number | null;
      top_reviewers?: Array<{
        reviewer: string;
        formal_reviews_count?: number;
        reviews_count?: number;
        share?: number;
      }>;
    };
    unreviewed_open_prs?: {
      open_prs_count?: number;
      unreviewed_open_prs_count?: number;
      count?: number;
      median_age_days?: number | null;
    };
  };
  // Legacy aliases
  total_prs?: number;
  turnaround_hours?: {
    p50?: number | null;
    p75?: number | null;
    p90?: number | null;
    mean?: number | null;
  };
  response_latencies?: {
    ttfresp?: { p50?: number | null };
    ttfr_formal?: { p50?: number | null };
    ttfr_inline?: { p50?: number | null };
  };
  outcomes?: {
    total_prs: number;
    merged_count: number;
    closed_unmerged_count: number;
    merge_rate?: number | null;
  };
  diff_size_turnaround?: Record<string, { count: number; p50_hours?: number | null }>;
  reviewer_metrics?: {
    unique_reviewers_count?: number;
    top_reviewers?: Array<{ login: string; review_count: number }>;
  };
}

export interface IssueIntelligenceData {
  status: string;
  scope?: {
    mode?: string;
    eligible_issues_count?: number;
  };
  observed_flow?: {
    status?: string;
    eligible_issues_count?: number;
    total_created_observed: number;
    total_closed_observed: number;
    flow_ratio?: number | null;
    net_flow_differential?: number;
    monthly_series?: Array<{ month: string; created: number; closed: number; net: number }>;
  };
  backlog_stock_summary?: {
    status?: string;
    github_metadata_snapshot?: {
      open_issues_and_prs_count?: number;
    };
  };
  resolution_distribution?: {
    status?: string;
    valid_resolved_count?: number;
    p25_hours?: number | null;
    p50_hours?: number | null;
    p75_hours?: number | null;
    p90_hours?: number | null;
    mean_hours?: number | null;
    speed_tier_distribution?: Record<string, number>;
  };
  survival_km?: {
    status?: string;
    s30d?: number | null;
    s60d?: number | null;
    s90d?: number | null;
    median_survival_days?: number | null;
    survival_curve?: Array<{ time_days: number; survival_probability: number }>;
  };
  backlog_aging?: {
    status?: string;
    open_issues_count: number;
    median_age_days?: number | null;
    stale_gt_90d_count?: number;
    stale_fraction?: number | null;
    bins?: Record<string, number>;
  };
  discussion_intensity?: {
    status?: string;
    mean_comments_per_issue?: number | null;
    median_comments_per_issue?: number | null;
    p90_comments_per_issue?: number | null;
  };
  canonical_categories?: {
    status?: string;
    categories?: Record<string, number>;
    top_raw_labels?: Array<{ label: string; count: number }>;
    unlabeled_share?: number | null;
  };
  // Legacy aliases
  total_issues?: number;
  flow?: {
    total_opened: number;
    total_closed: number;
    net_growth?: number;
    monthly_flow?: Array<{ month: string; opened: number; closed: number }>;
  };
  resolution_sla_hours?: {
    p50?: number | null;
    p75?: number | null;
    p90?: number | null;
  };
  aging?: {
    open_count?: number;
    open_stale_count?: number;
    median_age_days?: number | null;
  };
}

export interface ReleaseEvolutionData {
  status: string;
  scope?: {
    mode: string;
    is_bounded_or_partial: boolean;
  };
  cadence?: {
    total_releases: number;
    cv_release?: number | null;
    regularity_classification?: string;
    days_since_last_release?: number | null;
    latest_release_tag?: string | null;
    annualized_release_frequency?: number | null;
    mean_interval_days?: number | null;
    median_interval_days?: number | null;
  };
  semver_adoption?: {
    semver_adoption_rate?: number | null;
    major_releases_count?: number;
    minor_releases_count?: number;
    patch_releases_count?: number;
    major_releases?: number;
    minor_releases?: number;
    patch_releases?: number;
    prerelease_count?: number;
  };
  prerelease_surges?: {
    status?: string;
    mean_surge_ratio?: number | null;
    median_surge_ratio?: number | null;
    surge_detected?: boolean;
    surge_ratio?: number | null;
    valid_surge_releases_count?: number;
  };
}

export interface DiscussionsIntelligenceData {
  status: string;
  total_discussions: number;
  total_comments?: number;
  total_upvotes?: number;
  active_discussions_window?: number;
  dominant_category?: string | null;
  categories?: any;
  qna?: {
    total_qna_threads?: number;
    answered_qna_threads?: number;
    unanswered_qna_threads?: number;
    qna_answered_rate?: number | null;
    time_to_answer_hours?: {
      p50?: number | null;
      p50_hours?: number | null;
      p75?: number | null;
      p90?: number | null;
      mean?: number | null;
    };
  };
  engagement?: {
    mean_comments_per_discussion?: number | null;
    median_comments_per_discussion?: number | null;
    mean_upvotes_per_discussion?: number | null;
    median_upvotes_per_discussion?: number | null;
    unique_discussion_authors?: number;
    contributor_overlap_count?: number;
    contributor_overlap_rate?: number | null;
    top_discussed_threads?: Array<{
      discussion_number: number;
      title: string;
      category: string;
      comments_count: number;
      upvote_count: number;
      is_answered: boolean;
      created_at?: string | null;
    }>;
  };
}

export interface CrossDomainCorrelationPair {
  pair_id: string;
  name?: string;
  metric_x?: string;
  label_x?: string;
  metric_y?: string;
  label_y?: string;
  domain_x?: string;
  domain_y?: string;
  spearman_r?: number | null;
  p_value?: number | null;
  q_value?: number | null;
  status: string;
  k_months?: number;
  max_abs_ar1?: number | null;
  ar1_x?: number | null;
  ar1_y?: number | null;
  serial_dependence_guardrail_tripped?: boolean;
  significant_fdr_05?: boolean;
  note?: string;
  overlapping_months?: string[];
  monthly_points?: Array<{
    month: string;
    val_x: number | null;
    val_y: number | null;
  }>;
}

export interface CrossDomainSynthesisData {
  diagnostic_scorecard?: Record<string, any>;
  cross_domain_correlations?: {
    status?: string;
    methodology?: string;
    min_periods_required?: number;
    fdr_alpha?: number;
    tested_pairs_count?: number;
    pairs?: CrossDomainCorrelationPair[];
    causality_disclaimer?: string;
  };
  release_evolution?: ReleaseEvolutionData;
  discussions_intelligence?: DiscussionsIntelligenceData;
  orchestration_metadata?: Record<string, any>;
}

export interface AnalysisPayload {
  development_activity?: DevelopmentActivityData;
  contributor_ecosystem?: ContributorEcosystemData;
  pull_request_intelligence?: PullRequestIntelligenceData;
  issue_intelligence?: IssueIntelligenceData;
  repository_synthesis?: CrossDomainSynthesisData;
}

export interface AnalysisRunRead {
  id: number;
  repository_id: number;
  collection_run_id?: number | null;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  requested_source_revision: number;
  analytics_source_revision?: number | null;
  requested_methodology_version?: string | null;
  methodology_version?: string | null;
  analysis_reference_time?: string | null;
  source_content_digest?: string | null;
  worker_id?: string | null;
  lease_version: number;
  retry_count: number;
  max_retries: number;
  metrics_payload?: AnalysisPayload | null;
  coverage_summary?: Record<string, any> | null;
  confidence_summary?: ConfidenceSummary | null;
  rule_interpretations?: RuleInterpretation[] | null;
  started_at: string;
  completed_at?: string | null;
  error_message?: string | null;
  execution_time_ms: number;
  attempts: AnalysisRunAttemptRead[];
}

export interface AnalysisLatestResponse {
  repository: RepositoryFreshnessEnvelope;
  is_stale: boolean;
  active_run_id?: number | null;
  active_run_status?: string | null;
  run: AnalysisRunRead;
}

export interface AnalysisStatusResponse {
  repository: RepositoryFreshnessEnvelope;
  has_analysis: boolean;
  status?: string | null;
  is_stale: boolean;
  retry_count: number;
  max_retries: number;
  failure_category?: string | null;
  current_run?: AnalysisRunRead | null;
  attempts: AnalysisRunAttemptRead[];
  latest_completed_run_id?: number | null;
}

export interface AnalysisRecomputeRequest {
  force?: boolean;
  requested_methodology_version?: string;
}

export interface AnalysisRecomputeResponse {
  repository: RepositoryFreshnessEnvelope;
  run: AnalysisRunRead;
  enqueued: boolean;
  already_active: boolean;
  cached: boolean;
  message: string;
}

export interface AnalysisDomainSliceResponse {
  repository: RepositoryFreshnessEnvelope;
  run_id: number;
  domain: string;
  analytics_source_revision?: number | null;
  methodology_version?: string | null;
  analysis_reference_time?: string | null;
  is_stale: boolean;
  data?: Record<string, any> | null;
  coverage?: MetricCoverageItem | null;
  confidence_breakdown?: {
    score?: number;
    confidence_penalty?: number;
    coverage_status?: string;
    penalties?: ConfidencePenalty[];
  } | null;
  rules: RuleInterpretation[];
}

// ─── Classification Schemas ─────────────────────────────────────────────────

export interface ClassificationRequest {
  title: string;
  body?: string | null;
  repo_name?: string | null;
  model_type?: "distilbert" | "svm" | "logistic_regression";
  issue_id?: number | null;
}

export interface ClassificationResponse {
  predicted_label: string;
  canonical_label_id: number;
  confidence_score: number;
  probabilities: Record<string, number>;
  model_used: string;
  inference_time_ms: number;
  clean_title: string;
  clean_body: string | null;
  is_classified_category: boolean;
}

// ─── Model Registry Schemas ─────────────────────────────────────────────────

export interface ModelInfoItem {
  model_id: string;
  name: string;
  task_type: string;
  version: string;
  artifact_path: string;
  is_loaded: boolean;
  device: string;
  details: Record<string, any>;
  status: string;
}

export interface ModelRegistryResponse {
  total_models: number;
  models: ModelInfoItem[];
}

// ─── Error Schemas ──────────────────────────────────────────────────────────

export interface ErrorDetailItem {
  field: string;
  message: string;
  type: string;
}

export interface ErrorResponse {
  detail: string;
  error_code: string;
  status_code: number;
  path: string;
  timestamp: string;
  details?: Record<string, any>;
}

export interface ValidationErrorResponse {
  detail: string;
  error_code: string;
  status_code: number;
  path: string;
  timestamp: string;
  errors: ErrorDetailItem[];
}

export interface SimilarQueryRequest {
  query_text?: string | null;
  issue_number?: number | null;
  top_k?: number;
  threshold?: number;
  include_discussions?: boolean;
}

export interface SimilarCandidateItem {
  rank: number;
  entity_type: "issue" | "discussion";
  id: number;
  number: number;
  title: string;
  state: string;
  similarity_score: number;
  confidence_tier: "HIGH" | "MEDIUM" | "LOW";
  duplicate_candidate: boolean;
  canonical_label?: string | null;
  category?: string | null;
  author?: string | null;
  comments_count?: number | null;
  created_at?: string | null;
}

export interface SimilarQueryResponse {
  query: string;
  total_candidates_found: number;
  has_potential_duplicate: boolean;
  highest_similarity_score: number;
  primary_recommendation: string;
  latency_ms: number;
  candidates: SimilarCandidateItem[];
}

export interface DiscoveredTopicItem {
  cluster_id: number;
  title: string;
  keywords: string[];
  total_count: number;
  open_count: number;
  closed_count: number;
  representative_items: Array<{
    number: number;
    title: string;
    entity_type: string;
    state: string;
    similarity_to_centroid?: number;
  }>;
}

export interface TopicDiscoveryResponse {
  repo_name: string;
  total_topics_discovered: number;
  total_items_clustered: number;
  methodology: string;
  topics: DiscoveredTopicItem[];
  execution_time_ms: number;
}

export interface ClassifiedIssueItem {
  issue_number: number;
  title: string;
  state: string;
  author: string;
  created_at?: string | null;
  comments_count: number;
  html_url: string;
  github_labels: string[];
  predicted_category: "bug" | "feature" | "documentation" | "question" | "other";
  confidence: number;
  is_uncertain: boolean;
  probabilities: Record<string, number>;
}

export interface ClassifiedIssuesResponse {
  repository: string;
  state_filter: "open" | "closed" | "all";
  total_open: number;
  total_closed: number;
  total_matching: number;
  classified_count: number;
  model_used: string;
  category_counts: Record<string, number>;
  items: ClassifiedIssueItem[];
}

