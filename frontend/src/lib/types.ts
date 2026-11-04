/**
 * OpenSourceLens — Core TypeScript Data Contracts
 *
 * Direct 1:1 typed mapping matching backend FastAPI Pydantic schemas (Phase 9).
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

// ─── Repository Schemas ─────────────────────────────────────────────────────

export interface RepositorySummary {
  id: number;
  repo_name: string;
  owner: string;
  name: string;
  description: string | null;
  primary_language: string | null;
  stars_count: number;
  forks_count: number;
  open_issues_count: number;
  github_created_at: string | null;
  github_updated_at: string | null;
  collected_at: string;
}

export interface RepositoryDetail extends RepositorySummary {
  total_analyzed_issues: number;
  closed_issues_count: number;
  median_resolution_hours: number | null;
  category_distribution: Record<string, number> | null;
  speed_tier_distribution: Record<string, number> | null;
  snapshot_date: string | null;
}

// ─── Issue Schemas ──────────────────────────────────────────────────────────

export interface IssueLabelItem {
  label_name: string;
  is_canonical: boolean;
  canonical_category: string | null;
}

export interface IssueSummary {
  id: number;
  repo_name: string;
  issue_number: number;
  title: string;
  state: "open" | "closed";
  state_reason: string | null;
  author: string;
  author_association: string;
  is_internal_author: boolean;
  created_at: string;
  closed_at: string | null;
  canonical_label: string;
  canonical_label_id: number;
  speed_tier: string | null;
  comments_count: number;
  reactions_count: number;
  labels_count: number;
}

export interface IssueDetail extends IssueSummary {
  body: string | null;
  clean_title: string | null;
  clean_body: string | null;
  labels: IssueLabelItem[];
  resolution_hours: number | null;
}

export interface IssueListResponse {
  repo_name: string;
  total: number;
  limit: number;
  offset: number;
  state_filter: string | null;
  category_filter: string | null;
  speed_tier_filter: string | null;
  search_query: string | null;
  items: IssueSummary[];
}

export interface IssueQueryParams {
  state?: "open" | "closed";
  category?: string;
  speed_tier?: string;
  search?: string;
  limit?: number;
  offset?: number;
}

// ─── Analytics Schemas ──────────────────────────────────────────────────────

export interface ResolutionMetrics {
  median_hours: number;
  p25_hours: number;
  p75_hours: number;
  iqr_hours: number;
  day_1_closure_pct: number;
  week_1_closure_pct: number;
  month_1_closure_pct: number;
  min_hours: number;
  max_hours: number;
  mean_hours: number;
}

export interface RepositoryAnalyticsResponse {
  repo_name: string;
  total_issues: number;
  open_issues: number;
  closed_issues: number;
  closure_rate: number;
  survival_rate_30d: number | null;
  dominant_category: string;
  category_distribution: Record<string, number>;
  speed_tier_distribution: Record<string, number>;
  resolution_metrics: ResolutionMetrics;
  snapshot_date: string;
}

export interface RepositoryComparisonItem {
  repo_name: string;
  owner: string;
  name: string;
  stars_count: number;
  total_issues: number;
  open_issues: number;
  closed_issues: number;
  closure_rate: number;
  median_resolution_hours: number | null;
  survival_rate_30d: number | null;
  dominant_category: string;
  rapid_closure_percentage: number;
}

export interface RepositoryComparisonResponse {
  total_repositories: number;
  repositories: RepositoryComparisonItem[];
}

export interface GlobalAnalyticsOverview {
  total_repositories: number;
  total_issues: number;
  total_open_issues: number;
  total_closed_issues: number;
  global_closure_rate: number;
  category_distribution: Record<string, number>;
  speed_tier_distribution: Record<string, number>;
  fastest_repository: string | null;
  slowest_repository: string | null;
}

export interface TrendPoint {
  period: string;
  created_count: number;
  closed_count: number;
}

export interface RepositoryTrendsResponse {
  repo_name: string;
  total_periods: number;
  trends: TrendPoint[];
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

// ─── Related Issues Schemas ─────────────────────────────────────────────────

export interface RelatedIssueMatch {
  issue_id: number;
  repo_name: string;
  issue_number: number;
  title: string;
  state: string;
  canonical_label: string;
  similarity_score: number;
  rank: number;
  created_at: string;
  closed_at: string | null;
}

export interface RelatedIssuesResponse {
  query_issue_id: number | null;
  query_text: string | null;
  model_version: string;
  total_matches: number;
  matches: RelatedIssueMatch[];
  retrieval_time_ms: number;
}

export interface RelatedSearchRequest {
  title: string;
  body?: string | null;
  repo_name?: string | null;
  top_k?: number;
  state?: "open" | "closed" | null;
}

// ─── Resolution Prediction Schemas ──────────────────────────────────────────

export interface ResolutionPredictionRequest {
  repo_name: string;
  title: string;
  body?: string | null;
  canonical_label?: string;
  author_association?: string;
  author?: string | null;
  labels_count?: number;
  issue_id?: number | null;
}

export interface ResolutionPredictionResponse {
  repo_name: string;
  predicted_speed_tier: string;
  confidence_score: number;
  tier_probabilities: Record<string, number>;
  lower_bound_tier: string;
  upper_bound_tier: string;
  advisory_factors: string[];
  reliability_notice: string;
  model_used: string;
  inference_time_ms: number;
}

export interface ResolutionStatsResponse {
  repo_name: string;
  n_closed_issues: number;
  median_resolution_hours: number | null;
  p25_resolution_hours: number | null;
  p75_resolution_hours: number | null;
  pct_resolved_under_24h: number | null;
  pct_resolved_under_7d: number | null;
  survival_rate_30d: number | null;
  tier_distribution: Record<string, number>;
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
