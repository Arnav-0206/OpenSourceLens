/**
 * OpenSourceLens — Core TypeScript Data Contracts
 *
 * Direct typed mappings matching backend FastAPI schemas.
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
