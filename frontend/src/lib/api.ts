/**
 * OpenSourceLens — Strongly-Typed API Client
 *
 * Provides a client library for all FastAPI REST endpoints with error handling,
 * type safety, and query parameter serialization.
 */

import {
  ClassificationRequest,
  ClassificationResponse,
  ErrorResponse,
  GlobalAnalyticsOverview,
  HealthResponse,
  IssueDetail,
  IssueListResponse,
  IssueQueryParams,
  ModelInfoItem,
  ModelRegistryResponse,
  RelatedIssuesResponse,
  RelatedSearchRequest,
  RepositoryAnalyticsResponse,
  RepositoryComparisonResponse,
  RepositoryDetail,
  RepositorySummary,
  RepositoryTrendsResponse,
  ResolutionPredictionRequest,
  ResolutionPredictionResponse,
  ResolutionStatsResponse,
  RootResponse,
} from "./types";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";
const SERVER_ROOT_URL =
  API_BASE_URL.replace(/\/api\/v1\/?$/, "") || "http://localhost:8000";

export class ApiClientError extends Error {
  public statusCode: number;
  public errorCode: string;
  public path: string;
  public raw: ErrorResponse | any;

  constructor(
    statusCode: number,
    message: string,
    errorCode = "CLIENT_ERROR",
    path = "",
    raw: any = null
  ) {
    super(message);
    this.name = "ApiClientError";
    this.statusCode = statusCode;
    this.errorCode = errorCode;
    this.path = path;
    this.raw = raw;
  }
}

/**
 * Generic JSON fetch wrapper with standardized error decoding.
 */
async function fetchJson<T>(
  url: string,
  options?: RequestInit
): Promise<T> {
  const headers = new Headers(options?.headers);
  if (!headers.has("Accept")) {
    headers.set("Accept", "application/json");
  }
  if (options?.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(url, {
    ...options,
    headers,
  });

  if (!response.ok) {
    let errorData: any = null;
    let detailMessage = `Request failed with status ${response.status}`;
    let errorCode = "HTTP_ERROR";

    try {
      errorData = await response.json();
      if (typeof errorData?.detail === "string") {
        detailMessage = errorData.detail;
      } else if (Array.isArray(errorData?.detail)) {
        detailMessage = errorData.detail.map((e: any) => e.msg || JSON.stringify(e)).join(", ");
      }
      if (errorData?.error_code) {
        errorCode = errorData.error_code;
      }
    } catch {
      // Non-JSON error body
    }

    throw new ApiClientError(
      response.status,
      detailMessage,
      errorCode,
      url,
      errorData
    );
  }

  return response.json() as Promise<T>;
}

// ─── System & Health Endpoints ──────────────────────────────────────────────

export async function getSystemRoot(): Promise<RootResponse> {
  return fetchJson<RootResponse>(`${SERVER_ROOT_URL}/`);
}

export async function getHealthStatus(): Promise<HealthResponse> {
  return fetchJson<HealthResponse>(`${SERVER_ROOT_URL}/health`, {
    cache: "no-store",
  });
}

// ─── Repository & Issues Endpoints ──────────────────────────────────────────

export async function getRepositories(): Promise<RepositorySummary[]> {
  return fetchJson<RepositorySummary[]>(`${API_BASE_URL}/repos`);
}

export async function getRepository(
  owner: string,
  name: string
): Promise<RepositoryDetail> {
  return fetchJson<RepositoryDetail>(
    `${API_BASE_URL}/repos/${encodeURIComponent(owner)}/${encodeURIComponent(name)}`
  );
}

export async function getRepositoryIssues(
  owner: string,
  name: string,
  params?: IssueQueryParams
): Promise<IssueListResponse> {
  const url = new URL(
    `${API_BASE_URL}/repos/${encodeURIComponent(owner)}/${encodeURIComponent(name)}/issues`
  );
  if (params?.state) url.searchParams.set("state", params.state);
  if (params?.category) url.searchParams.set("category", params.category);
  if (params?.speed_tier) url.searchParams.set("speed_tier", params.speed_tier);
  if (params?.search) url.searchParams.set("search", params.search);
  if (params?.limit !== undefined) url.searchParams.set("limit", String(params.limit));
  if (params?.offset !== undefined) url.searchParams.set("offset", String(params.offset));

  return fetchJson<IssueListResponse>(url.toString());
}

export async function getIssue(
  owner: string,
  name: string,
  issueNumber: number
): Promise<IssueDetail> {
  return fetchJson<IssueDetail>(
    `${API_BASE_URL}/repos/${encodeURIComponent(owner)}/${encodeURIComponent(name)}/issues/${issueNumber}`
  );
}

// ─── Analytics Endpoints ────────────────────────────────────────────────────

export async function getAnalyticsOverview(): Promise<GlobalAnalyticsOverview> {
  return fetchJson<GlobalAnalyticsOverview>(`${API_BASE_URL}/analytics/overview`);
}

export async function getAnalyticsComparison(): Promise<RepositoryComparisonResponse> {
  return fetchJson<RepositoryComparisonResponse>(`${API_BASE_URL}/analytics/comparison`);
}

export async function getRepositoryAnalytics(
  owner: string,
  name: string
): Promise<RepositoryAnalyticsResponse> {
  return fetchJson<RepositoryAnalyticsResponse>(
    `${API_BASE_URL}/analytics/${encodeURIComponent(owner)}/${encodeURIComponent(name)}`
  );
}

export async function getRepositoryTrends(
  owner: string,
  name: string
): Promise<RepositoryTrendsResponse> {
  return fetchJson<RepositoryTrendsResponse>(
    `${API_BASE_URL}/analytics/${encodeURIComponent(owner)}/${encodeURIComponent(name)}/trends`
  );
}

// ─── Classification Endpoint ────────────────────────────────────────────────

export async function classifyIssue(
  request: ClassificationRequest
): Promise<ClassificationResponse> {
  return fetchJson<ClassificationResponse>(`${API_BASE_URL}/classify`, {
    method: "POST",
    body: JSON.stringify(request),
  });
}

// ─── Related Issues Endpoints ───────────────────────────────────────────────

export async function getRelatedIssues(
  issueId: number,
  params?: { top_k?: number; state?: string }
): Promise<RelatedIssuesResponse> {
  const url = new URL(`${API_BASE_URL}/related/${issueId}`);
  if (params?.top_k !== undefined) url.searchParams.set("top_k", String(params.top_k));
  if (params?.state) url.searchParams.set("state", params.state);
  return fetchJson<RelatedIssuesResponse>(url.toString());
}

export async function searchRelatedIssues(
  request: RelatedSearchRequest
): Promise<RelatedIssuesResponse> {
  return fetchJson<RelatedIssuesResponse>(`${API_BASE_URL}/related`, {
    method: "POST",
    body: JSON.stringify(request),
  });
}

// ─── Resolution Velocity Endpoints ─────────────────────────────────────────

export async function predictResolution(
  request: ResolutionPredictionRequest
): Promise<ResolutionPredictionResponse> {
  return fetchJson<ResolutionPredictionResponse>(`${API_BASE_URL}/resolution/predict`, {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export async function getResolutionStats(
  owner: string,
  name: string
): Promise<ResolutionStatsResponse> {
  return fetchJson<ResolutionStatsResponse>(
    `${API_BASE_URL}/resolution/stats/${encodeURIComponent(owner)}/${encodeURIComponent(name)}`
  );
}

// ─── Model Registry Endpoints ───────────────────────────────────────────────

export async function getModels(): Promise<ModelRegistryResponse> {
  return fetchJson<ModelRegistryResponse>(`${API_BASE_URL}/models`);
}

export async function getModelDetails(modelId: string): Promise<ModelInfoItem> {
  return fetchJson<ModelInfoItem>(`${API_BASE_URL}/models/${encodeURIComponent(modelId)}`);
}

export async function prewarmModels(): Promise<Record<string, string>> {
  return fetchJson<Record<string, string>>(`${API_BASE_URL}/models/prewarm`, {
    method: "POST",
  });
}
