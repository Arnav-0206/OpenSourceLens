/**
 * OpenSourceLens — Strongly-Typed API Client
 *
 * Provides a client library for FastAPI REST endpoints with error handling,
 * type safety, and query parameter serialization.
 */

import {
  ClassificationRequest,
  ClassificationResponse,
  ErrorResponse,
  HealthResponse,
  ModelInfoItem,
  ModelRegistryResponse,
  RootResponse,
} from "./types";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";
const SERVER_ROOT_URL =
  API_BASE_URL.replace(/\/api\/v1\/?$/, "") || "http://localhost:8000";

const DEFAULT_TIMEOUT_MS = 10000;

export interface RequestOptions extends RequestInit {
  timeoutMs?: number;
}

export interface BackendConnectionPingResult {
  ok: boolean;
  latencyMs: number;
  status: string;
  version?: string;
  databaseStatus?: string;
  databaseDialect?: string;
  error?: string;
}

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
 * Generic JSON fetch wrapper with standardized error decoding and timeout protection.
 */
export async function fetchJson<T>(
  url: string,
  options?: RequestOptions
): Promise<T> {
  const headers = new Headers(options?.headers);
  if (!headers.has("Accept")) {
    headers.set("Accept", "application/json");
  }
  if (options?.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const timeoutMs = options?.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      headers,
      signal: options?.signal || controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      let errorData: any = null;
      let detailMessage = `Request failed with status ${response.status}`;
      let errorCode = "HTTP_ERROR";

      try {
        errorData = await response.json();
        if (typeof errorData?.detail === "string") {
          detailMessage = errorData.detail;
        } else if (Array.isArray(errorData?.detail)) {
          detailMessage = errorData.detail
            .map((e: any) => e.msg || JSON.stringify(e))
            .join(", ");
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

    return (await response.json()) as T;
  } catch (err: any) {
    clearTimeout(timeoutId);
    if (err.name === "AbortError") {
      throw new ApiClientError(
        408,
        `Request to ${url} timed out after ${timeoutMs}ms`,
        "TIMEOUT",
        url
      );
    }
    throw err;
  }
}

// ─── System & Health Endpoints ──────────────────────────────────────────────

export async function getSystemRoot(): Promise<RootResponse> {
  return fetchJson<RootResponse>(`${SERVER_ROOT_URL}/`);
}

export async function getHealthStatus(): Promise<HealthResponse> {
  return fetchJson<HealthResponse>(`${SERVER_ROOT_URL}/health`, {
    cache: "no-store",
    timeoutMs: 5000,
  });
}

/**
 * Probe live backend connectivity, reporting round-trip latency and database health.
 */
export async function pingBackendConnection(): Promise<BackendConnectionPingResult> {
  const start = performance.now();
  try {
    const health = await getHealthStatus();
    const latency = Math.round(performance.now() - start);
    return {
      ok: health.status === "healthy",
      latencyMs: latency,
      status: health.status,
      version: health.version,
      databaseStatus: health.database?.status,
      databaseDialect: health.database?.dialect,
    };
  } catch (err: any) {
    const latency = Math.round(performance.now() - start);
    return {
      ok: false,
      latencyMs: latency,
      status: "unreachable",
      error: err?.message || "Failed to reach backend",
    };
  }
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
