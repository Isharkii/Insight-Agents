/**
 * API client for the InsightAgent backend.
 *
 * In development, Vite proxies /api/* to the backend.
 * In production, nginx handles the reverse proxy.
 */

import type { DashboardData } from "../components/IntelligenceDashboard/types";

const BASE = import.meta.env.VITE_API_BASE_URL ?? "";

interface BackendError {
  detail:
    | string
    | { code: string; message: string; context?: Record<string, unknown> };
}

function extractErrorMessage(body: BackendError): string {
  if (typeof body.detail === "string") return body.detail;
  return body.detail?.message ?? "Unknown backend error";
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, init);
  if (!res.ok) {
    let message = `HTTP ${res.status}`;
    try {
      const body: BackendError = await res.json();
      message = extractErrorMessage(body);
    } catch {
      /* use status code fallback */
    }
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

async function requestBlob(url: string): Promise<Blob> {
  const res = await fetch(`${BASE}${url}`);
  if (!res.ok) {
    let message = `HTTP ${res.status}`;
    try {
      const body: BackendError = await res.json();
      message = extractErrorMessage(body);
    } catch {
      /* use status code fallback */
    }
    throw new Error(message);
  }
  return res.blob();
}

// ─── Dashboard ───────────────────────────────────────────────────────────────

export async function fetchDashboard(
  entityName: string,
  businessType: string,
): Promise<DashboardData> {
  const params = new URLSearchParams({
    entity_name: entityName,
    business_type: businessType,
  });
  return request<DashboardData>(`/api/dashboard?${params}`);
}

// ─── Analysis ────────────────────────────────────────────────────────────────

export interface DiagnosticsData {
  warnings?: string[];
  confidence_score?: number;
  missing_signal?: string[];
  confidence_adjustments?: {
    signal: string;
    delta: number;
    reason: string;
  }[];
  [key: string]: unknown;
}

export interface AnalyzeResult {
  insight: string;
  evidence: string;
  impact: string;
  recommended_action: string;
  priority: string;
  confidence_score: number;
  pipeline_status: string;
  diagnostics: DiagnosticsData | null;
}

export interface AnalyzeRunResponse {
  result: AnalyzeResult;
  resolvedEntityName?: string;
  resolvedBusinessType?: string;
}

export async function runAnalysis(params: {
  prompt: string;
  file?: File;
  clientId?: string;
  businessType?: string;
  multiEntityBehavior?: string;
  model?: string;
}): Promise<AnalyzeRunResponse> {
  const form = new FormData();
  form.append("prompt", params.prompt);
  if (params.file) form.append("file", params.file);
  if (params.clientId) form.append("client_id", params.clientId);
  if (params.businessType) form.append("business_type", params.businessType);
  if (params.multiEntityBehavior)
    form.append("multi_entity_behavior", params.multiEntityBehavior);
  if (params.model && params.model !== "default")
    form.append("model", params.model);

  const res = await fetch(`${BASE}/analyze`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    let message = `HTTP ${res.status}`;
    try {
      const body: BackendError = await res.json();
      message = extractErrorMessage(body);
    } catch {
      /* use status code fallback */
    }
    throw new Error(message);
  }

  const result = (await res.json()) as AnalyzeResult;
  const resolvedEntityName =
    res.headers.get("X-Resolved-Entity-Name")?.trim() || undefined;
  const resolvedBusinessType =
    res.headers.get("X-Resolved-Business-Type")?.trim() || undefined;

  return {
    result,
    resolvedEntityName,
    resolvedBusinessType,
  };
}

// ─── Clients ─────────────────────────────────────────────────────────────────

export async function fetchClients(): Promise<string[]> {
  try {
    const data = await request<{ clients: string[] } | string[]>("/clients");
    if (Array.isArray(data)) return data;
    if (data && Array.isArray(data.clients)) return data.clients;
    return [];
  } catch {
    return [];
  }
}

// ─── Exports ─────────────────────────────────────────────────────────────────

export async function fetchExportBlob(
  dataset: string,
  format: string,
  entityName?: string,
): Promise<Blob> {
  const params = new URLSearchParams({ dataset, format });
  if (entityName) params.set("entity_name", entityName);
  return requestBlob(`/export/powerbi?${params}`);
}

export async function fetchExportJson(
  dataset: string,
  entityName?: string,
  limit = 2000,
): Promise<Record<string, unknown> | null> {
  const params = new URLSearchParams({
    dataset,
    format: "json",
    limit: String(limit),
  });
  if (entityName) params.set("entity_name", entityName);
  try {
    return await request<Record<string, unknown>>(`/export/powerbi?${params}`);
  } catch {
    return null;
  }
}

export async function fetchReportPayload(
  entityName: string,
  prompt: string,
  businessType?: string,
): Promise<Record<string, unknown> | null> {
  const params = new URLSearchParams({
    entity_name: entityName,
    prompt,
    format: "json",
  });
  if (businessType) params.set("business_type", businessType);
  try {
    return await request<Record<string, unknown>>(`/export/report?${params}`);
  } catch {
    return null;
  }
}

export async function fetchReportMarkdownBlob(
  entityName: string,
  prompt: string,
  businessType?: string,
): Promise<Blob> {
  const params = new URLSearchParams({
    entity_name: entityName,
    prompt,
    format: "md",
  });
  if (businessType) params.set("business_type", businessType);
  return requestBlob(`/export/report?${params}`);
}

// ─── Health ──────────────────────────────────────────────────────────────────

export async function checkHealth(): Promise<{ insight: string }> {
  return request<{ insight: string }>("/health");
}
