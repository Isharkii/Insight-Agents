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

/**
 * Fetch aggregated dashboard data from the backend.
 * Calls GET /api/dashboard which returns the full DashboardData shape.
 */
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

export interface AnalyzeResult {
  insight: string;
  evidence: string;
  impact: string;
  recommended_action: string;
  priority: string;
  confidence_score: number;
  pipeline_status: string;
  diagnostics: unknown;
}

/**
 * Run the full analysis pipeline.
 * Calls POST /analyze with form-data (prompt, optional CSV, etc.).
 */
export async function runAnalysis(params: {
  prompt: string;
  file?: File;
  clientId?: string;
  businessType?: string;
}): Promise<AnalyzeResult> {
  const form = new FormData();
  form.append("prompt", params.prompt);
  if (params.file) form.append("file", params.file);
  if (params.clientId) form.append("client_id", params.clientId);
  if (params.businessType) form.append("business_type", params.businessType);

  return request<AnalyzeResult>("/analyze", {
    method: "POST",
    body: form,
  });
}

/**
 * Check backend health.
 */
export async function checkHealth(): Promise<{ insight: string }> {
  return request<{ insight: string }>("/health");
}
