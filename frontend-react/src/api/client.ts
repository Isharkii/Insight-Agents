/**
 * API client for the InsightAgent backend.
 *
 * In development, Vite proxies /api/* to the backend.
 * In production, nginx handles the reverse proxy.
 */

import type { DashboardData } from "../components/IntelligenceDashboard/types";

const RAW_BASE = import.meta.env.VITE_API_BASE_URL;
const BASE =
  typeof RAW_BASE === "string" && RAW_BASE.trim()
    ? RAW_BASE.trim().replace(/\/+$/, "")
    : "";

function buildUrl(path: string): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  if (!BASE) return normalizedPath;
  return `${BASE}${normalizedPath}`;
}

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
  const res = await fetch(buildUrl(url), init);
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
  const res = await fetch(buildUrl(url));
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

export interface StructuredInsightOutput {
  competitive_analysis: {
    summary: string;
    market_position: string;
    relative_performance: string;
    key_advantages: string[];
    key_vulnerabilities: string[];
    confidence: number;
  };
  strategic_recommendations: {
    immediate_actions: string[];
    mid_term_moves: string[];
    defensive_strategies: string[];
    offensive_strategies: string[];
  };
}

export interface AnalyzeRunResponse {
  result: AnalyzeResult;
  resolvedEntityName?: string;
  resolvedBusinessType?: string;
}

function coerceNumber(value: unknown, fallback: number): number {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  return fallback;
}

function derivePriority(confidence: number): string {
  if (confidence >= 0.8) return "high";
  if (confidence >= 0.6) return "medium";
  if (confidence >= 0.4) return "low";
  return "critical";
}

function derivePipelineStatus(confidence: number): string {
  if (confidence <= 0) return "failed";
  if (confidence < 0.8) return "partial";
  return "success";
}

function asStructuredInsight(value: unknown): StructuredInsightOutput | null {
  if (!value || typeof value !== "object") return null;
  const payload = value as Record<string, unknown>;
  const competitive = payload.competitive_analysis;
  const strategic = payload.strategic_recommendations;
  if (!competitive || typeof competitive !== "object") return null;
  if (!strategic || typeof strategic !== "object") return null;
  return value as StructuredInsightOutput;
}

export function normalizeAnalyzeResult(value: unknown): AnalyzeResult {
  const structured = asStructuredInsight(value);
  if (structured) {
    const confidence = coerceNumber(
      structured.competitive_analysis?.confidence,
      0,
    );
    const firstImmediate =
      structured.strategic_recommendations?.immediate_actions?.[0] ?? "";
    return {
      insight: structured.competitive_analysis?.summary ?? "",
      evidence: structured.competitive_analysis?.relative_performance ?? "",
      impact: structured.competitive_analysis?.market_position ?? "",
      recommended_action: firstImmediate,
      priority: derivePriority(confidence),
      confidence_score: confidence,
      pipeline_status: derivePipelineStatus(confidence),
      diagnostics: null,
    };
  }

  const payload = (value || {}) as Record<string, unknown>;
  return {
    insight: String(payload.insight ?? ""),
    evidence: String(payload.evidence ?? ""),
    impact: String(payload.impact ?? ""),
    recommended_action: String(payload.recommended_action ?? ""),
    priority: String(payload.priority ?? "low"),
    confidence_score: coerceNumber(payload.confidence_score, 0),
    pipeline_status: String(payload.pipeline_status ?? "partial"),
    diagnostics:
      payload.diagnostics && typeof payload.diagnostics === "object"
        ? (payload.diagnostics as DiagnosticsData)
        : null,
  };
}

// —— Business Intelligence ———————————————————————————————————————————————————————

export interface BIPipelineStageResult {
  stage: string;
  status: "success" | "failed" | "skipped";
  duration_ms: number;
  error?: string | null;
}

export interface BusinessContextData {
  industry: string;
  business_model: string;
  target_market: string;
  macro_dependencies: string[];
  search_intents: string[];
  risk_factors: string[];
}

export interface BISignalReference {
  signal_id: string;
  metric_name: string;
  value: number;
  unit: string;
}

export interface BIEmergingSignal {
  title: string;
  description: string;
  supporting_signals: BISignalReference[];
  relevance: "high" | "medium" | "low";
}

export interface BIZone {
  title: string;
  description: string;
  supporting_signals: BISignalReference[];
}

export interface BIInsightBlock {
  emerging_signals: BIEmergingSignal[];
  macro_summary: string;
  opportunity_zones: BIZone[];
  risk_zones: BIZone[];
  momentum_score: number;
  confidence: number;
}

export interface BIStrategyAction {
  action: string;
  rationale: string;
  supporting_signals: BISignalReference[];
  priority: "critical" | "high" | "medium";
}

export interface BICompetitiveAngle {
  positioning: string;
  differentiation: string;
  supporting_signals: BISignalReference[];
}

export interface BIRiskMitigation {
  risk_title: string;
  mitigation: string;
  supporting_signals: BISignalReference[];
}

export interface BIStrategyBlock {
  short_term_actions: BIStrategyAction[];
  mid_term_actions: BIStrategyAction[];
  long_term_positioning: string;
  competitive_angle: BICompetitiveAngle;
  risk_mitigation: BIRiskMitigation[];
  confidence: number;
}

export interface BusinessIntelligenceResponse {
  status: "success" | "partial" | "failed";
  context: BusinessContextData | null;
  insights: BIInsightBlock | null;
  strategy: BIStrategyBlock | null;
  confidence: number;
  pipeline: BIPipelineStageResult[];
  warnings: string[];
  generated_at: string;
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

  const res = await fetch(buildUrl("/analyze"), {
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

  const raw = (await res.json()) as unknown;
  const result = normalizeAnalyzeResult(raw);
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

export async function runBusinessIntelligence(params: {
  businessPrompt: string;
}): Promise<BusinessIntelligenceResponse> {
  return request<BusinessIntelligenceResponse>("/api/business-intelligence", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      business_prompt: params.businessPrompt,
    }),
  });
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

export async function fetchBIWorkbookBlob(
  entityName: string,
  prompt: string,
  businessType?: string,
): Promise<Blob> {
  const params = new URLSearchParams({
    entity_name: entityName,
    prompt,
  });
  if (businessType) params.set("business_type", businessType);
  return requestBlob(`/export/bi-workbook?${params}`);
}

// ─── Health ──────────────────────────────────────────────────────────────────

export async function checkHealth(): Promise<{ insight: string }> {
  return request<{ insight: string }>("/health");
}
