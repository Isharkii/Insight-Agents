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

export interface PipelineSignalRisk {
  status: string;
  risk_score: number | null;
  risk_level: string | null;
  confidence: number;
  breakdown: Record<string, unknown> | null;
}

export interface PipelineSignalPrioritization {
  priority_level: string | null;
  recommended_focus: string | null;
  confidence_score: number | null;
  growth_short: number | null;
  growth_mid: number | null;
  growth_long: number | null;
  growth_trend_acceleration: number | null;
  cohort_signal_used: boolean | null;
  cohort_risk_hint: string | null;
  scenario_signal_used: boolean | null;
  scenario_worst_growth: number | null;
  scenario_best_growth: number | null;
  signal_conflict_count: number | null;
  signal_conflict_warnings: string[] | null;
}

export interface PipelineSignalGrowth {
  status: string;
  confidence: number;
  short_growth: number | null;
  mid_growth: number | null;
  long_growth: number | null;
  trend_acceleration: number | null;
  metric_series: Record<string, number[]> | null;
}

export interface PipelineSignalForecast {
  status: string;
  confidence: number;
  forecasts: Record<string, unknown> | null;
  metrics_queried: string[] | null;
}

export interface PipelineSignalCohort {
  status: string;
  confidence: number;
  retention_decay: number | null;
  churn_acceleration: number | null;
  worst_cohort: Record<string, unknown> | null;
  risk_hint: string | null;
}

export interface PipelineSignalConflicts {
  status: string;
  conflict_count: number;
  conflicts: unknown[] | null;
  total_severity: number | null;
  uncertainty_flag: boolean;
  warnings: string[] | null;
}

export interface PipelineSignalUnitEconomics {
  status: string;
  confidence: number;
  ltv: number | null;
  cac: number | null;
  ltv_cac_ratio: number | null;
  payback_months: number | null;
}

export interface PipelineSignalScenarios {
  status: string;
  confidence: number;
  scenario_simulation: Record<string, unknown> | null;
}

export interface PipelineSignalBenchmark {
  status: string;
  confidence: number;
  ranking?: {
    overall_rank?: number;
    total_participants?: number;
    overall_percentile?: number;
    tier?: string;
    peer_scores?: Record<string, number>;
    skipped_metrics?: Record<string, string>;
    metric_ranks?: Record<
      string,
      {
        rank?: number;
        percentile?: number;
        client_value?: number;
        field_mean?: number;
        field_median?: number;
      }
    >;
  } | null;
  composite?: {
    overall_score?: number;
    base_overall_score?: number;
    growth_score?: number;
    level_score?: number;
    stability_score?: number;
    confidence_score?: number;
    competitive_metrics?: {
      relative_growth_index?: number | null;
      market_share_proxy?: number | null;
      stability_score?: number;
      momentum_classification?: string;
      risk_divergence_score?: number | null;
    };
  } | null;
  peer_selection?: {
    peer_candidates?: string[];
    selected_peers?: string[];
  } | null;
  market_position?: string | null;
  metric_comparison_specs?: Record<
    string,
    {
      direction?: string;
      unit?: string;
      scale?: string;
      aggregation?: string;
      window_alignment?: string;
    }
  > | null;
}

export interface PipelineSignals {
  risk?: PipelineSignalRisk;
  prioritization?: PipelineSignalPrioritization;
  growth?: PipelineSignalGrowth;
  forecast?: PipelineSignalForecast;
  cohort?: PipelineSignalCohort;
  signal_integrity?: Record<string, unknown>;
  signal_conflicts?: PipelineSignalConflicts;
  unit_economics?: PipelineSignalUnitEconomics;
  scenarios?: PipelineSignalScenarios;
  benchmark?: PipelineSignalBenchmark;
  pipeline_status?: string;
  dataset_confidence?: number | null;
  synthesis_blocked?: boolean | null;
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
  pipeline_signals: PipelineSignals | null;
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

const _MIN_DISPLAY_CONFIDENCE = 0.15;

function derivePipelineStatus(confidence: number): string {
  if (confidence <= 0) return "partial";
  if (confidence < 0.8) return "partial";
  return "success";
}

function sanitizePipelineStatus(status: string): string {
  const normalized = String(status || "").trim().toLowerCase();
  if (normalized === "success" || normalized === "partial") return normalized;
  if (normalized === "failed" || normalized === "blocked") return "partial";
  return "partial";
}

function sanitizeDisplayConfidence(confidence: number, pipelineStatus: string): number {
  const status = sanitizePipelineStatus(pipelineStatus);
  const value = Number.isFinite(confidence) ? confidence : 0;
  if (value > 0) return value;
  // Do not surface a literal 0% in degraded/failed states.
  if (status === "partial") return _MIN_DISPLAY_CONFIDENCE;
  return _MIN_DISPLAY_CONFIDENCE;
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

function extractPipelineSignals(value: unknown): PipelineSignals | null {
  if (!value || typeof value !== "object") return null;
  const payload = value as Record<string, unknown>;
  const raw = payload.pipeline_signals;
  if (!raw || typeof raw !== "object") return null;
  return raw as PipelineSignals;
}

export function normalizeAnalyzeResult(value: unknown): AnalyzeResult {
  const pipelineSignals = extractPipelineSignals(value);
  const structured = asStructuredInsight(value);
  if (structured) {
    const rawConfidence = coerceNumber(
      structured.competitive_analysis?.confidence,
      0,
    );
    const pipelineStatus = sanitizePipelineStatus(
      derivePipelineStatus(rawConfidence)
    );
    const confidence = sanitizeDisplayConfidence(rawConfidence, pipelineStatus);
    const firstImmediate =
      structured.strategic_recommendations?.immediate_actions?.[0] ?? "";
    return {
      insight: structured.competitive_analysis?.summary ?? "",
      evidence: structured.competitive_analysis?.relative_performance ?? "",
      impact: structured.competitive_analysis?.market_position ?? "",
      recommended_action: firstImmediate,
      priority: derivePriority(confidence),
      confidence_score: confidence,
      pipeline_status: pipelineStatus,
      diagnostics: null,
      pipeline_signals: pipelineSignals,
    };
  }

  const payload = (value || {}) as Record<string, unknown>;
  const rawPipelineStatus = String(payload.pipeline_status ?? "partial");
  const pipelineStatus = sanitizePipelineStatus(rawPipelineStatus);
  const rawConfidence = coerceNumber(payload.confidence_score, 0);
  const confidenceScore = sanitizeDisplayConfidence(rawConfidence, pipelineStatus);
  return {
    insight: String(payload.insight ?? ""),
    evidence: String(payload.evidence ?? ""),
    impact: String(payload.impact ?? ""),
    recommended_action: String(payload.recommended_action ?? ""),
    priority: String(payload.priority ?? "low"),
    confidence_score: confidenceScore,
    pipeline_status: pipelineStatus,
    diagnostics:
      payload.diagnostics && typeof payload.diagnostics === "object"
        ? (payload.diagnostics as DiagnosticsData)
        : null,
    pipeline_signals: pipelineSignals,
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
  competitors?: string;
  selfAnalysisOnly?: boolean;
  model?: string;
}): Promise<AnalyzeRunResponse> {
  const form = new FormData();
  form.append("prompt", params.prompt);
  if (params.file) form.append("file", params.file);
  if (params.clientId) form.append("client_id", params.clientId);
  if (params.businessType) form.append("business_type", params.businessType);
  if (params.multiEntityBehavior)
    form.append("multi_entity_behavior", params.multiEntityBehavior);
  if (params.competitors) form.append("competitors", params.competitors);
  if (params.selfAnalysisOnly) form.append("self_analysis_only", "true");
  if (params.model && params.model !== "default")
    form.append("model", params.model);

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 110_000);
  let res: Response;
  try {
    res = await fetch(buildUrl("/analyze"), {
      method: "POST",
      body: form,
      signal: controller.signal,
    });
  } catch (err) {
    clearTimeout(timeoutId);
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error(
        "Analysis timed out. Try a smaller dataset or simpler query.",
      );
    }
    throw err;
  }
  clearTimeout(timeoutId);
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
  competitors?: string,
  selfAnalysisOnly?: boolean,
): Promise<Record<string, unknown> | null> {
  const params = new URLSearchParams({
    entity_name: entityName,
    prompt,
    format: "json",
  });
  if (businessType) params.set("business_type", businessType);
  if (competitors) params.set("competitors", competitors);
  if (selfAnalysisOnly) params.set("self_analysis_only", "true");
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
