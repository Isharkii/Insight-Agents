import { useState, useCallback, useEffect, useMemo } from "react";

import Sidebar, { type SidebarState } from "./components/Sidebar";
import CsvUpload from "./components/CsvUpload";
import InsightsDashboard from "./components/InsightsDashboard";
import ExportPanel from "./components/ExportPanel";
import IntelligenceDashboard from "./components/IntelligenceDashboard";

import type { DashboardData } from "./components/IntelligenceDashboard/types";
import {
  fetchDashboard,
  runAnalysis,
  runBusinessIntelligence,
  normalizeAnalyzeResult,
  fetchReportPayload,
  fetchExportJson,
  type AnalyzeResult,
  type BusinessIntelligenceResponse,
} from "./api/client";

// ─── Embed mode hook ─────────────────────────────────────────────────────────

function useEmbedParams() {
  const params = new URLSearchParams(window.location.search);
  const embed = params.get("embed") === "1";
  const entity = params.get("entity_name") ?? "";
  const btype = params.get("business_type") ?? "saas";
  return { embed, entity, btype };
}

// ─── Derived signals helpers ─────────────────────────────────────────────────

interface ReportDerivedSignals {
  risk?: { risk_score?: number; risk_level?: string };
  role_contribution?: {
    top_contributors?: { name: string; contribution_value: number }[];
  };
  multivariate_scenario?: {
    scenario_simulation?: {
      scenarios?: Record<
        string,
        { projected_value?: number; projected_growth?: number }
      >;
    };
  };
  competitive_benchmark?: {
    status?: string;
    reason?: string;
    peer_selection?: {
      peer_candidates?: string[];
      selected_peers?: string[];
    };
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
    };
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
    };
    metric_comparison_specs?: Record<
      string,
      {
        direction?: string;
        unit?: string;
        scale?: string;
        aggregation?: string;
        window_alignment?: string;
      }
    >;
  };
}

function extractDerivedSignals(
  reportPayload: Record<string, unknown> | null,
): ReportDerivedSignals {
  if (!reportPayload) return {};
  const ds = reportPayload.derived_signals;
  if (typeof ds === "object" && ds !== null) return ds as ReportDerivedSignals;
  return {};
}

// ─── KPI rows from export JSON ───────────────────────────────────────────────

interface KpiRow {
  period_end: string;
  metric_name: string;
  metric_value: number;
}

function extractKpiRows(
  exportJson: Record<string, unknown> | null,
): KpiRow[] {
  if (!exportJson) return [];
  const data = exportJson.data;
  if (!Array.isArray(data)) return [];
  return data.filter(
    (r): r is KpiRow =>
      typeof r === "object" &&
      r !== null &&
      "period_end" in r &&
      "metric_name" in r &&
      "metric_value" in r,
  );
}

// ─── Main App ────────────────────────────────────────────────────────────────

export default function App() {
  const { embed, entity: embedEntity, btype: embedBtype } = useEmbedParams();

  // Sidebar state
  const [sidebar, setSidebar] = useState<SidebarState>({
    mode: "LOCAL",
    clientId: "default",
    entityOverride: embedEntity,
    businessType: embedBtype || "auto",
    multiEntityBehavior: "auto",
    model: "default",
  });
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // Input state
  const [prompt, setPrompt] = useState("");
  const [file, setFile] = useState<File | null>(null);

  // Result state
  const [analyzeResult, setAnalyzeResult] = useState<AnalyzeResult | null>(
    null,
  );
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(
    null,
  );
  const [biResult, setBiResult] = useState<BusinessIntelligenceResponse | null>(
    null,
  );
  const [reportPayload, setReportPayload] = useState<Record<
    string,
    unknown
  > | null>(null);
  const [kpiExportJson, setKpiExportJson] = useState<Record<
    string,
    unknown
  > | null>(null);
  const [executionTime, setExecutionTime] = useState<number | undefined>();
  const [resolvedContext, setResolvedContext] = useState<{
    entityName?: string;
    businessType?: string;
  }>({});

  // UI state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Derived
  const requestedEntity = useMemo(() => {
    const override = sidebar.entityOverride.trim();
    if (override) return override;
    return sidebar.clientId !== "default" ? sidebar.clientId : "";
  }, [sidebar.entityOverride, sidebar.clientId]);

  const requestedBusinessType = useMemo(
    () => (sidebar.businessType === "auto" ? undefined : sidebar.businessType),
    [sidebar.businessType],
  );

  const effectiveEntity =
    resolvedContext.entityName?.trim() || requestedEntity || "";

  const effectiveBusinessType =
    resolvedContext.businessType?.trim() || requestedBusinessType;

  const reportInsight = useMemo(() => {
    const payload = reportPayload?.insight_payload;
    if (payload) {
      return normalizeAnalyzeResult(payload);
    }
    return null;
  }, [reportPayload]);

  const derivedSignals = useMemo(
    () => extractDerivedSignals(reportPayload),
    [reportPayload],
  );
  const kpiRows = useMemo(
    () => extractKpiRows(kpiExportJson),
    [kpiExportJson],
  );

  // ─── Handlers ──────────────────────────────────────────────────────────────

  const handleRun = useCallback(async () => {
    if (!prompt.trim()) {
      setError("Please enter a strategic business prompt.");
      return;
    }
    if (!file && !requestedEntity) {
      setError("Entity / Client ID is required when no CSV is uploaded.");
      return;
    }

    setLoading(true);
    setError(null);
    setAnalyzeResult(null);
    setDashboardData(null);
    setBiResult(null);
    setReportPayload(null);
    setKpiExportJson(null);
    setResolvedContext({});

    const started = performance.now();
    try {
      const biPromise = runBusinessIntelligence({
        businessPrompt: prompt.trim(),
      })
        .then((payload) => {
          setBiResult(payload);
          return payload;
        })
        .catch(() => {
          setBiResult(null);
          return null;
        });

      const run = await runAnalysis({
        prompt: prompt.trim(),
        file: file ?? undefined,
        clientId: requestedEntity || undefined,
        businessType: requestedBusinessType,
        multiEntityBehavior:
          sidebar.multiEntityBehavior === "auto"
            ? undefined
            : sidebar.multiEntityBehavior,
        model: sidebar.model,
      });
      const result = run.result;
      setAnalyzeResult(result);
      setExecutionTime((performance.now() - started) / 1000);

      // Fetch dashboard + report + KPI export in parallel
      const entity = (run.resolvedEntityName || requestedEntity || "").trim();
      const btype = (run.resolvedBusinessType || requestedBusinessType || "saas").trim();

      setResolvedContext({
        entityName: entity || undefined,
        businessType: btype || undefined,
      });

      const enrichmentTasks: Promise<unknown>[] = [biPromise];
      if (entity) {
        enrichmentTasks.push(
          fetchDashboard(entity, btype)
            .then(setDashboardData)
            .catch(() => {}),
          fetchReportPayload(entity, prompt.trim(), btype)
            .then(setReportPayload)
            .catch(() => {}),
          fetchExportJson("kpis", entity)
            .then(setKpiExportJson)
            .catch(() => {}),
        );
      }
      await Promise.all(enrichmentTasks);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [
    prompt,
    file,
    requestedEntity,
    requestedBusinessType,
    sidebar.multiEntityBehavior,
    sidebar.model,
  ]);

  const handleClear = useCallback(() => {
    setAnalyzeResult(null);
    setDashboardData(null);
    setBiResult(null);
    setReportPayload(null);
    setKpiExportJson(null);
    setError(null);
    setExecutionTime(undefined);
    setResolvedContext({});
    setPrompt("");
    setFile(null);
  }, []);

  // ─── Embed mode ────────────────────────────────────────────────────────────

  useEffect(() => {
    if (embed && embedEntity) {
      setLoading(true);
      setError(null);
      fetchDashboard(embedEntity, embedBtype)
        .then(setDashboardData)
        .catch((err) =>
          setError(err instanceof Error ? err.message : String(err)),
        )
        .finally(() => setLoading(false));
    }
  }, [embed, embedEntity, embedBtype]);

  if (embed) {
    return (
      <div className="bg-gray-50 dark:bg-gray-950">
        {loading && (
          <div className="flex items-center justify-center py-16">
            <p className="text-gray-400 dark:text-gray-500">
              Loading dashboard...
            </p>
          </div>
        )}
        {error && (
          <div className="max-w-7xl mx-auto px-4 py-4">
            <div className="rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 p-4 text-sm text-red-700 dark:text-red-300">
              {error}
            </div>
          </div>
        )}
        {dashboardData && <IntelligenceDashboard data={dashboardData} />}
      </div>
    );
  }

  // ─── Main layout ───────────────────────────────────────────────────────────

  return (
    <div className="flex h-screen bg-gray-50 dark:bg-gray-950">
      {/* Sidebar */}
      <Sidebar
        state={sidebar}
        onChange={setSidebar}
        onRun={handleRun}
        onClear={handleClear}
        loading={loading}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
      />

      {/* Main content */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-6xl mx-auto px-6 py-8 space-y-6">
          {/* Header */}
          <header>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
              InsightAgent
            </h1>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              Upload data, analyze, and explore insights in one place
            </p>
          </header>

          {/* Error banner */}
          {error && (
            <div className="rounded-xl bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 p-4 flex items-start gap-3">
              <svg
                className="w-5 h-5 text-red-500 shrink-0 mt-0.5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z"
                />
              </svg>
              <div className="flex-1">
                <p className="text-sm text-red-700 dark:text-red-300">
                  {error}
                </p>
              </div>
              <button
                onClick={() => setError(null)}
                className="text-red-400 hover:text-red-600 transition-colors"
              >
                <svg
                  className="w-4 h-4"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </div>
          )}

          {/* CSV Upload */}
          <section className="bg-white dark:bg-gray-900 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-800 p-6">
            <CsvUpload file={file} onFileChange={setFile} />
          </section>

          {/* Prompt Input */}
          <section className="bg-white dark:bg-gray-900 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-800 p-6">
            <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-3">
              Strategic Prompt
            </h3>
            <div className="flex gap-3">
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Enter your strategic business question..."
                rows={3}
                className="flex-1 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 px-4 py-3 text-sm text-gray-900 dark:text-gray-100 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
                    e.preventDefault();
                    handleRun();
                  }
                }}
              />
              <button
                onClick={handleRun}
                disabled={loading}
                className="self-end rounded-xl bg-blue-600 px-6 py-3 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shrink-0"
              >
                {loading ? "Analyzing..." : "Analyze"}
              </button>
            </div>
            <p className="mt-2 text-xs text-gray-400">Ctrl+Enter to run</p>
          </section>

          {/* Loading skeleton */}
          {loading && (
            <div className="space-y-4">
              {[1, 2, 3].map((i) => (
                <div
                  key={i}
                  className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-6 animate-pulse"
                >
                  <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/3 mb-4" />
                  <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-full mb-2" />
                  <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-2/3" />
                </div>
              ))}
            </div>
          )}

          {/* Unified Insights Dashboard (auto-scrolls into view) */}
          {analyzeResult && !loading && (
            <InsightsDashboard
              analyzeResult={analyzeResult}
              dashboardData={dashboardData}
              kpiRows={kpiRows}
              derivedSignals={derivedSignals}
              executionTime={executionTime}
              entityName={effectiveEntity || undefined}
              reportInsight={reportInsight || undefined}
              biData={biResult}
            />
          )}

          {/* Export Panel */}
          {analyzeResult && !loading && (
            <ExportPanel
              result={analyzeResult}
              entityName={effectiveEntity || undefined}
              prompt={prompt.trim() || undefined}
              businessType={effectiveBusinessType}
            />
          )}

          {/* Empty state */}
          {!loading && !analyzeResult && !error && (
            <div className="flex flex-col items-center justify-center py-24 text-center">
              <svg
                className="w-16 h-16 text-gray-300 dark:text-gray-700 mb-4"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={1}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                />
              </svg>
              <p className="text-lg text-gray-400 dark:text-gray-500">
                Upload data and enter a prompt to begin analysis
              </p>
              <p className="text-sm text-gray-400 dark:text-gray-600 mt-1">
                Or set an entity name in the sidebar and click "Run Analysis"
              </p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
