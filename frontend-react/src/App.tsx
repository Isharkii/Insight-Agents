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
  fetchReportPayload,
  fetchExportJson,
  type AnalyzeResult,
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
  const [reportPayload, setReportPayload] = useState<Record<
    string,
    unknown
  > | null>(null);
  const [kpiExportJson, setKpiExportJson] = useState<Record<
    string,
    unknown
  > | null>(null);
  const [executionTime, setExecutionTime] = useState<number | undefined>();

  // UI state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Derived
  const resolvedEntity = useMemo(() => {
    const override = sidebar.entityOverride.trim();
    if (override) return override;
    return sidebar.clientId !== "default" ? sidebar.clientId : "";
  }, [sidebar.entityOverride, sidebar.clientId]);

  const resolvedBusinessType = useMemo(
    () => (sidebar.businessType === "auto" ? undefined : sidebar.businessType),
    [sidebar.businessType],
  );

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
    if (!file && !resolvedEntity) {
      setError("Entity / Client ID is required when no CSV is uploaded.");
      return;
    }

    setLoading(true);
    setError(null);
    setAnalyzeResult(null);
    setDashboardData(null);
    setReportPayload(null);
    setKpiExportJson(null);

    const started = performance.now();
    try {
      const result = await runAnalysis({
        prompt: prompt.trim(),
        file: file ?? undefined,
        clientId: resolvedEntity || undefined,
        businessType: resolvedBusinessType,
        multiEntityBehavior:
          sidebar.multiEntityBehavior === "auto"
            ? undefined
            : sidebar.multiEntityBehavior,
        model: sidebar.model,
      });
      setAnalyzeResult(result);
      setExecutionTime((performance.now() - started) / 1000);

      // Fetch dashboard + report + KPI export in parallel
      const entity = resolvedEntity;
      const btype = resolvedBusinessType || "saas";

      if (entity) {
        await Promise.all([
          fetchDashboard(entity, btype)
            .then(setDashboardData)
            .catch(() => {}),
          fetchReportPayload(entity, prompt.trim(), resolvedBusinessType)
            .then(setReportPayload)
            .catch(() => {}),
          fetchExportJson("kpis", entity)
            .then(setKpiExportJson)
            .catch(() => {}),
        ]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [prompt, file, resolvedEntity, resolvedBusinessType, sidebar]);

  const handleClear = useCallback(() => {
    setAnalyzeResult(null);
    setDashboardData(null);
    setReportPayload(null);
    setKpiExportJson(null);
    setError(null);
    setExecutionTime(undefined);
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
              entityName={resolvedEntity || undefined}
            />
          )}

          {/* Export Panel */}
          {analyzeResult && !loading && (
            <ExportPanel
              result={analyzeResult}
              entityName={resolvedEntity || undefined}
              prompt={prompt.trim() || undefined}
              businessType={resolvedBusinessType}
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
