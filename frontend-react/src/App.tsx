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

function useEmbedParams() {
  const params = new URLSearchParams(window.location.search);
  const embed = params.get("embed") === "1";
  const entity = params.get("entity_name") ?? "";
  const btype = params.get("business_type") ?? "saas";
  return { embed, entity, btype };
}

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

export default function App() {
  const { embed, entity: embedEntity, btype: embedBtype } = useEmbedParams();

  const [sidebar, setSidebar] = useState<SidebarState>({
    mode: "LOCAL",
    clientId: "default",
    entityOverride: embedEntity,
    businessType: embedBtype || "auto",
    multiEntityBehavior: "auto",
    model: "default",
  });
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  const [prompt, setPrompt] = useState("");
  const [file, setFile] = useState<File | null>(null);

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

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
      const biPrompt = [
        prompt.trim(),
        requestedEntity ? `Entity: ${requestedEntity}.` : "",
        requestedBusinessType ? `Business type: ${requestedBusinessType}.` : "",
        "Prioritize macro environment, risk zones, and strategy recommendations from available internal and external signals.",
      ].filter(Boolean).join("\n");

      const biPromise = runBusinessIntelligence({
        businessPrompt: biPrompt,
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
        selfAnalysisOnly: true,
        model: sidebar.model,
      });
      const result = run.result;
      setAnalyzeResult(result);
      setExecutionTime((performance.now() - started) / 1000);

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
          fetchReportPayload(entity, prompt.trim(), btype, undefined, true)
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
      <div className="ia-shell-bg">
        {loading && (
          <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6">
            <div className="ia-surface p-8 text-center">
              <p className="ia-subtitle">Loading dashboard...</p>
            </div>
          </div>
        )}
        {error && (
          <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6">
            <div className="ia-surface border-red-200 bg-red-50/90 p-4 text-sm text-red-700">
              {error}
            </div>
          </div>
        )}
        {dashboardData && (
          <div className="relative z-10">
            <IntelligenceDashboard data={dashboardData} />
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="ia-shell-bg flex min-h-screen">
      <Sidebar
        state={sidebar}
        onChange={setSidebar}
        onRun={handleRun}
        onClear={handleClear}
        loading={loading}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
      />

      <main className="relative z-10 flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-[1240px] space-y-5 px-4 pb-10 pt-7 sm:px-6 lg:px-8">
          <header className="ia-surface ia-fade-up px-6 py-5 sm:px-7 sm:py-6">
            <div className="flex flex-wrap items-end justify-between gap-3">
              <div>
                <p className="ia-label mb-1">Decision Intelligence Platform</p>
                <h1 className="ia-title">InsightAgent Studio</h1>
                <p className="ia-subtitle mt-1.5">
                  Analyze historical data, generate intelligence, and export structured outputs.
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <span className="ia-chip">{loading ? "Running" : "Ready"}</span>
                {effectiveEntity && <span className="ia-chip">Entity: {effectiveEntity}</span>}
                {effectiveBusinessType && (
                  <span className="ia-chip">
                    Type: {effectiveBusinessType.replace(/_/g, " ")}
                  </span>
                )}
              </div>
            </div>
          </header>

          {error && (
            <div className="ia-surface border-red-200 bg-red-50/90 px-5 py-4">
              <div className="flex items-start gap-3">
                <svg
                  className="mt-0.5 h-5 w-5 shrink-0 text-red-500"
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
                <p className="flex-1 text-sm text-red-700">{error}</p>
                <button
                  onClick={() => setError(null)}
                  className="rounded-md p-1 text-red-400 transition-colors hover:bg-red-100 hover:text-red-600"
                  aria-label="Dismiss error"
                >
                  <svg
                    className="h-4 w-4"
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
            </div>
          )}

          <section className="ia-surface ia-fade-up p-5 sm:p-6">
            <div className="mb-4">
              <p className="ia-label">Data Input</p>
              <p className="ia-subtitle mt-1">
                Upload a CSV dataset to run analysis from raw records.
              </p>
            </div>
            <CsvUpload file={file} onFileChange={setFile} />
          </section>

          <section className="ia-surface ia-fade-up p-5 sm:p-6">
            <div className="mb-4 flex items-start justify-between gap-4">
              <div>
                <p className="ia-label">Strategic Prompt</p>
                <p className="ia-subtitle mt-1">
                  Define the business question to steer synthesis and recommendations.
                </p>
              </div>
              <span className="ia-kbd">Ctrl + Enter</span>
            </div>

            <div className="flex flex-col gap-3">
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Enter your strategic business question..."
                rows={4}
                className="ia-textarea min-h-[120px] flex-1"
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
                    e.preventDefault();
                    handleRun();
                  }
                }}
              />

              <div className="flex justify-end">
                <button
                  onClick={handleRun}
                  disabled={loading}
                  className="ia-btn-primary h-[46px] px-6"
                >
                  {loading ? (
                    <>
                      <svg
                        className="h-4 w-4 animate-spin"
                        viewBox="0 0 24 24"
                        fill="none"
                      >
                        <circle
                          className="opacity-25"
                          cx="12"
                          cy="12"
                          r="10"
                          stroke="currentColor"
                          strokeWidth="4"
                        />
                        <path
                          className="opacity-75"
                          fill="currentColor"
                          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                        />
                      </svg>
                      Analyzing
                    </>
                  ) : (
                    "Analyze"
                  )}
                </button>
              </div>
            </div>
          </section>

          {loading && (
            <div className="space-y-4">
              {[1, 2, 3].map((i) => (
                <div key={i} className="ia-surface p-6">
                  <div className="mb-4 h-4 w-1/3 animate-pulse rounded bg-slate-200" />
                  <div className="mb-2 h-3 w-full animate-pulse rounded bg-slate-200" />
                  <div className="h-3 w-2/3 animate-pulse rounded bg-slate-200" />
                </div>
              ))}
            </div>
          )}

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

          {analyzeResult && !loading && (
            <ExportPanel
              result={analyzeResult}
              entityName={effectiveEntity || undefined}
              prompt={prompt.trim() || undefined}
              businessType={effectiveBusinessType}
            />
          )}

          {!loading && !analyzeResult && !error && (
            <div className="ia-surface px-6 py-16 text-center sm:px-10">
              <svg
                className="mx-auto mb-4 h-14 w-14 text-slate-300"
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
              <p className="text-base font-medium text-slate-600">
                Upload data and enter a prompt to start analysis.
              </p>
              <p className="mt-2 text-sm text-slate-500">
                You can also set an entity name in the sidebar and run without CSV upload.
              </p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
