import { useRef, useEffect, type FC, useMemo } from "react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  RadialBarChart,
  RadialBar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import type { AnalyzeResult, BusinessIntelligenceResponse } from "../api/client";
import type { DashboardData } from "./IntelligenceDashboard/types";

// ─── Types ───────────────────────────────────────────────────────────────────

interface KpiRow {
  period_end: string;
  metric_name: string;
  metric_value: number;
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

interface InsightsDashboardProps {
  analyzeResult: AnalyzeResult;
  dashboardData: DashboardData | null;
  kpiRows: KpiRow[];
  derivedSignals: ReportDerivedSignals;
  executionTime?: number;
  entityName?: string;
  reportInsight?: AnalyzeResult;
  biData?: BusinessIntelligenceResponse | null;
}

// ─── Colors ──────────────────────────────────────────────────────────────────

const CHART_COLORS = [
  "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6",
  "#ef4444", "#06b6d4", "#f97316", "#ec4899",
];

const PIE_COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444", "#06b6d4"];

// ─── Helpers ─────────────────────────────────────────────────────────────────

function formatMetricValue(value: number, unit?: string): string {
  if (unit === "%" || unit === "percent") return `${value.toFixed(1)}%`;
  if (value >= 1_000_000) return `$${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `$${(value / 1_000).toFixed(1)}K`;
  if (Number.isInteger(value)) return value.toLocaleString();
  return value.toFixed(2);
}

function priorityColor(p: string): string {
  if (p === "critical") return "text-red-500";
  if (p === "high") return "text-orange-500";
  if (p === "medium") return "text-amber-500";
  return "text-gray-500";
}

function priorityBg(p: string): string {
  if (p === "critical") return "bg-red-500";
  if (p === "high") return "bg-orange-500";
  if (p === "medium") return "bg-amber-500";
  return "bg-gray-400";
}

function healthColor(score: number): string {
  if (score >= 80) return "#10b981";
  if (score >= 60) return "#f59e0b";
  return "#ef4444";
}

function riskLevelLabel(score: number): string {
  if (score <= 30) return "Low Risk";
  if (score <= 60) return "Moderate";
  if (score <= 80) return "High Risk";
  return "Critical";
}

function biStatusClass(status: string): string {
  if (status === "success") {
    return "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300";
  }
  if (status === "partial") {
    return "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300";
  }
  return "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300";
}

function stageStatusClass(status: string): string {
  if (status === "success") return "bg-emerald-500";
  if (status === "skipped") return "bg-gray-400";
  return "bg-red-500";
}

function signalIdList(ids: string[]): string {
  if (ids.length === 0) return "No signals";
  return ids.join(", ");
}

type StrategyStep = { title: string; detail: string };

function buildStrategies(insight: AnalyzeResult | null): StrategyStep[] {
  if (!insight) return [];
  return [
    {
      title: "Execute the recommended action",
      detail: `Implement: ${insight.recommended_action}`,
    },
    {
      title: "Validate the drivers cited in evidence",
      detail: `Confirm the drivers described: ${insight.evidence}`,
    },
    {
      title: "Monitor the stated impact",
      detail: `Track the outcome area: ${insight.impact}`,
    },
    {
      title: "Review priority and confidence",
      detail: `Align owners/SLAs to ${insight.priority} priority and re-score monthly (confidence ${(insight.confidence_score ?? 0).toFixed(2)})`,
    },
  ];
}

// ─── Sub-components ──────────────────────────────────────────────────────────

const KpiMetricCard: FC<{
  name: string;
  value: number;
  unit?: string;
  label?: string;
  index: number;
}> = ({ name, value, unit, label, index }) => (
  <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-5 relative overflow-hidden">
    <div
      className="absolute top-0 left-0 w-1 h-full rounded-l-2xl"
      style={{ backgroundColor: CHART_COLORS[index % CHART_COLORS.length] }}
    />
    <p className="text-xs font-medium uppercase tracking-wider text-gray-400 mb-1 pl-3">
      {label || name.replace(/_/g, " ")}
    </p>
    <p className="text-2xl font-bold text-gray-900 dark:text-gray-100 pl-3 tabular-nums">
      {formatMetricValue(value, unit)}
    </p>
  </div>
);

const ConfidenceGauge: FC<{ score: number }> = ({ score }) => {
  const pct = Math.round(score * 100);
  const data = [{ name: "confidence", value: pct, fill: healthColor(pct) }];
  return (
    <div className="flex flex-col items-center">
      <div className="w-32 h-32">
        <ResponsiveContainer width="100%" height="100%">
          <RadialBarChart
            cx="50%"
            cy="50%"
            innerRadius="70%"
            outerRadius="100%"
            startAngle={180}
            endAngle={0}
            data={data}
            barSize={10}
          >
            <RadialBar dataKey="value" cornerRadius={5} background={{ fill: "#e5e7eb" }} />
          </RadialBarChart>
        </ResponsiveContainer>
      </div>
      <p className="text-2xl font-bold tabular-nums -mt-8" style={{ color: healthColor(pct) }}>
        {pct}%
      </p>
      <p className="text-xs text-gray-400 mt-1">Confidence</p>
    </div>
  );
};

const HealthDonut: FC<{ score: number; label: string }> = ({ score, label }) => {
  const clamped = Math.max(0, Math.min(100, score));
  const data = [
    { name: "health", value: clamped },
    { name: "remaining", value: 100 - clamped },
  ];
  const color = healthColor(clamped);
  return (
    <div className="flex flex-col items-center">
      <div className="w-36 h-36 relative">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={48}
              outerRadius={62}
              startAngle={90}
              endAngle={-270}
              dataKey="value"
              strokeWidth={0}
            >
              <Cell fill={color} />
              <Cell fill="#e5e7eb" />
            </Pie>
          </PieChart>
        </ResponsiveContainer>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-3xl font-bold tabular-nums" style={{ color }}>
            {clamped}
          </span>
        </div>
      </div>
      <p className="text-sm font-semibold mt-1" style={{ color }}>{label}</p>
      <p className="text-xs text-gray-400">Health Index</p>
    </div>
  );
};

const RiskDonut: FC<{ score: number; label?: string }> = ({ score, label }) => {
  const clamped = Math.max(0, Math.min(100, score));
  const color = clamped <= 30 ? "#10b981" : clamped <= 60 ? "#f59e0b" : clamped <= 80 ? "#f97316" : "#ef4444";
  const data = [
    { name: "risk", value: clamped },
    { name: "safe", value: 100 - clamped },
  ];
  return (
    <div className="flex flex-col items-center">
      <div className="w-36 h-36 relative">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={48}
              outerRadius={62}
              startAngle={90}
              endAngle={-270}
              dataKey="value"
              strokeWidth={0}
            >
              <Cell fill={color} />
              <Cell fill="#e5e7eb" />
            </Pie>
          </PieChart>
        </ResponsiveContainer>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-3xl font-bold tabular-nums" style={{ color }}>
            {clamped}
          </span>
        </div>
      </div>
      <p className="text-sm font-semibold mt-1" style={{ color }}>
        {label || riskLevelLabel(clamped)}
      </p>
      <p className="text-xs text-gray-400">Risk Score</p>
    </div>
  );
};

// ─── Main Dashboard ──────────────────────────────────────────────────────────

const InsightsDashboard: FC<InsightsDashboardProps> = ({
  analyzeResult,
  dashboardData,
  kpiRows,
  derivedSignals,
  executionTime,
  entityName,
  reportInsight,
  biData,
}) => {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    ref.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [analyzeResult]);

  const pct = Math.round(analyzeResult.confidence_score * 100);
  const insightSource = reportInsight ?? analyzeResult;
  const strategies = useMemo(
    () => buildStrategies(insightSource ?? null),
    [insightSource],
  );

  // KPI metrics from dashboard
  const kpiEntries = useMemo(() => {
    if (!dashboardData?.kpi_metrics) return [];
    return Object.entries(dashboardData.kpi_metrics).map(([name, entry]) => ({
      name,
      value: entry.value,
      unit: entry.unit,
      label: entry.label,
    }));
  }, [dashboardData]);

  // Time-series chart data from kpiRows
  const { timeSeriesData, metricNames } = useMemo(() => {
    if (!kpiRows || kpiRows.length === 0) return { timeSeriesData: [], metricNames: [] };
    const grouped: Record<string, Record<string, number>> = {};
    const nameSet = new Set<string>();
    for (const row of kpiRows) {
      const period = row.period_end?.slice(0, 7) ?? "";
      if (!period) continue;
      if (!grouped[period]) grouped[period] = {};
      grouped[period][row.metric_name] = row.metric_value;
      nameSet.add(row.metric_name);
    }
    const sorted = Object.keys(grouped).sort();
    return {
      timeSeriesData: sorted.map((p) => ({ period: p, ...grouped[p] })),
      metricNames: Array.from(nameSet).sort(),
    };
  }, [kpiRows]);

  // Revenue trend from dashboard
  const revenueTrend = dashboardData?.revenue_trend ?? [];

  const chartEntityName = entityName || dashboardData?.entity_name || "selected entity";

  // Forecast chart data uses both historical revenue and projected points.
  const forecastSeries = useMemo(() => {
    const history = dashboardData?.revenue_trend ?? [];
    const projections = dashboardData?.market_share ?? [];

    if (history.length === 0 && projections.length === 0) {
      return { data: [], hasActual: false, hasProjection: false };
    }

    const byPeriod = new Map<string, { period: string; actual?: number; projected?: number }>();

    for (const point of history) {
      byPeriod.set(point.period, { period: point.period, actual: point.value });
    }

    for (const point of projections) {
      const existing = byPeriod.get(point.period) ?? { period: point.period };
      if (point.projected) {
        existing.projected = point.value;
      } else {
        existing.actual = point.value;
      }
      byPeriod.set(point.period, existing);
    }

    const data = Array.from(byPeriod.values()).sort((a, b) => a.period.localeCompare(b.period));
    const firstProjectedIdx = data.findIndex((point) => typeof point.projected === "number");
    if (
      firstProjectedIdx > 0 &&
      typeof data[firstProjectedIdx].actual !== "number" &&
      typeof data[firstProjectedIdx - 1].actual === "number"
    ) {
      data[firstProjectedIdx] = {
        ...data[firstProjectedIdx],
        actual: data[firstProjectedIdx - 1].actual,
      };
    }

    return {
      data,
      hasActual: data.some((point) => typeof point.actual === "number"),
      hasProjection: data.some((point) => typeof point.projected === "number"),
    };
  }, [dashboardData]);

  // KPI distribution for pie chart
  const kpiPieData = useMemo(() => {
    if (kpiEntries.length === 0) return [];
    const positives = kpiEntries.filter((k) => k.value > 0).slice(0, 6);
    return positives.map((k) => ({
      name: k.label || k.name.replace(/_/g, " "),
      value: Math.abs(k.value),
    }));
  }, [kpiEntries]);

  // Role contribution
  const roleContributors = derivedSignals.role_contribution?.top_contributors ?? [];

  // Scenario comparison
  const scenarios = derivedSignals.multivariate_scenario?.scenario_simulation?.scenarios;
  const scenarioData = useMemo(() => {
    if (!scenarios) return [];
    return Object.entries(scenarios).map(([name, vals]) => ({
      scenario: name,
      projected_value: vals.projected_value ?? 0,
      projected_growth: vals.projected_growth ?? 0,
    }));
  }, [scenarios]);

  // Insights from dashboard
  const structuredInsights = dashboardData?.insights ?? [];
  const biContext = biData?.context ?? null;
  const biInsights = biData?.insights ?? null;
  const biStrategy = biData?.strategy ?? null;
  const biPipeline = biData?.pipeline ?? [];
  const biWarnings = biData?.warnings ?? [];
  const benchmark = derivedSignals.competitive_benchmark;
  const benchmarkRanking = benchmark?.ranking;
  const benchmarkComposite = benchmark?.composite;
  const benchmarkMetrics = benchmarkComposite?.competitive_metrics;
  const benchmarkReason = benchmark?.reason
    ? benchmark.reason.replace(/_/g, " ")
    : "";

  return (
    <div ref={ref} className="space-y-6">
      {/* ── Section header ── */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">
            Intelligence Dashboard
          </h2>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-0.5">
            {entityName && <span>{entityName}</span>}
            {dashboardData?.business_type && (
              <>
                <span className="mx-2 text-gray-300 dark:text-gray-700">|</span>
                {dashboardData.business_type.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
              </>
            )}
            {executionTime !== undefined && (
              <>
                <span className="mx-2 text-gray-300 dark:text-gray-700">|</span>
                {executionTime.toFixed(1)}s
              </>
            )}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold capitalize ${
              analyzeResult.pipeline_status === "success"
                ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300"
                : analyzeResult.pipeline_status === "partial"
                  ? "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300"
                  : "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300"
            }`}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                analyzeResult.pipeline_status === "success"
                  ? "bg-emerald-500"
                  : analyzeResult.pipeline_status === "partial"
                    ? "bg-amber-500"
                    : "bg-red-500"
              }`}
            />
            {analyzeResult.pipeline_status}
          </span>
        </div>
      </div>

      {/* ── Row 1: Insight summary + gauges ── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Main insight card */}
        <div className="lg:col-span-7 bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-6 space-y-4">
          <div className="flex items-start gap-3">
            <div className={`w-1 shrink-0 self-stretch rounded-full ${priorityBg(analyzeResult.priority)}`} />
            <div className="space-y-3 flex-1">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span className={`text-xs font-bold uppercase tracking-wider ${priorityColor(analyzeResult.priority)}`}>
                    {analyzeResult.priority} priority
                  </span>
                </div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 leading-snug">
                  {analyzeResult.insight}
                </h3>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                {analyzeResult.evidence}
              </p>
              <div className="bg-blue-50 dark:bg-blue-900/10 border border-blue-200 dark:border-blue-800 rounded-lg p-3">
                <p className="text-xs font-semibold uppercase tracking-wider text-blue-500 mb-0.5">
                  Recommended Action
                </p>
                <p className="text-sm text-blue-900 dark:text-blue-200">
                  {analyzeResult.recommended_action}
                </p>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                <span className="font-medium text-gray-500">Impact:</span> {analyzeResult.impact}
              </p>
            </div>
          </div>
        </div>

        {/* Gauges column */}
        <div className="lg:col-span-5 grid grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-4 flex items-center justify-center">
            <ConfidenceGauge score={analyzeResult.confidence_score} />
          </div>
          {dashboardData ? (
            <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-4 flex items-center justify-center">
              <HealthDonut
                score={dashboardData.health_index}
                label={dashboardData.health_label}
              />
            </div>
          ) : derivedSignals.risk?.risk_score != null ? (
            <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-4 flex items-center justify-center">
              <RiskDonut
                score={derivedSignals.risk.risk_score}
                label={derivedSignals.risk.risk_level}
              />
            </div>
          ) : (
            <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-4 flex flex-col items-center justify-center">
              <p className="text-4xl font-bold tabular-nums text-gray-900 dark:text-gray-100">
                {pct}%
              </p>
              <p className="text-xs text-gray-400 mt-1">Pipeline Score</p>
            </div>
          )}
          {derivedSignals.risk?.risk_score != null && dashboardData && (
            <>
              <div className="col-span-2 bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-4 flex items-center justify-center">
                <RiskDonut
                  score={derivedSignals.risk.risk_score}
                  label={derivedSignals.risk.risk_level}
                />
              </div>
            </>
          )}
        </div>
      </div>

      {/* ── Strategies ── */}
      {biData && (
        <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-6 space-y-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h3 className="text-sm font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
                Business Intelligence Pipeline
              </h3>
              <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
                Generated {new Date(biData.generated_at).toLocaleString()}
              </p>
            </div>
            <div className="flex items-center gap-2">
              <span className={`px-3 py-1 rounded-full text-xs font-semibold capitalize ${biStatusClass(biData.status)}`}>
                {biData.status}
              </span>
              <span className="text-xs text-gray-400">
                Confidence {Math.round(biData.confidence * 100)}%
              </span>
            </div>
          </div>

          {biPipeline.length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
              {biPipeline.map((stage) => (
                <div key={stage.stage} className="rounded-xl border border-gray-200 dark:border-gray-700 p-3">
                  <div className="flex items-center justify-between gap-2">
                    <p className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
                      {stage.stage.replace(/_/g, " ")}
                    </p>
                    <span className={`w-2 h-2 rounded-full ${stageStatusClass(stage.status)}`} />
                  </div>
                  <p className="text-sm text-gray-700 dark:text-gray-200 capitalize mt-1">
                    {stage.status}
                  </p>
                  <p className="text-xs text-gray-400 mt-1">
                    {stage.duration_ms.toFixed(1)} ms
                  </p>
                  {stage.error && (
                    <p className="text-xs text-red-500 mt-1 line-clamp-2">
                      {stage.error}
                    </p>
                  )}
                </div>
              ))}
            </div>
          )}

          {biWarnings.length > 0 && (
            <div className="rounded-xl border border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-900/10 p-3">
              <p className="text-xs font-semibold uppercase tracking-wider text-amber-700 dark:text-amber-300 mb-2">
                BI Warnings
              </p>
              <ul className="space-y-1 text-xs text-amber-800 dark:text-amber-200">
                {biWarnings.slice(0, 5).map((item, idx) => (
                  <li key={idx}>• {item}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {(biContext || biInsights || biStrategy) && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {biContext && (
            <div className="lg:col-span-4 bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-6 space-y-4">
              <h3 className="text-sm font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
                BI Context
              </h3>
              <div className="space-y-2 text-sm">
                <p className="text-gray-700 dark:text-gray-200"><span className="text-gray-500">Industry:</span> {biContext.industry}</p>
                <p className="text-gray-700 dark:text-gray-200"><span className="text-gray-500">Business Model:</span> {biContext.business_model}</p>
                <p className="text-gray-700 dark:text-gray-200"><span className="text-gray-500">Target Market:</span> {biContext.target_market}</p>
              </div>
              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-2">Macro Dependencies</p>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-200">
                  {biContext.macro_dependencies.slice(0, 5).map((item, idx) => (
                    <li key={idx}>• {item}</li>
                  ))}
                </ul>
              </div>
              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-2">Risk Factors</p>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-200">
                  {biContext.risk_factors.slice(0, 5).map((item, idx) => (
                    <li key={idx}>• {item}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {biInsights && (
            <div className="lg:col-span-4 bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-6 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
                  BI Insights
                </h3>
                <span className="text-xs text-gray-400">
                  Momentum {(biInsights.momentum_score * 100).toFixed(0)}%
                </span>
              </div>
              <p className="text-sm text-gray-700 dark:text-gray-200">
                {biInsights.macro_summary}
              </p>
              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-2">Emerging Signals</p>
                <div className="space-y-2">
                  {biInsights.emerging_signals.slice(0, 3).map((signal, idx) => (
                    <div key={idx} className="rounded-xl border border-gray-200 dark:border-gray-700 p-3">
                      <p className="text-sm font-semibold text-gray-900 dark:text-gray-100">{signal.title}</p>
                      <p className="text-xs text-gray-600 dark:text-gray-300 mt-1">{signal.description}</p>
                      <p className="text-[11px] text-gray-400 mt-1">
                        Signals: {signalIdList(signal.supporting_signals.map((s) => s.signal_id))}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {biStrategy && (
            <div className="lg:col-span-4 bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-6 space-y-4">
              <h3 className="text-sm font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
                BI Strategy
              </h3>
              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-2">Short-Term Actions</p>
                <ol className="space-y-2 text-sm text-gray-700 dark:text-gray-200 list-decimal list-inside">
                  {biStrategy.short_term_actions.slice(0, 3).map((action, idx) => (
                    <li key={idx}>{action.action}</li>
                  ))}
                </ol>
              </div>
              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-2">Mid-Term Actions</p>
                <ol className="space-y-2 text-sm text-gray-700 dark:text-gray-200 list-decimal list-inside">
                  {biStrategy.mid_term_actions.slice(0, 3).map((action, idx) => (
                    <li key={idx}>{action.action}</li>
                  ))}
                </ol>
              </div>
              <div className="rounded-xl border border-gray-200 dark:border-gray-700 p-3">
                <p className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-1">Long-Term Positioning</p>
                <p className="text-sm text-gray-700 dark:text-gray-200">{biStrategy.long_term_positioning}</p>
              </div>
            </div>
          )}
        </div>
      )}

      {benchmark && (
        <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-6 space-y-4">
          <div className="flex items-center justify-between gap-3">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
              Competitive Benchmark Engine
            </h3>
            <div className="flex items-center gap-2 text-xs">
              <span className={`px-2.5 py-1 rounded-full font-semibold capitalize ${benchmark?.status === "success" ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300" : "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300"}`}>
                {benchmark?.status || "partial"}
              </span>
              {benchmarkRanking?.tier && (
                <span className="px-2.5 py-1 rounded-full font-semibold bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300">
                  {benchmarkRanking.tier}
                </span>
              )}
            </div>
          </div>
          {benchmarkReason && (
            <p className="text-xs text-amber-700 dark:text-amber-300">
              {benchmarkReason}
            </p>
          )}

          <div className="grid grid-cols-2 lg:grid-cols-6 gap-3">
            <div className="rounded-xl border border-gray-200 dark:border-gray-700 p-3">
              <p className="text-xs text-gray-400 uppercase tracking-wider">Overall</p>
              <p className="text-xl font-bold tabular-nums text-gray-900 dark:text-gray-100">
                {benchmarkComposite?.overall_score?.toFixed?.(2) ?? "N/A"}
              </p>
            </div>
            <div className="rounded-xl border border-gray-200 dark:border-gray-700 p-3">
              <p className="text-xs text-gray-400 uppercase tracking-wider">Growth</p>
              <p className="text-xl font-bold tabular-nums text-gray-900 dark:text-gray-100">
                {benchmarkComposite?.growth_score?.toFixed?.(2) ?? "N/A"}
              </p>
            </div>
            <div className="rounded-xl border border-gray-200 dark:border-gray-700 p-3">
              <p className="text-xs text-gray-400 uppercase tracking-wider">Level</p>
              <p className="text-xl font-bold tabular-nums text-gray-900 dark:text-gray-100">
                {benchmarkComposite?.level_score?.toFixed?.(2) ?? "N/A"}
              </p>
            </div>
            <div className="rounded-xl border border-gray-200 dark:border-gray-700 p-3">
              <p className="text-xs text-gray-400 uppercase tracking-wider">Stability</p>
              <p className="text-xl font-bold tabular-nums text-gray-900 dark:text-gray-100">
                {benchmarkComposite?.stability_score?.toFixed?.(2) ?? "N/A"}
              </p>
            </div>
            <div className="rounded-xl border border-gray-200 dark:border-gray-700 p-3">
              <p className="text-xs text-gray-400 uppercase tracking-wider">Confidence</p>
              <p className="text-xl font-bold tabular-nums text-gray-900 dark:text-gray-100">
                {benchmarkComposite?.confidence_score?.toFixed?.(2) ?? "N/A"}
              </p>
            </div>
            <div className="rounded-xl border border-gray-200 dark:border-gray-700 p-3">
              <p className="text-xs text-gray-400 uppercase tracking-wider">Rank</p>
              <p className="text-xl font-bold tabular-nums text-gray-900 dark:text-gray-100">
                {benchmarkRanking?.overall_rank ?? "N/A"}
                {benchmarkRanking?.total_participants ? `/${benchmarkRanking.total_participants}` : ""}
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="rounded-xl border border-gray-200 dark:border-gray-700 p-4">
              <p className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-2">
                Competitive Metrics
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-sm">
                <p className="text-gray-700 dark:text-gray-200">
                  Relative Growth Index: <span className="font-semibold">{benchmarkMetrics?.relative_growth_index ?? "N/A"}</span>
                </p>
                <p className="text-gray-700 dark:text-gray-200">
                  Market Share Proxy: <span className="font-semibold">{benchmarkMetrics?.market_share_proxy ?? "N/A"}</span>
                </p>
                <p className="text-gray-700 dark:text-gray-200">
                  Stability Score: <span className="font-semibold">{benchmarkMetrics?.stability_score ?? "N/A"}</span>
                </p>
                <p className="text-gray-700 dark:text-gray-200">
                  Momentum: <span className="font-semibold">{benchmarkMetrics?.momentum_classification ?? "N/A"}</span>
                </p>
                <p className="text-gray-700 dark:text-gray-200">
                  Risk Divergence: <span className="font-semibold">{benchmarkMetrics?.risk_divergence_score ?? "N/A"}</span>
                </p>
                <p className="text-gray-700 dark:text-gray-200">
                  Percentile: <span className="font-semibold">{benchmarkRanking?.overall_percentile ?? "N/A"}</span>
                </p>
              </div>
            </div>

            <div className="rounded-xl border border-gray-200 dark:border-gray-700 p-4">
              <p className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-2">
                Peer Sourcing
              </p>
              <p className="text-sm text-gray-700 dark:text-gray-200">
                Candidate peers: {benchmark?.peer_selection?.peer_candidates?.length ?? 0}
              </p>
              <p className="text-sm text-gray-700 dark:text-gray-200">
                Selected peers: {benchmark?.peer_selection?.selected_peers?.length ?? 0}
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
                Same category + aligned trailing window, client excluded.
              </p>
              {(benchmark?.peer_selection?.selected_peers?.length ?? 0) > 0 && (
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-2 line-clamp-3">
                  {benchmark?.peer_selection?.selected_peers?.join(", ")}
                </p>
              )}
            </div>
          </div>

          {benchmarkRanking?.metric_ranks && Object.keys(benchmarkRanking.metric_ranks).length > 0 && (
            <div className="rounded-xl border border-gray-200 dark:border-gray-700 p-4">
              <p className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-3">
                Metric Ranking Breakdown
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {Object.entries(benchmarkRanking.metric_ranks).map(([metric, rank]) => (
                  <div key={metric} className="rounded-lg border border-gray-200 dark:border-gray-700 p-3">
                    <p className="text-sm font-semibold text-gray-900 dark:text-gray-100">{metric}</p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      Rank {rank.rank ?? "N/A"} | Percentile {rank.percentile ?? "N/A"}
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      Value {rank.client_value ?? "N/A"} | Mean {rank.field_mean ?? "N/A"}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {benchmark?.metric_comparison_specs && Object.keys(benchmark.metric_comparison_specs).length > 0 && (
            <div className="rounded-xl border border-gray-200 dark:border-gray-700 p-4">
              <p className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-3">
                Metric Comparison Specs
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {Object.entries(benchmark.metric_comparison_specs).map(([metric, spec]) => (
                  <div key={metric} className="rounded-lg border border-gray-200 dark:border-gray-700 p-3 text-xs">
                    <p className="font-semibold text-sm text-gray-900 dark:text-gray-100 mb-1">{metric}</p>
                    <p className="text-gray-600 dark:text-gray-300">Direction: {spec.direction || "N/A"}</p>
                    <p className="text-gray-600 dark:text-gray-300">Unit: {spec.unit || "N/A"}</p>
                    <p className="text-gray-600 dark:text-gray-300">Scale: {spec.scale || "N/A"}</p>
                    <p className="text-gray-600 dark:text-gray-300">Aggregation: {spec.aggregation || "N/A"}</p>
                    <p className="text-gray-600 dark:text-gray-300">Window: {spec.window_alignment || "N/A"}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {benchmarkRanking?.skipped_metrics && Object.keys(benchmarkRanking.skipped_metrics).length > 0 && (
            <div className="rounded-xl border border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-900/10 p-4">
              <p className="text-xs font-semibold uppercase tracking-wider text-amber-700 dark:text-amber-300 mb-2">
                Skipped Metrics (Validation)
              </p>
              <div className="space-y-1 text-xs text-amber-800 dark:text-amber-200">
                {Object.entries(benchmarkRanking.skipped_metrics).map(([metric, reason]) => (
                  <p key={metric}>{metric}: {reason}</p>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {insightSource && (
        <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
              Insights Dashboard
            </h3>
            <div className="flex items-center gap-2">
              <span
                className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold capitalize ${
                  insightSource.priority === "critical"
                    ? "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300"
                    : insightSource.priority === "high"
                      ? "bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-300"
                      : insightSource.priority === "medium"
                        ? "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300"
                        : "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"
                }`}
              >
                {insightSource.priority} priority
              </span>
              <span className="text-xs text-gray-400">
                Confidence {Math.round((insightSource.confidence_score ?? 0) * 100)}%
              </span>
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div className="rounded-xl border border-gray-200 dark:border-gray-700 p-4">
              <p className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-2">Insight</p>
              <p className="text-gray-700 dark:text-gray-200">{insightSource.insight}</p>
            </div>
            <div className="rounded-xl border border-gray-200 dark:border-gray-700 p-4">
              <p className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-2">Evidence</p>
              <p className="text-gray-700 dark:text-gray-200">{insightSource.evidence}</p>
            </div>
            <div className="rounded-xl border border-gray-200 dark:border-gray-700 p-4">
              <p className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-2">Impact</p>
              <p className="text-gray-700 dark:text-gray-200">{insightSource.impact}</p>
            </div>
            <div className="rounded-xl border border-gray-200 dark:border-gray-700 p-4">
              <p className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-2">Recommended Action</p>
              <p className="text-gray-700 dark:text-gray-200">{insightSource.recommended_action}</p>
            </div>
          </div>
        </div>
      )}

      {/* ── Strategies ── */}
      {strategies.length > 0 && (
        <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-6">
          <div className="bg-blue-50 dark:bg-blue-900/10 border border-blue-200 dark:border-blue-800 rounded-xl p-4">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-blue-600 dark:text-blue-300 mb-1">
              Strategies
            </h3>
            <p className="text-xs text-blue-900 dark:text-blue-100 mb-3">
              Detailed step-by-step plan for {chartEntityName}.
            </p>
            <ol className="space-y-2 list-decimal list-inside text-sm text-blue-900 dark:text-blue-100">
              {strategies.map((step, idx) => (
                <li key={idx}>
                  <span className="font-semibold">{step.title}:</span> {step.detail}
                </li>
              ))}
            </ol>
          </div>
        </div>
      )}

      {/* ── Row 2: KPI metric cards ── */}
      {kpiEntries.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4">
          {kpiEntries.map((kpi, i) => (
            <KpiMetricCard
              key={kpi.name}
              name={kpi.name}
              value={kpi.value}
              unit={kpi.unit}
              label={kpi.label}
              index={i}
            />
          ))}
        </div>
      )}

      {/* ── Row 3: Revenue Trend (area) + KPI Distribution (pie) ── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {revenueTrend.length > 0 && (
          <div className={`bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-6 ${kpiPieData.length > 0 ? "lg:col-span-8" : "lg:col-span-12"}`}>
            <h3 className="text-sm font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-1">
              Revenue Trend
            </h3>
            <p className="text-xs text-gray-400 dark:text-gray-500 mb-4">
              Entity: {chartEntityName}. X-axis: period (month). Y-axis: revenue (USD).
            </p>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={revenueTrend} margin={{ top: 8, right: 16, left: 8, bottom: 0 }}>
                  <defs>
                    <linearGradient id="revGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
                  <XAxis dataKey="period" tick={{ fontSize: 11 }} />
                  <YAxis
                    tick={{ fontSize: 11 }}
                    tickFormatter={(v: number) =>
                      v >= 1_000_000 ? `$${(v / 1_000_000).toFixed(1)}M` : v >= 1_000 ? `$${(v / 1_000).toFixed(0)}K` : `$${v}`
                    }
                    width={60}
                  />
                  <Tooltip
                    formatter={(v: number) => [`$${v.toLocaleString()}`, "Revenue"]}
                    contentStyle={{ borderRadius: "0.5rem", fontSize: "0.75rem" }}
                  />
                  <Area type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={2.5} fill="url(#revGrad)" dot={{ r: 3, fill: "#3b82f6" }} activeDot={{ r: 5 }} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {kpiPieData.length > 0 && (
          <div className={`bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-6 ${revenueTrend.length > 0 ? "lg:col-span-4" : "lg:col-span-12"}`}>
            <h3 className="text-sm font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-1">
              KPI Distribution
            </h3>
            <p className="text-xs text-gray-400 dark:text-gray-500 mb-4">
              Entity: {chartEntityName}. Slice size: KPI contribution magnitude.
            </p>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={kpiPieData}
                    cx="50%"
                    cy="50%"
                    outerRadius={90}
                    innerRadius={50}
                    dataKey="value"
                    paddingAngle={2}
                    label={({ name }) => name.length > 12 ? name.slice(0, 12) + "..." : name}
                  >
                    {kpiPieData.map((_, i) => (
                      <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(v: number) => [formatMetricValue(v), ""]} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>

      {/* ── Row 4: KPI Time Series (multi-line) ── */}
      {timeSeriesData.length > 0 && (
        <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-6">
          <h3 className="text-sm font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-1">
            KPI Trends Over Time
          </h3>
          <p className="text-xs text-gray-400 dark:text-gray-500 mb-4">
            Entity: {chartEntityName}. X-axis: period (month). Y-axis: KPI metric values.
          </p>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={timeSeriesData} margin={{ top: 8, right: 16, left: 8, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
                <XAxis dataKey="period" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} width={60} />
                <Tooltip contentStyle={{ borderRadius: "0.5rem", fontSize: "0.75rem" }} />
                {metricNames.length > 1 && <Legend wrapperStyle={{ fontSize: "0.7rem" }} iconType="plainline" />}
                {metricNames.map((name, idx) => (
                  <Line
                    key={name}
                    type="monotone"
                    dataKey={name}
                    stroke={CHART_COLORS[idx % CHART_COLORS.length]}
                    strokeWidth={2}
                    dot={{ r: 2 }}
                    activeDot={{ r: 5 }}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* ── Row 5: Forecast Projection + Scenario Comparison ── */}
      {(forecastSeries.data.length > 0 || scenarioData.length > 0) && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {forecastSeries.data.length > 0 && (
            <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-6">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="text-sm font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-1">
                    Forecast Projection
                  </h3>
                  <p className="text-xs text-gray-400 dark:text-gray-500">
                    Entity: {chartEntityName}. X-axis: period (month). Y-axis: revenue (USD).
                  </p>
                </div>
                {(forecastSeries.hasActual || forecastSeries.hasProjection) && (
                  <div className="flex items-center gap-3 text-xs text-gray-400">
                    {forecastSeries.hasActual && (
                      <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-blue-500 inline-block" /> Actual</span>
                    )}
                    {forecastSeries.hasProjection && (
                      <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-violet-500 inline-block" /> Projected</span>
                    )}
                  </div>
                )}
              </div>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={forecastSeries.data} margin={{ top: 8, right: 16, left: 8, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
                    <XAxis dataKey="period" tick={{ fontSize: 11 }} />
                    <YAxis
                      tick={{ fontSize: 11 }}
                      width={60}
                      tickFormatter={(v: number) =>
                        v >= 1_000_000 ? `$${(v / 1_000_000).toFixed(1)}M` : v >= 1_000 ? `$${(v / 1_000).toFixed(0)}K` : `$${v}`
                      }
                    />
                    <Tooltip
                      contentStyle={{ borderRadius: "0.5rem", fontSize: "0.75rem" }}
                      formatter={(v: number) => [`$${v.toLocaleString()}`, ""]}
                    />
                    {forecastSeries.hasActual && (
                      <Line type="monotone" dataKey="actual" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} connectNulls={false} />
                    )}
                    {forecastSeries.hasProjection && (
                      <Line type="monotone" dataKey="projected" stroke="#8b5cf6" strokeWidth={2} strokeDasharray="6 3" dot={{ r: 3 }} connectNulls={false} />
                    )}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {scenarioData.length > 0 && (
            <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-6">
              <h3 className="text-sm font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-1">
                Scenario Comparison
              </h3>
              <p className="text-xs text-gray-400 dark:text-gray-500 mb-4">
                Entity: {chartEntityName}. X-axis: scenario name. Y-axis: projected value and projected growth.
              </p>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={scenarioData} margin={{ top: 8, right: 16, left: 8, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
                    <XAxis dataKey="scenario" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} width={55} />
                    <Tooltip contentStyle={{ borderRadius: "0.5rem", fontSize: "0.75rem" }} />
                    <Legend wrapperStyle={{ fontSize: "0.7rem" }} />
                    <Bar dataKey="projected_value" name="Value" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="projected_growth" name="Growth" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Row 6: Role Contribution ── */}
      {roleContributors.length > 0 && (
        <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-6">
          <h3 className="text-sm font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-1">
            Role Contribution
          </h3>
          <p className="text-xs text-gray-400 dark:text-gray-500 mb-4">
            Entity: {chartEntityName}. X-axis: contribution value. Y-axis: contributor name.
          </p>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={roleContributors} layout="vertical" margin={{ top: 4, right: 16, left: 8, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis dataKey="name" type="category" tick={{ fontSize: 11 }} width={100} />
                <Tooltip contentStyle={{ borderRadius: "0.5rem", fontSize: "0.75rem" }} />
                <Bar dataKey="contribution_value" fill="#3b82f6" radius={[0, 4, 4, 0]} barSize={20} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* ── Row 7: Structured Insights ── */}
      {structuredInsights.length > 0 && (
        <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-6">
          <h3 className="text-sm font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-4">
            Key Insights
          </h3>
          <div className="space-y-3">
            {structuredInsights.map((item, i) => {
              const dimColor =
                item.dimension === "kpi" ? "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300"
                : item.dimension === "macro" ? "bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-300"
                : item.dimension === "competitive" ? "bg-orange-100 text-orange-700 dark:bg-orange-900/40 dark:text-orange-300"
                : "bg-violet-100 text-violet-700 dark:bg-violet-900/40 dark:text-violet-300";
              return (
                <div key={i} className="border border-gray-200 dark:border-gray-700 rounded-xl p-4">
                  <div className="flex items-start justify-between gap-3 mb-2">
                    <p className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                      {item.title}
                    </p>
                    <span className={`shrink-0 px-2 py-0.5 rounded-full text-xs font-semibold ${dimColor}`}>
                      {item.dimension}
                    </span>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    {item.description}
                  </p>
                  <div className="flex items-center gap-3">
                    <span className="text-xs text-gray-400">Impact</span>
                    <div className="flex-1 max-w-xs h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full ${item.impact_score >= 75 ? "bg-emerald-500" : item.impact_score >= 50 ? "bg-amber-500" : "bg-red-500"}`}
                        style={{ width: `${item.impact_score}%` }}
                      />
                    </div>
                    <span className="text-xs font-medium tabular-nums text-gray-600 dark:text-gray-300">
                      {item.impact_score}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── Classification badge ── */}
      {dashboardData?.classification && (
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400 uppercase tracking-wider">Classification</span>
          <span
            className={`px-3 py-1 rounded-full text-sm font-medium border ${
              dashboardData.classification.confidence >= 85
                ? "bg-emerald-100 text-emerald-800 border-emerald-200 dark:bg-emerald-900/30 dark:text-emerald-300 dark:border-emerald-800"
                : dashboardData.classification.confidence >= 60
                  ? "bg-amber-100 text-amber-800 border-amber-200 dark:bg-amber-900/30 dark:text-amber-300 dark:border-amber-800"
                  : "bg-gray-100 text-gray-700 border-gray-200 dark:bg-gray-800 dark:text-gray-300 dark:border-gray-700"
            }`}
          >
            {dashboardData.classification.label} — {dashboardData.classification.confidence}%
          </span>
        </div>
      )}
    </div>
  );
};

export default InsightsDashboard;
