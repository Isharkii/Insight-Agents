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
import type { AnalyzeResult } from "../api/client";
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
}

interface InsightsDashboardProps {
  analyzeResult: AnalyzeResult;
  dashboardData: DashboardData | null;
  kpiRows: KpiRow[];
  derivedSignals: ReportDerivedSignals;
  executionTime?: number;
  entityName?: string;
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
}) => {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    ref.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [analyzeResult]);

  const pct = Math.round(analyzeResult.confidence_score * 100);

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

  // Market share / forecast from dashboard
  const marketShare = useMemo(() => {
    const raw = dashboardData?.market_share ?? [];
    if (raw.length === 0) return { data: [], hasProjection: false };
    const firstProj = raw.findIndex((p) => p.projected);
    const data = raw.map((p) => ({
      period: p.period,
      actual: p.projected ? undefined : p.value,
      projected: p.projected ? p.value : undefined,
    }));
    // bridge gap
    if (firstProj > 0) {
      data[firstProj] = { ...data[firstProj], projected: raw[firstProj - 1].value };
      data[firstProj - 1] = { ...data[firstProj - 1], projected: raw[firstProj - 1].value };
    }
    return { data, hasProjection: firstProj >= 0 };
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
            <h3 className="text-sm font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-4">
              Revenue Trend
            </h3>
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
            <h3 className="text-sm font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-4">
              KPI Distribution
            </h3>
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
          <h3 className="text-sm font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-4">
            KPI Trends Over Time
          </h3>
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
      {(marketShare.data.length > 0 || scenarioData.length > 0) && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {marketShare.data.length > 0 && (
            <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">
                  Forecast Projection
                </h3>
                {marketShare.hasProjection && (
                  <div className="flex items-center gap-3 text-xs text-gray-400">
                    <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-blue-500 inline-block" /> Actual</span>
                    <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-violet-500 inline-block" /> Projected</span>
                  </div>
                )}
              </div>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={marketShare.data} margin={{ top: 8, right: 16, left: 8, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
                    <XAxis dataKey="period" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} width={50} />
                    <Tooltip contentStyle={{ borderRadius: "0.5rem", fontSize: "0.75rem" }} />
                    <Line type="monotone" dataKey="actual" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} connectNulls={false} />
                    <Line type="monotone" dataKey="projected" stroke="#8b5cf6" strokeWidth={2} strokeDasharray="6 3" dot={{ r: 3 }} connectNulls={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {scenarioData.length > 0 && (
            <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-6">
              <h3 className="text-sm font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-4">
                Scenario Comparison
              </h3>
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
          <h3 className="text-sm font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-4">
            Role Contribution
          </h3>
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
