import { useMemo, type FC } from "react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  LineChart,
  Line,
  BarChart,
  Bar,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import { SectionHeader } from "../ui";
import type { PipelineSignalGrowth } from "../../api/client";

const CHART_COLORS = [
  "var(--chart-1)", "var(--chart-2)", "var(--chart-3)",
  "var(--chart-4)", "var(--chart-5)", "var(--chart-6)",
];

const yFmt = (v: number) =>
  v >= 1_000_000
    ? `$${(v / 1_000_000).toFixed(1)}M`
    : v >= 1_000
      ? `$${(v / 1_000).toFixed(0)}K`
      : `$${v}`;

interface KpiRow {
  period_end: string;
  metric_name: string;
  metric_value: number;
}

interface RevenueTrendPoint {
  period: string;
  value: number;
}

interface TrendChartsProps {
  kpiRows: KpiRow[];
  revenueTrend: RevenueTrendPoint[];
  growth: PipelineSignalGrowth | null;
  entityName: string;
}

/** Section 3: Revenue trend (area) + KPI multi-line + Growth horizons. */
const TrendCharts: FC<TrendChartsProps> = ({
  kpiRows,
  revenueTrend,
  growth,
  entityName,
}) => {
  // KPI time series grouped by period
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

  // Growth horizons bar data
  const growthData = useMemo(() => {
    if (!growth) return [];
    return [
      { horizon: "Short", growth: (growth.short_growth ?? 0) * 100 },
      { horizon: "Mid", growth: (growth.mid_growth ?? 0) * 100 },
      { horizon: "Long", growth: (growth.long_growth ?? 0) * 100 },
    ];
  }, [growth]);

  const accPct = (growth?.trend_acceleration ?? 0) * 100;
  const hasRevenue = revenueTrend.length > 0;
  const hasTimeSeries = timeSeriesData.length > 0;
  const hasGrowth = growthData.length > 0 && (growth?.short_growth != null || growth?.mid_growth != null || growth?.long_growth != null);

  if (!hasRevenue && !hasTimeSeries && !hasGrowth) return null;

  return (
    <div className="space-y-6">
      {/* Revenue area + Growth horizons side by side */}
      {(hasRevenue || hasGrowth) && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {hasRevenue && (
            <div className="ia-card p-5">
              <SectionHeader
                title="Revenue Trend"
                subtitle={entityName}
              />
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={revenueTrend} margin={{ top: 8, right: 16, left: 8, bottom: 0 }}>
                    <defs>
                      <linearGradient id="revGradNew" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="var(--chart-1)" stopOpacity={0.25} />
                        <stop offset="95%" stopColor="var(--chart-1)" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
                    <XAxis dataKey="period" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} tickFormatter={yFmt} width={60} />
                    <Tooltip
                      formatter={(v: number) => [`$${v.toLocaleString()}`, "Revenue"]}
                      contentStyle={{ borderRadius: "0.5rem", fontSize: "0.75rem" }}
                    />
                    <Area
                      type="monotone"
                      dataKey="value"
                      stroke="var(--chart-1)"
                      strokeWidth={2.5}
                      fill="url(#revGradNew)"
                      dot={{ r: 3, fill: "var(--chart-1)" }}
                      activeDot={{ r: 5 }}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {hasGrowth && (
            <div className="ia-card p-5">
              <SectionHeader
                title="Growth Horizons"
                subtitle="Short / Mid / Long-term growth rates"
                action={
                  <span className={`ia-mono text-xs font-semibold ${accPct >= 0 ? "text-emerald-600" : "text-red-500"}`}>
                    Accel: {accPct >= 0 ? "+" : ""}{accPct.toFixed(2)}%
                  </span>
                }
              />
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={growthData} margin={{ top: 8, right: 16, left: 8, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
                    <XAxis dataKey="horizon" tick={{ fontSize: 12 }} />
                    <YAxis tick={{ fontSize: 11 }} tickFormatter={(v: number) => `${v.toFixed(1)}%`} width={55} />
                    <Tooltip
                      formatter={(v: number) => [`${v.toFixed(2)}%`, "Growth"]}
                      contentStyle={{ borderRadius: "0.5rem", fontSize: "0.75rem" }}
                    />
                    <Bar dataKey="growth" radius={[6, 6, 0, 0]} barSize={48}>
                      {growthData.map((entry, i) => (
                        <Cell key={i} fill={entry.growth >= 0 ? "var(--ia-success)" : "var(--ia-danger)"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>
      )}

      {/* KPI multi-line time series */}
      {hasTimeSeries && (
        <div className="ia-card p-5">
          <SectionHeader
            title="KPI Trends Over Time"
            subtitle={entityName}
          />
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
    </div>
  );
};

export default TrendCharts;
