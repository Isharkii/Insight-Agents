import { useMemo, type FC } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import { SectionHeader } from "../ui";
import type { DashboardData } from "../IntelligenceDashboard/types";

interface ScenarioRecord {
  projected_value?: number;
  projected_growth?: number;
}

interface ForecastScenariosProps {
  dashboardData: DashboardData | null;
  scenarios?: Record<string, ScenarioRecord> | null;
  entityName: string;
  /** Unit economics data from pipeline signals. */
  unitEconomics?: {
    status: string;
    confidence: number;
    ltv: number | null;
    cac: number | null;
    ltv_cac_ratio: number | null;
    payback_months: number | null;
  } | null;
}

const yFmt = (v: number) =>
  v >= 1_000_000
    ? `$${(v / 1_000_000).toFixed(1)}M`
    : v >= 1_000
      ? `$${(v / 1_000).toFixed(0)}K`
      : `$${v}`;

/** Section 5: Forecast projection + Scenarios + Unit Economics. */
const ForecastScenarios: FC<ForecastScenariosProps> = ({
  dashboardData,
  scenarios,
  entityName,
  unitEconomics,
}) => {
  // Build forecast series from dashboard data
  const forecastSeries = useMemo(() => {
    const history = dashboardData?.revenue_trend ?? [];
    const projections = dashboardData?.market_share ?? [];
    if (history.length === 0 && projections.length === 0) {
      return { data: [] as { period: string; actual?: number; projected?: number }[], hasActual: false, hasProjection: false };
    }
    const byPeriod = new Map<string, { period: string; actual?: number; projected?: number }>();
    for (const p of history) byPeriod.set(p.period, { period: p.period, actual: p.value });
    for (const p of projections) {
      const existing = byPeriod.get(p.period) ?? { period: p.period };
      if (p.projected) existing.projected = p.value;
      else existing.actual = p.value;
      byPeriod.set(p.period, existing);
    }
    const data = Array.from(byPeriod.values()).sort((a, b) => a.period.localeCompare(b.period));
    return {
      data,
      hasActual: data.some((d) => d.actual != null),
      hasProjection: data.some((d) => d.projected != null),
    };
  }, [dashboardData]);

  // Scenario bar data
  const scenarioData = useMemo(() => {
    if (!scenarios) return [];
    return Object.entries(scenarios).map(([name, vals]) => ({
      scenario: name,
      projected_value: vals.projected_value ?? 0,
      projected_growth: vals.projected_growth ?? 0,
    }));
  }, [scenarios]);

  const hasForecast = forecastSeries.data.length > 0;
  const hasScenarios = scenarioData.length > 0;
  const hasUE = unitEconomics && unitEconomics.status !== "failed" && unitEconomics.status !== "skipped";

  if (!hasForecast && !hasScenarios && !hasUE) return null;

  return (
    <div className="space-y-6">
      {(hasForecast || hasScenarios) && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {hasForecast && (
            <div className="ia-card p-5">
              <SectionHeader
                title="Forecast Projection"
                subtitle={entityName}
                action={
                  <div className="flex items-center gap-3 text-xs ia-caption">
                    {forecastSeries.hasActual && (
                      <span className="flex items-center gap-1">
                        <span className="w-3 h-0.5 bg-blue-500 inline-block" /> Actual
                      </span>
                    )}
                    {forecastSeries.hasProjection && (
                      <span className="flex items-center gap-1">
                        <span className="w-3 h-0.5 bg-violet-500 inline-block" /> Projected
                      </span>
                    )}
                  </div>
                }
              />
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={forecastSeries.data} margin={{ top: 8, right: 16, left: 8, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
                    <XAxis dataKey="period" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} width={60} tickFormatter={yFmt} />
                    <Tooltip
                      contentStyle={{ borderRadius: "0.5rem", fontSize: "0.75rem" }}
                      formatter={(v: number) => [`$${v.toLocaleString()}`, ""]}
                    />
                    {forecastSeries.hasActual && (
                      <Line type="monotone" dataKey="actual" stroke="var(--chart-1)" strokeWidth={2} dot={{ r: 3 }} connectNulls={false} />
                    )}
                    {forecastSeries.hasProjection && (
                      <Line type="monotone" dataKey="projected" stroke="var(--chart-3)" strokeWidth={2} strokeDasharray="6 3" dot={{ r: 3 }} connectNulls={false} />
                    )}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {hasScenarios && (
            <div className="ia-card p-5">
              <SectionHeader title="Scenario Comparison" subtitle={entityName} />
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={scenarioData} margin={{ top: 8, right: 16, left: 8, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
                    <XAxis dataKey="scenario" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} width={55} />
                    <Tooltip contentStyle={{ borderRadius: "0.5rem", fontSize: "0.75rem" }} />
                    <Legend wrapperStyle={{ fontSize: "0.7rem" }} />
                    <Bar dataKey="projected_value" name="Value" fill="var(--chart-1)" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="projected_growth" name="Growth" fill="var(--chart-3)" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Unit Economics */}
      {hasUE && <UnitEconomicsPanel ue={unitEconomics!} />}
    </div>
  );
};

export default ForecastScenarios;

// ── Unit Economics sub-component ──────────────────────────────────────

const UnitEconomicsPanel: FC<{
  ue: NonNullable<ForecastScenariosProps["unitEconomics"]>;
}> = ({ ue }) => {
  const cards = [
    { label: "LTV", value: ue.ltv, fmt: (v: number) => `$${v >= 1000 ? (v / 1000).toFixed(1) + "K" : v.toFixed(0)}` },
    { label: "CAC", value: ue.cac, fmt: (v: number) => `$${v >= 1000 ? (v / 1000).toFixed(1) + "K" : v.toFixed(0)}` },
    { label: "LTV/CAC", value: ue.ltv_cac_ratio, fmt: (v: number) => v.toFixed(2) + "x" },
    { label: "Payback", value: ue.payback_months, fmt: (v: number) => v.toFixed(1) + " mo" },
  ];
  return (
    <div className="ia-card p-5">
      <SectionHeader
        title="Unit Economics"
        action={<span className="ia-caption">Confidence {Math.round(ue.confidence * 100)}%</span>}
      />
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {cards.map((card) => (
          <div key={card.label} className="ia-card-inline p-4 text-center">
            <p className="ia-caption uppercase tracking-wider mb-1">{card.label}</p>
            <p className="text-2xl font-bold ia-mono text-gray-900 dark:text-gray-100">
              {card.value != null ? card.fmt(card.value) : "N/A"}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
};
