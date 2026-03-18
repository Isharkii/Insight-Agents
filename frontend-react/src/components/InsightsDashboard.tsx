import { useRef, useEffect, useMemo, type FC } from "react";
import type { AnalyzeResult, BusinessIntelligenceResponse } from "../api/client";
import type { DashboardData } from "./IntelligenceDashboard/types";
import {
  HeroInsight,
  MetricCards,
  TrendCharts,
  DriverAnalysis,
  ForecastScenarios,
  RisksRecommendations,
} from "./sections";

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

// ─── Main Dashboard (thin orchestrator) ──────────────────────────────────────

const InsightsDashboard: FC<InsightsDashboardProps> = ({
  analyzeResult,
  dashboardData,
  kpiRows,
  derivedSignals,
  executionTime,
  entityName,
  reportInsight,
  biData: _biData,
}) => {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    ref.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [analyzeResult]);

  const ps = analyzeResult.pipeline_signals;
  const chartEntityName = entityName || dashboardData?.entity_name || "selected entity";

  // KPI metric entries from dashboard
  const kpiEntries = useMemo(() => {
    if (!dashboardData?.kpi_metrics) return [];
    return Object.entries(dashboardData.kpi_metrics).map(([name, entry]) => ({
      name,
      value: entry.value,
      unit: entry.unit,
      label: entry.label,
    }));
  }, [dashboardData]);

  // Build metric series lookup for sparklines
  const metricSeries = useMemo(() => {
    const series: Record<string, number[]> = {};
    if (ps?.growth?.metric_series) {
      for (const [name, values] of Object.entries(ps.growth.metric_series)) {
        if (Array.isArray(values)) series[name] = values;
      }
    }
    // Also build from KPI rows
    for (const row of kpiRows) {
      const name = row.metric_name;
      if (!series[name]) series[name] = [];
      series[name].push(row.metric_value);
    }
    return series;
  }, [ps, kpiRows]);

  // Scenarios from derived signals
  const scenarios = derivedSignals.multivariate_scenario?.scenario_simulation?.scenarios ?? null;

  return (
    <div ref={ref} className="ia-section-gap-lg">
      {/* ── Section 1: Hero Insight ── */}
      <HeroInsight
        result={analyzeResult}
        riskScore={derivedSignals.risk?.risk_score}
        riskLevel={derivedSignals.risk?.risk_level}
        healthIndex={dashboardData?.health_index}
        healthLabel={dashboardData?.health_label}
      />

      {/* ── Section 2: Key Metrics ── */}
      <MetricCards
        kpiEntries={kpiEntries}
        metricSeries={metricSeries}
      />

      {/* ── Section 3: Strategies & Recommendations (above charts) ── */}
      <RisksRecommendations
        result={reportInsight ?? analyzeResult}
        prioritization={ps?.prioritization}
        structuredInsights={dashboardData?.insights}
        entityName={chartEntityName}
      />

      {/* ── Section 4: Trends ── */}
      <TrendCharts
        kpiRows={kpiRows}
        revenueTrend={dashboardData?.revenue_trend ?? []}
        growth={ps?.growth ?? null}
        entityName={chartEntityName}
      />

      {/* ── Section 5: Driver Analysis ── */}
      {ps && (
        <DriverAnalysis
          signals={ps}
          overallConfidence={analyzeResult.confidence_score}
        />
      )}

      {/* ── Section 6: Forecast & Scenarios ── */}
      <ForecastScenarios
        dashboardData={dashboardData}
        scenarios={scenarios}
        entityName={chartEntityName}
        unitEconomics={ps?.unit_economics}
      />

      {/* ── Classification badge ── */}
      {dashboardData?.classification && (
        <div className="flex items-center gap-2 ia-fade-up">
          <span className="ia-caption uppercase tracking-wider">Classification</span>
          <span
            className={`px-3 py-1 rounded-full text-sm font-medium border ${
              dashboardData.classification.confidence >= 85
                ? "bg-emerald-100 text-emerald-800 border-emerald-200"
                : dashboardData.classification.confidence >= 60
                  ? "bg-amber-100 text-amber-800 border-amber-200"
                  : "bg-gray-100 text-gray-700 border-gray-200"
            }`}
          >
            {dashboardData.classification.label} — {dashboardData.classification.confidence}%
          </span>
        </div>
      )}

      {/* ── Execution time footer ── */}
      {executionTime !== undefined && (
        <p className="ia-caption text-right">
          Analysis completed in {executionTime.toFixed(1)}s
        </p>
      )}
    </div>
  );
};

export default InsightsDashboard;
