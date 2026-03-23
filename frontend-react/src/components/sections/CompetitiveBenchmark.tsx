import type { FC } from "react";
import { SectionHeader, Badge } from "../ui";

interface MetricRank {
  rank?: number;
  percentile?: number;
  client_value?: number;
  field_mean?: number;
  field_median?: number;
}

interface CompetitiveMetrics {
  relative_growth_index?: number | null;
  market_share_proxy?: number | null;
  stability_score?: number;
  momentum_classification?: string;
  risk_divergence_score?: number | null;
}

interface BenchmarkData {
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
    metric_ranks?: Record<string, MetricRank>;
  };
  composite?: {
    overall_score?: number;
    base_overall_score?: number;
    growth_score?: number;
    level_score?: number;
    stability_score?: number;
    confidence_score?: number;
    competitive_metrics?: CompetitiveMetrics;
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
}

interface CompetitiveBenchmarkProps {
  benchmark: BenchmarkData;
  entityName: string;
}

function tierVariant(tier: string): "success" | "warning" | "danger" | "info" | "neutral" {
  const t = tier.toLowerCase();
  if (t === "top_quartile" || t === "leader") return "success";
  if (t === "second_quartile" || t === "challenger") return "info";
  if (t === "third_quartile" || t === "stable") return "warning";
  return "danger";
}

function momentumVariant(m: string): "success" | "warning" | "danger" | "info" | "neutral" {
  const l = m.toLowerCase();
  if (l === "accelerating" || l === "leader") return "success";
  if (l === "stable" || l === "steady") return "info";
  if (l === "decelerating" || l === "declining") return "warning";
  return "danger";
}

function fmtPct(v: number | null | undefined): string {
  if (v == null) return "N/A";
  return `${(v * 100).toFixed(0)}%`;
}

function fmtScore(v: number | null | undefined): string {
  if (v == null) return "N/A";
  return v.toFixed(1);
}

/** Section: Competitive Benchmark — peer ranking, composite scores, metric-level detail. */
const CompetitiveBenchmark: FC<CompetitiveBenchmarkProps> = ({ benchmark, entityName }) => {
  const ranking = benchmark.ranking;
  const composite = benchmark.composite;
  const peers = benchmark.peer_selection?.selected_peers ?? [];
  const metricRanks = ranking?.metric_ranks ?? {};
  const compMetrics = composite?.competitive_metrics;
  const hasRanking = ranking && ranking.overall_rank != null;
  const hasComposite = composite && composite.overall_score != null;

  if (!hasRanking && !hasComposite) return null;

  return (
    <div className="space-y-6">
      {/* Market Position Summary */}
      <div className="ia-card p-5">
        <SectionHeader
          title="Competitive Benchmark"
          subtitle={`${entityName} vs ${peers.length} peer${peers.length !== 1 ? "s" : ""}`}
          action={
            ranking?.tier ? (
              <Badge variant={tierVariant(ranking.tier)}>
                {ranking.tier.replace(/_/g, " ")}
              </Badge>
            ) : undefined
          }
        />

        {/* Top-level rank + composite scores */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4 mb-6">
          {hasRanking && (
            <ScoreCard
              label="Overall Rank"
              value={`#${ranking!.overall_rank}`}
              sub={`of ${ranking!.total_participants ?? "?"}`}
            />
          )}
          {ranking?.overall_percentile != null && (
            <ScoreCard
              label="Percentile"
              value={`${Math.round(ranking.overall_percentile)}th`}
              color={ranking.overall_percentile >= 75 ? "text-emerald-600" : ranking.overall_percentile >= 50 ? "text-blue-600" : "text-amber-600"}
            />
          )}
          {hasComposite && (
            <>
              <ScoreCard label="Overall Score" value={fmtScore(composite!.overall_score)} />
              <ScoreCard label="Growth Score" value={fmtScore(composite!.growth_score)} />
              <ScoreCard label="Level Score" value={fmtScore(composite!.level_score)} />
              <ScoreCard label="Stability" value={fmtScore(composite!.stability_score)} />
            </>
          )}
        </div>

        {/* Competitive Metrics row */}
        {compMetrics && (
          <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
            <p className="ia-caption uppercase tracking-wider mb-3">Competitive Metrics</p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              {compMetrics.relative_growth_index != null && (
                <MiniMetric label="Relative Growth Index" value={compMetrics.relative_growth_index.toFixed(2)} />
              )}
              {compMetrics.market_share_proxy != null && (
                <MiniMetric label="Market Share Proxy" value={fmtPct(compMetrics.market_share_proxy)} />
              )}
              {compMetrics.stability_score != null && (
                <MiniMetric label="Stability Score" value={compMetrics.stability_score.toFixed(2)} />
              )}
              {compMetrics.momentum_classification && (
                <div className="ia-card-inline p-3">
                  <p className="ia-caption uppercase tracking-wider text-[10px] mb-1">Momentum</p>
                  <Badge variant={momentumVariant(compMetrics.momentum_classification)}>
                    {compMetrics.momentum_classification}
                  </Badge>
                </div>
              )}
              {compMetrics.risk_divergence_score != null && (
                <MiniMetric label="Risk Divergence" value={compMetrics.risk_divergence_score.toFixed(2)} />
              )}
            </div>
          </div>
        )}
      </div>

      {/* Metric-level ranks table */}
      {Object.keys(metricRanks).length > 0 && (
        <div className="ia-card p-5">
          <SectionHeader title="Metric-Level Ranking" subtitle="Per-metric rank, percentile, and field comparison" />
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left py-2 pr-4 ia-caption uppercase tracking-wider">Metric</th>
                  <th className="text-right py-2 px-3 ia-caption uppercase tracking-wider">Rank</th>
                  <th className="text-right py-2 px-3 ia-caption uppercase tracking-wider">Percentile</th>
                  <th className="text-right py-2 px-3 ia-caption uppercase tracking-wider">Your Value</th>
                  <th className="text-right py-2 px-3 ia-caption uppercase tracking-wider">Field Mean</th>
                  <th className="text-right py-2 pl-3 ia-caption uppercase tracking-wider">Field Median</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(metricRanks).map(([metric, rank]) => (
                  <MetricRankRow key={metric} metric={metric} rank={rank} />
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Peer scores */}
      {ranking?.peer_scores && Object.keys(ranking.peer_scores).length > 0 && (
        <div className="ia-card p-5">
          <SectionHeader title="Peer Comparison" subtitle="Composite scores across selected peers" />
          <div className="space-y-2">
            {Object.entries(ranking.peer_scores)
              .sort(([, a], [, b]) => b - a)
              .map(([peer, score]) => (
                <PeerBar
                  key={peer}
                  name={peer}
                  score={score}
                  isClient={peer.toLowerCase() === entityName.toLowerCase()}
                />
              ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default CompetitiveBenchmark;

// ── Sub-components ───────────────────────────────────────────────────

const ScoreCard: FC<{ label: string; value: string; sub?: string; color?: string }> = ({
  label, value, sub, color,
}) => (
  <div className="ia-card-inline p-3 text-center">
    <p className="ia-caption uppercase tracking-wider text-[10px] mb-1">{label}</p>
    <p className={`text-xl font-bold ia-mono ${color ?? "text-gray-900 dark:text-gray-100"}`}>{value}</p>
    {sub && <p className="ia-caption text-[10px] mt-0.5">{sub}</p>}
  </div>
);

const MiniMetric: FC<{ label: string; value: string }> = ({ label, value }) => (
  <div className="ia-card-inline p-3">
    <p className="ia-caption uppercase tracking-wider text-[10px] mb-1">{label}</p>
    <p className="text-sm font-semibold ia-mono text-gray-900 dark:text-gray-100">{value}</p>
  </div>
);

const MetricRankRow: FC<{ metric: string; rank: MetricRank }> = ({ metric, rank }) => {
  const pctColor = (rank.percentile ?? 0) >= 75
    ? "text-emerald-600"
    : (rank.percentile ?? 0) >= 50
      ? "text-blue-600"
      : (rank.percentile ?? 0) >= 25
        ? "text-amber-600"
        : "text-red-500";
  return (
    <tr className="border-b border-gray-100 dark:border-gray-800 last:border-0">
      <td className="py-2 pr-4 font-medium text-gray-900 dark:text-gray-100 capitalize">
        {metric.replace(/_/g, " ")}
      </td>
      <td className="text-right py-2 px-3 ia-mono">{rank.rank ?? "-"}</td>
      <td className={`text-right py-2 px-3 ia-mono font-semibold ${pctColor}`}>
        {rank.percentile != null ? `${Math.round(rank.percentile)}%` : "-"}
      </td>
      <td className="text-right py-2 px-3 ia-mono">
        {rank.client_value != null ? rank.client_value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : "-"}
      </td>
      <td className="text-right py-2 px-3 ia-mono text-gray-500">
        {rank.field_mean != null ? rank.field_mean.toLocaleString(undefined, { maximumFractionDigits: 2 }) : "-"}
      </td>
      <td className="text-right py-2 pl-3 ia-mono text-gray-500">
        {rank.field_median != null ? rank.field_median.toLocaleString(undefined, { maximumFractionDigits: 2 }) : "-"}
      </td>
    </tr>
  );
};

const PeerBar: FC<{ name: string; score: number; isClient: boolean }> = ({ name, score, isClient }) => (
  <div className="flex items-center gap-3">
    <span className={`w-32 truncate text-sm ${isClient ? "font-bold text-blue-700 dark:text-blue-300" : "text-gray-700 dark:text-gray-300"}`}>
      {name}
      {isClient && " (you)"}
    </span>
    <div className="flex-1 h-5 bg-gray-100 dark:bg-gray-800 rounded-full overflow-hidden">
      <div
        className={`h-full rounded-full transition-all ${isClient ? "bg-blue-500" : "bg-gray-400 dark:bg-gray-600"}`}
        style={{ width: `${Math.min(100, score)}%` }}
      />
    </div>
    <span className="ia-mono text-xs font-semibold w-10 text-right">{score.toFixed(1)}</span>
  </div>
);
