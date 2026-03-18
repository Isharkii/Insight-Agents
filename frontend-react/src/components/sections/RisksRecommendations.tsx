import type { FC } from "react";
import { SectionHeader, Badge } from "../ui";
import type { AnalyzeResult, PipelineSignalPrioritization } from "../../api/client";

interface StructuredInsight {
  title: string;
  description: string;
  impact_score: number;
  dimension: string;
}

interface RisksRecommendationsProps {
  result: AnalyzeResult;
  prioritization?: PipelineSignalPrioritization | null;
  structuredInsights?: StructuredInsight[];
  entityName: string;
}

/** Section 6: Prioritization + Strategies + Key Insights. */
const RisksRecommendations: FC<RisksRecommendationsProps> = ({
  result,
  prioritization,
  structuredInsights,
  entityName,
}) => {
  const hasPrioritization = !!prioritization;
  const hasInsights = structuredInsights && structuredInsights.length > 0;
  const strategies = buildStrategies(result);

  if (!hasPrioritization && !hasInsights && strategies.length === 0) return null;

  return (
    <div className="space-y-6">
      {/* Prioritization panel */}
      {hasPrioritization && <PrioritizationPanel p={prioritization!} />}

      {/* Strategy steps */}
      {strategies.length > 0 && (
        <div className="ia-card p-5">
          <div className="bg-blue-50 dark:bg-blue-900/10 border border-blue-200 dark:border-blue-800 rounded-xl p-4">
            <SectionHeader
              title="Strategies"
              subtitle={`Step-by-step plan for ${entityName}`}
            />
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

      {/* Key insights list */}
      {hasInsights && (
        <div className="ia-card p-5">
          <SectionHeader title="Key Insights" />
          <div className="space-y-3">
            {structuredInsights!.map((item, i) => (
              <InsightRow key={i} item={item} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default RisksRecommendations;

// ── Sub-components ───────────────────────────────────────────────────

function buildStrategies(insight: AnalyzeResult) {
  if (!insight.insight) return [];
  return [
    { title: "Execute the recommended action", detail: `Implement: ${insight.recommended_action}` },
    { title: "Validate the drivers cited in evidence", detail: `Confirm: ${insight.evidence}` },
    { title: "Monitor the stated impact", detail: `Track: ${insight.impact}` },
    { title: "Review priority and confidence", detail: `Align to ${insight.priority} priority, re-score monthly (confidence ${(insight.confidence_score ?? 0).toFixed(2)})` },
  ];
}

const PrioritizationPanel: FC<{ p: PipelineSignalPrioritization }> = ({ p }) => {
  const level = (p.priority_level || "low").toLowerCase();
  const levelVariant =
    level === "critical" ? "danger" as const
    : level === "high" ? "warning" as const
    : level === "moderate" ? "info" as const
    : "neutral" as const;

  return (
    <div className="ia-card p-5">
      <SectionHeader
        title="Prioritization"
        action={
          <div className="flex items-center gap-2">
            <Badge variant={levelVariant}>{level}</Badge>
            {p.confidence_score != null && (
              <span className="ia-caption">Confidence {Math.round(p.confidence_score * 100)}%</span>
            )}
          </div>
        }
      />
      {p.recommended_focus && (
        <p className="ia-body mb-4">
          <span className="font-medium text-gray-500">Focus:</span> {p.recommended_focus}
        </p>
      )}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-sm">
        {p.cohort_risk_hint && (
          <div className="ia-card-inline p-3">
            <p className="ia-caption uppercase tracking-wider mb-1">Cohort Risk</p>
            <p className="font-semibold text-gray-900 dark:text-gray-100 capitalize">{p.cohort_risk_hint}</p>
          </div>
        )}
        {p.scenario_worst_growth != null && (
          <div className="ia-card-inline p-3">
            <p className="ia-caption uppercase tracking-wider mb-1">Worst Scenario</p>
            <p className={`font-semibold ia-mono ${p.scenario_worst_growth >= 0 ? "text-emerald-600" : "text-red-500"}`}>
              {(p.scenario_worst_growth * 100).toFixed(1)}%
            </p>
          </div>
        )}
        {p.scenario_best_growth != null && (
          <div className="ia-card-inline p-3">
            <p className="ia-caption uppercase tracking-wider mb-1">Best Scenario</p>
            <p className={`font-semibold ia-mono ${p.scenario_best_growth >= 0 ? "text-emerald-600" : "text-red-500"}`}>
              {(p.scenario_best_growth * 100).toFixed(1)}%
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

const dimColorMap: Record<string, string> = {
  kpi: "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300",
  macro: "bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-300",
  competitive: "bg-orange-100 text-orange-700 dark:bg-orange-900/40 dark:text-orange-300",
  simulation: "bg-violet-100 text-violet-700 dark:bg-violet-900/40 dark:text-violet-300",
};

const InsightRow: FC<{ item: StructuredInsight }> = ({ item }) => (
  <div className="ia-card-inline p-4">
    <div className="flex items-start justify-between gap-3 mb-2">
      <p className="text-sm font-semibold text-gray-900 dark:text-gray-100">{item.title}</p>
      <span className={`shrink-0 px-2 py-0.5 rounded-full text-xs font-semibold ${dimColorMap[item.dimension] ?? dimColorMap.simulation}`}>
        {item.dimension}
      </span>
    </div>
    <p className="ia-body mb-3">{item.description}</p>
    <div className="flex items-center gap-3">
      <span className="ia-caption">Impact</span>
      <div className="flex-1 max-w-xs h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${
            item.impact_score >= 75 ? "bg-emerald-500" : item.impact_score >= 50 ? "bg-amber-500" : "bg-red-500"
          }`}
          style={{ width: `${item.impact_score}%` }}
        />
      </div>
      <span className="text-xs font-medium ia-mono text-gray-600 dark:text-gray-300">
        {item.impact_score}
      </span>
    </div>
  </div>
);
