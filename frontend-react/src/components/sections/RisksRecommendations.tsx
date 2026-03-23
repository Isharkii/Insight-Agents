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

function sanitizeLimitedText(text: string): string {
  const raw = String(text || "").trim();
  const noPrefix = raw.replace(/conditional:\s*/gi, "").trim();
  const noApiNoise = noPrefix.replace(/\(LLM API rate limit exceeded\.[^)]+\)/gi, "").trim();
  const softened = noApiNoise
    .replace(/\bblocked\b/gi, "limited")
    .replace(/\bfailed\b/gi, "limited");
  return softened
    .replace(/\s{2,}/g, " ")
    .replace(/\s+([.,;:])/g, "$1")
    .trim();
}

function normalizeImpactScore(value: number): number {
  if (!Number.isFinite(value)) return 15;
  const bounded = Math.max(0, Math.min(100, value));
  return bounded <= 0 ? 15 : bounded;
}

/** Section: Prioritization + Growth Horizons + Strategies + Key Insights. */
const RisksRecommendations: FC<RisksRecommendationsProps> = ({
  result,
  prioritization,
  structuredInsights,
  entityName,
}) => {
  const hasPrioritization = !!prioritization;
  const cleanedInsights = (structuredInsights ?? []).map((item) => ({
    ...item,
    title: sanitizeLimitedText(item.title),
    description: sanitizeLimitedText(item.description),
    impact_score: normalizeImpactScore(item.impact_score),
  }));
  const hasInsights = cleanedInsights.length > 0;
  const strategies = buildStrategies(result, prioritization);

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
            {cleanedInsights.map((item, i) => (
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

function isSynthesisBlocked(result: AnalyzeResult): boolean {
  const ps = result.pipeline_signals;
  if (ps?.synthesis_blocked === true) return true;
  if (!(result.insight ?? "").trim()) return true;
  return false;
}

function displayConfidencePct(value: number | null | undefined): number {
  if (value == null || !Number.isFinite(value)) return 15;
  const pct = Math.round(value * 100);
  return pct <= 0 ? 15 : pct;
}

function buildStrategies(
  insight: AnalyzeResult,
  prioritization?: PipelineSignalPrioritization | null,
) {
  // When synthesis produced output (even low-confidence "Conditional:"),
  // build strategies from the insight.  Strip the Conditional: prefix.
  if (insight.insight && !isSynthesisBlocked(insight)) {
    return [
      { title: "Execute the recommended action", detail: `Implement: ${sanitizeLimitedText(insight.recommended_action)}` },
      { title: "Validate the drivers cited in evidence", detail: `Confirm: ${sanitizeLimitedText(insight.evidence)}` },
      { title: "Monitor the stated impact", detail: `Track: ${sanitizeLimitedText(insight.impact)}` },
      { title: "Review priority and confidence", detail: `Align to ${insight.priority} priority, re-score monthly (confidence ${(insight.confidence_score ?? 0).toFixed(2)})` },
    ];
  }

  // When synthesis is genuinely blocked, build strategies from pipeline signals
  // (prioritization node output) so the user still sees actionable steps.
  if (!prioritization) return [];
  const steps: { title: string; detail: string }[] = [];

  if (prioritization.recommended_focus) {
    steps.push({ title: "Primary focus area", detail: prioritization.recommended_focus });
  }
  if (prioritization.cohort_risk_hint) {
    steps.push({ title: "Address cohort risk", detail: `Cohort risk is ${prioritization.cohort_risk_hint} — investigate retention and churn drivers` });
  }
  if (prioritization.growth_short != null) {
    const pct = (prioritization.growth_short * 100).toFixed(1);
    steps.push({ title: "Short-term growth trajectory", detail: `Projected at ${pct}% — ${prioritization.growth_short >= 0 ? "sustain momentum" : "identify and address decline drivers"}` });
  }
  if ((prioritization.signal_conflict_count ?? 0) > 0) {
    steps.push({ title: "Resolve signal conflicts", detail: `${prioritization.signal_conflict_count} conflicting signal(s) detected — review data quality and reconcile contradictory metrics` });
  }
  if (prioritization.scenario_worst_growth != null && prioritization.scenario_best_growth != null) {
    const worst = (prioritization.scenario_worst_growth * 100).toFixed(1);
    const best = (prioritization.scenario_best_growth * 100).toFixed(1);
    steps.push({ title: "Scenario range", detail: `Worst ${worst}% to best ${best}% — build contingency plans for downside scenarios` });
  }
  if (steps.length === 0) {
    steps.push({ title: "Improve data coverage", detail: "Upload additional metric data to enable full synthesis and actionable strategy generation" });
  }
  return steps;
}

const PrioritizationPanel: FC<{ p: PipelineSignalPrioritization }> = ({ p }) => {
  const level = (p.priority_level || "low").toLowerCase();
  const levelVariant =
    level === "critical" ? "danger" as const
    : level === "high" ? "warning" as const
    : level === "moderate" ? "info" as const
    : "neutral" as const;

  const hasGrowth = p.growth_short != null || p.growth_mid != null || p.growth_long != null;
  const hasConflicts = (p.signal_conflict_count ?? 0) > 0;

  return (
    <div className="ia-card p-5">
      <SectionHeader
        title="Prioritization"
        action={
          <div className="flex items-center gap-2">
            <Badge variant={levelVariant}>{level}</Badge>
            {p.confidence_score != null && (
              <span className="ia-caption">Confidence {displayConfidencePct(p.confidence_score)}%</span>
            )}
          </div>
        }
      />
      {p.recommended_focus && (
        <p className="ia-body mb-4">
          <span className="font-medium text-gray-500">Focus:</span> {p.recommended_focus}
        </p>
      )}

      {/* Primary stats row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm mb-4">
        {p.cohort_risk_hint && (
          <StatCard label="Cohort Risk" value={p.cohort_risk_hint} capitalize />
        )}
        {p.scenario_worst_growth != null && (
          <StatCard
            label="Worst Scenario"
            value={`${(p.scenario_worst_growth * 100).toFixed(1)}%`}
            color={p.scenario_worst_growth >= 0 ? "text-emerald-600" : "text-red-500"}
            mono
          />
        )}
        {p.scenario_best_growth != null && (
          <StatCard
            label="Best Scenario"
            value={`${(p.scenario_best_growth * 100).toFixed(1)}%`}
            color={p.scenario_best_growth >= 0 ? "text-emerald-600" : "text-red-500"}
            mono
          />
        )}
        {p.growth_trend_acceleration != null && (
          <StatCard
            label="Trend Acceleration"
            value={`${p.growth_trend_acceleration >= 0 ? "+" : ""}${(p.growth_trend_acceleration * 100).toFixed(2)}%`}
            color={p.growth_trend_acceleration >= 0 ? "text-emerald-600" : "text-red-500"}
            mono
          />
        )}
      </div>

      {/* Growth horizons inline */}
      {hasGrowth && (
        <div className="border-t border-gray-200 dark:border-gray-700 pt-4 mb-4">
          <p className="ia-caption uppercase tracking-wider mb-3">Growth Horizons</p>
          <div className="grid grid-cols-3 gap-3">
            <GrowthHorizon label="Short-Term" value={p.growth_short} />
            <GrowthHorizon label="Mid-Term" value={p.growth_mid} />
            <GrowthHorizon label="Long-Term" value={p.growth_long} />
          </div>
        </div>
      )}

      {/* Signal conflicts summary */}
      {hasConflicts && (
        <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
          <div className="flex items-center gap-3 mb-2">
            <p className="ia-caption uppercase tracking-wider">Signal Conflicts</p>
            <Badge variant="warning">{p.signal_conflict_count} conflict{p.signal_conflict_count !== 1 ? "s" : ""}</Badge>
          </div>
          {p.signal_conflict_warnings && p.signal_conflict_warnings.length > 0 && (
            <ul className="space-y-1">
              {p.signal_conflict_warnings.slice(0, 5).map((w, i) => (
                <li key={i} className="text-xs text-gray-600 dark:text-gray-400 flex gap-2">
                  <span className="text-amber-500 shrink-0">!</span>
                  <span className="line-clamp-2">{w}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
};

const StatCard: FC<{
  label: string;
  value: string;
  color?: string;
  capitalize?: boolean;
  mono?: boolean;
}> = ({ label, value, color, capitalize, mono }) => (
  <div className="ia-card-inline p-3">
    <p className="ia-caption uppercase tracking-wider text-[10px] mb-1">{label}</p>
    <p className={`font-semibold ${mono ? "ia-mono" : ""} ${capitalize ? "capitalize" : ""} ${color ?? "text-gray-900 dark:text-gray-100"}`}>
      {value}
    </p>
  </div>
);

const GrowthHorizon: FC<{ label: string; value: number | null }> = ({ label, value }) => {
  if (value == null) return null;
  const pct = value * 100;
  const positive = pct >= 0;
  return (
    <div className="ia-card-inline p-3 text-center">
      <p className="ia-caption uppercase tracking-wider text-[10px] mb-1">{label}</p>
      <p className={`text-lg font-bold ia-mono ${positive ? "text-emerald-600" : "text-red-500"}`}>
        {positive ? "+" : ""}{pct.toFixed(1)}%
      </p>
      <div className="mt-1.5 h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${positive ? "bg-emerald-500" : "bg-red-500"}`}
          style={{ width: `${Math.min(100, Math.abs(pct))}%`, marginLeft: positive ? 0 : "auto" }}
        />
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
        {Math.round(item.impact_score)}
      </span>
    </div>
  </div>
);
