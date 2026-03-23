import type { FC } from "react";
import { SectionHeader, Badge } from "../ui";
import type {
  BusinessIntelligenceResponse,
  BIEmergingSignal,
  BIZone,
  BIStrategyAction,
  BIRiskMitigation,
  BISignalReference,
  BIPipelineStageResult,
} from "../../api/client";

interface BusinessIntelligenceProps {
  biData: BusinessIntelligenceResponse;
}

function relevanceVariant(r: string): "success" | "warning" | "info" {
  if (r === "high") return "success";
  if (r === "medium") return "warning";
  return "info";
}

function priorityVariant(p: string): "danger" | "warning" | "info" {
  if (p === "critical") return "danger";
  if (p === "high") return "warning";
  return "info";
}

/** Section: Business Intelligence — emerging signals, opportunity/risk zones, strategy actions. */
const BusinessIntelligence: FC<BusinessIntelligenceProps> = ({ biData }) => {
  const insights = biData.insights;
  const strategy = biData.strategy;
  const context = biData.context;
  const pipeline = biData.pipeline ?? [];
  const warnings = biData.warnings ?? [];

  const hasInsights = insights && (
    insights.emerging_signals?.length > 0 ||
    insights.opportunity_zones?.length > 0 ||
    insights.risk_zones?.length > 0
  );
  const hasStrategy = strategy && (
    strategy.short_term_actions?.length > 0 ||
    strategy.mid_term_actions?.length > 0 ||
    strategy.risk_mitigation?.length > 0
  );
  const hasContext = !!context;
  const hasPipeline = pipeline.length > 0;
  const hasWarnings = warnings.length > 0;

  if (!hasInsights && !hasStrategy && !hasContext && !hasPipeline && !hasWarnings) return null;

  return (
    <div className="space-y-6">
      {/* Context bar */}
      {context && (
        <div className="ia-card p-5">
          <SectionHeader
            title="Business Context"
            action={
              <div className="flex items-center gap-2">
                {insights?.momentum_score != null && (
                  <span className="ia-mono text-xs font-semibold text-blue-600">
                    Momentum: {insights.momentum_score.toFixed(0)}
                  </span>
                )}
                {biData.confidence > 0 && (
                  <span className="ia-caption">
                    BI Confidence: {Math.round(biData.confidence * 100)}%
                  </span>
                )}
              </div>
            }
          />
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm">
            {context.industry && (
              <ContextItem label="Industry" value={context.industry} />
            )}
            {context.business_model && (
              <ContextItem label="Business Model" value={context.business_model} />
            )}
            {context.target_market && (
              <ContextItem label="Target Market" value={context.target_market} />
            )}
          </div>
          {context.risk_factors?.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-1.5">
              {context.risk_factors.map((f, i) => (
                <span key={i} className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-red-50 text-red-600 dark:bg-red-900/20 dark:text-red-300 border border-red-200 dark:border-red-800">
                  {f}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Macro Summary */}
      {insights?.macro_summary && (
        <div className="ia-card p-5">
          <SectionHeader title="Macro Environment" />
          <p className="ia-body">{insights.macro_summary}</p>
        </div>
      )}

      {/* Fallback macro context when synthesized macro summary is unavailable */}
      {!insights?.macro_summary && context && (
        <div className="ia-card p-5">
          <SectionHeader title="Macro Signals (Data-Driven)" />
          {context.macro_dependencies?.length > 0 ? (
            <div className="mb-3">
              <p className="ia-caption uppercase tracking-wider mb-2">Macro Dependencies</p>
              <div className="flex flex-wrap gap-1.5">
                {context.macro_dependencies.map((item, i) => (
                  <span key={i} className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-blue-50 text-blue-700 dark:bg-blue-900/20 dark:text-blue-300 border border-blue-200 dark:border-blue-800">
                    {item}
                  </span>
                ))}
              </div>
            </div>
          ) : null}
          {context.search_intents?.length > 0 ? (
            <div>
              <p className="ia-caption uppercase tracking-wider mb-2">Tracked Themes</p>
              <ul className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
                {context.search_intents.slice(0, 5).map((item, i) => (
                  <li key={i}>• {item}</li>
                ))}
              </ul>
            </div>
          ) : (
            <p className="ia-body">Macro context is limited; expand prompt context to improve external signal coverage.</p>
          )}
        </div>
      )}

      {/* Emerging Signals + Risk/Opportunity side by side */}
      {hasInsights && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Emerging Signals */}
          {insights!.emerging_signals?.length > 0 && (
            <div className="ia-card p-5">
              <SectionHeader title="Emerging Signals" subtitle="Early indicators detected from data" />
              <div className="space-y-3">
                {insights!.emerging_signals.map((sig, i) => (
                  <EmergingSignalCard key={i} signal={sig} />
                ))}
              </div>
            </div>
          )}

          {/* Opportunity + Risk Zones stacked */}
          <div className="space-y-6">
            {insights!.opportunity_zones?.length > 0 && (
              <div className="ia-card p-5">
                <SectionHeader title="Opportunity Zones" />
                <div className="space-y-3">
                  {insights!.opportunity_zones.map((zone, i) => (
                    <ZoneCard key={i} zone={zone} variant="opportunity" />
                  ))}
                </div>
              </div>
            )}
            {insights!.risk_zones?.length > 0 && (
              <div className="ia-card p-5">
                <SectionHeader title="Risk Zones" />
                <div className="space-y-3">
                  {insights!.risk_zones.map((zone, i) => (
                    <ZoneCard key={i} zone={zone} variant="risk" />
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Strategy Actions */}
      {hasStrategy && (
        <div className="ia-card p-5">
          <SectionHeader
            title="Strategic Playbook"
            subtitle="Actionable strategy with supporting signals"
            action={
              strategy!.confidence > 0 ? (
                <span className="ia-caption">
                  Strategy Confidence: {Math.round(strategy!.confidence * 100)}%
                </span>
              ) : undefined
            }
          />

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Short-term */}
            {strategy!.short_term_actions?.length > 0 && (
              <ActionList title="Short-Term Actions" actions={strategy!.short_term_actions} />
            )}

            {/* Mid-term */}
            {strategy!.mid_term_actions?.length > 0 && (
              <ActionList title="Mid-Term Actions" actions={strategy!.mid_term_actions} />
            )}
          </div>

          {/* Long-term positioning */}
          {strategy!.long_term_positioning && (
            <div className="mt-4 bg-blue-50 dark:bg-blue-900/10 border border-blue-200 dark:border-blue-800 rounded-xl p-4">
              <p className="ia-caption uppercase tracking-wider text-blue-600 mb-1">Long-Term Positioning</p>
              <p className="text-sm text-blue-900 dark:text-blue-100">{strategy!.long_term_positioning}</p>
            </div>
          )}

          {/* Competitive angle */}
          {strategy!.competitive_angle?.positioning && (
            <div className="mt-4 bg-violet-50 dark:bg-violet-900/10 border border-violet-200 dark:border-violet-800 rounded-xl p-4">
              <p className="ia-caption uppercase tracking-wider text-violet-600 mb-1">Competitive Positioning</p>
              <p className="text-sm text-violet-900 dark:text-violet-100 mb-2">{strategy!.competitive_angle.positioning}</p>
              {strategy!.competitive_angle.differentiation && (
                <>
                  <p className="ia-caption uppercase tracking-wider text-violet-600 mb-1">Differentiation</p>
                  <p className="text-sm text-violet-900 dark:text-violet-100">{strategy!.competitive_angle.differentiation}</p>
                </>
              )}
            </div>
          )}

          {/* Risk mitigation */}
          {strategy!.risk_mitigation?.length > 0 && (
            <div className="mt-4">
              <p className="ia-caption uppercase tracking-wider mb-3">Risk Mitigations</p>
              <div className="space-y-2">
                {strategy!.risk_mitigation.map((rm, i) => (
                  <RiskMitigationRow key={i} item={rm} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Fallback strategic/risk guidance when strategy synthesis is unavailable */}
      {!hasStrategy && (hasWarnings || (context?.risk_factors?.length ?? 0) > 0) && (
        <div className="ia-card p-5">
          <SectionHeader title="Strategy & Risk Signals" subtitle="Fallback guidance from available internal and macro context" />
          {(context?.risk_factors?.length ?? 0) > 0 && (
            <div className="mb-3">
              <p className="ia-caption uppercase tracking-wider mb-2">Risk Factors</p>
              <div className="flex flex-wrap gap-1.5">
                {context!.risk_factors.map((rf, i) => (
                  <span key={i} className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-amber-50 text-amber-700 dark:bg-amber-900/20 dark:text-amber-300 border border-amber-200 dark:border-amber-800">
                    {rf}
                  </span>
                ))}
              </div>
            </div>
          )}
          {hasWarnings && (
            <div>
              <p className="ia-caption uppercase tracking-wider mb-2">Execution Warnings</p>
              <ul className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
                {warnings.slice(0, 5).map((w, i) => (
                  <li key={i}>• {w}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Stage-level visibility for news/trends/search coverage */}
      {hasPipeline && (
        <div className="ia-card p-5">
          <SectionHeader title="Coverage Pipeline" subtitle="Stage-level status for macro/news/trend coverage" />
          <div className="space-y-2">
            {pipeline.map((stage, i) => (
              <PipelineStageRow key={i} stage={stage} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default BusinessIntelligence;

// ── Sub-components ───────────────────────────────────────────────────

const ContextItem: FC<{ label: string; value: string }> = ({ label, value }) => (
  <div className="ia-card-inline p-3">
    <p className="ia-caption uppercase tracking-wider text-[10px] mb-1">{label}</p>
    <p className="font-medium text-gray-900 dark:text-gray-100 capitalize">{value}</p>
  </div>
);

const SignalPills: FC<{ signals: BISignalReference[] }> = ({ signals }) => {
  if (signals.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-1 mt-2">
      {signals.slice(0, 4).map((s, i) => (
        <span key={i} className="px-1.5 py-0.5 rounded text-[10px] ia-mono bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400">
          {s.metric_name}: {s.value}{s.unit ? ` ${s.unit}` : ""}
        </span>
      ))}
    </div>
  );
};

const EmergingSignalCard: FC<{ signal: BIEmergingSignal }> = ({ signal }) => (
  <div className="ia-card-inline p-3">
    <div className="flex items-start justify-between gap-2 mb-1">
      <p className="text-sm font-semibold text-gray-900 dark:text-gray-100">{signal.title}</p>
      <Badge variant={relevanceVariant(signal.relevance)}>{signal.relevance}</Badge>
    </div>
    <p className="text-xs text-gray-600 dark:text-gray-400">{signal.description}</p>
    <SignalPills signals={signal.supporting_signals} />
  </div>
);

const ZoneCard: FC<{ zone: BIZone; variant: "opportunity" | "risk" }> = ({ zone, variant }) => {
  const borderColor = variant === "opportunity"
    ? "border-l-emerald-500"
    : "border-l-red-500";
  return (
    <div className={`ia-card-inline p-3 border-l-4 ${borderColor}`}>
      <p className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-1">{zone.title}</p>
      <p className="text-xs text-gray-600 dark:text-gray-400">{zone.description}</p>
      <SignalPills signals={zone.supporting_signals} />
    </div>
  );
};

const ActionList: FC<{ title: string; actions: BIStrategyAction[] }> = ({ title, actions }) => (
  <div>
    <p className="ia-caption uppercase tracking-wider mb-2">{title}</p>
    <div className="space-y-2">
      {actions.map((a, i) => (
        <div key={i} className="ia-card-inline p-3">
          <div className="flex items-start justify-between gap-2 mb-1">
            <p className="text-sm font-medium text-gray-900 dark:text-gray-100">{a.action}</p>
            <Badge variant={priorityVariant(a.priority)}>{a.priority}</Badge>
          </div>
          <p className="text-xs text-gray-600 dark:text-gray-400">{a.rationale}</p>
          <SignalPills signals={a.supporting_signals} />
        </div>
      ))}
    </div>
  </div>
);

const RiskMitigationRow: FC<{ item: BIRiskMitigation }> = ({ item }) => (
  <div className="ia-card-inline p-3 border-l-4 border-l-amber-500">
    <p className="text-sm font-semibold text-gray-900 dark:text-gray-100">{item.risk_title}</p>
    <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">{item.mitigation}</p>
    <SignalPills signals={item.supporting_signals} />
  </div>
);

const PipelineStageRow: FC<{ stage: BIPipelineStageResult }> = ({ stage }) => {
  const variant =
    stage.status === "success" ? "success" :
    stage.status === "failed" ? "danger" :
    "warning";
  return (
    <div className="ia-card-inline p-3 flex items-start justify-between gap-3">
      <div>
        <p className="text-sm font-semibold text-gray-900 dark:text-gray-100 capitalize">{stage.stage.replace(/_/g, " ")}</p>
        {stage.error && (
          <p className="text-xs text-gray-600 dark:text-gray-400 mt-0.5">{stage.error}</p>
        )}
      </div>
      <div className="flex items-center gap-2 shrink-0">
        <span className="ia-caption">{Math.round(stage.duration_ms)}ms</span>
        <Badge variant={variant}>{stage.status}</Badge>
      </div>
    </div>
  );
};
