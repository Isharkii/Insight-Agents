import type { FC } from "react";
import type { InsightDimension, StructuredInsight } from "./types";

interface KeyInsightsProps {
  insights: StructuredInsight[];
}

const DIMENSION_STYLES: Record<InsightDimension, string> = {
  kpi: "bg-sky-100 text-sky-800",
  macro: "bg-indigo-100 text-indigo-800",
  competitive: "bg-orange-100 text-orange-800",
  simulation: "bg-violet-100 text-violet-800",
};

function impactColor(score: number): string {
  if (score >= 75) return "bg-emerald-500";
  if (score >= 50) return "bg-amber-500";
  return "bg-rose-500";
}

const KeyInsights: FC<KeyInsightsProps> = ({ insights }) => {
  if (insights.length === 0) return null;

  return (
    <div className="ia-surface p-6">
      <p className="ia-label mb-5">Key Insights</p>
      <div className="space-y-4">
        {insights.map((item, index) => (
          <div
            key={index}
            className="rounded-xl border border-slate-200 bg-white/75 p-5"
          >
            <div className="mb-3 flex items-start justify-between gap-4">
              <p className="text-base font-semibold leading-snug text-slate-900">{item.title}</p>
              <span
                className={`shrink-0 inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold ${DIMENSION_STYLES[item.dimension]}`}
              >
                {item.dimension}
              </span>
            </div>

            <p className="mb-4 text-sm text-slate-700">{item.description}</p>

            <div className="flex items-center gap-3">
              <span className="text-xs font-medium text-slate-500">Impact</span>
              <div className="h-2 max-w-xs flex-1 overflow-hidden rounded-full bg-slate-200">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${impactColor(item.impact_score)}`}
                  style={{ width: `${item.impact_score}%` }}
                />
              </div>
              <span className="text-xs font-medium tabular-nums text-slate-600">{item.impact_score}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default KeyInsights;
