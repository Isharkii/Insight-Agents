import type { FC } from "react";
import type { InsightDimension, StructuredInsight } from "./types";

interface KeyInsightsProps {
  insights: StructuredInsight[];
}

const DIMENSION_STYLES: Record<InsightDimension, string> = {
  kpi: "bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-300",
  macro: "bg-purple-100 text-purple-800 dark:bg-purple-900/40 dark:text-purple-300",
  competitive: "bg-orange-100 text-orange-800 dark:bg-orange-900/40 dark:text-orange-300",
  simulation: "bg-violet-100 text-violet-800 dark:bg-violet-900/40 dark:text-violet-300",
};

function impactColor(score: number): string {
  if (score >= 75) return "bg-emerald-500";
  if (score >= 50) return "bg-amber-500";
  return "bg-red-500";
}

const KeyInsights: FC<KeyInsightsProps> = ({ insights }) => {
  if (insights.length === 0) return null;

  return (
    <div className="bg-white dark:bg-gray-900 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-800 p-6">
      <h3 className="text-sm font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-5">
        Key Insights
      </h3>
      <div className="space-y-4">
        {insights.map((item, index) => (
          <div
            key={index}
            className="border border-gray-200 dark:border-gray-700 rounded-xl p-5"
          >
            <div className="flex items-start justify-between gap-4 mb-3">
              <p className="text-base font-semibold text-gray-900 dark:text-gray-100 leading-snug">
                {item.title}
              </p>
              <span
                className={`shrink-0 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold ${DIMENSION_STYLES[item.dimension]}`}
              >
                {item.dimension}
              </span>
            </div>

            <p className="text-sm text-gray-700 dark:text-gray-300 mb-4">
              {item.description}
            </p>

            <div className="flex items-center gap-3">
              <span className="text-xs font-medium text-gray-500 dark:text-gray-400">
                Impact
              </span>
              <div className="flex-1 max-w-xs h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${impactColor(item.impact_score)}`}
                  style={{ width: `${item.impact_score}%` }}
                />
              </div>
              <span className="text-xs font-medium tabular-nums text-gray-600 dark:text-gray-300">
                {item.impact_score}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default KeyInsights;
