import { useState, type FC } from "react";
import type { AnalyzeResult } from "../api/client";

interface AnalysisResultsProps {
  result: AnalyzeResult;
  executionTime?: number;
}

const PRIORITY_STYLES: Record<string, string> = {
  low: "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300",
  medium: "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300",
  high: "bg-orange-100 text-orange-800 dark:bg-orange-900/40 dark:text-orange-300",
  critical:
    "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300",
};

const STATUS_STYLES: Record<string, { bg: string; dot: string; label: string }> = {
  success: {
    bg: "bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-800",
    dot: "bg-emerald-500",
    label: "Success",
  },
  partial: {
    bg: "bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-800",
    dot: "bg-amber-500",
    label: "Partial",
  },
  failed: {
    bg: "bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800",
    dot: "bg-red-500",
    label: "Failed",
  },
};

function confidenceColor(score: number): string {
  if (score >= 0.8) return "bg-emerald-500";
  if (score >= 0.6) return "bg-amber-500";
  return "bg-red-500";
}

const AnalysisResults: FC<AnalysisResultsProps> = ({
  result,
  executionTime,
}) => {
  const [showDiagnostics, setShowDiagnostics] = useState(false);
  const pct = Math.round(result.confidence_score * 100);
  const statusCfg = STATUS_STYLES[result.pipeline_status] ?? STATUS_STYLES.success;
  const priorityStyle =
    PRIORITY_STYLES[result.priority] ?? PRIORITY_STYLES.medium;

  return (
    <div className="bg-white dark:bg-gray-900 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-800 overflow-hidden">
      {/* Header bar */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100 dark:border-gray-800">
        <h3 className="text-base font-semibold text-gray-900 dark:text-gray-100">
          Analysis Results
        </h3>
        <div className="flex items-center gap-3">
          {executionTime !== undefined && (
            <span className="text-xs text-gray-400">
              {executionTime.toFixed(1)}s
            </span>
          )}
          <span
            className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${statusCfg.bg}`}
          >
            <span className={`w-1.5 h-1.5 rounded-full ${statusCfg.dot}`} />
            {statusCfg.label}
          </span>
        </div>
      </div>

      <div className="p-6 space-y-5">
        {/* Insight */}
        <div>
          <label className="block text-xs font-medium uppercase tracking-wider text-gray-400 mb-1">
            Insight
          </label>
          <p className="text-base text-gray-900 dark:text-gray-100 leading-relaxed">
            {result.insight}
          </p>
        </div>

        {/* Evidence */}
        <div>
          <label className="block text-xs font-medium uppercase tracking-wider text-gray-400 mb-1">
            Evidence
          </label>
          <p className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
            {result.evidence}
          </p>
        </div>

        {/* Impact */}
        <div>
          <label className="block text-xs font-medium uppercase tracking-wider text-gray-400 mb-1">
            Impact
          </label>
          <p className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
            {result.impact}
          </p>
        </div>

        {/* Recommended Action */}
        <div className="bg-blue-50 dark:bg-blue-900/10 border border-blue-200 dark:border-blue-800 rounded-xl p-4">
          <label className="block text-xs font-medium uppercase tracking-wider text-blue-600 dark:text-blue-400 mb-1">
            Recommended Action
          </label>
          <p className="text-sm text-blue-900 dark:text-blue-200 leading-relaxed">
            {result.recommended_action}
          </p>
        </div>

        {/* Priority + Confidence row */}
        <div className="flex items-center gap-6">
          <div>
            <label className="block text-xs font-medium uppercase tracking-wider text-gray-400 mb-1">
              Priority
            </label>
            <span
              className={`inline-block px-3 py-1 rounded-full text-xs font-semibold capitalize ${priorityStyle}`}
            >
              {result.priority}
            </span>
          </div>

          <div className="flex-1">
            <label className="block text-xs font-medium uppercase tracking-wider text-gray-400 mb-1">
              Confidence
            </label>
            <div className="flex items-center gap-3">
              <div className="flex-1 h-2.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-700 ${confidenceColor(result.confidence_score)}`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span className="text-sm font-semibold tabular-nums text-gray-700 dark:text-gray-300">
                {pct}%
              </span>
            </div>
          </div>
        </div>

        {/* Diagnostics */}
        {result.diagnostics && (
          <div>
            <button
              onClick={() => setShowDiagnostics(!showDiagnostics)}
              className="flex items-center gap-1.5 text-xs font-medium text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors"
            >
              <svg
                className={`w-3.5 h-3.5 transition-transform ${showDiagnostics ? "rotate-90" : ""}`}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M9 5l7 7-7 7"
                />
              </svg>
              Pipeline Diagnostics
            </button>
            {showDiagnostics && (
              <pre className="mt-2 p-4 rounded-lg bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-xs text-gray-700 dark:text-gray-300 overflow-x-auto max-h-64">
                {JSON.stringify(result.diagnostics, null, 2)}
              </pre>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default AnalysisResults;
