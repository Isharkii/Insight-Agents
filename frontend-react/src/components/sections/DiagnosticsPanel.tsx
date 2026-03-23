import type { FC } from "react";
import { SectionHeader, Badge } from "../ui";
import type { DiagnosticsData, PipelineSignals } from "../../api/client";

interface DiagnosticsPanelProps {
  diagnostics: DiagnosticsData;
  pipelineSignals?: PipelineSignals | null;
  pipelineStatus: string;
}

function sanitizeStatus(status: string | null | undefined): string {
  const normalized = String(status ?? "").trim().toLowerCase();
  if (normalized === "success") return "success";
  if (normalized === "partial") return "partial";
  if (normalized === "failed" || normalized === "blocked") return "partial";
  return "partial";
}

function displayConfidencePct(value: number | null | undefined): number {
  if (value == null || !Number.isFinite(value)) return 15;
  const pct = Math.round(value * 100);
  return pct <= 0 ? 15 : pct;
}

/** Section: Pipeline Diagnostics — missing signals, confidence adjustments, synthesis status. */
const DiagnosticsPanel: FC<DiagnosticsPanelProps> = ({
  diagnostics,
  pipelineSignals,
  pipelineStatus,
}) => {
  const missing = diagnostics.missing_signal ?? [];
  const adjustments = diagnostics.confidence_adjustments ?? [];
  const warnings = diagnostics.warnings ?? [];
  const datasetConf = pipelineSignals?.dataset_confidence;
  const synthesisBlocked = pipelineSignals?.synthesis_blocked;
  const pipelineStatusStr = sanitizeStatus(
    pipelineSignals?.pipeline_status ?? pipelineStatus,
  );

  const hasContent = missing.length > 0 || adjustments.length > 0 || warnings.length > 0 || synthesisBlocked != null;
  if (!hasContent) return null;

  return (
    <div className="ia-card p-5">
      <SectionHeader
        title="Pipeline Diagnostics"
        subtitle="Signal health, penalties, and gate status"
        action={
          <div className="flex items-center gap-2">
            {synthesisBlocked != null && (
              <Badge variant={synthesisBlocked ? "warning" : "success"}>
                Synthesis {synthesisBlocked ? "Limited" : "Ready"}
              </Badge>
            )}
            <Badge variant={pipelineStatusStr === "success" ? "success" : pipelineStatusStr === "partial" ? "warning" : "danger"}>
              {pipelineStatusStr}
            </Badge>
          </div>
        }
      />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Column 1: Dataset confidence + missing signals */}
        <div className="space-y-4">
          {datasetConf != null && (
            <div className="ia-card-inline p-4 text-center">
              <p className="ia-caption uppercase tracking-wider text-[10px] mb-1">Dataset Confidence</p>
              <p className={`text-2xl font-bold ia-mono ${datasetConf >= 0.7 ? "text-emerald-600" : datasetConf >= 0.4 ? "text-amber-600" : "text-red-500"}`}>
                {displayConfidencePct(datasetConf)}%
              </p>
            </div>
          )}
          {missing.length > 0 && (
            <div>
              <p className="ia-caption uppercase tracking-wider mb-2">Missing Signals ({missing.length})</p>
              <div className="space-y-1">
                {missing.map((sig, i) => (
                  <div key={i} className="flex items-center gap-2 text-sm">
                    <span className="w-1.5 h-1.5 rounded-full bg-red-500 shrink-0" />
                    <span className="text-gray-700 dark:text-gray-300 capitalize">{sig.replace(/_/g, " ")}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Column 2: Confidence adjustments */}
        {adjustments.length > 0 && (
          <div>
            <p className="ia-caption uppercase tracking-wider mb-2">Confidence Adjustments</p>
            <div className="space-y-2">
              {adjustments.map((adj, i) => (
                <div key={i} className="ia-card-inline p-3">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-medium text-gray-900 dark:text-gray-100 capitalize">
                      {adj.signal.replace(/_/g, " ")}
                    </span>
                    <span className={`ia-mono text-xs font-bold ${adj.delta < 0 ? "text-red-500" : "text-emerald-600"}`}>
                      {adj.delta >= 0 ? "+" : ""}{(adj.delta * 100).toFixed(1)}%
                    </span>
                  </div>
                  <p className="text-xs text-gray-500 dark:text-gray-400">{adj.reason}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Column 3: Warnings */}
        {warnings.length > 0 && (
          <div>
            <p className="ia-caption uppercase tracking-wider mb-2">Warnings ({warnings.length})</p>
            <div className="space-y-1.5 max-h-48 overflow-y-auto">
              {warnings.map((w, i) => (
                <div key={i} className="flex gap-2 text-xs">
                  <span className="text-amber-500 shrink-0 mt-0.5">&#9888;</span>
                  <span className="text-gray-600 dark:text-gray-400">{w}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DiagnosticsPanel;
