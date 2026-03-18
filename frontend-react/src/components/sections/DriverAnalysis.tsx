import type { FC } from "react";
import {
  ResponsiveContainer,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
} from "recharts";
import { SectionHeader } from "../ui";
import ConfidenceWaterfall from "../charts/ConfidenceWaterfall";
import type { PipelineSignalConflicts, PipelineSignals } from "../../api/client";

interface DriverAnalysisProps {
  signals: PipelineSignals;
  overallConfidence: number;
}

/** Section 4: Signal integrity radar + Confidence waterfall + Conflict panel. */
const DriverAnalysis: FC<DriverAnalysisProps> = ({
  signals,
  overallConfidence,
}) => {
  const integrity = signals.signal_integrity;
  const conflicts = signals.signal_conflicts;

  const hasIntegrity = !!integrity;
  const hasConflicts = conflicts && (conflicts.conflict_count > 0 || (conflicts.warnings?.length ?? 0) > 0);

  if (!hasIntegrity && !hasConflicts) return null;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Confidence decomposition waterfall */}
        <ConfidenceWaterfall
          datasetConfidence={signals.dataset_confidence ?? null}
          riskConfidence={signals.risk?.confidence ?? 0}
          growthConfidence={signals.growth?.confidence ?? 0}
          forecastConfidence={signals.forecast?.confidence ?? 0}
          cohortConfidence={signals.cohort?.confidence ?? 0}
          overallConfidence={overallConfidence}
        />

        {/* Signal integrity radar */}
        {hasIntegrity && <IntegrityRadar integrity={integrity!} />}
      </div>

      {/* Signal conflicts */}
      {hasConflicts && <ConflictsPanel conflicts={conflicts!} />}
    </div>
  );
};

export default DriverAnalysis;

// ── Sub-components (private to this section) ─────────────────────────

const IntegrityRadar: FC<{ integrity: Record<string, unknown> }> = ({ integrity }) => {
  const scores = (integrity.signal_scores ?? integrity) as Record<string, unknown>;
  const data = [
    { signal: "KPI", score: Number(scores.KPI_score ?? scores.kpi_score ?? 0) * 100 },
    { signal: "Forecast", score: Number(scores.Forecast_score ?? scores.forecast_score ?? 0) * 100 },
    { signal: "Competitive", score: Number(scores.Competitive_score ?? scores.competitive_score ?? 0) * 100 },
    { signal: "Cohort", score: Number(scores.Cohort_score ?? scores.cohort_score ?? 0) * 100 },
    { signal: "Segmentation", score: Number(scores.Segmentation_score ?? scores.segmentation_score ?? 0) * 100 },
  ];
  const unified = Number(scores.Unified_integrity_score ?? scores.unified_integrity_score ?? 0) * 100;

  // Threshold ring data (synthesis gate at 40%)
  const thresholdData = data.map((d) => ({ ...d, threshold: 40 }));

  return (
    <div className="ia-card p-5">
      <SectionHeader
        title="Signal Integrity"
        subtitle="Quality per signal category (0–100%)"
        action={
          <span className="ia-mono text-xs font-semibold text-blue-600">
            Unified: {unified.toFixed(0)}%
          </span>
        }
      />
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={thresholdData} cx="50%" cy="50%" outerRadius="70%">
            <PolarGrid stroke="#e5e7eb" />
            <PolarAngleAxis dataKey="signal" tick={{ fontSize: 11, fill: "#6b7280" }} />
            <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 10 }} />
            {/* Threshold ring at synthesis gate (40%) */}
            <Radar
              dataKey="threshold"
              stroke="#ef4444"
              strokeWidth={1}
              strokeDasharray="4 4"
              fill="none"
              strokeOpacity={0.5}
            />
            {/* Actual scores */}
            <Radar
              dataKey="score"
              stroke="var(--chart-1)"
              fill="var(--chart-1)"
              fillOpacity={0.2}
              strokeWidth={2}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

const ConflictsPanel: FC<{ conflicts: PipelineSignalConflicts }> = ({ conflicts }) => {
  const sev = conflicts.total_severity ?? 0;
  return (
    <div className="ia-card-alert p-5">
      <SectionHeader
        title="Signal Conflicts"
        action={
          <div className="flex items-center gap-3 text-xs">
            <span className="font-semibold text-amber-700 dark:text-amber-300">
              {conflicts.conflict_count} conflict{conflicts.conflict_count !== 1 ? "s" : ""}
            </span>
            <span className="ia-caption">Severity: {sev.toFixed(2)}</span>
          </div>
        }
      />
      {conflicts.warnings && conflicts.warnings.length > 0 && (
        <ul className="space-y-1.5">
          {conflicts.warnings.map((w, i) => (
            <li key={i} className="text-sm text-gray-700 dark:text-gray-300 flex gap-2">
              <span className="text-amber-500 shrink-0">!</span>
              {w}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};
