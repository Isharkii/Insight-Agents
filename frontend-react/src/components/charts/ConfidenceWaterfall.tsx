import type { FC } from "react";
import WaterfallChart from "./WaterfallChart";
import SectionHeader from "../ui/SectionHeader";

interface ConfidenceWaterfallProps {
  datasetConfidence: number | null;
  riskConfidence: number;
  growthConfidence: number;
  forecastConfidence: number;
  cohortConfidence: number;
  overallConfidence: number;
}

/**
 * Shows how confidence flows and degrades through the pipeline.
 * Renders as a waterfall: each stage's contribution = its value minus the previous stage.
 */
const ConfidenceWaterfall: FC<ConfidenceWaterfallProps> = ({
  datasetConfidence,
  riskConfidence,
  growthConfidence,
  forecastConfidence,
  cohortConfidence,
  overallConfidence,
}) => {
  // Build items as deltas from a 100% starting point
  const stages = [
    { label: "Dataset", value: (datasetConfidence ?? 1) * 100 },
    { label: "Risk", value: riskConfidence * 100 },
    { label: "Growth", value: growthConfidence * 100 },
    { label: "Forecast", value: forecastConfidence * 100 },
    { label: "Cohort", value: cohortConfidence * 100 },
  ];

  // Convert absolute confidences to relative contributions
  const avg = stages.reduce((sum, s) => sum + s.value, 0) / stages.length;
  const items = stages.map((s) => ({
    label: s.label,
    value: s.value - avg, // deviation from mean
  }));

  // Add overall as a positive "result" item
  items.push({ label: "Overall", value: overallConfidence * 100 });

  return (
    <div className="ia-card p-5">
      <SectionHeader
        title="Confidence Decomposition"
        subtitle="How each pipeline stage contributes to overall confidence"
        action={
          <span className="ia-mono text-xs font-semibold" style={{
            color: overallConfidence >= 0.6 ? "var(--ia-success)" : overallConfidence >= 0.4 ? "var(--ia-warning)" : "var(--ia-danger)",
          }}>
            {Math.round(overallConfidence * 100)}%
          </span>
        }
      />
      <WaterfallChart
        items={items}
        height={220}
        formatter={(v) => `${v >= 0 ? "+" : ""}${v.toFixed(0)}%`}
      />
    </div>
  );
};

export default ConfidenceWaterfall;
