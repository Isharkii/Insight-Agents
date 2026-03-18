import type { FC } from "react";
import type { AnalyzeResult } from "../../api/client";
import { Badge, priorityVariant, statusVariant, Gauge } from "../ui";

interface HeroInsightProps {
  result: AnalyzeResult;
  riskScore?: number | null;
  riskLevel?: string | null;
  healthIndex?: number | null;
  healthLabel?: string | null;
}

const priorityBg: Record<string, string> = {
  critical: "bg-red-500",
  high: "bg-orange-500",
  medium: "bg-amber-500",
  low: "bg-gray-400",
};

const HeroInsight: FC<HeroInsightProps> = ({
  result,
  riskScore,
  healthIndex,
  healthLabel,
}) => {
  const pct = Math.round(result.confidence_score * 100);
  const hasRisk = riskScore != null;
  const hasHealth = healthIndex != null;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 ia-fade-up">
      {/* Main insight card — hero tier */}
      <div className="lg:col-span-7 ia-card-hero p-6 space-y-4">
        <div className="flex items-start gap-3">
          <div className={`w-1 shrink-0 self-stretch rounded-full ${priorityBg[result.priority] ?? "bg-gray-400"}`} />
          <div className="space-y-3 flex-1">
            <div>
              <Badge variant={priorityVariant(result.priority)} dot>
                {result.priority} priority
              </Badge>
              <h3 className="ia-heading mt-2 leading-snug">
                {result.insight}
              </h3>
            </div>
            <p className="ia-body">{result.evidence}</p>
            <div className="bg-blue-50 dark:bg-blue-900/10 border border-blue-200 dark:border-blue-800 rounded-lg p-3">
              <p className="ia-caption text-blue-500 mb-0.5 uppercase font-semibold tracking-wider">
                Recommended Action
              </p>
              <p className="text-sm text-blue-900 dark:text-blue-200">
                {result.recommended_action}
              </p>
            </div>
            <p className="ia-body">
              <span className="font-medium text-gray-500">Impact:</span> {result.impact}
            </p>
          </div>
        </div>
      </div>

      {/* Gauges column */}
      <div className="lg:col-span-5 grid grid-cols-2 gap-4">
        <div className="ia-card p-4 flex items-center justify-center">
          <Gauge value={pct} label="Confidence" />
        </div>

        {hasHealth ? (
          <div className="ia-card p-4 flex items-center justify-center">
            <Gauge value={healthIndex!} label={healthLabel || "Health"} />
          </div>
        ) : hasRisk ? (
          <div className="ia-card p-4 flex items-center justify-center">
            <Gauge value={riskScore!} label="Risk Score" invertColor />
          </div>
        ) : (
          <div className="ia-card p-4 flex flex-col items-center justify-center">
            <p className="ia-display">{pct}%</p>
            <p className="ia-caption mt-1">Pipeline Score</p>
          </div>
        )}

        {/* Show risk below health when both exist */}
        {hasHealth && hasRisk && (
          <div className="col-span-2 ia-card p-4 flex items-center justify-center">
            <Gauge value={riskScore!} label="Risk Score" invertColor />
          </div>
        )}

        {/* Pipeline status badge */}
        <div className="col-span-2 flex justify-end">
          <Badge variant={statusVariant(result.pipeline_status)} dot>
            {result.pipeline_status}
          </Badge>
        </div>
      </div>
    </div>
  );
};

export default HeroInsight;
