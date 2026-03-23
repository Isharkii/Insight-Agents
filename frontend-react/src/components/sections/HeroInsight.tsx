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

/** Strip "Conditional: " prefix from failure-mode text. */
function cleanConditional(text: string): string {
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

/** True only when synthesis was genuinely blocked (no LLM output at all).
 *  "Conditional:" prefix is a tone marker for low-confidence results,
 *  NOT a synthesis failure — those still carry real LLM-generated content. */
function isSynthesisBlocked(result: AnalyzeResult): boolean {
  const ps = result.pipeline_signals;
  // Explicit block signal from synthesis_gate
  if (ps?.synthesis_blocked === true) return true;
  // No insight text at all
  if (!(result.insight ?? "").trim()) return true;
  return false;
}

const HeroInsight: FC<HeroInsightProps> = ({
  result,
  riskScore,
  healthIndex,
  healthLabel,
}) => {
  const pct = Math.round(result.confidence_score * 100);
  const hasRisk = riskScore != null;
  const hasHealth = healthIndex != null;
  const isBlocked = isSynthesisBlocked(result);

  // When synthesis is blocked, show a clear alert instead of the normal hero
  if (isBlocked) {
    return (
      <div className="space-y-4 ia-fade-up">
        {/* Alert banner */}
        <div className="ia-card border-l-4 border-l-red-500 p-6">
          <div className="flex items-start gap-4">
            <div className="shrink-0 mt-0.5">
              <svg className="w-6 h-6 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
              </svg>
            </div>
            <div className="flex-1 space-y-3">
              <div className="flex items-center gap-2">
                <Badge variant="warning" dot>Limited Insight</Badge>
                <Badge variant="warning" dot>
                  partial
                </Badge>
              </div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                {cleanConditional(result.insight)}
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                {cleanConditional(result.evidence)}
              </p>

              {/* What needs to happen */}
              <div className="bg-amber-50 dark:bg-amber-900/10 border border-amber-200 dark:border-amber-800 rounded-lg p-3">
                <p className="ia-caption text-amber-600 mb-0.5 uppercase font-semibold tracking-wider">
                  Suggested Next Step
                </p>
                <p className="text-sm text-amber-900 dark:text-amber-100">
                  {cleanConditional(result.recommended_action)}
                </p>
              </div>

              <p className="text-sm text-gray-500 dark:text-gray-400">
                <span className="font-medium">Current Risk:</span> {cleanConditional(result.impact)}
              </p>
            </div>

            {/* Keep confidence display non-zero in limited-output mode */}
            <div className="shrink-0 w-24">
              <Gauge value={Math.max(15, pct)} label="Confidence" />
            </div>
          </div>
        </div>

        {/* Still show health/risk gauges if available */}
        {(hasHealth || hasRisk) && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {hasHealth && (
              <div className="ia-card p-4 flex items-center justify-center">
                <Gauge value={healthIndex!} label={healthLabel || "Health"} />
              </div>
            )}
            {hasRisk && (
              <div className="ia-card p-4 flex items-center justify-center">
                <Gauge value={riskScore!} label="Risk Score" invertColor />
              </div>
            )}
          </div>
        )}
      </div>
    );
  }

  // Normal hero insight
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
                {cleanConditional(result.insight)}
              </h3>
            </div>
            <p className="ia-body">{cleanConditional(result.evidence)}</p>
            <div className="bg-blue-50 dark:bg-blue-900/10 border border-blue-200 dark:border-blue-800 rounded-lg p-3">
              <p className="ia-caption text-blue-500 mb-0.5 uppercase font-semibold tracking-wider">
                Recommended Action
              </p>
              <p className="text-sm text-blue-900 dark:text-blue-200">
                {cleanConditional(result.recommended_action)}
              </p>
            </div>
            <p className="ia-body">
              <span className="font-medium text-gray-500">Impact:</span> {cleanConditional(result.impact)}
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

