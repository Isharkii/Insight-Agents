import type { FC } from "react";

interface HealthIndexCardProps {
  score: number;
  label: string;
  entityName: string;
}

function scoreColor(score: number): string {
  if (score >= 80) return "text-emerald-600";
  if (score >= 60) return "text-amber-600";
  return "text-rose-600";
}

function ringColor(score: number): string {
  if (score >= 80) return "stroke-emerald-500";
  if (score >= 60) return "stroke-amber-500";
  return "stroke-rose-500";
}

const HealthIndexCard: FC<HealthIndexCardProps> = ({
  score,
  label,
  entityName,
}) => {
  const clampedScore = Math.max(0, Math.min(100, score));
  const circumference = 2 * Math.PI * 54;
  const offset = circumference - (clampedScore / 100) * circumference;

  return (
    <div className="ia-surface h-full p-8">
      <h2 className="ia-label mb-4 text-center">Enterprise Health Index</h2>

      <div className="relative mx-auto h-36 w-36">
        <svg viewBox="0 0 120 120" className="h-full w-full -rotate-90">
          <circle
            cx="60"
            cy="60"
            r="54"
            fill="none"
            strokeWidth="8"
            className="stroke-slate-200"
          />
          <circle
            cx="60"
            cy="60"
            r="54"
            fill="none"
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            className={`${ringColor(clampedScore)} transition-all duration-700 ease-out`}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`text-4xl font-bold tabular-nums ${scoreColor(clampedScore)}`}>
            {clampedScore}
          </span>
          <span className="text-xs text-slate-500">/100</span>
        </div>
      </div>

      <div className="mt-4 text-center">
        <p className={`text-lg font-semibold ${scoreColor(clampedScore)}`}>{label}</p>
        <p className="mt-1 text-sm text-slate-500">{entityName}</p>
      </div>
    </div>
  );
};

export default HealthIndexCard;
