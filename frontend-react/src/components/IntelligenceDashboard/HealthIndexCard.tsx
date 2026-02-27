import type { FC } from "react";

interface HealthIndexCardProps {
  score: number;
  label: string;
  entityName: string;
}

function scoreColor(score: number): string {
  if (score >= 80) return "text-emerald-600 dark:text-emerald-400";
  if (score >= 60) return "text-amber-600 dark:text-amber-400";
  return "text-red-600 dark:text-red-400";
}

function ringColor(score: number): string {
  if (score >= 80) return "stroke-emerald-500";
  if (score >= 60) return "stroke-amber-500";
  return "stroke-red-500";
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
    <div className="bg-white dark:bg-gray-900 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-800 p-8 flex flex-col items-center gap-4">
      <h2 className="text-sm font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">
        Enterprise Health Index
      </h2>

      <div className="relative w-36 h-36">
        <svg viewBox="0 0 120 120" className="w-full h-full -rotate-90">
          <circle
            cx="60"
            cy="60"
            r="54"
            fill="none"
            strokeWidth="8"
            className="stroke-gray-200 dark:stroke-gray-700"
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
          <span className="text-xs text-gray-500 dark:text-gray-400">/100</span>
        </div>
      </div>

      <div className="text-center">
        <p className={`text-lg font-semibold ${scoreColor(clampedScore)}`}>{label}</p>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">{entityName}</p>
      </div>
    </div>
  );
};

export default HealthIndexCard;
