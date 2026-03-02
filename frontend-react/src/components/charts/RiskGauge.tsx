import type { FC } from "react";

interface RiskGaugeProps {
  score: number;
  label?: string;
}

function gaugeColor(score: number): { ring: string; text: string } {
  if (score <= 30)
    return {
      ring: "stroke-emerald-500",
      text: "text-emerald-600 dark:text-emerald-400",
    };
  if (score <= 60)
    return {
      ring: "stroke-amber-500",
      text: "text-amber-600 dark:text-amber-400",
    };
  if (score <= 80)
    return {
      ring: "stroke-orange-500",
      text: "text-orange-600 dark:text-orange-400",
    };
  return {
    ring: "stroke-red-500",
    text: "text-red-600 dark:text-red-400",
  };
}

function riskLabel(score: number): string {
  if (score <= 30) return "Low";
  if (score <= 60) return "Moderate";
  if (score <= 80) return "High";
  return "Critical";
}

const RiskGauge: FC<RiskGaugeProps> = ({ score, label }) => {
  const clamped = Math.max(0, Math.min(100, score));
  const circumference = 2 * Math.PI * 54;
  const offset = circumference - (clamped / 100) * circumference;
  const colors = gaugeColor(clamped);

  return (
    <div className="bg-white dark:bg-gray-900 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-800 p-6 flex flex-col items-center gap-3">
      <h3 className="text-sm font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">
        Risk Score
      </h3>

      <div className="relative w-28 h-28">
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
            className={`${colors.ring} transition-all duration-700 ease-out`}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`text-2xl font-bold tabular-nums ${colors.text}`}>
            {clamped}
          </span>
        </div>
      </div>

      <p className={`text-sm font-semibold ${colors.text}`}>
        {label || riskLabel(clamped)}
      </p>
    </div>
  );
};

export default RiskGauge;
