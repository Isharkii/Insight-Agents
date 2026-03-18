import type { FC } from "react";
import SparkLine from "./SparkLine";

const CHART_COLORS = [
  "var(--chart-1)", "var(--chart-2)", "var(--chart-3)",
  "var(--chart-4)", "var(--chart-5)", "var(--chart-6)",
];

interface MetricCardProps {
  name: string;
  value: number;
  unit?: string;
  label?: string;
  /** Period-over-period change in percent. */
  deltaPct?: number | null;
  /** Historical values for sparkline (oldest → newest). */
  sparkValues?: number[];
  /** Color index for left accent bar. */
  index?: number;
}

function formatValue(value: number, unit?: string): string {
  if (unit === "%" || unit === "percent") return `${value.toFixed(1)}%`;
  if (value >= 1_000_000) return `$${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `$${(value / 1_000).toFixed(1)}K`;
  if (Number.isInteger(value)) return value.toLocaleString();
  return value.toFixed(2);
}

/** Premium metric card with accent bar, sparkline, and delta badge. */
const MetricCard: FC<MetricCardProps> = ({
  name,
  value,
  unit,
  label,
  deltaPct,
  sparkValues,
  index = 0,
}) => {
  const color = CHART_COLORS[index % CHART_COLORS.length];
  const hasDelta = deltaPct != null && deltaPct !== 0;
  const deltaPositive = (deltaPct ?? 0) >= 0;

  return (
    <div className="ia-card p-4 relative overflow-hidden">
      {/* Left accent bar */}
      <div
        className="absolute top-0 left-0 w-1 h-full rounded-l-2xl"
        style={{ backgroundColor: color }}
      />
      <div className="pl-3 flex flex-col gap-1.5">
        <p className="ia-caption truncate">
          {label || name.replace(/_/g, " ")}
        </p>
        <div className="flex items-end justify-between gap-2">
          <p className="text-xl font-bold ia-mono text-gray-900 dark:text-gray-100 leading-none">
            {formatValue(value, unit)}
          </p>
          {sparkValues && sparkValues.length >= 2 && (
            <SparkLine values={sparkValues} color={color} width={64} height={24} filled />
          )}
        </div>
        {hasDelta && (
          <span
            className={`text-xs font-semibold ia-mono ${
              deltaPositive ? "text-emerald-600" : "text-red-500"
            }`}
          >
            {deltaPositive ? "+" : ""}{deltaPct!.toFixed(1)}%
          </span>
        )}
      </div>
    </div>
  );
};

export default MetricCard;
