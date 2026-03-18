import type { FC } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";
import EmptyState from "../ui/EmptyState";

interface WaterfallItem {
  /** Label for the bar. */
  label: string;
  /** Signed value — positive = adds, negative = subtracts. */
  value: number;
}

interface WaterfallChartProps {
  items: WaterfallItem[];
  height?: number;
  /** Format the value for display. */
  formatter?: (v: number) => string;
}

const defaultFormatter = (v: number) => (v >= 0 ? `+${v.toFixed(1)}` : v.toFixed(1));

/**
 * Driver decomposition waterfall chart.
 * Uses stacked bars: an invisible "base" bar + a visible "delta" bar.
 * Positive contributions stack up, negative contributions stack down.
 */
const WaterfallChart: FC<WaterfallChartProps> = ({
  items,
  height = 240,
  formatter = defaultFormatter,
}) => {
  if (!items || items.length === 0) {
    return <EmptyState message="No decomposition data" height={height} />;
  }

  // Compute running totals for the invisible base
  let running = 0;
  const data = items.map((item) => {
    const base = item.value >= 0 ? running : running + item.value;
    const delta = Math.abs(item.value);
    running += item.value;
    return {
      label: item.label,
      base: Math.max(0, base),
      delta,
      rawValue: item.value,
      total: running,
    };
  });

  // Add total bar
  data.push({
    label: "Total",
    base: 0,
    delta: Math.max(0, running),
    rawValue: running,
    total: running,
  });

  return (
    <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 16, left: 8, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" vertical={false} />
          <XAxis dataKey="label" tick={{ fontSize: 10 }} interval={0} angle={-20} textAnchor="end" height={50} />
          <YAxis tick={{ fontSize: 11 }} width={45} />
          <Tooltip
            contentStyle={{ borderRadius: "0.5rem", fontSize: "0.75rem" }}
            formatter={(value: number) => [formatter(value), "Contribution"]}
          />
          {/* Invisible base bar */}
          <Bar dataKey="base" stackId="stack" fill="transparent" />
          {/* Visible delta bar */}
          <Bar dataKey="delta" stackId="stack" radius={[4, 4, 0, 0]}>
            {data.map((entry, i) => (
              <Cell
                key={i}
                fill={
                  entry.label === "Total"
                    ? "var(--chart-1)"
                    : entry.rawValue >= 0
                      ? "var(--ia-success)"
                      : "var(--ia-danger)"
                }
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default WaterfallChart;
