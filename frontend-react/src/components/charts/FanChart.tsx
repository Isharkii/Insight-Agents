import type { FC } from "react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";
import EmptyState from "../ui/EmptyState";

interface FanChartPoint {
  period: string;
  actual?: number;
  p50?: number;
  p25?: number;
  p75?: number;
  p10?: number;
  p90?: number;
}

interface FanChartProps {
  data: FanChartPoint[];
  height?: number;
  /** Format Y-axis values (e.g. currency). */
  yFormatter?: (v: number) => string;
}

const defaultYFormat = (v: number) =>
  v >= 1_000_000
    ? `$${(v / 1_000_000).toFixed(1)}M`
    : v >= 1_000
      ? `$${(v / 1_000).toFixed(0)}K`
      : `$${v}`;

/**
 * Historical line + forecast fan chart with P10/P25/P50/P75/P90 confidence bands.
 * Built on Recharts AreaChart with layered transparent areas.
 */
const FanChart: FC<FanChartProps> = ({
  data,
  height = 280,
  yFormatter = defaultYFormat,
}) => {
  if (!data || data.length === 0) {
    return <EmptyState message="No forecast data" height={height} />;
  }

  const hasActual = data.some((d) => d.actual != null);
  const hasBands = data.some((d) => d.p10 != null || d.p25 != null);
  const hasP50 = data.some((d) => d.p50 != null);

  return (
    <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 8, right: 16, left: 8, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
          <XAxis dataKey="period" tick={{ fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} tickFormatter={yFormatter} width={60} />
          <Tooltip
            contentStyle={{ borderRadius: "0.5rem", fontSize: "0.75rem" }}
            formatter={(v: number, name: string) => [yFormatter(v), name]}
          />

          {/* P10-P90 outer band */}
          {hasBands && (
            <>
              <Area type="monotone" dataKey="p90" stroke="none" fill="#8b5cf6" fillOpacity={0.06} />
              <Area type="monotone" dataKey="p10" stroke="none" fill="#ffffff" fillOpacity={1} />
            </>
          )}

          {/* P25-P75 inner band */}
          {hasBands && (
            <>
              <Area type="monotone" dataKey="p75" stroke="none" fill="#8b5cf6" fillOpacity={0.12} />
              <Area type="monotone" dataKey="p25" stroke="none" fill="#ffffff" fillOpacity={1} />
            </>
          )}

          {/* P50 median forecast line */}
          {hasP50 && (
            <Area
              type="monotone"
              dataKey="p50"
              stroke="#8b5cf6"
              strokeWidth={2}
              strokeDasharray="6 3"
              fill="none"
            />
          )}

          {/* Historical actual line */}
          {hasActual && (
            <Area
              type="monotone"
              dataKey="actual"
              stroke="var(--chart-1)"
              strokeWidth={2.5}
              fill="var(--chart-1)"
              fillOpacity={0.08}
              dot={{ r: 3, fill: "var(--chart-1)" }}
              activeDot={{ r: 5 }}
              connectNulls={false}
            />
          )}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};

export default FanChart;
