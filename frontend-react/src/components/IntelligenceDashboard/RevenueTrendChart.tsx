import type { FC } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";
import type { TimeSeriesPoint } from "./types";

interface RevenueTrendChartProps {
  data: TimeSeriesPoint[];
}

function formatCurrency(value: number): string {
  if (value >= 1_000_000) return `$${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `$${(value / 1_000).toFixed(0)}K`;
  return `$${value.toFixed(0)}`;
}

const RevenueTrendChart: FC<RevenueTrendChartProps> = ({ data }) => {
  if (data.length === 0) {
    return (
      <div className="ia-surface p-6">
        <p className="ia-label mb-4">KPI Revenue Trend</p>
        <p className="py-12 text-center text-sm text-slate-400">No revenue data available</p>
      </div>
    );
  }

  return (
    <div className="ia-surface p-6">
      <p className="ia-label mb-4">KPI Revenue Trend</p>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 16, left: 8, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-slate-200" />
            <XAxis dataKey="period" tick={{ fontSize: 12 }} className="text-slate-500" />
            <YAxis
              tickFormatter={formatCurrency}
              tick={{ fontSize: 12 }}
              className="text-slate-500"
              width={60}
            />
            <Tooltip
              formatter={(value: number) => [formatCurrency(value), "Revenue"]}
              contentStyle={{
                backgroundColor: "#fff",
                borderColor: "#e2e8f0",
                borderRadius: "0.5rem",
                fontSize: "0.875rem",
              }}
            />
            <Line
              type="monotone"
              dataKey="value"
              stroke="#0f766e"
              strokeWidth={2}
              dot={{ r: 3, fill: "#0f766e" }}
              activeDot={{ r: 5 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default RevenueTrendChart;
