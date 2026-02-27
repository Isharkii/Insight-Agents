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
      <div className="bg-white dark:bg-gray-900 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-800 p-6">
        <h3 className="text-sm font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-4">
          KPI Revenue Trend
        </h3>
        <p className="text-gray-400 dark:text-gray-500 text-center py-12">
          No revenue data available
        </p>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-900 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-800 p-6">
      <h3 className="text-sm font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-4">
        KPI Revenue Trend
      </h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 16, left: 8, bottom: 0 }}>
            <CartesianGrid
              strokeDasharray="3 3"
              className="stroke-gray-200 dark:stroke-gray-700"
            />
            <XAxis
              dataKey="period"
              tick={{ fontSize: 12 }}
              className="text-gray-500 dark:text-gray-400"
            />
            <YAxis
              tickFormatter={formatCurrency}
              tick={{ fontSize: 12 }}
              className="text-gray-500 dark:text-gray-400"
              width={60}
            />
            <Tooltip
              formatter={(value: number) => [formatCurrency(value), "Revenue"]}
              contentStyle={{
                backgroundColor: "var(--tooltip-bg, #fff)",
                borderColor: "var(--tooltip-border, #e5e7eb)",
                borderRadius: "0.5rem",
                fontSize: "0.875rem",
              }}
            />
            <Line
              type="monotone"
              dataKey="value"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={{ r: 3, fill: "#3b82f6" }}
              activeDot={{ r: 5 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default RevenueTrendChart;
