import type { FC } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from "recharts";
import type { ProjectedTimeSeriesPoint } from "./types";

interface MarketShareChartProps {
  data: ProjectedTimeSeriesPoint[];
}

const MarketShareChart: FC<MarketShareChartProps> = ({ data }) => {
  if (data.length === 0) {
    return (
      <div className="bg-white dark:bg-gray-900 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-800 p-6">
        <h3 className="text-sm font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-4">
          Market Share Projection
        </h3>
        <p className="text-gray-400 dark:text-gray-500 text-center py-12">
          No market share data available
        </p>
      </div>
    );
  }

  const firstProjectedIndex = data.findIndex((point) => point.projected);
  const projectionBoundary =
    firstProjectedIndex > 0 ? data[firstProjectedIndex - 1].period : null;

  const actualData = data.map((point) => ({
    ...point,
    actual: point.projected ? undefined : point.value,
    projected: point.projected ? point.value : undefined,
  }));

  // Bridge the gap: duplicate the last actual point as first projected point
  if (firstProjectedIndex > 0) {
    actualData[firstProjectedIndex] = {
      ...actualData[firstProjectedIndex],
      projected: data[firstProjectedIndex - 1].value,
    };
    // Also set the bridge on the last actual point
    actualData[firstProjectedIndex - 1] = {
      ...actualData[firstProjectedIndex - 1],
      projected: data[firstProjectedIndex - 1].value,
    };
  }

  return (
    <div className="bg-white dark:bg-gray-900 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-800 p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">
          Market Share Projection
        </h3>
        {firstProjectedIndex > 0 && (
          <div className="flex items-center gap-4 text-xs text-gray-500 dark:text-gray-400">
            <span className="flex items-center gap-1">
              <span className="w-3 h-0.5 bg-blue-500 inline-block" /> Actual
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-0.5 bg-violet-500 inline-block border-dashed" /> Projected
            </span>
          </div>
        )}
      </div>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={actualData} margin={{ top: 8, right: 16, left: 8, bottom: 0 }}>
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
              tickFormatter={(value: number) => `${value.toFixed(1)}%`}
              tick={{ fontSize: 12 }}
              className="text-gray-500 dark:text-gray-400"
              width={50}
            />
            <Tooltip
              formatter={(value: number, name: string) => [
                `${value.toFixed(2)}%`,
                name === "actual" ? "Actual" : "Projected",
              ]}
              contentStyle={{
                backgroundColor: "var(--tooltip-bg, #fff)",
                borderColor: "var(--tooltip-border, #e5e7eb)",
                borderRadius: "0.5rem",
                fontSize: "0.875rem",
              }}
            />
            {projectionBoundary && (
              <ReferenceLine
                x={projectionBoundary}
                stroke="#a78bfa"
                strokeDasharray="4 4"
                strokeWidth={1}
              />
            )}
            <Line
              type="monotone"
              dataKey="actual"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={{ r: 3, fill: "#3b82f6" }}
              connectNulls={false}
            />
            <Line
              type="monotone"
              dataKey="projected"
              stroke="#8b5cf6"
              strokeWidth={2}
              strokeDasharray="6 3"
              dot={{ r: 3, fill: "#8b5cf6" }}
              connectNulls={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default MarketShareChart;
