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
      <div className="ia-surface p-6">
        <p className="ia-label mb-4">Market Share Projection</p>
        <p className="py-12 text-center text-sm text-slate-400">No market share data available</p>
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

  if (firstProjectedIndex > 0) {
    actualData[firstProjectedIndex] = {
      ...actualData[firstProjectedIndex],
      projected: data[firstProjectedIndex - 1].value,
    };
    actualData[firstProjectedIndex - 1] = {
      ...actualData[firstProjectedIndex - 1],
      projected: data[firstProjectedIndex - 1].value,
    };
  }

  return (
    <div className="ia-surface p-6">
      <div className="mb-4 flex items-center justify-between">
        <p className="ia-label">Market Share Projection</p>
        {firstProjectedIndex > 0 && (
          <div className="flex items-center gap-4 text-xs text-slate-500">
            <span className="flex items-center gap-1">
              <span className="inline-block h-0.5 w-3 bg-teal-700" /> Actual
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-0.5 w-3 bg-slate-500" /> Projected
            </span>
          </div>
        )}
      </div>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={actualData} margin={{ top: 8, right: 16, left: 8, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-slate-200" />
            <XAxis dataKey="period" tick={{ fontSize: 12 }} className="text-slate-500" />
            <YAxis
              tickFormatter={(value: number) => `${value.toFixed(1)}%`}
              tick={{ fontSize: 12 }}
              className="text-slate-500"
              width={50}
            />
            <Tooltip
              formatter={(value: number, name: string) => [
                `${value.toFixed(2)}%`,
                name === "actual" ? "Actual" : "Projected",
              ]}
              contentStyle={{
                backgroundColor: "#fff",
                borderColor: "#e2e8f0",
                borderRadius: "0.5rem",
                fontSize: "0.875rem",
              }}
            />
            {projectionBoundary && (
              <ReferenceLine
                x={projectionBoundary}
                stroke="#64748b"
                strokeDasharray="4 4"
                strokeWidth={1}
              />
            )}
            <Line
              type="monotone"
              dataKey="actual"
              stroke="#0f766e"
              strokeWidth={2}
              dot={{ r: 3, fill: "#0f766e" }}
              connectNulls={false}
            />
            <Line
              type="monotone"
              dataKey="projected"
              stroke="#64748b"
              strokeWidth={2}
              strokeDasharray="6 3"
              dot={{ r: 3, fill: "#64748b" }}
              connectNulls={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default MarketShareChart;
