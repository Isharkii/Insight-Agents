import { type FC, useMemo } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

interface KpiRow {
  period_end: string;
  metric_name: string;
  metric_value: number;
}

interface KpiTimeSeriesChartProps {
  data: KpiRow[];
}

const COLORS = [
  "#3b82f6",
  "#10b981",
  "#f59e0b",
  "#8b5cf6",
  "#ef4444",
  "#06b6d4",
  "#f97316",
  "#ec4899",
];

const KpiTimeSeriesChart: FC<KpiTimeSeriesChartProps> = ({ data }) => {
  const { chartData, metricNames } = useMemo(() => {
    if (!data || data.length === 0) return { chartData: [], metricNames: [] };

    const grouped: Record<string, Record<string, number>> = {};
    const nameSet = new Set<string>();

    for (const row of data) {
      const period = row.period_end?.slice(0, 7) ?? "";
      if (!period) continue;
      if (!grouped[period]) grouped[period] = {};
      grouped[period][row.metric_name] = row.metric_value;
      nameSet.add(row.metric_name);
    }

    const sortedPeriods = Object.keys(grouped).sort();
    const chartData = sortedPeriods.map((period) => ({
      period,
      ...grouped[period],
    }));
    const metricNames = Array.from(nameSet).sort();

    return { chartData, metricNames };
  }, [data]);

  if (chartData.length === 0) {
    return (
      <div className="bg-white dark:bg-gray-900 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-800 p-6">
        <h3 className="text-sm font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-4">
          KPI Time Series
        </h3>
        <p className="text-gray-400 dark:text-gray-500 text-center py-12 text-sm">
          No KPI time-series data available
        </p>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-900 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-800 p-6">
      <h3 className="text-sm font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-4">
        KPI Time Series
      </h3>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={chartData}
            margin={{ top: 8, right: 16, left: 8, bottom: 0 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              className="stroke-gray-200 dark:stroke-gray-700"
            />
            <XAxis
              dataKey="period"
              tick={{ fontSize: 11 }}
              className="text-gray-500 dark:text-gray-400"
            />
            <YAxis
              tick={{ fontSize: 11 }}
              className="text-gray-500 dark:text-gray-400"
              width={55}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "var(--tooltip-bg, #fff)",
                borderColor: "var(--tooltip-border, #e5e7eb)",
                borderRadius: "0.5rem",
                fontSize: "0.75rem",
              }}
            />
            {metricNames.length > 1 && (
              <Legend
                wrapperStyle={{ fontSize: "0.75rem" }}
                iconType="plainline"
              />
            )}
            {metricNames.map((name, idx) => (
              <Line
                key={name}
                type="monotone"
                dataKey={name}
                stroke={COLORS[idx % COLORS.length]}
                strokeWidth={2}
                dot={{ r: 2 }}
                activeDot={{ r: 4 }}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default KpiTimeSeriesChart;
