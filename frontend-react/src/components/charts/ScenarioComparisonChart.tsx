import type { FC } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

interface ScenarioRow {
  scenario: string;
  projected_value: number | null;
  projected_growth: number | null;
}

interface ScenarioComparisonChartProps {
  scenarios: Record<
    string,
    { projected_value?: number; projected_growth?: number }
  >;
}

const ScenarioComparisonChart: FC<ScenarioComparisonChartProps> = ({
  scenarios,
}) => {
  if (!scenarios || Object.keys(scenarios).length === 0) return null;

  const data: ScenarioRow[] = Object.entries(scenarios).map(
    ([name, vals]) => ({
      scenario: name,
      projected_value: vals.projected_value ?? null,
      projected_growth: vals.projected_growth ?? null,
    }),
  );

  const hasValue = data.some((d) => d.projected_value != null);
  const hasGrowth = data.some((d) => d.projected_growth != null);

  return (
    <div className="bg-white dark:bg-gray-900 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-800 p-6">
      <h3 className="text-sm font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-4">
        Scenario Comparison
      </h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            margin={{ top: 8, right: 16, left: 8, bottom: 0 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              className="stroke-gray-200 dark:stroke-gray-700"
            />
            <XAxis
              dataKey="scenario"
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
            {(hasValue || hasGrowth) && (
              <Legend
                wrapperStyle={{ fontSize: "0.75rem" }}
              />
            )}
            {hasValue && (
              <Bar
                dataKey="projected_value"
                name="Projected Value"
                fill="#3b82f6"
                radius={[4, 4, 0, 0]}
              />
            )}
            {hasGrowth && (
              <Bar
                dataKey="projected_growth"
                name="Projected Growth"
                fill="#8b5cf6"
                radius={[4, 4, 0, 0]}
              />
            )}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default ScenarioComparisonChart;
