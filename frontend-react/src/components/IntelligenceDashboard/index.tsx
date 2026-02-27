import type { FC } from "react";
import type { DashboardData } from "./types";
import HealthIndexCard from "./HealthIndexCard";
import RevenueTrendChart from "./RevenueTrendChart";
import MarketShareChart from "./MarketShareChart";
import ClassificationBadges from "./ClassificationBadges";
import InsightCards from "./InsightCard";
import KeyInsights from "./KeyInsights";

interface IntelligenceDashboardProps {
  data: DashboardData;
}

const IntelligenceDashboard: FC<IntelligenceDashboardProps> = ({ data }) => {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950 px-4 py-8 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <header className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-2">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
              Intelligence Dashboard
            </h1>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              {data.entity_name}
              <span className="mx-2 text-gray-300 dark:text-gray-600">|</span>
              {data.business_type}
            </p>
          </div>

          {data.pipeline_errors && data.pipeline_errors.length > 0 && (
            <div className="text-xs text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg px-3 py-1.5">
              {data.pipeline_errors.length} pipeline warning(s)
            </div>
          )}
        </header>

        {/* Top row: Health Index + Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-3">
            <HealthIndexCard
              score={data.health_index}
              label={data.health_label}
              entityName={data.entity_name}
            />
          </div>

          <div className="lg:col-span-9 grid grid-cols-1 md:grid-cols-2 gap-6">
            <RevenueTrendChart data={data.revenue_trend} />
            <MarketShareChart data={data.market_share} />
          </div>
        </div>

        {/* Insight Cards */}
        {data.insight_cards && data.insight_cards.length > 0 && (
          <InsightCards cards={data.insight_cards} />
        )}

        {/* Classification */}
        <ClassificationBadges classification={data.classification} />

        {/* Key Insights */}
        <KeyInsights insights={data.insights} />
      </div>
    </div>
  );
};

export default IntelligenceDashboard;
