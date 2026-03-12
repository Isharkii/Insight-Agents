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
    <div className="min-h-screen px-4 py-8 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <header className="ia-surface px-6 py-5 sm:px-7">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p className="ia-label">Embedded Dashboard</p>
              <h1 className="ia-title mt-1">Intelligence Dashboard</h1>
              <p className="ia-subtitle mt-1.5">
                {data.entity_name}
                <span className="mx-2 text-slate-300">|</span>
                {data.business_type}
              </p>
            </div>

            {data.pipeline_errors && data.pipeline_errors.length > 0 && (
              <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-1.5 text-xs text-amber-700">
                {data.pipeline_errors.length} pipeline warning(s)
              </div>
            )}
          </div>
        </header>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
          <div className="lg:col-span-3">
            <HealthIndexCard
              score={data.health_index}
              label={data.health_label}
              entityName={data.entity_name}
            />
          </div>

          <div className="grid grid-cols-1 gap-6 lg:col-span-9 md:grid-cols-2">
            <RevenueTrendChart data={data.revenue_trend} />
            <MarketShareChart data={data.market_share} />
          </div>
        </div>

        {data.insight_cards && data.insight_cards.length > 0 && (
          <InsightCards cards={data.insight_cards} />
        )}

        <ClassificationBadges classification={data.classification} />

        <KeyInsights insights={data.insights} />
      </div>
    </div>
  );
};

export default IntelligenceDashboard;
