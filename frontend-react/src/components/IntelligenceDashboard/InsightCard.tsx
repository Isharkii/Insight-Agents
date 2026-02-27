import type { FC } from "react";
import type { InsightCardData } from "./types";

const directionConfig = {
  positive: {
    arrow: "↑",
    color: "text-emerald-600 dark:text-emerald-400",
    bg: "bg-emerald-50 dark:bg-emerald-900/20",
  },
  negative: {
    arrow: "↓",
    color: "text-red-600 dark:text-red-400",
    bg: "bg-red-50 dark:bg-red-900/20",
  },
  neutral: {
    arrow: "→",
    color: "text-gray-500 dark:text-gray-400",
    bg: "bg-gray-50 dark:bg-gray-800/40",
  },
} as const;

interface InsightCardsProps {
  cards: InsightCardData[];
}

const InsightCards: FC<InsightCardsProps> = ({ cards }) => {
  if (!cards || cards.length === 0) return null;

  return (
    <section>
      <h2 className="text-sm font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-3">
        Insight Cards
      </h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {cards.map((card, idx) => {
          const cfg = directionConfig[card.direction];
          return (
            <div
              key={idx}
              className="bg-white dark:bg-gray-900 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-800 p-5 flex flex-col gap-3"
            >
              <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100 leading-snug">
                {card.title}
              </h3>

              <p className="text-2xl font-bold text-gray-900 dark:text-gray-50">
                {card.metric}
              </p>

              <div className="flex items-center justify-between mt-auto">
                <span className="text-xs font-medium text-gray-500 dark:text-gray-400">
                  Impact&nbsp;
                  <span className="text-gray-900 dark:text-gray-100 font-semibold">
                    {card.impact_score}
                  </span>
                </span>

                <span
                  className={`inline-flex items-center gap-1 text-xs font-semibold px-2 py-0.5 rounded-full ${cfg.bg} ${cfg.color}`}
                >
                  {cfg.arrow}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
};

export default InsightCards;
