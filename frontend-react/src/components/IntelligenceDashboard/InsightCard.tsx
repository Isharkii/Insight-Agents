import type { FC } from "react";
import type { InsightCardData } from "./types";

const directionConfig = {
  positive: {
    arrow: "UP",
    color: "text-emerald-600",
    bg: "bg-emerald-50",
  },
  negative: {
    arrow: "DOWN",
    color: "text-rose-600",
    bg: "bg-rose-50",
  },
  neutral: {
    arrow: "FLAT",
    color: "text-slate-500",
    bg: "bg-slate-100",
  },
} as const;

interface InsightCardsProps {
  cards: InsightCardData[];
}

const InsightCards: FC<InsightCardsProps> = ({ cards }) => {
  if (!cards || cards.length === 0) return null;

  return (
    <section>
      <p className="ia-label mb-3">Insight Cards</p>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {cards.map((card, idx) => {
          const cfg = directionConfig[card.direction];
          return (
            <div
              key={idx}
              className="ia-surface p-5"
            >
              <h3 className="text-sm font-semibold leading-snug text-slate-900">{card.title}</h3>

              <p className="mt-3 text-2xl font-bold text-slate-900">{card.metric}</p>

              <div className="mt-4 flex items-center justify-between">
                <span className="text-xs font-medium text-slate-500">
                  Impact <span className="font-semibold text-slate-800">{card.impact_score}</span>
                </span>

                <span
                  className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold ${cfg.bg} ${cfg.color}`}
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
