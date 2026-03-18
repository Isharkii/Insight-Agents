import { useMemo, type FC } from "react";
import { MetricCard } from "../ui";
interface KpiEntry {
  name: string;
  value: number;
  unit?: string;
  label?: string;
}

interface MetricCardsProps {
  kpiEntries: KpiEntry[];
  /** Metric time-series keyed by metric name → ordered values. */
  metricSeries?: Record<string, number[]>;
}

/** Section 2: Key metric cards with sparklines and deltas. */
const MetricCards: FC<MetricCardsProps> = ({ kpiEntries, metricSeries }) => {
  const cards = useMemo(() => {
    return kpiEntries.map((kpi) => {
      const series = metricSeries?.[kpi.name];
      let deltaPct: number | null = null;
      if (series && series.length >= 2) {
        const prev = series[series.length - 2];
        const curr = series[series.length - 1];
        if (prev !== 0) {
          deltaPct = ((curr - prev) / Math.abs(prev)) * 100;
        }
      }
      return { ...kpi, sparkValues: series, deltaPct };
    });
  }, [kpiEntries, metricSeries]);

  if (cards.length === 0) return null;

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4 ia-stagger">
      {cards.map((card, i) => (
        <MetricCard
          key={card.name}
          name={card.name}
          value={card.value}
          unit={card.unit}
          label={card.label}
          deltaPct={card.deltaPct}
          sparkValues={card.sparkValues}
          index={i}
        />
      ))}
    </div>
  );
};

export default MetricCards;
