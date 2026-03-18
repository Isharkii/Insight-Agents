import type { FC } from "react";

interface GaugeProps {
  /** Value to display (0-100). */
  value: number;
  /** Label below the number (e.g. "Confidence", "Risk"). */
  label: string;
  /** Size in px (width & height). */
  size?: number;
  /** Color zones — array of {threshold, color}. Zones should be sorted ascending.
   *  Default: green→amber→red thresholds at 30/60/80. */
  zones?: { threshold: number; color: string }[];
  /** If true, higher values = worse (red), e.g. risk scores. */
  invertColor?: boolean;
}

const DEFAULT_ZONES = [
  { threshold: 30, color: "#10b981" },
  { threshold: 60, color: "#f59e0b" },
  { threshold: 80, color: "#f97316" },
  { threshold: 101, color: "#ef4444" },
];

const DEFAULT_ZONES_INVERTED = [
  { threshold: 30, color: "#ef4444" },
  { threshold: 60, color: "#f59e0b" },
  { threshold: 80, color: "#10b981" },
  { threshold: 101, color: "#10b981" },
];

function getColor(value: number, zones: { threshold: number; color: string }[]): string {
  for (const zone of zones) {
    if (value <= zone.threshold) return zone.color;
  }
  return zones[zones.length - 1]?.color ?? "#94a3b8";
}

/**
 * Reusable radial gauge for confidence, risk, health scores.
 * Pure SVG — no Recharts dependency.
 */
const Gauge: FC<GaugeProps> = ({
  value,
  label,
  size = 120,
  zones,
  invertColor = false,
}) => {
  const clamped = Math.max(0, Math.min(100, value));
  const effectiveZones = zones ?? (invertColor ? DEFAULT_ZONES : DEFAULT_ZONES_INVERTED);
  const color = invertColor
    ? getColor(clamped, DEFAULT_ZONES)
    : getColor(clamped, effectiveZones);

  const cx = size / 2;
  const cy = size / 2;
  const radius = (size - 12) / 2;
  const circumference = Math.PI * radius; // semicircle
  const offset = circumference - (clamped / 100) * circumference;

  return (
    <div className="flex flex-col items-center">
      <svg width={size} height={size * 0.65} viewBox={`0 0 ${size} ${size * 0.65}`}>
        {/* Background track */}
        <path
          d={`M ${cx - radius} ${cy} A ${radius} ${radius} 0 0 1 ${cx + radius} ${cy}`}
          fill="none"
          stroke="#e2e8f0"
          strokeWidth={8}
          strokeLinecap="round"
        />
        {/* Value arc */}
        <path
          d={`M ${cx - radius} ${cy} A ${radius} ${radius} 0 0 1 ${cx + radius} ${cy}`}
          fill="none"
          stroke={color}
          strokeWidth={8}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className="transition-all duration-700 ease-out"
        />
      </svg>
      <p className="text-2xl font-bold ia-mono -mt-3" style={{ color }}>
        {Math.round(clamped)}
      </p>
      <p className="ia-caption mt-0.5">{label}</p>
    </div>
  );
};

export default Gauge;
