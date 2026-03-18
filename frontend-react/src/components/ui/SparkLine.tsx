import type { FC } from "react";

interface SparkLineProps {
  /** Ordered numeric values (oldest → newest). */
  values: number[];
  /** SVG width in px. */
  width?: number;
  /** SVG height in px. */
  height?: number;
  /** Stroke color (CSS). */
  color?: string;
  /** Stroke width. */
  strokeWidth?: number;
  /** Show a filled area below the line. */
  filled?: boolean;
}

/**
 * Pure-SVG sparkline — no library dependency.
 * Designed for embedding inside metric cards.
 */
const SparkLine: FC<SparkLineProps> = ({
  values,
  width = 80,
  height = 28,
  color = "var(--chart-1)",
  strokeWidth = 1.5,
  filled = false,
}) => {
  if (values.length < 2) {
    return <svg width={width} height={height} />;
  }

  const pad = strokeWidth;
  const w = width - pad * 2;
  const h = height - pad * 2;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  const points = values.map((v, i) => {
    const x = pad + (i / (values.length - 1)) * w;
    const y = pad + h - ((v - min) / range) * h;
    return { x, y };
  });

  const linePath = points.map((p, i) => `${i === 0 ? "M" : "L"}${p.x},${p.y}`).join(" ");
  const areaPath = `${linePath} L${points[points.length - 1].x},${height} L${points[0].x},${height} Z`;

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="block">
      {filled && (
        <path d={areaPath} fill={color} fillOpacity={0.12} />
      )}
      <path d={linePath} fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" />
      {/* Latest-value dot */}
      <circle cx={points[points.length - 1].x} cy={points[points.length - 1].y} r={2} fill={color} />
    </svg>
  );
};

export default SparkLine;
