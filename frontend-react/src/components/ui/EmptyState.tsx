import type { FC } from "react";

interface EmptyStateProps {
  message?: string;
  height?: number;
}

/** Placeholder shown when a section has no data. */
const EmptyState: FC<EmptyStateProps> = ({
  message = "No data available",
  height = 120,
}) => (
  <div className="ia-empty" style={{ minHeight: height }}>
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="mb-2 opacity-40">
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="8" x2="12" y2="12" />
      <line x1="12" y1="16" x2="12.01" y2="16" />
    </svg>
    <span>{message}</span>
  </div>
);

export default EmptyState;
