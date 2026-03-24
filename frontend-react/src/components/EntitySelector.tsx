import type { FC } from "react";

interface EntitySelectorProps {
  entities: string[];
  selected: string;
  onSelect: (entity: string) => void;
}

const EntitySelector: FC<EntitySelectorProps> = ({
  entities,
  selected,
  onSelect,
}) => {
  if (entities.length < 2) return null;

  return (
    <div className="ia-surface ia-fade-up p-5 sm:p-6">
      <div className="mb-3">
        <p className="ia-label">Target Entity</p>
        <p className="ia-subtitle mt-1">
          {entities.length} entities detected in your dataset. Select the primary
          entity to analyze — the rest become benchmark peers.
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <select
          value={selected}
          onChange={(e) => onSelect(e.target.value)}
          className="ia-select max-w-xs"
        >
          {entities.map((entity) => (
            <option key={entity} value={entity}>
              {entity}
            </option>
          ))}
        </select>

        <div className="flex items-center gap-2 text-xs text-slate-500">
          <svg
            className="h-4 w-4 text-teal-600"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z"
            />
          </svg>
          <span>
            {entities.length - 1} peer{entities.length - 1 !== 1 ? "s" : ""} for
            benchmarking
          </span>
        </div>
      </div>
    </div>
  );
};

export default EntitySelector;
