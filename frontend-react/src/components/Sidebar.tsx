import { useState, useEffect, type FC } from "react";
import { fetchClients } from "../api/client";

const BUSINESS_TYPES = [
  { value: "auto", label: "Auto-detect" },
  { value: "saas", label: "SaaS" },
  { value: "ecommerce", label: "E-Commerce" },
  { value: "agency", label: "Agency" },
  { value: "general_timeseries", label: "General Timeseries" },
  { value: "financial_markets", label: "Financial Markets" },
  { value: "marketing_analytics", label: "Marketing Analytics" },
  { value: "operations", label: "Operations" },
  { value: "retail", label: "Retail" },
  { value: "healthcare", label: "Healthcare" },
  { value: "generic_timeseries", label: "Generic Timeseries" },
];

const MULTI_ENTITY_OPTIONS = [
  { value: "auto", label: "Auto" },
  { value: "split", label: "Split" },
  { value: "error", label: "Error" },
];

const MODEL_OPTIONS = [
  { value: "default", label: "Default" },
  { value: "gpt-4o-mini", label: "GPT-4o Mini" },
  { value: "gpt-4o", label: "GPT-4o" },
];

export interface SidebarState {
  mode: "LOCAL" | "CLOUD";
  clientId: string;
  entityOverride: string;
  businessType: string;
  multiEntityBehavior: string;
  model: string;
}

interface SidebarProps {
  state: SidebarState;
  onChange: (state: SidebarState) => void;
  onRun: () => void;
  onClear: () => void;
  loading: boolean;
  collapsed: boolean;
  onToggleCollapse: () => void;
}

const Sidebar: FC<SidebarProps> = ({
  state,
  onChange,
  onRun,
  onClear,
  loading,
  collapsed,
  onToggleCollapse,
}) => {
  const [clients, setClients] = useState<string[]>(["default"]);

  useEffect(() => {
    fetchClients().then((list) => {
      setClients(["default", ...list.filter((c) => c !== "default")]);
    });
  }, []);

  const update = (patch: Partial<SidebarState>) =>
    onChange({ ...state, ...patch });

  if (collapsed) {
    return (
      <div className="w-12 shrink-0 bg-gray-900 flex flex-col items-center py-4">
        <button
          onClick={onToggleCollapse}
          className="text-gray-400 hover:text-white transition-colors"
          title="Expand sidebar"
        >
          <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
            <path
              fillRule="evenodd"
              d="M7.21 14.77a.75.75 0 01.02-1.06L11.168 10 7.23 6.29a.75.75 0 111.04-1.08l4.5 4.25a.75.75 0 010 1.08l-4.5 4.25a.75.75 0 01-1.06-.02z"
              clipRule="evenodd"
            />
          </svg>
        </button>
      </div>
    );
  }

  return (
    <aside className="w-72 shrink-0 bg-gray-900 text-gray-100 flex flex-col h-screen overflow-y-auto">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-gray-800">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-gray-400">
          Controls
        </h2>
        <button
          onClick={onToggleCollapse}
          className="text-gray-500 hover:text-gray-300 transition-colors"
          title="Collapse sidebar"
        >
          <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
            <path
              fillRule="evenodd"
              d="M12.79 5.23a.75.75 0 01-.02 1.06L8.832 10l3.938 3.71a.75.75 0 11-1.04 1.08l-4.5-4.25a.75.75 0 010-1.08l4.5-4.25a.75.75 0 011.06.02z"
              clipRule="evenodd"
            />
          </svg>
        </button>
      </div>

      <div className="flex-1 px-5 py-4 space-y-5">
        {/* Mode Toggle */}
        <div>
          <label className="block text-xs font-medium text-gray-400 mb-1.5">
            Mode
          </label>
          <div className="flex rounded-lg overflow-hidden border border-gray-700">
            {(["LOCAL", "CLOUD"] as const).map((m) => (
              <button
                key={m}
                onClick={() => update({ mode: m })}
                className={`flex-1 px-3 py-1.5 text-xs font-medium transition-colors ${
                  state.mode === m
                    ? "bg-blue-600 text-white"
                    : "bg-gray-800 text-gray-400 hover:text-gray-200"
                }`}
              >
                {m}
              </button>
            ))}
          </div>
        </div>

        {/* Client Select */}
        <div>
          <label className="block text-xs font-medium text-gray-400 mb-1.5">
            Client
          </label>
          <select
            value={state.clientId}
            onChange={(e) => update({ clientId: e.target.value })}
            className="w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {clients.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>

        {/* Entity Override */}
        <div>
          <label className="block text-xs font-medium text-gray-400 mb-1.5">
            Entity / Client ID Override
          </label>
          <input
            type="text"
            value={state.entityOverride}
            onChange={(e) => update({ entityOverride: e.target.value })}
            placeholder="Override entity name"
            className="w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        {/* Business Type */}
        <div>
          <label className="block text-xs font-medium text-gray-400 mb-1.5">
            Business Type
          </label>
          <select
            value={state.businessType}
            onChange={(e) => update({ businessType: e.target.value })}
            className="w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {BUSINESS_TYPES.map((bt) => (
              <option key={bt.value} value={bt.value}>
                {bt.label}
              </option>
            ))}
          </select>
        </div>

        {/* Multi-Entity Behavior */}
        <div>
          <label className="block text-xs font-medium text-gray-400 mb-1.5">
            Multi-Entity Handling
          </label>
          <select
            value={state.multiEntityBehavior}
            onChange={(e) => update({ multiEntityBehavior: e.target.value })}
            className="w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {MULTI_ENTITY_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        {/* Model */}
        <div>
          <label className="block text-xs font-medium text-gray-400 mb-1.5">
            Model
          </label>
          <select
            value={state.model}
            onChange={(e) => update({ model: e.target.value })}
            className="w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {MODEL_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="px-5 py-4 border-t border-gray-800 space-y-2">
        <button
          onClick={onRun}
          disabled={loading}
          className="w-full rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <svg
                className="animate-spin h-4 w-4"
                viewBox="0 0 24 24"
                fill="none"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                />
              </svg>
              Running...
            </span>
          ) : (
            "Run Analysis"
          )}
        </button>
        <button
          onClick={onClear}
          className="w-full rounded-lg border border-gray-700 bg-gray-800 px-4 py-2 text-sm font-medium text-gray-300 hover:bg-gray-700 transition-colors"
        >
          Clear
        </button>
      </div>
    </aside>
  );
};

export default Sidebar;
