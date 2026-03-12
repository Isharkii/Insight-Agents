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
      <div className="relative z-20 flex w-14 shrink-0 flex-col items-center border-r border-slate-300/60 bg-slate-100/80 py-4 backdrop-blur">
        <button
          onClick={onToggleCollapse}
          className="rounded-lg border border-slate-300 bg-white/80 p-2 text-slate-500 transition-colors hover:text-slate-700"
          title="Expand sidebar"
          aria-label="Expand sidebar"
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
    <aside className="relative z-20 flex h-screen w-[300px] max-w-[88vw] shrink-0 flex-col border-r border-slate-300/60 bg-white/84 backdrop-blur">
      <div className="flex items-center justify-between border-b border-slate-300/70 px-5 py-4">
        <div>
          <p className="ia-label">Controls</p>
          <p className="mt-1 text-xs text-slate-500">Execution and context settings</p>
        </div>
        <button
          onClick={onToggleCollapse}
          className="rounded-lg border border-slate-300 bg-white/80 p-1.5 text-slate-500 transition-colors hover:text-slate-700"
          title="Collapse sidebar"
          aria-label="Collapse sidebar"
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

      <div className="flex-1 space-y-5 overflow-y-auto px-5 py-4">
        <div>
          <label className="ia-label mb-1.5 block">Mode</label>
          <div className="grid grid-cols-2 overflow-hidden rounded-xl border border-slate-300 bg-white/70">
            {(["LOCAL", "CLOUD"] as const).map((m) => (
              <button
                key={m}
                onClick={() => update({ mode: m })}
                className={`px-3 py-2 text-xs font-semibold transition-colors ${
                  state.mode === m
                    ? "bg-teal-700 text-white"
                    : "bg-transparent text-slate-600 hover:text-slate-800"
                }`}
              >
                {m}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="ia-label mb-1.5 block">Client</label>
          <select
            value={state.clientId}
            onChange={(e) => update({ clientId: e.target.value })}
            className="ia-select"
          >
            {clients.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="ia-label mb-1.5 block">Entity Override</label>
          <input
            type="text"
            value={state.entityOverride}
            onChange={(e) => update({ entityOverride: e.target.value })}
            placeholder="Override entity name"
            className="ia-input"
          />
        </div>

        <div>
          <label className="ia-label mb-1.5 block">Business Type</label>
          <select
            value={state.businessType}
            onChange={(e) => update({ businessType: e.target.value })}
            className="ia-select"
          >
            {BUSINESS_TYPES.map((bt) => (
              <option key={bt.value} value={bt.value}>
                {bt.label}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="ia-label mb-1.5 block">Multi-Entity Handling</label>
          <select
            value={state.multiEntityBehavior}
            onChange={(e) => update({ multiEntityBehavior: e.target.value })}
            className="ia-select"
          >
            {MULTI_ENTITY_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="ia-label mb-1.5 block">Model</label>
          <select
            value={state.model}
            onChange={(e) => update({ model: e.target.value })}
            className="ia-select"
          >
            {MODEL_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="space-y-2 border-t border-slate-300/70 px-5 py-4">
        <button
          onClick={onRun}
          disabled={loading}
          className="ia-btn-primary h-11 w-full"
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
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
              Running
            </span>
          ) : (
            "Run Analysis"
          )}
        </button>
        <button onClick={onClear} className="ia-btn-secondary h-10 w-full">
          Clear
        </button>
      </div>
    </aside>
  );
};

export default Sidebar;
