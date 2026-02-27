import { useState, useCallback, useEffect } from "react";
import IntelligenceDashboard from "./components/IntelligenceDashboard";
import type { DashboardData } from "./components/IntelligenceDashboard/types";
import { fetchDashboard, runAnalysis } from "./api/client";
import type { AnalyzeResult } from "./api/client";

const BUSINESS_TYPES = [
  { value: "saas", label: "SaaS" },
  { value: "ecommerce", label: "E-Commerce" },
  { value: "agency", label: "Agency" },
  { value: "marketing_analytics", label: "Marketing Analytics" },
  { value: "healthcare", label: "Healthcare" },
  { value: "retail", label: "Retail" },
  { value: "financial_markets", label: "Financial Markets" },
  { value: "operations", label: "Operations" },
  { value: "general_timeseries", label: "General Timeseries" },
];

function useEmbedParams() {
  const params = new URLSearchParams(window.location.search);
  const embed = params.get("embed") === "1";
  const entity = params.get("entity_name") ?? "";
  const btype = params.get("business_type") ?? "saas";
  return { embed, entity, btype };
}

export default function App() {
  const { embed, entity: embedEntity, btype: embedBtype } = useEmbedParams();

  const [entityName, setEntityName] = useState(embedEntity || "");
  const [businessType, setBusinessType] = useState(embedBtype || "saas");
  const [prompt, setPrompt] = useState("");
  const [file, setFile] = useState<File | null>(null);

  const [dashboardData, setDashboardData] = useState<DashboardData | null>(
    null,
  );
  const [analyzeResult, setAnalyzeResult] = useState<AnalyzeResult | null>(
    null,
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleLoadDashboard = useCallback(async () => {
    if (!entityName.trim()) {
      setError("Entity name is required.");
      return;
    }
    setLoading(true);
    setError(null);
    setAnalyzeResult(null);
    try {
      const data = await fetchDashboard(entityName.trim(), businessType);
      setDashboardData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [entityName, businessType]);

  const handleAnalyze = useCallback(async () => {
    if (!prompt.trim()) {
      setError("Prompt is required for analysis.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const result = await runAnalysis({
        prompt: prompt.trim(),
        file: file ?? undefined,
        clientId: entityName.trim() || undefined,
        businessType,
      });
      setAnalyzeResult(result);
      if (entityName.trim()) {
        const data = await fetchDashboard(entityName.trim(), businessType);
        setDashboardData(data);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [prompt, file, entityName, businessType]);

  // Embed mode: auto-load dashboard on mount
  useEffect(() => {
    if (embed && embedEntity) {
      setLoading(true);
      setError(null);
      fetchDashboard(embedEntity, embedBtype)
        .then(setDashboardData)
        .catch((err) =>
          setError(err instanceof Error ? err.message : String(err)),
        )
        .finally(() => setLoading(false));
    }
  }, [embed, embedEntity, embedBtype]);

  // Embed mode: render only the dashboard (no controls)
  if (embed) {
    return (
      <div className="bg-gray-50 dark:bg-gray-950">
        {loading && (
          <div className="flex items-center justify-center py-16">
            <p className="text-gray-400 dark:text-gray-500">
              Loading dashboard...
            </p>
          </div>
        )}
        {error && (
          <div className="max-w-7xl mx-auto px-4 py-4">
            <div className="rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 p-4 text-sm text-red-700 dark:text-red-300">
              {error}
            </div>
          </div>
        )}
        {dashboardData && <IntelligenceDashboard data={dashboardData} />}
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950">
      {/* Control Panel */}
      <div className="border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <h1 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-5">
            InsightAgent
          </h1>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Entity Name
              </label>
              <input
                type="text"
                value={entityName}
                onChange={(e) => setEntityName(e.target.value)}
                placeholder="e.g. Acme Corp"
                className="w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 text-sm text-gray-900 dark:text-gray-100 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Business Type
              </label>
              <select
                value={businessType}
                onChange={(e) => setBusinessType(e.target.value)}
                className="w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 text-sm text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {BUSINESS_TYPES.map((bt) => (
                  <option key={bt.value} value={bt.value}>
                    {bt.label}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                CSV File (optional)
              </label>
              <input
                type="file"
                accept=".csv"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                className="w-full text-sm text-gray-500 dark:text-gray-400 file:mr-3 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-medium file:bg-blue-50 file:text-blue-700 dark:file:bg-blue-900/30 dark:file:text-blue-300 hover:file:bg-blue-100"
              />
            </div>

            <div className="flex items-end gap-2">
              <button
                onClick={handleLoadDashboard}
                disabled={loading}
                className="flex-1 rounded-lg bg-gray-900 dark:bg-gray-100 px-4 py-2 text-sm font-medium text-white dark:text-gray-900 hover:bg-gray-700 dark:hover:bg-gray-300 disabled:opacity-50 transition-colors"
              >
                {loading ? "Loading\u2026" : "Load Dashboard"}
              </button>
            </div>
          </div>

          <div className="flex gap-3">
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Enter your analysis prompt\u2026"
              rows={2}
              className="flex-1 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 text-sm text-gray-900 dark:text-gray-100 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
            />
            <button
              onClick={handleAnalyze}
              disabled={loading}
              className="self-end rounded-lg bg-blue-600 px-6 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50 transition-colors"
            >
              {loading ? "Analyzing\u2026" : "Analyze"}
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-4">
          <div className="rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 p-4 text-sm text-red-700 dark:text-red-300">
            {error}
          </div>
        </div>
      )}

      {analyzeResult && !dashboardData && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6">
          <div className="bg-white dark:bg-gray-900 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-800 p-6">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
              Analysis Result
            </h2>
            <div className="space-y-3 text-sm">
              <div>
                <span className="font-medium text-gray-500 dark:text-gray-400">
                  Insight:{" "}
                </span>
                <span className="text-gray-800 dark:text-gray-200">
                  {analyzeResult.insight}
                </span>
              </div>
              <div>
                <span className="font-medium text-gray-500 dark:text-gray-400">
                  Evidence:{" "}
                </span>
                <span className="text-gray-800 dark:text-gray-200">
                  {analyzeResult.evidence}
                </span>
              </div>
              <div>
                <span className="font-medium text-gray-500 dark:text-gray-400">
                  Impact:{" "}
                </span>
                <span className="text-gray-800 dark:text-gray-200">
                  {analyzeResult.impact}
                </span>
              </div>
              <div>
                <span className="font-medium text-gray-500 dark:text-gray-400">
                  Action:{" "}
                </span>
                <span className="text-gray-800 dark:text-gray-200">
                  {analyzeResult.recommended_action}
                </span>
              </div>
              <div className="flex gap-4">
                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-300">
                  {analyzeResult.priority}
                </span>
                <span className="text-gray-500 dark:text-gray-400">
                  Confidence:{" "}
                  {(analyzeResult.confidence_score * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {dashboardData && <IntelligenceDashboard data={dashboardData} />}

      {!loading && !error && !dashboardData && !analyzeResult && (
        <div className="flex items-center justify-center py-32">
          <p className="text-gray-400 dark:text-gray-500 text-lg">
            Enter an entity name and click "Load Dashboard" to begin.
          </p>
        </div>
      )}
    </div>
  );
}
