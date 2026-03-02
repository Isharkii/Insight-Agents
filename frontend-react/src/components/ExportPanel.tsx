import { useState, useCallback, type FC } from "react";
import type { AnalyzeResult } from "../api/client";
import {
  fetchExportBlob,
  fetchReportMarkdownBlob,
} from "../api/client";

interface ExportPanelProps {
  result: AnalyzeResult;
  entityName?: string;
  prompt?: string;
  businessType?: string;
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, 100);
}

interface BtnProps {
  label: string;
  loading: boolean;
  disabled?: boolean;
  onClick: () => void;
}

const ExportBtn: FC<BtnProps> = ({ label, loading, disabled, onClick }) => (
  <button
    onClick={onClick}
    disabled={loading || disabled}
    className="flex-1 inline-flex items-center justify-center gap-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 px-4 py-2.5 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
  >
    {loading ? (
      <svg
        className="animate-spin h-4 w-4 text-gray-400"
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
    ) : (
      <svg
        className="w-4 h-4 text-gray-400"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth={2}
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
        />
      </svg>
    )}
    {label}
  </button>
);

const ExportPanel: FC<ExportPanelProps> = ({
  result,
  entityName,
  prompt,
  businessType,
}) => {
  const [loadingJson, setLoadingJson] = useState(false);
  const [loadingCsv, setLoadingCsv] = useState(false);
  const [loadingPbi, setLoadingPbi] = useState(false);
  const [loadingReport, setLoadingReport] = useState(false);

  const handleJson = useCallback(() => {
    setLoadingJson(true);
    try {
      const json = JSON.stringify(result, null, 2);
      const blob = new Blob([json], { type: "application/json" });
      downloadBlob(blob, "insight_output.json");
    } finally {
      setLoadingJson(false);
    }
  }, [result]);

  const handleCsv = useCallback(async () => {
    setLoadingCsv(true);
    try {
      const blob = await fetchExportBlob("records", "csv", entityName);
      downloadBlob(blob, "insight_records.csv");
    } catch {
      /* ignore */
    } finally {
      setLoadingCsv(false);
    }
  }, [entityName]);

  const handlePbi = useCallback(async () => {
    setLoadingPbi(true);
    try {
      const blob = await fetchExportBlob("kpis", "csv", entityName);
      downloadBlob(blob, "powerbi_dataset.csv");
    } catch {
      /* ignore */
    } finally {
      setLoadingPbi(false);
    }
  }, [entityName]);

  const handleReport = useCallback(async () => {
    if (!entityName || !prompt) return;
    setLoadingReport(true);
    try {
      const blob = await fetchReportMarkdownBlob(
        entityName,
        prompt,
        businessType,
      );
      downloadBlob(blob, "insight_report.md");
    } catch {
      /* ignore */
    } finally {
      setLoadingReport(false);
    }
  }, [entityName, prompt, businessType]);

  return (
    <div className="bg-white dark:bg-gray-900 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-800 p-5">
      <h3 className="text-sm font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-3">
        Export
      </h3>
      <div className="flex flex-wrap gap-2">
        <ExportBtn
          label="JSON"
          loading={loadingJson}
          onClick={handleJson}
        />
        <ExportBtn
          label="CSV Records"
          loading={loadingCsv}
          disabled={!entityName}
          onClick={handleCsv}
        />
        <ExportBtn
          label="PowerBI Dataset"
          loading={loadingPbi}
          disabled={!entityName}
          onClick={handlePbi}
        />
        <ExportBtn
          label="Report (.md)"
          loading={loadingReport}
          disabled={!entityName || !prompt}
          onClick={handleReport}
        />
      </div>
    </div>
  );
};

export default ExportPanel;
