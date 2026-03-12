import { useState, useCallback, type FC } from "react";
import type { AnalyzeResult } from "../api/client";
import {
  fetchExportBlob,
  fetchBIWorkbookBlob,
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
    className="ia-btn-secondary h-11 min-w-[170px] flex-1 px-4"
  >
    {loading ? (
      <svg className="h-4 w-4 animate-spin text-slate-500" viewBox="0 0 24 24" fill="none">
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
        className="h-4 w-4 text-slate-500"
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
      // No-op by design: caller surface already indicates availability.
    } finally {
      setLoadingCsv(false);
    }
  }, [entityName]);

  const handlePbi = useCallback(async () => {
    if (!entityName || !prompt) return;
    setLoadingPbi(true);
    try {
      const blob = await fetchBIWorkbookBlob(entityName, prompt, businessType);
      downloadBlob(blob, `${entityName}_insight_workbook.xlsx`);
    } catch {
      // No-op by design: caller surface already indicates availability.
    } finally {
      setLoadingPbi(false);
    }
  }, [entityName, prompt, businessType]);

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
      // No-op by design: caller surface already indicates availability.
    } finally {
      setLoadingReport(false);
    }
  }, [entityName, prompt, businessType]);

  return (
    <div className="ia-surface px-5 py-5 sm:px-6">
      <div className="mb-3 flex items-end justify-between gap-3">
        <div>
          <p className="ia-label">Export</p>
          <p className="ia-subtitle mt-1">Download analysis artifacts in operational formats.</p>
        </div>
        <span className="ia-chip">4 formats</span>
      </div>
      <div className="flex flex-wrap gap-2.5">
        <ExportBtn label="JSON" loading={loadingJson} onClick={handleJson} />
        <ExportBtn
          label="CSV Records"
          loading={loadingCsv}
          disabled={!entityName}
          onClick={handleCsv}
        />
        <ExportBtn
          label="BI Workbook (.xlsx)"
          loading={loadingPbi}
          disabled={!entityName || !prompt}
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
