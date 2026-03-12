import { useState, useCallback, useRef, type FC, type DragEvent } from "react";

interface CsvUploadProps {
  file: File | null;
  onFileChange: (file: File | null) => void;
}

interface CsvPreview {
  headers: string[];
  rows: string[][];
  totalRows: number;
}

function parseCsvText(text: string, maxRows: number): CsvPreview {
  const lines = text.split(/\r?\n/).filter((l) => l.trim() !== "");
  if (lines.length === 0) return { headers: [], rows: [], totalRows: 0 };

  const parseLine = (line: string): string[] => {
    const result: string[] = [];
    let current = "";
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (ch === '"') {
        inQuotes = !inQuotes;
      } else if (ch === "," && !inQuotes) {
        result.push(current.trim());
        current = "";
      } else {
        current += ch;
      }
    }
    result.push(current.trim());
    return result;
  };

  const headers = parseLine(lines[0]);
  const dataLines = lines.slice(1);
  const rows = dataLines.slice(0, maxRows).map(parseLine);

  return { headers, rows, totalRows: dataLines.length };
}

const CsvUpload: FC<CsvUploadProps> = ({ file, onFileChange }) => {
  const [preview, setPreview] = useState<CsvPreview | null>(null);
  const [previewRows, setPreviewRows] = useState(20);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const loadPreview = useCallback(
    (f: File, rows: number) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target?.result;
        if (typeof text === "string") {
          setPreview(parseCsvText(text, rows));
        }
      };
      reader.readAsText(f);
    },
    [],
  );

  const handleFile = useCallback(
    (f: File | null) => {
      onFileChange(f);
      if (f) {
        loadPreview(f, previewRows);
      } else {
        setPreview(null);
      }
    },
    [onFileChange, loadPreview, previewRows],
  );

  const handleDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const f = e.dataTransfer.files?.[0];
      if (f && f.name.endsWith(".csv")) handleFile(f);
    },
    [handleFile],
  );

  const handleRowSlider = useCallback(
    (value: number) => {
      setPreviewRows(value);
      if (file) loadPreview(file, value);
    },
    [file, loadPreview],
  );

  return (
    <div className="space-y-4">
      <div className="flex items-end justify-between gap-3">
        <div>
          <p className="ia-label">Historical CSV Upload</p>
          <p className="ia-subtitle mt-1">Drop or browse a file to analyze retained metrics.</p>
        </div>
        {file && (
          <span className="ia-chip">
            {(file.size / 1024).toFixed(1)} KB
          </span>
        )}
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={`relative cursor-pointer rounded-2xl border-2 border-dashed p-8 text-center transition-colors ${
          dragging
            ? "border-teal-600 bg-teal-50"
            : file
              ? "border-teal-300 bg-teal-50/70"
              : "border-slate-300 bg-white/60 hover:border-slate-400"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".csv"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
        />

        {file ? (
          <div className="flex flex-wrap items-center justify-center gap-3">
            <svg
              className="h-6 w-6 text-teal-600"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <div className="text-left">
              <p className="text-sm font-semibold text-slate-800">{file.name}</p>
              <p className="text-xs text-slate-500">Click to replace this file</p>
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleFile(null);
              }}
              className="rounded-md p-1 text-slate-400 transition-colors hover:bg-white hover:text-red-500"
              aria-label="Remove file"
            >
              <svg
                className="h-5 w-5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>
        ) : (
          <div>
            <svg
              className="mx-auto h-10 w-10 text-slate-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.5}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
              />
            </svg>
            <p className="mt-2 text-sm text-slate-600">Drop a CSV file here or click to browse</p>
          </div>
        )}
      </div>

      {preview && preview.headers.length > 0 && (
        <div className="space-y-3 rounded-2xl border border-slate-300/70 bg-white/70 p-4">
          <div className="flex items-center gap-3">
            <label className="ia-label whitespace-nowrap">Preview Rows</label>
            <input
              type="range"
              min={5}
              max={100}
              step={5}
              value={previewRows}
              onChange={(e) => handleRowSlider(Number(e.target.value))}
              className="flex-1 accent-teal-700"
            />
            <span className="text-xs font-semibold text-slate-600">{previewRows}</span>
          </div>

          <div className="flex flex-wrap gap-1.5">
            {preview.headers.map((col) => (
              <span
                key={col}
                className="rounded-full border border-slate-300 bg-slate-50 px-2 py-0.5 text-xs text-slate-600"
              >
                {col}
              </span>
            ))}
          </div>

          <div className="overflow-hidden rounded-xl border border-slate-300/70 bg-white">
            <div className="max-h-80 overflow-x-auto">
              <table className="min-w-full text-xs">
                <thead className="sticky top-0 bg-slate-100">
                  <tr>
                    <th className="w-10 px-3 py-2 text-left font-semibold text-slate-500">#</th>
                    {preview.headers.map((h) => (
                      <th
                        key={h}
                        className="whitespace-nowrap px-3 py-2 text-left font-semibold text-slate-500"
                      >
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {preview.rows.map((row, i) => (
                    <tr key={i} className="transition-colors hover:bg-slate-50">
                      <td className="px-3 py-1.5 text-slate-400">{i + 1}</td>
                      {row.map((cell, j) => (
                        <td key={j} className="whitespace-nowrap px-3 py-1.5 text-slate-700">
                          {cell}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="border-t border-slate-200 bg-slate-50 px-3 py-1.5 text-xs text-slate-500">
              Showing {preview.rows.length} of {preview.totalRows} rows
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CsvUpload;
