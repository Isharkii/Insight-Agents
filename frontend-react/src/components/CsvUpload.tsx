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
      <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
        Historical Data Upload
      </h3>

      {/* Drop zone */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={`relative border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${
          dragging
            ? "border-blue-500 bg-blue-50 dark:bg-blue-900/10"
            : file
              ? "border-emerald-300 dark:border-emerald-700 bg-emerald-50 dark:bg-emerald-900/10"
              : "border-gray-300 dark:border-gray-700 hover:border-gray-400 dark:hover:border-gray-600"
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
          <div className="flex items-center justify-center gap-3">
            <svg
              className="w-6 h-6 text-emerald-500"
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
              <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                {file.name}
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                {(file.size / 1024).toFixed(1)} KB
              </p>
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleFile(null);
              }}
              className="ml-4 text-gray-400 hover:text-red-500 transition-colors"
            >
              <svg
                className="w-5 h-5"
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
              className="mx-auto h-10 w-10 text-gray-400"
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
            <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
              Drop a CSV file here or click to browse
            </p>
          </div>
        )}
      </div>

      {/* Preview */}
      {preview && preview.headers.length > 0 && (
        <div className="space-y-3">
          {/* Row slider */}
          <div className="flex items-center gap-3">
            <label className="text-xs text-gray-500 dark:text-gray-400 whitespace-nowrap">
              Preview rows
            </label>
            <input
              type="range"
              min={5}
              max={100}
              step={5}
              value={previewRows}
              onChange={(e) => handleRowSlider(Number(e.target.value))}
              className="flex-1 accent-blue-600"
            />
            <span className="text-xs text-gray-500 dark:text-gray-400 w-8 text-right">
              {previewRows}
            </span>
          </div>

          {/* Column list */}
          <div className="flex flex-wrap gap-1.5">
            {preview.headers.map((col) => (
              <span
                key={col}
                className="inline-block px-2 py-0.5 text-xs rounded-full bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 border border-gray-200 dark:border-gray-700"
              >
                {col}
              </span>
            ))}
          </div>

          {/* Table */}
          <div className="border border-gray-200 dark:border-gray-700 rounded-xl overflow-hidden">
            <div className="overflow-x-auto max-h-80">
              <table className="min-w-full text-xs">
                <thead className="bg-gray-50 dark:bg-gray-800 sticky top-0">
                  <tr>
                    <th className="px-3 py-2 text-left font-medium text-gray-500 dark:text-gray-400 w-10">
                      #
                    </th>
                    {preview.headers.map((h) => (
                      <th
                        key={h}
                        className="px-3 py-2 text-left font-medium text-gray-500 dark:text-gray-400 whitespace-nowrap"
                      >
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100 dark:divide-gray-800">
                  {preview.rows.map((row, i) => (
                    <tr
                      key={i}
                      className="hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors"
                    >
                      <td className="px-3 py-1.5 text-gray-400">{i + 1}</td>
                      {row.map((cell, j) => (
                        <td
                          key={j}
                          className="px-3 py-1.5 text-gray-700 dark:text-gray-300 whitespace-nowrap"
                        >
                          {cell}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="px-3 py-1.5 bg-gray-50 dark:bg-gray-800 text-xs text-gray-500 dark:text-gray-400 border-t border-gray-200 dark:border-gray-700">
              Showing {preview.rows.length} of {preview.totalRows} rows
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CsvUpload;
