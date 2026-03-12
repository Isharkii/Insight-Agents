import type { FC } from "react";
import type { ClassificationData } from "./types";

interface ClassificationBadgesProps {
  classification: ClassificationData;
}

function badgeStyle(confidence: number): string {
  if (confidence >= 85) {
    return "border-emerald-200 bg-emerald-50 text-emerald-700";
  }
  if (confidence >= 60) {
    return "border-amber-200 bg-amber-50 text-amber-700";
  }
  return "border-slate-200 bg-slate-100 text-slate-700";
}

const ClassificationBadges: FC<ClassificationBadgesProps> = ({ classification }) => {
  if (!classification) return null;

  return (
    <div className="ia-surface p-6">
      <p className="ia-label mb-4">Classification</p>
      <div className="flex flex-wrap gap-2">
        <span
          className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-sm font-medium ${badgeStyle(classification.confidence)}`}
        >
          <span>{classification.label}</span>
          <span className="text-xs opacity-75">{classification.confidence}%</span>
        </span>
      </div>
    </div>
  );
};

export default ClassificationBadges;
