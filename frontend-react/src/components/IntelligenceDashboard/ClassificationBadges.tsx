import type { FC } from "react";
import type { ClassificationData } from "./types";

interface ClassificationBadgesProps {
  classification: ClassificationData;
}

function badgeStyle(confidence: number): string {
  if (confidence >= 85) {
    return "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300 border-emerald-200 dark:border-emerald-800";
  }
  if (confidence >= 60) {
    return "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300 border-amber-200 dark:border-amber-800";
  }
  return "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300 border-gray-200 dark:border-gray-700";
}

const ClassificationBadges: FC<ClassificationBadgesProps> = ({ classification }) => {
  if (!classification) return null;

  return (
    <div className="bg-white dark:bg-gray-900 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-800 p-6">
      <h3 className="text-sm font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-4">
        Classification
      </h3>
      <div className="flex flex-wrap gap-2">
        <span
          className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium border ${badgeStyle(classification.confidence)}`}
        >
          <span>{classification.label}</span>
          <span className="text-xs opacity-70">
            {classification.confidence}%
          </span>
        </span>
      </div>
    </div>
  );
};

export default ClassificationBadges;
