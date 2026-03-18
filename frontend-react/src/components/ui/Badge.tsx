import type { FC } from "react";

type Variant = "success" | "warning" | "danger" | "info" | "neutral";

const variantClasses: Record<Variant, string> = {
  success: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300",
  warning: "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300",
  danger: "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300",
  info: "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300",
  neutral: "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300",
};

interface BadgeProps {
  children: React.ReactNode;
  variant?: Variant;
  /** Show a leading dot indicator. */
  dot?: boolean;
}

/** Reusable status / priority badge. */
const Badge: FC<BadgeProps> = ({ children, variant = "neutral", dot = false }) => (
  <span
    className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold capitalize ${variantClasses[variant]}`}
  >
    {dot && (
      <span
        className={`w-1.5 h-1.5 rounded-full ${
          variant === "success" ? "bg-emerald-500"
          : variant === "warning" ? "bg-amber-500"
          : variant === "danger" ? "bg-red-500"
          : variant === "info" ? "bg-blue-500"
          : "bg-gray-400"
        }`}
      />
    )}
    {children}
  </span>
);

export default Badge;

export function priorityVariant(priority: string): Variant {
  if (priority === "critical") return "danger";
  if (priority === "high") return "warning";
  if (priority === "medium") return "info";
  return "neutral";
}

export function statusVariant(status: string): Variant {
  if (status === "success") return "success";
  if (status === "partial") return "warning";
  return "danger";
}
