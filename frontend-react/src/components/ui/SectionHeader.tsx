import type { FC, ReactNode } from "react";

interface SectionHeaderProps {
  title: string;
  subtitle?: string;
  /** Slot for badges / meta displayed to the right. */
  action?: ReactNode;
}

/** Consistent section header with title, optional subtitle, and right-side action slot. */
const SectionHeader: FC<SectionHeaderProps> = ({ title, subtitle, action }) => (
  <div className="flex items-center justify-between gap-3 mb-4">
    <div>
      <h3 className="ia-subhead">{title}</h3>
      {subtitle && <p className="ia-caption mt-0.5">{subtitle}</p>}
    </div>
    {action && <div className="flex items-center gap-2 shrink-0">{action}</div>}
  </div>
);

export default SectionHeader;
