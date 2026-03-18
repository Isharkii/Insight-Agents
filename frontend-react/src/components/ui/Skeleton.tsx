import type { FC } from "react";

interface SkeletonProps {
  /** Height in px or CSS string. */
  height?: number | string;
  /** Width — defaults to 100%. */
  width?: number | string;
  /** Border radius override. */
  rounded?: string;
}

/** Shimmer loading placeholder. */
const Skeleton: FC<SkeletonProps> = ({
  height = 16,
  width = "100%",
  rounded,
}) => (
  <div
    className="ia-skeleton"
    style={{
      height: typeof height === "number" ? `${height}px` : height,
      width: typeof width === "number" ? `${width}px` : width,
      borderRadius: rounded,
    }}
  />
);

export default Skeleton;
