/** TypeScript types aligned with the InsightAgent backend schemas. */

export interface TimeSeriesPoint {
  period: string;
  value: number;
}

export interface ProjectedTimeSeriesPoint extends TimeSeriesPoint {
  projected?: boolean;
}

export interface KPIMetricEntry {
  value: number;
  unit?: string;
  label?: string;
}

export interface ClassificationData {
  label: string;
  category: string;
  confidence: number;
}

export type InsightDimension = "kpi" | "macro" | "competitive" | "simulation";

export interface StructuredInsight {
  title: string;
  description: string;
  impact_score: number;
  dimension: InsightDimension;
}

export type InsightDirection = "positive" | "negative" | "neutral";

export interface InsightCardData {
  title: string;
  metric: string;
  impact_score: string;
  direction: InsightDirection;
}

export interface DashboardData {
  entity_name: string;
  business_type: string;
  health_index: number;
  health_label: string;
  kpi_metrics: Record<string, KPIMetricEntry>;
  revenue_trend: TimeSeriesPoint[];
  market_share: ProjectedTimeSeriesPoint[];
  classification: ClassificationData;
  insights: StructuredInsight[];
  competitive_benchmark?: Record<string, unknown> | null;
  insight_cards?: InsightCardData[];
  pipeline_errors?: string[];
}
