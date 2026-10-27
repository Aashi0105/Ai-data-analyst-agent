export interface ColumnSchema {
  column: string;
  type: string;
  missing_count: number;
  missing_pct: number;
}

export interface NumericColumnStats {
  count: number;
  missing_count: number;
  missing_pct: number;
  min?: number | null;
  max?: number | null;
  mean?: number | null;
  median?: number | null;
  std?: number | null;
  q25?: number | null;
  q75?: number | null;
}

export interface CategoryFrequency {
  value: string;
  count: number;
  pct: number;
}

export interface CategoricalColumnStats {
  count: number;
  missing_count: number;
  missing_pct: number;
  unique_count: number;
  top_categories: CategoryFrequency[];
}

export interface DateColumnStats {
  count: number;
  missing_count: number;
  missing_pct: number;
  min_date?: string | null;
  max_date?: string | null;
}

export interface ColumnProfile {
  name: string;
  dtype: string;
  semantic_type: "numeric" | "categorical" | "datetime" | "boolean" | "text" | "empty";
  missing_count: number;
  missing_pct: number;
  numeric_stats?: NumericColumnStats | null;
  categorical_stats?: CategoricalColumnStats | null;
  date_stats?: DateColumnStats | null;
}

export interface DataQualityWarning {
  code: string;
  column?: string | null;
  message: string;
  severity: "info" | "warning" | "critical";
}

export interface DatasetProfile {
  num_rows: number;
  num_cols: number;
  total_missing: number;
  missing_cell_pct: number;
  duplicate_rows: number;
  duplicate_row_pct: number;
  columns: string[];
  column_profiles: Record<string, ColumnProfile>;
  warnings: DataQualityWarning[];
}

export interface ColumnInsight {
  column: string;
  inferred_meaning: string;
}

export interface DatasetUnderstanding {
  overview: string;
  column_insights: ColumnInsight[];
  analytical_potential: string;
  data_quality_notes: string;
  suggested_questions: string[];
  status: "generated" | "fallback";
}

export interface InsightEvidence {

  metric_name: string;
  relevant_columns: string[];
  key_figures: Record<string, any>;
  sample_size?: number | null;
  comparison_basis?: string | null;
}

export interface Insight {
  id: string;
  type: "segmentation" | "trend" | "outlier" | "correlation" | "data_quality";
  headline: string;
  explanation: string;
  why_it_matters: string;
  uncertainty_or_limitations?: string | null;
  evidence: InsightEvidence;
  visualization?: VisualizationData | null;
  suggested_query?: string | null;
  score: number;
}

export interface DatasetMetadata {
  dataset_id: string;
  filename: string;
  num_rows: number;
  num_cols: number;
  total_missing: number;
  duplicate_rows: number;
  columns: string[];
  dtypes: Record<string, string>;
  schema: ColumnSchema[];
  preview: Record<string, any>[];
  profile?: DatasetProfile;
  understanding?: DatasetUnderstanding;
  insights?: Insight[];
}


export interface TableData {
  columns: string[];
  rows: Record<string, any>[];
}

export interface VisualizationData {
  chart_type: "Bar Chart" | "Line Chart" | "Scatter Plot" | "Histogram" | string;
  data: Record<string, any>[];
  x_key: string;
  y_keys: string[];
  title: string;
}

export interface AnswerEvidence {
  metric_name?: string | null;
  aggregation_type: string;
  is_additive: boolean;
  total_value?: number | null;
  denominator_description?: string | null;
  top_contributor?: string | null;
  top_value?: number | null;
  top_share_pct?: number | null;
  runner_up?: string | null;
  runner_up_value?: number | null;
  runner_up_share_pct?: number | null;
  bottom_contributor?: string | null;
  bottom_value?: number | null;
  disparity_ratio?: number | null;
  disparity_difference?: number | null;
  num_groups?: number | null;
  sample_size?: number | null;
  baseline_value?: number | null;
  baseline_description?: string | null;
  key_figures?: Record<string, any>;
  limitation_notes?: string | null;
}

export interface AnswerExplanation {
  finding: string;
  evidence: AnswerEvidence;
  interpretation: string;
  uncertainty_or_limitations?: string | null;
  next_steps: string[];
  status: "deterministic" | "synthesized";
}

export interface AnalyzeResponse {
  question: string;
  code: string;
  validation_status: "passed" | "failed";
  validation_message: string;
  execution_status: "passed" | "failed";
  error_message?: string | null;
  result_type?: "scalar" | "dataframe" | "series" | "dict" | "list" | "other" | null;
  formatted_result?: string | null;
  table_data?: TableData | null;
  visualization?: VisualizationData | null;
  execution_time_ms: number;
  explanation?: AnswerExplanation | null;
}

export interface HistoryItem {
  id: string;
  question: string;
  timestamp: string;
  response: AnalyzeResponse;
}
