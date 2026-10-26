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
}

export interface HistoryItem {
  id: string;
  question: string;
  timestamp: string;
  response: AnalyzeResponse;
}
