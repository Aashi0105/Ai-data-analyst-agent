export interface ColumnSchema {
  column: string;
  type: string;
  missing_count: number;
  missing_pct: number;
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
