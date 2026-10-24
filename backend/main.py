"""
backend/main.py
---------------
FastAPI REST API layer for the AI Data Analyst Agent.
Integrates the existing Python analysis core (agent.py, validator.py,
sandbox.py, charts.py, utils.py) into a clean, strongly-typed HTTP interface.
"""

import io
import os
import sys
import uuid
import time
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

# Ensure workspace root is in python path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import re
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from utils import load_csv, get_dataset_summary, format_value
from agent import DataAnalystAgent
from validator import clean_code, validate_code
from sandbox import execute_pandas_code
from backend.chart_utils import build_recharts_data

# ============================================================================
# Upload Constraints
# ============================================================================
MAX_UPLOAD_SIZE_MB = 25
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024  # 26,214,400 bytes
UPLOAD_CHUNK_SIZE_BYTES = 1024 * 1024  # 1 MiB chunk reading

# ============================================================================
# In-Memory Dataset Store
# NOTE: Server-memory based session storage. Suitable for development/demo.
# Not persistent across server restarts or multi-instance deployments.
# ============================================================================
DATASET_STORE: Dict[str, Dict[str, Any]] = {}

# FastAPI App Initialization
app = FastAPI(
    title="AI Data Analyst Agent API",
    description="REST API for natural-language data analysis, AST code validation, and isolated execution.",
    version="1.0.0"
)

# CORS Configuration (Dynamic with sensible development fallback)
frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
allowed_origins = [origin.strip() for origin in frontend_origin.split(",") if origin.strip()]
for default_origin in [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]:
    if default_origin not in allowed_origins:
        allowed_origins.append(default_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Pydantic Request & Response Models
# ============================================================================
class ColumnSchema(BaseModel):
    column: str
    type: str
    missing_count: int
    missing_pct: float


class DatasetMetadataResponse(BaseModel):
    dataset_id: str
    filename: str
    num_rows: int
    num_cols: int
    total_missing: int
    duplicate_rows: int
    columns: List[str]
    dtypes: Dict[str, str]
    schema_info: List[ColumnSchema] = Field(..., serialization_alias="schema")
    preview: List[Dict[str, Any]]


class AnalyzeRequest(BaseModel):
    dataset_id: str = Field(..., description="UUID of the uploaded dataset")
    question: str = Field(..., min_length=1, description="Natural language analytical query")


class TableData(BaseModel):
    columns: List[str]
    rows: List[Dict[str, Any]]


class VisualizationData(BaseModel):
    chart_type: str
    data: List[Dict[str, Any]]
    x_key: str
    y_keys: List[str]
    title: str


class AnalyzeResponse(BaseModel):
    question: str
    code: str
    validation_status: str  # "passed" | "failed"
    validation_message: str
    execution_status: str   # "passed" | "failed"
    error_message: Optional[str] = None
    result_type: Optional[str] = None  # "scalar" | "dataframe" | "series" | "dict" | "empty"
    formatted_result: Optional[str] = None
    table_data: Optional[TableData] = None
    visualization: Optional[VisualizationData] = None
    execution_time_ms: int


# ============================================================================
def _sanitize_error_message(msg: str) -> str:
    """Removes potential API keys, query params, and raw traceback dumps from error text."""
    if not msg:
        return "An internal error occurred."
    # Remove Google API keys (AIza...)
    cleaned = re.sub(r"AIza[0-9A-Za-z_-]{35}", "[REDACTED_API_KEY]", str(msg))
    # Remove key query parameters
    cleaned = re.sub(r"key=[0-9A-Za-z_-]+", "key=[REDACTED]", cleaned)
    # Truncate internal stack traces to first meaningful line
    first_line = cleaned.strip().split("\n")[0]
    return first_line[:200]


def _format_dataset_response(dataset_id: str, filename: str, df: pd.DataFrame) -> DatasetMetadataResponse:
    """Extracts summary and formats Pydantic response."""
    summary = get_dataset_summary(df)

    schema_list = []
    for col in df.columns:
        schema_list.append(ColumnSchema(
            column=str(col),
            type=str(df[col].dtype),
            missing_count=int(df[col].isnull().sum()),
            missing_pct=round((float(df[col].isnull().sum()) / len(df)) * 100, 2) if len(df) > 0 else 0.0
        ))

    # Convert head(10) to json-safe records
    preview_df = df.head(10).replace({np.nan: None})
    preview_records = preview_df.to_dict(orient="records")

    return DatasetMetadataResponse(
        dataset_id=dataset_id,
        filename=filename,
        num_rows=summary["num_rows"],
        num_cols=summary["num_cols"],
        total_missing=summary["total_missing"],
        duplicate_rows=summary["duplicate_rows"],
        columns=[str(c) for c in df.columns],
        dtypes={str(k): str(v) for k, v in summary["dtypes"].items()},
        schema_info=schema_list,
        preview=preview_records
    )


# ============================================================================
# API Routes
# ============================================================================
@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "AI Data Analyst Agent API",
        "version": "1.0.0",
        "datasets_in_memory": len(DATASET_STORE)
    }


@app.post("/api/datasets/upload", response_model=DatasetMetadataResponse)
async def upload_dataset(request: Request, file: UploadFile = File(...)) -> DatasetMetadataResponse:
    """
    Ingests a user-uploaded CSV file, profiles the schema,
    and stores the DataFrame in in-memory session storage.
    Enforces a strict maximum file size of 25 MiB via bounded chunk reading.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files (.csv) are currently supported.")

    # Early rejection if Content-Length header is present and significantly exceeds max allowed limit + multipart framing
    MULTIPART_OVERHEAD_ALLOWANCE = 64 * 1024  # 64 KiB allowance for multipart form headers and boundaries
    content_length_header = request.headers.get("content-length")
    if content_length_header and content_length_header.isdigit():
        if int(content_length_header) > (MAX_UPLOAD_SIZE_BYTES + MULTIPART_OVERHEAD_ALLOWANCE):
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds the maximum allowed limit of {MAX_UPLOAD_SIZE_MB} MiB."
            )

    buffer = io.BytesIO()
    total_bytes = 0

    try:
        while True:
            chunk = await file.read(UPLOAD_CHUNK_SIZE_BYTES)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_SIZE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File size exceeds the maximum allowed limit of {MAX_UPLOAD_SIZE_MB} MiB."
                )
            buffer.write(chunk)

        buffer.seek(0)
        df = load_csv(buffer)
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process CSV: {str(e)}")
    finally:
        buffer.close()
        await file.close()

    dataset_id = str(uuid.uuid4())
    DATASET_STORE[dataset_id] = {
        "df": df,
        "filename": file.filename,
        "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    return _format_dataset_response(dataset_id, file.filename, df)


@app.post("/api/datasets/sample", response_model=DatasetMetadataResponse)
def load_sample_dataset() -> DatasetMetadataResponse:
    """
    Loads the bundled synthetic evaluation benchmark dataset (data/ecommerce_sales.csv)
    for instant one-click demonstration.
    """
    sample_path = os.path.join(WORKSPACE_ROOT, "data", "ecommerce_sales.csv")
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail="Sample dataset not found on server.")

    try:
        df = load_csv(sample_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load sample dataset: {str(e)}")

    dataset_id = str(uuid.uuid4())
    DATASET_STORE[dataset_id] = {
        "df": df,
        "filename": "ecommerce_sales.csv (Sample)",
        "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    return _format_dataset_response(dataset_id, "ecommerce_sales.csv (Sample)", df)


@app.get("/api/datasets/{dataset_id}", response_model=DatasetMetadataResponse)
def get_dataset(dataset_id: str) -> DatasetMetadataResponse:
    """Retrieves dataset summary and schema for an active session."""
    if dataset_id not in DATASET_STORE:
        raise HTTPException(status_code=404, detail=f"Dataset with ID '{dataset_id}' was not found in server memory.")

    stored = DATASET_STORE[dataset_id]
    return _format_dataset_response(dataset_id, stored["filename"], stored["df"])


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze_data(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Translates a plain-English question into Pandas code, runs AST safety verification,
    executes inside the isolated sandbox, and formats results & Recharts visualization data.
    """
    if request.dataset_id not in DATASET_STORE:
        raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset_id}' not found. Please upload a CSV first.")

    df = DATASET_STORE[request.dataset_id]["df"]
    user_question = request.question.strip()

    start_time = time.time()
    generated_code = ""

    # Stage 1: Generate Code using existing DataAnalystAgent
    try:
        agent = DataAnalystAgent()
        generated_code = agent.generate_code(df, user_question)
        cleaned_code = clean_code(generated_code)
    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        err_str = str(e)
        
        # Detect Gemini quota / rate limit exhaustion (HTTP 429 / RESOURCE_EXHAUSTED)
        is_quota = any(marker in err_str for marker in [
            "429", "RESOURCE_EXHAUSTED", "GenerateRequestsPerDay", "ResourceExhausted", "quota", "usage limit"
        ])
        if is_quota:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "quota_exceeded",
                    "message": "AI analysis is temporarily unavailable because the current API usage limit has been reached.",
                    "retryable": False
                }
            )

        sanitized = _sanitize_error_message(err_str)
        return AnalyzeResponse(
            question=user_question,
            code=generated_code,
            validation_status="failed",
            validation_message="Code generation failed.",
            execution_status="failed",
            error_message=f"Code generation error: {sanitized}",
            execution_time_ms=elapsed_ms
        )

    # Stage 2: AST Validation
    is_valid, validation_msg = validate_code(cleaned_code)
    if not is_valid:
        elapsed_ms = int((time.time() - start_time) * 1000)
        return AnalyzeResponse(
            question=user_question,
            code=cleaned_code,
            validation_status="failed",
            validation_message=validation_msg,
            execution_status="failed",
            error_message=f"AST Security Violation: {validation_msg}",
            execution_time_ms=elapsed_ms
        )

    # Stage 3: Sandbox Execution (on deep copy)
    exec_success, result_or_error = execute_pandas_code(cleaned_code, df)
    elapsed_ms = int((time.time() - start_time) * 1000)

    if not exec_success:
        return AnalyzeResponse(
            question=user_question,
            code=cleaned_code,
            validation_status="passed",
            validation_message="Code passed AST validation.",
            execution_status="failed",
            error_message=str(result_or_error),
            execution_time_ms=elapsed_ms
        )

    # Stage 4: Structure Output and Visualization Data
    res = result_or_error
    result_type = "scalar"
    formatted_result = ""
    table_data = None
    viz_data = None

    if isinstance(res, (int, float, str, bool, np.number, np.bool_)):
        result_type = "scalar"
        formatted_result = format_value(res)

    elif isinstance(res, pd.DataFrame):
        result_type = "dataframe"
        formatted_result = f"Showing {len(res)} results."
        clean_df = res.head(50).replace({np.nan: None})
        table_data = TableData(
            columns=[str(c) for c in res.columns],
            rows=clean_df.to_dict(orient="records")
        )
        # Attempt to build structured chart data
        recharts_obj = build_recharts_data(res)
        if recharts_obj:
            viz_data = VisualizationData(**recharts_obj)

    elif isinstance(res, pd.Series):
        result_type = "series"
        formatted_result = f"Showing {len(res)} items."
        index_col = str(res.index.name or "Category")
        val_col = str(res.name or "Value")
        rows = []
        for idx, val in res.head(50).items():
            clean_val = None if pd.isnull(val) else (round(float(val), 2) if isinstance(val, (float, np.floating)) else val)
            rows.append({index_col: str(idx), val_col: clean_val})
        table_data = TableData(columns=[index_col, val_col], rows=rows)

        recharts_obj = build_recharts_data(res)
        if recharts_obj:
            viz_data = VisualizationData(**recharts_obj)

    elif isinstance(res, dict):
        result_type = "dict"
        formatted_result = f"{len(res)} keys returned."
        rows = [{"Key": str(k), "Value": str(v)} for k, v in res.items()]
        table_data = TableData(columns=["Key", "Value"], rows=rows)

    elif isinstance(res, (list, tuple, set)):
        result_type = "list"
        formatted_result = f"List of {len(res)} items."
        rows = [{"Item": str(item)} for item in list(res)[:50]]
        table_data = TableData(columns=["Item"], rows=rows)

    else:
        result_type = "other"
        formatted_result = str(res)

    return AnalyzeResponse(
        question=user_question,
        code=cleaned_code,
        validation_status="passed",
        validation_message="Code passed AST validation.",
        execution_status="passed",
        result_type=result_type,
        formatted_result=formatted_result,
        table_data=table_data,
        visualization=viz_data,
        execution_time_ms=elapsed_ms
    )
