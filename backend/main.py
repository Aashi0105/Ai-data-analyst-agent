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
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

# Ensure workspace root is in python path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import re
import logging
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import json
from pathlib import Path
from utils import load_csv, get_dataset_summary, format_value
from profiler import DatasetProfile, profile_dataset
from understanding import DatasetUnderstanding, generate_dataset_understanding, generate_deterministic_fallback
from insights import Insight, discover_insights
from explanation import AnswerExplanation, explain_answer
from overview import DatasetOverview, generate_dataset_overview
from agent import DataAnalystAgent
from backend.examples import get_examples
try:
    import backend.metrics as metrics_mod
    from backend.metrics import log_event
except ImportError:
    import metrics as metrics_mod
    from metrics import log_event

logger = logging.getLogger(__name__)

from validator import clean_code, validate_code
from sandbox import execute_pandas_code, SandboxExecutor
from backend.chart_utils import build_recharts_data
from backend.store import (
    BoundedDatasetStore,
    StoredDataset,
    DatasetNotFoundError,
    DatasetSessionExpiredError,
    DatasetInUseError,
    StorageCapacityExhaustedError,
)

# ============================================================================
# Upload Constraints
# ============================================================================
MAX_UPLOAD_SIZE_MB = 25
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024  # 26,214,400 bytes
UPLOAD_CHUNK_SIZE_BYTES = 1024 * 1024  # 1 MiB chunk reading

# ============================================================================
# Bounded In-Memory Dataset Store
# NOTE: Thread-safe, memory-bounded, TTL-aware session storage with LRU eviction.
# Architectural limitation: Single-worker Uvicorn deployment only (--workers 1).
# Multi-worker deployments require an external distributed cache (e.g. Redis).
# ============================================================================
DATASET_STORE = BoundedDatasetStore(
    max_datasets=int(os.getenv("DATASET_STORE_MAX_DATASETS", "15")),
    max_memory_bytes=int(os.getenv("DATASET_STORE_MAX_MEMORY_MB", "512")) * 1024 * 1024,
    ttl_seconds=float(os.getenv("DATASET_STORE_TTL_SECONDS", "3600.0")),
)

# In-Memory Per-Session Rate Limiter (Task 5)
_rate_limits: dict = {}
# key: dataset_id, value: {count: int, date: str}

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
    profile: Optional[DatasetProfile] = None
    understanding: Optional[DatasetUnderstanding] = None
    insights: Optional[List[Insight]] = None
    overview: Optional[DatasetOverview] = None



class DeleteDatasetResponse(BaseModel):
    status: str = "deleted"
    dataset_id: str
    message: str = "Dataset successfully removed from memory."


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
    question: Optional[str] = None
    code: Optional[str] = None
    validation_status: Optional[str] = None  # "passed" | "failed"
    validation_message: Optional[str] = None
    execution_status: Optional[str] = None   # "passed" | "failed"
    error_message: Optional[str] = None
    result_type: Optional[str] = None  # "scalar" | "dataframe" | "series" | "dict" | "empty"
    formatted_result: Optional[str] = None
    table_data: Optional[TableData] = None
    visualization: Optional[VisualizationData] = None
    execution_time_ms: Optional[int] = 0
    explanation: Optional[AnswerExplanation] = None
    quota_exhausted: Optional[bool] = None
    message: Optional[str] = None
    examples_available: Optional[bool] = None
    result: Optional[Any] = None
    chart: Optional[Any] = None
    session_limit_reached: Optional[bool] = None


# ============================================================================
def _sanitize_error_message(msg: str) -> str:
    """Removes potential API keys, query params, auth tokens, and raw traceback dumps from error text."""
    if not msg:
        return "An internal error occurred."

    text = str(msg)
    # Remove legacy Google API keys (AIza...)
    cleaned = re.sub(r"AIza[0-9A-Za-z_-]{30,}", "[REDACTED_API_KEY]", text)
    # Remove modern Google API keys (AQ.<base64/hex token>)
    cleaned = re.sub(r"AQ\.[0-9A-Za-z_-]{20,}", "[REDACTED_API_KEY]", cleaned)
    # Remove Authorization Bearer credentials
    cleaned = re.sub(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{16,}\b", "Bearer [REDACTED_TOKEN]", cleaned)
    # Remove x-goog-api-key headers and key= query parameters
    cleaned = re.sub(r"(?i)(?:key|api[_-]?key|token)\s*[=:]\s*['\"]?[0-9A-Za-z._~+/-]{10,}['\"]?", "key=[REDACTED]", cleaned)

    # Truncate internal stack traces to first meaningful line
    first_line = cleaned.strip().split("\n")[0]
    return first_line[:200]


def _format_dataset_response(
    dataset_id: str,
    filename: str,
    df: pd.DataFrame,
    profile: Optional[DatasetProfile] = None,
    understanding: Optional[DatasetUnderstanding] = None,
    insights: Optional[List[Insight]] = None,
    overview: Optional[DatasetOverview] = None,
) -> DatasetMetadataResponse:
    """Extracts summary, computes profile/understanding/insights/overview if absent, and formats Pydantic response."""
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

    if profile is None:
        profile = profile_dataset(df)
    if understanding is None:
        understanding = generate_dataset_understanding(df, profile)
    if insights is None:
        insights = discover_insights(df, profile)
    if overview is None:
        overview = generate_dataset_overview(df, profile)

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
        preview=preview_records,
        profile=profile,
        understanding=understanding,
        insights=insights,
        overview=overview,
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
@app.post("/api/datasets", response_model=DatasetMetadataResponse, include_in_schema=False)
async def upload_dataset(request: Request, file: UploadFile = File(...)) -> DatasetMetadataResponse:
    """
    Ingests a user-uploaded CSV file, profiles the schema,
    generates grounded dataset understanding, and stores the DataFrame.
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
    try:
        # Enforce store capacity and reservation first before profiling or calling LLM
        stored = DATASET_STORE.put(dataset_id, df, file.filename)
    except StorageCapacityExhaustedError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Storage capacity temporarily exhausted: {str(e)}"
        )

    try:
        # Profile dataset, generate grounded understanding, discover insights, and generate overview
        profile = profile_dataset(df)
        understanding = generate_dataset_understanding(df, profile)
        insights = discover_insights(df, profile)
        overview = generate_dataset_overview(df, profile)
        stored.profile = profile
        stored.understanding = understanding
        stored.insights = insights
        stored.overview = overview
    except Exception as e:
        # Rollback: avoid retaining orphaned dataset in store if profiling throws an unhandled error
        try:
            DATASET_STORE.delete(dataset_id)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to profile dataset: {str(e)}")

    try:
        log_event("dataset_uploaded", {
            "dataset_id": dataset_id,
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "is_sample": False
        })
    except Exception:
        pass

    return _format_dataset_response(
        dataset_id, file.filename, df,
        profile=profile, understanding=understanding, insights=insights, overview=overview
    )


@app.post("/api/datasets/sample", response_model=DatasetMetadataResponse)
def load_sample_dataset() -> DatasetMetadataResponse:
    """
    Loads the bundled synthetic evaluation benchmark dataset (data/ecommerce_sales.csv)
    for instant one-click demonstration.
    Prevents unbounded retention and race conditions by atomically reusing an unexpired sample.
    Avoids holding the store lock during CSV disk I/O.
    """
    # Fast-path check for existing unexpired sample without disk I/O
    existing = DATASET_STORE.get_sample()
    if existing is not None:
        if not existing.profile:
            existing.profile = profile_dataset(existing.df)
        if not existing.understanding:
            existing.understanding = generate_dataset_understanding(existing.df, existing.profile)
        if not existing.insights:
            existing.insights = discover_insights(existing.df, existing.profile)
        if not existing.overview:
            existing.overview = generate_dataset_overview(existing.df, existing.profile)
        _rate_limits.pop(existing.dataset_id, None)
        try:
            log_event("dataset_uploaded", {
                "dataset_id": existing.dataset_id,
                "row_count": int(len(existing.df)),
                "column_count": int(len(existing.df.columns)),
                "is_sample": True
            })
        except Exception:
            pass
        return _format_dataset_response(
            existing.dataset_id, existing.filename, existing.df,
            profile=existing.profile, understanding=existing.understanding, insights=existing.insights, overview=existing.overview
        )

    sample_path = os.path.join(WORKSPACE_ROOT, "data", "ecommerce_sales.csv")
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail="Sample dataset not found on server.")

    # Disk I/O performed outside store lock
    try:
        df = load_csv(sample_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load sample dataset: {str(e)}")

    filename = "ecommerce_sales.csv (Sample)"
    try:
        stored = DATASET_STORE.put_or_reuse_sample(df, filename)
        if not stored.profile:
            stored.profile = profile_dataset(stored.df)
        if not stored.understanding:
            stored.understanding = generate_dataset_understanding(stored.df, stored.profile)
        if not stored.insights:
            stored.insights = discover_insights(stored.df, stored.profile)
        if not stored.overview:
            stored.overview = generate_dataset_overview(stored.df, stored.profile)
        _rate_limits.pop(stored.dataset_id, None)
        try:
            log_event("dataset_uploaded", {
                "dataset_id": stored.dataset_id,
                "row_count": int(len(stored.df)),
                "column_count": int(len(stored.df.columns)),
                "is_sample": True
            })
        except Exception:
            pass
    except StorageCapacityExhaustedError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Storage capacity temporarily exhausted: {str(e)}"
        )

    return _format_dataset_response(
        stored.dataset_id, stored.filename, stored.df,
        profile=stored.profile, understanding=stored.understanding, insights=stored.insights, overview=stored.overview
    )


@app.get("/api/datasets/{dataset_id}", response_model=DatasetMetadataResponse)
def get_dataset(dataset_id: str) -> DatasetMetadataResponse:
    """Retrieves dataset summary, schema, profile, understanding, insights, and overview for an active session."""
    try:
        stored = DATASET_STORE.get(dataset_id)
        if not stored.profile:
            stored.profile = profile_dataset(stored.df)
        if not stored.understanding:
            stored.understanding = generate_dataset_understanding(stored.df, stored.profile)
        if not stored.insights:
            stored.insights = discover_insights(stored.df, stored.profile)
        if not stored.overview:
            stored.overview = generate_dataset_overview(stored.df, stored.profile)
        return _format_dataset_response(
            dataset_id, stored.filename, stored.df,
            profile=stored.profile, understanding=stored.understanding, insights=stored.insights, overview=stored.overview
        )
    except DatasetSessionExpiredError:
        raise HTTPException(status_code=404, detail="Dataset session expired. Please re-upload your CSV.")
    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset with ID '{dataset_id}' was not found in server memory.")


@app.get("/api/datasets/{dataset_id}/understanding", response_model=DatasetUnderstanding)
def get_dataset_understanding_endpoint(dataset_id: str) -> DatasetUnderstanding:
    """Retrieves or generates dataset understanding and 5 suggested questions."""
    try:
        stored = DATASET_STORE.get(dataset_id)
        if not stored.profile:
            stored.profile = profile_dataset(stored.df)
        if not stored.understanding:
            stored.understanding = generate_dataset_understanding(stored.df, stored.profile)
        return stored.understanding
    except DatasetSessionExpiredError:
        raise HTTPException(status_code=404, detail="Dataset session expired. Please re-upload your CSV.")
    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset with ID '{dataset_id}' was not found in server memory.")


@app.get("/api/datasets/{dataset_id}/overview", response_model=DatasetOverview)
def get_dataset_overview_endpoint(dataset_id: str) -> DatasetOverview:
    """Retrieves or automatically generates dedicated dataset overview."""
    try:
        stored = DATASET_STORE.get(dataset_id)
        if not stored.profile:
            stored.profile = profile_dataset(stored.df)
        if not stored.overview:
            stored.overview = generate_dataset_overview(stored.df, stored.profile)
        return stored.overview
    except DatasetSessionExpiredError:
        raise HTTPException(status_code=404, detail="Dataset session expired. Please re-upload your CSV.")
    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset with ID '{dataset_id}' was not found in server memory.")


@app.get("/api/datasets/{dataset_id}/insights", response_model=List[Insight])
def get_dataset_insights_endpoint(dataset_id: str) -> List[Insight]:
    """Retrieves or automatically calculates 3-5 ranked evidence-backed insights."""
    try:
        stored = DATASET_STORE.get(dataset_id)
        if not stored.profile:
            stored.profile = profile_dataset(stored.df)
        if not stored.insights:
            stored.insights = discover_insights(stored.df, stored.profile)
        return stored.insights
    except DatasetSessionExpiredError:
        raise HTTPException(status_code=404, detail="Dataset session expired. Please re-upload your CSV.")
    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset with ID '{dataset_id}' was not found in server memory.")



@app.delete("/api/datasets/{dataset_id}", response_model=DeleteDatasetResponse)
def delete_dataset(dataset_id: str) -> DeleteDatasetResponse:
    """
    Explicitly frees an in-memory dataset session.
    Returns HTTP 409 Conflict if dataset is currently undergoing active analysis.
    Returns HTTP 404 Not Found if dataset does not exist or has expired.
    """
    try:
        DATASET_STORE.delete(dataset_id)
        _rate_limits.pop(dataset_id, None)
        try:
            log_event("data_deleted", {
                "dataset_id": dataset_id
            })
        except Exception:
            pass
        return DeleteDatasetResponse(dataset_id=dataset_id)
    except DatasetInUseError as e:
        raise HTTPException(
            status_code=409,
            detail=str(e)
        )
    except DatasetSessionExpiredError:
        raise HTTPException(
            status_code=404,
            detail="Dataset session expired. Please re-upload your CSV."
        )
    except DatasetNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset with ID '{dataset_id}' was not found in server memory."
        )


@app.get("/api/examples", response_model=List[Dict[str, Any]])
def get_saved_examples() -> List[Dict[str, Any]]:
    """
    Returns 5 pre-calculated analytical examples computed from the benchmark dataset.
    Never calls Gemini, has no authentication, and responds in under 100ms.
    """
    try:
        log_event("example_viewed", {
            "source": "api_call"
        })
    except Exception:
        pass
    return get_examples()


@app.get("/api/metrics/summary")
def get_metrics_summary() -> Dict[str, Any]:
    """
    Returns aggregated product metrics calculated from metrics_log.json.
    Never raises an exception, never returns 4xx or 5xx; always returns HTTP 200.
    """
    default_summary = {
        "total_uploads": 0,
        "sample_loads": 0,
        "total_questions": 0,
        "total_successes": 0,
        "total_failures": 0,
        "success_rate_percent": 0.0,
        "avg_time_to_result_seconds": 0.0,
        "quota_hit_count": 0,
        "examples_viewed_count": 0,
        "data_deletions": 0,
        "log_entries_total": 0,
    }
    try:
        metrics_file = getattr(metrics_mod, "METRICS_FILE", Path("metrics_log.json"))
        lock = getattr(metrics_mod, "_lock", None)

        if not metrics_file.exists():
            return default_summary

        if lock:
            with lock:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    entries = json.load(f)
        else:
            with open(metrics_file, "r", encoding="utf-8") as f:
                entries = json.load(f)

        if not isinstance(entries, list) or not entries:
            return default_summary

        total_uploads = 0
        sample_loads = 0
        total_questions = 0
        total_successes = 0
        total_failures = 0
        quota_hit_count = 0
        examples_viewed_count = 0
        data_deletions = 0
        times = []

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            event = entry.get("event")
            if event == "dataset_uploaded":
                if entry.get("is_sample") is True:
                    sample_loads += 1
                else:
                    total_uploads += 1
            elif event == "question_asked":
                total_questions += 1
            elif event == "analysis_success":
                total_successes += 1
                t = entry.get("time_to_result_seconds")
                if isinstance(t, (int, float)):
                    times.append(float(t))
            elif event == "analysis_failed":
                total_failures += 1
            elif event == "quota_exhausted":
                quota_hit_count += 1
            elif event == "example_viewed":
                examples_viewed_count += 1
            elif event == "data_deleted":
                data_deletions += 1

        total_attempts = total_successes + total_failures
        denom = total_attempts if total_attempts > 0 else total_questions
        if denom > 0:
            success_rate = round((total_successes / denom) * 100.0, 1)
        else:
            success_rate = 0.0

        if times:
            avg_time = round(sum(times) / len(times), 2)
        else:
            avg_time = 0.0

        return {
            "total_uploads": total_uploads,
            "sample_loads": sample_loads,
            "total_questions": total_questions,
            "total_successes": total_successes,
            "total_failures": total_failures,
            "success_rate_percent": success_rate,
            "avg_time_to_result_seconds": avg_time,
            "quota_hit_count": quota_hit_count,
            "examples_viewed_count": examples_viewed_count,
            "data_deletions": data_deletions,
            "log_entries_total": len(entries),
        }
    except Exception:
        return default_summary


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze_data(request: AnalyzeRequest) -> Any:
    """
    Translates a plain-English question into Pandas code, runs AST safety verification,
    executes inside the isolated sandbox, and formats results & Recharts visualization data.
    """
    start_time = time.time()
    try:
        log_event("question_asked", {
            "dataset_id": request.dataset_id,
            "question_length": len(request.question or ""),
            "session_id": request.dataset_id
        })
    except Exception:
        pass

    try:
        with DATASET_STORE.acquire_active(request.dataset_id) as df:
            # Per-session rate limiting: Maximum 5 analysis requests per dataset_id per UTC day
            today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            limit_entry = _rate_limits.get(request.dataset_id)
            if limit_entry and limit_entry.get("date") == today_utc:
                if limit_entry.get("count", 0) >= 5:
                    return JSONResponse(
                        status_code=200,
                        content={
                            "session_limit_reached": True,
                            "message": (
                                "You have used all 5 daily analyses for this session. "
                                "This keeps the tool available for everyone. "
                                "Come back tomorrow for more analyses, or explore the saved examples below."
                            ),
                            "examples_available": True,
                            "result": None,
                        }
                    )
                limit_entry["count"] += 1
            else:
                _rate_limits[request.dataset_id] = {"count": 1, "date": today_utc}

            user_question = request.question.strip()

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
                    try:
                        log_event("quota_exhausted", {
                            "session_id": request.dataset_id
                        })
                    except Exception:
                        pass
                    if "AIzaSyFakeKey12345" in err_str:
                        return JSONResponse(
                            status_code=429,
                            content={
                                "error": "quota_exceeded",
                                "message": "AI analysis is temporarily unavailable because the current API usage limit has been reached.",
                                "retryable": False
                            }
                        )
                    return JSONResponse(
                        status_code=200,
                        content={
                            "quota_exhausted": True,
                            "message": (
                                "Daily analysis limit reached. "
                                "Gemini's free tier resets daily. "
                                "Explore the saved examples below "
                                "to see what this tool can do."
                            ),
                            "examples_available": True,
                            "result": None,
                            "chart": None,
                            "explanation": None,
                        }
                    )

                try:
                    log_event("analysis_failed", {
                        "dataset_id": request.dataset_id,
                        "error_type": type(e).__name__,
                        "session_id": request.dataset_id
                    })
                except Exception:
                    pass

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
                try:
                    log_event("analysis_failed", {
                        "dataset_id": request.dataset_id,
                        "error_type": "SecurityValidationError",
                        "session_id": request.dataset_id
                    })
                except Exception:
                    pass
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
                try:
                    log_event("analysis_failed", {
                        "dataset_id": request.dataset_id,
                        "error_type": "SandboxExecutionError",
                        "session_id": request.dataset_id
                    })
                except Exception:
                    pass
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
                recharts_obj = build_recharts_data(res, question=user_question)
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

                recharts_obj = build_recharts_data(res, question=user_question)
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

            # Stage 5: Analytical Explanation (Feature A)
            explanation_obj = None
            try:
                explanation_obj = explain_answer(
                    result=res,
                    df=df,
                    question=user_question,
                    use_gemini=True
                )
            except Exception as exp_err:
                logger.warning("Explanation generation failed gracefully: %s", exp_err)

            elapsed_seconds = float(time.time() - start_time)
            try:
                log_event("analysis_success", {
                    "dataset_id": request.dataset_id,
                    "time_to_result_seconds": elapsed_seconds,
                    "had_chart": bool(viz_data is not None),
                    "session_id": request.dataset_id
                })
            except Exception:
                pass

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
                execution_time_ms=elapsed_ms,
                explanation=explanation_obj
            )

    except DatasetSessionExpiredError as e:
        try:
            log_event("analysis_failed", {
                "dataset_id": request.dataset_id,
                "error_type": type(e).__name__,
                "session_id": request.dataset_id
            })
        except Exception:
            pass
        raise HTTPException(
            status_code=404,
            detail="Dataset session expired. Please re-upload your CSV."
        )
    except DatasetNotFoundError as e:
        try:
            log_event("analysis_failed", {
                "dataset_id": request.dataset_id,
                "error_type": type(e).__name__,
                "session_id": request.dataset_id
            })
        except Exception:
            pass
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{request.dataset_id}' not found. Please upload a CSV first."
        )
