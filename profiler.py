"""
profiler.py
-----------
Purpose:
    Deterministic dataset profiling and schema inspection layer for tabular DataFrames.
    Computes factual descriptive statistics, missing value ratios, duplicate row counts,
    categorical distributions, reliable date ranges, and data quality warnings.

Guarantees:
    - Pure pandas and numpy calculations; zero reliance on LLMs.
    - Non-destructive: caller DataFrame is never mutated.
    - Defensively handles empty datasets, all-null columns, infinities, and mixed types.
    - Bounded memory and computation: top categories capped, safe date coercion.
"""

import math
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


# ============================================================================
# Pydantic Models for Profiler Results
# ============================================================================

class NumericColumnStats(BaseModel):
    count: int
    missing_count: int
    missing_pct: float
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None


class CategoryFrequency(BaseModel):
    value: str
    count: int
    pct: float


class CategoricalColumnStats(BaseModel):
    count: int
    missing_count: int
    missing_pct: float
    unique_count: int
    top_categories: List[CategoryFrequency] = Field(default_factory=list)


class DateColumnStats(BaseModel):
    count: int
    missing_count: int
    missing_pct: float
    min_date: Optional[str] = None
    max_date: Optional[str] = None


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    semantic_type: str  # "numeric" | "categorical" | "datetime" | "boolean" | "text" | "empty"
    missing_count: int
    missing_pct: float
    numeric_stats: Optional[NumericColumnStats] = None
    categorical_stats: Optional[CategoricalColumnStats] = None
    date_stats: Optional[DateColumnStats] = None


class DataQualityWarning(BaseModel):
    code: str
    column: Optional[str] = None
    message: str
    severity: str  # "info" | "warning" | "critical"


class DatasetProfile(BaseModel):
    num_rows: int
    num_cols: int
    total_missing: int
    missing_cell_pct: float
    duplicate_rows: int
    duplicate_row_pct: float
    columns: List[str]
    column_profiles: Dict[str, ColumnProfile]
    warnings: List[DataQualityWarning] = Field(default_factory=list)


# ============================================================================
# Helper Functions for Reliable Type Inference and Safe Stats
# ============================================================================

def _clean_scalar_float(val: Any) -> Optional[float]:
    """Converts numpy/pandas floats to clean, JSON-serializable float or None if NaN/inf."""
    if val is None or pd.isna(val):
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, 4)
    except (ValueError, TypeError):
        return None


def _is_reliable_date_column(series: pd.Series, col_name: str) -> bool:
    """
    Determines whether a series reliably represents date or datetime values.
    Requires strong evidence to prevent misclassifying arbitrary text or numeric IDs.
    """
    # 1. Native datetime dtype
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    # Do not treat pure numeric types as dates
    if pd.api.types.is_numeric_dtype(series):
        return False

    # 2. Check for string / object series
    valid_samples = series.dropna().astype(str).str.strip()
    if len(valid_samples) == 0:
        return False

    # Check column name heuristics
    name_lower = col_name.lower().replace("_", " ").replace("-", " ")
    date_name_tokens = {"date", "time", "timestamp", "datetime", "created at", "updated at", "dob", "joined"}
    name_has_date_hint = any(token in name_lower for token in date_name_tokens)

    # Sample up to 20 non-empty values
    sample_size = min(len(valid_samples), 20)
    samples = valid_samples.iloc[:sample_size]

    # Quick regex check for common date/datetime formats (e.g. 2023-01-01, 01/15/2023, 2023-01-01 12:00:00)
    import re
    date_regex = re.compile(
        r"^(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?$"
    )
    matches = sum(bool(date_regex.match(s)) for s in samples)
    if matches / sample_size >= 0.7:
        try:
            parsed = pd.to_datetime(samples, errors="coerce", format="mixed")
            if parsed.notna().sum() / sample_size >= 0.7:
                return True
        except Exception:
            pass

    # If column name specifically hints at date, try stricter pd.to_datetime validation
    if name_has_date_hint:
        try:
            parsed = pd.to_datetime(samples, errors="coerce", format="mixed")
            if parsed.notna().sum() / sample_size >= 0.8:
                return True
        except Exception:
            pass

    return False


def _profile_numeric_column(series: pd.Series, num_rows: int) -> NumericColumnStats:
    """Profiles a numeric column, safely filtering infinities and NaNs."""
    # Replace infinities with NaN for safe calculations
    clean_series = series.replace([np.inf, -np.inf], np.nan).dropna()
    valid_count = int(clean_series.count())
    missing_count = int(num_rows - valid_count)
    missing_pct = round((missing_count / num_rows) * 100, 2) if num_rows > 0 else 0.0

    if valid_count == 0:
        return NumericColumnStats(
            count=0,
            missing_count=missing_count,
            missing_pct=missing_pct,
        )

    try:
        min_val = _clean_scalar_float(clean_series.min())
        max_val = _clean_scalar_float(clean_series.max())
        mean_val = _clean_scalar_float(clean_series.mean())
        median_val = _clean_scalar_float(clean_series.median())
        std_val = _clean_scalar_float(clean_series.std()) if valid_count > 1 else 0.0
        q25_val = _clean_scalar_float(clean_series.quantile(0.25))
        q75_val = _clean_scalar_float(clean_series.quantile(0.75))

        return NumericColumnStats(
            count=valid_count,
            missing_count=missing_count,
            missing_pct=missing_pct,
            min=min_val,
            max=max_val,
            mean=mean_val,
            median=median_val,
            std=std_val,
            q25=q25_val,
            q75=q75_val,
        )
    except Exception:
        return NumericColumnStats(
            count=valid_count,
            missing_count=missing_count,
            missing_pct=missing_pct,
        )


def _profile_categorical_column(
    series: pd.Series, num_rows: int, max_categories: int = 5
) -> CategoricalColumnStats:
    """Profiles a categorical/text column, bounding top frequency classes."""
    valid_series = series.dropna()
    valid_count = int(valid_series.count())
    missing_count = int(num_rows - valid_count)
    missing_pct = round((missing_count / num_rows) * 100, 2) if num_rows > 0 else 0.0

    if valid_count == 0:
        return CategoricalColumnStats(
            count=0,
            missing_count=missing_count,
            missing_pct=missing_pct,
            unique_count=0,
            top_categories=[],
        )

    try:
        unique_count = int(valid_series.nunique())
        # Compute top value counts (capped to max_categories)
        val_counts = valid_series.astype(str).value_counts().head(max_categories)
        top_cats: List[CategoryFrequency] = []
        for val, count in val_counts.items():
            top_cats.append(
                CategoryFrequency(
                    value=str(val)[:50],  # Bound value string length
                    count=int(count),
                    pct=round((int(count) / valid_count) * 100, 2) if valid_count > 0 else 0.0,
                )
            )

        return CategoricalColumnStats(
            count=valid_count,
            missing_count=missing_count,
            missing_pct=missing_pct,
            unique_count=unique_count,
            top_categories=top_cats,
        )
    except Exception:
        return CategoricalColumnStats(
            count=valid_count,
            missing_count=missing_count,
            missing_pct=missing_pct,
            unique_count=0,
            top_categories=[],
        )


def _profile_date_column(series: pd.Series, num_rows: int) -> DateColumnStats:
    """Profiles a reliable date column with min and max date boundaries."""
    try:
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        valid_dates = parsed.dropna()
        valid_count = int(valid_dates.count())
        missing_count = int(num_rows - valid_count)
        missing_pct = round((missing_count / num_rows) * 100, 2) if num_rows > 0 else 0.0

        min_date_str = str(valid_dates.min().isoformat()) if valid_count > 0 else None
        max_date_str = str(valid_dates.max().isoformat()) if valid_count > 0 else None

        return DateColumnStats(
            count=valid_count,
            missing_count=missing_count,
            missing_pct=missing_pct,
            min_date=min_date_str,
            max_date=max_date_str,
        )
    except Exception:
        valid_count = int(series.dropna().count())
        return DateColumnStats(
            count=valid_count,
            missing_count=int(num_rows - valid_count),
            missing_pct=round(((num_rows - valid_count) / num_rows) * 100, 2) if num_rows > 0 else 0.0,
        )


# ============================================================================
# Main Profiling Function
# ============================================================================

def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    """
    Computes a factual, deterministic dataset profile from a pandas DataFrame.

    Semantic Definitions:
    - `duplicate_rows`: The total count of completely duplicate rows where every column value matches
      an earlier row (`df.duplicated().sum()`). Distinguishable from column-level duplicate values.
    - `missing_cell_pct`: Total missing cells divided by (rows * cols).
    - `semantic_type`: High-level operational type ('numeric', 'categorical', 'datetime', 'boolean', 'text', 'empty').

    Args:
        df (pd.DataFrame): Target DataFrame.

    Returns:
        DatasetProfile: Strongly-typed, JSON-serializable profile model.
    """
    num_rows, num_cols = df.shape
    columns = [str(c) for c in df.columns]

    # Handle empty dataset edge case
    if num_rows == 0 or num_cols == 0:
        warnings = [
            DataQualityWarning(
                code="EMPTY_DATASET",
                message="The dataset contains zero rows or zero columns.",
                severity="critical",
            )
        ]
        return DatasetProfile(
            num_rows=num_rows,
            num_cols=num_cols,
            total_missing=0,
            missing_cell_pct=0.0,
            duplicate_rows=0,
            duplicate_row_pct=0.0,
            columns=columns,
            column_profiles={},
            warnings=warnings,
        )

    # Overall dataset metrics
    total_cells = num_rows * num_cols
    total_missing = int(df.isnull().sum().sum())
    missing_cell_pct = round((total_missing / total_cells) * 100, 2) if total_cells > 0 else 0.0

    # Duplicate rows semantics: complete row equality
    duplicate_rows = int(df.duplicated().sum())
    duplicate_row_pct = round((duplicate_rows / num_rows) * 100, 2) if num_rows > 0 else 0.0

    column_profiles: Dict[str, ColumnProfile] = {}
    warnings: List[DataQualityWarning] = []

    # Quality check: high overall duplicate percentage
    if duplicate_row_pct >= 5.0 and duplicate_rows > 0:
        warnings.append(
            DataQualityWarning(
                code="HIGH_DUPLICATES",
                message=f"Dataset contains {duplicate_rows} duplicate rows ({duplicate_row_pct}% of total records).",
                severity="warning" if duplicate_row_pct < 20.0 else "critical",
            )
        )

    # Quality check: high overall missing cell percentage
    if missing_cell_pct >= 25.0:
        warnings.append(
            DataQualityWarning(
                code="HIGH_TOTAL_MISSING",
                message=f"{missing_cell_pct}% of all data cells are missing across the dataset.",
                severity="warning" if missing_cell_pct < 50.0 else "critical",
            )
        )

    # Profile each column
    for col in df.columns:
        col_str = str(col)
        series = df[col]
        raw_dtype = str(series.dtype)
        null_count = int(series.isnull().sum())
        null_pct = round((null_count / num_rows) * 100, 2) if num_rows > 0 else 0.0

        # Check 1: All-null column
        if null_count == num_rows:
            warnings.append(
                DataQualityWarning(
                    code="ALL_NULL_COLUMN",
                    column=col_str,
                    message=f"Column '{col_str}' contains 100% missing values.",
                    severity="warning",
                )
            )
            column_profiles[col_str] = ColumnProfile(
                name=col_str,
                dtype=raw_dtype,
                semantic_type="empty",
                missing_count=null_count,
                missing_pct=null_pct,
            )
            continue

        # Check 2: High missing ratio
        if null_pct >= 50.0:
            warnings.append(
                DataQualityWarning(
                    code="HIGH_COLUMN_MISSING",
                    column=col_str,
                    message=f"Column '{col_str}' has {null_pct}% missing values ({null_count}/{num_rows}).",
                    severity="warning",
                )
            )

        # Check 3: Date/time detection
        if _is_reliable_date_column(series, col_str):
            date_stats = _profile_date_column(series, num_rows)
            column_profiles[col_str] = ColumnProfile(
                name=col_str,
                dtype=raw_dtype,
                semantic_type="datetime",
                missing_count=null_count,
                missing_pct=null_pct,
                date_stats=date_stats,
            )
            continue

        # Check 4: Numeric column
        if pd.api.types.is_numeric_dtype(series):
            # Check for boolean formatted as int (0 and 1 only)
            valid_vals = series.dropna()
            if len(valid_vals) > 0 and valid_vals.nunique() <= 2 and set(valid_vals.unique()).issubset({0, 1, 0.0, 1.0}):
                # Boolean / binary flag
                cat_stats = _profile_categorical_column(series, num_rows)
                column_profiles[col_str] = ColumnProfile(
                    name=col_str,
                    dtype=raw_dtype,
                    semantic_type="boolean",
                    missing_count=null_count,
                    missing_pct=null_pct,
                    categorical_stats=cat_stats,
                )
            else:
                num_stats = _profile_numeric_column(series, num_rows)
                column_profiles[col_str] = ColumnProfile(
                    name=col_str,
                    dtype=raw_dtype,
                    semantic_type="numeric",
                    missing_count=null_count,
                    missing_pct=null_pct,
                    numeric_stats=num_stats,
                )

                # Check constant column
                if num_stats.min == num_stats.max and num_stats.min is not None and num_rows > 1:
                    warnings.append(
                        DataQualityWarning(
                            code="CONSTANT_COLUMN",
                            column=col_str,
                            message=f"Column '{col_str}' is constant (all valid values equal {num_stats.min}).",
                            severity="info",
                        )
                    )
            continue

        # Check 5: Explicit boolean dtype
        if pd.api.types.is_bool_dtype(series):
            cat_stats = _profile_categorical_column(series, num_rows)
            column_profiles[col_str] = ColumnProfile(
                name=col_str,
                dtype=raw_dtype,
                semantic_type="boolean",
                missing_count=null_count,
                missing_pct=null_pct,
                categorical_stats=cat_stats,
            )
            continue

        # Check 6: Categorical or Text
        cat_stats = _profile_categorical_column(series, num_rows)
        unique_cnt = cat_stats.unique_count

        # Check for constant column
        if unique_cnt == 1 and num_rows > 1:
            warnings.append(
                DataQualityWarning(
                    code="CONSTANT_COLUMN",
                    column=col_str,
                    message=f"Column '{col_str}' is constant with only 1 unique value.",
                    severity="info",
                )
            )

        # Check for high cardinality text / potential primary key ID
        semantic_type = "categorical"
        if num_rows > 20 and unique_cnt >= 0.8 * num_rows:
            semantic_type = "text"
            warnings.append(
                DataQualityWarning(
                    code="HIGH_CARDINALITY",
                    column=col_str,
                    message=f"Column '{col_str}' has very high cardinality ({unique_cnt} unique values across {num_rows} rows).",
                    severity="info",
                )
            )

        column_profiles[col_str] = ColumnProfile(
            name=col_str,
            dtype=raw_dtype,
            semantic_type=semantic_type,
            missing_count=null_count,
            missing_pct=null_pct,
            categorical_stats=cat_stats,
        )

    return DatasetProfile(
        num_rows=num_rows,
        num_cols=num_cols,
        total_missing=total_missing,
        missing_cell_pct=missing_cell_pct,
        duplicate_rows=duplicate_rows,
        duplicate_row_pct=duplicate_row_pct,
        columns=columns,
        column_profiles=column_profiles,
        warnings=warnings,
    )
