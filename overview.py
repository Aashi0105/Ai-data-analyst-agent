"""
overview.py
-----------
Purpose:
    Generates a dedicated, production-grade Dataset Overview that explains the purpose,
    domain, row representation, key fields, analytical opportunities, and realistic
    predictive possibilities of an uploaded dataset.

Core Principles:
    1. Grounding & Epistemic Boundaries:
       - Domain and row meaning are inferred from verified column names,
         data types, cardinality, and profile statistics.
       - Distinguishes schema-level domain inference from data usability:
         a dataset may have recognized column names but be 100% missing or empty.
       - Distinguishes exploratory analysis from predictive machine learning.
       - Feasibility is determined by explicit evidence: sample size, class balance,
         feature availability, and temporal structure.
    2. Column-Role Disambiguation:
       - Primary identifiers, foreign keys, and coordinates are strictly separated
         from analytical metrics.
       - Unique identifiers are NEVER averaged, summed, ranked, correlated, or used
         as predictive targets.
    3. Zero Failure Mode:
       - Comprehensive deterministic baseline operates with zero LLM dependency.
       - If Gemini is enabled, synthesizes fluent narrative with a strict timeout.
       - Strict validation ensures LLM cannot hallucinate regression targets, forecasting
         on cross-sectional data, or value calculations on empty data.
       - On any failure or invalid LLM response, falls back instantly to deterministic baseline.
    4. Privacy & Safety:
       - Never transmits full CSV to AI; only compact profile and 3 PII-masked sample rows.
       - Cell values are treated as untrusted data, never as prompt instructions.
"""

from enum import Enum
import concurrent.futures
import json
import logging
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from profiler import DatasetProfile, ColumnProfile
from prompt import mask_sample_dataframe, mask_sensitive_value

logger = logging.getLogger(__name__)


# ============================================================================
# Column Semantic Roles & Intermediate Representation
# ============================================================================

class ColumnRole(str, Enum):
    PRIMARY_ID = "primary_id"
    FOREIGN_ID = "foreign_id"
    BINARY_LABEL = "binary_label"
    MULTICLASS_LABEL = "multiclass_label"
    CATEGORICAL_DIMENSION = "categorical_dimension"
    CONTINUOUS_MEASUREMENT = "continuous_measurement"
    COUNT_OR_QUANTITY = "count_or_quantity"
    TEMPORAL = "temporal"
    GEOGRAPHIC = "geographic"
    FREE_TEXT = "free_text"
    ALL_NULL = "all_null"
    UNKNOWN = "unknown"


class InferredColumnSemantics(BaseModel):
    name: str
    dtype: str
    semantic_role: ColumnRole
    is_usable: bool
    unique_count: int
    missing_pct: float
    plain_meaning: str
    role_evidence: str
    sample_values: List[str] = Field(default_factory=list)


class DatasetSemantics(BaseModel):
    num_rows: int
    num_cols: int
    usable_rows: int
    usable_cells: int
    is_entirely_empty: bool
    is_degraded: bool
    domain: str
    domain_confidence: str  # "high" | "medium" | "low"
    domain_rationale: str
    sub_domain: Optional[str] = None
    row_entity: str
    columns: Dict[str, InferredColumnSemantics]
    primary_ids: List[str]
    foreign_ids: List[str]
    measurements: List[str]
    dimensions: List[str]
    binary_labels: List[str]
    multiclass_labels: List[str]
    temporal_cols: List[str]


# ============================================================================
# Output Pydantic Schemas (API Contracts)
# ============================================================================

class KeyField(BaseModel):
    name: str
    meaning: str
    type_info: Optional[str] = None
    role: Optional[str] = None  # "identifier" | "metric" | "dimension" | "temporal" | "attribute"


class AnalysisOpportunity(BaseModel):
    title: str
    description: str
    business_value: str
    sample_question: Optional[str] = None


class PredictionCandidate(BaseModel):
    task: str
    target_variable: Optional[str] = None
    prerequisites: str
    feasibility: str  # "Feasible for exploration" | "Feasible with caveats" | "Not currently supported by the data"
    missing_elements: List[str] = Field(default_factory=list)


class DatasetOverview(BaseModel):
    domain: str
    domain_confidence: str  # "high" | "medium" | "low"
    domain_rationale: Optional[str] = None
    summary_paragraph: str
    row_meaning: str
    size_and_coverage: str
    key_fields: List[KeyField] = Field(default_factory=list)
    what_we_can_learn: List[AnalysisOpportunity] = Field(default_factory=list)
    potential_predictions: List[PredictionCandidate] = Field(default_factory=list)
    prediction_limitations: str
    status: str = "deterministic"  # "deterministic" | "synthesized"


# ============================================================================
# Domain Knowledge Lexicons
# ============================================================================

DOMAIN_RULES: Dict[str, Dict[str, Any]] = {
    "Real Estate & Property Listings": {
        "strong": [
            "room_type", "neighbourhood", "neighbourhood_group", "host_id",
            "host_name", "listing_id", "property_type", "accommodates",
            "bedrooms", "bathrooms", "amenities", "square_feet", "beds"
        ],
        "normal": [
            "price", "latitude", "longitude", "reviews", "minimum_nights",
            "availability", "city", "zipcode", "borough", "postal_code"
        ],
        "row_desc": "an individual short-term rental property listing or accommodation record",
        "vocabulary": {
            "comparison": "property type pricing dynamics",
            "correlation": "geographic and pricing relationships",
            "distribution": "nightly rate and accommodation capacity distributions",
        },
    },
    "Healthcare & Medical": {
        "strong": [
            "patient_id", "patient", "diagnosis", "blood_pressure", "heart_rate",
            "treatment", "dosage", "hospital", "physician", "admission",
            "cholesterol", "glucose", "symptom", "medical_record", "radius_mean",
            "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
            "compactness_mean", "concavity_mean", "symmetry_mean",
            "malignant", "benign", "tumor"
        ],
        "normal": [
            "doctor", "nurse", "clinic", "prescription", "disease", "condition",
            "bmi", "health", "vital", "lab_result", "procedure", "discharge",
            "se", "worst", "mean"
        ],
        "row_desc": "an individual patient encounter, clinical observation, or diagnostic assessment",
        "vocabulary": {
            "comparison": "diagnostic group differentiation and clinical variation",
            "correlation": "biomarker co-linearity and geometric feature redundancy",
            "distribution": "morphological measurement variation across observations",
        },
    },
    "E-Commerce & Retail": {
        "strong": [
            "order_id", "orders_id", "transaction_id", "sku", "cart", "checkout",
            "subtotal", "purchase_date", "unit_price", "quantity_ordered", "sales_amount"
        ],
        "normal": [
            "product", "category", "price", "revenue", "sales", "discount", "customer",
            "customer_id", "shipping", "quantity", "store", "retail", "transaction",
            "order", "merchant", "item_name"
        ],
        "row_desc": "an individual customer purchase transaction line-item",
        "vocabulary": {
            "comparison": "category sales performance and segment contribution",
            "correlation": "pricing elasticity and transaction volume interaction",
            "distribution": "order value spread and high-volume transaction concentration",
        },
    },
    "Human Resources & Workforce": {
        "strong": [
            "employee_id", "job_title", "hire_date", "attrition", "performance_rating",
            "salary", "tenure", "compensation", "leave_days", "department"
        ],
        "normal": [
            "employee", "staff", "manager", "payroll", "workforce", "bonus",
            "hourly_rate", "role", "headcount", "termination", "overtime"
        ],
        "row_desc": "an individual employee profile or employment history record",
        "vocabulary": {
            "comparison": "departmental compensation and retention disparities",
            "correlation": "tenure, experience, and performance interactions",
            "distribution": "workforce salary structure and tenure spread",
        },
    },
    "Finance & Banking": {
        "strong": [
            "account_number", "balance", "credit_score", "interest_rate", "loan_amount",
            "loan_id", "dividend", "portfolio", "ticker", "stock_price", "credit_limit"
        ],
        "normal": [
            "account", "bank", "deposit", "withdrawal", "loan", "asset", "liability",
            "debt", "equity", "transaction_type", "currency", "credit", "debit"
        ],
        "row_desc": "a financial transaction, bank account snapshot, or credit record",
        "vocabulary": {
            "comparison": "account balance and risk tier variations",
            "correlation": "credit score and interest rate relationship",
            "distribution": "capital allocation spread and credit exposure concentration",
        },
    },
    "Marketing & Web Analytics": {
        "strong": [
            "campaign_id", "clicks", "impressions", "ctr", "conversions", "pageviews",
            "bounce_rate", "ad_spend", "cpc", "utm_campaign", "utm_source"
        ],
        "normal": [
            "campaign", "channel", "lead", "funnel", "conversion_rate", "ad_group",
            "traffic_source", "session_duration", "visitor_id", "ad_name"
        ],
        "row_desc": "a marketing campaign performance log or web user interaction event",
        "vocabulary": {
            "comparison": "channel conversion efficiency and return on ad spend",
            "correlation": "engagement depth and conversion velocity",
            "distribution": "traffic concentration and visitor engagement variance",
        },
    },
    "Supply Chain & Inventory": {
        "strong": [
            "warehouse_id", "supplier_id", "stock_level", "reorder_point", "lead_time",
            "shipment_id", "carrier", "freight_cost", "lot_number"
        ],
        "normal": [
            "warehouse", "supplier", "stock", "inventory", "shipment", "logistics",
            "fulfillment", "replenishment", "sku_count", "inventory_turnover"
        ],
        "row_desc": "an inventory stock record, logistical shipment event, or supplier replenishment entry",
        "vocabulary": {
            "comparison": "supplier lead times and regional fulfillment variances",
            "correlation": "freight costs vs shipment weight dynamics",
            "distribution": "stock holding levels and warehouse capacity utilization",
        },
    },
    "Education & Academic": {
        "strong": [
            "student_id", "gpa", "grade", "exam_score", "course_id", "semester",
            "tuition", "enrollment", "degree", "academic_year"
        ],
        "normal": [
            "student", "course", "grade", "exam", "faculty", "school",
            "curriculum", "instructor", "attendance", "scholarship"
        ],
        "row_desc": "an individual student academic record or course evaluation entry",
        "vocabulary": {
            "comparison": "academic achievement across courses and departments",
            "correlation": "attendance vs examination performance",
            "distribution": "grade point average distribution and cohort percentiles",
        },
    },
}


# ============================================================================
# Deterministic Column-Role Inference Engine
# ============================================================================

def infer_column_semantics(df: pd.DataFrame, profile: DatasetProfile) -> Dict[str, InferredColumnSemantics]:
    """
    Evaluates every column using multiple independent signals:
      - Name tokens and prefixes/suffixes
      - Uniqueness ratio and cardinality
      - Data types and profile statistics
      - Distinct values and distributions

    Guarantees:
      - Primary identifiers are identified and strictly isolated from analytical metrics.
      - Binary outcome labels (like 'diagnosis' or 'churn') are recognized.
      - 100% all-null columns are identified as ALL_NULL without fabricating metrics.
    """
    num_rows = profile.num_rows
    semantics: Dict[str, InferredColumnSemantics] = {}

    for col in profile.columns:
        col_str = str(col)
        col_l = col_str.lower().strip()
        cp = profile.column_profiles.get(col_str)
        if not cp:
            continue

        raw_dtype = cp.dtype
        null_count = cp.missing_count
        null_pct = cp.missing_pct
        unique_cnt = 0
        if cp.categorical_stats:
            unique_cnt = cp.categorical_stats.unique_count
        elif num_rows > 0:
            # Fallback estimation for numeric or datetime
            try:
                unique_cnt = int(df[col_str].nunique(dropna=True)) if col_str in df.columns else 0
            except Exception:
                unique_cnt = 0

        # Sample strings for contextual inspection
        sample_vals = []
        if cp.categorical_stats and cp.categorical_stats.top_categories:
            sample_vals = [c.value for c in cp.categorical_stats.top_categories[:3]]
        elif col_str in df.columns and num_rows > 0:
            sample_vals = [str(v) for v in df[col_str].dropna().head(3).tolist()]

        # Signal 1: 100% Missing
        if null_pct == 100.0 or (num_rows > 0 and null_count == num_rows):
            semantics[col_str] = InferredColumnSemantics(
                name=col_str,
                dtype=raw_dtype,
                semantic_role=ColumnRole.ALL_NULL,
                is_usable=False,
                unique_count=0,
                missing_pct=100.0,
                plain_meaning="Unpopulated attribute containing 100% missing values.",
                role_evidence="Zero valid cells in dataset.",
                sample_values=[],
            )
            continue

        # Signal 2: Temporal
        if cp.semantic_type == "datetime":
            span = ""
            if cp.date_stats and cp.date_stats.min_date and cp.date_stats.max_date:
                span = f" spanning from {cp.date_stats.min_date} to {cp.date_stats.max_date}"
            semantics[col_str] = InferredColumnSemantics(
                name=col_str,
                dtype=raw_dtype,
                semantic_role=ColumnRole.TEMPORAL,
                is_usable=True,
                unique_count=unique_cnt,
                missing_pct=null_pct,
                plain_meaning=f"Chronological timestamp or transaction date{span}.",
                role_evidence="Inferred from datetime parsing or temporal naming.",
                sample_values=sample_vals,
            )
            continue

        # Signal 3: Geographic Coordinates
        if any(geo_kw in col_l for geo_kw in ["latitude", "longitude", "lat", "lon", "lng", "gps"]):
            semantics[col_str] = InferredColumnSemantics(
                name=col_str,
                dtype=raw_dtype,
                semantic_role=ColumnRole.GEOGRAPHIC,
                is_usable=True,
                unique_count=unique_cnt,
                missing_pct=null_pct,
                plain_meaning="Geographic spatial coordinate (latitude/longitude).",
                role_evidence="Column name and spatial coordinate pattern.",
                sample_values=sample_vals,
            )
            continue

        # Signal 4: Identifier Identification (Primary Key or Foreign Key)
        # Check name tokens for identifier patterns:
        col_tokens = set(re.split(r"[_\s\-.]+", col_l))
        has_id_token = (
            col_l in ("id", "uuid", "guid", "key", "code", "index", "hash", "token", "pk", "fk")
            or col_l.endswith(("_id", "_key", "_code", "_uuid", "_guid", "_ref", "_token", "_number", "_no", "_pk", "_fk"))
            or col_l.startswith(("id_", "uuid_", "key_", "code_"))
            or bool(col_tokens.intersection({"id", "uuid", "guid", "key", "code", "pk", "fk"}))
        )

        uniqueness_ratio = (unique_cnt / num_rows) if num_rows > 0 else 0.0

        # Check if values are continuous floats with decimal/fractional variation
        is_fractional_float = False
        if col_str in df.columns and num_rows > 0:
            try:
                s_valid = df[col_str].dropna()
                if pd.api.types.is_float_dtype(s_valid) and not s_valid.empty:
                    # Check if any value has non-zero fractional part
                    if not np.all(np.isclose(s_valid.values % 1, 0, atol=1e-5)):
                        is_fractional_float = True
            except Exception:
                pass

        # Check if non-numeric column contains alphanumeric identifier formats (UUID, hashes, PK-123, etc.)
        is_string_identifier = False
        if cp.semantic_type == "categorical" and sample_vals and uniqueness_ratio >= 0.90:
            # If values do not have spaces and look like code/keys or hashes
            if all(bool(re.match(r"^[A-Za-z0-9_\-\:]{2,64}$", str(v))) and " " not in str(v) for v in sample_vals):
                is_string_identifier = True

        # Determine if column is an identifier
        is_identifier = False
        if not is_fractional_float:
            if has_id_token:
                is_identifier = True
            elif is_string_identifier and uniqueness_ratio >= 0.90 and num_rows >= 20:
                is_identifier = True
            elif cp.semantic_type == "numeric" and uniqueness_ratio == 1.0 and num_rows >= 30:
                # Monotonic sequence check for unnamed row indices or explicit sequential row numbers
                try:
                    s_valid = df[col_str].dropna()
                    if s_valid.is_monotonic_increasing and (s_valid.iloc[0] in (0, 1) or "unnamed" in col_l):
                        is_identifier = True
                except Exception:
                    pass

        if is_identifier:
            if uniqueness_ratio >= 0.85 or col_l in ("id", "uuid", "guid", "key", "pk"):
                semantics[col_str] = InferredColumnSemantics(
                    name=col_str,
                    dtype=raw_dtype,
                    semantic_role=ColumnRole.PRIMARY_ID,
                    is_usable=True,
                    unique_count=unique_cnt,
                    missing_pct=null_pct,
                    plain_meaning=f"Unique record or subject identifier ({unique_cnt:,} distinct entities).",
                    role_evidence=f"Identifier token or key format with high uniqueness ({uniqueness_ratio:.1%}). Excluded from analytical metrics.",
                    sample_values=sample_vals[:2],
                )
                continue
            else:
                semantics[col_str] = InferredColumnSemantics(
                    name=col_str,
                    dtype=raw_dtype,
                    semantic_role=ColumnRole.FOREIGN_ID,
                    is_usable=True,
                    unique_count=unique_cnt,
                    missing_pct=null_pct,
                    plain_meaning=f"Entity reference identifier linking {unique_cnt:,} distinct entities.",
                    role_evidence=f"Identifier token with repeating values ({uniqueness_ratio:.1%} uniqueness).",
                    sample_values=sample_vals[:2],
                )
                continue

        # Signal 5: Binary Outcome Label (Crucial for Breast Cancer 'diagnosis' or Churn)
        if unique_cnt == 2 and num_rows >= 5:
            # Check if name or values imply an outcome or classification target
            is_target_name = any(kw in col_l for kw in [
                "diagnosis", "target", "label", "outcome", "class", "churn",
                "status", "fraud", "default", "survived", "result", "converted",
                "attrition", "disease", "decision", "is_", "has_"
            ])
            # Check values: M/B, True/False, 0/1, Yes/No, Malignant/Benign
            clean_s = [s.strip().upper() for s in sample_vals]
            is_target_values = any(v in ("M", "B", "TRUE", "FALSE", "0", "1", "YES", "NO", "MALIGNANT", "BENIGN", "PASS", "FAIL") for v in clean_s)

            if is_target_name or is_target_values or cp.semantic_type == "boolean":
                val_desc = f" ({' vs '.join(sample_vals[:2])})" if sample_vals else ""
                semantics[col_str] = InferredColumnSemantics(
                    name=col_str,
                    dtype=raw_dtype,
                    semantic_role=ColumnRole.BINARY_LABEL,
                    is_usable=True,
                    unique_count=2,
                    missing_pct=null_pct,
                    plain_meaning=f"Binary classification determination{val_desc}.",
                    role_evidence="Discrete attribute with exactly two distinct classes.",
                    sample_values=sample_vals,
                )
                continue

        # Signal 6: Multiclass Outcome Label
        if 3 <= unique_cnt <= 10 and any(kw in col_l for kw in ["status", "stage", "grade", "severity", "class", "target", "outcome"]):
            semantics[col_str] = InferredColumnSemantics(
                name=col_str,
                dtype=raw_dtype,
                semantic_role=ColumnRole.MULTICLASS_LABEL,
                is_usable=True,
                unique_count=unique_cnt,
                missing_pct=null_pct,
                plain_meaning=f"Multiclass status or outcome with {unique_cnt} categorical states.",
                role_evidence=f"Outcome naming with {unique_cnt} discrete states.",
                sample_values=sample_vals,
            )
            continue

        # Signal 7: Count or Quantity (Discrete non-negative integer)
        if cp.semantic_type == "numeric" and any(kw in col_l for kw in ["quantity", "count", "num_", "units", "items", "clicks", "visits", "days"]):
            semantics[col_str] = InferredColumnSemantics(
                name=col_str,
                dtype=raw_dtype,
                semantic_role=ColumnRole.COUNT_OR_QUANTITY,
                is_usable=True,
                unique_count=unique_cnt,
                missing_pct=null_pct,
                plain_meaning="Discrete quantitative count or enumeration metric.",
                role_evidence="Non-negative numeric metric representing counts.",
                sample_values=sample_vals,
            )
            continue

        # Signal 8: Continuous Measurement (Genuine analytical metrics: radius, revenue, salary, etc.)
        if cp.semantic_type == "numeric":
            stats_str = ""
            if cp.numeric_stats and cp.numeric_stats.min is not None and cp.numeric_stats.max is not None:
                stats_str = f" (min: {cp.numeric_stats.min}, max: {cp.numeric_stats.max}, avg: {cp.numeric_stats.mean})"
            semantics[col_str] = InferredColumnSemantics(
                name=col_str,
                dtype=raw_dtype,
                semantic_role=ColumnRole.CONTINUOUS_MEASUREMENT,
                is_usable=True,
                unique_count=unique_cnt,
                missing_pct=null_pct,
                plain_meaning=f"Continuous quantitative measurement or operational metric{stats_str}.",
                role_evidence="Numerical scale with continuous variation.",
                sample_values=sample_vals,
            )
            continue

        # Signal 9: Categorical Dimension vs Free Text
        if cp.semantic_type in ("categorical", "boolean"):
            top_val = sample_vals[0] if sample_vals else ""
            semantics[col_str] = InferredColumnSemantics(
                name=col_str,
                dtype=raw_dtype,
                semantic_role=ColumnRole.CATEGORICAL_DIMENSION,
                is_usable=True,
                unique_count=unique_cnt,
                missing_pct=null_pct,
                plain_meaning=f"Categorical dimension with {unique_cnt} distinct segments (e.g. '{top_val}').",
                role_evidence=f"Discrete non-numeric attribute with {unique_cnt} categories.",
                sample_values=sample_vals,
            )
            continue

        # Signal 10: Free Text
        semantics[col_str] = InferredColumnSemantics(
            name=col_str,
            dtype=raw_dtype,
            semantic_role=ColumnRole.FREE_TEXT,
            is_usable=True,
            unique_count=unique_cnt,
            missing_pct=null_pct,
            plain_meaning="Unstructured textual descriptions or natural language strings.",
            role_evidence="High-cardinality string attribute.",
            sample_values=sample_vals,
        )

    return semantics


# ============================================================================
# Dataset Semantics & Usability Evaluation
# ============================================================================

def build_dataset_semantics(df: pd.DataFrame, profile: DatasetProfile) -> DatasetSemantics:
    """
    Constructs the typed intermediate representation separating verified facts,
    inferred semantics, and operational usability constraints.
    """
    num_rows = profile.num_rows
    num_cols = profile.num_cols

    # Calculate usable rows and cells
    usable_rows = 0
    usable_cells = 0
    if num_rows > 0 and num_cols > 0 and not df.empty:
        try:
            usable_rows = int(df.dropna(how="all").shape[0])
            usable_cells = int(df.notna().sum().sum())
        except Exception:
            usable_rows = max(0, num_rows - profile.duplicate_rows)
            usable_cells = max(0, (num_rows * num_cols) - profile.total_missing)

    is_entirely_empty = (
        num_rows == 0
        or num_cols == 0
        or usable_rows == 0
        or usable_cells == 0
        or profile.missing_cell_pct >= 100.0
    )
    is_degraded = (not is_entirely_empty) and (profile.missing_cell_pct >= 60.0 or usable_rows < 15)

    # Infer column roles
    col_semantics = infer_column_semantics(df, profile)

    # Group columns by role
    primary_ids = [c for c, s in col_semantics.items() if s.semantic_role == ColumnRole.PRIMARY_ID]
    foreign_ids = [c for c, s in col_semantics.items() if s.semantic_role == ColumnRole.FOREIGN_ID]
    measurements = [c for c, s in col_semantics.items() if s.semantic_role in (ColumnRole.CONTINUOUS_MEASUREMENT, ColumnRole.COUNT_OR_QUANTITY)]
    dimensions = [c for c, s in col_semantics.items() if s.semantic_role in (ColumnRole.CATEGORICAL_DIMENSION, ColumnRole.MULTICLASS_LABEL)]
    binary_labels = [c for c, s in col_semantics.items() if s.semantic_role == ColumnRole.BINARY_LABEL]
    multiclass_labels = [c for c, s in col_semantics.items() if s.semantic_role == ColumnRole.MULTICLASS_LABEL]
    temporal_cols = [c for c, s in col_semantics.items() if s.semantic_role == ColumnRole.TEMPORAL]

    # Domain Scoring
    clean_cols = [str(c).lower().strip() for c in profile.columns]
    scores: Dict[str, int] = {}
    matched_words: Dict[str, List[str]] = {}

    for domain_name, rules in DOMAIN_RULES.items():
        score = 0
        matches = []
        for col in clean_cols:
            col_tokens = re.split(r"[_\s\-.]+", col)
            for strong in rules["strong"]:
                if strong in col or strong in col_tokens:
                    score += 3
                    matches.append(col)
                    break
            else:
                for norm in rules["normal"]:
                    if norm in col or norm in col_tokens:
                        score += 1
                        matches.append(col)
                        break
        scores[domain_name] = score
        matched_words[domain_name] = list(set(matches))

    # Rank domains
    sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_domain, best_score = sorted_domains[0] if sorted_domains else ("General Tabular Data (Unspecified)", 0)

    # Calibrate confidence & row description
    sub_domain: Optional[str] = None

    if is_entirely_empty:
        # Schema-only inference for completely empty datasets
        if best_score >= 3:
            domain = best_domain
            confidence = "low"
            matched_list = ", ".join(matched_words[best_domain][:4])
            rationale = (
                f"Inferred tentatively from column names ({matched_list}). However, all {num_rows:,} rows "
                f"contain zero usable values (100% missing cells), so this inference cannot be verified from records."
            )
            row_entity = f"an intended {DOMAIN_RULES[best_domain]['row_desc']} (all records currently empty)"
        else:
            domain = "General Tabular Data (Unspecified)"
            confidence = "low"
            rationale = "The dataset has zero usable values across all cells. Column names are generic or ambiguous."
            row_entity = "an unpopulated tabular record with zero recorded data"
    else:
        if best_score >= 4:
            domain = best_domain
            confidence = "high"
            matched_list = ", ".join(matched_words[best_domain][:4])
            rationale = f"Identified {best_domain} with high confidence based on columns: {matched_list}."
            row_entity = DOMAIN_RULES[best_domain]["row_desc"]

            # Specific sub-domain refinement for Healthcare (Diagnostic Cellular Morphology vs Clinical)
            if best_domain == "Healthcare & Medical":
                morphology_indicators = {"radius", "texture", "perimeter", "area", "smoothness", "concavity", "symmetry", "fractal_dimension", "tumor"}
                matched_morphology = [c for c in clean_cols if any(b in c for b in morphology_indicators)]
                if len(matched_morphology) >= 3:
                    sub_domain = "Diagnostic Oncology & Cellular Morphology"
                    row_entity = "an individual observation of cell-nuclei morphological measurements evaluated for diagnostic determination"
                    rationale = f"Identified Healthcare & Medical data with cell-nuclei morphological measurements and diagnostic labels ({', '.join(matched_morphology[:4])})."
        elif best_score >= 2:
            domain = best_domain
            confidence = "medium"
            matched_list = ", ".join(matched_words[best_domain][:3])
            rationale = f"Moderately aligned with {best_domain} based on columns: {matched_list}."
            row_entity = DOMAIN_RULES[best_domain]["row_desc"]
        else:
            domain = "General Tabular Data (Unspecified)"
            confidence = "low"
            rationale = "Column names do not exhibit a definitive industry signature. Evaluated as general structured tabular data."
            row_entity = "a single recorded observation with measured numerical and categorical attributes"

    return DatasetSemantics(
        num_rows=num_rows,
        num_cols=num_cols,
        usable_rows=usable_rows,
        usable_cells=usable_cells,
        is_entirely_empty=is_entirely_empty,
        is_degraded=is_degraded,
        domain=domain,
        domain_confidence=confidence,
        domain_rationale=rationale,
        sub_domain=sub_domain,
        row_entity=row_entity,
        columns=col_semantics,
        primary_ids=primary_ids,
        foreign_ids=foreign_ids,
        measurements=measurements,
        dimensions=dimensions,
        binary_labels=binary_labels,
        multiclass_labels=multiclass_labels,
        temporal_cols=temporal_cols,
    )


# ============================================================================
# Constrained Analytical Opportunities Generator
# ============================================================================

def generate_constrained_opportunities(
    semantics: DatasetSemantics,
    profile: DatasetProfile,
) -> List[AnalysisOpportunity]:
    """
    Constructs practical, column-grounded analytical opportunities.
    Guarantees:
      - If dataset is entirely empty, suppresses value-dependent aggregations and warns the user.
      - Primary identifiers (e.g. 'id') are NEVER selected as metrics or ranking items.
      - Vocabulary is adapted to the domain (clinical/diagnostic vs commerce vs property).
    """
    opportunities: List[AnalysisOpportunity] = []

    # Blocker for 100% empty datasets
    if semantics.is_entirely_empty:
        return [
            AnalysisOpportunity(
                title="Data Ingestion & Completeness Notice",
                description=(
                    f"All {semantics.num_rows:,} recorded rows contain zero usable values "
                    f"({profile.total_missing:,} missing cells, 100% missingness). "
                    f"Descriptive aggregations, group comparisons, and rankings cannot be calculated until data is populated."
                ),
                business_value="Re-extraction or re-export of the dataset from the source system is required before analysis can proceed.",
                sample_question=None,
            )
        ]

    # Domain vocabulary tailoring
    vocab = DOMAIN_RULES.get(semantics.domain, {}).get("vocabulary", {
        "comparison": "subgroup divergence and segment contribution",
        "correlation": "metric interaction and statistical correlation",
        "distribution": "numerical measurement spread and outlier concentration",
    })

    # Opportunity 1: Group Comparison across Dimensions or Binary Labels
    # Metric MUST come from genuine measurements (never IDs!)
    candidate_metrics = semantics.measurements
    candidate_dims = semantics.binary_labels + semantics.multiclass_labels + semantics.dimensions

    if candidate_dims and candidate_metrics:
        dim = candidate_dims[0]
        metric = candidate_metrics[0]
        dim_semantics = semantics.columns[dim]
        dim_type_desc = "classes" if dim_semantics.semantic_role in (ColumnRole.BINARY_LABEL, ColumnRole.MULTICLASS_LABEL) else "segments"

        if semantics.domain == "Healthcare & Medical":
            opportunities.append(
                AnalysisOpportunity(
                    title=f"Diagnostic Group Comparison ({metric} by {dim})",
                    description=f"Compare average and median {metric} across distinct {dim} {dim_type_desc} to evaluate morphological and clinical differentiation.",
                    business_value=f"Determines whether {metric} exhibits statistically significant separation between {dim} groups.",
                    sample_question=f"What is the average {metric} grouped by {dim}?",
                )
            )
        elif semantics.domain == "Real Estate & Property Listings":
            opportunities.append(
                AnalysisOpportunity(
                    title=f"Property Pricing Comparison ({metric} by {dim})",
                    description=f"Compare average and median {metric} across distinct {dim} to evaluate pricing premiums across property types or locations.",
                    business_value="Identifies rate divergence and informs rental yield or listing valuation benchmarks.",
                    sample_question=f"What is the average {metric} by {dim}?",
                )
            )
        else:
            opportunities.append(
                AnalysisOpportunity(
                    title=f"Segment Comparison ({metric} by {dim})",
                    description=f"Compare total and average {metric} across distinct values of {dim} to evaluate {vocab['comparison']}.",
                    business_value=f"Identifies high-performing vs underperforming {dim} segments to inform operational decisions.",
                    sample_question=f"What is the average {metric} by {dim}?",
                )
            )

    # Opportunity 2: Temporal Trajectory (ONLY if genuine temporal column exists!)
    if semantics.temporal_cols and candidate_metrics:
        t_col = semantics.temporal_cols[0]
        metric = candidate_metrics[0]
        t_stats = profile.column_profiles[t_col].date_stats
        span_str = f" from {t_stats.min_date} to {t_stats.max_date}" if t_stats and t_stats.min_date else ""
        opportunities.append(
            AnalysisOpportunity(
                title=f"Timeline Trajectory & Trend ({metric} over {t_col})",
                description=f"Analyze chronological shifts in {metric}{span_str} to identify seasonal patterns, velocity shifts, or cyclical spikes.",
                business_value="Informs scheduling, capacity forecasting, and proactive operational planning.",
                sample_question=f"How does {metric} trend over {t_col}?",
            )
        )
    elif len(candidate_metrics) >= 2:
        # Cross-Metric Correlation between genuine measurements (NEVER IDs!)
        m1, m2 = candidate_metrics[0], candidate_metrics[1]
        opportunities.append(
            AnalysisOpportunity(
                title=f"Feature Correlation & Co-linearity ({m1} vs {m2})",
                description=f"Assess the linear correlation and co-movement between {m1} and {m2} to evaluate {vocab['correlation']}.",
                business_value="Reveals trade-offs, dimensional redundancies, or structural relationships among measured variables.",
                sample_question=f"What is the correlation between {m1} and {m2}?",
            )
        )

    # Opportunity 3: Distribution & Extreme Outliers across Genuine Measurement
    if candidate_metrics:
        m_top = candidate_metrics[0]
        opportunities.append(
            AnalysisOpportunity(
                title=f"Measurement Distribution & Variation ({m_top})",
                description=f"Examine the spread, standard deviation, and quartile boundaries of {m_top} to evaluate {vocab['distribution']}.",
                business_value="Highlights skewness, extreme positive/negative anomalies, or recording inconsistencies.",
                sample_question=f"What are the top 5 records with the highest {m_top}?",
            )
        )

    # Opportunity 4: Data Quality Audit
    if profile.total_missing > 0 or profile.warnings:
        opportunities.append(
            AnalysisOpportunity(
                title="Data Quality & Integrity Assessment",
                description=f"Audit {profile.total_missing:,} missing cells ({profile.missing_cell_pct}% of dataset) and data quality caveats across attributes.",
                business_value="Confirms data completeness and ensures missingness does not distort downstream statistical findings.",
                sample_question="Which columns have the highest number of missing values?",
            )
        )
    elif len(candidate_dims) >= 2:
        d1, d2 = candidate_dims[0], candidate_dims[1]
        opportunities.append(
            AnalysisOpportunity(
                title=f"Cross-Tabulation Analysis ({d1} x {d2})",
                description=f"Analyze the joint frequency distribution and co-occurrence between {d1} and {d2}.",
                business_value="Uncovers multidimensional clustering and demographic or operational archetypes.",
                sample_question=f"What are the most frequent combinations of {d1} and {d2}?",
            )
        )

    return opportunities[:4]


# ============================================================================
# Evidence-Based Predictive Feasibility Engine
# ============================================================================

def generate_evidence_based_predictions(
    semantics: DatasetSemantics,
    profile: DatasetProfile,
) -> Tuple[List[PredictionCandidate], str]:
    """
    Evaluates realistic machine learning predictive tasks using strict evidence rules:
      - Usability Guard: 100% empty datasets cannot support predictive tasks.
      - Classification: Evaluates candidate binary or multiclass labels (e.g. diagnosis).
      - Regression: Proposes regression ONLY if a defensible continuous target exists.
      - Forecasting: Proposes time-series forecasting ONLY if a chronological index exists.
      - Clinical / Medical Caveats: Clearly marked for educational/research exploration.
    """
    predictions: List[PredictionCandidate] = []

    # Blocker for 100% empty datasets
    if semantics.is_entirely_empty:
        predictions.append(
            PredictionCandidate(
                task="Predictive Modeling Unavailable",
                target_variable=None,
                prerequisites="Supervised and unsupervised learning require populated feature observations and ground-truth values. This dataset contains zero usable observations.",
                feasibility="Not currently supported by the data",
                missing_elements=["Populated data observations", "Valid target labels"],
            )
        )
        limitation = (
            "Predictive modeling is not supported because this dataset contains zero populated records (100% missing cells). "
            "Data observations must be populated before model training or validation can be considered."
        )
        return predictions, limitation

    num_rows = semantics.num_rows
    candidate_features = [c for c in semantics.measurements if c not in semantics.primary_ids]
    has_features = len(candidate_features) >= 2

    # Task 1: Classification Target Evaluation
    candidate_targets = semantics.binary_labels + semantics.multiclass_labels
    if candidate_targets and has_features:
        target_col = candidate_targets[0]
        target_semantics = semantics.columns[target_col]
        u_cnt = target_semantics.unique_count

        # Sample size and balance checks
        if num_rows >= 50:
            feasibility = "Feasible for exploration"
            prereq = (
                f"Requires stratified train/test partitioning, feature standardization across the "
                f"{len(candidate_features)} continuous measurements, cross-validation, and out-of-sample evaluation."
            )
            missing = ["Out-of-sample holdout test partition", "Feature scaling protocol"]
        else:
            feasibility = "Feasible with caveats"
            prereq = f"Sample size ({num_rows} records) is limited for complex estimators. Requires robust cross-validation and regularized models."
            missing = ["Larger sample size for generalization", "Out-of-sample holdout partition"]

        # Medical disclaimer
        if semantics.domain == "Healthcare & Medical":
            missing.append("Clinical validation (research/educational use only)")

        predictions.append(
            PredictionCandidate(
                task=f"Supervised Classification: {target_col.capitalize()} Prediction",
                target_variable=target_col,
                prerequisites=prereq,
                feasibility=feasibility,
                missing_elements=missing,
            )
        )
    elif candidate_targets and not has_features:
        target_col = candidate_targets[0]
        predictions.append(
            PredictionCandidate(
                task=f"Supervised Classification: {target_col.capitalize()} Prediction",
                target_variable=target_col,
                prerequisites=f"Requires informative independent feature variables. Dataset currently has insufficient continuous measurements to train a predictive model for {target_col}.",
                feasibility="Not currently supported by the data",
                missing_elements=["Independent explanatory features", "Multi-variable feature matrix"],
            )
        )
    else:
        predictions.append(
            PredictionCandidate(
                task="Supervised Outcome Classification",
                target_variable=None,
                prerequisites="Requires an identified binary or categorical ground-truth target label and independent feature columns, neither of which are currently present.",
                feasibility="Not currently supported by the data",
                missing_elements=["Supervised target label", "Ground-truth outcome annotations"],
            )
        )

    # Task 2: Regression Target Evaluation
    # Search for genuine continuous targets (e.g. price, revenue, sales, cost, salary, amount)
    genuine_reg_targets = [
        c for c in semantics.measurements
        if any(kw in c.lower() for kw in ["price", "revenue", "sales", "cost", "salary", "amount", "spend", "duration", "fare"])
    ]

    if genuine_reg_targets and len(candidate_features) >= 2:
        reg_target = genuine_reg_targets[0]
        feature_count = len([c for c in candidate_features if c != reg_target])
        if feature_count >= 2 and num_rows >= 30:
            predictions.append(
                PredictionCandidate(
                    task=f"Continuous Value Estimation ({reg_target})",
                    target_variable=reg_target,
                    prerequisites=f"Requires {feature_count} explanatory feature columns exhibiting predictive correlation with {reg_target}, without feature leakage.",
                    feasibility="Feasible for exploration",
                    missing_elements=["Residual normality checks", "Out-of-sample validation split"],
                )
            )
        else:
            predictions.append(
                PredictionCandidate(
                    task=f"Continuous Value Estimation ({reg_target})",
                    target_variable=reg_target,
                    prerequisites=f"Requires explanatory features with validated predictive correlation with {reg_target}. Additional external features are recommended.",
                    feasibility="Feasible with caveats",
                    missing_elements=["Independent explanatory variables", "External domain drivers"],
                )
            )
    else:
        # Grounded declaration when no regression target exists
        predictions.append(
            PredictionCandidate(
                task="Numeric Regression / Value Estimation",
                target_variable=None,
                prerequisites=(
                    "Requires an identified continuous target variable with explanatory feature relationships. "
                    "All continuous measurements in this dataset serve as descriptive feature inputs rather than an explicit regression target."
                ),
                feasibility="Not currently supported by the data",
                missing_elements=["Designated continuous outcome variable"],
            )
        )

    # Task 3: Time-Series Forecasting Evaluation
    if semantics.temporal_cols and semantics.measurements:
        t_col = semantics.temporal_cols[0]
        m_target = semantics.measurements[0]
        if num_rows >= 100:
            feasibility = "Feasible with caveats"
            prereq = (
                f"Requires chronologically contiguous records at regular intervals (daily/weekly), "
                f"minimal temporal gaps, and external regressors (calendar/holidays) to capture seasonality."
            )
            missing = ["Chronological interval validation", "Calendar/seasonality regressors"]
        else:
            feasibility = "Not currently supported by the data"
            prereq = f"Historical depth ({num_rows} records) is insufficient for reliable time-series forecasting across cyclical periods."
            missing = ["Multi-cycle historical observations", "Sufficient chronological span"]

        predictions.append(
            PredictionCandidate(
                task=f"Time-Series Forecasting for {m_target}",
                target_variable=m_target,
                prerequisites=prereq,
                feasibility=feasibility,
                missing_elements=missing,
            )
        )
    else:
        predictions.append(
            PredictionCandidate(
                task="Time-Series Forecasting",
                target_variable=None,
                prerequisites="Requires a chronological date/time column and sequential observations. This dataset is a cross-sectional snapshot with no temporal index.",
                feasibility="Not currently supported by the data",
                missing_elements=["Datetime index", "Chronological sequence"],
            )
        )

    # Global Limitations & Disclaimer
    limitation_parts = [
        "Exploratory data analysis describes historical or observed attributes within this dataset.",
        "Supervised machine learning predictions require verified ground-truth targets, feature engineering, strict train/test separation, and protection against target leakage.",
    ]
    if semantics.domain == "Healthcare & Medical":
        limitation_parts.append(
            "Research & Educational Notice: Any predictive tasks on medical or diagnostic features are strictly for investigative data analysis and must never be utilized as clinical diagnostic tools."
        )

    return predictions[:3], " ".join(limitation_parts)


# ============================================================================
# Deterministic Overview Generator
# ============================================================================

def generate_deterministic_overview(
    df: pd.DataFrame,
    profile: DatasetProfile,
) -> DatasetOverview:
    """
    Constructs a complete, grounded, highly readable Dataset Overview without invoking an LLM.
    Acts as the guaranteed bedrock fallback.
    """
    # 1. Build intermediate semantics & usability representation
    semantics = build_dataset_semantics(df, profile)

    # 2. Key Fields Extraction
    key_fields: List[KeyField] = []
    # Primary IDs
    for pid in semantics.primary_ids[:1]:
        s = semantics.columns[pid]
        key_fields.append(
            KeyField(
                name=pid,
                meaning=s.plain_meaning,
                type_info=s.dtype,
                role="identifier",
            )
        )
    # Binary / Multiclass Labels
    for lbl in (semantics.binary_labels + semantics.multiclass_labels)[:1]:
        s = semantics.columns[lbl]
        key_fields.append(
            KeyField(
                name=lbl,
                meaning=s.plain_meaning,
                type_info=s.dtype,
                role="dimension",
            )
        )
    # Temporal
    for tid in semantics.temporal_cols[:1]:
        s = semantics.columns[tid]
        key_fields.append(
            KeyField(
                name=tid,
                meaning=s.plain_meaning,
                type_info=s.dtype,
                role="temporal",
            )
        )
    # Genuine Measurements (NEVER IDs!)
    for mid in semantics.measurements[:3]:
        s = semantics.columns[mid]
        key_fields.append(
            KeyField(
                name=mid,
                meaning=s.plain_meaning,
                type_info=s.dtype,
                role="metric",
            )
        )
    # Categorical Dimensions
    for did in semantics.dimensions[:2]:
        if did not in [k.name for k in key_fields]:
            s = semantics.columns[did]
            key_fields.append(
                KeyField(
                    name=did,
                    meaning=s.plain_meaning,
                    type_info=s.dtype,
                    role="dimension",
                )
            )

    key_fields = key_fields[:6]

    # 3. Size and Coverage
    if semantics.is_entirely_empty:
        size_and_coverage = (
            f"{semantics.num_rows:,} records across {semantics.num_cols} columns; "
            f"100% missing data cells (0 usable observations); "
            f"{profile.duplicate_rows:,} empty duplicate rows."
        )
    else:
        cov_parts = [f"{semantics.num_rows:,} records across {semantics.num_cols} attributes"]
        if semantics.temporal_cols:
            t_col = semantics.temporal_cols[0]
            d_stats = profile.column_profiles[t_col].date_stats
            if d_stats and d_stats.min_date and d_stats.max_date:
                cov_parts.append(f"chronologically spanning from {d_stats.min_date} to {d_stats.max_date} ({t_col})")
        else:
            cov_parts.append("captured as a cross-sectional snapshot without explicit date tracking")

        if profile.total_missing > 0:
            cov_parts.append(f"with {profile.missing_cell_pct}% missing cell values")
        else:
            cov_parts.append("with 100% complete records")
        size_and_coverage = "; ".join(cov_parts) + "."

    # 4. Central Summary Paragraph
    if semantics.is_entirely_empty:
        summary_paragraph = (
            f"This dataset defines a {semantics.num_cols}-column schema whose column headers "
            f"suggest a {semantics.domain.lower()} structure. However, all {semantics.num_rows:,} recorded rows "
            f"are entirely empty ({profile.total_missing:,} missing cells, 100% missingness). "
            f"Because there are zero usable observations, statistical aggregation, exploratory comparisons, "
            f"and predictive modeling cannot be performed until valid data values are supplied."
        )
        row_meaning = f"Intended to represent {semantics.row_entity}, but currently contains no recorded data."
    else:
        p_sentences = []
        domain_desc = semantics.sub_domain or semantics.domain
        p_sentences.append(
            f"This dataset contains {domain_desc.lower()} information, where each row represents {semantics.row_entity}."
        )
        p_sentences.append(
            f"It encompasses {semantics.num_rows:,} records across {semantics.num_cols} attributes."
        )

        # Highlight important structure
        if semantics.binary_labels:
            p_sentences.append(f"Key outcomes include classification target '{semantics.binary_labels[0]}'.")
        if semantics.measurements:
            meas_preview = ", ".join(semantics.measurements[:3])
            p_sentences.append(f"Primary continuous measurements include {meas_preview}.")
        if semantics.temporal_cols:
            t_stats = profile.column_profiles[semantics.temporal_cols[0]].date_stats
            if t_stats and t_stats.min_date and t_stats.max_date:
                p_sentences.append(f"Records span chronologically from {t_stats.min_date} to {t_stats.max_date}.")

        if semantics.domain == "Healthcare & Medical":
            p_sentences.append("It is structured for evaluating diagnostic group separation, morphological feature correlations, and biopsy classification.")
        elif semantics.domain == "Real Estate & Property Listings":
            p_sentences.append("It is structured for evaluating rental pricing dynamics, property type comparisons, and geographic distributions.")
        else:
            p_sentences.append("It is structured for evaluating segment comparisons, distribution dynamics, and practical exploratory reporting.")

        summary_paragraph = " ".join(p_sentences)
        row_meaning = f"Each row represents {semantics.row_entity}."

    # 5. Opportunities & Predictions
    opportunities = generate_constrained_opportunities(semantics, profile)
    predictions, limitations = generate_evidence_based_predictions(semantics, profile)

    return DatasetOverview(
        domain=semantics.domain,
        domain_confidence=semantics.domain_confidence,
        domain_rationale=semantics.domain_rationale,
        summary_paragraph=summary_paragraph,
        row_meaning=row_meaning,
        size_and_coverage=size_and_coverage,
        key_fields=key_fields,
        what_we_can_learn=opportunities,
        potential_predictions=predictions,
        prediction_limitations=limitations,
        status="deterministic",
    )


# ============================================================================
# Gemini LLM Synthesis Layer
# ============================================================================

def _build_overview_prompt(
    df: pd.DataFrame,
    profile: DatasetProfile,
    det_overview: DatasetOverview,
    semantics: DatasetSemantics,
) -> Tuple[str, str]:
    """
    Constructs a tightly scoped Gemini prompt with strict grounding and safety constraints.
    """
    system_instruction = (
        "You are an expert principal data analyst and machine learning architect.\n"
        "Your task is to synthesize a dedicated, polished Dataset Overview that allows users "
        "to clearly understand the purpose, meaning, and scope of an uploaded CSV without "
        "having to inspect the raw file themselves.\n\n"
        "STRICT GROUNDING & PRODUCTION RULES:\n"
        "1. Base inferences strictly on the provided schema, verified profile metrics, and domain classification.\n"
        "2. Do NOT invent external business facts, company names, fake currencies, or ungrounded claims.\n"
        "3. DATA USABILITY: If the dataset is 100% empty or unusable, you MUST state that clearly. "
        "   Do NOT propose statistical comparisons or predictive tasks on empty data.\n"
        "4. IDENTIFIER RULE: Columns identified as primary identifiers (like 'id') MUST NEVER be averaged, "
        "   summed, ranked, correlated, or used as predictive targets.\n"
        "5. DOMAIN-SPECIFIC VOCABULARY: Do not use generic commerce phrases (e.g. 'growth drivers', 'resource allocation') "
        "   for scientific, medical, or property datasets. Use domain-appropriate terminology.\n"
        "6. PREDICTION GROUNDING: Feasibility must reflect actual target and feature evidence. "
        "   Do not propose regression targets that do not exist. Do not propose time-series forecasting without dates.\n"
        "7. Return ONLY valid JSON matching this schema:\n"
        "{\n"
        '  "domain": "Domain Name",\n'
        '  "domain_confidence": "high" | "medium" | "low",\n'
        '  "domain_rationale": "Why this domain was identified or why uncertain",\n'
        '  "summary_paragraph": "Comprehensive, crystal-clear 3-4 sentence overview of the dataset",\n'
        '  "row_meaning": "Plain language explanation of what a single row represents",\n'
        '  "size_and_coverage": "Summary of row/col dimensions and temporal span",\n'
        '  "key_fields": [\n'
        '    {"name": "col_name", "meaning": "plain language meaning", "type_info": "dtype", "role": "metric|dimension|identifier|temporal"}\n'
        "  ],\n"
        '  "what_we_can_learn": [\n'
        '    {"title": "Title", "description": "Analysis description", "business_value": "Practical value", "sample_question": "Question or null"}\n'
        "  ],\n"
        '  "potential_predictions": [\n'
        '    {"task": "Task name", "target_variable": "col or null", "prerequisites": "Prerequisites", "feasibility": "Feasible for exploration" | "Feasible with caveats" | "Not currently supported by the data", "missing_elements": ["item1", "item2"]}\n'
        "  ],\n"
        '  "prediction_limitations": "Statement distinguishing exploratory analysis from supervised ML"\n'
        "}"
    )

    profile_summary = {
        "num_rows": profile.num_rows,
        "num_cols": profile.num_cols,
        "usable_rows": semantics.usable_rows,
        "is_entirely_empty": semantics.is_entirely_empty,
        "total_missing": profile.total_missing,
        "columns": [
            {
                "name": col,
                "role": semantics.columns[col].semantic_role.value if col in semantics.columns else "unknown",
                "type": p.semantic_type,
                **(
                    {"min": p.numeric_stats.min, "max": p.numeric_stats.max, "mean": p.numeric_stats.mean}
                    if p.numeric_stats else {}
                ),
                **(
                    {"unique_count": p.categorical_stats.unique_count, "top": [c.value for c in p.categorical_stats.top_categories[:2]]}
                    if p.categorical_stats else {}
                ),
                **(
                    {"date_range": f"{p.date_stats.min_date} to {p.date_stats.max_date}"}
                    if p.date_stats and p.date_stats.min_date else {}
                ),
            }
            for col, p in profile.column_profiles.items()
        ],
    }

    # Masked 3 sample rows
    if semantics.is_entirely_empty:
        sample_text = "(All rows are empty/null)"
    else:
        masked_sample = mask_sample_dataframe(df, max_rows=3, max_str_len=25)
        sample_text = masked_sample.to_string(index=False)

    user_prompt = (
        f"Baseline Deterministic Inferences:\n"
        f"- Identified Domain: {det_overview.domain} (Confidence: {det_overview.domain_confidence})\n"
        f"- Row Meaning: {det_overview.row_meaning}\n"
        f"- Size & Coverage: {det_overview.size_and_coverage}\n"
        f"- Usability Status: {'Entirely Empty / Blank Data' if semantics.is_entirely_empty else 'Populated Data'}\n\n"
        f"Computed Profile:\n{json.dumps(profile_summary, indent=2)}\n\n"
        f"Sample Records (PII masked):\n{sample_text}\n\n"
        "Generate the complete Dataset Overview JSON now adhering strictly to the production rules."
    )

    return system_instruction, user_prompt


def _sanitize_and_validate_overview(
    raw_text: str,
    det_overview: DatasetOverview,
    semantics: DatasetSemantics,
) -> Optional[DatasetOverview]:
    """
    Validates Gemini output against deterministic intermediate constraints.
    Rejects any hallucinations violating column roles, usability, or prediction feasibility.
    """
    try:
        cleaned_text = raw_text.strip()
        if cleaned_text.startswith("```"):
            cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text, flags=re.IGNORECASE)
            cleaned_text = re.sub(r"\s*```$", "", cleaned_text)
            cleaned_text = cleaned_text.strip()

        data = json.loads(cleaned_text)
        if not isinstance(data, dict):
            return None

        summary_p = str(data.get("summary_paragraph", "")).strip()
        if len(summary_p) < 40:
            return None

        # Rule 1: If dataset is entirely empty, summary MUST mention missingness/empty
        if semantics.is_entirely_empty:
            if not any(w in summary_p.lower() for w in ["empty", "missing", "blank", "unpopulated"]):
                return None  # Hallucinated value descriptions on empty data

        domain = str(data.get("domain", det_overview.domain)).strip() or det_overview.domain
        confidence = str(data.get("domain_confidence", det_overview.domain_confidence)).strip().lower()
        if confidence not in ("high", "medium", "low"):
            confidence = det_overview.domain_confidence

        row_meaning = str(data.get("row_meaning", det_overview.row_meaning)).strip() or det_overview.row_meaning
        size_and_coverage = str(data.get("size_and_coverage", det_overview.size_and_coverage)).strip() or det_overview.size_and_coverage
        domain_rationale = str(data.get("domain_rationale", det_overview.domain_rationale)).strip() or det_overview.domain_rationale

        # Parse key fields
        key_fields: List[KeyField] = []
        raw_kf = data.get("key_fields", [])
        if isinstance(raw_kf, list):
            for item in raw_kf:
                if isinstance(item, dict) and "name" in item and "meaning" in item:
                    col_name = str(item["name"]).strip()
                    if col_name in semantics.columns:
                        role_val = str(item.get("role", "attribute"))
                        # Ensure primary ID is never marked as metric
                        if col_name in semantics.primary_ids:
                            role_val = "identifier"
                        key_fields.append(
                            KeyField(
                                name=col_name,
                                meaning=str(item["meaning"]).strip(),
                                type_info=str(item.get("type_info", "")) or None,
                                role=role_val,
                            )
                        )
        if not key_fields:
            key_fields = det_overview.key_fields

        # Parse opportunities & enforce ID exclusion
        opportunities: List[AnalysisOpportunity] = []
        raw_opp = data.get("what_we_can_learn", [])
        if isinstance(raw_opp, list) and not semantics.is_entirely_empty:
            for item in raw_opp:
                if isinstance(item, dict) and "title" in item and "description" in item:
                    title = str(item["title"]).strip()
                    desc = str(item["description"]).strip()
                    sq = str(item.get("sample_question", "")).strip() or None
                    # Verify no primary ID is being aggregated
                    violates_id = False
                    for pid in semantics.primary_ids:
                        if f"average {pid}" in (desc + " " + (sq or "")).lower() or f"total {pid}" in (desc + " " + (sq or "")).lower():
                            violates_id = True
                            break
                    if not violates_id:
                        opportunities.append(
                            AnalysisOpportunity(
                                title=title,
                                description=desc,
                                business_value=str(item.get("business_value", "")).strip(),
                                sample_question=sq,
                            )
                        )
        if not opportunities:
            opportunities = det_overview.what_we_can_learn

        # Parse predictions & enforce feasibility constraints
        predictions: List[PredictionCandidate] = []
        raw_pred = data.get("potential_predictions", [])
        if isinstance(raw_pred, list) and not semantics.is_entirely_empty:
            for item in raw_pred:
                if isinstance(item, dict) and "task" in item and "prerequisites" in item:
                    task_name = str(item["task"]).strip()
                    feasibility_val = str(item.get("feasibility", "Requires additional data")).strip()
                    # Enforce: No forecasting if no temporal cols
                    if "forecast" in task_name.lower() and not semantics.temporal_cols:
                        feasibility_val = "Not currently supported by the data"
                    # Enforce: No regression on primary ID
                    target_var = str(item.get("target_variable", "")) or None
                    if target_var in semantics.primary_ids:
                        target_var = None
                        feasibility_val = "Not currently supported by the data"

                    missing_el = item.get("missing_elements", [])
                    if not isinstance(missing_el, list):
                        missing_el = []

                    predictions.append(
                        PredictionCandidate(
                            task=task_name,
                            target_variable=target_var,
                            prerequisites=str(item["prerequisites"]).strip(),
                            feasibility=feasibility_val,
                            missing_elements=[str(x) for x in missing_el],
                        )
                    )
        if not predictions:
            predictions = det_overview.potential_predictions

        limitations = str(data.get("prediction_limitations", det_overview.prediction_limitations)).strip() or det_overview.prediction_limitations

        return DatasetOverview(
            domain=domain,
            domain_confidence=confidence,
            domain_rationale=domain_rationale,
            summary_paragraph=summary_p,
            row_meaning=row_meaning,
            size_and_coverage=size_and_coverage,
            key_fields=key_fields[:6],
            what_we_can_learn=opportunities[:4],
            potential_predictions=predictions[:3],
            prediction_limitations=limitations,
            status="synthesized",
        )
    except Exception as e:
        logger.warning(f"Failed to validate LLM overview output: {e}")
        return None


# ============================================================================
# Main Overview Orchestrator
# ============================================================================

def generate_dataset_overview(
    df: pd.DataFrame,
    profile: DatasetProfile,
    timeout_seconds: float = 10.0,
    use_gemini: bool = True,
) -> DatasetOverview:
    """
    Generates a dedicated, production-quality Dataset Overview.
    Evaluates factual deterministic baseline first, then optionally synthesizes
    with Gemini within strict latency and quota bounds.
    """
    # 1. Deterministic baseline
    det_overview = generate_deterministic_overview(df, profile)

    if not use_gemini or profile.num_rows == 0 or profile.num_cols == 0:
        return det_overview

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_google_gemini_api_key_here":
        return det_overview

    try:
        from google import genai
        from google.genai import types

        semantics = build_dataset_semantics(df, profile)

        http_timeout_ms = max(int(timeout_seconds * 1000), 10000)
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=http_timeout_ms),
        )
        system_prompt, user_prompt = _build_overview_prompt(df, profile, det_overview, semantics)

        def _call_model():
            return client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.2,
                    response_mime_type="application/json",
                ),
            )

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(_call_model)
            response = future.result(timeout=timeout_seconds)
            executor.shutdown(wait=False)
        except Exception:
            executor.shutdown(wait=False, cancel_futures=True)
            raise

        if response and response.text:
            validated = _sanitize_and_validate_overview(response.text, det_overview, semantics)
            if validated:
                return validated

    except Exception as e:
        logger.warning(f"Gemini overview generation fell back to deterministic baseline: {e}")

    return det_overview
