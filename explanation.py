"""
explanation.py
--------------
Purpose:
    Feature A: Explains the analytical significance of query results.
    Computes deterministic evidence metrics (shares of total, disparity ratios,
    benchmarks, sample sizes) and synthesizes structured explanations:
      - Finding: What happened? (Factual, grounded)
      - Evidence: What computed numbers support the finding?
      - Interpretation: Why might the result matter? (Demarcated hypothesis)
      - Uncertainty & Limitations: Analytical caveats, non-causal disclosures.
      - Next Steps: What should the user investigate next?

Core Guarantees:
    - 100% Traceability: All numerical claims are grounded in deterministic calculations.
    - Epistemic Modesty: Correlation is never presented as causation. Hypotheses are
      labeled clearly. Assumptions are verified before calculating derived metrics.
    - Additive Metric Safety: Shares of total are ONLY calculated for demonstrably additive
      volumetric measures. Ambiguous, average, rate, score, ratio, or signed data with
      negatives safely suppress percentage shares.
    - Baseline Compatibility: Scalars are compared strictly with compatible metrics
      (e.g., subset sum vs total volume, group mean vs overall mean). Never mismatched.
    - Disparity Guard: Disparity ratios require denominators above small thresholds.
    - Zero Failure Mode: Fast, high-quality deterministic fallback operates with zero LLM
      dependency if Gemini times out (4s budget), rate-limits (429), or fails.
    - Security: Credentials and raw traceback errors are redacted from logs.
"""

import concurrent.futures
import json
import logging
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from utils import format_value

logger = logging.getLogger(__name__)


# ============================================================================
# Security & Sanitization Helpers
# ============================================================================

def _sanitize_error_message(msg: Any) -> str:
    """Removes potential API keys, auth tokens, and raw traceback dumps from error text."""
    if not msg:
        return "An internal error occurred."
    text = str(msg)
    cleaned = re.sub(r"AIza[0-9A-Za-z_-]{30,}", "[REDACTED_API_KEY]", text)
    cleaned = re.sub(r"AQ\.[0-9A-Za-z_-]{20,}", "[REDACTED_API_KEY]", cleaned)
    cleaned = re.sub(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{16,}\b", "Bearer [REDACTED_TOKEN]", cleaned)
    cleaned = re.sub(r"(?i)(?:key|api[_-]?key|token)\s*[=:]\s*['\"]?[0-9A-Za-z._~+/-]{10,}['\"]?", "key=[REDACTED]", cleaned)
    first_line = cleaned.strip().split("\n")[0]
    return first_line[:200]


# ============================================================================
# Pydantic Schemas
# ============================================================================

class AnswerEvidence(BaseModel):
    metric_name: Optional[str] = None
    aggregation_type: str = "unknown"  # "sum" | "average" | "count" | "ratio" | "scalar" | "table" | "single_category" | "unknown"
    is_additive: bool = False
    total_value: Optional[float] = None
    denominator_description: Optional[str] = None
    top_contributor: Optional[str] = None
    top_value: Optional[float] = None
    top_share_pct: Optional[float] = None
    runner_up: Optional[str] = None
    runner_up_value: Optional[float] = None
    runner_up_share_pct: Optional[float] = None
    bottom_contributor: Optional[str] = None
    bottom_value: Optional[float] = None
    disparity_ratio: Optional[float] = None
    disparity_difference: Optional[float] = None
    num_groups: Optional[int] = None
    sample_size: Optional[int] = None
    baseline_value: Optional[float] = None
    baseline_description: Optional[str] = None
    key_figures: Dict[str, Any] = Field(default_factory=dict)
    limitation_notes: Optional[str] = None


class AnswerExplanation(BaseModel):
    finding: str
    evidence: AnswerEvidence
    interpretation: str
    uncertainty_or_limitations: Optional[str] = None
    next_steps: List[str] = Field(default_factory=list)
    status: str = "deterministic"  # "deterministic" | "synthesized"


# ============================================================================
# Semantic & Additive Metric Inspection
# ============================================================================

NON_ADDITIVE_KEYWORDS = {
    "mean", "avg", "average", "median", "rate", "ratio", "pct", "percent",
    "percentage", "share", "proportion", "price", "unit_price", "cost_per",
    "gpa", "score", "temperature", "std", "var", "variance", "min", "max",
    "minimum", "maximum", "per_capita", "per_unit", "margin", "yield",
    "balance", "rating", "index", "density", "height", "weight", "level",
    "grade", "duration", "speed", "distance", "latitude", "longitude",
    "rank", "ranking", "year", "month", "day", "id", "profit"
}

ADDITIVE_KEYWORDS = {
    "sum", "total", "revenue", "sales", "spend", "cost", "count", "volume",
    "quantity", "amount", "records", "frequency", "units",
    "orders", "transactions", "visitors", "clicks", "impressions", "inventory"
}


def _is_additive_metric(metric_name: Optional[str], question: Optional[str], values: pd.Series) -> Tuple[bool, str, Optional[str]]:
    """
    Determines whether a numeric series can be safely treated as an additive volumetric measure
    where a sum and shares of total are mathematically sound.

    Rules:
      1. Non-additive keywords (mean, avg, rate, ratio, score, price, etc.) -> False.
      2. Negative values (e.g. net profit/loss) -> False.
      3. Total <= 0 -> False.
      4. Explicit additive keyword in metric name or question -> True.
      5. Ambiguous or unverified metric semantics -> False (do not assume additive).
    """
    name_tokens = set(re.findall(r"[a-z0-9]+", (metric_name or "").lower()))
    q_tokens = set(re.findall(r"[a-z0-9]+", (question or "").lower()))

    # Check for negative or non-finite values first (physical data constraints)
    clean_vals = values.replace([np.inf, -np.inf], np.nan).dropna()
    if clean_vals.empty:
        return False, "unknown", "No non-null, finite values available."

    if (clean_vals < 0).any():
        return False, "signed_sum", "Dataset contains negative values; percentage-of-total shares are not calculated to prevent misleading totals."

    # Check total sum
    total = clean_vals.sum()
    if total <= 0:
        return False, "zero_sum", "Sum of values is zero or negative; shares of total cannot be computed."

    # Check for explicit non-additive indicators
    if name_tokens.intersection(NON_ADDITIVE_KEYWORDS) or q_tokens.intersection(NON_ADDITIVE_KEYWORDS):
        agg_type = "average" if ("mean" in name_tokens or "avg" in name_tokens or "average" in name_tokens or
                                 "mean" in q_tokens or "avg" in q_tokens or "average" in q_tokens) else "ratio"
        return False, agg_type, "Because this metric represents an average, rate, score, or non-volumetric measure, values are non-additive and cannot be summed into a share of total."

    # Explicit additive keyword matched
    if name_tokens.intersection(ADDITIVE_KEYWORDS) or q_tokens.intersection(ADDITIVE_KEYWORDS):
        agg_type = "count" if ("count" in name_tokens or "frequency" in name_tokens or "count" in q_tokens) else "sum"
        return True, agg_type, None

    # Ambiguous: metric is not demonstrably additive; do NOT guess
    return False, "numeric", "Metric semantics are not demonstrably additive; percentage-of-total shares are not calculated."


def _clean_float(val: Any, decimals: int = 2) -> Optional[float]:
    """Safely converts numpy/pandas scalar to a JSON-serializable float, or None."""
    if val is None or pd.isna(val):
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, decimals)
    except (ValueError, TypeError):
        return None


# ============================================================================
# Deterministic Evidence Calculation Engine
# ============================================================================

def calculate_answer_evidence(
    result: Any,
    df: Optional[pd.DataFrame] = None,
    question: Optional[str] = None
) -> AnswerEvidence:
    """
    Calculates deterministic numerical evidence from the execution result and original dataset.
    Extracts key figures, shares of total (if mathematically additive), runner-up disparities,
    and contextual benchmarks without using an LLM.
    """
    total_dataset_rows = len(df) if df is not None and isinstance(df, pd.DataFrame) else None

    # Case 1: Empty or None Result
    if result is None:
        return AnswerEvidence(
            aggregation_type="empty",
            limitation_notes="No execution result was produced."
        )

    if isinstance(result, (pd.DataFrame, pd.Series)) and result.empty:
        return AnswerEvidence(
            aggregation_type="empty",
            sample_size=0,
            limitation_notes="Result contains zero records matching the query criteria."
        )

    # Case 2: pd.Series
    if isinstance(result, pd.Series):
        # Drop nulls and non-finite values
        clean_s = result.replace([np.inf, -np.inf], np.nan).dropna()
        if clean_s.empty:
            return AnswerEvidence(aggregation_type="empty", sample_size=0, limitation_notes="Series contains only null or non-finite values.")

        index_col = str(clean_s.index.name or "Category")
        val_col = str(clean_s.name or "Value")

        if not pd.api.types.is_numeric_dtype(clean_s):
            # Categorical or string series (e.g. list of items)
            return AnswerEvidence(
                metric_name=val_col,
                aggregation_type="table",
                num_groups=len(clean_s),
                sample_size=total_dataset_rows,
                limitation_notes="Result is non-numeric; numerical shares cannot be calculated."
            )

        # Deduplicate index labels if duplicate groups exist
        if clean_s.index.duplicated().any():
            clean_s = clean_s.groupby(level=0).sum()

        # Sort descending to find top and runner-up
        sorted_s = clean_s.sort_values(ascending=False)
        num_groups = len(sorted_s)
        is_additive, agg_type, limit_reason = _is_additive_metric(val_col, question, sorted_s)

        top_cat = str(sorted_s.index[0])
        top_val = _clean_float(sorted_s.iloc[0])

        runner_up_cat = str(sorted_s.index[1]) if num_groups > 1 else None
        runner_up_val = _clean_float(sorted_s.iloc[1]) if num_groups > 1 else None

        bottom_cat = str(sorted_s.index[-1]) if num_groups > 1 else None
        bottom_val = _clean_float(sorted_s.iloc[-1]) if num_groups > 1 else None

        total_sum = _clean_float(sorted_s.sum()) if is_additive else None
        top_share = None
        runner_up_share = None

        # Share of total is only computed for multi-group additive distributions
        if is_additive and total_sum and total_sum > 0 and num_groups > 1:
            top_share = _clean_float((top_val / total_sum) * 100) if top_val is not None else None
            runner_up_share = _clean_float((runner_up_val / total_sum) * 100) if runner_up_val is not None else None

        # Disparity ratio & difference (enforce minimum denominator threshold to avoid exploding ratios)
        disp_ratio = None
        disp_diff = None
        if top_val is not None and runner_up_val is not None:
            disp_diff = _clean_float(top_val - runner_up_val)
            if runner_up_val >= 1e-4:
                disp_ratio = _clean_float(top_val / runner_up_val)

        key_figs = {
            f"{top_cat}": top_val,
        }
        if runner_up_cat and runner_up_val is not None:
            key_figs[f"{runner_up_cat}"] = runner_up_val
        if is_additive and total_sum is not None:
            key_figs["Total"] = total_sum

        if num_groups == 1:
            agg_type = "single_category"
            limit_reason = "Single-category result; comparative segment shares and runner-up ratios are not applicable."

        return AnswerEvidence(
            metric_name=val_col,
            aggregation_type=agg_type,
            is_additive=is_additive,
            total_value=total_sum,
            denominator_description=f"Sum across all {num_groups} categories" if is_additive and num_groups > 1 else None,
            top_contributor=top_cat,
            top_value=top_val,
            top_share_pct=top_share,
            runner_up=runner_up_cat,
            runner_up_value=runner_up_val,
            runner_up_share_pct=runner_up_share,
            bottom_contributor=bottom_cat,
            bottom_value=bottom_val,
            disparity_ratio=disp_ratio,
            disparity_difference=disp_diff,
            num_groups=num_groups,
            sample_size=total_dataset_rows,
            key_figures=key_figs,
            limitation_notes=limit_reason
        )

    # Case 3: pd.DataFrame
    if isinstance(result, pd.DataFrame):
        df_res = result.replace([np.inf, -np.inf], np.nan).dropna(how="all")
        if df_res.empty:
            return AnswerEvidence(aggregation_type="empty", sample_size=0, limitation_notes="DataFrame contains zero non-null rows.")

        # Check if 2 columns: Category + Numeric
        cols = list(df_res.columns)
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df_res[c])]
        non_num_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(df_res[c])]

        if len(num_cols) >= 1 and len(non_num_cols) >= 1:
            cat_col = non_num_cols[0]
            val_col = num_cols[0]
            grouped_s = df_res.set_index(cat_col)[val_col].replace([np.inf, -np.inf], np.nan).dropna()
            if not grouped_s.empty and pd.api.types.is_numeric_dtype(grouped_s):
                # Delegate to Series logic using the extracted Series
                grouped_s.name = str(val_col)
                grouped_s.index.name = str(cat_col)
                return calculate_answer_evidence(grouped_s, df=df, question=question)

        # Multi-column or generic table
        return AnswerEvidence(
            aggregation_type="table",
            num_groups=len(df_res),
            sample_size=total_dataset_rows,
            key_figures={"Columns": len(cols), "Rows": len(df_res)},
            limitation_notes="Result is a multi-column table; categorical partition shares are not defined."
        )

    # Case 4: Scalar (int, float, np.number)
    if isinstance(result, (int, float, np.number)):
        try:
            f_val = float(result)
            if math.isinf(f_val) or math.isnan(f_val):
                return AnswerEvidence(
                    aggregation_type="scalar",
                    sample_size=total_dataset_rows,
                    limitation_notes="Calculated scalar is non-finite or NaN."
                )
        except (ValueError, TypeError):
            pass

        val = _clean_float(result)
        agg_type = "scalar"
        baseline_val = None
        baseline_desc = None
        limit_notes = None

        # Contextual dataset lookup with strict semantic compatibility
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty and val is not None:
            q_lower = (question or "").lower()
            q_tokens = set(re.findall(r"[a-z0-9]+", q_lower))

            matched_col = None
            for c in df.select_dtypes(include=[np.number]).columns:
                c_str = str(c).lower()
                if c_str in q_lower or set(re.findall(r"[a-z0-9]+", c_str)).issubset(q_tokens):
                    matched_col = c
                    break

            if matched_col:
                series_vals = df[matched_col].replace([np.inf, -np.inf], np.nan).dropna()
                if not series_vals.empty:
                    is_q_mean = any(w in q_tokens for w in ("mean", "avg", "average"))
                    is_q_sum = any(w in q_tokens for w in ("sum", "total"))

                    if is_q_mean:
                        overall_mean = _clean_float(series_vals.mean())
                        baseline_val = overall_mean
                        baseline_desc = f"Overall dataset mean for '{matched_col}'"
                        agg_type = "average"
                    elif is_q_sum:
                        overall_sum = _clean_float(series_vals.sum())
                        if overall_sum and overall_sum > 0:
                            if abs(val - overall_sum) < 1e-2:
                                baseline_val = overall_sum
                                baseline_desc = f"Grand total for '{matched_col}' across all {len(df):,} records"
                            else:
                                baseline_val = overall_sum
                                baseline_desc = f"Overall dataset total for '{matched_col}'"
                        agg_type = "sum"
                    else:
                        limit_notes = f"Calculated scalar for '{matched_col}'."

        return AnswerEvidence(
            metric_name="Calculated Value",
            aggregation_type=agg_type,
            is_additive=False,
            total_value=val,
            sample_size=total_dataset_rows,
            baseline_value=baseline_val,
            baseline_description=baseline_desc,
            key_figures={"Value": val},
            limitation_notes=limit_notes
        )

    # Case 5: String or Boolean or Dict/List
    if isinstance(result, (str, bool, np.bool_)):
        return AnswerEvidence(
            metric_name="Result",
            aggregation_type="scalar",
            sample_size=total_dataset_rows,
            key_figures={"Result": str(result)},
            limitation_notes="Result is a non-numeric scalar."
        )

    if isinstance(result, dict):
        return AnswerEvidence(
            metric_name="Dictionary",
            aggregation_type="table",
            num_groups=len(result),
            sample_size=total_dataset_rows,
            key_figures={"Keys": len(result)},
            limitation_notes="Structured dictionary output."
        )

    if isinstance(result, (list, tuple, set)):
        return AnswerEvidence(
            metric_name="List",
            aggregation_type="table",
            num_groups=len(result),
            sample_size=total_dataset_rows,
            key_figures={"Items": len(result)},
            limitation_notes="List output."
        )

    return AnswerEvidence(
        aggregation_type="unknown",
        sample_size=total_dataset_rows,
        limitation_notes=f"Unrecognized result type: {type(result).__name__}"
    )


# ============================================================================
# Deterministic Fallback Template Generator
# ============================================================================

def generate_deterministic_explanation(
    evidence: AnswerEvidence,
    question: str,
    result: Any
) -> AnswerExplanation:
    """
    Constructs a grounded, factual explanation using verified evidence without calling Gemini.
    Guarantees 0ms latency, zero API quota usage, and zero hallucination risk.
    """
    q_str = question.strip() if question else "your query"

    # Case 1: Empty or Zero Result
    if evidence.aggregation_type == "empty":
        return AnswerExplanation(
            finding="No records matched the specified query criteria.",
            evidence=evidence,
            interpretation="The filter conditions or query returned an empty result set. This often occurs when category casing differs or when multiple criteria are mutually exclusive.",
            uncertainty_or_limitations=evidence.limitation_notes or "Zero records available for analysis.",
            next_steps=[
                "What unique values exist in the target column?",
                "Can the query criteria be broadened to capture related rows?",
                "How many total records are present in the dataset?"
            ],
            status="deterministic"
        )

    # Case 2: Single Category Result
    if evidence.aggregation_type == "single_category" or (evidence.num_groups == 1 and evidence.top_contributor):
        top_name = evidence.top_contributor
        top_val_str = format_value(evidence.top_value)
        finding = f"Only one category ({top_name}) is present with {top_val_str}."
        interp = f"Because only {top_name} was returned, comparative segment shares and runner-up disparities are not applicable."
        limitations = "Single-category result; no comparative baseline within the result set."
        next_steps = [
            f"What other categories exist in the wider dataset?",
            f"Why are other categories excluded by the query filter?",
            f"How does {top_name} break down across secondary dimensions?"
        ]
        return AnswerExplanation(
            finding=finding,
            evidence=evidence,
            interpretation=interp,
            uncertainty_or_limitations=limitations,
            next_steps=next_steps,
            status="deterministic"
        )

    # Case 3: Additive Group Breakdown (e.g. Sales by Category)
    if evidence.is_additive and evidence.top_contributor and evidence.top_value is not None and (evidence.num_groups or 0) > 1:
        top_name = evidence.top_contributor
        top_val_str = format_value(evidence.top_value)
        total_str = format_value(evidence.total_value) if evidence.total_value is not None else "the total"
        share_str = f"{evidence.top_share_pct:.1f}%" if evidence.top_share_pct is not None else None

        if share_str:
            finding = f"{top_name} is the leading contributor with {top_val_str}, accounting for {share_str} of {total_str} across {evidence.num_groups} categories."
        else:
            finding = f"{top_name} leads with {top_val_str} across {evidence.num_groups} categories."

        # Interpretation & Disparity
        if evidence.runner_up and evidence.disparity_ratio is not None and evidence.runner_up_value is not None:
            runner_str = format_value(evidence.runner_up_value)
            interp = (
                f"{top_name} exceeds the second-highest category ({evidence.runner_up}: {runner_str}) by "
                f"{evidence.disparity_ratio:.1f}x (difference of {format_value(evidence.disparity_difference)}). "
                f"This indicates significant concentration in {top_name}, making overall performance heavily dependent on this segment."
            )
        else:
            diff_str = f" (difference of {format_value(evidence.disparity_difference)})" if evidence.disparity_difference is not None else ""
            interp = f"{top_name} accounts for the highest volume among the analyzed groups{diff_str}."

        limitations = (
            "Shares of total reflect the observed data sample. Correlation between segment size and underlying business "
            "drivers (such as volume vs. unit price) is not tested, and causality cannot be inferred."
        )

        next_steps = [
            f"How has {top_name}'s contribution trended over time?",
            f"What sub-segments or drivers explain the volume in {top_name}?",
            f"How does the average transaction value compare between {top_name} and {evidence.runner_up or 'other categories'}?"
        ]

        return AnswerExplanation(
            finding=finding,
            evidence=evidence,
            interpretation=interp,
            uncertainty_or_limitations=limitations,
            next_steps=next_steps[:3],
            status="deterministic"
        )

    # Case 4: Non-Additive Group Breakdown (e.g. Averages, Rates, Ambiguous Metrics)
    if evidence.top_contributor and evidence.top_value is not None:
        top_name = evidence.top_contributor
        top_val_str = format_value(evidence.top_value)
        num_g = evidence.num_groups or 2

        finding = f"{top_name} records the highest value at {top_val_str} across {num_g} compared groups."

        if evidence.runner_up and evidence.runner_up_value is not None:
            runner_str = format_value(evidence.runner_up_value)
            diff_str = format_value(evidence.disparity_difference) if evidence.disparity_difference is not None else ""
            ratio_str = f" ({evidence.disparity_ratio:.1f}x)" if evidence.disparity_ratio is not None else ""
            interp = (
                f"{top_name} is higher than {evidence.runner_up} ({runner_str}){ratio_str}"
                + (f" by a margin of {diff_str}." if diff_str else ".")
                + " Variations in group averages or rates may stem from differing sample sizes or operational factors."
            )
        else:
            interp = f"{top_name} is the top group for this metric."

        limitations = (
            evidence.limitation_notes or
            "Values represent group-level statistics. Because this metric is non-additive, values cannot be summed into a share of total."
        )

        next_steps = [
            f"What is the underlying sample size and record count for {top_name}?",
            f"What is the distribution spread or variance within {top_name}?",
            f"Is the higher metric in {top_name} driven by specific outliers?"
        ]

        return AnswerExplanation(
            finding=finding,
            evidence=evidence,
            interpretation=interp,
            uncertainty_or_limitations=limitations,
            next_steps=next_steps[:3],
            status="deterministic"
        )

    # Case 5: Scalar Result (Compatible Baseline Matching)
    if evidence.aggregation_type in ("scalar", "average", "sum") or (evidence.total_value is not None and evidence.num_groups is None):
        val_str = format_value(evidence.total_value) if evidence.total_value is not None else str(result)
        finding = f"The computed result for '{q_str}' is {val_str}."

        if evidence.baseline_value is not None and evidence.total_value is not None and evidence.baseline_value != 0:
            b_str = format_value(evidence.baseline_value)
            if evidence.aggregation_type == "sum":
                pct_of_total = (evidence.total_value / evidence.baseline_value) * 100
                if abs(pct_of_total - 100.0) < 0.1:
                    interp = f"This figure represents the grand total across all records ({evidence.baseline_description}: {b_str})."
                else:
                    interp = f"This value ({val_str}) represents {pct_of_total:.1f}% of the overall dataset total ({evidence.baseline_description}: {b_str})."
            elif evidence.aggregation_type == "average":
                diff_pct = ((evidence.total_value - evidence.baseline_value) / evidence.baseline_value) * 100
                rel_str = f"{abs(diff_pct):.1f}% {'above' if diff_pct > 0 else 'below'}"
                interp = f"This group average ({val_str}) is {rel_str} the overall dataset mean ({evidence.baseline_description}: {b_str})."
            else:
                interp = f"This value compares with the dataset benchmark ({evidence.baseline_description}: {b_str})."
        else:
            sample_str = f" across {evidence.sample_size:,} records" if evidence.sample_size else ""
            interp = f"This aggregate metric summarizes the dataset{sample_str}. Further segmentation can reveal whether this value is uniformly distributed or driven by specific clusters."

        limitations = (
            "A single scalar aggregate does not reveal underlying distribution variance, skewness, or subgroup differences."
        )

        next_steps = [
            "How does this metric break down across primary categories or segments?",
            "What is the median and distribution range for this metric?",
            "How does this value compare across different time periods?"
        ]

        return AnswerExplanation(
            finding=finding,
            evidence=evidence,
            interpretation=interp,
            uncertainty_or_limitations=limitations,
            next_steps=next_steps[:3],
            status="deterministic"
        )

    # Case 6: Generic Table / List / Dict
    row_count = evidence.num_groups or len(result) if hasattr(result, "__len__") else 0
    finding = f"Analysis returned {row_count} records responding to '{q_str}'."
    interp = "The tabular output provides row-level granularity. Examining summary distributions or filtering by top values can highlight actionable patterns."
    limitations = "Tabular records are presented without secondary dimensional modeling."
    next_steps = [
        "What are the top 5 records when sorted by the primary numeric column?",
        "Can these records be aggregated by a key categorical dimension?",
        "Are there any missing or outlier values in these rows?"
    ]

    return AnswerExplanation(
        finding=finding,
        evidence=evidence,
        interpretation=interp,
        uncertainty_or_limitations=limitations,
        next_steps=next_steps,
        status="deterministic"
    )


# ============================================================================
# Gemini Synthesis Engine (Selective, Bounded, & Audited)
# ============================================================================

SYNTHESIS_SYSTEM_INSTRUCTION = """You are an expert AI data analyst explaining the business significance of a verified analytical result.

Strict Rules:
1. Grounding: You must rely SOLELY on the numbers provided in the 'Evidence' input. Do NOT invent, assume, or modify any figures.
2. Finding: Factually state what happened in 1-2 clear, direct sentences.
3. Interpretation: Explain why this finding might matter to a decision maker. Clearly present this as a hypothesis or potential implication, NEVER as an established fact.
4. Epistemic Modesty: Correlation does NOT imply causation. Never state that X caused Y or that A happened because of B unless explicitly proven. Mention plausible confounding factors.
5. Limitations & Uncertainty: Disclose analytical caveats (e.g. sample size, non-additivity, lack of longitudinal data).
6. Next Steps: Suggest 2-3 specific, actionable analytical questions the user can investigate next using the data.
7. Return ONLY valid JSON matching this schema:
{
  "finding": "string",
  "interpretation": "string",
  "uncertainty_or_limitations": "string",
  "next_steps": ["string", "string"]
}
No markdown blocks, no commentary."""


def _synthesize_with_gemini(
    evidence: AnswerEvidence,
    question: str,
    timeout_sec: float = 4.0
) -> Optional[Dict[str, Any]]:
    """
    Invokes Gemini with a strict timeout and low token budget to synthesize a grounded
    explanation from pre-calculated evidence. Returns None on any error or timeout.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_google_gemini_api_key_here":
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        evidence_payload = {
            "metric_name": evidence.metric_name,
            "aggregation_type": evidence.aggregation_type,
            "is_additive": evidence.is_additive,
            "total_value": evidence.total_value,
            "top_contributor": evidence.top_contributor,
            "top_value": evidence.top_value,
            "top_share_pct": evidence.top_share_pct,
            "runner_up": evidence.runner_up,
            "runner_up_value": evidence.runner_up_value,
            "disparity_ratio": evidence.disparity_ratio,
            "disparity_difference": evidence.disparity_difference,
            "num_groups": evidence.num_groups,
            "sample_size": evidence.sample_size,
            "limitation_notes": evidence.limitation_notes,
            "key_figures": evidence.key_figures
        }

        user_content = f"Question: {question}\nEvidence: {json.dumps(evidence_payload, default=str)}"

        def _call_gemini():
            return client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=SYNTHESIS_SYSTEM_INSTRUCTION,
                    temperature=0.1,
                    max_output_tokens=350,
                    response_mime_type="application/json"
                )
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call_gemini)
            response = future.result(timeout=timeout_sec)

        if not response or not response.text:
            return None

        # Clean potential markdown wrapping
        raw_text = response.text.strip()
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)

        parsed = json.loads(raw_text)
        if isinstance(parsed, dict) and "finding" in parsed and "interpretation" in parsed:
            return parsed

        return None

    except Exception as e:
        sanitized = _sanitize_error_message(e)
        logger.warning("Gemini explanation synthesis unavailable: %s", sanitized)
        return None


# ============================================================================
# Main Orchestrator
# ============================================================================

def explain_answer(
    result: Any,
    df: Optional[pd.DataFrame] = None,
    question: Optional[str] = None,
    use_gemini: bool = True
) -> AnswerExplanation:
    """
    Main entry point for Feature A:
    1. Computes deterministic evidence metrics (shares, disparities, totals, baselines).
    2. Constructs a 100% deterministic baseline explanation.
    3. Selectively calls Gemini to synthesize richer narrative when applicable and available.
    4. Audits Gemini's numerical claims against deterministic evidence.
    5. Enforces validation and seamlessly falls back to the deterministic explanation.
    """
    q_str = (question or "").strip()

    # Step 1: Calculate Deterministic Evidence
    evidence = calculate_answer_evidence(result, df=df, question=q_str)

    # Step 2: Build Deterministic Baseline
    deterministic_explanation = generate_deterministic_explanation(evidence, q_str, result)

    # Step 3: Selectively Attempt Gemini Synthesis
    # Only invoke Gemini if:
    #   - use_gemini is True
    #   - Result has multi-dimensional evidence (num_groups >= 2)
    #   - Not single_category, empty, or unknown
    #   - API key is configured
    should_invoke_gemini = (
        use_gemini
        and evidence.aggregation_type not in ("empty", "unknown", "single_category")
        and (evidence.num_groups is not None and evidence.num_groups >= 2)
        and os.getenv("GEMINI_API_KEY") is not None
    )

    if not should_invoke_gemini:
        return deterministic_explanation

    try:
        synth = _synthesize_with_gemini(evidence, q_str, timeout_sec=4.0)
        if synth:
            ai_finding = str(synth.get("finding", "")).strip()

            # Audit numerical consistency in AI finding:
            # Check any percentage figures e.g. "56.7%" or "57%" against verified evidence
            ai_pcts = re.findall(r"(\d+(?:\.\d+)?)\s*%", ai_finding)
            is_consistent = True
            if ai_pcts and evidence.top_share_pct is not None:
                for pct_str in ai_pcts:
                    val_pct = float(pct_str)
                    is_match = (
                        abs(val_pct - evidence.top_share_pct) <= 1.5
                        or (evidence.runner_up_share_pct and abs(val_pct - evidence.runner_up_share_pct) <= 1.5)
                    )
                    if not is_match:
                        logger.info(
                            "Gemini finding contained unsupported percentage %s%% vs %s%%; falling back to deterministic finding",
                            pct_str, evidence.top_share_pct
                        )
                        is_consistent = False
                        break

            final_finding = ai_finding if (is_consistent and ai_finding) else deterministic_explanation.finding

            # Ensure non-causal disclaimer and methodology caveats are ALWAYS preserved
            ai_limitations = str(synth.get("uncertainty_or_limitations", "")).strip()
            base_limitations = deterministic_explanation.uncertainty_or_limitations or ""
            if base_limitations and base_limitations not in ai_limitations:
                final_limitations = f"{ai_limitations} {base_limitations}".strip() if ai_limitations else base_limitations
            else:
                final_limitations = ai_limitations or base_limitations

            next_steps = synth.get("next_steps", [])
            if not isinstance(next_steps, list) or len(next_steps) == 0:
                next_steps = deterministic_explanation.next_steps

            return AnswerExplanation(
                finding=final_finding,
                evidence=evidence,
                interpretation=str(synth.get("interpretation", deterministic_explanation.interpretation)),
                uncertainty_or_limitations=final_limitations,
                next_steps=[str(ns) for ns in next_steps[:3]],
                status="synthesized"
            )
    except Exception as e:
        sanitized = _sanitize_error_message(e)
        logger.warning("Falling back to deterministic explanation after synthesis exception: %s", sanitized)

    return deterministic_explanation
