"""
understanding.py
----------------
Purpose:
    Generates grounded dataset explanations and 5 analytical suggested questions
    using the existing Gemini integration and deterministic profile metadata.

Guarantees:
    - Strict Grounding: Explanations are anchored to computed profile metrics.
    - Bounded Transmission: Only compact metadata and 3 PII-masked sample rows are transmitted.
    - Robust Fallback: In event of quota limit, timeout, provider error, or malformed JSON,
      returns a deterministic, high-quality fallback explanation and 5 valid questions.
    - Safe Execution: Suggestions are never executed during generation.
    - Source of Truth: Profiler values are factual; LLM cannot calculate or override them.
"""

import concurrent.futures
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional
import pandas as pd
from pydantic import BaseModel, Field

from profiler import DatasetProfile, ColumnProfile
from prompt import mask_sample_dataframe, mask_sensitive_value

logger = logging.getLogger(__name__)


# ============================================================================
# Schema Models
# ============================================================================

class ColumnInsight(BaseModel):
    column: str
    inferred_meaning: str


class DatasetUnderstanding(BaseModel):
    overview: str
    column_insights: List[ColumnInsight] = Field(default_factory=list)
    analytical_potential: str
    data_quality_notes: str
    suggested_questions: List[str] = Field(..., min_length=5, max_length=5)
    status: str = "generated"  # "generated" | "fallback"


# ============================================================================
# Deterministic Fallback Generator
# ============================================================================

def generate_deterministic_fallback(
    profile: DatasetProfile,
    reason: str = "Deterministic fallback generated based on computed dataset profile."
) -> DatasetUnderstanding:
    """
    Constructs a factual, schema-grounded explanation and 5 relevant suggested questions
    without invoking an external LLM. Used when Gemini is unavailable, timed out, or exhausted.
    """
    num_rows = profile.num_rows
    num_cols = profile.num_cols

    if num_rows == 0 or num_cols == 0:
        return DatasetUnderstanding(
            overview="The dataset is empty and contains no records or columns.",
            column_insights=[],
            analytical_potential="No analysis can be performed on an empty dataset.",
            data_quality_notes="Dataset contains zero rows or columns.",
            suggested_questions=[
                "Why is the uploaded dataset empty?",
                "What columns were expected in this dataset?",
                "Can a valid CSV with data rows be uploaded?",
                "Are there any header rows present in the raw file?",
                "What schema should this dataset have?",
            ],
            status="fallback",
        )

    # Separate columns by semantic type
    numeric_cols = [
        c for c, p in profile.column_profiles.items() if p.semantic_type == "numeric"
    ]
    categorical_cols = [
        c for c, p in profile.column_profiles.items()
        if p.semantic_type in ("categorical", "boolean", "text")
    ]
    date_cols = [
        c for c, p in profile.column_profiles.items() if p.semantic_type == "datetime"
    ]

    # 1. Plain-English Overview
    col_breakdown = []
    if numeric_cols:
        col_breakdown.append(f"{len(numeric_cols)} numerical")
    if categorical_cols:
        col_breakdown.append(f"{len(categorical_cols)} categorical")
    if date_cols:
        col_breakdown.append(f"{len(date_cols)} date/time")

    breakdown_str = ", ".join(col_breakdown) if col_breakdown else "standard tabular"
    overview = (
        f"This dataset contains {num_rows:,} records across {num_cols} columns ({breakdown_str}). "
        f"It appears to be a structured tabular dataset suitable for aggregation, descriptive analysis, "
        f"and comparative reporting."
    )

    # 2. Inferred Column Insights (Top columns)
    column_insights: List[ColumnInsight] = []
    for col in profile.columns[:6]:
        p = profile.column_profiles.get(col)
        if not p:
            continue
        if p.semantic_type == "numeric" and p.numeric_stats:
            min_v = p.numeric_stats.min
            max_v = p.numeric_stats.max
            mean_v = p.numeric_stats.mean
            column_insights.append(
                ColumnInsight(
                    column=col,
                    inferred_meaning=f"Numerical metric ranging from {min_v} to {max_v} (average: {mean_v}).",
                )
            )
        elif p.semantic_type in ("categorical", "text", "boolean") and p.categorical_stats:
            u_count = p.categorical_stats.unique_count
            top_cat = (
                f", most frequent: '{p.categorical_stats.top_categories[0].value}'"
                if p.categorical_stats.top_categories
                else ""
            )
            column_insights.append(
                ColumnInsight(
                    column=col,
                    inferred_meaning=f"Categorical attribute with {u_count} unique values{top_cat}.",
                )
            )
        elif p.semantic_type == "datetime" and p.date_stats:
            column_insights.append(
                ColumnInsight(
                    column=col,
                    inferred_meaning=f"Timestamp or calendar date spanning from {p.date_stats.min_date or 'earliest'} to {p.date_stats.max_date or 'latest'}.",
                )
            )
        else:
            column_insights.append(
                ColumnInsight(
                    column=col,
                    inferred_meaning=f"Dataset attribute with data type '{p.dtype}'.",
                )
            )

    # 3. Analytical Potential
    capabilities = []
    if numeric_cols:
        capabilities.append(f"calculating totals, averages, and distributions for {', '.join(numeric_cols[:2])}")
    if categorical_cols and numeric_cols:
        capabilities.append(f"grouping and comparing metrics by {categorical_cols[0]}")
    if date_cols and numeric_cols:
        capabilities.append(f"tracking timeline trends across {date_cols[0]}")
    if not capabilities:
        capabilities.append("inspecting column frequency distributions and record previews")

    analytical_potential = (
        f"This dataset can support {'; '.join(capabilities)}."
    )

    # 4. Data Quality Notes
    if profile.warnings:
        warning_msgs = [w.message for w in profile.warnings[:3]]
        data_quality_notes = (
            f"Quality observations: {'; '.join(warning_msgs)}. "
            f"Total missing cells: {profile.total_missing} ({profile.missing_cell_pct}%); "
            f"Duplicate rows: {profile.duplicate_rows}."
        )
    else:
        data_quality_notes = (
            f"No critical data quality warnings detected. "
            f"Total missing cells: {profile.total_missing} ({profile.missing_cell_pct}%); "
            f"Duplicate rows: {profile.duplicate_rows}."
        )

    # 5. Exactly Five Grounded Suggested Questions
    questions: List[str] = []

    # Question 1: Aggregation
    if numeric_cols:
        questions.append(f"What is the total and average {numeric_cols[0]}?")
    else:
        questions.append(f"What is the total count of records across {profile.columns[0]}?")

    # Question 2: Grouping / Comparison
    if numeric_cols and categorical_cols:
        questions.append(f"What is the breakdown of {numeric_cols[0]} by {categorical_cols[0]}?")
    elif len(numeric_cols) >= 2:
        questions.append(f"How does {numeric_cols[0]} compare to {numeric_cols[1]}?")
    elif len(categorical_cols) >= 2:
        questions.append(f"How do counts in {categorical_cols[0]} compare across {categorical_cols[1]}?")
    else:
        questions.append(f"What are the summary statistics of the dataset?")

    # Question 3: Distribution / Frequency
    if categorical_cols:
        questions.append(f"What are the top 5 most frequent values in {categorical_cols[0]}?")
    elif numeric_cols:
        questions.append(f"What is the distribution and standard deviation of {numeric_cols[0]}?")
    else:
        questions.append(f"What are the unique values in {profile.columns[0]}?")

    # Question 4: Time-based or Relationship
    if date_cols and numeric_cols:
        questions.append(f"How does {numeric_cols[0]} trend over {date_cols[0]}?")
    elif len(numeric_cols) >= 2:
        questions.append(f"What is the correlation between {numeric_cols[0]} and {numeric_cols[1]}?")
    elif len(categorical_cols) >= 2:
        questions.append(f"What are the most common combinations of {categorical_cols[0]} and {categorical_cols[1]}?")
    else:
        questions.append(f"What are the minimum and maximum values in {profile.columns[0]}?")

    # Question 5: Outliers / Data Quality / Top items
    if numeric_cols:
        questions.append(f"What are the top 5 records with the highest {numeric_cols[0]}?")
    elif profile.total_missing > 0:
        questions.append(f"Which columns have the highest number of missing values?")
    else:
        questions.append(f"What are the first 5 records in the dataset?")

    # Ensure strictly 5 unique questions
    while len(questions) < 5:
        questions.append(f"What is the overview of column {profile.columns[len(questions) % len(profile.columns)]}?")
    questions = questions[:5]

    return DatasetUnderstanding(
        overview=overview,
        column_insights=column_insights,
        analytical_potential=analytical_potential,
        data_quality_notes=data_quality_notes,
        suggested_questions=questions,
        status="fallback",
    )


# ============================================================================
# LLM Understanding Prompt Construction
# ============================================================================

def _build_understanding_prompt(df: pd.DataFrame, profile: DatasetProfile) -> tuple[str, str]:
    """
    Constructs a tightly scoped system instruction and user prompt containing
    only compact metadata and 3 PII-masked sample rows.
    """
    system_instruction = (
        "You are an expert dataset analyst. Your task is to provide an objective, grounded "
        "dataset understanding and exactly 5 suggested analytical questions based strictly "
        "on the schema and profile provided.\n\n"
        "STRICT GROUNDING RULES:\n"
        "1. Base inferences strictly on the column names, data types, and statistics provided.\n"
        "2. Treat column meanings as inferred, not established facts, unless evident from standard naming.\n"
        "3. Do NOT invent external business context, proprietary company info, or causal assumptions.\n"
        "4. Numerical statistics in the profile are ground truth; do NOT recalculate or contradict them.\n"
        "5. If suggesting time-based analysis, do so ONLY if a datetime column is explicitly present.\n"
        "6. Provide EXACTLY 5 distinct, practical questions that can be answered using pandas code on the columns.\n"
        "7. Return ONLY valid JSON matching this schema:\n"
        "{\n"
        '  "overview": "Concise plain-English description (1-2 sentences, max 250 chars)",\n'
        '  "column_insights": [\n'
        '    {"column": "column_name", "inferred_meaning": "brief inferred meaning (max 80 chars)"}\n'
        "  ],\n"
        '  "analytical_potential": "Types of analysis this dataset supports (max 200 chars)",\n'
        '  "data_quality_notes": "Key data quality observations or caveats (max 200 chars)",\n'
        '  "suggested_questions": [\n'
        '    "Question 1 (aggregation / comparison)",\n'
        '    "Question 2 (distribution / frequency)",\n'
        '    "Question 3 (relationship between columns)",\n'
        '    "Question 4 (time-based trend if date exists, otherwise comparative)",\n'
        '    "Question 5 (ranking, outlier, or quality investigation)"\n'
        "  ]\n"
        "}"
    )

    # Format compact profile summary
    profile_summary = {
        "num_rows": profile.num_rows,
        "num_cols": profile.num_cols,
        "total_missing": profile.total_missing,
        "duplicate_rows": profile.duplicate_rows,
        "columns": [
            {
                "name": col,
                "type": p.semantic_type,
                "missing_pct": p.missing_pct,
                **(
                    {"min": p.numeric_stats.min, "max": p.numeric_stats.max, "mean": p.numeric_stats.mean}
                    if p.numeric_stats else {}
                ),
                **(
                    {
                        "unique_count": p.categorical_stats.unique_count,
                        "top": [
                            mask_sensitive_value(c.value, col_name=col)
                            for c in p.categorical_stats.top_categories[:3]
                        ] if p.semantic_type != "text" else []
                    }
                    if p.categorical_stats else {}
                ),
                **(
                    {"date_range": f"{p.date_stats.min_date} to {p.date_stats.max_date}"}
                    if p.date_stats and p.date_stats.min_date else {}
                ),
            }
            for col, p in profile.column_profiles.items()
        ],
        "warnings": [w.message for w in profile.warnings],
    }

    # Isolated 3 PII-masked sample rows
    masked_sample = mask_sample_dataframe(df, max_rows=3, max_str_len=25)
    sample_text = masked_sample.to_string(index=False)

    user_prompt = (
        f"Dataset Profile:\n{json.dumps(profile_summary, indent=2)}\n\n"
        f"Sample Data (First 3 rows, PII masked):\n{sample_text}\n\n"
        "Provide the dataset understanding JSON now."
    )

    return system_instruction, user_prompt


# ============================================================================
# LLM Response Validation and Sanitization
# ============================================================================

def _sanitize_and_validate_understanding(
    raw_text: str, profile: DatasetProfile
) -> Optional[DatasetUnderstanding]:
    """
    Parses and strictly validates the Gemini JSON response.
    Returns DatasetUnderstanding model if valid, None if invalid.
    """
    try:
        # Strip potential markdown formatting (```json ... ```)
        cleaned_text = raw_text.strip()
        if cleaned_text.startswith("```"):
            cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text, flags=re.IGNORECASE)
            cleaned_text = re.sub(r"\s*```$", "", cleaned_text)
            cleaned_text = cleaned_text.strip()

        data = json.loads(cleaned_text)
        if not isinstance(data, dict):
            return None

        overview = str(data.get("overview", "")).strip()[:400]
        analytical_potential = str(data.get("analytical_potential", "")).strip()[:350]
        data_quality_notes = str(data.get("data_quality_notes", "")).strip()[:350]

        if not overview:
            return None

        # Validate column insights
        raw_insights = data.get("column_insights", [])
        insights: List[ColumnInsight] = []
        valid_cols = set(profile.columns)

        if isinstance(raw_insights, list):
            for item in raw_insights:
                if isinstance(item, dict) and "column" in item and "inferred_meaning" in item:
                    col_name = str(item["column"]).strip()
                    if col_name in valid_cols:
                        insights.append(
                            ColumnInsight(
                                column=col_name,
                                inferred_meaning=str(item["inferred_meaning"]).strip()[:150],
                            )
                        )

        # Validate suggested questions
        raw_questions = data.get("suggested_questions", [])
        if not isinstance(raw_questions, list):
            return None

        cleaned_questions: List[str] = []
        for q in raw_questions:
            if isinstance(q, str) and q.strip():
                clean_q = q.strip().replace("`", "")[:180]
                if clean_q not in cleaned_questions:
                    cleaned_questions.append(clean_q)

        # If LLM didn't return at least 4 questions, reject and use fallback
        if len(cleaned_questions) < 4:
            return None

        # Pad or slice to exactly 5 questions
        if len(cleaned_questions) < 5:
            fallback = generate_deterministic_fallback(profile)
            for fq in fallback.suggested_questions:
                if fq not in cleaned_questions:
                    cleaned_questions.append(fq)
                if len(cleaned_questions) == 5:
                    break

        final_5_questions = cleaned_questions[:5]

        return DatasetUnderstanding(
            overview=overview,
            column_insights=insights[:10],
            analytical_potential=analytical_potential or "Supports tabular aggregations and distribution checks.",
            data_quality_notes=data_quality_notes or "No critical data quality issues noted.",
            suggested_questions=final_5_questions,
            status="generated",
        )
    except Exception as e:
        logger.warning(f"Failed to validate LLM understanding output: {e}")
        return None


# ============================================================================
# Main Understanding Generator
# ============================================================================

def generate_dataset_understanding(
    df: pd.DataFrame,
    profile: DatasetProfile,
    timeout_seconds: float = 12.0,
) -> DatasetUnderstanding:
    """
    Generates a grounded dataset understanding and 5 suggested questions.
    Attempts Gemini generation with strict error and timeout bounds, falling back
    instantly to deterministic generation upon any failure.

    Args:
        df (pd.DataFrame): Target DataFrame.
        profile (DatasetProfile): Computed profile.
        timeout_seconds (float): Maximum allowed Gemini latency (default: 12.0s, Google min HTTP deadline: 10s).

    Returns:
        DatasetUnderstanding: Validated understanding with status 'generated' or 'fallback'.
    """
    # Empty dataset handling
    if profile.num_rows == 0 or profile.num_cols == 0:
        return generate_deterministic_fallback(profile)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_google_gemini_api_key_here":
        return generate_deterministic_fallback(profile, reason="Gemini API key is not configured.")

    try:
        from google import genai
        from google.genai import types

        # Google API enforces a minimum deadline of 10s (10000ms) for HttpOptions
        http_timeout_ms = max(int(timeout_seconds * 1000), 10000)
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=http_timeout_ms),
        )
        system_prompt, user_prompt = _build_understanding_prompt(df, profile)

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
            validated = _sanitize_and_validate_understanding(response.text, profile)
            if validated:
                return validated

    except Exception as e:
        logger.warning(f"Gemini understanding generation failed gracefully: {e}")

    # Fallback guarantees seamless upload and feature continuity
    return generate_deterministic_fallback(profile)
