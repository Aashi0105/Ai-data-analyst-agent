"""
insights.py
-----------
Purpose:
    Deterministic Automatic Insight Discovery engine for tabular DataFrames.
    Automatically detects, calculates, and ranks the 3-5 most meaningful,
    evidence-backed insights before the user asks a question.

Core Principles:
    - 100% Deterministic: All numerical and statistical values are calculated via
      pandas and numpy. Zero hallucination or LLM recalculation.
    - Rigorous Evidence: Every insight contains structured metrics, key figures,
      sample size, and comparison baselines.
    - Epistemic Modesty: Clear uncertainty and limitation disclosures (e.g. correlation
      vs causation, statistical outliers vs fraud).
    - Diverse & Ranked: Prioritizes segmentation, trends, IQR outliers, correlations,
      and critical data quality issues without redundancy.
    - Defensive: Handles small datasets, empty frames, infinities, all-null columns,
      and constant columns safely.
"""

import math
import uuid
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from profiler import DatasetProfile, ColumnProfile


# ============================================================================
# Schema Models
# ============================================================================

class VisualizationData(BaseModel):
    chart_type: str  # "Bar Chart" | "Line Chart" | "Scatter Plot" | "Pie Chart" | "Histogram"
    data: List[Dict[str, Any]]
    x_key: str
    y_keys: List[str]
    title: str


class InsightEvidence(BaseModel):
    metric_name: str
    relevant_columns: List[str]
    key_figures: Dict[str, Any]
    sample_size: Optional[int] = None
    comparison_basis: Optional[str] = None


class Insight(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str  # "segmentation" | "trend" | "outlier" | "correlation" | "data_quality"
    headline: str
    explanation: str
    why_it_matters: str
    uncertainty_or_limitations: Optional[str] = None
    evidence: InsightEvidence
    visualization: Optional[VisualizationData] = None
    suggested_query: Optional[str] = None
    score: float = 0.0


# ============================================================================
# Helper Functions
# ============================================================================

def _clean_num(val: Any, decimals: int = 2) -> Optional[float]:
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


def _format_metric_val(val: float) -> str:
    """Formats numeric values into compact, human-readable strings (e.g. 1.2M, 4.5k, 42.50)."""
    if abs(val) >= 1_000_000:
        return f"{val / 1_000_000:.2f}M"
    if abs(val) >= 1_000:
        return f"{val / 1_000:.1f}k"
    return f"{val:.2f}" if isinstance(val, float) else f"{val}"


def _is_sensitive_or_id_column(col_name: str) -> bool:
    """
    Identifies PII-bearing or identifier columns that should not be used as
    segmentation dimensions, correlation candidates, or outlier targets.
    """
    clean = col_name.strip().lower().replace("-", "_")
    sensitive_exact = {
        "name", "first_name", "firstname", "last_name", "lastname", "full_name", "fullname",
        "customer_name", "client_name", "patient_name", "employee_name", "user_name", "username",
        "email", "e_mail", "phone", "telephone", "mobile", "cell", "contact",
        "ssn", "social_security", "aadhaar", "adhaar", "pan", "passport", "tax_id", "national_id",
        "card", "card_number", "credit_card", "debit_card", "cvv", "bank_account", "iban",
        "address", "street", "postal_code", "zipcode", "zip_code",
        "password", "secret", "token", "api_key", "credential",
    }
    if clean in sensitive_exact:
        return True

    id_terms = ["id", "uuid", "token", "hash", "email", "phone", "ssn", "password", "secret", "card", "zip", "postal"]
    if any(term in clean for term in id_terms):
        return True

    person_terms = ["customer", "client", "user", "employee", "patient", "person", "member", "author"]
    if any(p in clean for p in person_terms) and any(n in clean for n in ["name", "first", "last"]):
        return True

    return False


# ============================================================================
# 1. Highest & Lowest Performing Segments
# ============================================================================

def _analyze_segments(df: pd.DataFrame, profile: DatasetProfile) -> List[Insight]:
    """
    Identifies dominant or lagging categorical segments for appropriate numeric measures.
    Uses sums for volumetric columns (revenue, sales, quantity) and means for rates/ratings/prices.
    """
    insights: List[Insight] = []
    num_rows = len(df)
    if num_rows < 5:
        return insights

    cat_cols = [
        col for col, p in profile.column_profiles.items()
        if p.semantic_type in ("categorical", "boolean")
        and p.categorical_stats
        and 2 <= p.categorical_stats.unique_count <= 20
        and not _is_sensitive_or_id_column(col)
    ]

    num_cols = [
        col for col, p in profile.column_profiles.items()
        if p.semantic_type == "numeric"
        and p.numeric_stats
        and p.numeric_stats.count >= 5
        and not _is_sensitive_or_id_column(col)
        and "year" not in col.lower()
    ]

    for cat_col in cat_cols[:4]:
        for num_col in num_cols[:4]:
            try:
                # Decide aggregation method
                col_lower = num_col.lower()
                is_rate_or_average = any(
                    term in col_lower for term in [
                        "rating", "score", "rate", "pct", "percent", "ratio", "age",
                        "unit_price", "discount", "margin", "share", "prob", "probability"
                    ]
                )
                agg_func = "mean" if is_rate_or_average else "sum"

                clean_df = df[[cat_col, num_col]].dropna()
                if len(clean_df) < 5:
                    continue

                grouped = clean_df.groupby(cat_col)[num_col].agg(agg_func)
                # Filter out empty or non-finite values
                grouped = grouped.replace([np.inf, -np.inf], np.nan).dropna()
                if len(grouped) < 2:
                    continue

                sorted_grouped = grouped.sort_values(ascending=False)
                top_cat = str(sorted_grouped.index[0])
                top_val = float(sorted_grouped.iloc[0])
                bottom_cat = str(sorted_grouped.index[-1])
                bottom_val = float(sorted_grouped.iloc[-1])
                mean_val = float(sorted_grouped.mean())
                total_val = float(sorted_grouped.sum()) if agg_func == "sum" else None

                if top_val <= 0 and bottom_val <= 0:
                    continue

                # Check concentration or disparity
                ratio_vs_mean = (top_val / mean_val) if mean_val > 0 else 1.0
                all_non_negative = bool((sorted_grouped >= 0).all())
                pct_of_total = (top_val / total_val * 100) if (total_val and total_val > 0 and all_non_negative) else None

                # Must show meaningful variance (e.g. top is >= 1.3x mean or >= 30% of total)
                if (pct_of_total and pct_of_total >= 30.0 and len(sorted_grouped) >= 3) or (ratio_vs_mean >= 1.4):
                    measure_label = "total" if agg_func == "sum" else "average"
                    metric_str = _format_metric_val(top_val)
                    bottom_str = _format_metric_val(bottom_val)

                    if pct_of_total:
                        headline = f"'{top_cat}' leads {cat_col} with {pct_of_total:.1f}% of {measure_label} {num_col}"
                        explanation = (
                            f"'{top_cat}' generated {metric_str} in {measure_label} {num_col}, "
                            f"accounting for {pct_of_total:.1f}% across all {len(sorted_grouped)} {cat_col} categories. "
                            f"In contrast, '{bottom_cat}' recorded the lowest at {bottom_str}."
                        )
                    else:
                        headline = f"'{top_cat}' exhibits highest {measure_label} {num_col} ({metric_str})"
                        explanation = (
                            f"'{top_cat}' has an {measure_label} {num_col} of {metric_str}, which is "
                            f"{ratio_vs_mean:.1f}× the group benchmark ({_format_metric_val(mean_val)}). "
                            f"'{bottom_cat}' recorded {bottom_str}."
                        )

                    why_it_matters = (
                        f"Reveals high performance concentration in {cat_col}. "
                        f"Prioritizing or investigating '{top_cat}' offers the highest analytical leverage."
                    )

                    evidence = InsightEvidence(
                        metric_name=f"{measure_label.capitalize()} {num_col}",
                        relevant_columns=[cat_col, num_col],
                        key_figures={
                            "top_category": top_cat,
                            "top_value": _clean_num(top_val),
                            "bottom_category": bottom_cat,
                            "bottom_value": _clean_num(bottom_val),
                            "group_mean": _clean_num(mean_val),
                            **( {"share_of_total_pct": _clean_num(pct_of_total)} if pct_of_total else {} ),
                        },
                        sample_size=len(clean_df),
                        comparison_basis=f"Comparison across {len(sorted_grouped)} categories in {cat_col}",
                    )

                    # Build horizontal bar chart data
                    chart_data = [
                        {"category": str(k)[:20], "value": _clean_num(v)}
                        for k, v in sorted_grouped.head(7).items()
                    ]
                    vis = VisualizationData(
                        chart_type="Bar Chart",
                        data=chart_data,
                        x_key="category",
                        y_keys=["value"],
                        title=f"{measure_label.capitalize()} {num_col} by {cat_col}",
                    )

                    score = 75.0 + min((pct_of_total or (ratio_vs_mean * 20)), 20.0)
                    insights.append(
                        Insight(
                            type="segmentation",
                            headline=headline,
                            explanation=explanation,
                            why_it_matters=why_it_matters,
                            uncertainty_or_limitations=f"Aggregated on available records (N={len(clean_df)}). Unrecorded categories are excluded.",
                            evidence=evidence,
                            visualization=vis,
                            suggested_query=f"What is the breakdown of {num_col} by {cat_col}?",
                            score=score,
                        )
                    )
            except Exception:
                continue

    return insights


# ============================================================================
# 2. Trends Over Time
# ============================================================================

def _analyze_trends(df: pd.DataFrame, profile: DatasetProfile) -> List[Insight]:
    """
    Identifies clear temporal trajectories across reliable datetime columns.
    Requires at least 4 temporal periods to establish a meaningful trajectory.
    """
    insights: List[Insight] = []
    if len(df) < 8:
        return insights

    date_cols = [
        col for col, p in profile.column_profiles.items()
        if p.semantic_type == "datetime"
    ]
    num_cols = [
        col for col, p in profile.column_profiles.items()
        if p.semantic_type == "numeric"
        and p.numeric_stats
        and p.numeric_stats.count >= 8
        and not _is_sensitive_or_id_column(col)
        and "year" not in col.lower()
    ]

    for date_col in date_cols[:2]:
        for num_col in num_cols[:3]:
            try:
                temp_df = df[[date_col, num_col]].dropna().copy()
                if len(temp_df) < 8:
                    continue

                parsed_dates = pd.to_datetime(temp_df[date_col], errors="coerce", format="mixed")
                temp_df["__dt__"] = parsed_dates
                temp_df = temp_df.dropna(subset=["__dt__"]).sort_values("__dt__")
                if len(temp_df) < 8:
                    continue

                min_date = temp_df["__dt__"].min()
                max_date = temp_df["__dt__"].max()
                day_span = (max_date - min_date).days
                if day_span < 3:
                    continue

                # Choose temporal grouping interval
                if day_span > 365 * 2:
                    freq = "QE"  # quarterly
                    date_format = "%Y-Q%q"
                elif day_span > 90:
                    freq = "ME"  # monthly
                    date_format = "%Y-%m"
                elif day_span > 14:
                    freq = "W-MON"  # weekly
                    date_format = "%Y-%m-%d"
                else:
                    freq = "D"  # daily
                    date_format = "%Y-%m-%d"

                col_lower = num_col.lower()
                is_rate = any(
                    term in col_lower for term in [
                        "rating", "score", "rate", "pct", "percent", "ratio", "age",
                        "unit_price", "discount", "margin", "share", "prob", "probability"
                    ]
                )
                agg_type = "mean" if is_rate else "sum"

                resampled = (
                    temp_df.set_index("__dt__")[num_col]
                    .resample(freq)
                    .agg(agg_type)
                    .dropna()
                )

                if len(resampled) < 4:
                    continue

                # Measure net progression from start to finish
                start_val = float(resampled.iloc[0])
                end_val = float(resampled.iloc[-1])
                # Temporal percentage change is only statistically well-defined for strictly positive baselines
                if start_val <= 0 or end_val <= 0:
                    continue

                pct_change = ((end_val - start_val) / start_val) * 100.0

                # Statistical Integrity: Verify directional trajectory consistency across intervals
                time_steps = np.arange(len(resampled))
                series_vals = resampled.values.astype(float)
                if np.std(series_vals) == 0:
                    continue

                corr_matrix = np.corrcoef(time_steps, series_vals)
                trend_r = float(corr_matrix[0, 1])
                if math.isnan(trend_r) or abs(trend_r) < 0.50:
                    continue

                # Direction of the linear trend must match the net endpoint change
                if (pct_change > 0 and trend_r < 0) or (pct_change < 0 and trend_r > 0):
                    continue

                # Check if change is noticeable (>= 12% change)
                if abs(pct_change) >= 12.0:
                    direction = "growth" if pct_change > 0 else "decline"
                    trend_adj = "increased" if pct_change > 0 else "contracted"
                    verb = "surged" if abs(pct_change) > 40 else "moved"

                    start_period = resampled.index[0].strftime("%b %Y" if day_span > 60 else "%b %d")
                    end_period = resampled.index[-1].strftime("%b %Y" if day_span > 60 else "%b %d")

                    headline = f"{num_col} {direction} of {abs(pct_change):.1f}% from {start_period} to {end_period}"
                    explanation = (
                        f"{num_col} {trend_adj} from {_format_metric_val(start_val)} to {_format_metric_val(end_val)} "
                        f"over {len(resampled)} observed intervals (trend r = {trend_r:.2f}), representing a net {pct_change:+.1f}% shift."
                    )
                    why_it_matters = (
                        f"Establishes historical velocity for {num_col}. "
                        f"Understanding this trajectory helps validate growth drivers or address contractions."
                    )

                    evidence = InsightEvidence(
                        metric_name=f"{num_col} Net Trend",
                        relevant_columns=[date_col, num_col],
                        key_figures={
                            "start_period": start_period,
                            "start_value": _clean_num(start_val),
                            "end_period": end_period,
                            "end_value": _clean_num(end_val),
                            "net_pct_change": _clean_num(pct_change),
                            "trend_correlation": _clean_num(trend_r, 2),
                            "time_buckets": len(resampled),
                        },
                        sample_size=len(temp_df),
                        comparison_basis=f"Resampled by {freq} over {day_span} days",
                    )

                    chart_data = [
                        {
                            "date": idx.strftime("%Y-%m-%d" if day_span <= 30 else "%Y-%m"),
                            "value": _clean_num(v),
                        }
                        for idx, v in resampled.items()
                    ]

                    vis = VisualizationData(
                        chart_type="Line Chart",
                        data=chart_data,
                        x_key="date",
                        y_keys=["value"],
                        title=f"{num_col} Trend ({date_col})",
                    )

                    score = 75.0 + (abs(trend_r) * 15.0) + min(abs(pct_change) / 5.0, 10.0)
                    insights.append(
                        Insight(
                            type="trend",
                            headline=headline,
                            explanation=explanation,
                            why_it_matters=why_it_matters,
                            uncertainty_or_limitations=(
                                f"Observed across {len(resampled)} aggregate periods. "
                                f"Does not account for cyclical or seasonal fluctuations outside this observation window."
                            ),
                            evidence=evidence,
                            visualization=vis,
                            suggested_query=f"How does {num_col} trend over {date_col}?",
                            score=score,
                        )
                    )
            except Exception:
                continue

    return insights


# ============================================================================
# 3. Outliers & Unusual Distributions (Tukey IQR Rule)
# ============================================================================

def _analyze_outliers(df: pd.DataFrame, profile: DatasetProfile) -> List[Insight]:
    """
    Identifies numerical outliers using Tukey's robust Interquartile Range (IQR) rule:
    Outliers = Values < Q25 - 1.5 * IQR or > Q75 + 1.5 * IQR.
    """
    insights: List[Insight] = []
    num_rows = len(df)
    if num_rows < 15:
        return insights

    num_cols = [
        col for col, p in profile.column_profiles.items()
        if p.semantic_type == "numeric"
        and p.numeric_stats
        and p.numeric_stats.count >= 15
        and not _is_sensitive_or_id_column(col)
        and "year" not in col.lower()
    ]

    for col in num_cols[:4]:
        try:
            series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(series) < 15:
                continue

            q25 = float(series.quantile(0.25))
            q75 = float(series.quantile(0.75))
            iqr = q75 - q25

            # If IQR is 0, dataset has low variance / discrete clusters; skip to avoid false positives
            if iqr <= 0:
                continue

            lower_bound = q25 - (1.5 * iqr)
            upper_bound = q75 + (1.5 * iqr)

            upper_outliers = series[series > upper_bound]
            lower_outliers = series[series < lower_bound]
            outliers_count = len(upper_outliers) + len(lower_outliers)
            outlier_pct = (outliers_count / len(series)) * 100.0

            # Meaningful outlier threshold: between 1 and 15% of records
            if 1 <= outliers_count and outlier_pct <= 15.0:
                max_val = float(series.max())
                median_val = float(series.median())

                if len(upper_outliers) > 0 and len(lower_outliers) == 0:
                    boundary_desc = f"exceeding the upper boundary of {upper_bound:.2f}"
                elif len(lower_outliers) > 0 and len(upper_outliers) == 0:
                    boundary_desc = f"falling below the lower boundary of {lower_bound:.2f}"
                else:
                    boundary_desc = f"falling outside the boundaries [{lower_bound:.2f}, {upper_bound:.2f}]"

                headline = f"{outliers_count} extreme statistical outliers detected in '{col}'"
                explanation = (
                    f"'{col}' contains {outliers_count} record{'s' if outliers_count > 1 else ''} "
                    f"({outlier_pct:.1f}% of values) {boundary_desc} "
                    f"(median: {median_val:.2f}, peak outlier: {max_val:.2f})."
                )
                why_it_matters = (
                    f"Extreme values can heavily distort aggregate sums, means, and standard deviations. "
                    f"Filtering or segmenting these observations is recommended for baseline reporting."
                )

                evidence = InsightEvidence(
                    metric_name=f"{col} Outliers (IQR)",
                    relevant_columns=[col],
                    key_figures={
                        "outlier_count": outliers_count,
                        "outlier_pct": _clean_num(outlier_pct, 1),
                        "upper_threshold": _clean_num(upper_bound),
                        "lower_threshold": _clean_num(lower_bound),
                        "peak_outlier": _clean_num(max_val),
                        "median": _clean_num(median_val),
                        "iqr": _clean_num(iqr),
                    },
                    sample_size=len(series),
                    comparison_basis="Tukey 1.5x IQR method (Q75 + 1.5*IQR)",
                )

                # Histogram / Distribution visualization
                hist, bin_edges = np.histogram(series, bins=min(len(series), 10))
                chart_data = []
                for i in range(len(hist)):
                    label = f"{_format_metric_val(bin_edges[i])}-{_format_metric_val(bin_edges[i+1])}"
                    chart_data.append({"bucket": label, "count": int(hist[i])})

                vis = VisualizationData(
                    chart_type="Bar Chart",
                    data=chart_data,
                    x_key="bucket",
                    y_keys=["count"],
                    title=f"Distribution & Outlier Range of {col}",
                )

                score = 70.0 + min(outliers_count * 2.0, 20.0)
                insights.append(
                    Insight(
                        type="outlier",
                        headline=headline,
                        explanation=explanation,
                        why_it_matters=why_it_matters,
                        uncertainty_or_limitations=(
                            "Statistical outliers reflect distribution extremes and do not automatically "
                            "imply data corruption, operational failure, or fraudulent entries."
                        ),
                        evidence=evidence,
                        visualization=vis,
                        suggested_query=f"What are the top 5 records with the highest {col}?",
                        score=score,
                    )
                )
        except Exception:
            continue

    return insights


# ============================================================================
# 4. Pearson Correlations
# ============================================================================

def _analyze_correlations(df: pd.DataFrame, profile: DatasetProfile) -> List[Insight]:
    """
    Computes Pearson correlation coefficients across relevant numeric pairs.
    Identifies strong positive or negative associations (|r| >= 0.50).
    """
    insights: List[Insight] = []
    num_rows = len(df)
    if num_rows < 10:
        return insights

    num_cols = [
        col for col, p in profile.column_profiles.items()
        if p.semantic_type == "numeric"
        and p.numeric_stats
        and p.numeric_stats.count >= 10
        and not _is_sensitive_or_id_column(col)
        and "year" not in col.lower()
    ]

    # Bound pairwise computation to first 12 columns to maintain bounded upload latency
    eval_cols = num_cols[:12]
    if len(eval_cols) < 2:
        return insights

    try:
        clean_df = df[eval_cols].replace([np.inf, -np.inf], np.nan)
        corr_matrix = clean_df.corr(method="pearson")

        seen_pairs = set()
        for i, col1 in enumerate(eval_cols):
            for j, col2 in enumerate(eval_cols):
                if i >= j:
                    continue

                pair_key = tuple(sorted([col1, col2]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                r_val = corr_matrix.loc[col1, col2]
                if pd.isna(r_val) or math.isnan(r_val):
                    continue

                abs_r = abs(float(r_val))
                if abs_r >= 0.50:  # Moderate to strong correlation
                    pair_df = clean_df[[col1, col2]].dropna()
                    if len(pair_df) < 10:
                        continue

                    strength = "strong" if abs_r >= 0.70 else "moderate"
                    direction = "positive" if r_val > 0 else "inverse"

                    headline = f"{strength.capitalize()} {direction} correlation (r = {r_val:.2f}) between {col1} and {col2}"
                    if r_val > 0:
                        explanation = (
                            f"'{col1}' and '{col2}' exhibit a {strength} positive relationship (Pearson r = {r_val:.2f}, N={len(pair_df)}). "
                            f"As {col1} rises, {col2} systematically tends to increase."
                        )
                    else:
                        explanation = (
                            f"'{col1}' and '{col2}' exhibit an inverse relationship (Pearson r = {r_val:.2f}, N={len(pair_df)}). "
                            f"Increases in {col1} coincide with systematic decreases in {col2}."
                        )

                    why_it_matters = (
                        f"Reveals high co-movement between {col1} and {col2}. "
                        f"Useful for predictive modeling, regression, and joint strategic planning."
                    )

                    evidence = InsightEvidence(
                        metric_name="Pearson Correlation Coefficient",
                        relevant_columns=[col1, col2],
                        key_figures={
                            "r_coefficient": _clean_num(r_val, 3),
                            "r_squared": _clean_num(r_val ** 2, 3),
                            "strength": strength,
                            "direction": direction,
                        },
                        sample_size=len(pair_df),
                        comparison_basis="Bivariate Pearson correlation across non-null finite rows",
                    )

                    # Subsample up to 40 data points for responsive scatter plot
                    step = max(1, len(pair_df) // 40)
                    scatter_samples = pair_df.iloc[::step].head(40)
                    chart_data = [
                        {"x": _clean_num(row[col1]), "y": _clean_num(row[col2])}
                        for _, row in scatter_samples.iterrows()
                        if _clean_num(row[col1]) is not None and _clean_num(row[col2]) is not None
                    ]

                    vis = VisualizationData(
                        chart_type="Scatter Plot",
                        data=chart_data,
                        x_key=col1,
                        y_keys=[col2],
                        title=f"{col2} vs {col1} (r = {r_val:.2f})",
                    )

                    score = 80.0 + (abs_r * 15.0)
                    insights.append(
                        Insight(
                            type="correlation",
                            headline=headline,
                            explanation=explanation,
                            why_it_matters=why_it_matters,
                            uncertainty_or_limitations=(
                                "Correlation establishes statistical association, NOT causation. "
                                "Unobserved confounding variables may drive the relationship."
                            ),
                            evidence=evidence,
                            visualization=vis,
                            suggested_query=f"What is the correlation between {col1} and {col2}?",
                            score=score,
                        )
                    )
    except Exception:
        pass

    return insights


# ============================================================================
# 5. Data Quality Problems
# ============================================================================

def _analyze_data_quality(df: pd.DataFrame, profile: DatasetProfile) -> List[Insight]:
    """
    Identifies high-impact data hygiene issues that could distort analysis.
    Only surfaced if critical threshold is met (e.g. duplicate rows >= 5%, high missingness).
    """
    insights: List[Insight] = []
    num_rows = len(df)
    if num_rows == 0:
        return insights

    # Check 1: Significant duplicate rows
    if profile.duplicate_rows > 0 and profile.duplicate_row_pct >= 5.0:
        headline = f"Dataset contains {profile.duplicate_rows} duplicate rows ({profile.duplicate_row_pct}% of total records)"
        explanation = (
            f"There are {profile.duplicate_rows} identical row entries out of {num_rows:,} records. "
            f"This represents complete multi-column duplication."
        )
        why_it_matters = (
            "Duplicate rows artificially inflate sums, counts, and statistical significance. "
            "Deduplication is advised before performing aggregate business reporting."
        )
        evidence = InsightEvidence(
            metric_name="Complete Duplicate Rows",
            relevant_columns=profile.columns[:5],
            key_figures={
                "duplicate_count": profile.duplicate_rows,
                "duplicate_pct": _clean_num(profile.duplicate_row_pct, 1),
                "total_rows": num_rows,
            },
            sample_size=num_rows,
            comparison_basis="Exact row-wise equality (df.duplicated())",
        )
        insights.append(
            Insight(
                type="data_quality",
                headline=headline,
                explanation=explanation,
                why_it_matters=why_it_matters,
                uncertainty_or_limitations="Evaluated strictly on identical column match.",
                evidence=evidence,
                visualization=None,
                suggested_query="How many duplicate rows are in this dataset?",
                score=85.0,  # High priority to inform analysts early
            )
        )

    # Check 2: Severe column-level missingness (>= 35%)
    for col, p in profile.column_profiles.items():
        if p.missing_pct >= 35.0:
            headline = f"Column '{col}' is missing {p.missing_pct}% of its entries ({p.missing_count:,} nulls)"
            explanation = (
                f"Column '{col}' has {p.missing_count:,} unpopulated cells out of {num_rows:,} records, "
                f"yielding an effective completeness rate of only {100.0 - p.missing_pct:.1f}%."
            )
            why_it_matters = (
                "High missingness restricts the reliability of sub-group analyses and can introduce "
                "survivorship or reporting bias if values are missing non-randomly."
            )
            evidence = InsightEvidence(
                metric_name=f"Missing Ratio in {col}",
                relevant_columns=[col],
                key_figures={
                    "missing_count": p.missing_count,
                    "missing_pct": _clean_num(p.missing_pct, 1),
                    "valid_count": num_rows - p.missing_count,
                },
                sample_size=num_rows,
                comparison_basis="Null cell evaluation (df[col].isnull())",
            )
            insights.append(
                Insight(
                    type="data_quality",
                    headline=headline,
                    explanation=explanation,
                    why_it_matters=why_it_matters,
                    uncertainty_or_limitations="Assumes unrecorded cells reflect missing data rather than implicit zero/false values.",
                    evidence=evidence,
                    visualization=None,
                    suggested_query=f"Which columns have the highest number of missing values?",
                    score=72.0,
                )
            )

    return insights


# ============================================================================
# Main Insight Discovery Orchestrator & Ranking
# ============================================================================

def discover_insights(df: pd.DataFrame, profile: DatasetProfile) -> List[Insight]:
    """
    Orchestrates deterministic insight discovery, rankings, and diversity filtering.

    Guarantees:
    - Returns between 3 and 5 insights when data supports it.
    - Returns fewer findings for small, empty, or uninformative datasets.
    - Balances insight diversity (segments, trends, outliers, correlations, quality).
    - Prevents duplicate findings or conflicting claims.
    """
    if len(df) == 0 or len(df.columns) == 0:
        return []

    candidate_insights: List[Insight] = []

    # 1. Harvest candidates across all categories
    candidate_insights.extend(_analyze_segments(df, profile))
    candidate_insights.extend(_analyze_trends(df, profile))
    candidate_insights.extend(_analyze_outliers(df, profile))
    candidate_insights.extend(_analyze_correlations(df, profile))
    candidate_insights.extend(_analyze_data_quality(df, profile))

    if not candidate_insights:
        # Fallback for uninformative/uniform small datasets: basic metric distribution
        numeric_cols = [c for c, p in profile.column_profiles.items() if p.semantic_type == "numeric" and p.numeric_stats]
        if numeric_cols:
            col = numeric_cols[0]
            stats = profile.column_profiles[col].numeric_stats
            if stats:
                candidate_insights.append(
                    Insight(
                        type="segmentation",
                        headline=f"Summary benchmark for {col}: Mean {_format_metric_val(stats.mean or 0)}",
                        explanation=f"Values range from {stats.min} to {stats.max} across {stats.count} recorded observations.",
                        why_it_matters="Provides baseline scale and central tendency for initial analysis.",
                        uncertainty_or_limitations="Descriptive distribution summary.",
                        evidence=InsightEvidence(
                            metric_name=f"{col} Overview",
                            relevant_columns=[col],
                            key_figures={"min": stats.min, "max": stats.max, "mean": stats.mean, "median": stats.median},
                            sample_size=stats.count,
                        ),
                        visualization=None,
                        suggested_query=f"What is the average {col}?",
                        score=60.0,
                    )
                )

    # 2. Sort by ranking score descending
    candidate_insights.sort(key=lambda x: x.score, reverse=True)

    # 3. Apply Diversity Selection:
    # Ensure a rich mix of types and columns; cap at most 2 insights of the same type
    selected: List[Insight] = []
    type_counts: Dict[str, int] = {}
    seen_headlines: set = set()

    for item in candidate_insights:
        if item.headline in seen_headlines:
            continue

        c_type = item.type
        current_type_count = type_counts.get(c_type, 0)
        if current_type_count >= 2:
            continue

        selected.append(item)
        seen_headlines.add(item.headline)
        type_counts[c_type] = current_type_count + 1

        if len(selected) == 5:
            break

    # If diversity filter left open slots and more candidates exist, fill from remaining
    if len(selected) < 5:
        for item in candidate_insights:
            if item.headline not in seen_headlines:
                selected.append(item)
                seen_headlines.add(item.headline)
                if len(selected) == 5:
                    break

    # Slice to at most 5 insights
    return selected[:5]
