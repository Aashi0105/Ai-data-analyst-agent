"""
backend/chart_utils.py
----------------------
Converts Pandas Series or DataFrame execution results into lightweight,
Recharts-compatible structured JSON dictionaries.
Matches the deterministic charting heuristics from charts.py.
"""

from typing import Any, Dict, Optional, List
import pandas as pd
import numpy as np

from charts import detect_chart_type, _is_datetime_series


def build_recharts_data(result: Any) -> Optional[Dict[str, Any]]:
    """
    Transforms sandbox result into a structured JSON representation
    tailored for Recharts frontend rendering.

    Returns:
        Optional[Dict[str, Any]]: {
            "chart_type": str ("Bar Chart" | "Line Chart" | "Scatter Plot" | "Histogram"),
            "data": List[Dict[str, Any]],
            "x_key": str,
            "y_keys": List[str],
            "title": str
        } or None if not chart-compatible.
    """
    chart_type = detect_chart_type(result)
    if not chart_type:
        return None

    try:
        # Case 1: Series
        if isinstance(result, pd.Series):
            s = result.dropna()
            if s.empty or not pd.api.types.is_numeric_dtype(s):
                return None

            data = []
            for idx, val in s.items():
                data.append({
                    "label": str(idx),
                    "value": round(float(val), 2) if isinstance(val, (float, np.floating)) else int(val)
                })

            return {
                "chart_type": "Bar Chart",
                "data": data[:30],  # Keep reasonable ceiling
                "x_key": "label",
                "y_keys": ["value"],
                "title": f"{s.name or 'Values'} by {s.index.name or 'Category'}"
            }

        # Case 2: DataFrame
        if isinstance(result, pd.DataFrame):
            df = result.copy()
            if df.empty:
                return None

            cols = list(df.columns)

            if chart_type == "Bar Chart":
                cat_col = cols[0] if not pd.api.types.is_numeric_dtype(df[cols[0]]) else cols[1]
                num_col = cols[1] if cat_col == cols[0] else cols[0]

                data = []
                for _, row in df.iterrows():
                    val = row[num_col]
                    clean_val = round(float(val), 2) if pd.notnull(val) else 0.0
                    data.append({
                        "label": str(row[cat_col]),
                        "value": clean_val
                    })

                return {
                    "chart_type": "Bar Chart",
                    "data": data[:30],
                    "x_key": "label",
                    "y_keys": ["value"],
                    "title": f"{num_col} by {cat_col}"
                }

            elif chart_type == "Line Chart":
                dt_col = cols[0] if _is_datetime_series(df[cols[0]]) else cols[1]
                num_col = cols[1] if dt_col == cols[0] else cols[0]

                if not pd.api.types.is_datetime64_any_dtype(df[dt_col]):
                    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")

                df_sorted = df.dropna(subset=[dt_col]).sort_values(by=dt_col)
                data = []
                for _, row in df_sorted.iterrows():
                    val = row[num_col]
                    clean_val = round(float(val), 2) if pd.notnull(val) else 0.0
                    dt_str = row[dt_col].strftime("%Y-%m-%d") if hasattr(row[dt_col], "strftime") else str(row[dt_col])
                    data.append({
                        "date": dt_str,
                        "value": clean_val
                    })

                return {
                    "chart_type": "Line Chart",
                    "data": data[:100],
                    "x_key": "date",
                    "y_keys": ["value"],
                    "title": f"{num_col} Trend over {dt_col}"
                }

            elif chart_type == "Scatter Plot":
                x_col, y_col = cols[0], cols[1]
                data = []
                for _, row in df.dropna(subset=[x_col, y_col]).iterrows():
                    data.append({
                        "x": round(float(row[x_col]), 2),
                        "y": round(float(row[y_col]), 2)
                    })

                return {
                    "chart_type": "Scatter Plot",
                    "data": data[:200],
                    "x_key": "x",
                    "y_keys": ["y"],
                    "title": f"{y_col} vs {x_col}"
                }

            elif chart_type == "Histogram":
                num_col = cols[0]
                series = df[num_col].dropna().astype(float)
                if series.empty:
                    return None

                counts, bin_edges = np.histogram(series, bins=10)
                data = []
                for i in range(len(counts)):
                    bin_label = f"{bin_edges[i]:.0f}–{bin_edges[i+1]:.0f}"
                    data.append({
                        "bin": bin_label,
                        "count": int(counts[i])
                    })

                return {
                    "chart_type": "Histogram",
                    "data": data,
                    "x_key": "bin",
                    "y_keys": ["count"],
                    "title": f"Distribution of {num_col}"
                }

    except Exception:
        return None

    return None
