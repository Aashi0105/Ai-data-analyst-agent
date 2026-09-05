"""
charts.py
---------
Purpose:
    Handles deterministic Python-based chart selection and automated Matplotlib visualization generation
    based on the data structure and data types of the execution result.
"""

from typing import Tuple, Optional, Any
import pandas as pd
import matplotlib.pyplot as plt


def _is_datetime_series(series: pd.Series) -> bool:
    """
    Checks if a series is datetime dtype or can be converted to datetime without high null conversion rate.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        try:
            converted = pd.to_datetime(series, errors="coerce")
            valid_ratio = converted.notnull().sum() / len(series) if len(series) > 0 else 0
            return valid_ratio >= 0.8
        except Exception:
            return False
            
    return False


def detect_chart_type(result: Any) -> Optional[str]:
    """
    Analyzes the structure and column dtypes of result to determine the appropriate chart type.

    Rules:
        - DataFrame (2 cols: 1 categorical/object, 1 numeric) -> "Bar Chart"
        - DataFrame (2 cols: both numeric) -> "Scatter Plot"
        - DataFrame (2 cols: 1 datetime [or convertible], 1 numeric) -> "Line Chart"
        - DataFrame (1 numeric col) -> "Histogram"
        - Series (numeric values) -> "Bar Chart"
        - Otherwise -> None

    Args:
        result (Any): Output result object from sandbox execution.

    Returns:
        Optional[str]: Chart type identifier string or None.
    """
    if isinstance(result, pd.Series):
        if pd.api.types.is_numeric_dtype(result):
            return "Bar Chart"
        return None

    if not isinstance(result, pd.DataFrame) or result.empty:
        return None

    cols = result.columns
    num_cols = len(cols)

    # Case 4: Single numeric column -> Histogram
    if num_cols == 1:
        if pd.api.types.is_numeric_dtype(result.iloc[:, 0]):
            return "Histogram"
        return None

    # Case 1, 2, 3: Exactly 2 columns
    if num_cols == 2:
        col1 = result.iloc[:, 0]
        col2 = result.iloc[:, 1]

        is_dt1 = _is_datetime_series(col1)
        is_dt2 = _is_datetime_series(col2)

        is_num1 = pd.api.types.is_numeric_dtype(col1)
        is_num2 = pd.api.types.is_numeric_dtype(col2)

        # Case 3: Datetime column + numeric column -> Line Chart
        if (is_dt1 and is_num2) or (is_dt2 and is_num1):
            return "Line Chart"

        # Case 2: Exactly 2 numeric columns -> Scatter Plot
        if is_num1 and is_num2:
            return "Scatter Plot"

        is_cat1 = pd.api.types.is_object_dtype(col1) or isinstance(col1.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(col1)
        is_cat2 = pd.api.types.is_object_dtype(col2) or isinstance(col2.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(col2)

        # Case 1: First column object/category and second column numeric -> Bar Chart
        if (is_cat1 and is_num2) or (is_cat2 and is_num1):
            return "Bar Chart"

    return None


def generate_chart(result: Any) -> Tuple[Optional[plt.Figure], Optional[str]]:
    """
    Generates a stylized Matplotlib figure based on deterministic chart type detection.

    Args:
        result (Any): Output result object from execution sandbox.

    Returns:
        Tuple[Optional[plt.Figure], Optional[str]]: (fig, chart_name) or (None, None)
    """
    chart_type = detect_chart_type(result)
    if not chart_type:
        return None, None

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)

    # Modern visual defaults
    accent_color = "#1f77b4"
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.5)

    try:
        if isinstance(result, pd.Series):
            # Case 5: Series -> Bar Chart
            x = [str(i) for i in result.index]
            y = result.values
            ax.bar(x, y, color=accent_color, alpha=0.85)
            ax.set_xlabel(str(result.index.name or "Index"))
            ax.set_ylabel(str(result.name or "Values"))
            ax.set_title(f"{result.name or 'Series'} Distribution", fontsize=12, fontweight="bold")
            if len(x) > 5:
                plt.xticks(rotation=45, ha="right")

        elif isinstance(result, pd.DataFrame):
            cols = result.columns
            if chart_type == "Bar Chart":
                cat_col = cols[0] if not pd.api.types.is_numeric_dtype(result[cols[0]]) else cols[1]
                num_col = cols[1] if cat_col == cols[0] else cols[0]

                x = [str(val) for val in result[cat_col]]
                y = result[num_col].values
                ax.bar(x, y, color=accent_color, alpha=0.85)
                ax.set_xlabel(str(cat_col))
                ax.set_ylabel(str(num_col))
                ax.set_title(f"{num_col} by {cat_col}", fontsize=12, fontweight="bold")
                if len(x) > 5:
                    plt.xticks(rotation=45, ha="right")

            elif chart_type == "Scatter Plot":
                x_col, y_col = cols[0], cols[1]
                ax.scatter(result[x_col], result[y_col], color=accent_color, alpha=0.7, edgecolors="none")
                ax.set_xlabel(str(x_col))
                ax.set_ylabel(str(y_col))
                ax.set_title(f"{y_col} vs {x_col}", fontsize=12, fontweight="bold")

            elif chart_type == "Line Chart":
                dt_col = cols[0] if _is_datetime_series(result[cols[0]]) else cols[1]
                num_col = cols[1] if dt_col == cols[0] else cols[0]

                df_temp = result.copy()
                if not pd.api.types.is_datetime64_any_dtype(df_temp[dt_col]):
                    df_temp[dt_col] = pd.to_datetime(df_temp[dt_col], errors="coerce")

                df_sorted = df_temp.sort_values(by=dt_col)
                ax.plot(df_sorted[dt_col], df_sorted[num_col], color=accent_color, marker="o", linewidth=2)
                ax.set_xlabel(str(dt_col))
                ax.set_ylabel(str(num_col))
                ax.set_title(f"{num_col} over Time ({dt_col})", fontsize=12, fontweight="bold")
                plt.xticks(rotation=45, ha="right")

            elif chart_type == "Histogram":
                num_col = cols[0]
                ax.hist(result[num_col].dropna(), bins=15, color=accent_color, edgecolor="white", alpha=0.85)
                ax.set_xlabel(str(num_col))
                ax.set_ylabel("Frequency")
                ax.set_title(f"Distribution of {num_col}", fontsize=12, fontweight="bold")

        plt.tight_layout()
        return fig, chart_type

    except Exception:
        plt.close(fig)
        return None, None
