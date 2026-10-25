"""
tests/test_robustness.py
------------------------
Robustness and pipeline edge-case tests:
- Malformed syntax
- Unsupported AST operations
- Missing values in dataset
- Empty datasets
- Unexpected column data types
- Chart detection robustness
- CSV ingestion edge cases
"""

import pytest
import pandas as pd
import numpy as np
from validator import validate_code, clean_code
from sandbox import execute_pandas_code
from charts import detect_chart_type, generate_chart
from utils import load_csv, get_dataset_summary


def test_malformed_syntax_fails_gracefully():
    """Verify malformed python syntax returns a clear Syntax Error message."""
    bad_codes = [
        "result = df[",
        "result = = 5",
        "def foo(",
        "result = df['a' +]"
    ]
    for code in bad_codes:
        is_valid, msg = validate_code(code)
        assert is_valid is False
        assert "Syntax Error" in msg or "Invalid code" in msg


def test_unsupported_operations_rejected():
    """Verify code with unsupported constructs is rejected cleanly."""
    unsupported_snippets = [
        ("with open('f') as f: result = 1", "With statements are not allowed"),
        ("async def foo(): pass\nresult = 1", "Function definitions are not allowed"),
        ("result = [x for x in range(10)]", "ListComp")  # or disallowed construct
    ]
    for code, expected_kw in unsupported_snippets:
        is_valid, msg = validate_code(code)
        assert is_valid is False


def test_dataset_with_missing_values_handled():
    """Verify operations on datasets with NaN values succeed without unhandled crashes."""
    df_missing = pd.DataFrame({
        "category": ["A", "B", "A", None],
        "sales": [100.0, np.nan, 200.0, 50.0],
        "discount": [np.nan, np.nan, 0.1, 0.2]
    })
    
    # mean ignores NaN by default
    code = "result = df['sales'].mean()"
    is_valid, _ = validate_code(code)
    assert is_valid is True
    success, result = execute_pandas_code(code, df_missing)
    assert success is True
    assert round(result, 2) == 116.67

    # null check
    code_null = "result = int(df['discount'].isnull().sum())"
    is_valid, _ = validate_code(code_null)
    assert is_valid is True
    success, result = execute_pandas_code(code_null, df_missing)
    assert success is True
    assert result == 2


def test_empty_dataframe_handling():
    """Verify operations on an empty DataFrame return expected empty results or fail safely."""
    df_empty = pd.DataFrame(columns=["col1", "col2"])
    code = "result = len(df)"
    success, result = execute_pandas_code(code, df_empty)
    assert success is True
    assert result == 0


def test_chart_detection_robustness():
    """Verify chart engine handles edge cases and empty data without raising exceptions."""
    # Empty DataFrame -> None
    assert detect_chart_type(pd.DataFrame()) is None
    fig, name = generate_chart(pd.DataFrame())
    assert fig is None and name is None

    # Scalar value -> None
    assert detect_chart_type(42.0) is None
    assert detect_chart_type("A") is None

    # Series with numbers -> Bar Chart
    s = pd.Series([10, 20, 30], index=["A", "B", "C"], name="Units")
    assert detect_chart_type(s) == "Bar Chart"

    # 2 cols: Category + Numeric -> Bar Chart
    df_bar = pd.DataFrame({"Region": ["East", "West"], "Sales": [100, 200]})
    assert detect_chart_type(df_bar) == "Bar Chart"

    # 2 cols: Numeric + Numeric -> Scatter Plot
    df_scatter = pd.DataFrame({"Height": [170, 180], "Weight": [65, 75]})
    assert detect_chart_type(df_scatter) == "Scatter Plot"

    # 2 cols: Datetime + Numeric -> Line Chart
    df_line = pd.DataFrame({
        "Date": pd.date_range("2023-01-01", periods=3),
        "Revenue": [100, 150, 200]
    })
    assert detect_chart_type(df_line) == "Line Chart"

    # 1 col numeric -> Histogram
    df_hist = pd.DataFrame({"Ages": [21, 25, 30, 45, 50]})
    assert detect_chart_type(df_hist) == "Histogram"

    # Pie Chart Case A: Categorical + Proportions (sum ~= 1.0)
    df_prop = pd.DataFrame({"diagnosis": ["M", "B"], "proportion": [0.63, 0.37]})
    assert detect_chart_type(df_prop) == "Pie Chart"
    fig, name = generate_chart(df_prop)
    assert fig is not None and name == "Pie Chart"

    # Pie Chart Case B: Categorical + Percentages (sum ~= 100.0)
    df_pct = pd.DataFrame({"Category": ["A", "B", "C"], "pct": [40.0, 35.0, 25.0]})
    assert detect_chart_type(df_pct) == "Pie Chart"
    fig, name = generate_chart(df_pct)
    assert fig is not None and name == "Pie Chart"

    # Pie Chart Case C: Explicit question intent
    df_custom = pd.DataFrame({"Segment": ["Corp", "Consumer"], "Revenue": [5000, 3000]})
    # Without question, ordinary comparison is Bar Chart
    assert detect_chart_type(df_custom) == "Bar Chart"
    # With explicit pie chart question, detected as Pie Chart
    assert detect_chart_type(df_custom, question="can you create me a pie chart to analyze my data?") == "Pie Chart"
    fig, name = generate_chart(df_custom, question="show me a pie chart")
    assert fig is not None and name == "Pie Chart"

    # Series with explicit pie chart question
    s_pie = pd.Series([10, 20, 30], index=["A", "B", "C"], name="Counts")
    assert detect_chart_type(s_pie, question="make a donut chart") == "Pie Chart"
    fig, name = generate_chart(s_pie, question="make a pie chart")
    assert fig is not None and name == "Pie Chart"

    # Negative numbers or arbitrary large sums do not accidentally trigger Pie Chart without question
    df_neg = pd.DataFrame({"Category": ["A", "B"], "Value": [-0.5, 1.5]})
    assert detect_chart_type(df_neg) == "Bar Chart"


def test_dataset_summary_handles_mixed_types():
    """Verify get_dataset_summary correctly handles nulls and calculates statistics."""
    df = pd.DataFrame({
        "name": ["Alice", "Bob", None],
        "age": [25, 30, np.nan],
        "score": [95.5, 88.0, 72.5]
    })
    summary = get_dataset_summary(df)
    assert summary["num_rows"] == 3
    assert summary["num_cols"] == 3
    assert summary["total_missing"] == 2
    assert summary["duplicate_rows"] == 0
    assert "name" in summary["columns"]


def test_load_csv_rejects_empty_file(tmp_path):
    """Verify load_csv raises descriptive ValueError for completely empty CSVs."""
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("")
    with pytest.raises(ValueError, match="contains no data"):
        load_csv(str(empty_file))
