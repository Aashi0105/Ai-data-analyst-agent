"""
tests/test_sandbox.py
---------------------
Tests execution sandbox isolation, namespace restrictions,
dataset immutability, and error handling diagnostics.
"""

import pytest
import pandas as pd
import numpy as np
from sandbox import execute_pandas_code, SAFE_BUILTINS


@pytest.fixture
def sample_df():
    """Provides a controlled test DataFrame."""
    return pd.DataFrame({
        "product": ["Laptop", "Mouse", "Keyboard", "Monitor"],
        "price": [1000.0, 25.0, 75.0, 300.0],
        "quantity": [2, 10, 5, 1],
        "region": ["North", "South", "East", "West"]
    })


def test_sandbox_executes_valid_pandas_code(sample_df):
    """Verify standard valid pandas code runs and returns result."""
    code = "result = (df['price'] * df['quantity']).sum()"
    success, result = execute_pandas_code(code, sample_df)
    assert success is True
    assert result == 2925.0


def test_sandbox_preserves_dataset_immutability(sample_df):
    """Verify in-place modifications in the sandbox do NOT mutate the original DataFrame."""
    code = "df['price'] = 0.0\nresult = df['price'].sum()"
    success, result = execute_pandas_code(code, sample_df)
    assert success is True
    assert result == 0.0
    # Original DataFrame must remain unchanged
    assert sample_df["price"].sum() == 1400.0


def test_sandbox_catches_keyerror_with_fuzzy_suggestions(sample_df):
    """Verify KeyError provides useful column suggestions."""
    code = "result = df['prices'].mean()"  # typo in 'price'
    success, error_msg = execute_pandas_code(code, sample_df)
    assert success is False
    assert "Column 'prices' was not found" in error_msg
    assert "'price'" in error_msg
    assert "Available columns are:" in error_msg


def test_sandbox_catches_keyerror_without_suggestions_lists_columns(sample_df):
    """Verify KeyError on non-existent column lists available columns."""
    code = "result = df['non_existent_column_xyz'].mean()"
    success, error_msg = execute_pandas_code(code, sample_df)
    assert success is False
    assert "Column 'non_existent_column_xyz' was not found" in error_msg
    assert "Available columns are:" in error_msg


def test_sandbox_handles_zero_division():
    """Verify division by zero returns clean error rather than unhandled crash."""
    df = pd.DataFrame({"a": [1, 2], "b": [0, 0]})
    code = "result = 10 / 0"
    success, error_msg = execute_pandas_code(code, df)
    assert success is False
    assert "Division by zero" in error_msg


def test_sandbox_handles_name_error(sample_df):
    """Verify undefined variables return clean diagnostic message."""
    code = "result = undefined_variable + 1"
    success, error_msg = execute_pandas_code(code, sample_df)
    assert success is False
    assert "Undefined reference" in error_msg


def test_sandbox_handles_type_error(sample_df):
    """Verify type mismatches return clean error message."""
    code = "result = df['product'] + 5"
    success, error_msg = execute_pandas_code(code, sample_df)
    assert success is False
    assert "Data type mismatch" in error_msg or "Data operation error" in error_msg


def test_sandbox_handles_empty_code_string(sample_df):
    """Verify empty code string fails gracefully."""
    success, error_msg = execute_pandas_code("", sample_df)
    assert success is False
    assert "empty" in error_msg.lower()


def test_sandbox_prevents_unauthorized_builtins(sample_df):
    """Verify builtins outside SAFE_BUILTINS cannot be executed."""
    for forbidden in ["eval", "exec", "open", "input", "__import__"]:
        assert forbidden not in SAFE_BUILTINS
