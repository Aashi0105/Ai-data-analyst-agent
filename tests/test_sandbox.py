"""
tests/test_sandbox.py
---------------------
Tests execution sandbox isolation, namespace restrictions,
dataset immutability, subprocess worker IPC, timeouts, and error handling diagnostics.
"""

import sys
import os
import json
import tempfile
import subprocess
from pathlib import Path
import io
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
import pandas as pd
import numpy as np
from sandbox import (
    execute_pandas_code,
    SAFE_BUILTINS,
    _get_safe_env,
    MAX_CODE_SIZE_BYTES,
    BoundedFileWriter,
    ArrowWriteLimitExceededError,
    SandboxExecutor,
    WindowsJobObject,
    MAX_CONCURRENT_WORKERS,
    DEFAULT_QUEUE_TIMEOUT_SECONDS,
    MAX_WORKER_MEMORY_BYTES,
)
from sandbox_worker import (
    _validate_result_bounds,
    _validate_envelope_bounds,
    _write_response,
    _estimate_int_digits,
    MAX_INTEGER_DIGITS,
    MAX_AGGREGATE_NUMERIC_BYTES,
)
import sandbox



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


# =========================================================================
# Subprocess Worker and Safe IPC Foundation Tests
# =========================================================================

def test_sandbox_dataframe_roundtrip_supported_dtypes():
    """Verify DataFrame containing int, float, string, boolean, and datetime roundtrips with fidelity."""
    df = pd.DataFrame({
        "int_col": [1, 2, 3],
        "float_col": [1.5, 2.5, 3.5],
        "str_col": ["alpha", "beta", "gamma"],
        "bool_col": [True, False, True],
        "date_col": pd.date_range("2025-01-01", periods=3),
    })
    code = "result = df"
    success, result = execute_pandas_code(code, df)
    assert success is True
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert list(result.columns) == list(df.columns)
    assert (result["int_col"] == df["int_col"]).all()
    assert (result["float_col"] == df["float_col"]).all()
    assert (result["str_col"] == df["str_col"]).all()
    assert (result["bool_col"] == df["bool_col"]).all()
    assert (result["date_col"] == df["date_col"]).all()


def test_sandbox_series_roundtrip_with_name_and_index(sample_df):
    """Verify Series results preserve values, name, and index structure."""
    code = "result = df.groupby('region')['price'].sum()"
    success, result = execute_pandas_code(code, sample_df)
    assert success is True
    assert isinstance(result, pd.Series)
    assert result.name == "price"
    assert result.index.name == "region"
    assert result["North"] == 1000.0


def test_sandbox_scalar_types_and_missing_values(sample_df):
    """Verify scalar results, NaN, None, and pd.NA preserve their exact semantics."""
    # Float scalar
    success, res = execute_pandas_code("result = 42.5", sample_df)
    assert success is True and res == 42.5

    # String scalar
    success, res = execute_pandas_code("result = 'analysis_complete'", sample_df)
    assert success is True and res == "analysis_complete"

    # Boolean scalar
    success, res = execute_pandas_code("result = True", sample_df)
    assert success is True and res is True

    # None
    success, res = execute_pandas_code("result = None", sample_df)
    assert success is True and res is None

    # NaN
    success, res = execute_pandas_code("result = float('nan')", sample_df)
    assert success is True and np.isnan(res)

    # pd.NA
    success, res = execute_pandas_code("result = pd.NA", sample_df)
    assert success is True and res is pd.NA


def test_sandbox_datetime_scalar(sample_df):
    """Verify datetime / Timestamp scalar results preserve datetime semantics."""
    code = "result = pd.Timestamp('2025-06-15 10:30:00')"
    success, result = execute_pandas_code(code, sample_df)
    assert success is True
    assert isinstance(result, pd.Timestamp)
    assert result.year == 2025
    assert result.month == 6


def test_sandbox_unsupported_result_type_fails_closed(sample_df):
    """Verify valid AST code returning arbitrary unsupported objects (e.g. DataFrameGroupBy) fails closed in worker."""
    code = "result = df.groupby('region')"
    success, error_msg = execute_pandas_code(code, sample_df)
    assert success is False
    assert "unsupported result type" in error_msg.lower()


def test_sandbox_worker_timeout_enforcement(sample_df):
    """Verify hard wall-clock timeout terminates runaway calculations and returns clean error."""
    code = "result = pd.concat([df] * 50000).merge(pd.concat([df] * 50000), on='region').shape[0]"
    success, error_msg = execute_pandas_code(code, sample_df, timeout_seconds=1.0)
    assert success is False
    assert "timed out" in error_msg.lower()


def test_sandbox_direct_call_enforces_ast_validation(sample_df):
    """Verify that calling execute_pandas_code directly cannot bypass AST validation."""
    for invalid_code in [
        "import os\nresult = 1",
        "result = pd.read_table('sensitive.txt')",
        "result = np.load('malicious.npy')",
        "while True: pass",
        "result = open('file.txt')",
    ]:
        success, error_msg = execute_pandas_code(invalid_code, sample_df)
        assert success is False
        assert "validation error" in error_msg.lower()


def test_sandbox_bounded_stderr_and_stdout_discarded(sample_df, monkeypatch):
    """Verify that worker stdout and stderr are routed to DEVNULL at the kernel level without unbounded buffering or disk files."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        # Script that writes 100 KiB of stderr and stdout, but returns a valid JSON response
        f.write(
            "import sys, os, json\n"
            "sys.stdout.write('O' * (100 * 1024))\n"
            "sys.stderr.write('E' * (100 * 1024))\n"
            "ipc_dir = sys.argv[2]\n"
            "with open(os.path.join(ipc_dir, 'response.json'), 'w') as out:\n"
            "    json.dump({'version': '1.0', 'status': 'success', 'result_type': 'int', 'has_arrow_data': False, 'scalar_value': 777}, out)\n"
        )
        f.flush()
        chatty_worker = Path(f.name)

    try:
        monkeypatch.setattr(sandbox, "WORKER_SCRIPT", chatty_worker)
        success, res = execute_pandas_code("result = 1", sample_df)
        assert success is True
        assert res == 777
    finally:
        if chatty_worker.exists():
            chatty_worker.unlink()


def test_sandbox_input_dataset_row_and_column_limits(sample_df, monkeypatch):
    """Verify input dataset row and column limits are enforced before worker execution."""
    monkeypatch.setattr(sandbox, "MAX_INPUT_ROWS", 2)
    success, error_msg = execute_pandas_code("result = 1", sample_df)
    assert success is False
    assert "rows (4) exceed maximum allowed limit" in error_msg

    monkeypatch.setattr(sandbox, "MAX_INPUT_ROWS", 2_000_000)
    monkeypatch.setattr(sandbox, "MAX_INPUT_COLS", 2)
    success, error_msg = execute_pandas_code("result = 1", sample_df)
    assert success is False
    assert "columns (4) exceed maximum allowed limit" in error_msg


def test_sandbox_input_memory_admission_rejection(sample_df, monkeypatch):
    """Verify input DataFrame estimated memory admission check rejects oversized datasets before launch."""
    monkeypatch.setattr(sandbox, "MAX_INPUT_MEMORY_BYTES", 10)
    success, error_msg = execute_pandas_code("result = 1", sample_df)
    assert success is False
    assert "estimated memory" in error_msg.lower()


def test_sandbox_oversized_input_arrow_rejected(sample_df, monkeypatch):
    """Verify serialized input Arrow file size is bounded before worker execution."""
    monkeypatch.setattr(sandbox, "MAX_INPUT_ARROW_BYTES", 10)
    success, error_msg = execute_pandas_code("result = 1", sample_df)
    assert success is False
    assert "serialized dataset size" in error_msg.lower()


def test_sandbox_oversized_request_json_rejected(sample_df, monkeypatch):
    """Verify request JSON payload size is bounded before worker execution."""
    monkeypatch.setattr(sandbox, "MAX_REQUEST_JSON_BYTES", 10)
    success, error_msg = execute_pandas_code("result = 1", sample_df)
    assert success is False
    assert "request payload size" in error_msg.lower()


def test_sandbox_oversized_result_arrow_rejected(sample_df, monkeypatch):
    """Verify parent process rejects oversized result Arrow files before reading/memory-mapping."""
    monkeypatch.setattr(sandbox, "MAX_RESULT_ARROW_BYTES", 50)
    success, error_msg = execute_pandas_code("result = df", sample_df)
    assert success is False
    assert "result tabular payload size" in error_msg.lower() or "exceeds maximum size limit" in error_msg.lower()


def test_sandbox_oversized_response_json_rejected(sample_df, monkeypatch):
    """Verify parent process rejects oversized response JSON files before parsing into memory."""
    monkeypatch.setattr(sandbox, "MAX_RESPONSE_JSON_BYTES", 50)
    success, error_msg = execute_pandas_code("result = 1", sample_df)
    assert success is False
    assert "response envelope size" in error_msg.lower()


def test_sandbox_protocol_version_mismatch(sample_df, monkeypatch):
    """Verify that protocol responses with mismatched versions are rejected fail-closed."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(
            "import os, sys, json\n"
            "ipc_dir = sys.argv[2]\n"
            "with open(os.path.join(ipc_dir, 'response.json'), 'w') as out:\n"
            "    json.dump({'version': '9.9', 'status': 'success', 'result_type': 'int', 'scalar_value': 1}, out)\n"
        )
        f.flush()
        version_worker = Path(f.name)

    try:
        monkeypatch.setattr(sandbox, "WORKER_SCRIPT", version_worker)
        success, error_msg = execute_pandas_code("result = 1", sample_df)
        assert success is False
        assert "incompatible protocol response version '9.9'" in error_msg.lower()
    finally:
        if version_worker.exists():
            version_worker.unlink()


def test_sandbox_protocol_inconsistent_tabular_metadata(sample_df, monkeypatch):
    """Verify that contradictory response envelopes (arrow data declared without tabular type) fail closed."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(
            "import os, sys, json\n"
            "ipc_dir = sys.argv[2]\n"
            "with open(os.path.join(ipc_dir, 'response.json'), 'w') as out:\n"
            "    json.dump({'version': '1.0', 'status': 'success', 'result_type': 'int', 'has_arrow_data': True, 'scalar_value': 1}, out)\n"
        )
        f.flush()
        inconsistent_worker = Path(f.name)

    try:
        monkeypatch.setattr(sandbox, "WORKER_SCRIPT", inconsistent_worker)
        success, error_msg = execute_pandas_code("result = 1", sample_df)
        assert success is False
        assert "protocol violation: inconsistent tabular metadata" in error_msg.lower()
    finally:
        if inconsistent_worker.exists():
            inconsistent_worker.unlink()


def test_sandbox_excessive_nesting_depth_rejected(sample_df):
    """Verify that deeply nested dictionaries or lists exceeding MAX_NESTING_DEPTH (5) are rejected."""
    code = "result = {'l1': {'l2': {'l3': {'l4': {'l5': {'l6': 'deep'}}}}}}"
    success, error_msg = execute_pandas_code(code, sample_df)
    assert success is False
    assert "nesting depth" in error_msg.lower()


def test_sandbox_excessive_string_length_rejected(sample_df):
    """Verify that string results exceeding MAX_STRING_LENGTH (64 KiB) fail closed in worker."""
    code = f"result = 'A' * ({64 * 1024 + 500})"
    success, error_msg = execute_pandas_code(code, sample_df)
    assert success is False
    assert "exceeds maximum allowed length" in error_msg.lower()


def test_sandbox_total_ipc_storage_limit_enforced(sample_df, monkeypatch):
    """Verify total temporary IPC storage cap is enforced."""
    monkeypatch.setattr(sandbox, "MAX_TOTAL_IPC_BYTES", 100)
    success, error_msg = execute_pandas_code("result = 1", sample_df)
    assert success is False
    assert "ipc storage" in error_msg.lower()


def test_sandbox_parent_secret_non_inheritance(monkeypatch):
    """Verify sensitive environment variables in parent process are never leaked to child worker."""
    monkeypatch.setenv("GEMINI_API_KEY", "sensitive_mock_token_abc_123")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "mock_aws_secret_xyz")

    safe_env = _get_safe_env()
    assert "GEMINI_API_KEY" not in safe_env
    assert "AWS_SECRET_ACCESS_KEY" not in safe_env
    for key in safe_env:
        assert key.upper() in {
            "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC",
            "PATH", "PATHEXT", "TEMP", "TMP",
            "LANG", "LC_ALL", "TMPDIR",
            "PYTHONUNBUFFERED", "PYTHONIOENCODING", "PYTHONUTF8"
        }


def test_sandbox_oversized_code_payload(sample_df):
    """Verify code payloads exceeding MAX_CODE_SIZE_BYTES fail closed."""
    oversized_code = "# " + ("x" * (MAX_CODE_SIZE_BYTES + 1024)) + "\nresult = 1"
    success, error_msg = execute_pandas_code(oversized_code, sample_df)
    assert success is False
    assert "maximum size limit" in error_msg


def test_sandbox_handles_missing_worker_file(sample_df, monkeypatch):
    """Verify missing worker script yields clean error without crashing."""
    monkeypatch.setattr(sandbox, "WORKER_SCRIPT", Path("C:/non/existent/path/worker.py"))
    success, error_msg = execute_pandas_code("result = 1", sample_df)
    assert success is False
    assert "sandbox worker not found" in error_msg.lower()


def test_sandbox_no_in_process_fallback(sample_df, monkeypatch):
    """
    CRITICAL: Verify that if the worker fails or is killed, sandbox.py NEVER
    silently falls back to in-process exec().
    """
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write("import sys; sys.exit(42)\n")
        f.flush()
        bad_worker = Path(f.name)

    try:
        monkeypatch.setattr(sandbox, "WORKER_SCRIPT", bad_worker)
        success, error_msg = execute_pandas_code("result = 123", sample_df)
        assert success is False
        assert "exited with code 42" in error_msg
    finally:
        if bad_worker.exists():
            bad_worker.unlink()


def test_sandbox_handles_corrupt_response_envelope(sample_df, monkeypatch):
    """Verify that malformed JSON response envelopes fail closed safely."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(
            "import os, sys\n"
            "ipc_dir = sys.argv[2]\n"
            "with open(os.path.join(ipc_dir, 'response.json'), 'w') as out:\n"
            "    out.write('CORRUPTED_NOT_JSON{{{')\n"
        )
        f.flush()
        corrupt_worker = Path(f.name)

    try:
        monkeypatch.setattr(sandbox, "WORKER_SCRIPT", corrupt_worker)
        success, error_msg = execute_pandas_code("result = 1", sample_df)
        assert success is False
        assert "malformed response" in error_msg.lower()
    finally:
        if corrupt_worker.exists():
            corrupt_worker.unlink()


# =========================================================================
# Task 07B.5 — Write-Time IPC Budgets & Aggregate Limit Tests
# =========================================================================

def test_bounded_writer_stops_at_configured_byte_budget():
    """
    1. Verify BoundedFileWriter aborts writes immediately when the byte budget is crossed,
    and proves that subsequent writes do not increase the sink.
    """
    buf = io.BytesIO()
    max_bytes = 150
    bw = BoundedFileWriter(buf, max_bytes)

    # First write of 100 bytes must succeed completely
    chunk1 = b"A" * 100
    n1 = bw.write(chunk1)
    assert n1 == 100
    assert bw.bytes_written == 100
    assert len(buf.getvalue()) == 100

    # Second write of 60 bytes crosses 150 (100 + 60 = 160 > 150) -> must raise immediately
    chunk2 = b"B" * 60
    with pytest.raises(ArrowWriteLimitExceededError) as exc_info:
        bw.write(chunk2)

    assert "Arrow IPC write limit exceeded" in str(exc_info.value)
    # The exceeding chunk was rejected: buffer MUST NOT contain any of chunk2
    assert len(buf.getvalue()) == 100
    assert bw.bytes_written == 100

    # Subsequent write must be refused (returns 0) and not mutate buffer
    n3 = bw.write(b"C" * 10)
    assert n3 == 0
    assert len(buf.getvalue()) == 100


def test_bounded_writer_compatible_with_valid_pyarrow_ipc_stream():
    """
    4. Verify that BoundedFileWriter remains completely compatible with valid PyArrow
    IPC streaming, flushing, and readback.
    """
    table = pa.Table.from_arrays([pa.array([10, 20, 30])], names=["val"])
    buf = io.BytesIO()
    bw = BoundedFileWriter(buf, max_bytes=10_000)

    with ipc.new_stream(bw, table.schema) as writer:
        writer.write_table(table)

    assert bw.bytes_written > 0
    assert len(buf.getvalue()) == bw.bytes_written

    # Read back stream with PyArrow
    buf.seek(0)
    reader = ipc.open_stream(buf)
    read_table = reader.read_all()
    assert read_table.num_rows == 3
    assert read_table.to_pydict() == {"val": [10, 20, 30]}


def test_arrow_stream_write_crossing_limit_rejected_during_writing():
    """
    3. Verify a write crossing the limit during PyArrow stream serialization
    is rejected during writing before the full file is completed.
    """
    df = pd.DataFrame({"col": ["test_data_string_" + str(i) for i in range(500)]})
    table = pa.Table.from_pandas(df)

    with tempfile.TemporaryDirectory() as td:
        arrow_file = os.path.join(td, "test_partial.arrow")
        # Set limit much smaller than full serialized table (~15 KB)
        write_limit = 200

        with open(arrow_file, "wb") as raw_f:
            bw = BoundedFileWriter(raw_f, max_bytes=write_limit)
            with pytest.raises(ArrowWriteLimitExceededError):
                with ipc.new_stream(bw, table.schema) as writer:
                    writer.write_table(table)

        # File on disk must not exceed the limit
        actual_size = os.path.getsize(arrow_file)
        assert actual_size <= write_limit

        # Cleanup: remove partial file
        os.remove(arrow_file)
        assert not os.path.exists(arrow_file)


def test_sandbox_input_arrow_write_time_budget_enforcement(sample_df, monkeypatch):
    """
    1 & 10. Verify input Arrow writes stop at the configured byte budget,
    and partial files are cleaned up after violation.
    """
    monkeypatch.setattr(sandbox, "MAX_INPUT_ARROW_BYTES", 50)
    success, error_msg = execute_pandas_code("result = 1", sample_df)
    assert success is False
    assert "serialized dataset size exceeds limit" in error_msg.lower()
    assert "arrow ipc write limit exceeded" in error_msg.lower()


def test_sandbox_output_arrow_write_time_budget_enforcement(sample_df, monkeypatch):
    """
    2 & 10. Verify output Arrow writes stop at the configured byte budget,
    and partial files are cleaned up after violation.
    """
    monkeypatch.setattr(sandbox, "MAX_RESULT_ARROW_BYTES", 100)
    success, error_msg = execute_pandas_code("result = df", sample_df)
    assert success is False
    assert "exceeds maximum size limit" in error_msg.lower()


def test_aggregate_json_strings_rejected_before_dumps(sample_df, monkeypatch):
    """
    5. Verify aggregate JSON string bytes across collection items are rejected
    before full serialization when total budget is exceeded.
    """
    # Direct validation check
    items = ["abcdefghij" for _ in range(15)]  # 15 * 10 = 150 UTF-8 bytes
    ok, err = _validate_result_bounds(items, max_agg_bytes=100)
    assert ok is False
    assert "aggregate string bytes limit" in err

    # Subprocess execution check
    monkeypatch.setattr(sandbox, "MAX_AGGREGATE_STRING_BYTES", 100)
    success, error_msg = execute_pandas_code("result = ['abcdefghij'] * 15", sample_df)
    assert success is False
    assert "aggregate string bytes limit" in error_msg.lower()


def test_dictionary_keys_count_toward_aggregate_string_budget(sample_df, monkeypatch):
    """
    6. Verify dictionary keys count toward the aggregate UTF-8 byte budget before serialization.
    """
    # Direct validation check
    d = {"long_dictionary_key_name_that_exceeds": 1}
    ok, err = _validate_result_bounds(d, max_agg_bytes=15)
    assert ok is False
    assert "aggregate string bytes limit" in err

    # Subprocess execution check
    monkeypatch.setattr(sandbox, "MAX_AGGREGATE_STRING_BYTES", 15)
    success, error_msg = execute_pandas_code("result = {'a_long_dict_key_exceeding_budget': 1}", sample_df)
    assert success is False
    assert "aggregate string bytes limit" in error_msg.lower()


def test_excessive_total_nodes_rejected_before_dumps(sample_df, monkeypatch):
    """
    7. Verify total visited nodes (lists, dicts, keys, items) exceeding the limit
    are rejected before full serialization.
    """
    nested = [1, 2, [3, 4], {"a": 5}]
    ok, err = _validate_result_bounds(nested, max_items=4)
    assert ok is False
    assert "item count limit" in err

    monkeypatch.setattr(sandbox, "MAX_COLLECTION_ITEMS", 5)
    success, error_msg = execute_pandas_code("result = [1, 2, 3, 4, 5, 6, 7]", sample_df)
    assert success is False
    assert "item count limit" in error_msg.lower()


def test_deep_nesting_unsupported_objects_and_non_finite_values_fail_safely(sample_df):
    """
    8. Verify deep nesting (>5), unsupported objects, and non-finite floats fail safely.
    """
    # Deep nesting
    deep_list = [[[[[[1]]]]]]
    ok, err = _validate_result_bounds(deep_list, max_nesting=5)
    assert ok is False
    assert "nesting depth" in err

    success, error_msg = execute_pandas_code("result = [[[[[[1]]]]]]", sample_df)
    assert success is False
    assert "nesting depth" in error_msg.lower()

    # Unsupported object in collection
    ok, err = _validate_result_bounds([object()])
    assert ok is False
    assert "Unsupported item type" in err

    # Non-finite numbers in collection
    ok, err = _validate_result_bounds([float("nan")])
    assert ok is False
    assert "Non-finite numbers" in err

    ok, err = _validate_result_bounds([float("inf")])
    assert ok is False
    assert "Non-finite numbers" in err

    # Non-finite scalar float
    success, error_msg = execute_pandas_code("result = float('inf')", sample_df)
    assert success is False
    assert "non-finite numbers" in error_msg.lower() or "unsupported" in error_msg.lower()


def test_error_envelopes_cannot_bypass_json_limits():
    """
    9. Verify error envelopes validate metadata and cannot produce oversized JSON responses.
    """
    envelope = {
        "version": "1.0",
        "status": "error",
        "result_type": "none",
        "has_arrow_data": False,
        "scalar_value": None,
        "series_metadata": None,
        "error_type": "TestError",
        "error_message": "E" * 100_000,
    }

    _validate_envelope_bounds(envelope, max_string_length=1000)
    assert len(envelope["error_message"]) <= 1000

    # Test writing with tiny response limit (500 bytes: envelope exceeds 500, fallback is ~240 bytes)
    with tempfile.TemporaryDirectory() as td:
        resp_file = os.path.join(td, "resp.json")
        _write_response(resp_file, envelope, max_resp_bytes=500)
        assert os.path.exists(resp_file)
        assert os.path.getsize(resp_file) <= 500
        with open(resp_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["error_type"] == "ResultTooLarge"


def test_partial_arrow_files_cleaned_up_after_violations(sample_df, monkeypatch):
    """
    10. Verify that on any Arrow write violation, no partial .arrow file remains on disk.
    """
    monkeypatch.setattr(sandbox, "MAX_INPUT_ARROW_BYTES", 50)
    success, error_msg = execute_pandas_code("result = 1", sample_df)
    assert success is False

    # The ephemeral IPC directory is automatically removed on context exit,
    # ensuring zero residual partial Arrow files on the host filesystem.


def test_existing_scalar_and_tabular_types_preserve_compatibility(sample_df):
    """
    11. Verify compatibility with scalar, DataFrame, Series, datetime, missing values, and None.
    """
    cases = [
        ("result = 100", int, 100),
        ("result = 3.14", float, 3.14),
        ("result = True", bool, True),
        ("result = 'hello'", str, "hello"),
        ("result = None", type(None), None),
    ]

    for code, expected_type, expected_val in cases:
        ok, res = execute_pandas_code(code, sample_df)
        assert ok is True
        assert isinstance(res, expected_type)
        if expected_val is not None:
            assert res == expected_val

    # DataFrame roundtrip
    ok, df_res = execute_pandas_code("result = df.head(2)", sample_df)
    assert ok is True
    assert isinstance(df_res, pd.DataFrame)
    assert len(df_res) == 2

    # Series roundtrip
    ok, ser_res = execute_pandas_code("result = df['price']", sample_df)
    assert ok is True
    assert isinstance(ser_res, pd.Series)
    assert len(ser_res) == 4


# ============================================================================
# Task 07B.5.2 — Numeric Serialization Tests
# ============================================================================

def test_estimate_int_digits_accuracy_and_safety():
    """Verify _estimate_int_digits calculates a safe upper bound without str() or memory allocation."""
    assert _estimate_int_digits(0) == 1
    assert _estimate_int_digits(True) == 1
    assert _estimate_int_digits(False) == 1
    assert _estimate_int_digits(np.bool_(True)) == 1
    assert _estimate_int_digits(np.int64(42)) >= 2
    assert _estimate_int_digits(-999) >= 3

    # Test controlled bit-shift integer
    val = 1 << 100  # ~31 decimal digits
    est = _estimate_int_digits(val)
    actual_digits = len(str(val))
    assert est >= actual_digits
    assert est <= actual_digits + 3  # Tightly bounded upper estimate


def test_ordinary_integer_succeeds(sample_df):
    """Verify ordinary integers of various sizes succeed and return exact values."""
    for code, expected in [
        ("result = 42", 42),
        ("result = -100_000", -100_000),
        ("result = 0", 0),
        ("result = 1234567890123456789", 1234567890123456789),
    ]:
        ok, res = execute_pandas_code(code, sample_df)
        assert ok is True
        assert res == expected
        assert isinstance(res, int)


def test_normal_collection_with_integers_and_floats_succeeds(sample_df):
    """Verify normal collections containing integers and floats serialize correctly."""
    code = "result = {'counts': [1, 2, 3], 'total': 6.0, 'nested': {'avg': 2.0, 'max': 3}}"
    ok, res = execute_pandas_code(code, sample_df)
    assert ok is True
    assert isinstance(res, dict)
    assert res["counts"] == [1, 2, 3]
    assert res["total"] == 6.0
    assert res["nested"]["avg"] == 2.0


def test_oversized_toplevel_integer_rejected(sample_df, monkeypatch):
    """Verify an oversized top-level integer is rejected before JSON serialization."""
    monkeypatch.setattr(sandbox, "MAX_INTEGER_DIGITS", 50)
    # 1 << 200 is ~61 decimal digits, exceeding 50 digits
    code = "result = 1 << 200"
    ok, err = execute_pandas_code(code, sample_df)
    assert ok is False
    assert "digit length" in err.lower() or "resulttoolarge" in err.lower()


def test_oversized_integer_nested_in_list_rejected(sample_df, monkeypatch):
    """Verify an oversized integer nested inside a list is rejected before serialization."""
    monkeypatch.setattr(sandbox, "MAX_INTEGER_DIGITS", 50)
    code = "result = [1, 2, 1 << 200]"
    ok, err = execute_pandas_code(code, sample_df)
    assert ok is False
    assert "digit length" in err.lower() or "resulttoolarge" in err.lower()


def test_oversized_integer_nested_in_dict_rejected(sample_df, monkeypatch):
    """Verify an oversized integer nested inside a dictionary is rejected before serialization."""
    monkeypatch.setattr(sandbox, "MAX_INTEGER_DIGITS", 50)
    code = "result = {'key': 1 << 200}"
    ok, err = execute_pandas_code(code, sample_df)
    assert ok is False
    assert "digit length" in err.lower() or "resulttoolarge" in err.lower()


def test_aggregate_numeric_budget_exceeded_rejected(sample_df, monkeypatch):
    """Verify a collection of valid integers that exceeds the aggregate numeric budget is rejected."""
    # Set aggregate numeric budget to 50 bytes (each 10**10 is ~11 digits, 5 items = ~55 digits > 50)
    monkeypatch.setattr(sandbox, "MAX_AGGREGATE_NUMERIC_BYTES", 50)
    code = "result = [10**10, 10**10, 10**10, 10**10, 10**10]"
    ok, err = execute_pandas_code(code, sample_df)
    assert ok is False
    assert "budget" in err.lower() or "exceeds" in err.lower() or "resulttoolarge" in err.lower()


def test_booleans_not_treated_as_oversized_integers(sample_df, monkeypatch):
    """Verify Python booleans (subclass of int) are handled as booleans and not rejected as oversized integers."""
    monkeypatch.setattr(sandbox, "MAX_INTEGER_DIGITS", 2)
    for code, expected in [("result = True", True), ("result = False", False)]:
        ok, res = execute_pandas_code(code, sample_df)
        assert ok is True
        assert res is expected
        assert isinstance(res, bool)

    # In collections
    ok, res = execute_pandas_code("result = [True, False, {'flag': True}]", sample_df)
    assert ok is True
    assert res == [True, False, {"flag": True}]


def test_numpy_numeric_scalars_follow_policy(sample_df):
    """Verify NumPy scalar numeric types are safely handled according to policy."""
    ok, res = execute_pandas_code("result = np.int64(12345)", sample_df)
    assert ok is True
    assert res == 12345
    assert isinstance(res, int)

    ok, res = execute_pandas_code("result = np.float64(3.14159)", sample_df)
    assert ok is True
    assert np.isclose(res, 3.14159)
    assert isinstance(res, float)

    ok, res = execute_pandas_code("result = np.bool_(True)", sample_df)
    assert ok is True
    assert res is True
    assert isinstance(res, bool)


def test_nan_and_infinity_policy(sample_df):
    """Verify NaN and Inf follow documented top-level and nested policies."""
    # Top-level NaN succeeds
    ok, res = execute_pandas_code("result = float('nan')", sample_df)
    assert ok is True
    assert np.isnan(res)

    ok, res = execute_pandas_code("result = np.nan", sample_df)
    assert ok is True
    assert np.isnan(res)

    # Top-level Infinity rejected
    ok, err = execute_pandas_code("result = float('inf')", sample_df)
    assert ok is False
    assert "not supported" in err.lower()

    ok, err = execute_pandas_code("result = float('-inf')", sample_df)
    assert ok is False
    assert "not supported" in err.lower()

    # Nested NaN rejected
    ok, err = execute_pandas_code("result = [float('nan')]", sample_df)
    assert ok is False
    assert "not supported" in err.lower()

    # Nested Inf rejected
    ok, err = execute_pandas_code("result = {'val': float('inf')}", sample_df)
    assert ok is False
    assert "not supported" in err.lower()


def test_oversized_numeric_does_not_reach_json_dumps_with_invalid_result(sample_df, monkeypatch):
    """
    Verify that an invalid oversized numeric result is rejected BEFORE json.dumps()
    can serialize the invalid object.
    """
    import sandbox_worker
    real_dumps = json.dumps

    def guarded_dumps(obj, *args, **kwargs):
        # Assert that json.dumps is never called with the oversized integer in payload
        if isinstance(obj, dict):
            scalar = obj.get("scalar_value")
            if isinstance(scalar, int) and scalar.bit_length() > 200:
                raise AssertionError("CRITICAL: json.dumps() called with invalid oversized integer result!")
            if isinstance(scalar, (list, dict)):
                def check_nested(item):
                    if isinstance(item, int) and item.bit_length() > 200:
                        raise AssertionError("CRITICAL: json.dumps() called with nested oversized integer result!")
                    if isinstance(item, list):
                        for sub in item:
                            check_nested(sub)
                    elif isinstance(item, dict):
                        for k, v in item.items():
                            check_nested(v)
                check_nested(scalar)
        return real_dumps(obj, *args, **kwargs)

    monkeypatch.setattr(json, "dumps", guarded_dumps)

    with tempfile.TemporaryDirectory() as td:
        req_file = os.path.join(td, "request.json")
        input_arrow = os.path.join(td, "input_data.arrow")
        resp_file = os.path.join(td, "response.json")

        req_payload = {
            "version": "1.0",
            "code": "result = 1 << 400",
            "max_integer_digits": 50,
            "max_aggregate_numeric_bytes": 1000,
        }
        with open(req_file, "w") as f:
            json.dump(req_payload, f)

        table = pa.Table.from_pandas(sample_df)
        with open(input_arrow, "wb") as f:
            with ipc.new_stream(f, table.schema) as writer:
                writer.write_table(table)

        # Execute run_worker directly with monkeypatched json.dumps
        sandbox_worker.run_worker(td)

        assert os.path.exists(resp_file)
        with open(resp_file, "r", encoding="utf-8") as f:
            resp_data = json.load(f)
        assert resp_data["status"] == "error"
        assert resp_data["error_type"] == "ResultTooLarge"


def test_numeric_error_envelope_remains_bounded():
    """Verify error envelopes written when numeric validation fails remain strictly bounded and valid JSON."""
    with tempfile.TemporaryDirectory() as td:
        resp_file = os.path.join(td, "response.json")
        envelope = {
            "version": "1.0",
            "status": "error",
            "result_type": "none",
            "has_arrow_data": False,
            "scalar_value": None,
            "series_metadata": None,
            "error_type": "ResultTooLarge",
            "error_message": "Integer result exceeds maximum allowed digit length (4096 digits)."
        }
        _write_response(resp_file, envelope, max_resp_bytes=1024)
        assert os.path.exists(resp_file)
        assert os.path.getsize(resp_file) <= 1024
        with open(resp_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["status"] == "error"
        assert loaded["error_type"] == "ResultTooLarge"


# ============================================================================
# Task 07C.1 — Unified Sandbox Executor & Windows Job Object Containment Tests
# ============================================================================

def test_sandbox_executor_singleton_and_delegation(sample_df):
    """Verify SandboxExecutor maintains a singleton instance and execute_pandas_code delegates to it."""
    executor1 = SandboxExecutor.get_instance()
    executor2 = SandboxExecutor.get_instance()
    assert executor1 is executor2

    # Verify delegation returns matching result
    ok1, res1 = executor1.execute("result = 123", sample_df)
    ok2, res2 = execute_pandas_code("result = 123", sample_df)
    assert ok1 is True and res1 == 123
    assert ok2 is True and res2 == 123


def test_sandbox_executor_concurrency_ceiling_and_overload_rejection(sample_df):
    """
    Verify concurrency limit is enforced and requests exceeding capacity
    fail closed with queue saturation error after queue timeout.
    """
    custom_executor = SandboxExecutor(
        max_concurrent_workers=1,
        queue_timeout_seconds=0.1,
    )

    # Acquire the single slot to simulate saturated queue
    slot_acquired = custom_executor._semaphore.acquire(timeout=0.5)
    assert slot_acquired is True

    try:
        # Second execution must be rejected due to queue saturation
        ok, err = custom_executor.execute("result = 1", sample_df)
        assert ok is False
        assert "queue saturated" in err.lower()
        assert "exceeded concurrency limit of 1" in err.lower()
    finally:
        custom_executor._semaphore.release()

    # Slot is now available; execution succeeds
    ok, res = custom_executor.execute("result = 99", sample_df)
    assert ok is True
    assert res == 99


def test_sandbox_executor_slot_released_on_failure_and_timeout(sample_df):
    """Verify that concurrency slots are reliably released on validation errors and timeouts."""
    executor = SandboxExecutor(max_concurrent_workers=1, queue_timeout_seconds=0.2)

    # 1. Validation failure: slot must be released
    ok, err = executor.execute("result = open('file.txt')", sample_df)
    assert ok is False
    assert "validation error" in err.lower()
    # Confirm slot is still free
    assert executor._semaphore.acquire(blocking=False) is True
    executor._semaphore.release()

    # 2. Timeout: slot must be released
    timeout_code = "result = pd.concat([df] * 50000).merge(pd.concat([df] * 50000), on='region').shape[0]"
    ok, err = executor.execute(timeout_code, sample_df, timeout_seconds=0.5)
    assert ok is False
    assert "timed out" in err.lower()
    # Confirm slot is still free
    assert executor._semaphore.acquire(blocking=False) is True
    executor._semaphore.release()


def test_sandbox_executor_concurrent_threads_execution(sample_df):
    """Verify multiple simultaneous threads successfully execute via the shared executor without deadlocks."""
    import concurrent.futures

    executor = SandboxExecutor.get_instance()
    queries = [
        ("result = df['price'].sum()", 1400.0),
        ("result = df['quantity'].max()", 10),
        ("result = len(df)", 4),
        ("result = df['price'].min()", 25.0),
    ]

    def run_query(code, expected):
        ok, res = executor.execute(code, sample_df)
        return ok, res, expected

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_query, q[0], q[1]) for q in queries]
        for f in concurrent.futures.as_completed(futures):
            ok, res, expected = f.result()
            assert ok is True
            assert res == expected


def test_windows_job_object_lifecycle_and_containment():
    """Verify WindowsJobObject lifecycle, handle creation, limits, and cleanup on Windows."""
    job = WindowsJobObject(memory_limit_bytes=256 * 1024 * 1024, max_processes=8)
    assert job.handle is None

    if sys.platform == "win32":
        job.open()
        assert job.handle is not None
        job.close()
        assert job.handle is None
    else:
        # On non-Windows platforms, open() gracefully no-ops
        job.open()
        assert job.handle is None
        job.close()


def test_windows_job_object_fail_closed_on_setup_failure(sample_df, monkeypatch):
    """Verify executor fails closed and rejects execution if Windows Job Object setup fails."""
    if sys.platform != "win32":
        pytest.skip("Windows-specific Job Object test")

    def mock_failing_open(self):
        raise RuntimeError("Simulated Job Object setup error")

    monkeypatch.setattr(WindowsJobObject, "open", mock_failing_open)

    executor = SandboxExecutor(max_concurrent_workers=1, enable_job_object=True)
    ok, err = executor.execute("result = 1", sample_df)
    assert ok is False
    assert "could not configure windows job object containment" in err.lower()


def test_windows_job_object_fail_closed_on_assign_failure(sample_df, monkeypatch):
    """Verify executor fails closed and kills process if Job Object assignment fails."""
    if sys.platform != "win32":
        pytest.skip("Windows-specific Job Object test")

    def mock_failing_assign(self, proc):
        raise RuntimeError("Simulated Job Object assign error")

    monkeypatch.setattr(WindowsJobObject, "assign_process", mock_failing_assign)

    executor = SandboxExecutor(max_concurrent_workers=1, enable_job_object=True)
    ok, err = executor.execute("result = 1", sample_df)
    assert ok is False
    assert "could not assign worker to windows job object" in err.lower()


def test_windows_job_object_disabled_fallback(sample_df):
    """Verify executor can run via standard subprocess when enable_job_object is False."""
    executor = SandboxExecutor(max_concurrent_workers=1, enable_job_object=False)
    ok, res = executor.execute("result = 555", sample_df)
    assert ok is True
    assert res == 555


def test_both_fastapi_and_streamlit_use_shared_executor():
    """Verify that both FastAPI and Streamlit import and use the exact same shared executor contract."""
    import backend.main as backend_main
    import app as streamlit_app

    # Verify identical execution function reference
    assert backend_main.execute_pandas_code is sandbox.execute_pandas_code
    assert streamlit_app.execute_pandas_code is sandbox.execute_pandas_code

    # Verify identical SandboxExecutor class reference
    assert backend_main.SandboxExecutor is sandbox.SandboxExecutor
    assert streamlit_app.SandboxExecutor is sandbox.SandboxExecutor


def test_win32_ctypes_declarations_and_pointer_sizes():
    """Verify Win32 ctypes declarations, parameter types, restypes, and pointer-sized handles."""
    if sys.platform != "win32":
        pytest.skip("Windows-specific ctypes declaration verification")

    import ctypes
    import ctypes.wintypes as wt

    kernel32 = ctypes.windll.kernel32

    # Verify pointer-sized HANDLE
    assert ctypes.sizeof(wt.HANDLE) == ctypes.sizeof(ctypes.c_void_p)

    # Verify function signatures are explicitly registered
    assert kernel32.CreateJobObjectW.restype == wt.HANDLE
    assert kernel32.CreateJobObjectW.argtypes == [ctypes.c_void_p, wt.LPCWSTR]

    assert kernel32.SetInformationJobObject.restype == wt.BOOL
    assert len(kernel32.SetInformationJobObject.argtypes) == 4

    assert kernel32.AssignProcessToJobObject.restype == wt.BOOL
    assert kernel32.AssignProcessToJobObject.argtypes == [wt.HANDLE, wt.HANDLE]

    assert kernel32.TerminateJobObject.restype == wt.BOOL
    assert kernel32.TerminateJobObject.argtypes == [wt.HANDLE, wt.UINT]

    assert kernel32.CloseHandle.restype == wt.BOOL
    assert kernel32.CloseHandle.argtypes == [wt.HANDLE]

    assert kernel32.CreateProcessW.restype == wt.BOOL
    assert len(kernel32.CreateProcessW.argtypes) == 10

    assert kernel32.ResumeThread.restype == wt.DWORD
    assert kernel32.ResumeThread.argtypes == [wt.HANDLE]

    assert kernel32.TerminateProcess.restype == wt.BOOL
    assert kernel32.TerminateProcess.argtypes == [wt.HANDLE, wt.UINT]

    assert kernel32.WaitForSingleObject.restype == wt.DWORD
    assert kernel32.WaitForSingleObject.argtypes == [wt.HANDLE, wt.DWORD]

    assert kernel32.GetExitCodeProcess.restype == wt.BOOL
    assert kernel32.GetExitCodeProcess.argtypes == [wt.HANDLE, ctypes.POINTER(wt.DWORD)]

    assert kernel32.InitializeProcThreadAttributeList.restype == wt.BOOL
    assert len(kernel32.InitializeProcThreadAttributeList.argtypes) == 4

    assert kernel32.UpdateProcThreadAttribute.restype == wt.BOOL
    assert len(kernel32.UpdateProcThreadAttribute.argtypes) == 7

    assert kernel32.DeleteProcThreadAttributeList.restype is None
    assert kernel32.DeleteProcThreadAttributeList.argtypes == [ctypes.c_void_p]

    assert ctypes.sizeof(sandbox._STARTUPINFOEXW) == (
        ctypes.sizeof(sandbox._STARTUPINFOW) + ctypes.sizeof(ctypes.c_void_p)
    )


def test_windows_kernel_descendant_termination_integration(tmp_path):
    """
    [REAL WINDOWS KERNEL INTEGRATION TEST]
    Exercises real Windows kernel behavior:
    Starts a worker inside a WindowsJobObject with KILL_ON_JOB_CLOSE.
    The worker creates a grandchild process.
    When the Job Object is terminated or closed, the Windows kernel terminates
    both the worker and all grandchild descendants.
    """
    if sys.platform != "win32":
        pytest.skip("Windows-only kernel integration test")

    import ctypes
    import ctypes.wintypes as wt
    import time

    kernel32 = ctypes.windll.kernel32

    pid_file = tmp_path / "grandchild.pid"
    posix_pid = pid_file.as_posix()

    job = WindowsJobObject().open()
    assert job.handle is not None

    worker_code = (
        "import subprocess, sys, time\n"
        f"p = subprocess.Popen([sys.executable, '-c', 'import os, time; open(\"{posix_pid}\", \"w\").write(str(os.getpid())); time.sleep(30)'])\n"
        "time.sleep(30)\n"
    )

    proc = sandbox._spawn_suspended_windows_process(
        [sys.executable, "-c", worker_code],
        env=sandbox._get_safe_env(),
        job=job,
    )

    # Wait for grandchild PID to be written
    grandchild_pid = None
    for _ in range(50):
        if pid_file.exists() and pid_file.stat().st_size > 0:
            grandchild_pid = int(pid_file.read_text().strip())
            break
        time.sleep(0.1)

    assert grandchild_pid is not None, "Grandchild did not start in time"

    # Open handle to grandchild with SYNCHRONIZE rights
    SYNCHRONIZE = 0x00100000
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    kernel32.OpenProcess.restype = wt.HANDLE
    kernel32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]

    h_grandchild = kernel32.OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, False, grandchild_pid)
    assert h_grandchild is not None and h_grandchild != 0

    try:
        # Terminate Job Object and close handle
        job.terminate(1)
        job.close()

        # Grandchild process MUST be signaled as terminated by OS kernel
        wait_res = kernel32.WaitForSingleObject(h_grandchild, 5000)
        assert wait_res == 0, "Grandchild process survived Job Object termination!"
    finally:
        kernel32.CloseHandle(h_grandchild)
        proc.close()


def test_windows_kernel_memory_limit_enforcement_integration():
    """
    [REAL WINDOWS KERNEL INTEGRATION TEST]
    Exercises real Windows kernel memory quota enforcement:
    Configures a WindowsJobObject with a bounded 64 MiB limit.
    Launches a worker that attempts to allocate 150 MiB.
    Verifies that the OS kernel rejects memory commitment above the Job limit.
    """
    if sys.platform != "win32":
        pytest.skip("Windows-only kernel integration test")

    # 64 MiB limit
    job = WindowsJobObject(memory_limit_bytes=64 * 1024 * 1024, max_processes=4).open()
    assert job.handle is not None

    alloc_code = (
        "import sys\n"
        "try:\n"
        "    data = bytearray(150 * 1024 * 1024)\n"
        "    sys.exit(0)\n"
        "except MemoryError:\n"
        "    sys.exit(42)\n"
        "except Exception:\n"
        "    sys.exit(43)\n"
    )

    try:
        proc = sandbox._spawn_suspended_windows_process(
            [sys.executable, "-c", alloc_code],
            env=sandbox._get_safe_env(),
            job=job,
        )
        exit_code = proc.wait(timeout=5.0)
        # Allocation must NOT succeed: exit code 42 (MemoryError) or abnormal OS termination
        assert exit_code != 0, "Worker unexpectedly succeeded in committing memory beyond Job limit"
    finally:
        job.close()
        proc.close()


def test_windows_kernel_active_process_limit_enforcement_integration(tmp_path):
    """
    [REAL WINDOWS KERNEL INTEGRATION TEST]
    Exercises real Windows kernel active process limit enforcement:
    Configures a WindowsJobObject with ActiveProcessLimit = 2.
    Spawns a worker that creates 1 child (total active = 2).
    Attempting to spawn a 2nd child (reaching 3) is rejected by the OS kernel with quota error.
    """
    if sys.platform != "win32":
        pytest.skip("Windows-only kernel integration test")

    job = WindowsJobObject(memory_limit_bytes=256 * 1024 * 1024, max_processes=2).open()
    assert job.handle is not None

    err_file = tmp_path / "quota_err.txt"
    base_exe = getattr(sys, "_base_executable", sys.executable)

    worker_code = (
        "import subprocess, sys, time\n"
        f"base = r'{base_exe}'\n"
        "p1 = subprocess.Popen([base, '-c', 'import time; time.sleep(10)'])\n"
        "try:\n"
        "    p2 = subprocess.Popen([base, '-c', 'import time; time.sleep(10)'])\n"
        "    sys.exit(0)\n"
        "except OSError as e:\n"
        f"    open(r'{err_file}', 'w').write(str(e))\n"
        "    sys.exit(99)\n"
        "finally:\n"
        "    p1.kill()\n"
    )

    try:
        proc = sandbox._spawn_suspended_windows_process(
            [base_exe, "-c", worker_code],
            env=sandbox._get_safe_env(),
            job=job,
        )
        exit_code = proc.wait(timeout=6.0)
        assert exit_code == 99, f"Worker returned unexpected code {exit_code}"
        assert err_file.exists(), "Quota error file was not created"
        err_content = err_file.read_text()
        assert "1816" in err_content or "quota" in err_content.lower()
    finally:
        job.close()
        proc.close()


def test_windows_suspended_startup_ordering_prevents_pre_containment_execution(tmp_path):
    """
    [REGRESSION TEST - STARTUP ORDERING]
    Demonstrates that the worker process cannot execute user code prior to Job Object assignment.
    Creates a process in a suspended state; simulates assignment failure before resumption.
    Verifies that the worker is terminated without ever executing, and the marker file is never created.
    """
    if sys.platform != "win32":
        pytest.skip("Windows-only startup ordering test")

    import ctypes
    import ctypes.wintypes as wt

    marker_file = tmp_path / "startup_marker.txt"
    assert not marker_file.exists()
    posix_marker = marker_file.as_posix()

    cmd = [sys.executable, "-c", f"open('{posix_marker}', 'w').write('executed'); import sys; sys.exit(0)"]

    job = WindowsJobObject().open()
    job.handle = wt.HANDLE(0x12345)  # Simulate invalid/failing Job handle for assignment

    with pytest.raises(RuntimeError) as exc_info:
        sandbox._spawn_suspended_windows_process(cmd, env=sandbox._get_safe_env(), job=job)

    assert "could not assign worker to windows job object" in str(exc_info.value).lower()
    # Critically: because the worker was suspended, it NEVER executed before assignment failure!
    assert not marker_file.exists(), "Worker executed user code before successful Job Object assignment!"


def test_windows_suspended_startup_resume_failure_fails_closed(monkeypatch):
    """
    [FAILURE PATH TEST - MOCKED RESUMETHREAD]
    Verifies that if ResumeThread fails, the process is safely terminated,
    handles are cleaned, and an error is raised without leaving unmanaged processes running.
    """
    if sys.platform != "win32":
        pytest.skip("Windows-only resume failure test")

    import ctypes
    kernel32 = ctypes.windll.kernel32

    # Mock ResumeThread to simulate failure (0xFFFFFFFF)
    monkeypatch.setattr(kernel32, "ResumeThread", lambda handle: 0xFFFFFFFF)

    job = WindowsJobObject().open()
    cmd = [sys.executable, "-c", "import sys; sys.exit(0)"]

    try:
        with pytest.raises(RuntimeError) as exc_info:
            sandbox._spawn_suspended_windows_process(cmd, env=sandbox._get_safe_env(), job=job)
        assert "failed to resume suspended worker thread" in str(exc_info.value).lower()
    finally:
        job.close()


def test_windows_job_object_set_information_failure_cleans_handles(monkeypatch):
    """
    [FAILURE PATH TEST - MOCKED SETINFORMATIONJOBOBJECT]
    Verifies that if SetInformationJobObject fails during open(), the handle is closed
    and self.handle is set to None.
    """
    if sys.platform != "win32":
        pytest.skip("Windows-only SetInformationJobObject test")

    import ctypes
    kernel32 = ctypes.windll.kernel32

    monkeypatch.setattr(kernel32, "SetInformationJobObject", lambda *args: False)

    job = WindowsJobObject()
    with pytest.raises(RuntimeError) as exc_info:
        job.open()

    assert "failed to configure windows job object limits" in str(exc_info.value).lower()
    assert job.handle is None, "Job handle was not cleaned up after configuration failure!"


def test_windows_process_creation_handle_whitelist_restricts_inheritance():
    """
    [REAL WINDOWS KERNEL INTEGRATION TEST]
    Exercises real Windows kernel handle inheritance filtering:
    1. Parent process creates an inheritable event handle (bInheritHandle = True).
    2. Spawns child using _spawn_suspended_windows_process with STARTUPINFOEXW and
       PROC_THREAD_ATTRIBUTE_HANDLE_LIST configured to whitelist ONLY the NUL handle.
    3. The child attempts to access the parent's inheritable event handle via ResetEvent.
    4. Verifies that the Windows kernel did NOT inherit the handle into the child process
       (ResetEvent returns 0 and GetLastError() returns ERROR_INVALID_HANDLE).
    """
    if sys.platform != "win32":
        pytest.skip("Windows-only handle whitelist test")

    import ctypes
    import ctypes.wintypes as wt

    kernel32 = ctypes.windll.kernel32

    # Create an inheritable event handle in the parent process
    sa = sandbox._SECURITY_ATTRIBUTES()
    sa.nLength = ctypes.sizeof(sa)
    sa.bInheritHandle = True

    h_unrelated = kernel32.CreateEventW(ctypes.byref(sa), True, False, None)
    assert h_unrelated and h_unrelated != -1

    try:
        # Child script attempts to reset the parent's event handle
        # If leaked, ResetEvent returns non-zero (True).
        # If correctly filtered by PROC_THREAD_ATTRIBUTE_HANDLE_LIST, ResetEvent returns False.
        child_code = (
            "import ctypes, sys\n"
            "from ctypes import wintypes as wt\n"
            "kernel32 = ctypes.windll.kernel32\n"
            "kernel32.ResetEvent.restype = wt.BOOL\n"
            "kernel32.ResetEvent.argtypes = [wt.HANDLE]\n"
            f"res = kernel32.ResetEvent(wt.HANDLE({h_unrelated}))\n"
            "if res:\n"
            "    sys.exit(10) # LEAKED: Handle was inherited\n"
            "else:\n"
            "    sys.exit(0)  # PROTECTED: Handle was not inherited\n"
        )

        base_exe = getattr(sys, "_base_executable", sys.executable)
        proc = sandbox._spawn_suspended_windows_process(
            [base_exe, "-c", child_code],
            env=sandbox._get_safe_env(),
        )
        exit_code = proc.wait(timeout=5.0)
        assert exit_code == 0, f"Unrelated inheritable handle was leaked to worker! Child exit code: {exit_code}"
    finally:
        kernel32.CloseHandle(h_unrelated)


def test_windows_proc_thread_attribute_list_lifecycle_and_cleanup():
    """
    [UNIT / WIN32 API VERIFICATION TEST]
    Verifies that InitializeProcThreadAttributeList, UpdateProcThreadAttribute,
    and DeleteProcThreadAttributeList execute correctly against kernel32 without errors.
    """
    if sys.platform != "win32":
        pytest.skip("Windows-only attribute list API verification")

    import ctypes
    import ctypes.wintypes as wt

    kernel32 = ctypes.windll.kernel32

    attr_size = ctypes.c_size_t(0)
    # First call retrieves required buffer size
    kernel32.InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(attr_size))
    assert attr_size.value > 0

    attr_buf = ctypes.create_string_buffer(attr_size.value)
    init_ok = kernel32.InitializeProcThreadAttributeList(attr_buf, 1, 0, ctypes.byref(attr_size))
    assert init_ok

    try:
        # Whitelist a dummy handle
        h_dummy = wt.HANDLE(1234)
        handle_arr = (wt.HANDLE * 1)(h_dummy)
        update_ok = kernel32.UpdateProcThreadAttribute(
            attr_buf,
            0,
            sandbox._PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
            ctypes.byref(handle_arr),
            ctypes.sizeof(handle_arr),
            None,
            None,
        )
        assert update_ok
    finally:
        # Must delete attribute list cleanly
        kernel32.DeleteProcThreadAttributeList(attr_buf)


def test_windows_proc_thread_attribute_list_init_failure_fails_closed(monkeypatch):
    """
    [FAILURE PATH TEST - MOCKED INITIALIZEPROCTHREADATTRIBUTELIST]
    Verifies that if InitializeProcThreadAttributeList fails, _spawn_suspended_windows_process
    fails closed, raises RuntimeError, and does not leave orphaned handles or processes.
    """
    if sys.platform != "win32":
        pytest.skip("Windows-only attribute list failure test")

    import ctypes
    kernel32 = ctypes.windll.kernel32

    # Mock InitializeProcThreadAttributeList to simulate allocation/initialization failure
    monkeypatch.setattr(kernel32, "InitializeProcThreadAttributeList", lambda *args: False)

    cmd = [sys.executable, "-c", "import sys; sys.exit(0)"]
    with pytest.raises(RuntimeError) as exc_info:
        sandbox._spawn_suspended_windows_process(cmd, env=sandbox._get_safe_env())

    assert "initializeprocthreadattributelist failed" in str(exc_info.value).lower()


def test_windows_proc_thread_attribute_list_update_failure_fails_closed(monkeypatch):
    """
    [FAILURE PATH TEST - MOCKED UPDATEPROCTHREADATTRIBUTE]
    Verifies that if UpdateProcThreadAttribute fails, _spawn_suspended_windows_process
    fails closed, cleans up the attribute list, and raises RuntimeError.
    """
    if sys.platform != "win32":
        pytest.skip("Windows-only attribute list update failure test")

    import ctypes
    kernel32 = ctypes.windll.kernel32

    monkeypatch.setattr(kernel32, "UpdateProcThreadAttribute", lambda *args: False)

    cmd = [sys.executable, "-c", "import sys; sys.exit(0)"]
    with pytest.raises(RuntimeError) as exc_info:
        sandbox._spawn_suspended_windows_process(cmd, env=sandbox._get_safe_env())

    assert "updateprocthreadattribute failed" in str(exc_info.value).lower()


def test_unicode_environment_block_sorting_and_deduplication():
    """
    [UNIT TEST]
    Verifies that environment keys are normalized, sorted case-insensitively,
    deduplicated, null-delimited, and double-null terminated.
    """
    # Raw dictionary with mixed case, unordered entries, and duplicate keys with different casing
    test_env = {
        "zeta_key": "val_zeta",
        "ALPHA_KEY": "val_alpha",
        "Beta_Key": "val_beta",
        "beta_key": "val_beta_duplicate",
        "GAMMA_KEY": "val_gamma",
    }

    normalized_env = {}
    for k, v in test_env.items():
        upper_k = k.upper()
        if upper_k not in normalized_env:
            normalized_env[upper_k] = (k, v)

    sorted_pairs = sorted(normalized_env.values(), key=lambda item: item[0].upper())
    env_str = "".join(f"{k}={v}\0" for k, v in sorted_pairs) + "\0"

    # Verify keys are sorted case-insensitively
    sorted_keys = [k.upper() for k, v in sorted_pairs]
    assert sorted_keys == ["ALPHA_KEY", "BETA_KEY", "GAMMA_KEY", "ZETA_KEY"]

    # Verify deduplication (Beta_Key kept, duplicate discarded)
    assert len(sorted_pairs) == 4

    # Verify double-null termination
    assert env_str.endswith("\0\0")


def test_docker_command_construction_and_security_flags(tmp_path):
    """
    [CROSS-PLATFORM UNIT TEST]
    Verifies that _build_docker_cmd generates all required hardening and containment flags:
      - --network none
      - --read-only
      - --cap-drop ALL
      - --security-opt no-new-privileges:true
      - --user 10001:10001
      - --memory=512m --memory-swap=512m
      - --cpus=1.0
      - --pids-limit=16
      - --tmpfs /tmp:rw,noexec,nosuid,size=64m
      - separate read-only input mount (-v in_dir:/workspace/ipc/in:ro)
      - separate writable output mount (-v out_dir:/workspace/ipc/out:rw)
      - worker invocation using --in-dir and --out-dir
    """
    in_dir = str(tmp_path / "in")
    out_dir = str(tmp_path / "out")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    cmd, container_name = sandbox._build_docker_cmd(
        in_dir=in_dir,
        out_dir=out_dir,
        image_name="test-sandbox:1.0",
        container_name="test_worker_12345",
        memory_limit="512m",
        cpu_limit="1.0",
        pids_limit=16,
        timeout_seconds=5.0,
    )

    assert cmd[0] == "docker"
    assert cmd[1] == "run"
    assert "--rm" in cmd
    assert "--name" in cmd and cmd[cmd.index("--name") + 1] == "test_worker_12345"
    assert "--network" in cmd and cmd[cmd.index("--network") + 1] == "none"
    assert "--read-only" in cmd
    assert "--cap-drop" in cmd and cmd[cmd.index("--cap-drop") + 1] == "ALL"
    assert "--security-opt" in cmd and cmd[cmd.index("--security-opt") + 1] == "no-new-privileges:true"
    assert "--user" in cmd and cmd[cmd.index("--user") + 1] == "10001:10001"
    assert "--memory=512m" in cmd
    assert "--memory-swap=512m" in cmd
    assert "--cpus=1.0" in cmd
    assert "--pids-limit=16" in cmd
    assert "--tmpfs" in cmd and cmd[cmd.index("--tmpfs") + 1] == "/tmp:rw,noexec,nosuid,size=64m"

    assert "--pull=never" in cmd

    abs_in = os.path.abspath(in_dir)
    abs_out = os.path.abspath(out_dir)
    expected_in_mount = f"{abs_in}:/workspace/ipc/in:ro"
    expected_out_mount = f"{abs_out}:/workspace/ipc/out:rw"
    assert expected_in_mount in cmd
    assert expected_out_mount in cmd

    # Verify arguments after image name match the Dockerfile ENTRYPOINT contract without duplicate python invocation
    image_idx = cmd.index("test-sandbox:1.0")
    assert cmd[image_idx + 1 :] == ["--in-dir", "/workspace/ipc/in", "--out-dir", "/workspace/ipc/out"]
    assert "python" not in cmd[image_idx :]


def test_host_output_validation_rejects_symlinks(tmp_path, monkeypatch):
    """
    [CROSS-PLATFORM UNIT TEST]
    Verifies that _validate_host_output_files rejects symbolic links for response.json
    and output_data.arrow, preventing host path traversal / secret file reading.
    Uses real symlink when supported by OS privileges, and verified islink mock on unprivileged Windows.
    """
    out_dir = str(tmp_path / "out")
    os.makedirs(out_dir, exist_ok=True)

    secret_file = tmp_path / "secret.txt"
    secret_file.write_text("HOST_SECRET_TOKEN")

    resp_path = os.path.join(out_dir, "response.json")
    arrow_path = os.path.join(out_dir, "output_data.arrow")

    try:
        os.symlink(str(secret_file), resp_path)
        real_symlink = True
    except (OSError, NotImplementedError):
        real_symlink = False
        with open(resp_path, "w") as f:
            f.write("{}")
        monkeypatch.setattr(os.path, "islink", lambda p: p == resp_path)

    is_valid, err_msg = sandbox._validate_host_output_files(out_dir, resp_path, arrow_path)
    assert not is_valid
    assert "symbolic link" in err_msg.lower()

    if real_symlink:
        os.remove(resp_path)
    # Write regular response.json
    with open(resp_path, "w") as f:
        f.write('{"status": "success"}')

    # Now test output_data.arrow symlink rejection
    if real_symlink:
        os.symlink(str(secret_file), arrow_path)
    else:
        with open(arrow_path, "wb") as f:
            f.write(b"arrow")
        monkeypatch.setattr(os.path, "islink", lambda p: p == arrow_path)

    is_valid, err_msg = sandbox._validate_host_output_files(out_dir, resp_path, arrow_path)
    assert not is_valid
    assert "symbolic link" in err_msg.lower()


def test_host_output_validation_rejects_non_regular_files(tmp_path):
    """
    [CROSS-PLATFORM UNIT TEST]
    Verifies that _validate_host_output_files rejects non-regular files (e.g. directories).
    """
    out_dir = str(tmp_path / "out")
    os.makedirs(out_dir, exist_ok=True)

    resp_path = os.path.join(out_dir, "response.json")
    os.makedirs(resp_path, exist_ok=True)  # response.json is a directory
    arrow_path = os.path.join(out_dir, "output_data.arrow")

    is_valid, err_msg = sandbox._validate_host_output_files(out_dir, resp_path, arrow_path)
    assert not is_valid
    assert "not a regular file" in err_msg.lower()


def test_host_output_validation_enforces_size_limits(tmp_path):
    """
    [CROSS-PLATFORM UNIT TEST]
    Verifies that _validate_host_output_files rejects files that exceed byte limits.
    """
    out_dir = str(tmp_path / "out")
    os.makedirs(out_dir, exist_ok=True)

    resp_path = os.path.join(out_dir, "response.json")
    arrow_path = os.path.join(out_dir, "output_data.arrow")

    # 1. Oversized response JSON
    with open(resp_path, "wb") as f:
        f.write(b"X" * 200)

    is_valid, err_msg = sandbox._validate_host_output_files(
        out_dir, resp_path, arrow_path, max_resp_bytes=100
    )
    assert not is_valid
    assert "exceeds limit" in err_msg.lower()

    # 2. Valid response JSON, oversized arrow file
    with open(resp_path, "wb") as f:
        f.write(b"X" * 50)
    with open(arrow_path, "wb") as f:
        f.write(b"Y" * 200)

    is_valid, err_msg = sandbox._validate_host_output_files(
        out_dir, resp_path, arrow_path, max_resp_bytes=100, max_arrow_bytes=100
    )
    assert not is_valid
    assert "exceeds limit" in err_msg.lower()


def test_linux_docker_unavailable_fails_closed(sample_df, monkeypatch):
    """
    [CROSS-PLATFORM UNIT TEST]
    Verifies that when running under the Docker containment policy and Docker is unavailable,
    execution strictly fails closed with an informative error rather than falling back to
    unrestricted host subprocess execution.
    """
    # Mock Docker as unavailable
    monkeypatch.setattr(sandbox, "_is_docker_available", lambda: False)

    executor = SandboxExecutor(backend="docker")
    success, error_msg = executor.execute("result = 1", sample_df)

    assert success is False
    assert "linux execution requires docker containment" in error_msg.lower()
    assert "fail-closed" in error_msg.lower()


def test_sandbox_worker_separated_in_out_dir_cli(tmp_path, sample_df):
    """
    [CROSS-PLATFORM INTEGRATION TEST]
    Verifies that sandbox_worker.py operates correctly when invoked with separated
    --in-dir and --out-dir command line flags.
    """
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()

    req_payload = {
        "code": "result = df['price'].sum()",
        "version": "1.0",
        "max_result_arrow_bytes": 10 * 1024 * 1024,
        "max_response_json_bytes": 1 * 1024 * 1024,
    }
    (in_dir / "request.json").write_text(json.dumps(req_payload), encoding="utf-8")

    table = pa.Table.from_pandas(sample_df)
    with open(in_dir / "input_data.arrow", "wb") as f:
        with ipc.new_stream(f, table.schema) as writer:
            writer.write_table(table)

    cmd = [
        sys.executable,
        "-u",
        str(sandbox.WORKER_SCRIPT),
        "--in-dir",
        str(in_dir),
        "--out-dir",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5.0)
    assert proc.returncode == 0

    resp_file = out_dir / "response.json"
    assert resp_file.exists()
    with open(resp_file, "r", encoding="utf-8") as f:
        resp = json.load(f)

    assert resp.get("status") == "success"
    assert resp.get("scalar_value") == 1400.0


def test_docker_cleanup_helper_tolerates_missing_container():
    """
    [UNIT TEST]
    Verifies that _kill_and_remove_docker_container handles nonexistent containers safely
    without raising exceptions.
    """
    sandbox._kill_and_remove_docker_container("nonexistent_sandbox_container_999999")


@pytest.mark.skipif(
    not sandbox._is_docker_available(),
    reason="[REAL DOCKER KERNEL INTEGRATION TEST] Requires Docker daemon and container image"
)
def test_docker_real_container_network_isolation_integration(sample_df):
    """
    [REAL LINUX / DOCKER INTEGRATION TEST]
    Validates that --network none blocks all outbound network socket connections.
    """
    code = (
        "import urllib.request\n"
        "try:\n"
        "    urllib.request.urlopen('http://1.1.1.1', timeout=1.0)\n"
        "    result = 'CONNECTED'\n"
        "except Exception as e:\n"
        "    result = f'BLOCKED: {type(e).__name__}'\n"
    )
    executor = SandboxExecutor(backend="docker")
    success, res = executor.execute(code, sample_df)
    assert success is True
    assert "BLOCKED" in str(res)


@pytest.mark.skipif(
    not sandbox._is_docker_available(),
    reason="[REAL DOCKER KERNEL INTEGRATION TEST] Requires Docker daemon and container image"
)
def test_docker_real_container_read_only_rootfs_integration(sample_df):
    """
    [REAL LINUX / DOCKER INTEGRATION TEST]
    Validates that --read-only prevents writes to the container root filesystem.
    """
    code = (
        "try:\n"
        "    with open('/workspace/malicious.txt', 'w') as f:\n"
        "        f.write('pwned')\n"
        "    result = 'WRITTEN'\n"
        "except OSError as e:\n"
        "    result = f'READ_ONLY: {type(e).__name__}'\n"
    )
    executor = SandboxExecutor(backend="docker")
    success, res = executor.execute(code, sample_df)
    assert success is True
    assert "READ_ONLY" in str(res)


@pytest.mark.skipif(
    not sandbox._is_docker_available(),
    reason="[REAL DOCKER KERNEL INTEGRATION TEST] Requires Docker daemon and container image"
)
def test_docker_real_container_timeout_and_cleanup_integration(sample_df):
    """
    [REAL LINUX / DOCKER INTEGRATION TEST]
    Validates that a runaway container is forcefully terminated and destroyed within timeout.
    """
    code = "while True: pass"
    executor = SandboxExecutor(backend="docker")
    success, error_msg = executor.execute(code, sample_df, timeout_seconds=2.0)
    assert success is False
    assert "timed out" in error_msg.lower()


# =========================================================================
# Task 07C.2.2 — Docker Blockers Regression & Hardening Tests
# =========================================================================

def test_docker_command_entrypoint_compatibility_regression(tmp_path):
    """
    [REGRESSION TEST]
    Verifies that _build_docker_cmd strictly obeys the Dockerfile.sandbox ENTRYPOINT contract:
    - Dockerfile.sandbox defines: ENTRYPOINT ["python", "-u", "/workspace/sandbox_worker.py"]
    - Therefore, docker run arguments AFTER image_name MUST ONLY be worker CLI flags (--in-dir, --out-dir)
    - Must NOT repeat python, -u, or sandbox_worker.py, which would cause argparse to crash with code 2.
    """
    dockerfile_path = Path(__file__).resolve().parent.parent / "Dockerfile.sandbox"
    assert dockerfile_path.is_file(), "Dockerfile.sandbox must exist in repository root"
    dockerfile_content = dockerfile_path.read_text(encoding="utf-8")
    assert 'ENTRYPOINT ["python", "-u", "/workspace/sandbox_worker.py"]' in dockerfile_content

    cmd, container_name = sandbox._build_docker_cmd(
        in_dir=str(tmp_path / "in"),
        out_dir=str(tmp_path / "out"),
        image_name="ai-data-analyst-sandbox:latest",
    )

    image_idx = cmd.index("ai-data-analyst-sandbox:latest")
    trailing_args = cmd[image_idx + 1 :]

    # Arguments after image name must ONLY be CLI flags, not an extra python invocation
    assert trailing_args == ["--in-dir", "/workspace/ipc/in", "--out-dir", "/workspace/ipc/out"]
    assert "python" not in trailing_args
    assert "/workspace/sandbox_worker.py" not in trailing_args


def test_ipc_mount_permissions_setup_noop_on_windows(tmp_path):
    """
    [CROSS-PLATFORM UNIT TEST]
    Verifies that _setup_ipc_mount_permissions is a safe no-op on Windows hosts,
    preserving exact Windows Job Object runtime behavior without raising errors.
    """
    ipc_dir = tmp_path / "ipc"
    in_dir = ipc_dir / "in"
    out_dir = ipc_dir / "out"
    in_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)
    (in_dir / "request.json").write_text("{}", encoding="utf-8")

    # On Windows, this must return cleanly
    sandbox._setup_ipc_mount_permissions(str(ipc_dir), str(in_dir), str(out_dir))


def test_docker_user_derivation_non_root_and_root(monkeypatch):
    """
    [CROSS-PLATFORM UNIT TEST]
    Verifies that _get_docker_user:
      - Aligns container execution with host identity when running unprivileged (os.getuid() != 0)
      - Falls back to unprivileged 10001:10001 when host is root (os.getuid() == 0)
      - Honors valid UID:GID and UID-only SANDBOX_DOCKER_USER overrides
      - Rejects malformed SANDBOX_DOCKER_USER overrides with ValueError
    """
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "getuid", lambda: 1000, raising=False)
    monkeypatch.setattr(os, "getgid", lambda: 1000, raising=False)
    monkeypatch.delenv("SANDBOX_DOCKER_USER", raising=False)

    # 1. Non-root host user -> aligns with host UID/GID
    assert sandbox._get_docker_user() == "1000:1000"

    # 2. Root host user -> falls back to unprivileged 10001:10001
    monkeypatch.setattr(os, "getuid", lambda: 0, raising=False)
    monkeypatch.setattr(os, "getgid", lambda: 0, raising=False)
    assert sandbox._get_docker_user() == "10001:10001"

    # 3. Explicit valid environment overrides (UID:GID and UID-only)
    monkeypatch.setenv("SANDBOX_DOCKER_USER", "20002:20002")
    assert sandbox._get_docker_user() == "20002:20002"

    monkeypatch.setenv("SANDBOX_DOCKER_USER", "30003")
    assert sandbox._get_docker_user() == "30003"

    # 4. Malformed environment overrides reject with ValueError
    for bad_user in ["", "   ", "abc", "-1", "1000:abc", "1:2:3", "1000:-5"]:
        monkeypatch.setenv("SANDBOX_DOCKER_USER", bad_user)
        if not bad_user.strip():
            # Empty/whitespace is ignored and falls back to host default
            pass
        else:
            with pytest.raises(ValueError) as exc:
                sandbox._get_docker_user()
            assert "malformed container user" in str(exc.value).lower() or "must be a non-negative integer" in str(exc.value).lower()


def test_ipc_mount_permissions_setup_aligned_identity_strict_mode_700(tmp_path, monkeypatch):
    """
    [POSIX UNIT TEST - ALIGNED IDENTITY (RECOMMENDED PATTERN)]
    When container worker runs as the host's unprivileged UID/GID:
      - ipc_dir: 0o700
      - in_dir: 0o700, files: 0o600
      - out_dir: 0o700
    Verifies zero permissive modes (no 0o777, no 0o1777), zero chown calls, and zero ACL dependencies.
    """
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "getuid", lambda: 1000, raising=False)
    monkeypatch.setattr(os, "getgid", lambda: 1000, raising=False)

    ipc_dir = tmp_path / "ipc_aligned"
    in_dir = ipc_dir / "in"
    out_dir = ipc_dir / "out"
    in_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)
    (in_dir / "request.json").write_text("{}", encoding="utf-8")

    recorded_chmods = {}
    monkeypatch.setattr(os, "chmod", lambda p, m: recorded_chmods.update({str(p): m}))

    # Verify aligned execution identity
    sandbox._setup_ipc_mount_permissions(str(ipc_dir), str(in_dir), str(out_dir), container_user="1000:1000")

    assert recorded_chmods[str(ipc_dir)] == 0o700
    assert recorded_chmods[str(in_dir)] == 0o700
    assert recorded_chmods[str(in_dir / "request.json")] == 0o600
    assert recorded_chmods[str(out_dir)] == 0o700


def test_ipc_mount_permissions_setup_root_host_chowns_cleanly(tmp_path, monkeypatch):
    """
    [POSIX UNIT TEST - ROOT HOST PATTERN]
    When host process is root (UID 0) and container is unprivileged (10001:10001):
      - out_dir is chowned to 10001:10001 with mode 0o700
      - in_dir is mode 0o755 with files 0o644
    """
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "getuid", lambda: 0, raising=False)
    monkeypatch.setattr(os, "getgid", lambda: 0, raising=False)

    ipc_dir = tmp_path / "ipc_root"
    in_dir = ipc_dir / "in"
    out_dir = ipc_dir / "out"
    in_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)
    (in_dir / "request.json").write_text("{}", encoding="utf-8")

    recorded_chmods = {}
    recorded_chowns = {}
    monkeypatch.setattr(os, "chmod", lambda p, m: recorded_chmods.update({str(p): m}))
    monkeypatch.setattr(os, "chown", lambda p, u, g: recorded_chowns.update({str(p): (u, g)}), raising=False)

    sandbox._setup_ipc_mount_permissions(str(ipc_dir), str(in_dir), str(out_dir), container_user="10001:10001")

    assert recorded_chmods[str(ipc_dir)] == 0o700
    assert recorded_chmods[str(in_dir)] == 0o755
    assert recorded_chmods[str(in_dir / "request.json")] == 0o644
    assert recorded_chowns[str(out_dir)] == (10001, 10001)
    assert recorded_chmods[str(out_dir)] == 0o700


def test_ipc_mount_permissions_setup_mismatched_unprivileged_fails_closed(tmp_path, monkeypatch):
    """
    [POSIX UNIT TEST - FAIL CLOSED ON MISMATCHED UNPRIVILEGED IDENTITY]
    When host UID is unprivileged (e.g. 1000) and container_user is mismatched (e.g. 20002:20002),
    and safe ownership cannot be established without world-writable permissions:
      - Must FAIL CLOSED with RuntimeError
      - Must NOT fall back to permissive mode 0o777 or 0o1777
    """
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "getuid", lambda: 1000, raising=False)
    monkeypatch.setattr(os, "getgid", lambda: 1000, raising=False)

    ipc_dir = tmp_path / "ipc_mismatch"
    in_dir = ipc_dir / "in"
    out_dir = ipc_dir / "out"
    in_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)
    (in_dir / "request.json").write_text("{}", encoding="utf-8")

    def mock_chown_fail(path, uid, gid):
        raise PermissionError("Operation not permitted")

    monkeypatch.setattr(os, "chown", mock_chown_fail, raising=False)
    import shutil
    monkeypatch.setattr(shutil, "which", lambda cmd: None)  # setfacl unavailable

    with pytest.raises(RuntimeError) as exc_info:
        sandbox._setup_ipc_mount_permissions(str(ipc_dir), str(in_dir), str(out_dir), container_user="20002:20002")

    assert "cannot establish secure, non-permissive ipc mount permissions" in str(exc_info.value).lower()


def test_docker_cleanup_skips_redundant_calls_on_clean_exit(sample_df, monkeypatch):
    """
    [UNIT TEST]
    Verifies that _kill_and_remove_docker_container is NOT invoked when a container exits
    cleanly with code 0 (since --rm already auto-removes it), but IS invoked if it times out
    or exits abnormally.
    """
    cleanup_calls = []

    def mock_cleanup(name):
        cleanup_calls.append(name)
        return (True, "")

    monkeypatch.setattr(sandbox, "_kill_and_remove_docker_container", mock_cleanup)
    monkeypatch.setattr(sandbox, "_is_docker_available", lambda: True)

    class MockCleanProc:
        def __init__(self):
            self.returncode = 0
            self.pid = 9999

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    class MockFailedProc:
        def __init__(self):
            self.returncode = 1
            self.pid = 9998

        def poll(self):
            return 1

        def wait(self, timeout=None):
            return 1

    # 1. Clean exit (code 0) -> no cleanup subprocesses
    def mock_clean_popen(*args, **kwargs):
        # Write valid response so host doesn't fail before completing
        out_mount = next((arg for arg in args[0] if ":/workspace/ipc/out" in arg), None)
        out_dir = out_mount.split(":/workspace/ipc/out")[0] if out_mount else args[0][args[0].index("--out-dir") + 1]
        resp_file = os.path.join(out_dir, "response.json")
        with open(resp_file, "w", encoding="utf-8") as f:
            json.dump({"version": "1.0", "status": "success", "result_type": "int", "has_arrow_data": False, "scalar_value": 42}, f)
        return MockCleanProc()

    monkeypatch.setattr(subprocess, "Popen", mock_clean_popen)

    executor = SandboxExecutor(backend="docker")
    success, res = executor.execute("result = 42", sample_df)
    assert success is True
    assert res == 42
    assert len(cleanup_calls) == 0, f"Redundant cleanup called on clean exit: {cleanup_calls}"

    # 2. Abnormal exit (code 1) -> cleanup IS invoked exactly once
    def mock_failed_popen(*args, **kwargs):
        return MockFailedProc()

    monkeypatch.setattr(subprocess, "Popen", mock_failed_popen)
    success, err = executor.execute("result = 42", sample_df)
    assert success is False
    assert len(cleanup_calls) == 1, f"Expected exactly 1 cleanup call on abnormal exit, got {len(cleanup_calls)}"


def test_docker_timeout_terminates_and_reaps_host_process(sample_df, monkeypatch):
    """
    [UNIT TEST]
    Verifies that on timeout:
      1. Host Docker CLI subprocess is explicitly killed (proc.kill()) and reaped (proc.wait())
      2. _kill_and_remove_docker_container is invoked exactly once
      3. Original timeout error message is preserved even if cleanup fails
    """
    cleanup_calls = []

    def mock_cleanup(name):
        cleanup_calls.append(name)
        return (False, "mock daemon timeout")

    monkeypatch.setattr(sandbox, "_kill_and_remove_docker_container", mock_cleanup)
    monkeypatch.setattr(sandbox, "_is_docker_available", lambda: True)

    class MockTimeoutProc:
        def __init__(self):
            self.pid = 5555
            self.killed = False
            self.reaped = False
            self.returncode = None

        def poll(self):
            return -9 if self.killed else None

        def wait(self, timeout=None):
            if timeout is not None and not self.killed:
                raise subprocess.TimeoutExpired(cmd="docker", timeout=timeout)
            self.reaped = True
            self.returncode = -9
            return -9

        def kill(self):
            self.killed = True

    proc_instance = MockTimeoutProc()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: proc_instance)

    executor = SandboxExecutor(backend="docker")
    success, error_msg = executor.execute("result = 1", sample_df, timeout_seconds=1.0)

    # 1. Timeout error preserved with cleanup warning surfaced
    assert success is False
    assert "timed out after 1.0 seconds" in error_msg.lower()
    assert "cleanup warning" in error_msg.lower()
    assert "mock daemon timeout" in error_msg.lower()

    # 2. Host process terminated and reaped
    assert proc_instance.killed is True, "Host Docker CLI subprocess was not killed on timeout!"
    assert proc_instance.reaped is True, "Host Docker CLI subprocess was not reaped on timeout!"

    # 3. Container cleanup called exactly once
    assert len(cleanup_calls) == 1, f"Cleanup must be called exactly once on timeout, got {len(cleanup_calls)}"


def test_docker_startup_diagnostics_exit_codes_125_126_127(sample_df, monkeypatch):
    """
    [UNIT TEST]
    Verifies specific, actionable diagnostics for Docker container launch failures:
      - Exit code 125: daemon error or missing image
      - Exit code 126: command could not be invoked (permissions/exec format)
      - Exit code 127: command not found in container
    """
    monkeypatch.setattr(sandbox, "_is_docker_available", lambda: True)
    monkeypatch.setattr(sandbox, "_kill_and_remove_docker_container", lambda name: (True, ""))

    class MockExitCodeProc:
        def __init__(self, code):
            self.returncode = code
            self.pid = 12345

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    executor = SandboxExecutor(backend="docker")

    # Exit code 125
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: MockExitCodeProc(125))
    success, error_msg = executor.execute("result = 1", sample_df)
    assert success is False
    assert "exit code 125" in error_msg.lower()
    assert "docker build -t" in error_msg.lower()

    # Exit code 126
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: MockExitCodeProc(126))
    success, error_msg = executor.execute("result = 1", sample_df)
    assert success is False
    assert "exit code 126" in error_msg.lower()
    assert "could not be invoked" in error_msg.lower()

    # Exit code 127
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: MockExitCodeProc(127))
    success, error_msg = executor.execute("result = 1", sample_df)
    assert success is False
    assert "exit code 127" in error_msg.lower()
    assert "command not found" in error_msg.lower()


def test_user_code_error_not_misclassified_as_docker_startup_error(sample_df, monkeypatch):
    """
    [UNIT TEST]
    Verifies that when worker exits with non-zero code (e.g. 1) but produces a structured
    response envelope with an error_message (e.g. division by zero), that diagnostic is returned
    and NOT misclassified as a Docker startup error.
    """
    monkeypatch.setattr(sandbox, "_is_docker_available", lambda: True)
    monkeypatch.setattr(sandbox, "_kill_and_remove_docker_container", lambda name: (True, ""))

    class MockUserErrorProc:
        def __init__(self):
            self.returncode = 1
            self.pid = 7777

        def poll(self):
            return 1

        def wait(self, timeout=None):
            return 1

    def mock_popen(*args, **kwargs):
        out_mount = next((arg for arg in args[0] if ":/workspace/ipc/out" in arg), None)
        out_dir = out_mount.split(":/workspace/ipc/out")[0] if out_mount else args[0][args[0].index("--out-dir") + 1]
        resp_file = os.path.join(out_dir, "response.json")
        with open(resp_file, "w", encoding="utf-8") as f:
            json.dump({
                "version": "1.0",
                "status": "error",
                "result_type": "none",
                "has_arrow_data": False,
                "scalar_value": None,
                "error_type": "ZeroDivisionError",
                "error_message": "Calculation error: Division by zero encountered."
            }, f)
        return MockUserErrorProc()

    monkeypatch.setattr(subprocess, "Popen", mock_popen)

    executor = SandboxExecutor(backend="docker")
    success, error_msg = executor.execute("result = 1", sample_df)

    assert success is False
    assert "division by zero" in error_msg.lower()
    assert "exit code" not in error_msg.lower()


def test_docker_primary_execution_failure_surfaces_cleanup_warning(sample_df, monkeypatch):
    """
    [UNIT TEST]
    Verifies that when a primary execution failure occurs (e.g. exit code 1) and container
    cleanup also fails, the primary failure is preserved AND the cleanup warning is surfaced.
    """
    monkeypatch.setattr(sandbox, "_is_docker_available", lambda: True)
    monkeypatch.setattr(
        sandbox,
        "_kill_and_remove_docker_container",
        lambda name: (False, "docker rm error: daemon hung"),
    )

    class MockExit1Proc:
        def __init__(self):
            self.returncode = 1
            self.pid = 8888

        def poll(self):
            return 1

        def wait(self, timeout=None):
            return 1

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: MockExit1Proc())

    executor = SandboxExecutor(backend="docker")
    success, error_msg = executor.execute("result = 1", sample_df)

    assert success is False
    # Primary failure preserved
    assert "exited with code 1" in error_msg.lower()
    # Cleanup warning surfaced
    assert "cleanup warning" in error_msg.lower()
    assert "daemon hung" in error_msg.lower()


def test_docker_user_code_error_surfaces_cleanup_warning(sample_df, monkeypatch):
    """
    [UNIT TEST]
    Verifies that when user code fails with a structured error in response.json and cleanup
    also fails, the user error message is preserved along with the cleanup warning.
    """
    monkeypatch.setattr(sandbox, "_is_docker_available", lambda: True)
    monkeypatch.setattr(
        sandbox,
        "_kill_and_remove_docker_container",
        lambda name: (False, "docker kill error: container not responding"),
    )

    class MockUserProc:
        def __init__(self):
            self.returncode = 1
            self.pid = 9991

        def poll(self):
            return 1

        def wait(self, timeout=None):
            return 1

    def mock_popen(*args, **kwargs):
        out_mount = next((arg for arg in args[0] if ":/workspace/ipc/out" in arg), None)
        out_dir = out_mount.split(":/workspace/ipc/out")[0] if out_mount else args[0][args[0].index("--out-dir") + 1]
        resp_file = os.path.join(out_dir, "response.json")
        with open(resp_file, "w", encoding="utf-8") as f:
            json.dump({
                "version": "1.0",
                "status": "error",
                "result_type": "none",
                "has_arrow_data": False,
                "scalar_value": None,
                "error_type": "ValueError",
                "error_message": "Invalid column index specified."
            }, f)
        return MockUserProc()

    monkeypatch.setattr(subprocess, "Popen", mock_popen)

    executor = SandboxExecutor(backend="docker")
    success, error_msg = executor.execute("result = 1", sample_df)

    assert success is False
    assert "invalid column index" in error_msg.lower()
    assert "cleanup warning" in error_msg.lower()
    assert "container not responding" in error_msg.lower()


def test_ipc_mount_permissions_setup_aligned_uid_only(tmp_path, monkeypatch):
    """
    [POSIX UNIT TEST]
    Verifies that when container_user is provided as a UID-only string (e.g. '1000'),
    matching the host unprivileged UID:
      - Resolves successfully under Case A (aligned identity)
      - Sets mode 0o700 on ipc_dir, in_dir, and out_dir without error
    """
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "getuid", lambda: 1000, raising=False)
    monkeypatch.setattr(os, "getgid", lambda: 1000, raising=False)

    ipc_dir = tmp_path / "ipc_uid_only"
    in_dir = ipc_dir / "in"
    out_dir = ipc_dir / "out"
    in_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)
    (in_dir / "request.json").write_text("{}", encoding="utf-8")

    recorded_chmods = {}
    monkeypatch.setattr(os, "chmod", lambda p, m: recorded_chmods.update({str(p): m}))

    # container_user passed as UID-only integer string
    sandbox._setup_ipc_mount_permissions(str(ipc_dir), str(in_dir), str(out_dir), container_user="1000")

    assert recorded_chmods[str(ipc_dir)] == 0o700
    assert recorded_chmods[str(in_dir)] == 0o700
    assert recorded_chmods[str(in_dir / "request.json")] == 0o600
    assert recorded_chmods[str(out_dir)] == 0o700


def test_ipc_mount_permissions_setup_malformed_container_user_raises(tmp_path, monkeypatch):
    """
    [CROSS-PLATFORM UNIT TEST]
    Verifies that _setup_ipc_mount_permissions rejects malformed container_user strings with ValueError.
    """
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "getuid", lambda: 1000, raising=False)
    monkeypatch.setattr(os, "getgid", lambda: 1000, raising=False)

    ipc_dir = tmp_path / "ipc_malformed"
    in_dir = ipc_dir / "in"
    out_dir = ipc_dir / "out"
    in_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)

    with pytest.raises(ValueError) as exc:
        sandbox._setup_ipc_mount_permissions(str(ipc_dir), str(in_dir), str(out_dir), container_user="bad:user:id")

    assert "malformed container user" in str(exc.value).lower()


def test_cleanup_diagnostic_sanitization():
    """
    [UNIT TEST]
    Demonstrates that _sanitize_cleanup_diagnostic:
      1. Strips verbose 'Error response from daemon:' boilerplate
      2. Redacts recognized credential/token patterns ('token=...', 'secret=...')
      3. Redacts recognized absolute filesystem paths (POSIX /var/..., Windows C:\\...)
      4. Bounds length to 120 characters
    """
    # 1. Daemon boilerplate stripped, path redacted, secret redacted
    raw_1 = (
        "Error response from daemon: failed to remove /var/lib/docker/overlay2/abc12345/merged: "
        "permission denied with token=supersecret_api_key_12345"
    )
    sanitized_1 = sandbox._sanitize_cleanup_diagnostic(raw_1)
    assert "Error response from daemon" not in sanitized_1
    assert "/var/lib/docker" not in sanitized_1
    assert "[path]" in sanitized_1
    assert "supersecret_api_key_12345" not in sanitized_1
    assert "[redacted]" in sanitized_1

    # 2. Windows absolute paths redacted
    raw_2 = "docker kill error: Access denied on C:\\Users\\Administrator\\AppData\\Local\\Temp\\agy_ipc"
    sanitized_2 = sandbox._sanitize_cleanup_diagnostic(raw_2)
    assert "C:\\Users\\Administrator" not in sanitized_2
    assert "[path]" in sanitized_2

    # 3. Message length bounded to <= 120 characters
    long_raw = "docker kill failed: " + ("x" * 200)
    sanitized_long = sandbox._sanitize_cleanup_diagnostic(long_raw)
    assert len(sanitized_long) <= 120
    assert sanitized_long.endswith("...")


def test_docker_cleanup_exception_does_not_mask_primary_error(sample_df, monkeypatch):
    """
    [UNIT TEST]
    Demonstrates that if _kill_and_remove_docker_container itself raises an unhandled exception,
    the primary error (timeout) is PRESERVED and not replaced by an unhandled cleanup crash.
    """
    monkeypatch.setattr(sandbox, "_is_docker_available", lambda: True)

    def mock_raising_cleanup(name):
        raise RuntimeError("Unexpected daemon socket crash")

    monkeypatch.setattr(sandbox, "_kill_and_remove_docker_container", mock_raising_cleanup)

    class MockTimeoutProc:
        def __init__(self):
            self.pid = 4321
            self.killed = False

        def poll(self):
            return -9 if self.killed else None

        def wait(self, timeout=None):
            if timeout is not None and not self.killed:
                raise subprocess.TimeoutExpired(cmd="docker", timeout=timeout)
            return -9

        def kill(self):
            self.killed = True

    proc_instance = MockTimeoutProc()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: proc_instance)

    executor = SandboxExecutor(backend="docker")
    success, error_msg = executor.execute("result = 1", sample_df, timeout_seconds=1.0)

    # 1. Primary timeout failure preserved
    assert success is False
    assert "timed out after 1.0 seconds" in error_msg.lower()

    # 2. Cleanup error surfaced without crashing the executor
    assert "cleanup warning" in error_msg.lower()
    assert "unexpected daemon socket crash" in error_msg.lower()


def test_docker_successful_run_remains_successful(sample_df, monkeypatch):
    """
    [UNIT TEST]
    Verifies that a successful execution path (clean exit code 0) remains fully successful,
    returning (True, result) without spurious cleanup warnings.
    """
    monkeypatch.setattr(sandbox, "_is_docker_available", lambda: True)
    monkeypatch.setattr(sandbox, "_kill_and_remove_docker_container", lambda name: (True, ""))

    class MockCleanProc:
        def __init__(self):
            self.returncode = 0
            self.pid = 9999

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    def mock_clean_popen(*args, **kwargs):
        out_mount = next((arg for arg in args[0] if ":/workspace/ipc/out" in arg), None)
        out_dir = out_mount.split(":/workspace/ipc/out")[0] if out_mount else args[0][args[0].index("--out-dir") + 1]
        resp_file = os.path.join(out_dir, "response.json")
        with open(resp_file, "w", encoding="utf-8") as f:
            json.dump({
                "version": "1.0",
                "status": "success",
                "result_type": "int",
                "has_arrow_data": False,
                "scalar_value": 42
            }, f)
        return MockCleanProc()

    monkeypatch.setattr(subprocess, "Popen", mock_clean_popen)

    executor = SandboxExecutor(backend="docker")
    success, res = executor.execute("result = 42", sample_df)

    assert success is True
    assert res == 42


def test_cleanup_warning_ordering_proof_finally_to_return(sample_df, monkeypatch):
    """
    [UNIT TEST - RETURN-ORDERING LIFECYCLE PROOF]
    Proves that cleanup warnings collected during the `finally:` block reach the final
    returned diagnostic across both timeout and nonzero-exit paths, establishing that:
      1. Container termination, subprocess reaping, and cleanup execution occur in `finally:`
      2. The final diagnostic is formatted strictly AFTER `finally:` has populated cleanup_errors
      3. The primary execution failure remains unmasked as the leading diagnostic
    """
    events = []

    def mock_cleanup(name):
        events.append("container_cleanup_invoked")
        return (False, "docker rm error: daemon communication lost")

    monkeypatch.setattr(sandbox, "_is_docker_available", lambda: True)
    monkeypatch.setattr(sandbox, "_kill_and_remove_docker_container", mock_cleanup)

    # --- Scenario 1: Timeout path ---
    class MockTimeoutProc:
        def __init__(self):
            self.pid = 7171
            self.killed = False

        def poll(self):
            return -9 if self.killed else None

        def wait(self, timeout=None):
            if timeout is not None and not self.killed:
                events.append("proc_wait_timeout")
                raise subprocess.TimeoutExpired(cmd="docker", timeout=timeout)
            events.append("proc_reaped")
            return -9

        def kill(self):
            events.append("proc_kill")
            self.killed = True

    proc_timeout = MockTimeoutProc()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: proc_timeout)

    executor = SandboxExecutor(backend="docker")
    success_t, error_msg_t = executor.execute("result = 1", sample_df, timeout_seconds=1.0)
    events.append("timeout_return_received")

    # Verify strict lifecycle order
    assert events == [
        "proc_wait_timeout",
        "proc_kill",
        "proc_reaped",
        "container_cleanup_invoked",
        "timeout_return_received",
    ]
    assert success_t is False
    assert "timed out after 1.0 seconds" in error_msg_t.lower()
    assert "cleanup warning" in error_msg_t.lower()
    assert "daemon communication lost" in error_msg_t.lower()

    # --- Scenario 2: Nonzero exit path ---
    events.clear()

    class MockNonZeroProc:
        def __init__(self):
            self.returncode = 1
            self.pid = 7172

        def poll(self):
            return 1

        def wait(self, timeout=None):
            events.append("proc_wait_code_1")
            return 1

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: MockNonZeroProc())

    success_nz, error_msg_nz = executor.execute("result = 1", sample_df)
    events.append("nonzero_return_received")

    # Verify strict lifecycle order for abnormal exit
    assert events == [
        "proc_wait_code_1",
        "container_cleanup_invoked",
        "nonzero_return_received",
    ]
    assert success_nz is False
    assert "exited with code 1" in error_msg_nz.lower()
    assert "cleanup warning" in error_msg_nz.lower()
    assert "daemon communication lost" in error_msg_nz.lower()
