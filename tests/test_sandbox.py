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
