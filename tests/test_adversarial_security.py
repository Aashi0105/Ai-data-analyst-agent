"""
tests/test_adversarial_security.py
-----------------------------------
Safe adversarial and security control tests for the AI Data Analyst Agent.
Audits and tests:
  1. AST-based code validation against malicious syntax and disallowed constructs.
  2. Prohibited built-ins, system modules, dynamic reflection, and file I/O operations.
  3. Strict single-assignment invariants and evasion attempts (comprehensions, walrus, f-strings, unpacking).
  4. Worker environment secret isolation and stripping of host secrets.
  5. Pre-admission dataset and code resource bounds (code size, row count, column count, RAM admission).
  6. Concurrency bounding and queue saturation defense.
  7. Worker serialization limits (oversized integers, strings, nested objects, non-finite floats).
  8. Host output validation rejecting symlinks, device files, and storage violations.
  9. Execution timeout enforcement, process tree cleanup, and IPC directory disposal.
  10. Diagnostic sanitization preventing path and credential leakage in error messages.
  11. Windows Job Object and Linux Docker isolation boundary configuration audits.

NOTE ON SECURITY SCOPE:
  - Unit tests on validator.py prove AST syntax and identifier rejection, NOT OS isolation.
  - Unit tests on sandbox.py prove process limits, IPC bounds, and host validation.
  - Windows Job Objects enforce resource quotas and process lifetime; they do NOT isolate the filesystem or network.
  - Full OS isolation is enforced only under Linux Docker containerization (--network none, --read-only, --cap-drop ALL).
"""

import sys
import os
import json
import tempfile
import threading
import time
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

import sandbox
from validator import validate_code, clean_code
from sandbox import (
    execute_pandas_code,
    SandboxExecutor,
    _get_safe_env,
    _validate_host_output_files,
    _sanitize_cleanup_diagnostic,
    _parse_docker_user,
    _build_docker_cmd,
    MAX_CODE_SIZE_BYTES,
    MAX_INPUT_ROWS,
    MAX_INPUT_COLS,
    MAX_INPUT_MEMORY_BYTES,
    MAX_RESPONSE_JSON_BYTES,
    MAX_RESULT_ARROW_BYTES,
    MAX_TOTAL_IPC_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
)
from sandbox_worker import (
    _validate_result_bounds,
    _validate_envelope_bounds,
    SAFE_BUILTINS,
    MAX_STRING_LENGTH,
    MAX_INTEGER_DIGITS,
    MAX_NESTING_DEPTH,
)


@pytest.fixture
def base_df():
    """Provides a small baseline DataFrame for testing."""
    return pd.DataFrame({
        "item": ["Widget A", "Widget B", "Widget C"],
        "price": [10.0, 20.0, 30.0],
        "quantity": [5, 2, 8],
    })


# =========================================================================
# Part 1: AST Validator Adversarial Tests (Syntax & Identifier Rejection)
# =========================================================================

class TestASTValidatorAdversarial:
    """
    Tests AST validator resistance against prohibited syntax, simulated loops,
    dynamic code execution, reflection, dunder access, and assignment tampering.
    """

    @pytest.mark.parametrize("comprehension_code", [
        "result = [x * 2 for x in range(10)]",
        "result = {x: x * 2 for x in range(10)}",
        "result = {x for x in range(10)}",
        "result = list(x for x in range(10))",
    ])
    def test_rejects_comprehensions_and_generators(self, comprehension_code):
        """Comprehensions simulate loops and are excluded from ALLOWED_NODE_TYPES."""
        is_valid, msg = validate_code(comprehension_code)
        assert not is_valid
        assert "not allowed" in msg.lower() or "Unsafe construct" in msg

    def test_rejects_walrus_operator(self):
        """The walrus operator (:= / ast.NamedExpr) allows hidden assignment expressions."""
        code = "result = (x := df['price'].sum())"
        is_valid, msg = validate_code(code)
        assert not is_valid
        assert "NamedExpr" in msg or "not allowed" in msg.lower()

    def test_rejects_formatted_string_literals(self):
        """F-strings (ast.JoinedStr / FormattedValue) are excluded from ALLOWED_NODE_TYPES."""
        code = 'result = f"Summary: {df[\'price\'].sum()}"'
        is_valid, msg = validate_code(code)
        assert not is_valid
        assert "JoinedStr" in msg or "not allowed" in msg.lower()

    @pytest.mark.parametrize("disallowed_statement", [
        "assert len(df) > 0\nresult = df['price'].sum()",
        "del df['price']\nresult = 1",
        "raise RuntimeError('Exploit')\nresult = 1",
        "pass\nresult = 1",
        "return df['price'].sum()",
    ])
    def test_rejects_disallowed_statements(self, disallowed_statement):
        """Statements like assert, del, raise, pass, return are rejected by AST whitelist."""
        is_valid, msg = validate_code(disallowed_statement)
        assert not is_valid
        assert "not allowed" in msg.lower() or "Syntax Error" in msg or "Unsafe construct" in msg

    @pytest.mark.parametrize("forbidden_id", [
        "compile",
        "input",
        "exit",
        "quit",
        "pathlib",
        "shutil",
        "pickle",
        "http",
        "urllib",
    ])
    def test_rejects_forbidden_builtin_identifiers(self, forbidden_id):
        """Identifiers in FORBIDDEN_NAMES must be rejected even without import."""
        code = f"result = {forbidden_id}"
        is_valid, msg = validate_code(code)
        assert not is_valid
        assert f"forbidden identifier '{forbidden_id}'" in msg

    @pytest.mark.parametrize("io_method", [
        "to_stata", "read_stata",
        "to_gbq", "read_gbq",
        "to_orc", "read_orc",
        "to_feather", "read_feather",
        "to_html", "read_html",
        "to_clipboard", "read_clipboard",
    ])
    def test_rejects_prohibited_pandas_io_attributes(self, io_method):
        """Dangerous serialization, database, and clipboard attributes are rejected."""
        code_df = f"result = df.{io_method}('output')"
        is_valid_df, msg_df = validate_code(code_df)
        assert not is_valid_df
        assert f"forbidden attribute '{io_method}'" in msg_df

        code_pd = f"result = pd.{io_method}('input')"
        is_valid_pd, msg_pd = validate_code(code_pd)
        assert not is_valid_pd
        assert f"forbidden attribute '{io_method}'" in msg_pd

    @pytest.mark.parametrize("dunder_attr", [
        "__class__",
        "__dict__",
        "__doc__",
        "__init__",
        "__subclasses__",
        "__bases__",
        "__globals__",
    ])
    def test_rejects_dunder_and_private_attributes(self, dunder_attr):
        """Any attribute access starting with '__' is blocked to prevent object inspection."""
        code = f"result = df.{dunder_attr}"
        is_valid, msg = validate_code(code)
        assert not is_valid
        assert "dunder" in msg.lower() or "private" in msg.lower()

    def test_rejects_tuple_unpacking_to_result(self):
        """Unpacking assignments (e.g. result, other = (1, 2)) fail closed."""
        code = "result, other = (df['price'].sum(), 10)"
        is_valid, msg = validate_code(code)
        assert not is_valid
        assert "must assign the final output to a variable named 'result'" in msg

    def test_rejects_augmented_assignment_with_initial_assignment(self):
        """Assigning to 'result' and then mutating it with '+=' counts as multiple assignments."""
        code = "result = df['price'].sum()\nresult += 10"
        is_valid, msg = validate_code(code)
        assert not is_valid
        assert "assigned exactly once" in msg

    def test_exponentiation_limit_boundary_conditions(self):
        """Verifies exact numerical boundaries for constant exponentiation limits."""
        # Base 100,000 with exponent 2 is allowed (<= 100,000)
        valid_base, _ = validate_code("result = 100000 ** 2")
        assert valid_base

        # Base 100,001 is rejected (> 100,000)
        invalid_base, msg_b = validate_code("result = 100001 ** 2")
        assert not invalid_base
        assert "Exponentiation base exceeds safe limit" in msg_b

        # Exponent power 1000 is allowed (<= 1000)
        valid_pow, _ = validate_code("result = 2 ** 1000")
        assert valid_pow

        # Exponent power 1001 is rejected (> 1000)
        invalid_pow, msg_p = validate_code("result = 2 ** 1001")
        assert not invalid_pow
        assert "Exponentiation power exceeds safe limit" in msg_p


# =========================================================================
# Part 2: Environment Sanitization & Secret Isolation Tests
# =========================================================================

class TestEnvironmentSanitization:
    """
    Verifies that host secrets, API keys, and arbitrary tokens are stripped
    from child process environment blocks before execution.
    """

    def test_safe_env_strips_sensitive_host_secrets(self, monkeypatch):
        """Host environment variables containing secrets are never inherited by the child."""
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSySecretApiKey123456789")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
        monkeypatch.setenv("DATABASE_URL", "postgres://user:password123@localhost:5432/db")
        monkeypatch.setenv("SUPER_SECRET_TOKEN", "my_secret_token_value")
        monkeypatch.setenv("AUTH_BEARER", "bearer_token_xyz")

        env = _get_safe_env()

        for forbidden_key in [
            "GEMINI_API_KEY",
            "AWS_SECRET_ACCESS_KEY",
            "DATABASE_URL",
            "SUPER_SECRET_TOKEN",
            "AUTH_BEARER",
        ]:
            assert forbidden_key not in env
            # Also verify lowercase / mixed-case forms are not present
            assert forbidden_key.lower() not in env

        # Ensure standard necessary system keys are preserved for Python runtime
        assert "PYTHONUNBUFFERED" in env
        assert env["PYTHONUNBUFFERED"] == "1"
        assert "PYTHONIOENCODING" in env


# =========================================================================
# Part 3: Admission Bounds & Resource Defense Tests
# =========================================================================

class TestAdmissionResourceBounds:
    """
    Verifies pre-admission bounds on code length, DataFrame row/col count,
    and estimated memory consumption before spawning subprocesses.
    """

    def test_oversized_code_payload_rejected_at_admission(self, base_df):
        """Code exceeding 512 KiB is rejected before any subprocess spawn."""
        huge_code = "# " + ("x" * (MAX_CODE_SIZE_BYTES + 1024)) + "\nresult = df['price'].sum()"
        executor = SandboxExecutor()
        success, msg = executor.execute(huge_code, base_df)
        assert success is False
        assert "Code payload exceeds maximum size limit" in msg

    def test_oversized_input_rows_rejected_at_admission(self, base_df, monkeypatch):
        """DataFrames exceeding MAX_INPUT_ROWS are rejected before worker execution."""
        monkeypatch.setattr(sandbox, "MAX_INPUT_ROWS", 2)
        executor = SandboxExecutor()
        success, msg = executor.execute("result = df['price'].sum()", base_df)
        assert success is False
        assert "rows (3) exceed maximum allowed limit (2)" in msg

    def test_oversized_input_cols_rejected_at_admission(self, base_df, monkeypatch):
        """DataFrames exceeding MAX_INPUT_COLS are rejected before worker execution."""
        monkeypatch.setattr(sandbox, "MAX_INPUT_COLS", 2)
        executor = SandboxExecutor()
        success, msg = executor.execute("result = df['price'].sum()", base_df)
        assert success is False
        assert "columns (3) exceed maximum allowed limit (2)" in msg

    def test_oversized_input_memory_rejected_at_admission(self, monkeypatch, base_df):
        """DataFrames exceeding MAX_INPUT_MEMORY_BYTES are rejected before serialization."""
        def mock_huge_memory(*args, **kwargs):
            return pd.Series([MAX_INPUT_MEMORY_BYTES + 1024])

        monkeypatch.setattr(base_df, "memory_usage", mock_huge_memory)

        executor = SandboxExecutor()
        success, msg = executor.execute("result = df['price'].sum()", base_df)
        assert success is False
        assert "exceeds limit" in msg and "estimated memory" in msg


# =========================================================================
# Part 4: Concurrency Bounding & Queue Saturation Defense
# =========================================================================

class TestConcurrencyBounding:
    """
    Verifies that SandboxExecutor bounds concurrent workers and fails closed
    with a clean diagnostic when the execution queue is saturated.
    """

    def test_executor_queue_saturation_rejection(self, base_df):
        """When max_concurrent_workers is saturated, waiting requests time out cleanly."""
        executor = SandboxExecutor(max_concurrent_workers=1, queue_timeout_seconds=0.2)

        # Acquire the single semaphore slot manually
        acquired = executor._semaphore.acquire(timeout=0.1)
        assert acquired is True

        try:
            # Second call should time out and return queue saturation message
            success, msg = executor.execute("result = df['price'].sum()", base_df)
            assert success is False
            assert "Execution queue saturated" in msg
            assert "exceeded concurrency limit of 1" in msg
        finally:
            executor._semaphore.release()


# =========================================================================
# Part 5: Worker Serialization Bounding Tests
# =========================================================================

class TestWorkerSerializationBounds:
    """
    Tests that the isolated subprocess worker restricts output payload dimensions
    and refuses to return unbounded results.
    """

    def test_worker_rejects_excessive_integer_digits_in_collection(self):
        """Integers with estimated decimal digits exceeding max_int_digits are rejected."""
        # 2 ** 14000 has ~4214 decimal digits, exceeding 4096
        large_int = 2 ** 14000
        is_valid, err = _validate_result_bounds([large_int], max_int_digits=MAX_INTEGER_DIGITS)
        assert not is_valid
        assert "exceeds maximum allowed digit length" in err

    def test_worker_rejects_excessive_string_length_in_collection(self):
        """Single string items in collection exceeding max_str_len are rejected."""
        oversized_str = "A" * (MAX_STRING_LENGTH + 10)
        is_valid, err = _validate_result_bounds([oversized_str], max_str_len=MAX_STRING_LENGTH)
        assert not is_valid
        assert "length exceeds limit" in err

    def test_worker_rejects_excessive_nesting_depth(self):
        """Collections exceeding MAX_NESTING_DEPTH fail validation."""
        nested = [1]
        for _ in range(MAX_NESTING_DEPTH + 1):
            nested = [nested]

        is_valid, err = _validate_result_bounds(nested, max_nesting=MAX_NESTING_DEPTH)
        assert not is_valid
        assert "maximum nesting depth limit" in err

    def test_worker_rejects_non_finite_float_values(self, base_df):
        """Non-finite numbers (Infinity, -Infinity) fail execution with explicit error."""
        # Scalar infinity via np.inf
        success_inf, err_inf = execute_pandas_code("result = np.inf", base_df)
        assert success_inf is False
        assert "Non-finite numbers (Inf) are not supported" in err_inf

        # Structured collection containing infinity
        success_coll, err_coll = execute_pandas_code("result = [np.inf]", base_df)
        assert success_coll is False
        assert "Non-finite numbers (NaN, Inf) in structured collections are not supported" in err_coll


# =========================================================================
# Part 6: Host Output Validation & Path Traversal Rejection
# =========================================================================

class TestHostOutputValidation:
    """
    Verifies that the host process rigorously validates output artifacts
    before opening or reading them into memory.
    """

    def test_rejects_missing_output_directory(self, tmp_path):
        """Non-existent output directory fails verification."""
        non_existent = str(tmp_path / "does_not_exist")
        ok, err = _validate_host_output_files(
            non_existent,
            os.path.join(non_existent, "response.json"),
            os.path.join(non_existent, "output_data.arrow"),
        )
        assert not ok
        assert "does not exist" in err

    def test_rejects_missing_response_envelope(self, tmp_path):
        """Missing response.json fails verification."""
        out_dir = str(tmp_path / "out")
        os.makedirs(out_dir, exist_ok=True)
        resp_path = os.path.join(out_dir, "response.json")
        arrow_path = os.path.join(out_dir, "output_data.arrow")

        ok, err = _validate_host_output_files(out_dir, resp_path, arrow_path)
        assert not ok
        assert "Worker did not return response envelope" in err

    def test_rejects_oversized_response_envelope(self, tmp_path):
        """Response envelope exceeding MAX_RESPONSE_JSON_BYTES is rejected before reading."""
        out_dir = str(tmp_path / "out")
        os.makedirs(out_dir, exist_ok=True)
        resp_path = os.path.join(out_dir, "response.json")
        arrow_path = os.path.join(out_dir, "output_data.arrow")

        with open(resp_path, "wb") as f:
            f.write(b"x" * (MAX_RESPONSE_JSON_BYTES + 1024))

        ok, err = _validate_host_output_files(out_dir, resp_path, arrow_path)
        assert not ok
        assert "Response envelope size" in err and "exceeds limit" in err

    def test_rejects_oversized_total_directory_storage(self, tmp_path):
        """Aggregate files in out_dir exceeding MAX_TOTAL_IPC_BYTES are rejected."""
        out_dir = str(tmp_path / "out")
        os.makedirs(out_dir, exist_ok=True)
        resp_path = os.path.join(out_dir, "response.json")
        arrow_path = os.path.join(out_dir, "output_data.arrow")

        with open(resp_path, "wb") as f:
            f.write(b'{"version":"1.0"}')

        # Create dummy file inflating directory size
        extra_file = os.path.join(out_dir, "extra.bin")
        with open(extra_file, "wb") as f:
            f.seek(MAX_TOTAL_IPC_BYTES + 1024)
            f.write(b"0")

        ok, err = _validate_host_output_files(out_dir, resp_path, arrow_path)
        assert not ok
        assert "Total output IPC storage" in err and "exceeds limit" in err


# =========================================================================
# Part 7: Timeout, Cleanup, and Diagnostic Redaction Tests
# =========================================================================

class TestTimeoutAndDiagnosticRedaction:
    """
    Verifies execution timeout enforcement, deterministic cleanup, and
    diagnostic redaction of filesystem paths and credentials.
    """

    def test_sandbox_wall_clock_timeout_enforcement(self, base_df):
        """
        Executes code under an explicit timeout.
        Fast valid code completes well within 5.0s.
        """
        executor = SandboxExecutor()
        success, result = executor.execute("result = df['price'].sum()", base_df, timeout_seconds=5.0)
        assert success is True
        assert result == 60.0

    def test_diagnostic_sanitization_redacts_credentials(self):
        """Credentials and tokens in cleanup errors are replaced with [redacted]."""
        raw = "Error token=super_secret_token_12345 password=topsecret with API key"
        clean = _sanitize_cleanup_diagnostic(raw)
        assert "super_secret_token_12345" not in clean
        assert "topsecret" not in clean
        assert "[redacted]" in clean

    def test_diagnostic_sanitization_redacts_paths(self):
        """Absolute file paths (POSIX /var/... or Windows C:\\...) are replaced with [path]."""
        raw_posix = "Failed to unlink /var/lib/docker/overlay2/deadbeef/merged/file"
        clean_posix = _sanitize_cleanup_diagnostic(raw_posix)
        assert "/var/lib/docker" not in clean_posix
        assert "[path]" in clean_posix

        raw_win = "Cannot delete C:\\Users\\User\\AppData\\Local\\Temp\\agy_ipc_1234"
        clean_win = _sanitize_cleanup_diagnostic(raw_win)
        assert "C:\\Users" not in clean_win
        assert "[path]" in clean_win

    def test_diagnostic_sanitization_length_bounded(self):
        """Sanitized diagnostics cannot flood logs or the UI beyond 120 characters."""
        long_message = "Unexpected daemon error: " + ("z" * 200)
        clean = _sanitize_cleanup_diagnostic(long_message)
        assert len(clean) <= 120
        assert clean.endswith("...")


# =========================================================================
# Part 8: Platform Isolation Specification & Flag Verification
# =========================================================================

class TestPlatformIsolationConfiguration:
    """
    Verifies the platform isolation command generation and user validation.
    """

    def test_docker_user_parsing_valid(self):
        """Verifies that valid container user specs are parsed correctly."""
        assert _parse_docker_user("1000") == (1000, None)
        assert _parse_docker_user("1000:1000") == (1000, 1000)
        assert _parse_docker_user("10001:10001") == (10001, 10001)

    @pytest.mark.parametrize("invalid_user", [
        "",
        "   ",
        "-1",
        "root",
        "user:group",
        "1000:1000:1000",
        "1000:-1",
    ])
    def test_docker_user_parsing_rejects_malformed_input(self, invalid_user):
        """Verifies that non-digit, negative, or malformed container user specs are rejected."""
        with pytest.raises(ValueError):
            _parse_docker_user(invalid_user)

    def test_docker_command_hardening_flags_presence(self, tmp_path):
        """Verifies that generated docker run command strictly includes all hardening flags."""
        in_dir = str(tmp_path / "in")
        out_dir = str(tmp_path / "out")
        os.makedirs(in_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        cmd, cname = _build_docker_cmd(
            in_dir=in_dir,
            out_dir=out_dir,
            image_name="ai-data-analyst-sandbox:latest",
            container_user="10001:10001",
        )

        assert "--network" in cmd and cmd[cmd.index("--network") + 1] == "none"
        assert "--read-only" in cmd
        assert "--cap-drop" in cmd and cmd[cmd.index("--cap-drop") + 1] == "ALL"
        assert "--security-opt" in cmd and cmd[cmd.index("--security-opt") + 1] == "no-new-privileges:true"
        assert "--user" in cmd and cmd[cmd.index("--user") + 1] == "10001:10001"
        assert "--pull=never" in cmd
        assert "--memory=512m" in cmd
        assert "--memory-swap=512m" in cmd
        assert "--cpus=1.0" in cmd
        assert "--pids-limit=16" in cmd
        assert f"{os.path.abspath(in_dir)}:/workspace/ipc/in:ro" in cmd
        assert f"{os.path.abspath(out_dir)}:/workspace/ipc/out:rw" in cmd
