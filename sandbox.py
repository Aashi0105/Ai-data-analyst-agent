"""
sandbox.py
----------
Purpose:
    Executes validated Python pandas code inside an isolated standalone subprocess worker.
    Uses safe, bounded Apache Arrow IPC streaming and structured JSON envelopes.
    Enforces strict pre-admission dataset bounds, input/output Arrow size caps,
    response schema and payload bounds, total IPC directory storage limits,
    kernel-level DEVNULL diagnostic bounding, and hard wall-clock process timeouts.
    Never uses Python pickle and strictly forbids legacy in-process fallback.
"""

import sys
import os
import io
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Tuple, Any, Dict
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from validator import validate_code, clean_code

# Path to the standalone worker script
WORKER_SCRIPT = Path(__file__).resolve().parent / "sandbox_worker.py"

# =========================================================================
# Resource Budget Constants (Configurable Defaults)
# =========================================================================
MAX_CODE_SIZE_BYTES = 512 * 1024                # 512 KiB max validated code string
MAX_INPUT_ROWS = 2_000_000                      # 2,000,000 max rows for input dataset
MAX_INPUT_COLS = 500                            # 500 max columns for input dataset
MAX_INPUT_MEMORY_BYTES = 250 * 1024 * 1024      # 250 MiB estimated DataFrame memory admission bound
MAX_REQUEST_JSON_BYTES = 1 * 1024 * 1024        # 1 MiB max request envelope
MAX_INPUT_ARROW_BYTES = 100 * 1024 * 1024       # 100 MiB max input serialized Arrow file
MAX_RESULT_ARROW_BYTES = 100 * 1024 * 1024      # 100 MiB max output serialized Arrow file
MAX_RESPONSE_JSON_BYTES = 2 * 1024 * 1024       # 2 MiB max response envelope
MAX_TOTAL_IPC_BYTES = 250 * 1024 * 1024         # 250 MiB max temporary storage per execution
DEFAULT_TIMEOUT_SECONDS = 5.0                   # 5.0s hard wall-clock execution deadline
MAX_STRING_LENGTH = 64 * 1024                   # 64 KiB individual string limit
MAX_AGGREGATE_STRING_BYTES = 1 * 1024 * 1024    # 1 MiB max aggregate string/key UTF-8 bytes
MAX_COLLECTION_ITEMS = 10_000                   # 10,000 max collection items / nodes
MAX_NESTING_DEPTH = 5                           # 5 max collection nesting depth
MAX_INTEGER_DIGITS = 4096                       # 4,096 estimated decimal digits per integer
MAX_AGGREGATE_NUMERIC_BYTES = 1 * 1024 * 1024   # 1 MiB max aggregate numeric representation budget



class ArrowWriteLimitExceededError(ValueError):
    """Raised when an Arrow IPC stream write exceeds the configured byte budget."""
    pass


class BoundedFileWriter(io.RawIOBase):
    """
    A file-like write wrapper that enforces a strict byte budget during stream writing.
    Aborts immediately upon any write call that would cause total written bytes to exceed max_bytes.
    """
    def __init__(self, raw_file, max_bytes: int):
        self.raw = raw_file
        self.max_bytes = max_bytes
        self.bytes_written = 0
        self.aborted = False

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    def write(self, b) -> int:
        if self.aborted:
            # Once aborted, refuse subsequent writes and drop trailing writer flush/close writes
            return 0
        data_len = len(b)
        if self.bytes_written + data_len > self.max_bytes:
            self.aborted = True
            raise ArrowWriteLimitExceededError(
                f"Arrow IPC write limit exceeded: write of {data_len} bytes "
                f"would bring total written ({self.bytes_written + data_len} bytes) "
                f"over budget ({self.max_bytes} bytes)."
            )
        n = self.raw.write(b)
        self.bytes_written += n
        return n

    def flush(self) -> None:
        if not self.aborted and hasattr(self.raw, "flush"):
            self.raw.flush()

    def close(self) -> None:
        if hasattr(self.raw, "close"):
            self.raw.close()


# Safe built-in functions exported for callers and tests
SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
}


def _get_safe_env() -> Dict[str, str]:
    """
    Constructs a minimal, safe environment for the child worker process.
    Explicitly strips API keys, tokens, and unrelated host secrets.
    """
    allowed_keys = {
        # Windows system keys required for runtime initialization
        "SYSTEMROOT",
        "SYSTEMDRIVE",
        "WINDIR",
        "COMSPEC",
        "PATH",
        "PATHEXT",
        "TEMP",
        "TMP",
        # POSIX keys
        "LANG",
        "LC_ALL",
        "TMPDIR",
        # Python encoding configuration
        "PYTHONIOENCODING",
        "PYTHONUTF8",
    }
    safe_env = {k: v for k, v in os.environ.items() if k.upper() in allowed_keys}
    safe_env["PYTHONUNBUFFERED"] = "1"
    safe_env["PYTHONIOENCODING"] = "utf-8"
    return safe_env


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    """
    Terminates the worker process and attempts to kill descendants across Windows and POSIX.
    Note: On Windows, taskkill /F /T is a userland helper; true kernel-enforced termination
    requires Windows Job Objects (planned for OS-level containment).
    """
    if proc.poll() is not None:
        return

    pid = proc.pid
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3.0,
                check=False
            )
        except Exception:
            pass
    else:
        try:
            import signal
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:
            pass

    try:
        proc.kill()
        proc.wait(timeout=1.0)
    except Exception:
        pass


def execute_pandas_code(
    code_str: str,
    df: pd.DataFrame,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
) -> Tuple[bool, Any]:
    """
    Executes validated pandas code in an isolated subprocess worker with strict resource budgeting.

    Args:
        code_str (str): AST-validated Python pandas code.
        df (pd.DataFrame): Input dataset DataFrame.
        timeout_seconds (float): Maximum wall-clock execution time allowed.

    Returns:
        Tuple[bool, Any]: (success_flag, result_object_or_user_friendly_error)
    """
    # 1. Input pre-admission checks and structural validation
    if not code_str or not code_str.strip():
        return False, "Execution failed: Code string is empty."

    encoded_code = code_str.encode("utf-8")
    if len(encoded_code) > MAX_CODE_SIZE_BYTES:
        return False, f"Execution failed: Code payload exceeds maximum size limit ({MAX_CODE_SIZE_BYTES // 1024} KiB)."

    if not isinstance(df, pd.DataFrame):
        return False, "Execution failed: Input dataset must be a pandas DataFrame."

    # Dataset dimension pre-admission checks
    if len(df) > MAX_INPUT_ROWS:
        return False, f"Execution failed: Input dataset rows ({len(df)}) exceed maximum allowed limit ({MAX_INPUT_ROWS})."

    if len(df.columns) > MAX_INPUT_COLS:
        return False, f"Execution failed: Input dataset columns ({len(df.columns)}) exceed maximum allowed limit ({MAX_INPUT_COLS})."

    # Input memory footprint admission check
    try:
        est_mem_bytes = int(df.memory_usage(deep=True).sum())
    except Exception:
        est_mem_bytes = 0

    if est_mem_bytes > MAX_INPUT_MEMORY_BYTES:
        return False, f"Execution failed: Input dataset estimated memory ({est_mem_bytes // (1024*1024)} MiB) exceeds limit ({MAX_INPUT_MEMORY_BYTES // (1024*1024)} MiB)."

    # Enforce AST validation defense-in-depth at execution boundary
    is_valid, val_msg = validate_code(code_str)
    if not is_valid:
        return False, f"Validation Error: {val_msg}"

    code_to_run = clean_code(code_str)

    if not WORKER_SCRIPT.is_file():
        return False, f"Execution failed: Sandbox worker not found at {WORKER_SCRIPT}."

    # 2. Ephemeral IPC directory setup with deterministic cleanup
    try:
        with tempfile.TemporaryDirectory(prefix="agy_ipc_") as ipc_dir:
            req_path = os.path.join(ipc_dir, "request.json")
            input_arrow_path = os.path.join(ipc_dir, "input_data.arrow")
            resp_path = os.path.join(ipc_dir, "response.json")
            output_arrow_path = os.path.join(ipc_dir, "output_data.arrow")

            # Write bounded request envelope
            try:
                req_payload = {
                    "code": code_to_run,
                    "version": "1.0",
                    "max_result_arrow_bytes": MAX_RESULT_ARROW_BYTES,
                    "max_response_json_bytes": MAX_RESPONSE_JSON_BYTES,
                    "max_string_length": MAX_STRING_LENGTH,
                    "max_aggregate_string_bytes": MAX_AGGREGATE_STRING_BYTES,
                    "max_collection_items": MAX_COLLECTION_ITEMS,
                    "max_nesting_depth": MAX_NESTING_DEPTH,
                    "max_integer_digits": MAX_INTEGER_DIGITS,
                    "max_aggregate_numeric_bytes": MAX_AGGREGATE_NUMERIC_BYTES,
                }
                req_bytes = json.dumps(req_payload).encode("utf-8")
                if len(req_bytes) > MAX_REQUEST_JSON_BYTES:
                    return False, f"Execution setup failed: Request payload size ({len(req_bytes)} bytes) exceeds limit ({MAX_REQUEST_JSON_BYTES} bytes)."
                with open(req_path, "wb") as f:
                    f.write(req_bytes)
            except Exception as e:
                return False, f"Execution setup failed: Unable to write request: {str(e)}"

            # Serialize input DataFrame via PyArrow stream with write-time byte budget enforcement
            try:
                table = pa.Table.from_pandas(df)
                with open(input_arrow_path, "wb") as raw_f:
                    bounded_f = BoundedFileWriter(raw_f, MAX_INPUT_ARROW_BYTES)
                    with ipc.new_stream(bounded_f, table.schema) as writer:
                        writer.write_table(table)

                input_arrow_size = os.path.getsize(input_arrow_path)
                if input_arrow_size > MAX_INPUT_ARROW_BYTES:
                    if os.path.exists(input_arrow_path):
                        try:
                            os.remove(input_arrow_path)
                        except OSError:
                            pass
                    return False, f"Execution setup failed: Serialized dataset size ({input_arrow_size // (1024*1024)} MiB) exceeds limit ({MAX_INPUT_ARROW_BYTES // (1024*1024)} MiB)."
            except ArrowWriteLimitExceededError as e:
                if os.path.exists(input_arrow_path):
                    try:
                        os.remove(input_arrow_path)
                    except OSError:
                        pass
                return False, f"Execution setup failed: Serialized dataset size exceeds limit: {str(e)}"
            except Exception as e:
                if os.path.exists(input_arrow_path):
                    try:
                        os.remove(input_arrow_path)
                    except OSError:
                        pass
                return False, f"Execution setup failed: Unable to serialize dataset: {str(e)}"

            # Pre-launch total IPC storage check
            initial_ipc_size = sum(
                os.path.getsize(os.path.join(ipc_dir, f))
                for f in os.listdir(ipc_dir)
                if os.path.isfile(os.path.join(ipc_dir, f))
            )
            if initial_ipc_size > MAX_TOTAL_IPC_BYTES:
                return False, f"Execution setup failed: Initial IPC storage ({initial_ipc_size // (1024*1024)} MiB) exceeds total limit ({MAX_TOTAL_IPC_BYTES // (1024*1024)} MiB)."

            # 3. Spawn isolated worker subprocess with zero diagnostic memory/disk overhead
            cmd = [sys.executable, "-u", str(WORKER_SCRIPT), "--ipc-dir", ipc_dir]
            safe_env = _get_safe_env()

            popen_kwargs: Dict[str, Any] = {
                "stdout": subprocess.DEVNULL,  # Discard worker stdout at OS kernel level
                "stderr": subprocess.DEVNULL,  # Discard worker stderr at OS kernel level; diagnostics in response.json
                "env": safe_env,
            }

            if sys.platform == "win32":
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                popen_kwargs["start_new_session"] = True

            try:
                proc = subprocess.Popen(cmd, **popen_kwargs)
            except Exception as e:
                return False, f"Execution failed: Could not launch worker process: {str(e)}"

            # 4. Wait for execution under hard wall-clock deadline
            try:
                proc.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                _terminate_process_tree(proc)
                return False, f"Execution timed out after {timeout_seconds} seconds."
            except Exception as e:
                _terminate_process_tree(proc)
                return False, f"Execution process failure: {str(e)}"

            # 5. Process exit verification and error diagnostic handling
            if proc.returncode != 0:
                if os.path.exists(resp_path) and os.path.getsize(resp_path) <= MAX_RESPONSE_JSON_BYTES:
                    try:
                        with open(resp_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if data.get("error_message"):
                            return False, data["error_message"]
                    except Exception:
                        pass

                return False, f"Execution worker exited with code {proc.returncode}."

            # 6. Post-execution IPC directory total size verification
            total_ipc_size = sum(
                os.path.getsize(os.path.join(ipc_dir, f))
                for f in os.listdir(ipc_dir)
                if os.path.isfile(os.path.join(ipc_dir, f))
            )
            if total_ipc_size > MAX_TOTAL_IPC_BYTES:
                return False, f"Execution failed: Total IPC storage ({total_ipc_size // (1024*1024)} MiB) exceeds limit ({MAX_TOTAL_IPC_BYTES // (1024*1024)} MiB)."

            # 7. Response envelope pre-read size check and schema validation
            if not os.path.exists(resp_path):
                return False, "Execution failed: Worker did not return response envelope."

            resp_file_size = os.path.getsize(resp_path)
            if resp_file_size > MAX_RESPONSE_JSON_BYTES:
                return False, f"Execution failed: Response envelope size ({resp_file_size} bytes) exceeds limit ({MAX_RESPONSE_JSON_BYTES} bytes)."

            try:
                with open(resp_path, "r", encoding="utf-8") as f:
                    response_data = json.load(f)
            except Exception as e:
                return False, f"Execution failed: Malformed response envelope: {str(e)}"

            if not isinstance(response_data, dict):
                return False, "Execution failed: Response envelope root must be a JSON object."

            # Validate protocol version
            if response_data.get("version") != "1.0":
                return False, f"Execution failed: Incompatible protocol response version '{response_data.get('version')}'."

            # Validate status
            status = response_data.get("status")
            if status not in ("success", "error"):
                return False, f"Execution failed: Unknown protocol status '{status}'."

            if status == "error":
                return False, response_data.get("error_message", "Execution failed.")

            # Validate result_type
            result_type = response_data.get("result_type")
            valid_types = {
                "dataframe", "series", "bool", "int", "float",
                "str", "datetime", "nan", "pd_na", "none", "dict", "list"
            }
            if result_type not in valid_types:
                return False, f"Protocol violation: Unrecognized result type '{result_type}'."

            has_arrow_data = bool(response_data.get("has_arrow_data", False))

            # Contradiction checks
            if has_arrow_data and result_type not in ("dataframe", "series"):
                return False, "Protocol violation: Inconsistent tabular metadata (arrow data declared for non-tabular result)."

            if not has_arrow_data and result_type in ("dataframe", "series"):
                return False, "Protocol violation: Missing expected tabular payload for tabular result type."

            # 8. Deserialize result payload safely
            if has_arrow_data:
                if not os.path.exists(output_arrow_path):
                    return False, "Execution failed: Output tabular payload missing from IPC directory."

                out_arrow_size = os.path.getsize(output_arrow_path)
                if out_arrow_size > MAX_RESULT_ARROW_BYTES:
                    return False, f"Execution failed: Result tabular payload size ({out_arrow_size // (1024*1024)} MiB) exceeds limit ({MAX_RESULT_ARROW_BYTES // (1024*1024)} MiB)."

                try:
                    with open(output_arrow_path, "rb") as f:
                        reader = ipc.open_stream(f)
                        df_res = reader.read_pandas()
                except Exception as e:
                    return False, f"Execution failed: Could not deserialize result dataset: {str(e)}"

                if result_type == "series":
                    series_res = df_res.iloc[:, 0]
                    meta = response_data.get("series_metadata") or {}
                    series_res.name = meta.get("name")
                    if meta.get("index_name"):
                        series_res.index.name = meta["index_name"]
                    return True, series_res

                return True, df_res

            if result_type == "nan":
                return True, np.nan

            if result_type == "pd_na":
                return True, pd.NA

            if result_type == "none":
                return True, None

            if result_type == "datetime":
                try:
                    return True, pd.Timestamp(response_data.get("scalar_value"))
                except Exception:
                    return True, response_data.get("scalar_value")

            if result_type == "bool":
                return True, bool(response_data.get("scalar_value"))

            if result_type == "int":
                return True, int(response_data.get("scalar_value"))

            if result_type == "float":
                val = response_data.get("scalar_value")
                if val == "Infinity":
                    return True, float("inf")
                elif val == "-Infinity":
                    return True, float("-inf")
                return True, float(val)

            if result_type == "str":
                return True, str(response_data.get("scalar_value"))

            # dicts and lists
            return True, response_data.get("scalar_value")

    except Exception as e:
        # Catch unexpected host exceptions during execution setup/cleanup
        return False, f"Execution error: {str(e)}"
