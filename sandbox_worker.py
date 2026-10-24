"""
sandbox_worker.py
-----------------
Purpose:
    Standalone worker process executed in an isolated child process by sandbox.py.
    Reads input DataFrame and validated code via safe Arrow IPC and JSON protocol,
    executes with restricted builtins, and writes structured results back.
    Contains zero host secrets, never uses Python pickle, and enforces strict
    output resource bounds on result sizes, string lengths, collection counts, and nesting.
"""

import sys
import os
import io
import argparse
import json
import difflib
import datetime
from typing import Any, Dict, Tuple, Optional
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

# Bounded payload and complexity limits
MAX_REQUEST_JSON_BYTES = 1 * 1024 * 1024        # 1 MiB
MAX_INPUT_ARROW_BYTES = 100 * 1024 * 1024       # 100 MiB
MAX_RESULT_ROWS = 500_000
MAX_RESULT_COLS = 500
MAX_RESULT_ARROW_BYTES = 100 * 1024 * 1024      # 100 MiB
MAX_RESPONSE_JSON_BYTES = 2 * 1024 * 1024       # 2 MiB
MAX_STRING_LENGTH = 64 * 1024                   # 64 KiB
MAX_AGGREGATE_STRING_BYTES = 1 * 1024 * 1024    # 1 MiB max aggregate string/key UTF-8 bytes
MAX_COLLECTION_ITEMS = 10_000
MAX_NESTING_DEPTH = 5
MAX_TOTAL_IPC_BYTES = 250 * 1024 * 1024         # 250 MiB
MAX_INTEGER_DIGITS = 4096                       # 4,096 estimated decimal digits per integer
MAX_AGGREGATE_NUMERIC_BYTES = 1 * 1024 * 1024   # 1 MiB max aggregate numeric representation budget


def _estimate_int_digits(val: Any) -> int:
    """
    Safely and inexpensively estimates the decimal digit length of an integer
    using bit_length() without calling str() or allocating string memory.
    Handles Python int and NumPy integer scalar types.
    """
    if isinstance(val, (bool, np.bool_)):
        return 1
    if isinstance(val, (int, np.integer)):
        py_int = int(val)
        if py_int == 0:
            return 1
        # (bit_length * 302 // 1000) + 1 is guaranteed >= actual decimal digits
        digits = (py_int.bit_length() * 302 // 1000) + 1
        if py_int < 0:
            digits += 1
        return digits
    return 1



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


# Safe built-in functions permitted in worker scope
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


def _format_error(e: Exception, df: pd.DataFrame) -> Dict[str, str]:
    """Generates structured, user-friendly diagnostic messages for runtime errors with strict bounds."""
    err_type = type(e).__name__[:128]
    if isinstance(e, KeyError):
        missing_col = str(e).strip("'\"")[:256]
        str_columns = [str(col)[:256] for col in df.columns]
        matches = difflib.get_close_matches(missing_col, str_columns, n=3, cutoff=0.6)
        suggestion = f" Did you mean one of these: {', '.join(repr(m) for m in matches)}?" if matches else ""
        avail_preview = ", ".join(repr(c) for c in str_columns[:8])
        if len(str_columns) > 8:
            avail_preview += f", ... ({len(str_columns)} total)"
        msg = f"Column '{missing_col}' was not found in the dataset.{suggestion} Available columns are: [{avail_preview}]."
        return {
            "error_type": "KeyError",
            "error_message": msg[:4096]
        }

    if isinstance(e, NameError):
        return {
            "error_type": "NameError",
            "error_message": f"Undefined reference: {str(e)[:1024]}. Ensure calculations only reference the existing 'df' DataFrame."[:4096]
        }

    if isinstance(e, IndexError):
        return {
            "error_type": "IndexError",
            "error_message": f"Index out of range: {str(e)[:1024]}. Verify that filtering or indexing did not result in an empty dataset."[:4096]
        }

    if isinstance(e, AttributeError):
        return {
            "error_type": "AttributeError",
            "error_message": f"Data operation error: {str(e)[:1024]}"[:4096]
        }

    if isinstance(e, TypeError):
        return {
            "error_type": "TypeError",
            "error_message": f"Data type mismatch: {str(e)[:1024]}"[:4096]
        }

    if isinstance(e, ZeroDivisionError):
        return {
            "error_type": "ZeroDivisionError",
            "error_message": "Calculation error: Division by zero encountered."
        }

    return {
        "error_type": err_type,
        "error_message": f"Execution failed: {err_type} - {str(e)[:1024]}"[:4096]
    }


def _validate_result_bounds(
    obj: Any,
    depth: int = 1,
    node_counter: list = None,
    byte_counter: list = None,
    num_byte_counter: list = None,
    max_nesting: int = MAX_NESTING_DEPTH,
    max_items: int = MAX_COLLECTION_ITEMS,
    max_str_len: int = MAX_STRING_LENGTH,
    max_agg_bytes: int = MAX_AGGREGATE_STRING_BYTES,
    max_int_digits: int = MAX_INTEGER_DIGITS,
    max_agg_num_bytes: int = MAX_AGGREGATE_NUMERIC_BYTES,
) -> Tuple[bool, str]:
    """Recursively validates nesting depth, total nodes, string lengths, aggregate UTF-8 bytes, numeric budgets, and supported types."""
    if node_counter is None:
        node_counter = [0]
    if byte_counter is None:
        byte_counter = [0]
    if num_byte_counter is None:
        num_byte_counter = [0]

    if depth > max_nesting:
        return False, f"Structured result exceeds maximum nesting depth limit ({max_nesting})."

    if isinstance(obj, dict):
        node_counter[0] += 1
        if node_counter[0] > max_items:
            return False, f"Structured result exceeds maximum item count limit ({max_items})."

        for k, v in obj.items():
            node_counter[0] += 1
            if node_counter[0] > max_items:
                return False, f"Structured result exceeds maximum item count limit ({max_items})."

            if not isinstance(k, str):
                return False, f"Dictionary key must be string, got '{type(k).__name__}'."
            if len(k) > max_str_len:
                return False, f"Dictionary key string length exceeds limit ({max_str_len} chars)."

            k_bytes = len(k.encode("utf-8"))
            byte_counter[0] += k_bytes
            if byte_counter[0] > max_agg_bytes:
                return False, f"Structured result exceeds aggregate string bytes limit ({max_agg_bytes} bytes)."

            if isinstance(v, (dict, list, tuple)):
                ok, err = _validate_result_bounds(
                    v, depth + 1, node_counter, byte_counter, num_byte_counter,
                    max_nesting, max_items, max_str_len, max_agg_bytes,
                    max_int_digits, max_agg_num_bytes
                )
                if not ok:
                    return False, err
            elif isinstance(v, str):
                node_counter[0] += 1
                if node_counter[0] > max_items:
                    return False, f"Structured result exceeds maximum item count limit ({max_items})."
                if len(v) > max_str_len:
                    return False, f"Collection string item length exceeds limit ({max_str_len} chars)."
                v_bytes = len(v.encode("utf-8"))
                byte_counter[0] += v_bytes
                if byte_counter[0] > max_agg_bytes:
                    return False, f"Structured result exceeds aggregate string bytes limit ({max_agg_bytes} bytes)."
            elif isinstance(v, (bool, np.bool_)):
                node_counter[0] += 1
                if node_counter[0] > max_items:
                    return False, f"Structured result exceeds maximum item count limit ({max_items})."
            elif isinstance(v, (int, np.integer)):
                node_counter[0] += 1
                if node_counter[0] > max_items:
                    return False, f"Structured result exceeds maximum item count limit ({max_items})."
                digits = _estimate_int_digits(v)
                if digits > max_int_digits:
                    return False, f"Integer exceeds maximum allowed digit length ({max_int_digits} digits)."
                num_byte_counter[0] += digits
                if num_byte_counter[0] > max_agg_num_bytes:
                    return False, f"Structured result exceeds aggregate numeric representation budget ({max_agg_num_bytes} bytes)."
            elif isinstance(v, (float, np.floating)):
                node_counter[0] += 1
                if node_counter[0] > max_items:
                    return False, f"Structured result exceeds maximum item count limit ({max_items})."
                if np.isnan(v) or np.isinf(v):
                    return False, "Non-finite numbers (NaN, Inf) in structured collections are not supported."
                num_byte_counter[0] += 24
                if num_byte_counter[0] > max_agg_num_bytes:
                    return False, f"Structured result exceeds aggregate numeric representation budget ({max_agg_num_bytes} bytes)."
            elif v is None:
                node_counter[0] += 1
                if node_counter[0] > max_items:
                    return False, f"Structured result exceeds maximum item count limit ({max_items})."
            else:
                return False, f"Unsupported item type '{type(v).__name__}' in structured collection."
        return True, ""

    elif isinstance(obj, (list, tuple)):
        node_counter[0] += 1
        if node_counter[0] > max_items:
            return False, f"Structured result exceeds maximum item count limit ({max_items})."

        for item in obj:
            if isinstance(item, (dict, list, tuple)):
                ok, err = _validate_result_bounds(
                    item, depth + 1, node_counter, byte_counter, num_byte_counter,
                    max_nesting, max_items, max_str_len, max_agg_bytes,
                    max_int_digits, max_agg_num_bytes
                )
                if not ok:
                    return False, err
            elif isinstance(item, str):
                node_counter[0] += 1
                if node_counter[0] > max_items:
                    return False, f"Structured result exceeds maximum item count limit ({max_items})."
                if len(item) > max_str_len:
                    return False, f"Collection string item length exceeds limit ({max_str_len} chars)."
                item_bytes = len(item.encode("utf-8"))
                byte_counter[0] += item_bytes
                if byte_counter[0] > max_agg_bytes:
                    return False, f"Structured result exceeds aggregate string bytes limit ({max_agg_bytes} bytes)."
            elif isinstance(item, (bool, np.bool_)):
                node_counter[0] += 1
                if node_counter[0] > max_items:
                    return False, f"Structured result exceeds maximum item count limit ({max_items})."
            elif isinstance(item, (int, np.integer)):
                node_counter[0] += 1
                if node_counter[0] > max_items:
                    return False, f"Structured result exceeds maximum item count limit ({max_items})."
                digits = _estimate_int_digits(item)
                if digits > max_int_digits:
                    return False, f"Integer exceeds maximum allowed digit length ({max_int_digits} digits)."
                num_byte_counter[0] += digits
                if num_byte_counter[0] > max_agg_num_bytes:
                    return False, f"Structured result exceeds aggregate numeric representation budget ({max_agg_num_bytes} bytes)."
            elif isinstance(item, (float, np.floating)):
                node_counter[0] += 1
                if node_counter[0] > max_items:
                    return False, f"Structured result exceeds maximum item count limit ({max_items})."
                if np.isnan(item) or np.isinf(item):
                    return False, "Non-finite numbers (NaN, Inf) in structured collections are not supported."
                num_byte_counter[0] += 24
                if num_byte_counter[0] > max_agg_num_bytes:
                    return False, f"Structured result exceeds aggregate numeric representation budget ({max_agg_num_bytes} bytes)."
            elif item is None:
                node_counter[0] += 1
                if node_counter[0] > max_items:
                    return False, f"Structured result exceeds maximum item count limit ({max_items})."
            else:
                return False, f"Unsupported item type '{type(item).__name__}' in structured collection."
        return True, ""

    return True, ""


def _validate_collection_bounds(obj: Any, depth: int = 1, counter: list = None) -> Tuple[bool, str]:
    """Backward-compatible alias for _validate_result_bounds."""
    return _validate_result_bounds(obj, depth=depth, node_counter=counter)


def _validate_envelope_bounds(
    envelope: Dict[str, Any],
    max_string_length: int = MAX_STRING_LENGTH,
    max_aggregate_bytes: int = MAX_AGGREGATE_STRING_BYTES,
) -> Tuple[bool, str]:
    """Ensures metadata strings in the response envelope cannot exceed bounds."""
    for field in ("version", "status", "result_type", "error_type"):
        val = envelope.get(field)
        if val is not None and (not isinstance(val, str) or len(val) > 128):
            return False, f"Envelope field '{field}' must be a short string (<= 128 chars)."

    err_msg = envelope.get("error_message")
    if err_msg is not None:
        if not isinstance(err_msg, str):
            envelope["error_message"] = str(err_msg)[:4096]
        elif len(err_msg) > max_string_length:
            envelope["error_message"] = err_msg[:max_string_length]
        if len(envelope["error_message"].encode("utf-8")) > max_aggregate_bytes:
            envelope["error_message"] = envelope["error_message"][:1024]

    meta = envelope.get("series_metadata")
    if meta is not None and isinstance(meta, dict):
        for k, v in meta.items():
            if v is not None and (not isinstance(v, str) or len(v) > 1024):
                meta[k] = str(v)[:1024]

    return True, ""


def _write_response(
    resp_path: str,
    payload: Dict[str, Any],
    max_resp_bytes: int = MAX_RESPONSE_JSON_BYTES,
    max_string_length: int = MAX_STRING_LENGTH,
    max_aggregate_bytes: int = MAX_AGGREGATE_STRING_BYTES,
) -> None:
    """Serializes and writes the response JSON envelope with strict bounds and fallbacks."""
    _validate_envelope_bounds(payload, max_string_length, max_aggregate_bytes)
    try:
        resp_encoded = json.dumps(payload).encode("utf-8")
        if len(resp_encoded) > max_resp_bytes:
            fallback = {
                "version": "1.0",
                "status": "error",
                "result_type": "none",
                "has_arrow_data": False,
                "scalar_value": None,
                "series_metadata": None,
                "error_type": "ResultTooLarge",
                "error_message": f"Response envelope size ({len(resp_encoded)} bytes) exceeds limit ({max_resp_bytes} bytes)."
            }
            resp_encoded = json.dumps(fallback).encode("utf-8")

        with open(resp_path, "wb") as f:
            f.write(resp_encoded)
    except Exception as e:
        fallback = {
            "version": "1.0",
            "status": "error",
            "result_type": "none",
            "has_arrow_data": False,
            "scalar_value": None,
            "series_metadata": None,
            "error_type": "SerializationError",
            "error_message": f"Failed to serialize response JSON: {str(e)[:256]}"
        }
        with open(resp_path, "wb") as f:
            f.write(json.dumps(fallback).encode("utf-8"))



def run_worker(
    ipc_dir: Optional[str] = None,
    in_dir: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> None:
    """Executes the analysis query inside the worker process."""
    if in_dir is None and out_dir is None:
        if ipc_dir is None:
            raise ValueError("Either ipc_dir or both in_dir and out_dir must be provided.")
        in_dir = ipc_dir
        out_dir = ipc_dir
    elif in_dir is None or out_dir is None:
        raise ValueError("Both in_dir and out_dir must be provided if one is specified.")

    req_path = os.path.join(in_dir, "request.json")
    input_arrow_path = os.path.join(in_dir, "input_data.arrow")
    resp_path = os.path.join(out_dir, "response.json")
    output_arrow_path = os.path.join(out_dir, "output_data.arrow")

    if not os.path.exists(req_path) or not os.path.exists(input_arrow_path):
        _write_response(resp_path, {
            "version": "1.0",
            "status": "error",
            "result_type": "none",
            "has_arrow_data": False,
            "scalar_value": None,
            "series_metadata": None,
            "error_type": "ProtocolError",
            "error_message": "Worker failed: Missing input files in IPC directory."
        })
        sys.exit(1)

    # Validate input file sizes before reading into memory
    if os.path.getsize(req_path) > MAX_REQUEST_JSON_BYTES:
        _write_response(resp_path, {
            "version": "1.0",
            "status": "error",
            "result_type": "none",
            "has_arrow_data": False,
            "scalar_value": None,
            "series_metadata": None,
            "error_type": "PayloadTooLarge",
            "error_message": f"Request JSON size ({os.path.getsize(req_path)} bytes) exceeds limit ({MAX_REQUEST_JSON_BYTES} bytes)."
        })
        sys.exit(1)

    if os.path.getsize(input_arrow_path) > MAX_INPUT_ARROW_BYTES:
        _write_response(resp_path, {
            "version": "1.0",
            "status": "error",
            "result_type": "none",
            "has_arrow_data": False,
            "scalar_value": None,
            "series_metadata": None,
            "error_type": "PayloadTooLarge",
            "error_message": f"Input Arrow file size exceeds limit ({MAX_INPUT_ARROW_BYTES // (1024*1024)} MiB)."
        })
        sys.exit(1)

    # 1. Read request metadata
    try:
        with open(req_path, "r", encoding="utf-8") as f:
            request_data = json.load(f)
    except Exception as e:
        _write_response(resp_path, {
            "version": "1.0",
            "status": "error",
            "result_type": "none",
            "has_arrow_data": False,
            "scalar_value": None,
            "series_metadata": None,
            "error_type": "ProtocolError",
            "error_message": f"Malformed request JSON: {str(e)}"
        })
        sys.exit(1)

    if not isinstance(request_data, dict) or request_data.get("version") != "1.0" or "code" not in request_data:
        _write_response(resp_path, {
            "version": "1.0",
            "status": "error",
            "result_type": "none",
            "has_arrow_data": False,
            "scalar_value": None,
            "series_metadata": None,
            "error_type": "ProtocolError",
            "error_message": "Invalid request envelope schema."
        })
        sys.exit(1)

    # Extract configurable bounds from request or default constants
    max_result_arrow_bytes = request_data.get("max_result_arrow_bytes", MAX_RESULT_ARROW_BYTES)
    max_response_json_bytes = request_data.get("max_response_json_bytes", MAX_RESPONSE_JSON_BYTES)
    max_string_length = request_data.get("max_string_length", MAX_STRING_LENGTH)
    max_aggregate_string_bytes = request_data.get("max_aggregate_string_bytes", MAX_AGGREGATE_STRING_BYTES)
    max_collection_items = request_data.get("max_collection_items", MAX_COLLECTION_ITEMS)
    max_nesting_depth = request_data.get("max_nesting_depth", MAX_NESTING_DEPTH)
    max_integer_digits = request_data.get("max_integer_digits", MAX_INTEGER_DIGITS)
    max_aggregate_numeric_bytes = request_data.get("max_aggregate_numeric_bytes", MAX_AGGREGATE_NUMERIC_BYTES)

    code_str = request_data.get("code", "")

    # 2. Deserialize DataFrame via safe Arrow IPC stream
    try:
        with open(input_arrow_path, "rb") as f:
            reader = ipc.open_stream(f)
            df = reader.read_pandas()
    except Exception as e:
        _write_response(resp_path, {
            "version": "1.0",
            "status": "error",
            "result_type": "none",
            "has_arrow_data": False,
            "scalar_value": None,
            "series_metadata": None,
            "error_type": "SerializationError",
            "error_message": f"Failed to deserialize input dataset: {str(e)}"
        }, max_resp_bytes=max_response_json_bytes)
        sys.exit(1)

    # Deep copy so operations in worker cannot alter the deserialized input frame
    df_sandbox = df.copy(deep=True)

    restricted_globals = {
        "__builtins__": SAFE_BUILTINS,
        "pd": pd,
        "np": np,
    }
    restricted_locals = {
        "df": df_sandbox
    }

    try:
        # Execute validated code
        exec(code_str, restricted_globals, restricted_locals)

        if "result" in restricted_locals:
            result = restricted_locals["result"]
        elif "result" in restricted_globals:
            result = restricted_globals["result"]
        else:
            _write_response(resp_path, {
                "version": "1.0",
                "status": "error",
                "result_type": "none",
                "has_arrow_data": False,
                "scalar_value": None,
                "series_metadata": None,
                "error_type": "ResultMissing",
                "error_message": "Execution completed, but output variable 'result' was not assigned."
            }, max_resp_bytes=max_response_json_bytes)
            return

    except Exception as e:
        diag = _format_error(e, df)
        _write_response(resp_path, {
            "version": "1.0",
            "status": "error",
            "result_type": "none",
            "has_arrow_data": False,
            "scalar_value": None,
            "series_metadata": None,
            "error_type": diag["error_type"],
            "error_message": diag["error_message"]
        }, max_resp_bytes=max_response_json_bytes)
        return

    # 3. Format result and serialize output
    response_payload: Dict[str, Any] = {
        "version": "1.0",
        "status": "success",
        "result_type": "none",
        "has_arrow_data": False,
        "scalar_value": None,
        "series_metadata": None,
        "error_type": None,
        "error_message": None,
    }

    # Handle missing / null values first
    if result is None:
        response_payload["result_type"] = "none"
        response_payload["scalar_value"] = None

    elif result is pd.NA:
        response_payload["result_type"] = "pd_na"
        response_payload["scalar_value"] = None

    elif isinstance(result, (float, np.floating)) and np.isnan(result):
        response_payload["result_type"] = "nan"
        response_payload["scalar_value"] = None

    elif isinstance(result, (pd.Timestamp, datetime.datetime, datetime.date, np.datetime64)):
        response_payload["result_type"] = "datetime"
        response_payload["scalar_value"] = str(result)

    elif isinstance(result, (bool, np.bool_)):
        response_payload["result_type"] = "bool"
        response_payload["scalar_value"] = bool(result)

    elif isinstance(result, (int, np.integer)):
        digits = _estimate_int_digits(result)
        if digits > max_integer_digits:
            _write_response(resp_path, {
                "version": "1.0",
                "status": "error",
                "result_type": "none",
                "has_arrow_data": False,
                "scalar_value": None,
                "series_metadata": None,
                "error_type": "ResultTooLarge",
                "error_message": f"Integer result exceeds maximum allowed digit length ({max_integer_digits} digits)."
            }, max_resp_bytes=max_response_json_bytes)
            return
        if digits > max_aggregate_numeric_bytes:
            _write_response(resp_path, {
                "version": "1.0",
                "status": "error",
                "result_type": "none",
                "has_arrow_data": False,
                "scalar_value": None,
                "series_metadata": None,
                "error_type": "ResultTooLarge",
                "error_message": f"Integer result exceeds aggregate numeric budget ({max_aggregate_numeric_bytes} bytes)."
            }, max_resp_bytes=max_response_json_bytes)
            return
        response_payload["result_type"] = "int"
        response_payload["scalar_value"] = int(result)

    elif isinstance(result, (float, np.floating)):
        if np.isinf(result):
            _write_response(resp_path, {
                "version": "1.0",
                "status": "error",
                "result_type": "none",
                "has_arrow_data": False,
                "scalar_value": None,
                "series_metadata": None,
                "error_type": "UnsupportedResultType",
                "error_message": "Non-finite numbers (Inf) are not supported."
            }, max_resp_bytes=max_response_json_bytes)
            return
        if 24 > max_aggregate_numeric_bytes:
            _write_response(resp_path, {
                "version": "1.0",
                "status": "error",
                "result_type": "none",
                "has_arrow_data": False,
                "scalar_value": None,
                "series_metadata": None,
                "error_type": "ResultTooLarge",
                "error_message": f"Float result exceeds aggregate numeric budget ({max_aggregate_numeric_bytes} bytes)."
            }, max_resp_bytes=max_response_json_bytes)
            return
        response_payload["result_type"] = "float"
        response_payload["scalar_value"] = float(result)

    elif isinstance(result, str):
        if len(result) > max_string_length:
            _write_response(resp_path, {
                "version": "1.0",
                "status": "error",
                "result_type": "none",
                "has_arrow_data": False,
                "scalar_value": None,
                "series_metadata": None,
                "error_type": "ResultTooLarge",
                "error_message": f"String result ({len(result)} chars) exceeds maximum allowed length ({max_string_length} chars)."
            }, max_resp_bytes=max_response_json_bytes)
            return
        res_bytes = len(result.encode("utf-8"))
        if res_bytes > max_aggregate_string_bytes:
            _write_response(resp_path, {
                "version": "1.0",
                "status": "error",
                "result_type": "none",
                "has_arrow_data": False,
                "scalar_value": None,
                "series_metadata": None,
                "error_type": "ResultTooLarge",
                "error_message": f"String result size ({res_bytes} bytes) exceeds aggregate string limit ({max_aggregate_string_bytes} bytes)."
            }, max_resp_bytes=max_response_json_bytes)
            return
        response_payload["result_type"] = "str"
        response_payload["scalar_value"] = str(result)

    elif isinstance(result, pd.DataFrame):
        if len(result) > MAX_RESULT_ROWS:
            _write_response(resp_path, {
                "version": "1.0",
                "status": "error",
                "result_type": "none",
                "has_arrow_data": False,
                "scalar_value": None,
                "series_metadata": None,
                "error_type": "ResultTooLarge",
                "error_message": f"DataFrame result row count ({len(result)}) exceeds maximum limit ({MAX_RESULT_ROWS})."
            }, max_resp_bytes=max_response_json_bytes)
            return

        if len(result.columns) > MAX_RESULT_COLS:
            _write_response(resp_path, {
                "version": "1.0",
                "status": "error",
                "result_type": "none",
                "has_arrow_data": False,
                "scalar_value": None,
                "series_metadata": None,
                "error_type": "ResultTooLarge",
                "error_message": f"DataFrame result column count ({len(result.columns)}) exceeds maximum limit ({MAX_RESULT_COLS})."
            }, max_resp_bytes=max_response_json_bytes)
            return

        response_payload["result_type"] = "dataframe"
        response_payload["has_arrow_data"] = True
        try:
            table = pa.Table.from_pandas(result)
            with open(output_arrow_path, "wb") as raw_f:
                bounded_f = BoundedFileWriter(raw_f, max_result_arrow_bytes)
                with ipc.new_stream(bounded_f, table.schema) as writer:
                    writer.write_table(table)

            if os.path.getsize(output_arrow_path) > max_result_arrow_bytes:
                if os.path.exists(output_arrow_path):
                    try:
                        os.remove(output_arrow_path)
                    except OSError:
                        pass
                _write_response(resp_path, {
                    "version": "1.0",
                    "status": "error",
                    "result_type": "none",
                    "has_arrow_data": False,
                    "scalar_value": None,
                    "series_metadata": None,
                    "error_type": "ResultTooLarge",
                    "error_message": f"Serialized DataFrame result exceeds maximum size limit ({max_result_arrow_bytes // (1024*1024)} MiB)."
                }, max_resp_bytes=max_response_json_bytes)
                return
        except ArrowWriteLimitExceededError as e:
            if os.path.exists(output_arrow_path):
                try:
                    os.remove(output_arrow_path)
                except OSError:
                    pass
            _write_response(resp_path, {
                "version": "1.0",
                "status": "error",
                "result_type": "none",
                "has_arrow_data": False,
                "scalar_value": None,
                "series_metadata": None,
                "error_type": "ResultTooLarge",
                "error_message": f"Serialized DataFrame result exceeds maximum size limit: {str(e)}"
            }, max_resp_bytes=max_response_json_bytes)
            return
        except Exception as e:
            if os.path.exists(output_arrow_path):
                try:
                    os.remove(output_arrow_path)
                except OSError:
                    pass
            _write_response(resp_path, {
                "version": "1.0",
                "status": "error",
                "result_type": "none",
                "has_arrow_data": False,
                "scalar_value": None,
                "series_metadata": None,
                "error_type": "SerializationError",
                "error_message": f"Failed to serialize DataFrame result: {str(e)}"
            }, max_resp_bytes=max_response_json_bytes)
            return

    elif isinstance(result, pd.Series):
        if len(result) > MAX_RESULT_ROWS:
            _write_response(resp_path, {
                "version": "1.0",
                "status": "error",
                "result_type": "none",
                "has_arrow_data": False,
                "scalar_value": None,
                "series_metadata": None,
                "error_type": "ResultTooLarge",
                "error_message": f"Series result item count ({len(result)}) exceeds maximum limit ({MAX_RESULT_ROWS})."
            }, max_resp_bytes=max_response_json_bytes)
            return

        response_payload["result_type"] = "series"
        response_payload["has_arrow_data"] = True
        response_payload["series_metadata"] = {
            "name": str(result.name) if result.name is not None else None,
            "index_name": str(result.index.name) if result.index.name is not None else None,
        }
        try:
            series_df = result.to_frame()
            table = pa.Table.from_pandas(series_df)
            with open(output_arrow_path, "wb") as raw_f:
                bounded_f = BoundedFileWriter(raw_f, max_result_arrow_bytes)
                with ipc.new_stream(bounded_f, table.schema) as writer:
                    writer.write_table(table)

            if os.path.getsize(output_arrow_path) > max_result_arrow_bytes:
                if os.path.exists(output_arrow_path):
                    try:
                        os.remove(output_arrow_path)
                    except OSError:
                        pass
                _write_response(resp_path, {
                    "version": "1.0",
                    "status": "error",
                    "result_type": "none",
                    "has_arrow_data": False,
                    "scalar_value": None,
                    "series_metadata": None,
                    "error_type": "ResultTooLarge",
                    "error_message": f"Serialized Series result exceeds maximum size limit ({max_result_arrow_bytes // (1024*1024)} MiB)."
                }, max_resp_bytes=max_response_json_bytes)
                return
        except ArrowWriteLimitExceededError as e:
            if os.path.exists(output_arrow_path):
                try:
                    os.remove(output_arrow_path)
                except OSError:
                    pass
            _write_response(resp_path, {
                "version": "1.0",
                "status": "error",
                "result_type": "none",
                "has_arrow_data": False,
                "scalar_value": None,
                "series_metadata": None,
                "error_type": "ResultTooLarge",
                "error_message": f"Serialized Series result exceeds maximum size limit: {str(e)}"
            }, max_resp_bytes=max_response_json_bytes)
            return
        except Exception as e:
            if os.path.exists(output_arrow_path):
                try:
                    os.remove(output_arrow_path)
                except OSError:
                    pass
            _write_response(resp_path, {
                "version": "1.0",
                "status": "error",
                "result_type": "none",
                "has_arrow_data": False,
                "scalar_value": None,
                "series_metadata": None,
                "error_type": "SerializationError",
                "error_message": f"Failed to serialize Series result: {str(e)}"
            }, max_resp_bytes=max_response_json_bytes)
            return

    elif isinstance(result, dict):
        ok, err = _validate_result_bounds(
            result, depth=1, max_nesting=max_nesting_depth,
            max_items=max_collection_items, max_str_len=max_string_length,
            max_agg_bytes=max_aggregate_string_bytes,
            max_int_digits=max_integer_digits,
            max_agg_num_bytes=max_aggregate_numeric_bytes
        )
        if not ok:
            _write_response(resp_path, {
                "version": "1.0",
                "status": "error",
                "result_type": "none",
                "has_arrow_data": False,
                "scalar_value": None,
                "series_metadata": None,
                "error_type": "ResultTooLarge" if any(w in err.lower() for w in ("limit", "budget", "exceeds")) else "UnsupportedResultType",
                "error_message": err
            }, max_resp_bytes=max_response_json_bytes)
            return
        response_payload["result_type"] = "dict"
        response_payload["scalar_value"] = result

    elif isinstance(result, (list, tuple)):
        ok, err = _validate_result_bounds(
            result, depth=1, max_nesting=max_nesting_depth,
            max_items=max_collection_items, max_str_len=max_string_length,
            max_agg_bytes=max_aggregate_string_bytes,
            max_int_digits=max_integer_digits,
            max_agg_num_bytes=max_aggregate_numeric_bytes
        )
        if not ok:
            _write_response(resp_path, {
                "version": "1.0",
                "status": "error",
                "result_type": "none",
                "has_arrow_data": False,
                "scalar_value": None,
                "series_metadata": None,
                "error_type": "ResultTooLarge" if any(w in err.lower() for w in ("limit", "budget", "exceeds")) else "UnsupportedResultType",
                "error_message": err
            }, max_resp_bytes=max_response_json_bytes)
            return
        response_payload["result_type"] = "list"
        response_payload["scalar_value"] = list(result)

    else:
        # Fail closed on unsupported types rather than guessing or arbitrary serialization
        _write_response(resp_path, {
            "version": "1.0",
            "status": "error",
            "result_type": "none",
            "has_arrow_data": False,
            "scalar_value": None,
            "series_metadata": None,
            "error_type": "UnsupportedResultType",
            "error_message": f"Execution returned unsupported result type '{type(result).__name__}'. Result must be a DataFrame, Series, scalar, datetime, dict, or list."
        }, max_resp_bytes=max_response_json_bytes)
        return

    # Final response write with validated bounds
    _write_response(
        resp_path,
        response_payload,
        max_resp_bytes=max_response_json_bytes,
        max_string_length=max_string_length,
        max_aggregate_bytes=max_aggregate_string_bytes
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated Python execution worker")
    parser.add_argument("--ipc-dir", required=False, default=None, help="Directory containing both input and output IPC files (legacy/shared)")
    parser.add_argument("--in-dir", required=False, default=None, help="Directory containing read-only input IPC files (request.json, input_data.arrow)")
    parser.add_argument("--out-dir", required=False, default=None, help="Directory containing writable output IPC files (response.json, output_data.arrow)")
    args = parser.parse_args()

    if args.in_dir and args.out_dir:
        run_worker(in_dir=args.in_dir, out_dir=args.out_dir)
    elif args.ipc_dir:
        run_worker(ipc_dir=args.ipc_dir)
    else:
        parser.error("Either --ipc-dir or both --in-dir and --out-dir must be specified.")


if __name__ == "__main__":
    main()
