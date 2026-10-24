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
import threading
import uuid
from pathlib import Path
from typing import Tuple, Any, Dict, Optional, List
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

# Concurrency and Process Resource Budget Constants
MAX_CONCURRENT_WORKERS = int(os.environ.get("SANDBOX_MAX_CONCURRENT_WORKERS", "4"))
DEFAULT_QUEUE_TIMEOUT_SECONDS = float(os.environ.get("SANDBOX_QUEUE_TIMEOUT_SECONDS", "5.0"))
MAX_WORKER_MEMORY_BYTES = int(os.environ.get("SANDBOX_WORKER_MEMORY_BYTES", str(512 * 1024 * 1024)))  # 512 MiB
MAX_WORKER_PROCESSES = int(os.environ.get("SANDBOX_WORKER_PROCESSES", "16"))
DOCKER_SANDBOX_IMAGE = os.environ.get("SANDBOX_DOCKER_IMAGE", "ai-data-analyst-sandbox:latest")



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
    safe_env: Dict[str, str] = {}
    for k, v in os.environ.items():
        upper_k = k.upper()
        if upper_k in allowed_keys and upper_k not in safe_env:
            safe_env[upper_k] = v

    safe_env["PYTHONUNBUFFERED"] = "1"
    safe_env["PYTHONIOENCODING"] = "utf-8"
    return safe_env


# =========================================================================
# Windows Job Object OS-Level Process Containment
# =========================================================================
# =========================================================================
# Windows Job Object OS-Level Process Containment
# =========================================================================
if sys.platform == "win32":
    import ctypes
    import ctypes.wintypes as wt

    kernel32 = ctypes.windll.kernel32

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wt.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wt.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wt.DWORD),
            ("SchedulingClass", wt.DWORD),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryLimit", ctypes.c_size_t),
            ("PeakJobMemoryLimit", ctypes.c_size_t),
        ]

    class _STARTUPINFOW(ctypes.Structure):
        _fields_ = [
            ("cb", wt.DWORD),
            ("lpReserved", wt.LPWSTR),
            ("lpDesktop", wt.LPWSTR),
            ("lpTitle", wt.LPWSTR),
            ("dwX", wt.DWORD),
            ("dwY", wt.DWORD),
            ("dwXSize", wt.DWORD),
            ("dwYSize", wt.DWORD),
            ("dwXCountChars", wt.DWORD),
            ("dwYCountChars", wt.DWORD),
            ("dwFillAttribute", wt.DWORD),
            ("dwFlags", wt.DWORD),
            ("wShowWindow", wt.WORD),
            ("cbReserved2", wt.WORD),
            ("lpReserved2", ctypes.c_void_p),
            ("hStdInput", wt.HANDLE),
            ("hStdOutput", wt.HANDLE),
            ("hStdError", wt.HANDLE),
        ]

    class _PROCESS_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("hProcess", wt.HANDLE),
            ("hThread", wt.HANDLE),
            ("dwProcessId", wt.DWORD),
            ("dwThreadId", wt.DWORD),
        ]

    class _STARTUPINFOEXW(ctypes.Structure):
        _fields_ = [
            ("StartupInfo", _STARTUPINFOW),
            ("lpAttributeList", ctypes.c_void_p),
        ]

    class _SECURITY_ATTRIBUTES(ctypes.Structure):
        _fields_ = [
            ("nLength", wt.DWORD),
            ("lpSecurityDescriptor", ctypes.c_void_p),
            ("bInheritHandle", wt.BOOL),
        ]

    # Explicit Win32 API signatures to prevent 64-bit handle truncation or pointer sign-extension bugs
    kernel32.CreateJobObjectW.restype = wt.HANDLE
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wt.LPCWSTR]

    kernel32.SetInformationJobObject.restype = wt.BOOL
    kernel32.SetInformationJobObject.argtypes = [
        wt.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wt.DWORD,
    ]

    kernel32.AssignProcessToJobObject.restype = wt.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]

    kernel32.TerminateJobObject.restype = wt.BOOL
    kernel32.TerminateJobObject.argtypes = [wt.HANDLE, wt.UINT]

    kernel32.CloseHandle.restype = wt.BOOL
    kernel32.CloseHandle.argtypes = [wt.HANDLE]

    kernel32.CreateProcessW.restype = wt.BOOL
    kernel32.CreateProcessW.argtypes = [
        wt.LPCWSTR,
        wt.LPWSTR,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wt.BOOL,
        wt.DWORD,
        ctypes.c_void_p,
        wt.LPCWSTR,
        ctypes.c_void_p,  # Accepts pointer to _STARTUPINFOW or _STARTUPINFOEXW
        ctypes.POINTER(_PROCESS_INFORMATION),
    ]

    kernel32.ResumeThread.restype = wt.DWORD
    kernel32.ResumeThread.argtypes = [wt.HANDLE]

    kernel32.TerminateProcess.restype = wt.BOOL
    kernel32.TerminateProcess.argtypes = [wt.HANDLE, wt.UINT]

    kernel32.WaitForSingleObject.restype = wt.DWORD
    kernel32.WaitForSingleObject.argtypes = [wt.HANDLE, wt.DWORD]

    kernel32.GetExitCodeProcess.restype = wt.BOOL
    kernel32.GetExitCodeProcess.argtypes = [wt.HANDLE, ctypes.POINTER(wt.DWORD)]

    kernel32.CreateFileW.restype = wt.HANDLE
    kernel32.CreateFileW.argtypes = [
        wt.LPCWSTR,
        wt.DWORD,
        wt.DWORD,
        ctypes.POINTER(_SECURITY_ATTRIBUTES),
        wt.DWORD,
        wt.DWORD,
        wt.HANDLE,
    ]

    kernel32.InitializeProcThreadAttributeList.restype = wt.BOOL
    kernel32.InitializeProcThreadAttributeList.argtypes = [
        ctypes.c_void_p,
        wt.DWORD,
        wt.DWORD,
        ctypes.POINTER(ctypes.c_size_t),
    ]

    kernel32.UpdateProcThreadAttribute.restype = wt.BOOL
    kernel32.UpdateProcThreadAttribute.argtypes = [
        ctypes.c_void_p,
        wt.DWORD,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]

    kernel32.DeleteProcThreadAttributeList.restype = None
    kernel32.DeleteProcThreadAttributeList.argtypes = [ctypes.c_void_p]

    kernel32.GetLastError.restype = wt.DWORD
    kernel32.GetLastError.argtypes = []

    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
    _JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
    _JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
    _JobObjectExtendedLimitInformation = 9

    _EXTENDED_STARTUPINFO_PRESENT = 0x00080000
    _PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x00020002


class WindowsJobObject:
    """
    Encapsulates a Windows Job Object for OS-level subprocess containment.

    Enforces:
      - Kernel-enforced termination of all descendant processes on job handle closure
        or explicit job termination (prevents orphan processes and runaway trees).
      - Active process count limits (limits concurrent child processes, mitigating
        fork bombs; note: ActiveProcessLimit restricts process count, not thread count).
      - Hard committed memory limits (sets per-process and per-job VirtualAlloc commit limits,
        preventing host memory exhaustion).

    Security Boundaries & Limitations:
      - Windows Job Objects enforce resource quotas and process lifecycle containment.
      - They DO NOT independently isolate the filesystem, network stack, or kernel IPC namespaces.
      - This mechanism mitigates resource exhaustion but does NOT constitute a full sandbox
        safe for arbitrary public untrusted code execution.
    """

    def __init__(
        self,
        memory_limit_bytes: int = MAX_WORKER_MEMORY_BYTES,
        max_processes: int = MAX_WORKER_PROCESSES,
    ):
        self.memory_limit_bytes = memory_limit_bytes
        self.max_processes = max_processes
        self.handle = None
        self._is_windows = (sys.platform == "win32")

    def open(self) -> "WindowsJobObject":
        """Creates and configures the Job Object with kernel limit flags."""
        if not self._is_windows:
            return self

        kernel32 = ctypes.windll.kernel32
        self.handle = kernel32.CreateJobObjectW(None, None)
        if not self.handle:
            err = kernel32.GetLastError()
            raise RuntimeError(f"Failed to create Windows Job Object: error code {err}")

        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        flags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if self.max_processes > 0:
            flags |= _JOB_OBJECT_LIMIT_ACTIVE_PROCESS
            info.BasicLimitInformation.ActiveProcessLimit = self.max_processes

        if self.memory_limit_bytes > 0:
            flags |= _JOB_OBJECT_LIMIT_PROCESS_MEMORY | _JOB_OBJECT_LIMIT_JOB_MEMORY
            info.ProcessMemoryLimit = self.memory_limit_bytes
            info.JobMemoryLimit = self.memory_limit_bytes

        info.BasicLimitInformation.LimitFlags = flags

        success = kernel32.SetInformationJobObject(
            self.handle,
            _JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not success:
            err = kernel32.GetLastError()
            self.close()
            raise RuntimeError(f"Failed to configure Windows Job Object limits: error code {err}")

        return self

    def assign_process(self, proc_or_handle: Any) -> None:
        """Assigns a process (subprocess.Popen, WindowsProcess, or raw HANDLE) to this Job Object."""
        if not self._is_windows:
            return

        if not self.handle:
            raise RuntimeError("Windows Job Object is not active.")

        raw_handle = getattr(proc_or_handle, "_handle", proc_or_handle)
        if raw_handle is None:
            raise RuntimeError("Process does not expose a native OS process handle.")

        if hasattr(proc_or_handle, "poll") and proc_or_handle.poll() is not None:
            # Subprocess already exited; nothing left to contain
            return

        kernel32 = ctypes.windll.kernel32
        if isinstance(raw_handle, wt.HANDLE):
            h_proc = raw_handle
        elif hasattr(raw_handle, "value") and raw_handle.value is not None:
            h_proc = wt.HANDLE(raw_handle.value)
        else:
            h_proc = wt.HANDLE(int(raw_handle))

        success = kernel32.AssignProcessToJobObject(self.handle, h_proc)
        if not success:
            # Check if process exited right during the call
            if hasattr(proc_or_handle, "poll") and proc_or_handle.poll() is not None:
                return
            err = kernel32.GetLastError()
            raise RuntimeError(f"Failed to assign process to Windows Job Object: error code {err}")

    def terminate(self, exit_code: int = 1) -> None:
        """Immediately terminates all processes associated with the Job Object via kernel call."""
        if not self._is_windows or not self.handle:
            return
        kernel32 = ctypes.windll.kernel32
        kernel32.TerminateJobObject(self.handle, exit_code)

    def close(self) -> None:
        """Closes the Job Object handle. Kills descendants if KILL_ON_JOB_CLOSE is set."""
        if self._is_windows and self.handle:
            kernel32 = ctypes.windll.kernel32
            kernel32.CloseHandle(self.handle)
            self.handle = None

    def __enter__(self) -> "WindowsJobObject":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class WindowsProcess:
    """
    Manages a native Windows process handle created via CreateProcessW in a suspended state.
    Implements a subprocess.Popen-compatible interface for poll(), wait(), kill(), and terminate().
    """

    def __init__(self, h_process: Any, pid: int):
        self._handle = h_process
        self.pid = pid
        self.returncode: Optional[int] = None

    def poll(self) -> Optional[int]:
        if self.returncode is not None:
            return self.returncode
        if not self._handle:
            return self.returncode
        status = kernel32.WaitForSingleObject(self._handle, 0)
        if status == 0:  # WAIT_OBJECT_0
            code = wt.DWORD()
            if kernel32.GetExitCodeProcess(self._handle, ctypes.byref(code)):
                self.returncode = int(code.value)
            else:
                self.returncode = 1
            return self.returncode
        return None

    def wait(self, timeout: Optional[float] = None) -> int:
        if self.returncode is not None:
            return self.returncode
        if not self._handle:
            return self.returncode or 0

        timeout_ms = wt.DWORD(int(timeout * 1000) if timeout is not None and timeout >= 0 else 0xFFFFFFFF)
        status = kernel32.WaitForSingleObject(self._handle, timeout_ms)
        if status == 0x00000102:  # WAIT_TIMEOUT
            raise subprocess.TimeoutExpired(cmd=f"pid_{self.pid}", timeout=timeout)
        elif status == 0:  # WAIT_OBJECT_0
            code = wt.DWORD()
            if kernel32.GetExitCodeProcess(self._handle, ctypes.byref(code)):
                self.returncode = int(code.value)
            else:
                self.returncode = 1
            return self.returncode
        else:
            err = kernel32.GetLastError()
            raise RuntimeError(f"WaitForSingleObject failed with status {status}, error code {err}")

    def terminate(self) -> None:
        if self.poll() is None and self._handle:
            kernel32.TerminateProcess(self._handle, 1)
            self.poll()

    def kill(self) -> None:
        self.terminate()

    def close(self) -> None:
        if self._handle:
            kernel32.CloseHandle(self._handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _spawn_suspended_windows_process(
    cmd: List[str],
    env: Dict[str, str],
    job: Optional[WindowsJobObject] = None,
    cwd: Optional[str] = None,
) -> WindowsProcess:
    """
    Creates a child process in a genuinely suspended state on Windows,
    assigns it to the configured Job Object, and resumes execution only
    after assignment succeeds.

    If creation, assignment, or resumption fails, the process is terminated,
    all open handles are safely closed, and an exception is raised.
    Never allows untrusted code to run outside the Job Object.
    """
    cmd_str = subprocess.list2cmdline(cmd)
    cmd_buf = ctypes.create_unicode_buffer(cmd_str)

    # Format Unicode environment block: null-separated, double-null-terminated,
    # sorted in alphabetical order, case-insensitive (required by Windows SDK).
    # Keys are deduplicated case-insensitively.
    normalized_env: Dict[str, str] = {}
    for k, v in env.items():
        upper_k = k.upper()
        if upper_k not in normalized_env:
            normalized_env[upper_k] = (k, v)

    sorted_pairs = sorted(normalized_env.values(), key=lambda item: item[0].upper())
    env_str = "".join(f"{k}={v}\0" for k, v in sorted_pairs) + "\0"
    env_buf = ctypes.create_unicode_buffer(env_str)

    # Route standard handles to NUL with bInheritHandle=True so worker print/logging does not raise OSError
    sa = _SECURITY_ATTRIBUTES()
    sa.nLength = ctypes.sizeof(sa)
    sa.bInheritHandle = True

    h_nul = kernel32.CreateFileW(
        "NUL",
        0x80000000 | 0x40000000,  # GENERIC_READ | GENERIC_WRITE
        0x00000001 | 0x00000002,  # FILE_SHARE_READ | FILE_SHARE_WRITE
        ctypes.byref(sa),
        3,  # OPEN_EXISTING
        0,
        None,
    )
    h_nul_valid = (
        h_nul is not None
        and h_nul != 0
        and h_nul != wt.HANDLE(-1).value
        and h_nul != -1
    )

    attr_buf = None
    attr_initialized = False

    # SUSPENDED (0x04) | NEW_PROCESS_GROUP (0x200) | NO_WINDOW (0x08000000) | UNICODE_ENVIRONMENT (0x400)
    flags = 0x00000004 | 0x00000200 | 0x08000000 | 0x00000400
    cwd_buf = ctypes.create_unicode_buffer(cwd) if cwd else None
    pi = _PROCESS_INFORMATION()

    try:
        if h_nul_valid:
            # Initialize thread attribute list to explicitly whitelist ONLY h_nul
            attr_size = ctypes.c_size_t(0)
            kernel32.InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(attr_size))
            if attr_size.value == 0:
                err = kernel32.GetLastError()
                raise RuntimeError(f"InitializeProcThreadAttributeList failed to calculate size: error code {err}")

            attr_buf = ctypes.create_string_buffer(attr_size.value)
            if not kernel32.InitializeProcThreadAttributeList(attr_buf, 1, 0, ctypes.byref(attr_size)):
                err = kernel32.GetLastError()
                raise RuntimeError(f"InitializeProcThreadAttributeList failed: error code {err}")
            attr_initialized = True

            handle_array = (wt.HANDLE * 1)(h_nul)
            if not kernel32.UpdateProcThreadAttribute(
                attr_buf,
                0,
                _PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                ctypes.byref(handle_array),
                ctypes.sizeof(handle_array),
                None,
                None,
            ):
                err = kernel32.GetLastError()
                raise RuntimeError(f"UpdateProcThreadAttribute failed: error code {err}")

            siex = _STARTUPINFOEXW()
            siex.StartupInfo.cb = ctypes.sizeof(siex)
            siex.StartupInfo.dwFlags = 0x00000100  # STARTF_USESTDHANDLES
            siex.StartupInfo.hStdInput = h_nul
            siex.StartupInfo.hStdOutput = h_nul
            siex.StartupInfo.hStdError = h_nul
            siex.lpAttributeList = ctypes.cast(attr_buf, ctypes.c_void_p)
            startup_info_ptr = ctypes.byref(siex)
            flags |= _EXTENDED_STARTUPINFO_PRESENT
            b_inherit = True
        else:
            si = _STARTUPINFOW()
            si.cb = ctypes.sizeof(si)
            startup_info_ptr = ctypes.byref(si)
            b_inherit = False

        success = kernel32.CreateProcessW(
            None,
            cmd_buf,
            None,
            None,
            b_inherit,
            flags,
            ctypes.byref(env_buf),
            cwd_buf,
            startup_info_ptr,
            ctypes.byref(pi),
        )
    finally:
        if attr_initialized and attr_buf:
            kernel32.DeleteProcThreadAttributeList(attr_buf)
        if h_nul_valid:
            kernel32.CloseHandle(h_nul)

    if not success:
        err = kernel32.GetLastError()
        raise RuntimeError(f"CreateProcessW failed: error code {err}")

    h_proc = pi.hProcess
    h_thread = pi.hThread
    pid = int(pi.dwProcessId)

    # Step 3: Assign suspended worker to configured Job Object
    if job is not None and getattr(job, "handle", None):
        try:
            job.assign_process(h_proc)
        except Exception as e:
            # Terminate immediately before process ever runs
            kernel32.TerminateProcess(h_proc, 1)
            kernel32.CloseHandle(h_thread)
            kernel32.CloseHandle(h_proc)
            raise RuntimeError(f"Could not assign worker to Windows Job Object: {str(e)}") from e

    # Step 4: Resume execution only after successful assignment
    prev_suspend_count = kernel32.ResumeThread(h_thread)
    kernel32.CloseHandle(h_thread)

    if prev_suspend_count == 0xFFFFFFFF:
        err = kernel32.GetLastError()
        kernel32.TerminateProcess(h_proc, 1)
        kernel32.CloseHandle(h_proc)
        raise RuntimeError(f"Failed to resume suspended worker thread: error code {err}")

    return WindowsProcess(h_proc, pid)


def _terminate_process_tree(proc: Any, job: Optional[WindowsJobObject] = None) -> None:
    """
    Terminates the worker process and attempts to kill descendants across Windows and POSIX.
    Uses kernel Job Object termination on Windows when available.
    """
    if job is not None and getattr(job, "handle", None):
        try:
            job.terminate(1)
        except Exception:
            pass

    if hasattr(proc, "poll") and proc.poll() is not None:
        return

    pid = getattr(proc, "pid", None)
    if pid is not None:
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


def _is_docker_available() -> bool:
    """
    Checks if the Docker CLI binary exists and the Docker daemon is responding.
    """
    try:
        res = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=2.0,
            check=False,
            text=True,
        )
        return res.returncode == 0 and bool(res.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError, Exception):
        return False


def _build_docker_cmd(
    in_dir: str,
    out_dir: str,
    image_name: str = DOCKER_SANDBOX_IMAGE,
    container_name: Optional[str] = None,
    memory_limit: str = "512m",
    cpu_limit: str = "1.0",
    pids_limit: int = 16,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> Tuple[List[str], str]:
    """
    Constructs the hardened docker run command line enforcing:
      - --network none: Complete network isolation (no internet, no LAN, no host loopback).
      - --read-only: Read-only container root filesystem.
      - --cap-drop ALL: All Linux capabilities dropped.
      - --security-opt no-new-privileges:true: Prevents gaining privileges.
      - --user 10001:10001: Unprivileged non-root user.
      - --memory and --memory-swap: Strict memory limit with swap disabled.
      - --cpus: CPU quota enforcement.
      - --pids-limit: Process table limit preventing fork bombs.
      - --tmpfs /tmp: Narrow, non-executable, bounded temporary storage.
      - -v in_dir:/workspace/ipc/in:ro: Input files mounted strictly READ-ONLY.
      - -v out_dir:/workspace/ipc/out:rw: Output files mounted in narrowly scoped writable directory.
      - --stop-timeout: Container stop timeout matching execution deadline.
      - --rm: Automatic removal on exit.
    """
    if not container_name:
        container_name = f"sandbox_worker_{uuid.uuid4().hex[:16]}"

    abs_in = os.path.abspath(in_dir)
    abs_out = os.path.abspath(out_dir)

    cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        "--user",
        "10001:10001",
        f"--memory={memory_limit}",
        f"--memory-swap={memory_limit}",
        f"--cpus={cpu_limit}",
        f"--pids-limit={pids_limit}",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "-v",
        f"{abs_in}:/workspace/ipc/in:ro",
        "-v",
        f"{abs_out}:/workspace/ipc/out:rw",
        f"--stop-timeout={max(1, int(timeout_seconds))}",
        image_name,
        "python",
        "-u",
        "/workspace/sandbox_worker.py",
        "--in-dir",
        "/workspace/ipc/in",
        "--out-dir",
        "/workspace/ipc/out",
    ]
    return cmd, container_name


def _kill_and_remove_docker_container(container_name: str) -> None:
    """Best-effort termination and removal of an isolated Docker worker container."""
    try:
        subprocess.run(
            ["docker", "kill", container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5.0,
            check=False,
        )
    except Exception:
        pass
    try:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5.0,
            check=False,
        )
    except Exception:
        pass


def _validate_host_output_files(
    out_dir: str,
    resp_path: str,
    output_arrow_path: str,
    max_resp_bytes: int = MAX_RESPONSE_JSON_BYTES,
    max_arrow_bytes: int = MAX_RESULT_ARROW_BYTES,
    max_total_bytes: int = MAX_TOTAL_IPC_BYTES,
) -> Tuple[bool, str]:
    """
    Rigorously validates output files produced by the worker before the host reads them.
    Guarantees:
      - Symlinks are rejected to prevent arbitrary host file reading / path traversal attacks.
      - Non-regular files (FIFOs, device files, directories) are rejected.
      - Individual file sizes are strictly bounded before reading into host memory.
      - Aggregate directory storage budget is enforced.
    """
    if not os.path.exists(out_dir) or not os.path.isdir(out_dir):
        return False, "Output directory does not exist or is not a directory."

    if not os.path.exists(resp_path):
        return False, "Worker did not return response envelope."

    if os.path.islink(resp_path):
        return False, "Response envelope cannot be a symbolic link."

    if not os.path.isfile(resp_path):
        return False, "Response envelope path is not a regular file."

    resp_size = os.path.getsize(resp_path)
    if resp_size > max_resp_bytes:
        return False, f"Response envelope size ({resp_size} bytes) exceeds limit ({max_resp_bytes} bytes)."

    if os.path.exists(output_arrow_path):
        if os.path.islink(output_arrow_path):
            return False, "Output tabular payload cannot be a symbolic link."
        if not os.path.isfile(output_arrow_path):
            return False, "Output tabular payload path is not a regular file."
        arrow_size = os.path.getsize(output_arrow_path)
        if arrow_size > max_arrow_bytes:
            return False, f"Output tabular payload size ({arrow_size} bytes) exceeds limit ({max_arrow_bytes} bytes)."

    try:
        total_out_size = sum(
            os.path.getsize(os.path.join(out_dir, f))
            for f in os.listdir(out_dir)
            if os.path.isfile(os.path.join(out_dir, f)) and not os.path.islink(os.path.join(out_dir, f))
        )
        if total_out_size > max_total_bytes:
            return False, f"Total output IPC storage ({total_out_size} bytes) exceeds limit ({max_total_bytes} bytes)."
    except OSError as e:
        return False, f"Failed to verify output directory storage: {str(e)}"

    return True, ""


class SandboxExecutor:
    """
    Centralized execution orchestration engine for AI-generated pandas code.
    Shared across FastAPI (backend/main.py) and Streamlit (app.py).

    Enforces:
      - Admission constraints (code size, row/column counts, memory admission).
      - AST static code safety validation.
      - Process-local concurrency limits via threading.BoundedSemaphore with queue timeout
        and saturation rejection (Note: this coordinates execution strictly within the
        current Python host process; it does not enforce machine-wide or multi-worker limits).
      - Ephemeral Arrow IPC and JSON envelope exchange with write-time budgets.
      - Separated IPC permissions: read-only input directory and narrowly scoped writable output directory.
      - Race-free OS-level containment:
          * On Windows: worker is created suspended (CREATE_SUSPENDED), assigned to a configured
            Windows Job Object with strict handle inheritance filtering, and resumed only after assignment.
          * On Linux: dedicated, short-lived Docker container with non-root user, --network none,
            --read-only rootfs, dropped capabilities, cgroups v2 memory/pids/cpu limits, and no-new-privileges.
          * Fail-Closed Policy: If required isolation runtime (e.g. Docker on Linux) is unavailable,
            execution is strictly rejected. Never falls back to unrestricted host subprocess execution.
      - Hard wall-clock timeout and process-tree/container termination.
      - Bounded numeric, string, collection, and response serialization limits.
      - Host-side output file validation rejecting symbolic links and size violations before reading.
      - Zero fallback to in-process execution.

    Host Security Notice:
      - The API host process communicates with the Docker daemon. To protect the host, the Docker socket
        (/var/run/docker.sock) is NEVER mounted into worker containers. For production deployments,
        using Rootless Docker or an API-level socket proxy with a strict endpoint allowlist is recommended.
      - Docker isolation does not eliminate shared Linux kernel attack surfaces (e.g., kernel 0-day bugs).
        For public multi-tenant platforms, microVMs (e.g., Firecracker) or gVisor are recommended.
    """
    _instance: Optional["SandboxExecutor"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        max_concurrent_workers: int = MAX_CONCURRENT_WORKERS,
        queue_timeout_seconds: float = DEFAULT_QUEUE_TIMEOUT_SECONDS,
        worker_memory_limit_bytes: int = MAX_WORKER_MEMORY_BYTES,
        worker_process_limit: int = MAX_WORKER_PROCESSES,
        enable_job_object: bool = True,
        backend: Optional[str] = None,
        docker_image: str = DOCKER_SANDBOX_IMAGE,
    ):
        self.max_concurrent_workers = max_concurrent_workers
        self.queue_timeout_seconds = queue_timeout_seconds
        self.worker_memory_limit_bytes = worker_memory_limit_bytes
        self.worker_process_limit = worker_process_limit
        self.enable_job_object = enable_job_object
        self.docker_image = docker_image

        env_backend = os.environ.get("SANDBOX_BACKEND", "").lower().strip()
        if backend:
            self.backend = backend.lower().strip()
        elif env_backend:
            self.backend = env_backend
        elif sys.platform == "win32":
            self.backend = "windows"
        else:
            self.backend = "docker"

        self._semaphore = threading.BoundedSemaphore(self.max_concurrent_workers)

    @classmethod
    def get_instance(cls) -> "SandboxExecutor":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Resets the singleton executor instance (primarily for isolated test fixtures)."""
        with cls._lock:
            cls._instance = None

    def execute(
        self,
        code_str: str,
        df: pd.DataFrame,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> Tuple[bool, Any]:
        """
        Executes validated pandas code under the shared concurrency and containment policy.
        """
        acquired = self._semaphore.acquire(timeout=self.queue_timeout_seconds)
        if not acquired:
            return False, f"Execution failed: Execution queue saturated (exceeded concurrency limit of {self.max_concurrent_workers}). Please retry shortly."

        try:
            return self._execute_internal(code_str, df, timeout_seconds)
        finally:
            self._semaphore.release()

    def _execute_internal(
        self,
        code_str: str,
        df: pd.DataFrame,
        timeout_seconds: float,
    ) -> Tuple[bool, Any]:
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

        # 2. Ephemeral IPC directory setup with deterministic cleanup and separated permissions
        try:
            with tempfile.TemporaryDirectory(prefix="agy_ipc_") as ipc_dir:
                in_dir = os.path.join(ipc_dir, "in")
                out_dir = os.path.join(ipc_dir, "out")
                os.makedirs(in_dir, exist_ok=True)
                os.makedirs(out_dir, exist_ok=True)

                req_path = os.path.join(in_dir, "request.json")
                input_arrow_path = os.path.join(in_dir, "input_data.arrow")
                resp_path = os.path.join(out_dir, "response.json")
                output_arrow_path = os.path.join(out_dir, "output_data.arrow")

                # Write bounded request envelope into read-only input location
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

                # Pre-launch total input storage check
                initial_ipc_size = sum(
                    os.path.getsize(os.path.join(in_dir, f))
                    for f in os.listdir(in_dir)
                    if os.path.isfile(os.path.join(in_dir, f))
                )
                if initial_ipc_size > MAX_TOTAL_IPC_BYTES:
                    return False, f"Execution setup failed: Initial IPC storage ({initial_ipc_size // (1024*1024)} MiB) exceeds total limit ({MAX_TOTAL_IPC_BYTES // (1024*1024)} MiB)."

                # 3. Spawn isolated worker with platform containment
                job = None
                container_name = None
                proc = None

                if self.backend == "docker":
                    if not _is_docker_available():
                        return False, (
                            "Execution setup failed: Linux execution requires Docker containment, "
                            "but Docker daemon is unreachable or docker command is not found. "
                            "Execution rejected under fail-closed security policy."
                        )
                    docker_cmd, container_name = _build_docker_cmd(
                        in_dir,
                        out_dir,
                        image_name=self.docker_image,
                        timeout_seconds=timeout_seconds,
                    )
                    try:
                        proc = subprocess.Popen(
                            docker_cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    except Exception as e:
                        return False, f"Execution failed: Could not launch Docker sandbox container: {str(e)}"

                elif self.backend == "windows":
                    if sys.platform != "win32":
                        return False, "Execution setup failed: Windows Job Object backend is only supported on Windows hosts."
                    cmd = [
                        sys.executable,
                        "-u",
                        str(WORKER_SCRIPT),
                        "--ipc-dir",
                        ipc_dir,
                        "--in-dir",
                        in_dir,
                        "--out-dir",
                        out_dir,
                    ]
                    safe_env = _get_safe_env()
                    if self.enable_job_object:
                        try:
                            job = WindowsJobObject(
                                memory_limit_bytes=self.worker_memory_limit_bytes,
                                max_processes=self.worker_process_limit,
                            ).open()
                        except Exception as e:
                            return False, f"Execution setup failed: Could not configure Windows Job Object containment: {str(e)}"

                    try:
                        proc = _spawn_suspended_windows_process(cmd, safe_env, job=job)
                    except Exception as e:
                        if job is not None:
                            job.close()
                        err_msg = str(e)
                        if "could not assign worker to windows job object" in err_msg.lower():
                            return False, f"Execution failed: {err_msg}"
                        return False, f"Execution failed: Could not launch worker process: {err_msg}"
                else:
                    return False, f"Execution setup failed: Unrecognized sandbox backend '{self.backend}'."

                # 4. Wait for execution under hard wall-clock deadline
                try:
                    try:
                        proc.wait(timeout=timeout_seconds)
                    except subprocess.TimeoutExpired:
                        if container_name:
                            _kill_and_remove_docker_container(container_name)
                        else:
                            _terminate_process_tree(proc, job=job)
                        return False, f"Execution timed out after {timeout_seconds} seconds."
                    except Exception as e:
                        if container_name:
                            _kill_and_remove_docker_container(container_name)
                        else:
                            _terminate_process_tree(proc, job=job)
                        return False, f"Execution process failure: {str(e)}"
                finally:
                    if container_name:
                        _kill_and_remove_docker_container(container_name)
                    if job is not None:
                        job.close()
                    if hasattr(proc, "close"):
                        proc.close()

                # Resolve output paths (handling legacy/mock test workers that write to ipc_dir or in_dir)
                if not os.path.exists(resp_path):
                    if os.path.exists(os.path.join(ipc_dir, "response.json")):
                        resp_path = os.path.join(ipc_dir, "response.json")
                        out_dir = ipc_dir
                    elif os.path.exists(os.path.join(in_dir, "response.json")):
                        resp_path = os.path.join(in_dir, "response.json")
                        out_dir = in_dir

                if not os.path.exists(output_arrow_path):
                    if os.path.exists(os.path.join(ipc_dir, "output_data.arrow")):
                        output_arrow_path = os.path.join(ipc_dir, "output_data.arrow")
                    elif os.path.exists(os.path.join(in_dir, "output_data.arrow")):
                        output_arrow_path = os.path.join(in_dir, "output_data.arrow")

                # 5. Process exit verification and error diagnostic handling
                if proc.returncode != 0:
                    if (
                        os.path.exists(resp_path)
                        and not os.path.islink(resp_path)
                        and os.path.getsize(resp_path) <= MAX_RESPONSE_JSON_BYTES
                    ):
                        try:
                            with open(resp_path, "r", encoding="utf-8") as f:
                                data = json.load(f)
                            if data.get("error_message"):
                                return False, data["error_message"]
                        except Exception:
                            pass

                    return False, f"Execution worker exited with code {proc.returncode}."

                # 6. Rigorous host-side validation of output files before reading
                is_valid_out, out_err = _validate_host_output_files(
                    out_dir,
                    resp_path,
                    output_arrow_path,
                    max_resp_bytes=MAX_RESPONSE_JSON_BYTES,
                    max_arrow_bytes=MAX_RESULT_ARROW_BYTES,
                    max_total_bytes=MAX_TOTAL_IPC_BYTES,
                )
                if not is_valid_out:
                    return False, f"Execution failed: {out_err}"

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


def execute_pandas_code(
    code_str: str,
    df: pd.DataFrame,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
) -> Tuple[bool, Any]:
    """
    Executes validated pandas code in an isolated subprocess worker with strict resource budgeting.
    Delegates to the unified SandboxExecutor singleton shared across FastAPI and Streamlit.

    Args:
        code_str (str): AST-validated Python pandas code.
        df (pd.DataFrame): Input dataset DataFrame.
        timeout_seconds (float): Maximum wall-clock execution time allowed.

    Returns:
        Tuple[bool, Any]: (success_flag, result_object_or_user_friendly_error)
    """
    return SandboxExecutor.get_instance().execute(
        code_str=code_str,
        df=df,
        timeout_seconds=timeout_seconds,
    )
