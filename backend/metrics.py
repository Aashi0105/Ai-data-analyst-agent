import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

METRICS_FILE = Path("metrics_log.json")
_lock = threading.Lock()

def log_event(event_type: str, data: dict) -> None:
    """
    Log a product metric event.
    Thread-safe, non-blocking, fail-silent.
    Never raises an exception.
    """
    try:
        entry = {
            "timestamp": datetime.now(timezone.utc)
                         .isoformat(),
            "event": event_type,
            **data
        }
        with _lock:
            existing = []
            if METRICS_FILE.exists():
                try:
                    with open(METRICS_FILE, "r",
                              encoding="utf-8") as f:
                        existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = []
                except (json.JSONDecodeError, 
                        OSError):
                    existing = []
            existing.append(entry)
            with open(METRICS_FILE, "w",
                      encoding="utf-8") as f:
                json.dump(existing, f, 
                          indent=2, 
                          default=str)
    except Exception:
        pass  # Never crash the main application

