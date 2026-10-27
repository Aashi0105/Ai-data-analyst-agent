"""
backend/store.py
----------------
Thread-safe, memory-bounded, TTL-aware in-process dataset store.
Replaces unbounded dictionary storage with LRU eviction and active query immunity.

Architectural Notes:
- Deployment limitation: This store is process-local and held in Python memory.
  Uvicorn MUST be run in single-worker mode (i.e., `--workers 1`).
  Multi-process or horizontally scaled deployments require a distributed
  cache or object store (e.g. Redis + S3/MinIO).
- Memory calculation: Uses pandas `df.memory_usage(deep=True).sum()`.
  This measures DataFrame internal buffer allocations. It is an estimate of
  heap consumption, not a hard operating system process-level memory limit.
- Active analysis protection: Datasets actively being analyzed by Gemini or
  executing in the sandbox have an active query reference count > 0.
  Active datasets are immune from LRU eviction and manual deletion.
"""

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import threading
import time
import uuid
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple
import pandas as pd


_DEFAULT = object()


# ============================================================================
# Exceptions
# ============================================================================

class DatasetStoreError(Exception):
    """Base exception for all dataset store operations."""
    pass


class DatasetNotFoundError(DatasetStoreError, KeyError):
    """Raised when the requested dataset ID does not exist in store."""
    pass


class DatasetSessionExpiredError(DatasetStoreError, KeyError):
    """Raised when a dataset has exceeded its idle time-to-live (TTL)."""
    pass


class DatasetInUseError(DatasetStoreError):
    """Raised when an operation (e.g., deletion) is rejected because the dataset is actively in use."""
    pass


class StorageCapacityExhaustedError(DatasetStoreError):
    """Raised when store capacity cannot be acquired because all candidates are actively in use."""
    pass


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class StoredDataset:
    """
    Metadata and DataFrame payload for an in-memory dataset entry.
    Provides dict-like subscription for backward-compatibility with existing code.
    """
    dataset_id: str
    df: pd.DataFrame
    filename: str
    uploaded_at: str
    last_accessed: float
    memory_bytes: int
    active_queries: int = 0
    is_sample: bool = False
    profile: Optional[Any] = None
    understanding: Optional[Any] = None
    insights: Optional[Any] = None

    def __getitem__(self, key: str) -> Any:
        """Emulate legacy dictionary access for backwards compatibility."""
        if key == "df":
            return self.df
        elif key == "filename":
            return self.filename
        elif key == "uploaded_at":
            return self.uploaded_at
        elif key == "dataset_id":
            return self.dataset_id
        elif key == "memory_bytes":
            return self.memory_bytes
        elif key == "active_queries":
            return self.active_queries
        elif key == "last_accessed":
            return self.last_accessed
        elif key == "is_sample":
            return self.is_sample
        elif key == "profile":
            return self.profile
        elif key == "understanding":
            return self.understanding
        elif key == "insights":
            return self.insights
        raise KeyError(f"StoredDataset has no key '{key}'")

    def __contains__(self, key: str) -> bool:
        return key in ("df", "filename", "uploaded_at", "dataset_id", "memory_bytes", "active_queries", "last_accessed", "is_sample", "profile", "understanding", "insights")

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> List[str]:
        return ["df", "filename", "uploaded_at", "dataset_id", "memory_bytes", "active_queries", "last_accessed", "is_sample", "profile", "understanding", "insights"]

    def values(self) -> List[Any]:
        return [self[k] for k in self.keys()]

    def items(self) -> List[Tuple[str, Any]]:
        return [(k, self[k]) for k in self.keys()]


# ============================================================================
# Bounded Dataset Store
# ============================================================================

class BoundedDatasetStore:
    """
    Thread-safe, bounded, TTL-aware dataset storage with LRU eviction of inactive entries.

    Guarantees:
    1. Bounds on item count (max_datasets) and estimated memory (max_memory_bytes).
    2. Lazy TTL expiration on access and capacity checks.
    3. Active analysis immunity: active datasets (active_queries > 0) are never evicted
       or manually deleted during analysis.
    4. Exception-safe acquire/release semantics via `acquire_active()` context manager.
    5. Store lock is NEVER held during code generation or sandbox execution.
    6. Dict-like interface compatibility so legacy tests and `len(DATASET_STORE)` continue to work.
    7. Strict all-or-nothing insertion: existing datasets are never evicted if the incoming
       dataset cannot be accommodated.
    """

    def __init__(
        self,
        max_datasets: int = 15,
        max_memory_bytes: int = 512 * 1024 * 1024,  # 512 MiB
        ttl_seconds: float = 3600.0,                 # 1 hour idle TTL
        time_provider: Callable[[], float] = time.time
    ) -> None:
        if max_datasets < 1:
            raise ValueError("max_datasets must be at least 1")
        if max_memory_bytes < 1:
            raise ValueError("max_memory_bytes must be at least 1")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")

        self.max_datasets = max_datasets
        self.max_memory_bytes = max_memory_bytes
        self.ttl_seconds = ttl_seconds
        self._time_provider = time_provider

        self._store: OrderedDict[str, StoredDataset] = OrderedDict()
        self._total_memory_bytes: int = 0
        self._lock = threading.RLock()

    @staticmethod
    def estimate_dataframe_memory(df: pd.DataFrame) -> int:
        """
        Calculates estimated deep memory usage of the DataFrame in bytes.
        Note: This is an internal buffer allocation estimate, not a hard process-level memory limit.
        """
        try:
            return int(df.memory_usage(deep=True).sum())
        except Exception:
            return int(df.memory_usage().sum())

    # ------------------------------------------------------------------------
    # Internal Helpers (Must be called under self._lock)
    # ------------------------------------------------------------------------

    def _is_expired(self, item: StoredDataset, current_time: float) -> bool:
        return (current_time - item.last_accessed) > self.ttl_seconds

    def _purge_expired(self, current_time: float) -> List[str]:
        """Purges inactive expired datasets. Active datasets are never purged."""
        purged = []
        for ds_id, item in list(self._store.items()):
            if item.active_queries == 0 and self._is_expired(item, current_time):
                del self._store[ds_id]
                self._total_memory_bytes -= item.memory_bytes
                purged.append(ds_id)
        return purged

    # ------------------------------------------------------------------------
    # Public Core API
    # ------------------------------------------------------------------------

    def put(
        self,
        dataset_id: str,
        df: pd.DataFrame,
        filename: str,
        uploaded_at: Optional[str] = None,
        is_sample: bool = False,
        memory_bytes: Optional[int] = None
    ) -> StoredDataset:
        """
        Inserts a dataset into storage, enforcing limits and LRU eviction atomically.
        If capacity cannot be obtained, raises StorageCapacityExhaustedError without
        modifying or evicting existing store contents.
        Thread-safe.
        """
        if memory_bytes is None:
            memory_bytes = self.estimate_dataframe_memory(df)
        uploaded_str = uploaded_at or time.strftime("%Y-%m-%d %H:%M:%S")

        with self._lock:
            return self._put_locked(
                dataset_id=dataset_id,
                df=df,
                filename=filename,
                uploaded_str=uploaded_str,
                memory_bytes=memory_bytes,
                is_sample=is_sample
            )

    def _put_locked(
        self,
        dataset_id: str,
        df: pd.DataFrame,
        filename: str,
        uploaded_str: str,
        memory_bytes: int,
        is_sample: bool = False
    ) -> StoredDataset:
        """Internal insertion implementation. Must be called while holding self._lock."""
        now = self._time_provider()

        # 1. Immediate rejection if single item exceeds total capacity
        if memory_bytes > self.max_memory_bytes:
            raise StorageCapacityExhaustedError(
                f"Dataset estimated size ({memory_bytes / (1024 * 1024):.2f} MiB) "
                f"exceeds maximum allowed store capacity ({self.max_memory_bytes / (1024 * 1024):.2f} MiB)."
            )

        # 2. Check if replacing an active dataset
        if dataset_id in self._store and self._store[dataset_id].active_queries > 0:
            raise DatasetInUseError(
                f"Dataset '{dataset_id}' is actively being analyzed and cannot be overwritten."
            )

        # 3. Simulate all-or-nothing capacity reservation before mutating store
        # Identify inactive expired items as first eviction candidates
        expired_candidates = [
            k for k, item in self._store.items()
            if item.active_queries == 0 and self._is_expired(item, now)
        ]

        evict_candidates: List[str] = list(expired_candidates)
        simulated_count = len(self._store) - len(expired_candidates)
        current_mem = self._total_memory_bytes - sum(self._store[k].memory_bytes for k in expired_candidates)

        # If replacing an existing unexpired key, account for its slot and memory release
        if dataset_id in self._store and dataset_id not in expired_candidates:
            current_mem -= self._store[dataset_id].memory_bytes
        else:
            simulated_count += 1

        # Identify additional inactive LRU eviction candidates if limits exceeded
        for k, item in self._store.items():
            if simulated_count <= self.max_datasets and (current_mem + memory_bytes) <= self.max_memory_bytes:
                break
            if k == dataset_id or k in evict_candidates:
                continue
            if item.active_queries == 0:
                evict_candidates.append(k)
                simulated_count -= 1
                current_mem -= item.memory_bytes

        # Check if capacity can be satisfied
        if simulated_count > self.max_datasets or (current_mem + memory_bytes) > self.max_memory_bytes:
            # Capacity cannot be obtained. Do NOT modify the store!
            raise StorageCapacityExhaustedError(
                "Dataset storage capacity reached and all existing datasets are currently being analyzed. "
                "Please wait for ongoing analyses to complete and retry."
            )

        # 4. Apply planned evictions atomically
        for k in evict_candidates:
            removed_item = self._store.pop(k)
            self._total_memory_bytes -= removed_item.memory_bytes

        # If replacing an existing key that wasn't among evict_candidates
        if dataset_id in self._store:
            old_item = self._store.pop(dataset_id)
            self._total_memory_bytes -= old_item.memory_bytes

        stored = StoredDataset(
            dataset_id=dataset_id,
            df=df,
            filename=filename,
            uploaded_at=uploaded_str,
            last_accessed=now,
            memory_bytes=memory_bytes,
            active_queries=0,
            is_sample=is_sample
        )
        self._store[dataset_id] = stored
        self._total_memory_bytes += memory_bytes
        return stored

    def get(self, dataset_id: str, default: Any = _DEFAULT) -> Any:
        """
        Retrieves a stored dataset and updates its LRU access time.
        If default is not provided:
          Raises DatasetNotFoundError if not found.
          Raises DatasetSessionExpiredError if idle TTL expired.
        If default is provided:
          Returns default if not found or if idle TTL expired.
        Thread-safe.
        """
        now = self._time_provider()
        with self._lock:
            if dataset_id not in self._store:
                if default is not _DEFAULT:
                    return default
                raise DatasetNotFoundError(f"Dataset with ID '{dataset_id}' was not found in server memory.")

            item = self._store[dataset_id]
            if item.active_queries == 0 and self._is_expired(item, now):
                # Lazy TTL expiration
                del self._store[dataset_id]
                self._total_memory_bytes -= item.memory_bytes
                if default is not _DEFAULT:
                    return default
                raise DatasetSessionExpiredError("Dataset session expired. Please re-upload your CSV.")

            item.last_accessed = now
            self._store.move_to_end(dataset_id)
            return item

    def touch(self, dataset_id: str) -> None:
        """Touches a dataset to refresh its last_accessed timestamp without returning it."""
        self.get(dataset_id)

    def delete(self, dataset_id: str) -> None:
        """
        Explicitly removes a dataset from storage.
        Raises DatasetInUseError if dataset is currently undergoing analysis.
        Raises DatasetNotFoundError if not found.
        Raises DatasetSessionExpiredError if dataset was expired (and purges it).
        Thread-safe.
        """
        now = self._time_provider()
        with self._lock:
            if dataset_id not in self._store:
                raise DatasetNotFoundError(f"Dataset with ID '{dataset_id}' was not found in server memory.")

            item = self._store[dataset_id]
            if item.active_queries > 0:
                raise DatasetInUseError(
                    f"Dataset '{dataset_id}' is currently being analyzed and cannot be deleted."
                )

            del self._store[dataset_id]
            self._total_memory_bytes -= item.memory_bytes

            if self._is_expired(item, now):
                raise DatasetSessionExpiredError("Dataset session expired. Please re-upload your CSV.")

    @contextmanager
    def acquire_active(self, dataset_id: str) -> Generator[pd.DataFrame, None, None]:
        """
        Context manager for active analysis queries.
        Increments active_queries during analysis to protect dataset from LRU eviction
        and deletion. Releases the store lock while yielding so that concurrent requests
        and Gemini / sandbox execution do NOT block the store lock.
        Guarantees reference decrement and TTL refresh in `finally`.
        """
        now = self._time_provider()
        with self._lock:
            if dataset_id not in self._store:
                raise DatasetNotFoundError(f"Dataset '{dataset_id}' not found. Please upload a CSV first.")

            item = self._store[dataset_id]
            if item.active_queries == 0 and self._is_expired(item, now):
                del self._store[dataset_id]
                self._total_memory_bytes -= item.memory_bytes
                raise DatasetSessionExpiredError("Dataset session expired. Please re-upload your CSV.")

            item.active_queries += 1
            item.last_accessed = now
            self._store.move_to_end(dataset_id)
            df = item.df

        try:
            yield df
        finally:
            now_finish = self._time_provider()
            with self._lock:
                if dataset_id in self._store:
                    item = self._store[dataset_id]
                    item.active_queries -= 1
                    if item.active_queries <= 0:
                        item.active_queries = 0
                        item.last_accessed = now_finish
                        self._store.move_to_end(dataset_id)

    def get_sample(self) -> Optional[StoredDataset]:
        """
        Returns an existing, unexpired sample dataset if available, updating its LRU position.
        Prevents repeated sample loads from allocating duplicate DataFrames.
        If a sample entry has expired, purges it and continues scanning for another
        unexpired sample rather than immediately returning None.
        """
        now = self._time_provider()
        with self._lock:
            for ds_id, item in list(self._store.items()):
                if item.is_sample:
                    if item.active_queries == 0 and self._is_expired(item, now):
                        del self._store[ds_id]
                        self._total_memory_bytes -= item.memory_bytes
                        continue  # purged; keep scanning for another unexpired sample
                    item.last_accessed = now
                    self._store.move_to_end(ds_id)
                    return item
            return None

    def put_or_reuse_sample(self, df: pd.DataFrame, filename: str) -> StoredDataset:
        """
        Atomically registers a sample dataset, or reuses an existing unexpired sample.
        Memory estimation occurs outside the store lock to avoid lock contention,
        while sample deduplication and capacity reservation remain atomic.
        """
        # Fast-path check without acquiring write lock or estimating memory
        existing = self.get_sample()
        if existing is not None:
            return existing

        # Expensive deep memory estimation outside the lock
        memory_bytes = self.estimate_dataframe_memory(df)
        uploaded_str = time.strftime("%Y-%m-%d %H:%M:%S")

        with self._lock:
            # Double-check if another thread registered a sample while memory was being estimated
            existing = self.get_sample()
            if existing is not None:
                return existing
            dataset_id = str(uuid.uuid4())
            return self._put_locked(
                dataset_id=dataset_id,
                df=df,
                filename=filename,
                uploaded_str=uploaded_str,
                memory_bytes=memory_bytes,
                is_sample=True
            )

    def pop(self, dataset_id: str, default: Any = _DEFAULT) -> Any:
        """Removes specified dataset and returns it, or returns default if not found."""
        with self._lock:
            try:
                item = self.get(dataset_id)
                self.delete(dataset_id)
                return item
            except (DatasetNotFoundError, DatasetSessionExpiredError):
                if default is not _DEFAULT:
                    return default
                raise KeyError(f"Dataset '{dataset_id}' not found.")

    def clear(self) -> None:
        """
        Clears all stored datasets and resets retained memory counter.
        Refuses to clear if any dataset is actively undergoing analysis,
        raising DatasetInUseError to prevent invalidating in-flight queries.
        """
        with self._lock:
            if any(item.active_queries > 0 for item in self._store.values()):
                raise DatasetInUseError("Cannot clear dataset store while active analysis queries are in progress.")
            self._store.clear()
            self._total_memory_bytes = 0

    # ------------------------------------------------------------------------
    # Properties & Status
    # ------------------------------------------------------------------------

    @property
    def total_memory_bytes(self) -> int:
        with self._lock:
            return self._total_memory_bytes

    @property
    def total_memory_mb(self) -> float:
        with self._lock:
            return round(self._total_memory_bytes / (1024 * 1024), 2)

    @property
    def active_count(self) -> int:
        with self._lock:
            return sum(1 for item in self._store.values() if item.active_queries > 0)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "dataset_count": len(self._store),
                "max_datasets": self.max_datasets,
                "total_memory_bytes": self._total_memory_bytes,
                "total_memory_mb": self.total_memory_mb,
                "max_memory_bytes": self.max_memory_bytes,
                "max_memory_mb": round(self.max_memory_bytes / (1024 * 1024), 2),
                "active_queries": self.active_count,
                "ttl_seconds": self.ttl_seconds,
            }

    # ------------------------------------------------------------------------
    # Dictionary Compatibility Layer
    # ------------------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            # Purge any expired items for accurate count
            now = self._time_provider()
            self._purge_expired(now)
            return len(self._store)

    def __contains__(self, dataset_id: Any) -> bool:
        if not isinstance(dataset_id, str):
            return False
        now = self._time_provider()
        with self._lock:
            if dataset_id not in self._store:
                return False
            item = self._store[dataset_id]
            if item.active_queries == 0 and self._is_expired(item, now):
                del self._store[dataset_id]
                self._total_memory_bytes -= item.memory_bytes
                return False
            return True

    def __getitem__(self, dataset_id: str) -> StoredDataset:
        return self.get(dataset_id)

    def __setitem__(self, dataset_id: str, value: Any) -> None:
        if isinstance(value, StoredDataset):
            self.put(
                dataset_id=dataset_id,
                df=value.df,
                filename=value.filename,
                uploaded_at=value.uploaded_at,
                is_sample=value.is_sample
            )
        elif isinstance(value, dict):
            self.put(
                dataset_id=dataset_id,
                df=value["df"],
                filename=value.get("filename", "dataset.csv"),
                uploaded_at=value.get("uploaded_at"),
                is_sample=value.get("is_sample", False)
            )
        else:
            raise TypeError(f"Cannot store value of type {type(value)}")

    def __delitem__(self, dataset_id: str) -> None:
        self.delete(dataset_id)

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            now = self._time_provider()
            self._purge_expired(now)
            return iter(list(self._store.keys()))

    def __bool__(self) -> bool:
        return len(self) > 0

    def copy(self) -> Dict[str, StoredDataset]:
        with self._lock:
            now = self._time_provider()
            self._purge_expired(now)
            return dict(self._store)

    def keys(self) -> List[str]:
        with self._lock:
            now = self._time_provider()
            self._purge_expired(now)
            return list(self._store.keys())

    def values(self) -> List[StoredDataset]:
        with self._lock:
            now = self._time_provider()
            self._purge_expired(now)
            return list(self._store.values())

    def items(self) -> List[Tuple[str, StoredDataset]]:
        with self._lock:
            now = self._time_provider()
            self._purge_expired(now)
            return list(self._store.items())
