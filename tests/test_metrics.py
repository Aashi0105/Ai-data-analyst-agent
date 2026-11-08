"""
tests/test_metrics.py
---------------------
Tests for lightweight product metrics logging and GET /api/metrics/summary endpoint.
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

import backend.metrics as metrics_mod
from backend.metrics import log_event
from backend.main import app


@pytest.fixture
def client():
    """Provides a TestClient for the FastAPI app."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def cleanup_metrics():
    """Ensure any metrics_log.json created during a test is cleaned up before and after."""
    orig_path = metrics_mod.METRICS_FILE
    if metrics_mod.METRICS_FILE.exists():
        try:
            metrics_mod.METRICS_FILE.unlink()
        except OSError:
            pass
    yield
    metrics_mod.METRICS_FILE = orig_path
    if metrics_mod.METRICS_FILE.exists():
        try:
            metrics_mod.METRICS_FILE.unlink()
        except OSError:
            pass


def test_log_event_creates_file_with_correct_structure():
    """Call log_event and verify metrics_log.json is created with proper list structure and keys."""
    log_event("test_event", {"key": "value"})

    assert metrics_mod.METRICS_FILE.exists()
    with open(metrics_mod.METRICS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) == 1
    entry = data[0]
    assert "timestamp" in entry
    assert entry.get("event") == "test_event"
    assert entry.get("key") == "value"


def test_log_event_appends_multiple_entries():
    """Call log_event twice with different events and verify entries are appended."""
    log_event("event_one", {"data_a": 123})
    log_event("event_two", {"data_b": 456})

    assert metrics_mod.METRICS_FILE.exists()
    with open(metrics_mod.METRICS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) == 2
    events = [entry.get("event") for entry in data]
    assert "event_one" in events
    assert "event_two" in events


def test_log_event_never_raises_on_bad_json():
    """Corrupt metrics_log.json with invalid JSON and verify log_event recovers without raising."""
    with open(metrics_mod.METRICS_FILE, "w", encoding="utf-8") as f:
        f.write("{invalid-json-content;;;}")

    # Must not raise
    log_event("recovered_event", {"status": "ok"})

    assert metrics_mod.METRICS_FILE.exists()
    with open(metrics_mod.METRICS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0].get("event") == "recovered_event"


def test_log_event_never_raises_on_missing_dir():
    """Verify log_event fails silently and never raises even when METRICS_FILE is in a nonexistent directory."""
    orig_path = metrics_mod.METRICS_FILE
    try:
        metrics_mod.METRICS_FILE = Path("/nonexistent_directory_987654321/sub/metrics_log.json")
        # Must not raise
        log_event("test_missing_dir", {"payload": 1})
    finally:
        metrics_mod.METRICS_FILE = orig_path


def test_metrics_summary_endpoint_returns_zeros_when_no_file(client):
    """GET /api/metrics/summary returns HTTP 200 with all zeros when metrics_log.json does not exist."""
    if metrics_mod.METRICS_FILE.exists():
        metrics_mod.METRICS_FILE.unlink()

    response = client.get("/api/metrics/summary")
    assert response.status_code == 200
    data = response.json()

    assert data["total_uploads"] == 0
    assert data["sample_loads"] == 0
    assert data["total_questions"] == 0
    assert data["total_successes"] == 0
    assert data["total_failures"] == 0
    assert data["success_rate_percent"] == 0.0
    assert data["avg_time_to_result_seconds"] == 0.0
    assert data["quota_hit_count"] == 0
    assert data["examples_viewed_count"] == 0
    assert data["data_deletions"] == 0
    assert data["log_entries_total"] == 0


def test_metrics_summary_endpoint_counts_correctly(client):
    """GET /api/metrics/summary correctly aggregates a known set of logged events."""
    known_events = [
        {"timestamp": "2026-11-08T12:00:00Z", "event": "dataset_uploaded", "dataset_id": "d1", "row_count": 100, "column_count": 5, "is_sample": False},
        {"timestamp": "2026-11-08T12:01:00Z", "event": "dataset_uploaded", "dataset_id": "d2", "row_count": 200, "column_count": 6, "is_sample": False},
        {"timestamp": "2026-11-08T12:02:00Z", "event": "question_asked", "dataset_id": "d1", "question_length": 25, "session_id": "d1"},
        {"timestamp": "2026-11-08T12:02:02Z", "event": "analysis_success", "dataset_id": "d1", "time_to_result_seconds": 1.5, "had_chart": True, "session_id": "d1"},
        {"timestamp": "2026-11-08T12:03:00Z", "event": "question_asked", "dataset_id": "d1", "question_length": 30, "session_id": "d1"},
        {"timestamp": "2026-11-08T12:03:02Z", "event": "analysis_success", "dataset_id": "d1", "time_to_result_seconds": 2.5, "had_chart": False, "session_id": "d1"},
        {"timestamp": "2026-11-08T12:04:00Z", "event": "question_asked", "dataset_id": "d2", "question_length": 15, "session_id": "d2"},
        {"timestamp": "2026-11-08T12:04:01Z", "event": "analysis_failed", "dataset_id": "d2", "error_type": "ValueError", "session_id": "d2"},
    ]
    with open(metrics_mod.METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(known_events, f)

    response = client.get("/api/metrics/summary")
    assert response.status_code == 200
    data = response.json()

    assert data["total_uploads"] == 2
    assert data["sample_loads"] == 0
    assert data["total_questions"] == 3
    assert data["total_successes"] == 2
    assert data["total_failures"] == 1
    assert data["success_rate_percent"] == 66.7
    assert data["avg_time_to_result_seconds"] == 2.0
    assert data["quota_hit_count"] == 0
    assert data["examples_viewed_count"] == 0
    assert data["data_deletions"] == 0
    assert data["log_entries_total"] == 8

