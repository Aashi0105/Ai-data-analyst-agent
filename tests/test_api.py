"""
tests/test_api.py
-----------------
Integration tests for the FastAPI REST API layer (backend/main.py).
Tests dataset uploads, sample loading, health checks, error responses,
and schema profiling.
"""

import io
import pytest
from fastapi.testclient import TestClient
import pandas as pd

from backend.main import app, DATASET_STORE


@pytest.fixture
def client():
    """Provides a TestClient for the FastAPI app."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_gemini_api_keys(monkeypatch):
    """Set dummy Gemini API keys so API tests don't depend on a real .env key."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")


def test_health_check(client):
    """Verify health check endpoint returns 200 and expected status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "datasets_in_memory" in data


def test_upload_csv(client):
    """Verify uploading a CSV file returns dataset ID, summary KPIs, and schema."""
    csv_content = b"region,revenue,quantity\nNorth,500.0,5\nSouth,300.0,3\nEast,400.0,4"
    file_obj = io.BytesIO(csv_content)

    response = client.post(
        "/api/datasets/upload",
        files={"file": ("test_data.csv", file_obj, "text/csv")}
    )

    assert response.status_code == 200
    data = response.json()
    assert "dataset_id" in data
    assert data["filename"] == "test_data.csv"
    assert data["num_rows"] == 3
    assert data["num_cols"] == 3
    assert "region" in data["columns"]
    assert "revenue" in data["columns"]
    assert len(data["schema"]) == 3
    assert len(data["preview"]) == 3


def test_upload_invalid_extension(client):
    """Verify uploading a non-CSV file is rejected with a 400 error."""
    file_obj = io.BytesIO(b"some content")
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("document.txt", file_obj, "text/plain")}
    )
    assert response.status_code == 400
    assert "Only CSV files" in response.json()["detail"]


def test_load_sample_dataset(client):
    """Verify the sample dataset endpoint loads ecommerce_sales.csv with correct dimensions."""
    response = client.post("/api/datasets/sample")
    assert response.status_code == 200
    data = response.json()
    assert "dataset_id" in data
    assert "ecommerce_sales.csv" in data["filename"]
    # Verify dimensions derived from actual synthetic benchmark dataset
    assert data["num_rows"] == 100
    assert data["num_cols"] == 13
    assert "order_id" in data["columns"]
    assert "revenue" in data["columns"]


def test_get_dataset_by_id(client):
    """Verify retrieving an uploaded dataset by its UUID."""
    # First load sample
    sample_res = client.post("/api/datasets/sample")
    ds_id = sample_res.json()["dataset_id"]

    # Now get it
    get_res = client.get(f"/api/datasets/{ds_id}")
    assert get_res.status_code == 200
    assert get_res.json()["dataset_id"] == ds_id
    assert get_res.json()["num_rows"] == 100


def test_get_nonexistent_dataset(client):
    """Verify 404 response for unknown dataset ID."""
    response = client.get("/api/datasets/non-existent-uuid-12345")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_analyze_nonexistent_dataset(client):
    """Verify analyze endpoint returns 404 for unknown dataset ID."""
    response = client.post(
        "/api/analyze",
        json={"dataset_id": "non-existent-id", "question": "What is total sales?"}
    )
    assert response.status_code == 404


def test_analyze_quota_exceeded_structured_error(client, monkeypatch):
    """
    Verify that when Gemini raises 429 / RESOURCE_EXHAUSTED / GenerateRequestsPerDay,
    the API returns a clean structured 429 response without leaking stack traces or keys.
    """
    sample_res = client.post("/api/datasets/sample")
    ds_id = sample_res.json()["dataset_id"]

    from agent import DataAnalystAgent

    def mock_generate_code(self, df, question, max_retries=3):
        raise RuntimeError(
            "Gemini API daily quota exceeded (429 RESOURCE_EXHAUSTED): "
            "GenerateRequestsPerDayPerModel-FreeTier limit: 20 per day. key=AIzaSyFakeKey12345"
        )

    monkeypatch.setattr(DataAnalystAgent, "generate_code", mock_generate_code)

    response = client.post(
        "/api/analyze",
        json={"dataset_id": ds_id, "question": "Which region generated the most revenue?"}
    )

    assert response.status_code == 429
    data = response.json()
    assert data["error"] == "quota_exceeded"
    assert "usage limit has been reached" in data["message"]
    assert data["retryable"] is False
    # Verify no secret key or stack trace dump leaked
    assert "AIzaSyFakeKey12345" not in response.text
    assert "Traceback" not in response.text
