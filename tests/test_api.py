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
import numpy as np
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


def test_upload_empty_file_rejected(client):
    """Verify that uploading an empty CSV (0 bytes) is rejected with HTTP 400."""
    file_obj = io.BytesIO(b"")
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("empty.csv", file_obj, "text/csv")}
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower() or "no data" in response.json()["detail"].lower()


def test_upload_exact_size_boundary(client, monkeypatch):
    """Verify an upload at the exact accepted byte boundary succeeds."""
    import backend.main as backend_main
    csv_content = b"col1,col2\nval1,10\nval2,20"
    content_len = len(csv_content)
    monkeypatch.setattr(backend_main, "MAX_UPLOAD_SIZE_BYTES", content_len)
    monkeypatch.setattr(backend_main, "MAX_UPLOAD_SIZE_MB", round(content_len / (1024 * 1024), 4))

    file_obj = io.BytesIO(csv_content)
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("boundary.csv", file_obj, "text/csv")}
    )
    assert response.status_code == 200
    assert response.json()["num_rows"] == 2


def test_upload_oversized_rejected_with_413(client, monkeypatch):
    """Verify an upload exceeding the size limit is rejected with HTTP 413 and no dataset leak."""
    import backend.main as backend_main
    csv_content = b"col1,col2\nval1,10\nval2,20\nval3,30"
    monkeypatch.setattr(backend_main, "MAX_UPLOAD_SIZE_BYTES", 15)
    monkeypatch.setattr(backend_main, "MAX_UPLOAD_SIZE_MB", 25)

    initial_datasets_count = len(backend_main.DATASET_STORE)
    file_obj = io.BytesIO(csv_content)
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("oversized.csv", file_obj, "text/csv")}
    )

    assert response.status_code == 413
    assert "exceeds" in response.json()["detail"].lower()
    # Ensure no dataset entry was created in DATASET_STORE
    assert len(backend_main.DATASET_STORE) == initial_datasets_count


def test_upload_oversized_misleading_content_length_rejected(client, monkeypatch):
    """
    Verify that an upload with a misleadingly small Content-Length header
    is still caught and rejected with HTTP 413 during stream reading.
    """
    import backend.main as backend_main
    csv_content = b"col1,col2\nval1,10\nval2,20\nval3,30"
    monkeypatch.setattr(backend_main, "MAX_UPLOAD_SIZE_BYTES", 15)

    initial_datasets_count = len(backend_main.DATASET_STORE)
    file_obj = io.BytesIO(csv_content)
    # Header claims 5 bytes, but payload is larger than limit (15)
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("spoofed.csv", file_obj, "text/csv")},
        headers={"content-length": "5"}
    )

    assert response.status_code == 413
    assert "exceeds" in response.json()["detail"].lower()
    assert len(backend_main.DATASET_STORE) == initial_datasets_count


def test_upload_oversized_content_length_early_rejection(client, monkeypatch):
    """Verify that an oversized Content-Length header triggers early 413 rejection."""
    import backend.main as backend_main
    monkeypatch.setattr(backend_main, "MAX_UPLOAD_SIZE_BYTES", 50)

    initial_datasets_count = len(backend_main.DATASET_STORE)
    file_obj = io.BytesIO(b"col1,col2\n1,2")
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("declared_large.csv", file_obj, "text/csv")},
        headers={"content-length": str(100 * 1024 * 1024)}
    )

    assert response.status_code == 413
    assert "exceeds" in response.json()["detail"].lower()
    assert len(backend_main.DATASET_STORE) == initial_datasets_count


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


# ============================================================================
# End-to-End Analysis Pipeline Verification (/api/analyze)
# ============================================================================

def test_analyze_successful_end_to_end_pipeline(client, monkeypatch):
    """
    Verify complete successful end-to-end analysis pipeline:
    DataAnalystAgent (mocked) -> AST validation (real) -> Sandbox execution (real)
    -> Result formatting (real) -> Recharts visualization builder (real) -> explain_answer (mocked).
    Asserts regional revenue values match independently verified figures for ecommerce_sales.csv.
    """
    import backend.main as backend_main
    from agent import DataAnalystAgent
    from explanation import AnswerExplanation, AnswerEvidence

    # 1. Load sample ecommerce dataset (100 rows, 13 columns)
    sample_res = client.post("/api/datasets/sample")
    assert sample_res.status_code == 200
    dataset_id = sample_res.json()["dataset_id"]

    # 2. Candidate code accepted by AST validator
    candidate_code = "result = df.groupby('region')['revenue'].sum().sort_values(ascending=False).reset_index()"

    monkeypatch.setattr(
        DataAnalystAgent,
        "generate_code",
        lambda self, df, q, max_retries=3: candidate_code
    )

    mock_exp = AnswerExplanation(
        finding="North region generated the highest revenue ($16,886.30), followed by West ($12,664.85), South ($12,607.00), and East ($11,190.70).",
        evidence=AnswerEvidence(
            metric_name="revenue",
            aggregation_type="sum",
            is_additive=True,
            total_value=53348.85,
            top_contributor="North",
            top_value=16886.30,
            runner_up="West",
            runner_up_value=12664.85,
            bottom_contributor="East",
            bottom_value=11190.70,
            num_groups=4,
            sample_size=100
        ),
        interpretation="North accounted for approximately 31.7% of total regional revenue.",
        uncertainty_or_limitations="Observational aggregation without causal controls.",
        next_steps=["Examine product category breakdowns within the North region."],
        status="deterministic"
    )
    monkeypatch.setattr(backend_main, "explain_answer", lambda **kwargs: mock_exp)

    # 3. Dispatch analytical query
    response = client.post(
        "/api/analyze",
        json={"dataset_id": dataset_id, "question": "What is the total revenue by region?"}
    )

    # 4. Assert HTTP & Pipeline Statuses
    assert response.status_code == 200
    data = response.json()
    assert data["validation_status"] == "passed"
    assert data["validation_message"] == "Code passed AST validation."
    assert data["execution_status"] == "passed"
    assert data["error_message"] is None
    assert data["result_type"] == "dataframe"
    assert data["code"] == candidate_code
    assert data["execution_time_ms"] >= 0

    # 5. Assert Independently Verified Regional Revenue Values
    table = data["table_data"]
    assert table is not None
    assert table["columns"] == ["region", "revenue"]
    rows = table["rows"]
    assert len(rows) == 4

    row_map = {r["region"]: round(float(r["revenue"]), 2) for r in rows}
    assert row_map["North"] == 16886.30
    assert row_map["West"] == 12664.85
    assert row_map["South"] == 12607.00
    assert row_map["East"] == 11190.70

    # 6. Assert Recharts Visualization Data
    viz = data["visualization"]
    assert viz is not None
    assert viz["chart_type"] == "Bar Chart"
    assert viz["x_key"] == "label"
    assert viz["y_keys"] == ["value"]
    assert "revenue by region" in viz["title"]

    viz_map = {d["label"]: round(float(d["value"]), 2) for d in viz["data"]}
    assert viz_map["North"] == 16886.30
    assert viz_map["West"] == 12664.85
    assert viz_map["South"] == 12607.00
    assert viz_map["East"] == 11190.70

    # 7. Assert Explanation Model
    assert data["explanation"] is not None
    assert data["explanation"]["finding"] == mock_exp.finding
    assert data["explanation"]["evidence"]["top_contributor"] == "North"
    assert data["explanation"]["evidence"]["top_value"] == 16886.30


def test_analyze_ast_security_violation_handled(client, monkeypatch):
    """Verify that code violating the AST safety whitelist is rejected without executing in the sandbox."""
    sample_res = client.post("/api/datasets/sample")
    dataset_id = sample_res.json()["dataset_id"]

    from agent import DataAnalystAgent
    # Disallowed import in generated code
    monkeypatch.setattr(
        DataAnalystAgent,
        "generate_code",
        lambda self, df, q, max_retries=3: "import os\nresult = df.head()"
    )

    response = client.post(
        "/api/analyze",
        json={"dataset_id": dataset_id, "question": "Attempt malicious import"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["validation_status"] == "failed"
    assert data["execution_status"] == "failed"
    assert "AST Security Violation" in data["error_message"]
    assert data["table_data"] is None
    assert data["visualization"] is None
    assert data["explanation"] is None


def test_analyze_sandbox_execution_error_handled(client, monkeypatch):
    """Verify that valid AST code causing runtime sandbox errors is captured cleanly."""
    sample_res = client.post("/api/datasets/sample")
    dataset_id = sample_res.json()["dataset_id"]

    from agent import DataAnalystAgent
    # Code passes AST check but references a nonexistent column
    monkeypatch.setattr(
        DataAnalystAgent,
        "generate_code",
        lambda self, df, q, max_retries=3: "result = df['non_existent_column_for_testing'].sum()"
    )

    response = client.post(
        "/api/analyze",
        json={"dataset_id": dataset_id, "question": "Calculate missing column"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["validation_status"] == "passed"
    assert data["execution_status"] == "failed"
    assert data["error_message"] is not None
    assert "non_existent_column_for_testing" in data["error_message"]
    assert data["table_data"] is None
    assert data["visualization"] is None
    assert data["explanation"] is None


def test_analyze_graceful_degradation_on_explanation_failure(client, monkeypatch):
    """Verify that when explanation synthesis throws an error, the core analysis response still succeeds."""
    import backend.main as backend_main
    from agent import DataAnalystAgent

    sample_res = client.post("/api/datasets/sample")
    dataset_id = sample_res.json()["dataset_id"]

    monkeypatch.setattr(
        DataAnalystAgent,
        "generate_code",
        lambda self, df, q, max_retries=3: "result = df['revenue'].sum()"
    )

    def mock_failing_explain(**kwargs):
        raise RuntimeError("Explanation LLM synthesis timed out.")

    monkeypatch.setattr(backend_main, "explain_answer", mock_failing_explain)

    response = client.post(
        "/api/analyze",
        json={"dataset_id": dataset_id, "question": "What is total revenue?"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["validation_status"] == "passed"
    assert data["execution_status"] == "passed"
    assert data["result_type"] == "scalar"
    assert data["explanation"] is None  # Degraded gracefully


def test_analyze_generic_code_generation_error_handled(client, monkeypatch):
    """Verify that generic non-quota code generation errors return an execution failure response."""
    sample_res = client.post("/api/datasets/sample")
    dataset_id = sample_res.json()["dataset_id"]

    from agent import DataAnalystAgent

    def mock_failing_generate(self, df, q, max_retries=3):
        raise ValueError("Invalid prompt parameters generated by client.")

    monkeypatch.setattr(DataAnalystAgent, "generate_code", mock_failing_generate)

    response = client.post(
        "/api/analyze",
        json={"dataset_id": dataset_id, "question": "Trigger generic error"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["validation_status"] == "failed"
    assert data["validation_message"] == "Code generation failed."
    assert data["execution_status"] == "failed"
    assert "Code generation error" in data["error_message"]


# ============================================================================
# Bounded Dataset Storage Tests (Task 08A.2.2 & 08A.2.3)
# ============================================================================

def test_repeated_sample_dataset_does_not_duplicate_retention(client):
    """Verify repeated sequential requests for sample dataset reuse existing entry and prevent unbounded growth."""
    import backend.main as backend_main
    backend_main.DATASET_STORE.clear()

    res1 = client.post("/api/datasets/sample")
    assert res1.status_code == 200
    id1 = res1.json()["dataset_id"]

    res2 = client.post("/api/datasets/sample")
    assert res2.status_code == 200
    id2 = res2.json()["dataset_id"]

    # Reuses existing unexpired sample dataset
    assert id1 == id2
    assert len(backend_main.DATASET_STORE) == 1


def test_repeated_and_concurrent_sample_dataset_requests(client):
    """Verify concurrent requests for sample dataset atomically reuse identical instance without race conditions."""
    import threading
    import backend.main as backend_main
    backend_main.DATASET_STORE.clear()

    results = []
    errors = []

    def load_worker():
        try:
            res = client.post("/api/datasets/sample")
            assert res.status_code == 200
            results.append(res.json()["dataset_id"])
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=load_worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert len(results) == 10
    # Every concurrent thread must receive the exact same dataset ID
    assert len(set(results)) == 1
    # Exactly one sample dataset retained in memory
    assert len(backend_main.DATASET_STORE) == 1


def test_delete_dataset_success(client):
    """Verify DELETE /api/datasets/{id} frees dataset and subsequent GET returns 404."""
    csv_content = b"col1,col2\nval1,10\nval2,20"
    upload_res = client.post(
        "/api/datasets/upload",
        files={"file": ("to_delete.csv", io.BytesIO(csv_content), "text/csv")}
    )
    assert upload_res.status_code == 200
    ds_id = upload_res.json()["dataset_id"]

    # Delete dataset
    del_res = client.delete(f"/api/datasets/{ds_id}")
    assert del_res.status_code == 200
    del_data = del_res.json()
    assert del_data["status"] == "deleted"
    assert del_data["dataset_id"] == ds_id

    # Subsequent GET returns 404
    get_res = client.get(f"/api/datasets/{ds_id}")
    assert get_res.status_code == 404


def test_delete_nonexistent_dataset(client):
    """Verify DELETE /api/datasets/{id} returns 404 for unknown dataset ID."""
    del_res = client.delete("/api/datasets/nonexistent-uuid-99999")
    assert del_res.status_code == 404


def test_delete_dataset_conflict_during_active_analysis(client):
    """Verify deletion attempt during active analysis returns HTTP 409 Conflict."""
    import backend.main as backend_main
    csv_content = b"col1,col2\nval1,10\nval2,20"
    upload_res = client.post(
        "/api/datasets/upload",
        files={"file": ("in_use.csv", io.BytesIO(csv_content), "text/csv")}
    )
    ds_id = upload_res.json()["dataset_id"]

    with backend_main.DATASET_STORE.acquire_active(ds_id):
        del_res = client.delete(f"/api/datasets/{ds_id}")
        assert del_res.status_code == 409
        assert "currently being analyzed" in del_res.json()["detail"].lower()

    # Once released, deletion succeeds
    del_res_after = client.delete(f"/api/datasets/{ds_id}")
    assert del_res_after.status_code == 200


def test_active_reference_released_on_success_and_exception():
    """Verify active analysis references are safely released on both normal return and exception."""
    from backend.store import BoundedDatasetStore
    store = BoundedDatasetStore(max_datasets=5)
    df = pd.DataFrame({"a": [1, 2, 3]})
    store.put("ds1", df, "test.csv")

    assert store["ds1"].active_queries == 0

    # Normal exit
    with store.acquire_active("ds1") as active_df:
        assert active_df is not None
        assert store["ds1"].active_queries == 1

    assert store["ds1"].active_queries == 0

    # Exception exit
    try:
        with store.acquire_active("ds1"):
            assert store["ds1"].active_queries == 1
            raise RuntimeError("simulated pipeline error")
    except RuntimeError:
        pass

    assert store["ds1"].active_queries == 0


def test_lru_eviction_at_item_limit(client, monkeypatch):
    """Verify least-recently-used inactive datasets are evicted when max_datasets limit is reached."""
    import backend.main as backend_main
    monkeypatch.setattr(backend_main.DATASET_STORE, "max_datasets", 3)
    backend_main.DATASET_STORE.clear()

    # Upload 3 datasets
    ids = []
    for i in range(1, 4):
        res = client.post(
            "/api/datasets/upload",
            files={"file": (f"data_{i}.csv", io.BytesIO(f"col\n{i}".encode()), "text/csv")}
        )
        assert res.status_code == 200
        ids.append(res.json()["dataset_id"])

    # Access dataset 0 (making it most recently used; dataset 1 is now LRU)
    get_res = client.get(f"/api/datasets/{ids[0]}")
    assert get_res.status_code == 200

    # Upload 4th dataset -> must evict dataset 1 (LRU)
    res4 = client.post(
        "/api/datasets/upload",
        files={"file": ("data_4.csv", io.BytesIO(b"col\n4"), "text/csv")}
    )
    assert res4.status_code == 200
    id4 = res4.json()["dataset_id"]

    # Verify dataset 1 (index 1) was evicted
    assert client.get(f"/api/datasets/{ids[1]}").status_code == 404
    # Verify dataset 0, 2, and 4 are still present
    assert client.get(f"/api/datasets/{ids[0]}").status_code == 200
    assert client.get(f"/api/datasets/{ids[2]}").status_code == 200
    assert client.get(f"/api/datasets/{id4}").status_code == 200


def test_aggregate_memory_limit_eviction():
    """Verify store evicts inactive LRU dataset when aggregate memory exceeds threshold."""
    from backend.store import BoundedDatasetStore
    df = pd.DataFrame({"col": range(100)})
    single_mem = BoundedDatasetStore.estimate_dataframe_memory(df)

    # Set capacity enough for 2 datasets, but not 3
    store = BoundedDatasetStore(
        max_datasets=10,
        max_memory_bytes=single_mem * 2 + 100
    )

    store.put("ds1", df, "ds1.csv")
    store.put("ds2", df, "ds2.csv")
    assert len(store) == 2
    assert "ds1" in store
    assert "ds2" in store

    # Putting ds3 must evict ds1 to stay within memory ceiling
    store.put("ds3", df, "ds3.csv")
    assert len(store) == 2
    assert "ds1" not in store
    assert "ds2" in store
    assert "ds3" in store
    assert store.total_memory_bytes <= store.max_memory_bytes


def test_ttl_expiration_with_mock_time(client, monkeypatch):
    """Verify datasets past idle TTL are expired and return 404 with session expired message."""
    import backend.main as backend_main
    backend_main.DATASET_STORE.clear()

    current_time = [1000.0]
    monkeypatch.setattr(backend_main.DATASET_STORE, "_time_provider", lambda: current_time[0])

    res = client.post(
        "/api/datasets/upload",
        files={"file": ("expiring.csv", io.BytesIO(b"col\n100"), "text/csv")}
    )
    assert res.status_code == 200
    ds_id = res.json()["dataset_id"]

    # Immediately accessible
    assert client.get(f"/api/datasets/{ds_id}").status_code == 200

    # Advance time beyond 1-hour TTL (3600 seconds)
    current_time[0] += 3605.0

    # Request after TTL expiration
    expired_res = client.get(f"/api/datasets/{ds_id}")
    assert expired_res.status_code == 404
    assert "session expired" in expired_res.json()["detail"].lower()


def test_active_analysis_prevents_eviction():
    """Verify an active dataset cannot be evicted even if it is the oldest item in store."""
    from backend.store import BoundedDatasetStore
    store = BoundedDatasetStore(max_datasets=2)
    df = pd.DataFrame({"col": [1, 2, 3]})

    store.put("ds1", df, "ds1.csv")
    store.put("ds2", df, "ds2.csv")

    # Mark ds1 as active
    with store.acquire_active("ds1"):
        # Put ds3: capacity limit reached (max_datasets=2).
        # Since ds1 is active, ds2 (inactive) must be evicted instead of ds1!
        store.put("ds3", df, "ds3.csv")
        assert "ds1" in store
        assert "ds2" not in store
        assert "ds3" in store


def test_capacity_exhaustion_when_all_candidates_active(client, monkeypatch):
    """Verify upload returns HTTP 503 when all stored datasets are active and cannot be evicted."""
    import backend.main as backend_main
    monkeypatch.setattr(backend_main.DATASET_STORE, "max_datasets", 2)
    backend_main.DATASET_STORE.clear()

    res1 = client.post(
        "/api/datasets/upload",
        files={"file": ("data_1.csv", io.BytesIO(b"col\n1"), "text/csv")}
    )
    res2 = client.post(
        "/api/datasets/upload",
        files={"file": ("data_2.csv", io.BytesIO(b"col\n2"), "text/csv")}
    )
    id1 = res1.json()["dataset_id"]
    id2 = res2.json()["dataset_id"]

    # Hold active locks on both retained datasets
    with backend_main.DATASET_STORE.acquire_active(id1):
        with backend_main.DATASET_STORE.acquire_active(id2):
            res3 = client.post(
                "/api/datasets/upload",
                files={"file": ("data_3.csv", io.BytesIO(b"col\n3"), "text/csv")}
            )
            assert res3.status_code == 503
            assert "capacity temporarily exhausted" in res3.json()["detail"].lower()


def test_oversized_dataset_rejected_without_modifying_store():
    """Verify dataset larger than max_memory_bytes is rejected without altering existing store contents."""
    from backend.store import BoundedDatasetStore, StorageCapacityExhaustedError
    df_small = pd.DataFrame({"col": [1, 2]})
    small_mem = BoundedDatasetStore.estimate_dataframe_memory(df_small)

    store = BoundedDatasetStore(max_datasets=5, max_memory_bytes=small_mem * 3)
    store.put("valid1", df_small, "small.csv")
    initial_mem = store.total_memory_bytes
    assert len(store) == 1

    df_oversized = pd.DataFrame({"col": range(10000)})
    assert BoundedDatasetStore.estimate_dataframe_memory(df_oversized) > store.max_memory_bytes

    try:
        store.put("oversized", df_oversized, "large.csv")
        assert False, "Should have raised StorageCapacityExhaustedError"
    except StorageCapacityExhaustedError:
        pass

    # Ensure store was untouched
    assert len(store) == 1
    assert "valid1" in store
    assert store.total_memory_bytes == initial_mem


def test_failed_insertion_does_not_evict_existing_valid_datasets():
    """Verify that if an insertion cannot succeed, zero existing datasets are evicted speculatively."""
    from backend.store import BoundedDatasetStore, StorageCapacityExhaustedError
    df = pd.DataFrame({"col": range(50)})
    mem = BoundedDatasetStore.estimate_dataframe_memory(df)

    # Store with max 2 datasets, memory for 2 datasets
    store = BoundedDatasetStore(max_datasets=2, max_memory_bytes=mem * 2 + 50)
    store.put("active_ds", df, "active.csv")
    store.put("inactive_ds", df, "inactive.csv")

    with store.acquire_active("active_ds"):
        # Attempt to insert a dataset that requires more memory than can be obtained by evicting inactive_ds
        df_large = pd.DataFrame({"col": range(150)})
        large_mem = BoundedDatasetStore.estimate_dataframe_memory(df_large)
        assert large_mem + mem > store.max_memory_bytes

        try:
            store.put("new_ds", df_large, "new.csv")
            assert False, "Should have raised StorageCapacityExhaustedError"
        except StorageCapacityExhaustedError:
            pass

        # Inactive dataset must NOT have been speculatively evicted!
        assert "active_ds" in store
        assert "inactive_ds" in store
        assert len(store) == 2


def test_memory_accounting_after_delete_and_expiration():
    """Verify total_memory_bytes accurately tracks allocations through insertion, deletion, and TTL expiration."""
    from backend.store import BoundedDatasetStore
    clock = [100.0]
    store = BoundedDatasetStore(max_datasets=5, ttl_seconds=50.0, time_provider=lambda: clock[0])

    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"b": [4, 5, 6, 7, 8]})
    m1 = store.estimate_dataframe_memory(df1)
    m2 = store.estimate_dataframe_memory(df2)

    store.put("ds1", df1, "f1.csv")
    assert store.total_memory_bytes == m1

    store.put("ds2", df2, "f2.csv")
    assert store.total_memory_bytes == (m1 + m2)

    # Delete ds1
    store.delete("ds1")
    assert store.total_memory_bytes == m2

    # Expire ds2 by advancing clock
    clock[0] += 60.0
    # Lazy expiration on count or access
    assert len(store) == 0
    assert store.total_memory_bytes == 0


def test_concurrent_store_access():
    """Verify thread-safe concurrent reads and writes do not corrupt internal store state."""
    import threading
    from backend.store import BoundedDatasetStore

    store = BoundedDatasetStore(max_datasets=10, max_memory_bytes=10 * 1024 * 1024)
    errors = []

    def worker(worker_id: int):
        try:
            df = pd.DataFrame({"col": range(50)})
            ds_id = f"ds_{worker_id}"
            store.put(ds_id, df, f"worker_{worker_id}.csv")
            with store.acquire_active(ds_id) as active_df:
                assert len(active_df) == 50
            item = store.get(ds_id)
            assert item.filename == f"worker_{worker_id}.csv"
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert len(store) <= 10


def test_dictionary_compatibility_protocol():
    """Verify BoundedDatasetStore satisfies dict interface: len, in, get, pop, copy, iteration, KeyError."""
    from backend.store import BoundedDatasetStore
    store = BoundedDatasetStore(max_datasets=5)
    df = pd.DataFrame({"x": [10, 20]})

    store["d1"] = {"df": df, "filename": "data.csv"}
    assert len(store) == 1
    assert "d1" in store
    assert "d_missing" not in store
    assert bool(store) is True

    # Subscript access
    item = store["d1"]
    assert item["filename"] == "data.csv"
    assert "df" in item
    assert "filename" in item.keys()

    # get with default
    assert store.get("missing", "fallback") == "fallback"

    # copy snapshot
    snap = store.copy()
    assert isinstance(snap, dict)
    assert "d1" in snap

    # pop
    popped = store.pop("d1")
    assert popped.filename == "data.csv"
    assert len(store) == 0
    assert bool(store) is False

    # pop missing with default
    assert store.pop("missing", None) is None

    # Missing subscript raises KeyError
    try:
        _ = store["missing"]
        assert False, "Should raise KeyError"
    except KeyError:
        pass


def test_clear_refuses_when_datasets_active():
    """Verify clear() refuses to clear store when active queries are running, preventing query invalidation."""
    from backend.store import BoundedDatasetStore, DatasetInUseError
    store = BoundedDatasetStore(max_datasets=5)
    df = pd.DataFrame({"col": [1, 2, 3]})
    store.put("active_ds", df, "active.csv")

    with store.acquire_active("active_ds"):
        try:
            store.clear()
            assert False, "Should raise DatasetInUseError"
        except DatasetInUseError:
            pass

        # Dataset must remain intact in store
        assert "active_ds" in store
        assert len(store) == 1

    # Once active analysis completes, clear() succeeds
    store.clear()
    assert len(store) == 0
    assert store.total_memory_bytes == 0


def test_replacement_failure_preserves_previous_dataset():
    """Verify attempting to replace an existing dataset with an unfulfillable one preserves the previous dataset."""
    from backend.store import BoundedDatasetStore, StorageCapacityExhaustedError
    df_small = pd.DataFrame({"col": range(10)})
    small_mem = BoundedDatasetStore.estimate_dataframe_memory(df_small)

    store = BoundedDatasetStore(max_datasets=2, max_memory_bytes=small_mem * 2 + 50)
    store.put("ds_target", df_small, "original.csv")
    store.put("ds_active", df_small, "active.csv")

    with store.acquire_active("ds_active"):
        # Attempt to replace ds_target with a payload that cannot fit alongside ds_active
        df_too_big = pd.DataFrame({"col": range(100)})
        assert BoundedDatasetStore.estimate_dataframe_memory(df_too_big) + small_mem > store.max_memory_bytes

        try:
            store.put("ds_target", df_too_big, "replacement.csv")
            assert False, "Should raise StorageCapacityExhaustedError"
        except StorageCapacityExhaustedError:
            pass

        # Previous dataset must remain untouched
        assert "ds_target" in store
        assert store["ds_target"].filename == "original.csv"
        assert len(store) == 2


def test_active_operation_refreshes_ttl_on_release():
    """
    Verify a long-running active operation refreshes last_accessed on release
    so the dataset does not immediately expire after completion.

    Timeline (TTL = 100 s):
      t=1000  put() → last_accessed=1000
      t=1000  acquire_active() → active_queries=1, last_accessed=1000
      t=1150  clock advanced inside context (150 s > TTL, but dataset is active)
      t=1150  finally: active_queries→0, last_accessed refreshed to 1150
      t=1170  clock +20 → store.get() → last_accessed refreshed to 1170 (TTL restarted)
      t=1275  clock +105 → 1275-1170=105 > 100 → DatasetSessionExpiredError raised
    """
    from backend.store import BoundedDatasetStore, DatasetSessionExpiredError
    clock = [1000.0]
    # 100 seconds idle TTL
    store = BoundedDatasetStore(max_datasets=5, ttl_seconds=100.0, time_provider=lambda: clock[0])
    df = pd.DataFrame({"col": [1, 2, 3]})
    store.put("ds_long", df, "long.csv")

    # Begin active query at t=1000.0
    with store.acquire_active("ds_long"):
        # Advance clock by 150.0 seconds (to 1150.0), which exceeds TTL but dataset is active
        clock[0] += 150.0

    # Query finished at t=1150.0; finally block refreshed last_accessed to 1150.0.
    # Advance clock +20 to 1170.0 — well within the new 100 s TTL window from 1150.0.
    clock[0] += 20.0
    assert "ds_long" in store
    # store.get() also refreshes last_accessed to t=1170.0 and returns the item.
    item = store.get("ds_long")
    assert item.filename == "long.csv"
    assert item.active_queries == 0

    # last_accessed is now 1170.0 (refreshed by store.get() above).
    # Advance clock by 105 s to t=1275.0 → 1275-1170=105 > TTL(100) → expires.
    clock[0] += 105.0
    try:
        store.get("ds_long")
        assert False, "Should raise DatasetSessionExpiredError after refreshed TTL expires"
    except DatasetSessionExpiredError:
        pass



def test_concurrent_sample_registration_deduplicated_and_memory_accounted():
    """Verify concurrent sample registration with deterministic barrier deduplicates atomically with correct memory accounting."""
    import threading
    from backend.store import BoundedDatasetStore
    store = BoundedDatasetStore(max_datasets=5, max_memory_bytes=50 * 1024 * 1024)
    df = pd.DataFrame({"a": range(100), "b": range(100)})
    expected_mem = store.estimate_dataframe_memory(df)

    num_threads = 8
    barrier = threading.Barrier(num_threads)
    results = []
    errors = []

    def worker():
        try:
            barrier.wait()
            stored = store.put_or_reuse_sample(df, "sample.csv")
            results.append(stored.dataset_id)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert len(results) == num_threads
    # All threads received the identical dataset ID
    assert len(set(results)) == 1
    # Exactly one sample retained in store
    assert len(store) == 1
    # Memory accounted for exactly one dataset, not 8x
    assert store.total_memory_bytes == expected_mem


def test_recharts_pie_chart_serialization_and_null_handling():
    """Verify build_recharts_data serializes pie charts with proper formatting and null handling."""
    from backend.chart_utils import build_recharts_data

    # 1. Proportion DataFrame
    df_prop = pd.DataFrame({"diagnosis": ["Malignant", "Benign"], "proportion": [0.627, 0.373]})
    viz = build_recharts_data(df_prop)
    assert viz is not None
    assert viz["chart_type"] == "Pie Chart"
    assert viz["x_key"] == "name"
    assert viz["y_keys"] == ["value"]
    assert len(viz["data"]) == 2
    assert viz["data"][0] == {"name": "Malignant", "value": 0.627}
    assert viz["data"][1] == {"name": "Benign", "value": 0.373}

    # 2. Percentage Series with question intent
    s_pct = pd.Series([45.0, 55.0], index=["Product A", "Product B"], name="Share")
    viz_series = build_recharts_data(s_pct, question="create a pie chart")
    assert viz_series is not None
    assert viz_series["chart_type"] == "Pie Chart"
    assert len(viz_series["data"]) == 2
    assert viz_series["data"][0]["name"] == "Product A"
    assert viz_series["data"][0]["value"] == 45.0

    # 3. Non-finite or missing values cleaned safely
    df_dirty = pd.DataFrame({"Category": ["Alpha", None], "proportion": [0.7, np.nan]})
    viz_dirty = build_recharts_data(df_dirty, question="pie chart")
    assert viz_dirty is not None
    assert viz_dirty["data"][0] == {"name": "Alpha", "value": 0.7}
    assert viz_dirty["data"][1] == {"name": "Unknown", "value": 0.0}

    # 4. Ordinary comparison defaults to Bar Chart without question
    df_bar = pd.DataFrame({"Dept": ["HR", "Eng"], "Headcount": [10, 50]})
    viz_bar = build_recharts_data(df_bar)
    assert viz_bar["chart_type"] == "Bar Chart"


def test_credential_sanitization_comprehensive():
    """Verify that _sanitize_error_message redacts legacy AIza keys, modern AQ. keys, and bearer tokens."""
    from backend.main import _sanitize_error_message

    # 1. Legacy AIza Google API key
    msg1 = "GoogleGenAIError: 400 Invalid key AIzaSyD3x9FakeKey123456789012345678901 in request URL"
    clean1 = _sanitize_error_message(msg1)
    assert "AIzaSyD3x9FakeKey123456789012345678901" not in clean1
    assert "[REDACTED_API_KEY]" in clean1

    # 2. Modern AQ. Google API key
    msg2 = "ClientError: 403 request failed with token AQ.Ab8RN6I-FakeKey1234567890abcdef at https://api.endpoint"
    clean2 = _sanitize_error_message(msg2)
    assert "AQ.Ab8RN6I-FakeKey1234567890abcdef" not in clean2
    assert "[REDACTED_API_KEY]" in clean2

    # 3. Authorization Bearer token
    msg3 = "HTTPError: 401 Authorization: Bearer ya29.a0AfH6SMFakeBearerToken9876543210"
    clean3 = _sanitize_error_message(msg3)
    assert "ya29.a0AfH6SMFakeBearerToken9876543210" not in clean3
    assert "Bearer [REDACTED_TOKEN]" in clean3

    # 4. key= query parameter
    msg4 = "Request failed: https://generativelanguage.googleapis.com/v1beta?key=secret_arbitrary_key_value_12345"
    clean4 = _sanitize_error_message(msg4)
    assert "secret_arbitrary_key_value_12345" not in clean4
    assert "key=[REDACTED]" in clean4

    # 5. Normal error message is preserved
    normal = "Column 'sales' does not exist in dataset."
    assert _sanitize_error_message(normal) == normal


def test_recharts_infinity_and_truncation_handling():
    """Verify build_recharts_data cleanly handles positive/negative infinity and marks truncation in titles."""
    from backend.chart_utils import build_recharts_data

    # 1. Positive and negative infinity in DataFrame values
    df_inf = pd.DataFrame({
        "Category": ["PosInf", "NegInf", "Valid"],
        "Value": [float("inf"), float("-inf"), 42.5]
    })
    viz_inf = build_recharts_data(df_inf)
    assert viz_inf is not None
    assert viz_inf["data"][0]["value"] == 0.0
    assert viz_inf["data"][1]["value"] == 0.0
    assert viz_inf["data"][2]["value"] == 42.5

    # 2. Category truncation (over 30 items) appends '(Top 30)'
    df_many = pd.DataFrame({
        "Category": [f"Cat_{i}" for i in range(45)],
        "Value": [i * 10 for i in range(45)]
    })
    viz_many = build_recharts_data(df_many)
    assert viz_many is not None
    assert len(viz_many["data"]) == 30
    assert "(Top 30)" in viz_many["title"]
