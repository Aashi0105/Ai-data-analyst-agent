"""
tests/test_dataset_understanding.py
-----------------------------------
Automated tests for Phase 1 — Dataset Understanding:
- Pure deterministic profiler correctness and edge cases.
- Duplicate row semantics vs repeated column values.
- Safe infinity and all-null column handling.
- Date inference reliability.
- Grounded explanation and question generation.
- Mocked Gemini success, timeout, quota exhaustion, and malformed JSON fallback.
- Suggested questions submission through existing analysis workflow.
- API endpoints integration (/api/datasets/upload, /api/datasets/sample, /api/datasets/{id}/understanding).
"""

import io
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app, DATASET_STORE
from profiler import profile_dataset, DatasetProfile
from understanding import (
    generate_dataset_understanding,
    generate_deterministic_fallback,
    DatasetUnderstanding,
    _sanitize_and_validate_understanding,
)


@pytest.fixture
def client():
    return TestClient(app)


# ============================================================================
# 1. Deterministic Profiler Tests
# ============================================================================

class TestDeterministicProfiler:
    def test_known_dataframe_profile_metrics(self):
        """Validates exact numerical statistics and shape metrics on known data."""
        df = pd.DataFrame({
            "revenue": [100.0, 200.0, 300.0, 400.0, 500.0],
            "category": ["A", "B", "A", "B", "C"],
            "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
        })
        profile = profile_dataset(df)

        assert profile.num_rows == 5
        assert profile.num_cols == 3
        assert profile.total_missing == 0
        assert profile.duplicate_rows == 0

        # Numeric stats
        rev_stats = profile.column_profiles["revenue"].numeric_stats
        assert rev_stats is not None
        assert rev_stats.min == 100.0
        assert rev_stats.max == 500.0
        assert rev_stats.mean == 300.0
        assert rev_stats.median == 300.0
        assert rev_stats.q25 == 200.0
        assert rev_stats.q75 == 400.0

        # Categorical stats
        cat_stats = profile.column_profiles["category"].categorical_stats
        assert cat_stats is not None
        assert cat_stats.unique_count == 3
        top_names = [c.value for c in cat_stats.top_categories]
        assert "A" in top_names
        assert "B" in top_names

        # Date stats
        date_stats = profile.column_profiles["date"].date_stats
        assert date_stats is not None
        assert profile.column_profiles["date"].semantic_type == "datetime"
        assert "2023-01-01" in str(date_stats.min_date)
        assert "2023-01-05" in str(date_stats.max_date)

    def test_duplicate_row_semantics_vs_column_repeats(self):
        """
        Confirms duplicate row count strictly measures complete row identity,
        distinguishing it from repeated values in individual columns.
        """
        # Column 'val' has repeated values, but rows are unique
        df_unique_rows = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "val": [10, 10, 10, 10],  # Column has repeated values
        })
        profile_unique = profile_dataset(df_unique_rows)
        assert profile_unique.duplicate_rows == 0
        assert profile_unique.duplicate_row_pct == 0.0

        # DataFrame with 2 completely duplicate rows
        df_with_dupes = pd.DataFrame({
            "id": [1, 2, 1, 3, 1],
            "val": [10, 20, 10, 30, 10],
        })
        # Rows 0, 2, and 4 are identical: row 2 and row 4 are duplicates of row 0
        profile_dupes = profile_dataset(df_with_dupes)
        assert profile_dupes.duplicate_rows == 2
        assert profile_dupes.duplicate_row_pct == 40.0

    def test_safe_infinities_and_nans(self):
        """Ensures inf, -inf, and NaNs in numeric columns do not crash or corrupt stats."""
        df = pd.DataFrame({
            "metric": [10.0, np.inf, -np.inf, np.nan, 20.0, 30.0],
            "label": ["a", "b", "c", None, "e", "f"],
        })
        profile = profile_dataset(df)

        assert profile.num_rows == 6
        metric_profile = profile.column_profiles["metric"]
        assert metric_profile.semantic_type == "numeric"
        stats = metric_profile.numeric_stats
        assert stats is not None
        # 3 finite values: 10, 20, 30
        assert stats.count == 3
        assert stats.min == 10.0
        assert stats.max == 30.0
        assert stats.mean == 20.0

    def test_empty_dataframe_profiling(self):
        """Ensures 0-row or 0-column DataFrames profile cleanly with appropriate warning."""
        df_empty = pd.DataFrame()
        profile = profile_dataset(df_empty)
        assert profile.num_rows == 0
        assert profile.num_cols == 0
        assert len(profile.warnings) == 1
        assert profile.warnings[0].code == "EMPTY_DATASET"

    def test_all_null_column_profiling(self):
        """Ensures all-null columns are identified with 'empty' semantic type and warning."""
        df = pd.DataFrame({
            "valid_col": [1, 2, 3],
            "empty_col": [None, np.nan, None],
        })
        profile = profile_dataset(df)
        empty_prof = profile.column_profiles["empty_col"]
        assert empty_prof.semantic_type == "empty"
        assert empty_prof.missing_pct == 100.0

        warning_codes = [w.code for w in profile.warnings]
        assert "ALL_NULL_COLUMN" in warning_codes

    def test_constant_column_detection(self):
        """Ensures constant columns generate an informational warning."""
        df = pd.DataFrame({
            "constant_num": [42, 42, 42, 42],
            "constant_str": ["same", "same", "same", "same"],
        })
        profile = profile_dataset(df)
        warning_codes = [w.code for w in profile.warnings]
        assert "CONSTANT_COLUMN" in warning_codes

    def test_high_cardinality_detection(self):
        """Ensures high cardinality non-numeric attributes are flagged as text."""
        num_rows = 50
        df = pd.DataFrame({
            "uuid_token": [f"user_{i}" for i in range(num_rows)],
            "score": list(range(num_rows)),
        })
        profile = profile_dataset(df)
        assert profile.column_profiles["uuid_token"].semantic_type == "text"
        warning_codes = [w.code for w in profile.warnings]
        assert "HIGH_CARDINALITY" in warning_codes

    def test_reliable_date_detection(self):
        """Ensures true date strings are recognized while arbitrary text/numbers are not."""
        df = pd.DataFrame({
            "order_date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
            "not_a_date": ["apple", "banana", "cherry", "date fruit"],
            "numeric_id": [20230101, 20230102, 20230103, 20230104],
        })
        profile = profile_dataset(df)
        assert profile.column_profiles["order_date"].semantic_type == "datetime"
        assert profile.column_profiles["not_a_date"].semantic_type == "categorical"
        assert profile.column_profiles["numeric_id"].semantic_type == "numeric"


# ============================================================================
# 2. Grounded Explanation & Question Generator Tests
# ============================================================================

class TestUnderstandingGenerator:
    def test_deterministic_fallback_properties(self):
        """Validates that fallback produces 5 valid questions and grounded overview."""
        df = pd.DataFrame({
            "price": [10.0, 20.0, 30.0],
            "item": ["A", "B", "C"],
        })
        profile = profile_dataset(df)
        fallback = generate_deterministic_fallback(profile)

        assert fallback.status == "fallback"
        assert len(fallback.suggested_questions) == 5
        assert all(isinstance(q, str) and len(q) > 0 for q in fallback.suggested_questions)
        assert "price" in fallback.overview or "records" in fallback.overview
        assert len(fallback.column_insights) > 0

    def test_fallback_time_question_only_when_date_exists(self):
        """Ensures time-based questions are only suggested when a date column is present."""
        # No date column
        df_no_date = pd.DataFrame({"sales": [10, 20], "dept": ["X", "Y"]})
        fb_no_date = generate_deterministic_fallback(profile_dataset(df_no_date))
        assert not any("trend over" in q for q in fb_no_date.suggested_questions)

        # With date column
        df_date = pd.DataFrame({
            "sales": [10, 20],
            "dept": ["X", "Y"],
            "timestamp": ["2023-01-01", "2023-01-02"],
        })
        fb_date = generate_deterministic_fallback(profile_dataset(df_date))
        assert any("trend over timestamp" in q for q in fb_date.suggested_questions)

    def test_llm_success_mocked(self):
        """Ensures valid Gemini JSON response is parsed into a 'generated' understanding."""
        df = pd.DataFrame({"sales": [100, 200], "region": ["North", "South"]})
        profile = profile_dataset(df)

        mock_json_response = """
        {
          "overview": "This dataset appears to record regional sales transactions.",
          "column_insights": [
            {"column": "sales", "inferred_meaning": "Revenue generated per sale"},
            {"column": "region", "inferred_meaning": "Geographical sales territory"}
          ],
          "analytical_potential": "Supports regional revenue comparison and sales volume aggregations.",
          "data_quality_notes": "Clean data with zero missing values.",
          "suggested_questions": [
            "What is the total sales by region?",
            "What is the average sales amount?",
            "Which region has the highest recorded sale?",
            "How does sales compare between North and South?",
            "What is the overall distribution of sales?"
          ]
        }
        """

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_resp = MagicMock()
            mock_resp.text = mock_json_response
            mock_client.models.generate_content.return_value = mock_resp

            with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
                understanding = generate_dataset_understanding(df, profile)

            assert understanding.status == "generated"
            assert len(understanding.suggested_questions) == 5
            assert "regional sales transactions" in understanding.overview
            assert understanding.suggested_questions[0] == "What is the total sales by region?"

    def test_llm_malformed_json_fallback(self):
        """Ensures invalid JSON or garbage output from Gemini triggers deterministic fallback."""
        df = pd.DataFrame({"metric": [1, 2, 3]})
        profile = profile_dataset(df)

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_resp = MagicMock()
            mock_resp.text = "This is not valid JSON at all: 42!!!"
            mock_client.models.generate_content.return_value = mock_resp

            with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
                understanding = generate_dataset_understanding(df, profile)

            assert understanding.status == "fallback"
            assert len(understanding.suggested_questions) == 5

    def test_llm_timeout_or_network_error_fallback(self):
        """Ensures LLM timeout or connection failure triggers deterministic fallback."""
        df = pd.DataFrame({"colA": [10, 20]})
        profile = profile_dataset(df)

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.models.generate_content.side_effect = TimeoutError("Gemini call timed out")

            with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
                understanding = generate_dataset_understanding(df, profile)

            assert understanding.status == "fallback"
            assert len(understanding.suggested_questions) == 5

    def test_llm_quota_exhaustion_fallback(self):
        """Ensures 429 quota exhaustion during understanding triggers deterministic fallback."""
        df = pd.DataFrame({"colA": [10, 20]})
        profile = profile_dataset(df)

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.models.generate_content.side_effect = RuntimeError(
                "429 RESOURCE_EXHAUSTED: GenerateRequestsPerDay exceeded"
            )

            with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
                understanding = generate_dataset_understanding(df, profile)

            assert understanding.status == "fallback"
            assert len(understanding.suggested_questions) == 5

    def test_insufficient_questions_padded_to_five(self):
        """Ensures if LLM provides 4 questions, it is padded to strictly 5 distinct questions."""
        df = pd.DataFrame({"sales": [1, 2], "rep": ["A", "B"]})
        profile = profile_dataset(df)

        raw_json = """
        {
          "overview": "Short test overview.",
          "column_insights": [],
          "analytical_potential": "Analysis potential.",
          "data_quality_notes": "Notes.",
          "suggested_questions": [
            "Question 1?",
            "Question 2?",
            "Question 3?",
            "Question 4?"
          ]
        }
        """
        validated = _sanitize_and_validate_understanding(raw_json, profile)
        assert validated is not None
        assert len(validated.suggested_questions) == 5

    def test_prompt_masks_pii_in_categorical_top_categories(self):
        """Ensures raw PII in categorical top categories is redacted before prompt reaches Gemini."""
        from understanding import _build_understanding_prompt

        df = pd.DataFrame({
            "customer_email": ["secret_john@corp.com", "secret_jane@corp.com", "secret_john@corp.com"],
            "amount": [100, 200, 300],
        })
        profile = profile_dataset(df)
        system_instruction, user_prompt = _build_understanding_prompt(df, profile)

        # Raw emails must NOT be transmitted in user prompt (neither in profile_summary nor sample rows)
        assert "secret_john@corp.com" not in user_prompt
        assert "secret_jane@corp.com" not in user_prompt
        assert "[EMAIL]" in user_prompt

    def test_hard_timeout_enforcement_in_generate_understanding(self):
        """Ensures long-running Gemini call is aborted within timeout_seconds and triggers fallback."""
        import time

        df = pd.DataFrame({"colA": [10, 20]})
        profile = profile_dataset(df)

        def slow_generate(*args, **kwargs):
            time.sleep(2.0)
            return MagicMock(text="{}")

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.models.generate_content.side_effect = slow_generate

            with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
                start = time.time()
                understanding = generate_dataset_understanding(df, profile, timeout_seconds=0.2)
                elapsed = time.time() - start

            # Must have returned fallback in under 1 second
            assert elapsed < 1.0
            assert understanding.status == "fallback"
            assert len(understanding.suggested_questions) == 5

    def test_sanitize_strips_backticks_from_suggested_questions(self):
        """Ensures markdown code backticks around column names are cleaned from suggested questions."""
        df = pd.DataFrame({"sales": [1, 2], "rep": ["A", "B"]})
        profile = profile_dataset(df)

        raw_json = """
        {
          "overview": "Overview.",
          "column_insights": [],
          "analytical_potential": "Potential.",
          "data_quality_notes": "Notes.",
          "suggested_questions": [
            "What is the total `sales` by `rep`?",
            "What is the average `sales`?",
            "Who is the top `rep`?",
            "What is the median `sales`?",
            "How many `sales` occurred?"
          ]
        }
        """
        validated = _sanitize_and_validate_understanding(raw_json, profile)
        assert validated is not None
        assert validated.suggested_questions[0] == "What is the total sales by rep?"
        assert not any("`" in q for q in validated.suggested_questions)


# ============================================================================
# 3. API Endpoints Integration Tests
# ============================================================================

class TestApiDatasetUnderstanding:
    def test_sample_endpoint_includes_profile_and_understanding(self, client):
        """Calls POST /api/datasets/sample and verifies profile & understanding are present."""
        resp = client.post("/api/datasets/sample")
        assert resp.status_code == 200
        data = resp.json()

        assert "profile" in data and data["profile"] is not None
        assert "understanding" in data and data["understanding"] is not None
        assert len(data["understanding"]["suggested_questions"]) == 5
        assert data["profile"]["num_rows"] == 100
        assert data["profile"]["num_cols"] == 13

    def test_upload_endpoint_includes_profile_and_understanding(self, client):
        """Uploads a CSV and verifies enriched response schema."""
        csv_content = b"department,budget,headcount\nSales,500000,12\nEngineering,1200000,25\nMarketing,300000,8\n"
        files = {"file": ("departments.csv", io.BytesIO(csv_content), "text/csv")}

        resp = client.post("/api/datasets/upload", files=files)
        assert resp.status_code == 200
        data = resp.json()

        assert data["filename"] == "departments.csv"
        assert data["num_rows"] == 3
        assert data["num_cols"] == 3
        assert data["profile"]["columns"] == ["department", "budget", "headcount"]
        assert len(data["understanding"]["suggested_questions"]) == 5

    def test_get_dataset_understanding_endpoint(self, client):
        """Retrieves understanding via GET /api/datasets/{dataset_id}/understanding."""
        # First ensure sample is loaded
        sample_resp = client.post("/api/datasets/sample")
        dataset_id = sample_resp.json()["dataset_id"]

        resp = client.get(f"/api/datasets/{dataset_id}/understanding")
        assert resp.status_code == 200
        data = resp.json()

        assert "overview" in data
        assert "suggested_questions" in data
        assert len(data["suggested_questions"]) == 5

    def test_upload_succeeds_even_when_gemini_fails(self, client):
        """
        Critical requirement: A failure in Gemini during upload must NEVER fail the upload.
        Upload must return 200 with fallback understanding.
        """
        csv_content = b"product,price\nWidget,19.99\nGadget,29.99\n"
        files = {"file": ("products.csv", io.BytesIO(csv_content), "text/csv")}

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.models.generate_content.side_effect = Exception("API connection dropped")

            resp = client.post("/api/datasets/upload", files=files)
            assert resp.status_code == 200
            data = resp.json()

            assert data["understanding"]["status"] == "fallback"
            assert len(data["understanding"]["suggested_questions"]) == 5

    def test_suggested_question_execution_in_existing_analysis_pipeline(self, client):
        """
        Verifies that suggested questions use the existing analysis pipeline
        and can be dispatched to POST /api/analyze with valid AST and sandbox execution.
        """
        sample_resp = client.post("/api/datasets/sample")
        sample_data = sample_resp.json()
        dataset_id = sample_data["dataset_id"]
        suggested_questions = sample_data["understanding"]["suggested_questions"]

        assert len(suggested_questions) == 5

        # Mock the DataAnalystAgent.generate_code so the test is 100% deterministic and quota-free
        mock_pandas_code = "result = df.groupby('region')['revenue'].sum()"

        with patch("backend.main.DataAnalystAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.generate_code.return_value = mock_pandas_code
            mock_agent_cls.return_value = mock_agent

            # Submit the first suggested question
            analyze_resp = client.post("/api/analyze", json={
                "dataset_id": dataset_id,
                "question": suggested_questions[0]
            })

            assert analyze_resp.status_code == 200
            analyze_data = analyze_resp.json()

            assert analyze_data["validation_status"] == "passed"
            assert analyze_data["execution_status"] == "passed"
            assert analyze_data["table_data"] is not None
            assert len(analyze_data["table_data"]["rows"]) > 0

    def test_upload_rollback_on_profiling_exception(self, client):
        """
        Confirms that if profile_dataset raises an unhandled exception during upload,
        the dataset is rolled back from DATASET_STORE and no orphaned dataset remains.
        """
        csv_content = b"colA,colB\n1,2\n3,4\n"
        files = {"file": ("crash_rollback.csv", io.BytesIO(csv_content), "text/csv")}

        with patch("backend.main.profile_dataset", side_effect=RuntimeError("Simulated profiling failure")):
            resp = client.post("/api/datasets/upload", files=files)
            assert resp.status_code == 500
            assert "Failed to profile dataset" in resp.json()["detail"]

        # Confirm that crash_rollback.csv is not retained in DATASET_STORE
        assert not any(item.filename == "crash_rollback.csv" for item in DATASET_STORE.values())
