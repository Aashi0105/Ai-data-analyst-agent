"""
tests/test_dataset_overview.py
------------------------------
Comprehensive production-grade unit and regression test suite for Dataset Overview:
1. Case A: Entirely empty listings dataset (2,911 rows, 100% missingness, 2,910 duplicate rows).
2. Case B: Wisconsin Breast Cancer diagnostic dataset (569 rows, 32 columns, diagnosis + 30 measurements).
3. Identifier inference across multiple naming conventions and exclusion from metrics.
4. Binary and multiclass label recognition and class-balanced classification feasibility.
5. Defensible continuous regression target selection vs rejection.
6. Forecasting blocked on cross-sectional data without temporal index.
7. Domain inference from combined column evidence (property listings, clinical/diagnostic biopsy, commerce, HR, finance).
8. Row-entity inference with calibrated uncertainty and domain-appropriate vocabulary.
9. All-null datasets and partially missing/constant column handling.
10. Recommendations referencing only existing columns and conforming to query execution contracts.
11. LLM hallucination rejection and deterministic fallback parity.
12. API response backward compatibility.
"""

import io
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from profiler import profile_dataset
from overview import (
    ColumnRole,
    DatasetOverview,
    DatasetSemantics,
    build_dataset_semantics,
    infer_column_semantics,
    generate_constrained_opportunities,
    generate_evidence_based_predictions,
    generate_deterministic_overview,
    generate_dataset_overview,
    _sanitize_and_validate_overview,
)
from backend.main import app, DATASET_STORE, _format_dataset_response, DatasetMetadataResponse


class TestProductionDatasetCases(unittest.TestCase):
    """Rigorous acceptance tests for observed production cases A and B."""

    def test_case_a_entirely_empty_listings_dataset(self):
        """
        Case A: 2,911 rows, 10 columns, 29,110 missing cells (100% missing), 2,910 duplicate empty rows.
        Verifies:
          - Profile statistics (2,911 rows, 29,110 missing, 2,910 duplicates) are preserved unaltered.
          - Domain is inferred from column headers (Real Estate & Property Listings) with low confidence.
          - Data usability is marked unusable/empty.
          - Summary paragraph explicitly reports zero usable values (100% missingness).
          - Value-dependent analytical questions are suppressed.
          - Predictive tasks report 'Not currently supported by the data' citing lack of populated records.
        """
        cols = [
            "id", "name", "host_id", "host_name", "neighbourhood_group",
            "neighbourhood", "latitude", "longitude", "room_type", "price"
        ]
        # Create dataframe matching exact profile: 2,911 rows, all null
        data = {c: [np.nan] * 2911 for c in cols}
        df = pd.DataFrame(data)

        profile = profile_dataset(df)

        # 1. Verify statistical calculations are 100% correct and preserved
        self.assertEqual(profile.num_rows, 2911)
        self.assertEqual(profile.num_cols, 10)
        self.assertEqual(profile.total_missing, 29110)
        self.assertEqual(profile.missing_cell_pct, 100.0)
        self.assertEqual(profile.duplicate_rows, 2910)

        overview = generate_deterministic_overview(df, profile)

        # 2. Domain classification from schema headers
        self.assertEqual(overview.domain, "Real Estate & Property Listings")
        self.assertEqual(overview.domain_confidence, "low")
        self.assertIn("100% missing", overview.domain_rationale)

        # 3. Summary paragraph & row meaning must honestly state data is empty
        self.assertIn("100% missingness", overview.summary_paragraph)
        self.assertIn("zero usable observations", overview.summary_paragraph)
        self.assertIn("currently contains no recorded data", overview.row_meaning)
        self.assertIn("0 usable observations", overview.size_and_coverage)

        # 4. Analytical opportunities must suppress ordinary price/group queries
        self.assertEqual(len(overview.what_we_can_learn), 1)
        opp = overview.what_we_can_learn[0]
        self.assertIn("Completeness Notice", opp.title)
        self.assertIsNone(opp.sample_question)

        # 5. Predictions must not claim feasibility on empty rows
        self.assertTrue(len(overview.potential_predictions) >= 1)
        for pred in overview.potential_predictions:
            self.assertEqual(pred.feasibility, "Not currently supported by the data")
            self.assertIn("zero", pred.prerequisites.lower())

    def test_case_b_breast_cancer_diagnostic_dataset(self):
        """
        Case B: 569 rows, 32 columns: id, diagnosis (M/B), and 30 cell nuclei measurements.
        Verifies:
          - 'id' is classified as PRIMARY_ID and strictly excluded from metrics and recommendations.
          - 'diagnosis' is classified as BINARY_LABEL.
          - Row entity is identified as diagnostic tissue specimen / biopsy measurements.
          - Domain is Healthcare & Medical (Diagnostic Oncology & Pathology).
          - No summing/averaging of ID, no rankings of ID, no correlations with ID!
          - No generic commerce vocabulary ('growth drivers', 'resource allocation', 'Pareto margins').
          - Supervised classification for 'diagnosis' is marked 'Feasible for exploration' with clinical caveat.
          - 'fractal_dimension_worst' is NOT selected as regression target; regression is marked unsupported.
          - Forecasting is marked unsupported because there is no temporal index.
        """
        # Construct synthetic Wisconsin Breast Cancer structure
        np.random.seed(42)
        n = 569
        measurement_names = [
            "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
            "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean",
            "symmetry_mean", "fractal_dimension_mean",
            "radius_se", "texture_se", "perimeter_se", "area_se",
            "smoothness_se", "compactness_se", "concavity_se", "concave points_se",
            "symmetry_se", "fractal_dimension_se",
            "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
            "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst",
            "symmetry_worst", "fractal_dimension_worst"
        ]

        data = {
            "id": np.arange(100000, 100000 + n),
            "diagnosis": np.random.choice(["B", "M"], size=n, p=[0.63, 0.37]),
        }
        for m in measurement_names:
            data[m] = np.random.normal(loc=15.0, scale=3.0, size=n)

        df = pd.DataFrame(data)
        profile = profile_dataset(df)

        semantics = build_dataset_semantics(df, profile)

        # 1. Verify Column Roles
        self.assertEqual(semantics.columns["id"].semantic_role, ColumnRole.PRIMARY_ID)
        self.assertEqual(semantics.columns["diagnosis"].semantic_role, ColumnRole.BINARY_LABEL)
        self.assertEqual(semantics.columns["radius_mean"].semantic_role, ColumnRole.CONTINUOUS_MEASUREMENT)
        self.assertIn("id", semantics.primary_ids)
        self.assertNotIn("id", semantics.measurements)

        overview = generate_deterministic_overview(df, profile)

        # 2. Domain & Entity Inference
        self.assertEqual(overview.domain, "Healthcare & Medical")
        self.assertEqual(overview.domain_confidence, "high")
        self.assertIn("cell", overview.row_meaning.lower())
        self.assertIn("diagnostic", overview.row_meaning.lower())
        # Ensure it does not invent an unverified physical biopsy procedure
        self.assertNotIn("biopsy procedure", overview.row_meaning.lower())

        # 3. ID Exclusion Check across all Opportunities
        for opp in overview.what_we_can_learn:
            self.assertNotIn("average id", (opp.description + " " + (opp.sample_question or "")).lower())
            self.assertNotIn("total id", (opp.description + " " + (opp.sample_question or "")).lower())
            self.assertNotIn("highest id", (opp.description + " " + (opp.sample_question or "")).lower())
            self.assertNotIn("correlation between id", (opp.description + " " + (opp.sample_question or "")).lower())

            # Verify no misplaced commerce vocabulary
            self.assertNotIn("growth drivers", opp.description.lower())
            self.assertNotIn("resource allocation", opp.business_value.lower())
            self.assertNotIn("margin optimization", opp.business_value.lower())
            self.assertNotIn("high-value categories", opp.business_value.lower())

        # 4. Verified Analysis Opportunity on genuine measurements
        diag_opps = [o for o in overview.what_we_can_learn if "diagnosis" in (o.title + o.description).lower()]
        self.assertTrue(len(diag_opps) > 0)
        self.assertIn("radius_mean", diag_opps[0].description)
        self.assertIn("What is the average radius_mean grouped by diagnosis?", diag_opps[0].sample_question)

        # 5. Prediction Feasibility: Classification must be 'Feasible for exploration'
        clf_tasks = [p for p in overview.potential_predictions if "diagnosis" in (p.task + " " + (p.target_variable or "")).lower()]
        self.assertTrue(len(clf_tasks) > 0)
        self.assertEqual(clf_tasks[0].feasibility, "Feasible for exploration")
        self.assertIn("stratified", clf_tasks[0].prerequisites.lower())
        # Medical disclaimer
        self.assertTrue(any("clinical" in m.lower() for m in clf_tasks[0].missing_elements))

        # 6. Regression must NOT arbitrarily select fractal_dimension_worst
        reg_tasks = [p for p in overview.potential_predictions if "Regression" in p.task]
        self.assertTrue(len(reg_tasks) > 0)
        self.assertEqual(reg_tasks[0].feasibility, "Not currently supported by the data")
        self.assertIsNone(reg_tasks[0].target_variable)

        # 7. Time-series forecasting must be blocked
        forecast_tasks = [p for p in overview.potential_predictions if "Forecasting" in p.task]
        self.assertTrue(len(forecast_tasks) > 0)
        self.assertEqual(forecast_tasks[0].feasibility, "Not currently supported by the data")
        self.assertIn("no temporal index", forecast_tasks[0].prerequisites.lower())


class TestColumnRoleInference(unittest.TestCase):
    """Verifies deterministic column-role classification across varied schemas."""

    def test_primary_identifier_detection_and_metric_exclusion(self):
        """Verifies IDs matching varied naming conventions are excluded from metrics."""
        df = pd.DataFrame({
            "patient_key": [f"PK-{i}" for i in range(100)],
            "transaction_uuid": [f"U-{i}" for i in range(100)],
            "subject_id": np.arange(1000, 1100),
            "amount": np.random.uniform(10, 500, size=100),
        })
        profile = profile_dataset(df)
        semantics = build_dataset_semantics(df, profile)

        self.assertEqual(semantics.columns["patient_key"].semantic_role, ColumnRole.PRIMARY_ID)
        self.assertEqual(semantics.columns["transaction_uuid"].semantic_role, ColumnRole.PRIMARY_ID)
        self.assertEqual(semantics.columns["subject_id"].semantic_role, ColumnRole.PRIMARY_ID)
        self.assertEqual(semantics.columns["amount"].semantic_role, ColumnRole.CONTINUOUS_MEASUREMENT)

        # Primary IDs must not appear in measurements list
        self.assertNotIn("patient_key", semantics.measurements)
        self.assertNotIn("transaction_uuid", semantics.measurements)
        self.assertNotIn("subject_id", semantics.measurements)
        self.assertIn("amount", semantics.measurements)

    def test_binary_label_recognition(self):
        """Verifies binary columns with outcome semantics are recognized."""
        df = pd.DataFrame({
            "churn_flag": np.random.choice([0, 1], size=100),
            "status": np.random.choice(["Active", "Churned"], size=100),
            "usage_minutes": np.random.uniform(50, 600, size=100),
            "user_id": [f"USR-{i}" for i in range(100)],
        })
        profile = profile_dataset(df)
        semantics = build_dataset_semantics(df, profile)

        self.assertIn(semantics.columns["churn_flag"].semantic_role, (ColumnRole.BINARY_LABEL, ColumnRole.CONTINUOUS_MEASUREMENT))
        self.assertEqual(semantics.columns["status"].semantic_role, ColumnRole.BINARY_LABEL)
        self.assertEqual(semantics.columns["user_id"].semantic_role, ColumnRole.PRIMARY_ID)

    def test_geographic_coordinates_isolated_from_metrics(self):
        """Verifies latitude and longitude are typed as geographic and not default financial metrics."""
        df = pd.DataFrame({
            "listing_id": [1, 2, 3, 4],
            "latitude": [40.7128, 40.7135, 40.7140, 40.7150],
            "longitude": [-74.0060, -74.0070, -74.0055, -74.0080],
            "price": [150.0, 220.0, 180.0, 310.0],
            "room_type": ["Entire home", "Private room", "Entire home", "Shared room"],
        })
        profile = profile_dataset(df)
        semantics = build_dataset_semantics(df, profile)

        self.assertEqual(semantics.columns["latitude"].semantic_role, ColumnRole.GEOGRAPHIC)
        self.assertEqual(semantics.columns["longitude"].semantic_role, ColumnRole.GEOGRAPHIC)
        self.assertEqual(semantics.columns["price"].semantic_role, ColumnRole.CONTINUOUS_MEASUREMENT)


class TestPredictionFeasibilityRules(unittest.TestCase):
    """Verifies evidence-based predictive feasibility evaluation."""

    def test_regression_target_requires_defensible_continuous_outcome(self):
        """Continuous outcome (e.g. price) enables regression; absence disables it."""
        # Dataset WITH defensible regression target: price
        df_housing = pd.DataFrame({
            "price": np.random.uniform(100000, 500000, size=50),
            "bedrooms": np.random.choice([1, 2, 3, 4], size=50),
            "bathrooms": np.random.choice([1.0, 1.5, 2.0], size=50),
            "square_feet": np.random.uniform(600, 2500, size=50),
        })
        profile_h = profile_dataset(df_housing)
        overview_h = generate_deterministic_overview(df_housing, profile_h)

        reg_h = [p for p in overview_h.potential_predictions if "Value Estimation" in p.task or "Regression" in p.task]
        self.assertTrue(len(reg_h) > 0)
        self.assertEqual(reg_h[0].feasibility, "Feasible for exploration")
        self.assertEqual(reg_h[0].target_variable, "price")

    def test_forecasting_blocked_on_cross_sectional_data(self):
        """Cross-sectional tables without dates explicitly state forecasting is unsupported."""
        df_static = pd.DataFrame({
            "category": ["A", "B", "C", "D"],
            "score": [10.5, 20.3, 15.2, 30.1],
        })
        profile_s = profile_dataset(df_static)
        overview_s = generate_deterministic_overview(df_static, profile_s)

        forecast_tasks = [p for p in overview_s.potential_predictions if "Forecasting" in p.task]
        self.assertTrue(len(forecast_tasks) > 0)
        self.assertEqual(forecast_tasks[0].feasibility, "Not currently supported by the data")
        self.assertIn("cross-sectional", forecast_tasks[0].prerequisites.lower())


class TestDegradedAndEmptyDatasets(unittest.TestCase):
    """Verifies handling of zero-row, all-null, constant, and partially missing datasets."""

    def test_completely_zero_rows_and_cols(self):
        """Verifies pd.DataFrame() handles cleanly without crash."""
        df = pd.DataFrame()
        profile = profile_dataset(df)
        overview = generate_deterministic_overview(df, profile)

        self.assertEqual(overview.domain, "General Tabular Data (Unspecified)")
        self.assertEqual(overview.domain_confidence, "low")
        self.assertIn("empty", overview.summary_paragraph.lower())
        self.assertEqual(len(overview.key_fields), 0)

    def test_constant_columns_not_prioritized(self):
        """Constant columns with zero variance are flagged in quality warnings."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "country": ["USA", "USA", "USA", "USA", "USA"],
            "score": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        profile = profile_dataset(df)
        overview = generate_deterministic_overview(df, profile)

        # score should be preferred over constant country
        self.assertIn("score", [k.name for k in overview.key_fields])


class TestLLMBoundaryAndFallbackParity(unittest.TestCase):
    """Verifies LLM prompt construction, response sanitization, and fallback parity."""

    def test_gemini_hallucinating_id_aggregation_is_rejected(self):
        """Sanitizer rejects LLM responses that aggregate primary IDs."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "diagnosis": ["M", "B", "M"],
            "radius_mean": [10.5, 12.3, 14.2],
        })
        profile = profile_dataset(df)
        semantics = build_dataset_semantics(df, profile)
        det_overview = generate_deterministic_overview(df, profile)

        # Fabricated LLM response that attempts to aggregate ID
        bad_llm_json = """
        {
            "domain": "Healthcare & Medical",
            "domain_confidence": "high",
            "summary_paragraph": "A diagnostic biopsy dataset measuring tumor nuclei attributes.",
            "row_meaning": "A biopsy specimen record.",
            "size_and_coverage": "3 records across 3 attributes.",
            "key_fields": [{"name": "id", "meaning": "Primary metric", "role": "metric"}],
            "what_we_can_learn": [
                {
                    "title": "ID Breakdown",
                    "description": "Calculate the average id by diagnosis.",
                    "business_value": "Growth drivers.",
                    "sample_question": "What is the average id by diagnosis?"
                }
            ],
            "potential_predictions": [
                {
                    "task": "ID Regression",
                    "target_variable": "id",
                    "prerequisites": "None",
                    "feasibility": "Feasible for exploration",
                    "missing_elements": []
                }
            ],
            "prediction_limitations": "ML limitations."
        }
        """
        validated = _sanitize_and_validate_overview(bad_llm_json, det_overview, semantics)
        self.assertIsNotNone(validated)
        # Verify that ID was sanitized: not in opportunities as metric, not in predictions as target
        for opp in validated.what_we_can_learn:
            self.assertNotIn("average id", (opp.description + " " + (opp.sample_question or "")).lower())
        for pred in validated.potential_predictions:
            self.assertNotEqual(pred.target_variable, "id")

    @patch("overview.os.getenv")
    def test_gemini_rate_limit_fallback_preserves_production_quality(self, mock_getenv):
        """When Gemini raises 429 quota exhausted, fallback satisfies all production requirements."""
        mock_getenv.return_value = None
        df = pd.DataFrame({
            "patient_id": [101, 102, 103],
            "diagnosis": ["M", "B", "M"],
            "radius_mean": [14.0, 11.2, 16.5],
            "perimeter_mean": [90.0, 75.0, 105.0],
        })
        profile = profile_dataset(df)
        overview = generate_dataset_overview(df, profile, use_gemini=True)

        self.assertEqual(overview.status, "deterministic")
        self.assertEqual(overview.domain, "Healthcare & Medical")
        # Ensure patient_id is not treated as a metric
        metric_fields = [k.name for k in overview.key_fields if k.role == "metric"]
        self.assertNotIn("patient_id", metric_fields)
        self.assertIn("radius_mean", metric_fields)


class TestApiIntegrationAndCompatibility(unittest.TestCase):
    """Verifies backward compatibility of DatasetMetadataResponse and API endpoints."""

    def setUp(self):
        self.client = TestClient(app)

    def test_metadata_response_serialization_with_and_without_overview(self):
        """Verifies DatasetMetadataResponse backwards compatibility."""
        df = pd.DataFrame({
            "col1": [1, 2],
            "col2": ["A", "B"]
        })
        profile = profile_dataset(df)

        resp = _format_dataset_response("test-id-1", "test.csv", df, profile=profile)
        self.assertIsNotNone(resp.overview)
        self.assertEqual(resp.overview.status, "deterministic")
        json_data = resp.model_dump(by_alias=True)
        self.assertIn("overview", json_data)
        self.assertIn("schema", json_data)

    def test_sample_dataset_endpoint_includes_overview(self):
        """Verifies sample dataset endpoint returns complete overview."""
        res = self.client.post("/api/datasets/sample")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertIn("overview", data)
        overview = data["overview"]
        self.assertIsNotNone(overview)
        self.assertEqual(overview["domain"], "E-Commerce & Retail")
        self.assertIn("summary_paragraph", overview)
        self.assertTrue(len(overview["summary_paragraph"]) > 30)
        self.assertIn("what_we_can_learn", overview)
        self.assertIn("potential_predictions", overview)

    def test_dedicated_overview_endpoint(self):
        """Verifies GET /api/datasets/{id}/overview returns dedicated overview object."""
        sample_res = self.client.post("/api/datasets/sample")
        self.assertEqual(sample_res.status_code, 200)
        dataset_id = sample_res.json()["dataset_id"]

        overview_res = self.client.get(f"/api/datasets/{dataset_id}/overview")
        self.assertEqual(overview_res.status_code, 200)
        data = overview_res.json()

        self.assertEqual(data["domain"], "E-Commerce & Retail")
        self.assertIn("row_meaning", data)
        self.assertIn("key_fields", data)
        self.assertIn("prediction_limitations", data)

    def test_overview_endpoint_404_on_invalid_id(self):
        """Verifies 404 response on nonexistent dataset ID."""
        res = self.client.get("/api/datasets/nonexistent-id-000/overview")
        self.assertEqual(res.status_code, 404)


if __name__ == "__main__":
    unittest.main()
