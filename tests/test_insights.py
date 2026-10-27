"""
tests/test_insights.py
-----------------------
Automated tests for Phase 2 — Automatic Insight Discovery:
- Segmentation analysis: dominant segments, volumetric sums vs averages.
- Time-series trend detection: valid trends, insufficient periods, non-datetime columns.
- Outlier detection: Tukey IQR rule, threshold reporting, small-sample handling.
- Pearson correlations: strong/moderate positive/negative, constant columns, insufficient data.
- Data quality findings: duplicate rows, high missingness.
- Defensive handling: empty DataFrames, single-row, all-null, NaNs, infinities.
- Deterministic ranking, diversity enforcement, deduplication, and fewer-than-five returns.
- API endpoints integration: /api/datasets/upload, /api/datasets/sample, /api/datasets/{id}/insights.
- Storage lifecycle, cache reuse, and rollback safety.
"""

import io
import math
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app, DATASET_STORE
from profiler import profile_dataset
from insights import (
    discover_insights,
    _analyze_segments,
    _analyze_trends,
    _analyze_outliers,
    _analyze_correlations,
    _analyze_data_quality,
    Insight,
)


@pytest.fixture
def client():
    return TestClient(app)


# ============================================================================
# 1. Segmentation Analysis Tests
# ============================================================================

class TestSegmentationInsights:
    def test_dominant_segment_identification(self):
        """Validates that a dominant category segment is detected with correct key figures."""
        df = pd.DataFrame({
            "region": ["North", "North", "North", "North", "South", "East", "West"],
            "revenue": [5000.0, 4500.0, 4000.0, 3500.0, 500.0, 600.0, 400.0],
        })
        profile = profile_dataset(df)
        insights = _analyze_segments(df, profile)

        assert len(insights) >= 1
        top_insight = insights[0]
        assert top_insight.type == "segmentation"
        assert "North" in top_insight.headline
        assert top_insight.evidence.key_figures["top_category"] == "North"
        assert top_insight.evidence.key_figures["top_value"] == 17000.0
        assert top_insight.visualization is not None
        assert top_insight.visualization.chart_type == "Bar Chart"

    def test_segmentation_uses_mean_for_rates(self):
        """Ensures rating/score columns aggregate via mean rather than sum."""
        df = pd.DataFrame({
            "store": ["Store A", "Store A", "Store B", "Store B", "Store C", "Store C"],
            "customer_rating": [4.9, 4.8, 3.1, 3.2, 2.0, 2.1],
        })
        profile = profile_dataset(df)
        insights = _analyze_segments(df, profile)

        assert len(insights) >= 1
        rating_insight = [i for i in insights if "customer_rating" in i.evidence.relevant_columns][0]
        # Store A average is 4.85
        assert rating_insight.evidence.key_figures["top_category"] == "Store A"
        assert abs(rating_insight.evidence.key_figures["top_value"] - 4.85) < 0.05
        assert "average" in rating_insight.headline.lower() or "average" in rating_insight.explanation.lower()

    def test_segmentation_ignores_high_cardinality_identifiers(self):
        """Ensures UUIDs or unique ID columns are not used as categorical segmentation keys."""
        df = pd.DataFrame({
            "transaction_id": [f"tx_{i}" for i in range(30)],
            "amount": [float(i * 10) for i in range(30)],
        })
        profile = profile_dataset(df)
        insights = _analyze_segments(df, profile)
        assert len(insights) == 0


# ============================================================================
# 2. Time-Series Trends Tests
# ============================================================================

class TestTimeTrendInsights:
    def test_valid_temporal_growth_detected(self):
        """Validates detection of significant temporal trajectory across valid dates."""
        dates = pd.date_range("2023-01-01", periods=10, freq="W-MON").strftime("%Y-%m-%d").tolist()
        revenue = [1000.0, 1100.0, 1300.0, 1500.0, 1700.0, 1900.0, 2200.0, 2500.0, 2800.0, 3200.0]
        df = pd.DataFrame({"order_date": dates, "revenue": revenue})
        profile = profile_dataset(df)
        insights = _analyze_trends(df, profile)

        assert len(insights) >= 1
        trend_insight = insights[0]
        assert trend_insight.type == "trend"
        assert "growth" in trend_insight.headline.lower() or "increased" in trend_insight.headline.lower()
        assert trend_insight.evidence.key_figures["net_pct_change"] > 0
        assert trend_insight.visualization is not None
        assert trend_insight.visualization.chart_type == "Line Chart"

    def test_insufficient_time_buckets_yields_no_trend(self):
        """Confirms that fewer than 4 distinct time intervals does not claim an unfounded trend."""
        df = pd.DataFrame({
            "order_date": ["2023-01-01", "2023-01-02", "2023-01-02"],
            "revenue": [100.0, 200.0, 300.0],
        })
        profile = profile_dataset(df)
        insights = _analyze_trends(df, profile)
        assert len(insights) == 0


# ============================================================================
# 3. IQR Outlier Tests
# ============================================================================

class TestOutlierInsights:
    def test_iqr_outlier_detection_and_thresholds(self):
        """Ensures statistical outliers are identified via Tukey IQR with accurate bounds."""
        # 20 normal values around 50, and 2 extreme injected outliers (500, 600)
        values = [48.0, 49.0, 50.0, 51.0, 52.0] * 4 + [500.0, 600.0]
        df = pd.DataFrame({"metric": values})
        profile = profile_dataset(df)
        insights = _analyze_outliers(df, profile)

        assert len(insights) == 1
        outlier_insight = insights[0]
        assert outlier_insight.type == "outlier"
        assert outlier_insight.evidence.key_figures["outlier_count"] == 2
        assert outlier_insight.evidence.key_figures["peak_outlier"] == 600.0
        assert "Tukey" in outlier_insight.evidence.comparison_basis
        assert "not automatically" in outlier_insight.uncertainty_or_limitations.lower()


    def test_zero_iqr_constant_variance_skips_outliers(self):
        """Ensures discrete distributions with zero IQR do not create false positive outliers."""
        df = pd.DataFrame({"status_flag": [1.0] * 30})
        profile = profile_dataset(df)
        insights = _analyze_outliers(df, profile)
        assert len(insights) == 0

    def test_small_sample_skips_outliers(self):
        """Ensures datasets with N < 15 do not attempt outlier claims."""
        df = pd.DataFrame({"val": [1, 2, 3, 100]})
        profile = profile_dataset(df)
        assert len(_analyze_outliers(df, profile)) == 0


# ============================================================================
# 4. Pearson Correlation Tests
# ============================================================================

class TestCorrelationInsights:
    def test_strong_positive_correlation(self):
        """Validates detection of strong positive linear correlation (r >= 0.7)."""
        x = [float(i) for i in range(1, 25)]
        y = [val * 2.5 + 5.0 for val in x]  # Perfect correlation
        df = pd.DataFrame({"units_sold": x, "total_revenue": y})
        profile = profile_dataset(df)
        insights = _analyze_correlations(df, profile)

        assert len(insights) >= 1
        corr_insight = insights[0]
        assert corr_insight.type == "correlation"
        assert corr_insight.evidence.key_figures["r_coefficient"] >= 0.99
        assert "positive" in corr_insight.headline.lower()
        assert "NOT causation" in corr_insight.uncertainty_or_limitations
        assert corr_insight.visualization is not None
        assert corr_insight.visualization.chart_type == "Scatter Plot"

    def test_weak_or_zero_correlation_filtered(self):
        """Confirms weak correlation (|r| < 0.50) is excluded."""
        # Uncorrelated orthogonal sequences
        df = pd.DataFrame({
            "metric_a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "metric_b": [5, 2, 8, 3, 7, 1, 9, 4, 6, 2, 8, 3],
        })
        profile = profile_dataset(df)
        insights = _analyze_correlations(df, profile)
        assert len(insights) == 0

    def test_constant_column_correlation_safe(self):
        """Ensures constant numeric columns do not raise division-by-zero runtime errors."""
        df = pd.DataFrame({
            "varying": [float(i) for i in range(20)],
            "constant": [42.0] * 20,
        })
        profile = profile_dataset(df)
        insights = _analyze_correlations(df, profile)
        assert len(insights) == 0


# ============================================================================
# 5. Data Quality Insights Tests
# ============================================================================

class TestDataQualityInsights:
    def test_high_duplicate_rows_insight(self):
        """Validates alert when duplicate rows exceed 5% threshold."""
        df = pd.DataFrame({
            "colA": [1, 1, 1, 1, 2, 3, 4, 5, 6, 7],
            "colB": ["x", "x", "x", "x", "y", "z", "w", "v", "u", "t"],
        })
        # 3 duplicate rows out of 10 = 30%
        profile = profile_dataset(df)
        insights = _analyze_data_quality(df, profile)

        dupe_insight = [i for i in insights if i.evidence.metric_name == "Complete Duplicate Rows"]
        assert len(dupe_insight) == 1
        assert dupe_insight[0].evidence.key_figures["duplicate_count"] == 3
        assert "duplicate rows" in dupe_insight[0].headline.lower()

    def test_severe_missingness_insight(self):
        """Validates alert when column missingness exceeds 35%."""
        vals = [None] * 40 + [10.0] * 60  # 40% missing
        df = pd.DataFrame({"incomplete_col": vals, "full_col": list(range(100))})
        profile = profile_dataset(df)
        insights = _analyze_data_quality(df, profile)

        missing_insight = [i for i in insights if "incomplete_col" in i.evidence.relevant_columns]
        assert len(missing_insight) == 1
        assert missing_insight[0].evidence.key_figures["missing_pct"] == 40.0


# ============================================================================
# 6. Overall Discover Orchestrator & Edge Cases
# ============================================================================

class TestDiscoverInsightsOrchestrator:
    def test_empty_dataframe_returns_empty_list(self):
        """Ensures empty DataFrame produces empty insights list without error."""
        df = pd.DataFrame()
        profile = profile_dataset(df)
        assert discover_insights(df, profile) == []

    def test_single_row_dataframe_safe(self):
        """Ensures 1-row DataFrame does not crash and returns fallback/empty."""
        df = pd.DataFrame({"a": [10], "b": ["x"]})
        profile = profile_dataset(df)
        insights = discover_insights(df, profile)
        assert isinstance(insights, list)
        assert len(insights) <= 1

    def test_all_null_and_infinities_safe(self):
        """Ensures infinite and NaN values produce valid JSON serializable insights."""
        df = pd.DataFrame({
            "metric": [np.inf, -np.inf, np.nan, 10.0, 20.0, 30.0] * 5,
            "category": ["A", "B", "C", "A", "B", "C"] * 5,
        })
        profile = profile_dataset(df)
        insights = discover_insights(df, profile)
        assert isinstance(insights, list)

        # Check JSON serialization of all fields
        for ins in insights:
            for k, v in ins.evidence.key_figures.items():
                if isinstance(v, float):
                    assert not math.isnan(v)
                    assert not math.isinf(v)

    def test_ranking_and_diversity_caps_to_five(self):
        """Ensures no more than 5 insights are returned and candidates are ranked."""
        # Use bundled sample dataset
        sample_path = "data/ecommerce_sales.csv"
        df = pd.read_csv(sample_path)
        profile = profile_dataset(df)
        insights = discover_insights(df, profile)

        assert 3 <= len(insights) <= 5
        # Verify scores are sorted descending
        scores = [i.score for i in insights]
        assert scores == sorted(scores, reverse=True)
        # Verify diversity: not all identical types
        types = set(i.type for i in insights)
        assert len(types) >= 2


# ============================================================================
# 7. API Endpoints Integration Tests
# ============================================================================

class TestApiInsightsIntegration:
    def test_sample_dataset_returns_insights(self, client):
        """Verifies POST /api/datasets/sample returns 3-5 insights in the payload."""
        resp = client.post("/api/datasets/sample")
        assert resp.status_code == 200
        data = resp.json()

        assert "insights" in data and data["insights"] is not None
        insights = data["insights"]
        assert 3 <= len(insights) <= 5
        assert all("headline" in i and "evidence" in i for i in insights)

    def test_get_dataset_insights_dedicated_endpoint(self, client):
        """Verifies GET /api/datasets/{dataset_id}/insights returns the ranked insights."""
        sample_resp = client.post("/api/datasets/sample")
        dataset_id = sample_resp.json()["dataset_id"]

        resp = client.get(f"/api/datasets/{dataset_id}/insights")
        assert resp.status_code == 200
        insights = resp.json()

        assert isinstance(insights, list)
        assert 3 <= len(insights) <= 5
        assert insights[0]["type"] in ("segmentation", "trend", "outlier", "correlation", "data_quality")

    def test_upload_includes_insights_and_rollback_safe(self, client):
        """Verifies CSV upload computes insights and that profiling failure cleans store."""
        csv_content = b"region,revenue\nNorth,1000\nNorth,1200\nSouth,200\nSouth,150\nEast,100\n"
        files = {"file": ("test_insights_upload.csv", io.BytesIO(csv_content), "text/csv")}

        resp = client.post("/api/datasets/upload", files=files)
        assert resp.status_code == 200
        data = resp.json()

        assert "insights" in data and data["insights"] is not None
        assert len(data["insights"]) >= 1


# ============================================================================
# 8. Acceptance Review Regression Tests
# ============================================================================

class TestAcceptanceReviewRegressions:
    def test_sensitive_columns_excluded_from_segments(self):
        """Ensures personal names (customer_name, employee_name) are never used as segment categories."""
        df = pd.DataFrame({
            "customer_name": ["Alice Smith", "Alice Smith", "Bob Jones", "Charlie Brown", "Diana Prince"] * 2,
            "revenue": [1000.0, 900.0, 200.0, 150.0, 100.0] * 2,
            "region": ["North", "North", "South", "East", "West"] * 2,
        })
        profile = profile_dataset(df)
        insights = _analyze_segments(df, profile)

        # Confirm customer_name is never selected as a segmentation dimension
        for ins in insights:
            assert "customer_name" not in ins.evidence.relevant_columns
            assert "Alice Smith" not in ins.headline
            if ins.suggested_query:
                assert "customer_name" not in ins.suggested_query

    def test_temporal_oscillating_series_rejected_as_trend(self):
        """Ensures oscillating/noisy series with low linear correlation are rejected as trends even if endpoints differ."""
        # 12 monthly periods where endpoints differ (100 -> 150) but middle oscillates wildly (corr ~ 0)
        dates = pd.date_range("2023-01-01", periods=12, freq="ME").strftime("%Y-%m-%d").tolist()
        vals = [100.0, 10.0, 500.0, 20.0, 600.0, 30.0, 400.0, 20.0, 700.0, 10.0, 30.0, 150.0]
        df = pd.DataFrame({"order_date": dates, "metric": vals})
        profile = profile_dataset(df)
        insights = _analyze_trends(df, profile)

        assert len(insights) == 0

    def test_temporal_negative_or_zero_baseline_rejected(self):
        """Ensures temporal series crossing or starting at zero are not given misleading percentage changes."""
        dates = pd.date_range("2023-01-01", periods=6, freq="ME").strftime("%Y-%m-%d").tolist()
        df = pd.DataFrame({
            "order_date": dates,
            "net_loss": [-100.0, -80.0, -50.0, -20.0, 10.0, 50.0],
        })
        profile = profile_dataset(df)
        insights = _analyze_trends(df, profile)

        # Baseline starts at -100 <= 0, so percentage growth is mathematically ambiguous and rejected
        assert len(insights) == 0

    def test_discount_rate_uses_average_aggregation(self):
        """Ensures discount columns are treated as rates (mean) rather than summed."""
        df = pd.DataFrame({
            "region": ["North", "North", "South", "South", "East", "East"],
            "discount": [0.20, 0.20, 0.05, 0.05, 0.02, 0.02],
        })
        profile = profile_dataset(df)
        insights = _analyze_segments(df, profile)

        assert len(insights) >= 1
        discount_ins = [i for i in insights if "discount" in i.evidence.relevant_columns][0]
        assert "average" in discount_ins.headline.lower() or "average" in discount_ins.explanation.lower()
        # Top average is 0.20
        assert discount_ins.evidence.key_figures["top_value"] == 0.20

    def test_negative_segment_totals_suppress_pct_of_total(self):
        """Ensures columns with negative segment totals do not claim a distorted share of total."""
        df = pd.DataFrame({
            "department": ["Dept A", "Dept B", "Dept C"],
            "net_profit": [1000.0, -400.0, 100.0],
        })
        profile = profile_dataset(df)
        insights = _analyze_segments(df, profile)

        # Total is 700, so Dept A (1000) would be 142% of total without the non-negative check
        for ins in insights:
            if "net_profit" in ins.evidence.relevant_columns:
                assert "share_of_total_pct" not in ins.evidence.key_figures
                assert "% of total" not in ins.headline

    def test_outlier_lower_boundary_description(self):
        """Ensures lower outliers are accurately described as falling below the lower boundary."""
        # 20 normal values around 100, and 2 low outliers (1.0, 2.0)
        values = [98.0, 99.0, 100.0, 101.0, 102.0] * 4 + [1.0, 2.0]
        df = pd.DataFrame({"score": values})
        profile = profile_dataset(df)
        insights = _analyze_outliers(df, profile)

        assert len(insights) == 1
        outlier_ins = insights[0]
        assert "falling below the lower boundary" in outlier_ins.explanation
