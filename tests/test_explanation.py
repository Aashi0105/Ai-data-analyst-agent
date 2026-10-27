"""
tests/test_explanation.py
-------------------------
Tests for Feature A: Analytical Explanation Engine (explanation.py) & API Integration.

Acceptance Test Matrix:
1. Evidence calculation: Additive group breakdown (sales/revenue sum) with share-of-total.
2. Evidence calculation: Non-additive metrics (averages, rates, ratios, scores, balances) safely suppress shares.
3. Ambiguous metrics: Semantics not demonstrably additive safely suppress shares without guessing.
4. Negative values guard: Net profit/losses suppress percentage-of-total.
5. Zero denominator and zero sum safely handled without ZeroDivisionError.
6. Non-finite values: Infinity and -infinity safely dropped without crashing.
7. Duplicate category labels: Duplicate index labels deduplicate correctly.
8. Single category results: Suppresses comparative ratios and percentage shares cleanly.
9. Extremely small denominators: Suppresses runaway disparity ratios.
10. Scalar results: Strictly compatible baselines (sum vs total sum, mean vs dataset mean).
11. Empty / null / non-numeric results handle gracefully.
12. Epistemic modesty: Non-causal disclosures, no unsupported causal assertions.
13. Robust fallback: Gemini failure, timeout (4s budget), rate-limit (429), or malformed JSON.
14. Gemini consistency audit: Hallucinated percentages trigger fallback to deterministic finding.
15. Error sanitization: Credentials, bearer tokens, and API keys are redacted from logs/errors.
16. API backward compatibility: POST /api/analyze returns valid explanation without breaking schema.
"""

import math
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from explanation import (
    calculate_answer_evidence,
    generate_deterministic_explanation,
    explain_answer,
    _is_additive_metric,
    _sanitize_error_message,
    AnswerEvidence,
    AnswerExplanation,
)
from backend.main import app


@pytest.fixture
def sample_sales_df():
    """Sample retail transactions DataFrame for testing explanations."""
    return pd.DataFrame({
        "Category": ["Technology", "Furniture", "Office Supplies", "Technology", "Furniture"],
        "Region": ["West", "East", "West", "East", "Central"],
        "Revenue": [1000.0, 500.0, 300.0, 700.0, 500.0],
        "Discount": [0.1, 0.2, 0.15, 0.05, 0.25],
        "Profit": [200.0, -50.0, 40.0, 150.0, -20.0],
        "Rating": [4.5, 3.8, 4.2, 4.9, 3.5],
        "Balance": [1500.0, 200.0, 350.0, 1200.0, 800.0],
    })


# ============================================================================
# 1. Additive Metrics & Share of Total Tests
# ============================================================================

def test_additive_group_breakdown_calculates_shares_correctly(sample_sales_df):
    """
    Volumetric non-negative sums (e.g. Revenue by Category) must compute
    share of total, top contributor, runner-up, and disparity ratio.
    """
    grouped_revenue = sample_sales_df.groupby("Category")["Revenue"].sum()
    # Technology: 1700, Furniture: 1000, Office Supplies: 300 -> Total = 3000

    evidence = calculate_answer_evidence(
        result=grouped_revenue,
        df=sample_sales_df,
        question="What is the total revenue by category?"
    )

    assert evidence.is_additive is True
    assert evidence.aggregation_type == "sum"
    assert evidence.total_value == 3000.0
    assert evidence.top_contributor == "Technology"
    assert evidence.top_value == 1700.0
    assert round(evidence.top_share_pct, 1) == 56.7

    assert evidence.runner_up == "Furniture"
    assert evidence.runner_up_value == 1000.0
    assert round(evidence.runner_up_share_pct, 1) == 33.3

    assert evidence.bottom_contributor == "Office Supplies"
    assert evidence.bottom_value == 300.0

    # Disparity: 1700 / 1000 = 1.7x
    assert evidence.disparity_ratio == 1.7
    assert evidence.disparity_difference == 700.0
    assert evidence.num_groups == 3
    assert evidence.sample_size == len(sample_sales_df)


def test_additive_dataframe_input_delegates_properly(sample_sales_df):
    """Two-column DataFrame (Category, Sum) calculates identical evidence to Series."""
    df_res = sample_sales_df.groupby("Category", as_index=False)["Revenue"].sum()

    evidence = calculate_answer_evidence(
        result=df_res,
        df=sample_sales_df,
        question="Show revenue by category"
    )

    assert evidence.is_additive is True
    assert evidence.top_contributor == "Technology"
    assert evidence.top_value == 1700.0
    assert round(evidence.top_share_pct, 1) == 56.7


# ============================================================================
# 2. Non-Additive Metrics (Averages, Rates) Safeguards
# ============================================================================

def test_averages_safely_suppress_shares_of_total(sample_sales_df):
    """
    Averages (e.g. Average Discount by Category) must NOT sum into shares of total.
    Must set is_additive=False, top_share_pct=None, and state limitation.
    """
    grouped_avg = sample_sales_df.groupby("Category")["Discount"].mean()
    # Office Supplies: 0.15, Furniture: 0.225, Technology: 0.075

    evidence = calculate_answer_evidence(
        result=grouped_avg,
        df=sample_sales_df,
        question="What is the average discount across categories?"
    )

    assert evidence.is_additive is False
    assert evidence.aggregation_type == "average"
    assert evidence.top_share_pct is None
    assert evidence.total_value is None
    assert "average" in evidence.limitation_notes.lower()
    assert evidence.top_contributor == "Furniture"

    # Deterministic explanation must not claim shares of total
    explanation = generate_deterministic_explanation(evidence, "Average discount", grouped_avg)
    assert "% of" not in explanation.finding
    assert "share of total" not in explanation.finding.lower()
    assert "non-additive" in explanation.uncertainty_or_limitations.lower()


def test_is_additive_metric_keywords():
    """Verify non-additive keywords are safely recognized."""
    s = pd.Series([10.5, 20.0, 15.2])
    is_add, agg_type, limit = _is_additive_metric("avg_price", "What is the avg price?", s)
    assert is_add is False
    assert "average" in agg_type or "ratio" in agg_type

    is_add2, agg_type2, _ = _is_additive_metric("median_salary", "median salary", s)
    assert is_add2 is False

    is_add3, agg_type3, _ = _is_additive_metric("conversion_rate", "rate by campaign", s)
    assert is_add3 is False

    is_add4, agg_type4, _ = _is_additive_metric("customer_rating", "rating by category", s)
    assert is_add4 is False

    is_add5, agg_type5, _ = _is_additive_metric("account_balance", "balance by user", s)
    assert is_add5 is False


def test_ambiguous_column_semantics_suppress_additive_shares():
    """Columns without explicit additive volume keywords must not guess additive shares."""
    s = pd.Series([100.0, 200.0, 150.0], index=["X", "Y", "Z"], name="metric_alpha")
    is_add, agg_type, limit = _is_additive_metric("metric_alpha", "compare metric_alpha", s)
    assert is_add is False
    assert "not demonstrably additive" in limit.lower()

    ev = calculate_answer_evidence(s, question="compare metric_alpha")
    assert ev.is_additive is False
    assert ev.top_share_pct is None


# ============================================================================
# 3. Negative Values Guard
# ============================================================================

def test_negative_values_suppress_shares_of_total(sample_sales_df):
    """
    When net figures contain negative values (e.g. losses), percentage of total is invalid.
    Must set is_additive=False and report signed sum limitation.
    """
    grouped_profit = sample_sales_df.groupby("Category")["Profit"].sum()
    # Furniture has negative profit: -70.0

    evidence = calculate_answer_evidence(
        result=grouped_profit,
        df=sample_sales_df,
        question="What is the total profit by category?"
    )

    assert evidence.is_additive is False
    assert evidence.aggregation_type == "signed_sum"
    assert evidence.top_share_pct is None
    assert "negative values" in evidence.limitation_notes.lower()
    assert evidence.top_contributor == "Technology"


# ============================================================================
# 4. Zero Totals and Zero Division Safety
# ============================================================================

def test_zero_total_and_runner_up_zero_division():
    """Zero sum or zero runner-up must not raise ZeroDivisionError."""
    # Zero sum series
    s_zero = pd.Series([0.0, 0.0, 0.0], index=["A", "B", "C"], name="Units")
    ev_zero = calculate_answer_evidence(s_zero, question="Units by group")
    assert ev_zero.is_additive is False
    assert ev_zero.top_share_pct is None
    assert "zero" in ev_zero.limitation_notes.lower()

    # Runner-up is 0 -> disparity_ratio should be None, disparity_difference computed
    s_runner_zero = pd.Series([100.0, 0.0], index=["A", "B"], name="Revenue")
    ev_rz = calculate_answer_evidence(s_runner_zero, question="Total revenue")
    assert ev_rz.disparity_ratio is None
    assert ev_rz.disparity_difference == 100.0


def test_tiny_denominator_disparity_ratio_guard():
    """Extremely small runner-up values (< 1e-4) must suppress runaway disparity ratios."""
    s_tiny = pd.Series([500.0, 0.00001], index=["Leader", "Trailer"], name="Revenue")
    ev = calculate_answer_evidence(s_tiny, question="Revenue comparison")
    assert ev.disparity_ratio is None
    assert ev.disparity_difference == 500.0


# ============================================================================
# 5. Non-Finite Values & Duplicate Indices
# ============================================================================

def test_non_finite_infinity_handled_safely():
    """Series containing inf or -inf must drop non-finite values and calculate cleanly."""
    s_inf = pd.Series([100.0, np.inf, 50.0, -np.inf, np.nan], index=["A", "B", "C", "D", "E"], name="Revenue")
    ev = calculate_answer_evidence(s_inf, question="Revenue sum")
    assert ev.top_contributor == "A"
    assert ev.top_value == 100.0
    assert ev.runner_up == "C"
    assert ev.runner_up_value == 50.0
    assert ev.num_groups == 2
    assert ev.total_value == 150.0


def test_duplicate_category_labels_handled_safely():
    """Series with duplicate index labels must group properly without reporting duplicate top/runner-up."""
    s_dup = pd.Series([100.0, 150.0, 50.0], index=["Alpha", "Alpha", "Beta"], name="Revenue")
    ev = calculate_answer_evidence(s_dup, question="Total revenue")
    assert ev.top_contributor == "Alpha"
    assert ev.top_value == 250.0
    assert ev.runner_up == "Beta"
    assert ev.runner_up_value == 50.0
    assert ev.num_groups == 2


def test_single_category_result_handling():
    """Single category results must report single_category status and suppress ratios/shares."""
    s_single = pd.Series([500.0], index=["OnlyOne"], name="Revenue")
    ev = calculate_answer_evidence(s_single, question="Revenue for OnlyOne")
    assert ev.aggregation_type == "single_category"
    assert ev.runner_up is None
    assert ev.disparity_ratio is None
    assert ev.top_share_pct is None

    exp = generate_deterministic_explanation(ev, "Revenue for OnlyOne", s_single)
    assert "Only one category (OnlyOne)" in exp.finding
    assert "comparative segment shares and runner-up disparities are not applicable" in exp.interpretation


# ============================================================================
# 6. Scalar Results Context & Compatible Baselines
# ============================================================================

def test_scalar_numeric_evidence_and_context(sample_sales_df):
    """Scalars extract dataset baseline context matching the question aggregation."""
    # 1. Total revenue sum across dataset: 3000.0
    total_rev = 3000.0
    ev_sum = calculate_answer_evidence(
        result=total_rev,
        df=sample_sales_df,
        question="What is the total revenue?"
    )

    assert ev_sum.aggregation_type == "sum"
    assert ev_sum.total_value == 3000.0
    assert ev_sum.sample_size == len(sample_sales_df)
    assert ev_sum.baseline_value == 3000.0
    assert "grand total" in ev_sum.baseline_description.lower()

    exp_sum = generate_deterministic_explanation(ev_sum, "What is the total revenue?", total_rev)
    assert "grand total across all records" in exp_sum.interpretation

    # 2. Average discount across dataset: 0.15
    avg_discount = 0.15
    ev_avg = calculate_answer_evidence(
        result=avg_discount,
        df=sample_sales_df,
        question="What is the average discount?"
    )

    assert ev_avg.aggregation_type == "average"
    assert ev_avg.baseline_value == 0.15
    assert "mean" in ev_avg.baseline_description.lower()

    # 3. Subset sum: 1700.0 (Technology revenue) compared to total revenue 3000.0
    ev_subset = calculate_answer_evidence(
        result=1700.0,
        df=sample_sales_df,
        question="What is the total revenue for Technology?"
    )
    assert ev_subset.aggregation_type == "sum"
    assert ev_subset.baseline_value == 3000.0
    exp_subset = generate_deterministic_explanation(ev_subset, "Total revenue for Technology", 1700.0)
    assert "56.7%" in exp_subset.interpretation


# ============================================================================
# 7. Empty and Null Result Handlers
# ============================================================================

def test_empty_dataframe_and_series():
    """Empty results handle gracefully with informative finding and next steps."""
    empty_df = pd.DataFrame()
    ev_df = calculate_answer_evidence(empty_df, question="Find orders with status='Unknown'")
    assert ev_df.aggregation_type == "empty"
    assert ev_df.sample_size == 0

    exp = generate_deterministic_explanation(ev_df, "Find orders with status='Unknown'", empty_df)
    assert "No records matched" in exp.finding
    assert len(exp.next_steps) >= 1

    # None result
    ev_none = calculate_answer_evidence(None)
    assert ev_none.aggregation_type == "empty"


# ============================================================================
# 8. Epistemic Modesty & Non-Causation Guarantees
# ============================================================================

def test_epistemic_modesty_in_explanations(sample_sales_df):
    """Explanations must include caveats and disclaim causal assertions."""
    grouped = sample_sales_df.groupby("Category")["Revenue"].sum()
    explanation = explain_answer(grouped, df=sample_sales_df, question="Revenue by category", use_gemini=False)

    assert explanation.uncertainty_or_limitations is not None
    assert "causality cannot be inferred" in explanation.uncertainty_or_limitations.lower()
    # Check that finding strictly states what happened
    assert "Technology" in explanation.finding


# ============================================================================
# 9. Gemini Fallback & Timeout Budget Tests
# ============================================================================

def test_gemini_fallback_on_exception(monkeypatch, sample_sales_df):
    """When Gemini raises an exception (or rate limits), seamlessly returns deterministic explanation."""
    import explanation as exp_module

    def mock_synthesize(evidence, question, timeout_sec=4.0):
        raise RuntimeError("Quota exceeded 429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr(exp_module, "_synthesize_with_gemini", mock_synthesize)

    grouped = sample_sales_df.groupby("Category")["Revenue"].sum()
    explanation = exp_module.explain_answer(
        result=grouped,
        df=sample_sales_df,
        question="Revenue by category",
        use_gemini=True
    )

    assert explanation is not None
    assert explanation.status == "deterministic"
    assert "Technology" in explanation.finding
    assert len(explanation.next_steps) >= 1


def test_gemini_fallback_on_malformed_json(monkeypatch, sample_sales_df):
    """When Gemini returns non-JSON or partial response, seamlessly falls back."""
    import explanation as exp_module

    def mock_synthesize(evidence, question, timeout_sec=4.0):
        return None  # simulates malformed or empty synthesis

    monkeypatch.setattr(exp_module, "_synthesize_with_gemini", mock_synthesize)

    grouped = sample_sales_df.groupby("Category")["Revenue"].sum()
    explanation = exp_module.explain_answer(
        result=grouped,
        df=sample_sales_df,
        question="Revenue by category",
        use_gemini=True
    )

    assert explanation.status == "deterministic"
    assert "Technology" in explanation.finding


def test_gemini_numerical_consistency_fallback(monkeypatch, sample_sales_df):
    """When Gemini returns a finding with an unsupported percentage, it falls back to deterministic finding."""
    import explanation as exp_module

    # Return an unsupported percentage (claims 92% when actual is 56.7%)
    def mock_synthesize(evidence, question, timeout_sec=4.0):
        return {
            "finding": "Technology dominates with 92% of all revenue.",
            "interpretation": "High concentration.",
            "uncertainty_or_limitations": "Observational data.",
            "next_steps": ["Check sub-categories"]
        }

    monkeypatch.setattr(exp_module, "_synthesize_with_gemini", mock_synthesize)

    grouped = sample_sales_df.groupby("Category")["Revenue"].sum()
    explanation = exp_module.explain_answer(
        result=grouped,
        df=sample_sales_df,
        question="Revenue by category",
        use_gemini=True
    )

    # Must fall back to the factual deterministic finding citing verified 56.7%
    assert "92%" not in explanation.finding
    assert "56.7%" in explanation.finding
    assert "causality cannot be inferred" in explanation.uncertainty_or_limitations.lower()


# ============================================================================
# 10. Security & Credential Redaction Tests
# ============================================================================

def test_error_sanitization_redacts_credentials():
    """Verify that credentials, bearer tokens, and API keys are redacted from error logs."""
    raw_key_err = "API error: key=AIzaSyA1234567890abcdef1234567890abcdef failed with 429"
    cleaned = _sanitize_error_message(raw_key_err)
    assert "AIzaSy" not in cleaned
    assert "[REDACTED_API_KEY]" in cleaned or "key=[REDACTED]" in cleaned

    bearer_err = "Unauthorized Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xyz token expired"
    cleaned_bearer = _sanitize_error_message(bearer_err)
    assert "Bearer [REDACTED_TOKEN]" in cleaned_bearer


# ============================================================================
# 11. API Backward Compatibility & Schema Verification
# ============================================================================

def test_api_analyze_returns_structured_explanation():
    """Verify that POST /api/analyze returns 200 with complete explanation model."""
    client = TestClient(app)

    # 1. Load sample dataset
    sample_res = client.post("/api/datasets/sample")
    assert sample_res.status_code == 200
    dataset_id = sample_res.json()["dataset_id"]

    # 2. Dispatch analytical query
    from agent import DataAnalystAgent
    import unittest.mock as mock

    # Mock code generation to return deterministically executed pandas query
    with mock.patch.object(
        DataAnalystAgent,
        "generate_code",
        return_value="result = df.groupby('category')['revenue'].sum()"
    ):
        res = client.post(
            "/api/analyze",
            json={"dataset_id": dataset_id, "question": "What is total revenue by category?"}
        )

    assert res.status_code == 200
    data = res.json()

    # Verify all legacy fields are preserved intact
    assert data["validation_status"] == "passed"
    assert data["execution_status"] == "passed"
    assert "formatted_result" in data
    assert "table_data" in data
    assert "visualization" in data
    assert "execution_time_ms" in data

    # Verify Feature A explanation field
    assert "explanation" in data
    explanation = data["explanation"]
    assert explanation is not None
    assert "finding" in explanation
    assert "evidence" in explanation
    assert "interpretation" in explanation
    assert "next_steps" in explanation
    assert len(explanation["next_steps"]) >= 1

    # Verify evidence fields
    evidence = explanation["evidence"]
    assert evidence["is_additive"] is True
    assert evidence["top_contributor"] is not None
    assert evidence["top_share_pct"] is not None
    assert evidence["top_share_pct"] > 0
