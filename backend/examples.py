"""
backend/examples.py
-------------------
Pre-calculated saved analytical examples from the benchmark e-commerce dataset (data/ecommerce_sales.csv).
Provides zero-latency, quota-free fallback demonstrations for review when API limits are reached.
"""

import os
from typing import Any, Dict, List, Optional
import pandas as pd
from backend.chart_utils import build_recharts_data

_CACHED_EXAMPLES: Optional[List[Dict[str, Any]]] = None


def _calculate_examples() -> List[Dict[str, Any]]:
    """Calculates 5 structured analytical examples using pure Pandas on the authentic dataset."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "data", "ecommerce_sales.csv")

    if not os.path.exists(csv_path):
        # Fallback to current working directory
        csv_path = os.path.join(os.getcwd(), "data", "ecommerce_sales.csv")

    df = pd.read_csv(csv_path)

    # Q1: Which region has the highest total revenue?
    q1_q = "Which region has the highest total revenue?"
    q1_res = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    q1_top_region = str(q1_res.index[0])
    q1_top_rev = float(q1_res.iloc[0])
    q1_total_rev = float(q1_res.sum())
    q1_pct = (q1_top_rev / q1_total_rev) * 100
    q1_answer = f"{q1_top_region} region leads with ₹{q1_top_rev:,.2f} total revenue, contributing {q1_pct:.1f}% of overall revenue."
    q1_chart = build_recharts_data(q1_res, question=q1_q) or {
        "chart_type": "Bar Chart",
        "data": [{"label": str(k), "value": round(float(v), 2)} for k, v in q1_res.items()],
        "x_key": "label",
        "y_keys": ["value"],
        "title": "revenue by region",
    }

    # Q2: What is the top-selling product category?
    q2_q = "What is the top-selling product category?"
    q2_res = df.groupby("category")["quantity"].sum().sort_values(ascending=False)
    q2_top_cat = str(q2_res.index[0])
    q2_top_qty = int(q2_res.iloc[0])
    q2_answer = f"{q2_top_cat} is the top-selling category with {q2_top_qty} units sold."
    q2_chart = build_recharts_data(q2_res, question=q2_q) or {
        "chart_type": "Bar Chart",
        "data": [{"label": str(k), "value": int(v)} for k, v in q2_res.items()],
        "x_key": "label",
        "y_keys": ["value"],
        "title": "quantity by category",
    }

    # Q3: Which region has the highest profit margin?
    q3_q = "Which region has the highest profit margin?"
    q3_res = (df.groupby("region")["profit"].sum() / df.groupby("region")["revenue"].sum() * 100).sort_values(ascending=False)
    q3_top_region = str(q3_res.index[0])
    q3_margin = float(q3_res.iloc[0])
    q3_answer = f"{q3_top_region} region has the highest profit margin at {q3_margin:.2f}%."
    q3_chart = build_recharts_data(q3_res, question=q3_q) or {
        "chart_type": "Bar Chart",
        "data": [{"label": str(k), "value": round(float(v), 2)} for k, v in q3_res.items()],
        "x_key": "label",
        "y_keys": ["value"],
        "title": "profit margin (%) by region",
    }

    # Q4: What is the revenue trend across regions?
    q4_q = "What is the revenue trend across regions?"
    q4_res = df.groupby("region")["revenue"].sum()
    parts = [f"{reg}: ₹{float(val):,.2f}" for reg, val in q4_res.items()]
    q4_answer = f"Total revenue across regions: {', '.join(parts)}."
    q4_chart = build_recharts_data(q4_res, question=q4_q) or {
        "chart_type": "Bar Chart",
        "data": [{"label": str(k), "value": round(float(v), 2)} for k, v in q4_res.items()],
        "x_key": "label",
        "y_keys": ["value"],
        "title": "revenue by region",
    }

    # Q5: Which product has the highest profit?
    q5_q = "Which product has the highest profit?"
    q5_res = df.groupby("product_name")["profit"].sum().sort_values(ascending=False).head(5)
    q5_top_prod = str(q5_res.index[0])
    q5_top_profit = float(q5_res.iloc[0])
    q5_answer = f"{q5_top_prod} is the most profitable product with ₹{q5_top_profit:,.2f} profit."
    q5_chart = build_recharts_data(q5_res, question=q5_q) or {
        "chart_type": "Bar Chart",
        "data": [{"label": str(k), "value": round(float(v), 2)} for k, v in q5_res.items()],
        "x_key": "label",
        "y_keys": ["value"],
        "title": "profit by product_name",
    }

    return [
        {
            "question": q1_q,
            "answer_text": q1_answer,
            "chart_type": "bar",
            "chart_data": q1_chart,
            "label": "Saved Example — E-commerce Sales Dataset",
            "is_example": True,
        },
        {
            "question": q2_q,
            "answer_text": q2_answer,
            "chart_type": "bar",
            "chart_data": q2_chart,
            "label": "Saved Example — E-commerce Sales Dataset",
            "is_example": True,
        },
        {
            "question": q3_q,
            "answer_text": q3_answer,
            "chart_type": "bar",
            "chart_data": q3_chart,
            "label": "Saved Example — E-commerce Sales Dataset",
            "is_example": True,
        },
        {
            "question": q4_q,
            "answer_text": q4_answer,
            "chart_type": "bar",
            "chart_data": q4_chart,
            "label": "Saved Example — E-commerce Sales Dataset",
            "is_example": True,
        },
        {
            "question": q5_q,
            "answer_text": q5_answer,
            "chart_type": "bar",
            "chart_data": q5_chart,
            "label": "Saved Example — E-commerce Sales Dataset",
            "is_example": True,
        },
    ]


def get_examples() -> List[Dict[str, Any]]:
    """
    Returns 5 pre-calculated analytical examples.
    Guaranteed:
    - Never invokes Gemini or any network services.
    - Cached in-memory: returns in under 1ms with zero repeated disk I/O.
    """
    global _CACHED_EXAMPLES
    if _CACHED_EXAMPLES is None:
        _CACHED_EXAMPLES = _calculate_examples()
    return _CACHED_EXAMPLES
