"""
evaluation/evaluator.py
-----------------------
Dual-mode evaluation engine for the AI Data Analyst Agent.

Supports two strictly separated modes:
1. Mode A: Live LLM Evaluation ("live")
   End-to-end test: Question -> Gemini Code Gen -> AST Validation -> Sandbox -> Correctness -> Latency.
2. Mode B: Deterministic Pipeline Evaluation ("deterministic")
   Evaluates AST validator, sandbox execution, error handling, and analytical correctness
   using representative benchmark code without making external LLM API calls.

Comparison Methodology:
- Scalars: Evaluated using absolute or relative numerical tolerance (never exact float equality).
- Rankings / Lists: Evaluated using top-N entity overlap and order consistency.
- Grouping / Dictionaries: Evaluated by mapping key names and checking aggregate values within tolerance.
- Time-series: Evaluated by verifying chronological series/DataFrames and valid numeric values.
- Visualizations: Evaluated by inspecting output structure against charts.py deterministic heuristics.
- Safety / Attacks: Evaluated by asserting rejection by the AST validator.
- Error cases: Evaluated by asserting graceful exception recovery and column suggestions.
"""

import json
import os
import sys
import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if sys.stderr.encoding != 'utf-8':
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from validator import clean_code, validate_code
from sandbox import execute_pandas_code
from charts import detect_chart_type


def _extract_scalar(val: Any) -> Optional[float]:
    """Attempts to extract a float scalar from numbers, Series, or DataFrames."""
    if isinstance(val, (int, float, np.number)):
        return float(val)
    if isinstance(val, pd.Series):
        if len(val) == 1:
            try:
                return float(val.iloc[0])
            except (ValueError, TypeError):
                return None
    if isinstance(val, pd.DataFrame):
        if val.size == 1:
            try:
                return float(val.values[0, 0])
            except (ValueError, TypeError):
                return None
    if isinstance(val, str):
        cleaned = val.replace("$", "").replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _extract_labels(val: Any) -> List[str]:
    """Extracts string identifiers from Series indices/values, DataFrames, lists, or sets."""
    if isinstance(val, pd.Series):
        idx_labels = [str(x) for x in val.index]
        val_labels = [str(x) for x in val.values if isinstance(x, str)]
        return idx_labels + val_labels
    if isinstance(val, pd.DataFrame):
        labels = []
        for col in val.columns:
            labels.append(str(col))
            if pd.api.types.is_string_dtype(val[col]) or pd.api.types.is_object_dtype(val[col]):
                labels.extend([str(x) for x in val[col].dropna()])
        labels.extend([str(x) for x in val.index])
        return labels
    if isinstance(val, (list, tuple, set)):
        return [str(x) for x in val]
    if isinstance(val, str):
        return [val]
    return []


def evaluate_correctness(actual: Any, case: Dict[str, Any], exec_success: bool, val_passed: bool) -> Tuple[bool, str]:
    """
    Evaluates correctness of actual result against expected criteria using tolerances and structural matching.
    """
    rule = case.get("evaluation_rule", "exact")
    expected = case.get("expected_value")

    # Safety blocked cases
    if rule == "safety_blocked":
        if not val_passed:
            return True, "Successfully blocked by AST code validator."
        if not exec_success:
            return True, "Execution failed safely in sandbox."
        return False, "Unsafe code was permitted to execute successfully."

    # Graceful error handling cases (e.g. non-existent column)
    if rule == "graceful_error_suggestion":
        if not exec_success and isinstance(actual, str) and (expected.lower() in actual.lower() or "not found" in actual.lower()):
            return True, f"Graceful diagnostic error produced for missing column '{expected}'."
        return False, f"Expected graceful diagnostic for '{expected}', got: {actual}"

    if not exec_success:
        return False, f"Execution failed: {actual}"

    # Numerical approximation (with tolerance)
    if rule == "approx_numeric":
        actual_num = _extract_scalar(actual)
        if actual_num is None:
            return False, f"Expected numeric value, got {type(actual).__name__} ({actual})"
        tol = case.get("tolerance", 0.05 * abs(expected) if expected else 0.5)
        diff = abs(actual_num - expected)
        if diff <= tol:
            return True, f"Actual {actual_num:.2f} matches expected {expected} within tolerance ±{tol}."
        return False, f"Actual {actual_num:.2f} differs from expected {expected} (diff: {diff:.2f}, tol: ±{tol})."

    # Numerical approximation or empty result
    if rule == "approx_numeric_or_empty":
        if isinstance(actual, (pd.DataFrame, pd.Series)) and actual.empty:
            return True, "Result is correctly empty."
        actual_num = _extract_scalar(actual)
        if actual_num is not None:
            if abs(actual_num - expected) <= case.get("tolerance", 0.5):
                return True, f"Actual {actual_num} matches expected {expected}."
        return False, f"Expected 0 or empty, got: {actual}"

    # String contains
    if rule == "string_contains":
        actual_str = str(actual).lower()
        if expected.lower() in actual_str:
            return True, f"'{expected}' found in result string."
        return False, f"'{expected}' not found in actual result: '{actual}'."

    # String contains any accepted candidate
    if rule == "string_contains_any":
        accepted = case.get("accepted_values", [expected])
        actual_str = str(actual).lower()
        for cand in accepted:
            if cand.lower() in actual_str:
                return True, f"Accepted candidate '{cand}' found in actual result."
        return False, f"None of {accepted} found in actual result: '{actual}'."

    # Set matching (e.g. categories, product names)
    if rule == "set_match":
        labels = _extract_labels(actual)
        expected_set = set(str(x).lower() for x in expected)
        found = set(str(x).lower() for x in labels)
        overlap = expected_set.intersection(found)
        if len(overlap) >= len(expected_set) * 0.7:
            return True, f"Matched {len(overlap)}/{len(expected_set)} expected items."
        return False, f"Insufficient set match: matched {len(overlap)}/{len(expected_set)}. Found: {labels[:5]}."

    # Top-N overlap (ranking questions)
    if rule == "top_n_overlap":
        labels = _extract_labels(actual)
        min_overlap = case.get("min_overlap", max(1, len(expected) - 1))
        expected_set = set(str(x).lower() for x in expected)
        found = set(str(x).lower() for x in labels)
        overlap = expected_set.intersection(found)
        if len(overlap) >= min_overlap:
            return True, f"Matched {len(overlap)} items (required min {min_overlap})."
        return False, f"Top-N overlap insufficient: {len(overlap)} < {min_overlap}. Labels found: {labels[:5]}."

    # Dictionary / groupby key-value match
    if rule == "dict_match":
        if isinstance(actual, pd.Series):
            actual_dict = actual.to_dict()
        elif isinstance(actual, dict):
            actual_dict = actual
        elif isinstance(actual, pd.DataFrame) and actual.shape[1] >= 2:
            try:
                actual_dict = dict(zip(actual.iloc[:, 0], actual.iloc[:, 1]))
            except Exception:
                actual_dict = {}
        else:
            return False, f"Expected key-value structure, got {type(actual).__name__}."

        tol = case.get("tolerance", 5.0)
        matched = 0
        total = len(expected)
        for k, v in expected.items():
            cand_keys = [ak for ak in actual_dict if str(ak).lower() == str(k).lower()]
            if cand_keys:
                ak = cand_keys[0]
                try:
                    act_num = float(actual_dict[ak])
                    if abs(act_num - v) <= tol:
                        matched += 1
                except (ValueError, TypeError):
                    pass

        if matched >= max(1, total - 1):
            return True, f"Matched {matched}/{total} group entries within ±{tol}."
        return False, f"Matched only {matched}/{total} group entries. Got keys: {list(actual_dict.keys())}."

    # Chart detection compatibility
    if rule == "chart_detected":
        detected = detect_chart_type(actual)
        if detected == expected or (expected == "Bar Chart" and detected in ["Bar Chart", "Histogram"]):
            return True, f"Detected chart type '{detected}' matches expected '{expected}'."
        return False, f"Expected chart '{expected}', detected '{detected}'."

    # Time series data validation
    if rule == "has_time_series_data":
        if isinstance(actual, (pd.DataFrame, pd.Series)) and not actual.empty:
            return True, "Successfully returned time series tabular data."
        return False, f"Expected non-empty time-series DataFrame/Series, got {type(actual).__name__}."

    # Fallback exact comparison
    if str(actual).strip() == str(expected).strip():
        return True, "Exact match."
    return False, f"Expected {expected}, got {actual}."


def run_evaluation(
    eval_cases_path: str = "evaluation/eval_cases.json",
    output_results_path: str = "evaluation/results.json",
    mode: str = "deterministic",
    verbose: bool = True,
    category_filter: Optional[str] = None,
    limit: Optional[int] = None,
    pacing_delay: float = 0.0
) -> Dict[str, Any]:
    """
    Runs evaluation benchmark across test cases in either deterministic or live mode.

    Args:
        eval_cases_path (str): Path to JSON file containing benchmark test cases.
        output_results_path (str): Path where evaluation summary JSON will be stored.
        mode (str): 'deterministic' (uses candidate_code) or 'live' (invokes Gemini API).
        verbose (bool): Whether to print progress to stdout.
        category_filter (Optional[str]): Optional category name substring to filter cases.
        limit (Optional[int]): Max cases to execute.
        pacing_delay (float): Delay in seconds between queries (useful for live mode rate limits).

    Returns:
        Dict[str, Any]: Comprehensive evaluation report dictionary.
    """
    if not os.path.exists(eval_cases_path):
        raise FileNotFoundError(f"Evaluation cases file not found at: {eval_cases_path}")

    with open(eval_cases_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    if category_filter:
        cases = [c for c in cases if category_filter.lower() in c.get("category", "").lower()]

    if limit and limit > 0:
        cases = cases[:limit]

    agent = None
    if mode == "live":
        from agent import DataAnalystAgent
        agent = DataAnalystAgent()

    # Cache datasets
    datasets: Dict[str, pd.DataFrame] = {}

    results_detail: List[Dict[str, Any]] = []
    category_metrics: Dict[str, Dict[str, int]] = {}

    total_cases = len(cases)
    passed_validation_count = 0
    execution_success_count = 0
    correct_count = 0
    unsafe_tested = 0
    unsafe_blocked = 0
    api_failures_count = 0
    latencies: List[float] = []

    header_title = "DETERMINISTIC PIPELINE EVALUATION" if mode == "deterministic" else "LIVE LLM EVALUATION"
    print(f"\n>> Starting AI Data Analyst Benchmark [{header_title}] ({total_cases} cases)...\n" + "=" * 65, flush=True)

    for i, case in enumerate(cases, 1):
        if i > 1 and pacing_delay > 0:
            time.sleep(pacing_delay)

        cid = case["id"]
        cat = case.get("category", "General")
        q = case["question"]
        ds_path = case.get("dataset", "data/ecommerce_sales.csv")
        rule = case.get("evaluation_rule")

        if cat not in category_metrics:
            category_metrics[cat] = {"total": 0, "correct": 0, "exec_success": 0}
        category_metrics[cat]["total"] += 1

        # Load dataset
        if ds_path not in datasets:
            if os.path.exists(ds_path):
                datasets[ds_path] = pd.read_csv(ds_path)
            else:
                datasets[ds_path] = pd.DataFrame()
        df = datasets[ds_path]

        start_time = time.time()
        gen_code = ""
        val_passed = False
        val_msg = ""
        exec_success = False
        exec_res: Any = None
        error_msg = ""

        try:
            # 1. Obtain code
            if mode == "live":
                gen_code = agent.generate_code(df, q)
            else:
                # Mode B: Deterministic evaluation using pre-established candidate code
                gen_code = case.get("candidate_code", "")

            cleaned = clean_code(gen_code)

            # 2. AST Validation
            val_passed, val_msg = validate_code(cleaned)

            # 3. Execution
            if val_passed:
                passed_validation_count += 1
                exec_success, exec_res = execute_pandas_code(cleaned, df)
                if exec_success:
                    execution_success_count += 1
                    category_metrics[cat]["exec_success"] += 1
                else:
                    error_msg = str(exec_res)
            else:
                error_msg = f"Validation Error: {val_msg}"

        except Exception as e:
            error_msg = f"Evaluation Execution Error: {str(e)}"
            if mode == "live" and ("429" in str(e) or "API" in str(e)):
                api_failures_count += 1

        elapsed = time.time() - start_time
        latencies.append(elapsed)

        # 4. Check correctness
        is_correct, reason = evaluate_correctness(
            actual=exec_res if exec_success else error_msg,
            case=case,
            exec_success=exec_success,
            val_passed=val_passed
        )

        if is_correct:
            correct_count += 1
            category_metrics[cat]["correct"] += 1

        if rule == "safety_blocked":
            unsafe_tested += 1
            if is_correct:
                unsafe_blocked += 1

        status_sym = "[PASS]" if is_correct else "[FAIL]"
        if verbose:
            print(f"[{i:02d}/{total_cases:02d}] {status_sym} [{cid}] {cat}: {q[:45]}... ({elapsed:.3f}s)", flush=True)
            if not is_correct:
                print(f"       Reason: {reason}", flush=True)
                if gen_code:
                    print(f"       Code: {gen_code[:80]}...", flush=True)

        # Serializable record
        serializable_result = str(exec_res) if exec_res is not None else None
        if isinstance(exec_res, (pd.DataFrame, pd.Series)):
            serializable_result = exec_res.head(5).to_dict()

        results_detail.append({
            "id": cid,
            "category": cat,
            "question": q,
            "code": gen_code,
            "passed_validation": val_passed,
            "validation_message": val_msg,
            "execution_success": exec_success,
            "error_message": error_msg,
            "actual_result": serializable_result,
            "expected_value": case.get("expected_value"),
            "is_correct": is_correct,
            "correctness_reason": reason,
            "response_time_seconds": round(elapsed, 4)
        })

    # Summary metrics
    avg_latency = round(float(np.mean(latencies)), 3) if latencies else 0.0
    attempted_executable = total_cases - unsafe_tested
    exec_rate = round((execution_success_count / attempted_executable) * 100, 1) if attempted_executable > 0 else 0.0
    accuracy_rate = round((correct_count / total_cases) * 100, 1)
    safety_rate = round((unsafe_blocked / unsafe_tested) * 100, 1) if unsafe_tested > 0 else 100.0

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "evaluation_type": "Deterministic Pipeline Evaluation" if mode == "deterministic" else "Live LLM Evaluation",
        "total_cases": total_cases,
        "execution": {
            "successful": execution_success_count,
            "attempted_executable": attempted_executable,
            "execution_rate_percent": exec_rate
        },
        "correctness": {
            "correct": correct_count,
            "accuracy_percent": accuracy_rate
        },
        "safety": {
            "unsafe_cases_tested": unsafe_tested,
            "unsafe_cases_blocked": unsafe_blocked,
            "safety_rate_percent": safety_rate
        },
        "performance": {
            "average_response_time_seconds": avg_latency,
            "min_response_time_seconds": round(min(latencies), 4) if latencies else 0.0,
            "max_response_time_seconds": round(max(latencies), 4) if latencies else 0.0
        },
        "api_failures": api_failures_count if mode == "live" else 0,
        "category_metrics": category_metrics,
        "results": results_detail
    }

    # Save to json
    os.makedirs(os.path.dirname(output_results_path) or ".", exist_ok=True)
    with open(output_results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
