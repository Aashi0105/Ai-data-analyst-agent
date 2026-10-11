"""
evaluate.py
-----------
Command-line evaluation runner for the AI Data Analyst Agent.
Supports two distinct benchmarking modes:

1. Deterministic Pipeline Evaluation (--mode deterministic):
   Evaluates the AST validator, sandbox execution, error handling,
   and analytical correctness against all 43 benchmark queries using
   predetermined candidate code without calling the external LLM API.

2. Live LLM Evaluation (--mode live):
   Runs real end-to-end questions through Google Gemini, AST validation,
   sandbox execution, and auto-charting.

Usage:
    python evaluate.py --mode deterministic
    python evaluate.py --mode live [--limit N] [--pacing SECONDS]
"""

import sys
import os
import argparse

# Ensure UTF-8 output encoding on Windows consoles
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

from evaluation.evaluator import run_evaluation


def print_deterministic_report(summary: dict) -> None:
    """Prints formatted report for deterministic pipeline evaluation."""
    total = summary["total_cases"]
    exec_data = summary["execution"]
    corr_data = summary["correctness"]
    safety_data = summary["safety"]
    perf_data = summary["performance"]
    categories = summary["category_metrics"]
    results = summary["results"]

    print("\n" + "=" * 70)
    print("                      AI DATA ANALYST")
    print("           DETERMINISTIC PIPELINE EVALUATION REPORT")
    print("=" * 70)
    print(f"Timestamp: {summary['timestamp']}")
    print(f"Mode: Deterministic Pipeline Evaluation (Validator + Sandbox + Matcher)")
    print(f"Total Benchmark Cases: {total}")
    print("-" * 70)

    print("\n## [1] Execution & Pipeline Reliability")
    print(f"  Attempted Executable: {exec_data['attempted_executable']}")
    print(f"  Execution Succeeded: {exec_data['successful']}/{exec_data['attempted_executable']}")
    print(f"  Execution Rate:      {exec_data['execution_rate_percent']}%")

    print("\n## [2] Analytical Correctness")
    print(f"  Correct Cases:       {corr_data['correct']}/{total}")
    print(f"  Accuracy:            {corr_data['accuracy_percent']}%")

    print("\n## [3] Security & AST Safety Invariants")
    print(f"  Unsafe cases tested: {safety_data['unsafe_cases_tested']}")
    print(f"  Blocked correctly:   {safety_data['unsafe_cases_blocked']}/{safety_data['unsafe_cases_tested']}")
    print(f"  Safety Rejection:    {safety_data['safety_rate_percent']}%")

    print("\n## [4] Execution Latency")
    print(f"  Average pipeline:    {perf_data['average_response_time_seconds']} seconds")
    print(f"  Min / Max pipeline:  {perf_data['min_response_time_seconds']}s / {perf_data['max_response_time_seconds']}s")

    print("\n## [5] Category Breakdown")
    print(f"  {'Category':<24} | {'Total':<6} | {'Correct':<8} | {'Accuracy'}")
    print("  " + "-" * 56)
    for cat_name, metrics in categories.items():
        c_tot = metrics["total"]
        c_cor = metrics["correct"]
        c_pct = round((c_cor / c_tot) * 100, 1) if c_tot > 0 else 0.0
        print(f"  {cat_name:<24} | {c_tot:<6} | {c_cor:<8} | {c_pct}%")

    failures = [r for r in results if not r["is_correct"]]
    print(f"\n## [6] Failures & Discrepancies ({len(failures)} total)")
    if not failures:
        print("  None! All cases passed evaluation criteria.")
    else:
        for f in failures:
            print(f"\n  * [{f['id']}] {f['category']}: \"{f['question']}\"")
            print(f"    Reason:  {f['correctness_reason']}")
            if f.get("code"):
                print(f"    Code:    {f['code'].strip()}")
            if f.get("error_message"):
                print(f"    Error:   {f['error_message']}")

    print("\n" + "=" * 70)
    print("Detailed JSON log saved to: evaluation/results.json\n")


def print_live_report(summary: dict) -> None:
    """Prints formatted report for live LLM evaluation."""
    total = summary["total_cases"]
    exec_data = summary["execution"]
    corr_data = summary["correctness"]
    perf_data = summary["performance"]
    categories = summary["category_metrics"]
    results = summary["results"]
    api_failures = summary.get("api_failures", 0)

    print("\n" + "=" * 70)
    print("                      AI DATA ANALYST")
    print("                    LIVE LLM EVALUATION")
    print("=" * 70)
    print(f"Timestamp: {summary['timestamp']}")
    print(f"Mode: Live End-to-End Evaluation (Gemini 2.5 Flash -> AST -> Sandbox)")
    print(f"Total Attempted: {total}")
    print("-" * 70)

    print("\n## [1] LLM Generation & Pipeline Reliability")
    print(f"  Total Attempted:     {total}")
    print(f"  Execution Succeeded: {exec_data['successful']}/{exec_data['attempted_executable']}")
    print(f"  Execution Rate:      {exec_data['execution_rate_percent']}%")
    print(f"  API / Quota Errors:  {api_failures}")

    print("\n## [2] Analytical Correctness")
    print(f"  Correct:             {corr_data['correct']}/{total}")
    print(f"  Accuracy:            {corr_data['accuracy_percent']}%")

    print("\n## [3] Response Latency")
    print(f"  Average response:    {perf_data['average_response_time_seconds']} seconds")
    print(f"  Min / Max response:  {perf_data['min_response_time_seconds']}s / {perf_data['max_response_time_seconds']}s")

    print("\n## [4] Category Breakdown")
    print(f"  {'Category':<24} | {'Total':<6} | {'Correct':<8} | {'Accuracy'}")
    print("  " + "-" * 56)
    for cat_name, metrics in categories.items():
        c_tot = metrics["total"]
        c_cor = metrics["correct"]
        c_pct = round((c_cor / c_tot) * 100, 1) if c_tot > 0 else 0.0
        print(f"  {cat_name:<24} | {c_tot:<6} | {c_cor:<8} | {c_pct}%")

    failures = [r for r in results if not r["is_correct"]]
    print(f"\n## [5] Failures & API Errors ({len(failures)} total)")
    if not failures:
        print("  None! All live cases passed evaluation criteria.")
    else:
        for f in failures:
            print(f"\n  * [{f['id']}] {f['category']}: \"{f['question']}\"")
            print(f"    Reason:  {f['correctness_reason']}")
            if f.get("code"):
                print(f"    Code:    {f['code'].strip()}")
            if f.get("error_message"):
                print(f"    Error:   {f['error_message']}")

    print("\n" + "=" * 70)
    print("Detailed JSON log saved to: evaluation/results.json\n")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run AI Data Analyst Agent evaluation benchmark and generate audit report."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["deterministic", "live"],
        default="deterministic",
        help="Evaluation mode: 'deterministic' (no LLM calls) or 'live' (calls Gemini API). Default: deterministic."
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter evaluation to a specific category (e.g. 'Basic', 'Filtering', 'Grouping', 'Security')."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of evaluation test cases to run."
    )
    parser.add_argument(
        "--pacing",
        type=float,
        default=0.0,
        help="Pacing delay in seconds between consecutive queries."
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="evaluation/eval_cases.json",
        help="Path to evaluation test cases JSON file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/results.json",
        help="Path to save detailed benchmark results JSON."
    )

    args = parser.parse_args()

    try:
        summary = run_evaluation(
            eval_cases_path=args.cases,
            output_results_path=args.output,
            mode=args.mode,
            verbose=True,
            category_filter=args.category,
            limit=args.limit,
            pacing_delay=args.pacing
        )

        if args.mode == "deterministic":
            print_deterministic_report(summary)
        else:
            print_live_report(summary)

    except Exception as e:
        print(f"\n[ERROR] Evaluation aborted due to error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
