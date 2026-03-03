"""
Evaluation harness for AgenOptimic against the OptiBench benchmark dataset.

Usage:
    python -m tests.evaluate [--limit N] [--type TYPE] [--concurrency N]

Examples:
    python -m tests.evaluate --limit 10
    python -m tests.evaluate --type linear-notable --limit 20
    python -m tests.evaluate --concurrency 4
"""

import argparse
import asyncio
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

DATA_PATH = Path(__file__).parent / "data" / "OptiBench.json"
TOLERANCE = 1e-2  # relative tolerance for numeric comparison


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_numbers_from_text(text: str) -> list[float]:
    """Extract all floating-point numbers from a string."""
    pattern = r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?"
    return [float(m.replace(",", "")) for m in re.findall(pattern, text)]


def _is_close(a: float, b: float, tol: float = TOLERANCE) -> bool:
    if abs(b) < 1e-9:
        return abs(a) < 1e-9
    return abs(a - b) / abs(b) <= tol


def _check_result(solution: Optional[str], expected: dict[str, str]) -> tuple[int, int]:
    """
    Returns (matched, total) for a single problem.
    A key is matched if any number in the solution line is within tolerance
    of the expected numeric value.
    """
    if not solution:
        return 0, len(expected)

    solution_lower = solution.lower()
    matched = 0

    for key, val_str in expected.items():
        try:
            expected_val = float(val_str)
        except ValueError:
            continue

        key_lower = key.lower()
        # Look for lines containing keywords from the expected key
        key_words = [w for w in re.split(r"\W+", key_lower) if len(w) > 2]
        candidate_lines = [
            line for line in solution_lower.splitlines()
            if any(w in line for w in key_words)
        ]
        if not candidate_lines:
            # Fall back: search entire output
            candidate_lines = solution_lower.splitlines()

        found = False
        for line in candidate_lines:
            for num in _extract_numbers_from_text(line):
                if _is_close(num, expected_val):
                    found = True
                    break
            if found:
                break

        if found:
            matched += 1

    return matched, len(expected)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

async def _run_problem(problem: dict[str, Any], semaphore: asyncio.Semaphore) -> dict[str, Any]:
    from app.graph.graph import compiled_graph
    from app.graph.states import AgenOptimicState

    idx = problem["index"]
    question = problem["question"]
    expected = problem["results"]

    async with semaphore:
        print(f"  [{idx}] Running...", flush=True)
        try:
            initial = AgenOptimicState(question=question)
            result: AgenOptimicState = await compiled_graph.ainvoke(initial)
            solution = result.solution
            error = result.error
            retry_count = result.retry_count
        except Exception as exc:
            solution = None
            error = str(exc)
            retry_count = 0

    matched, total = _check_result(solution, expected)
    status = "OK" if matched == total else f"PARTIAL {matched}/{total}" if matched > 0 else "FAIL"
    print(f"  [{idx}] {status} — retries={retry_count}", flush=True)

    return {
        "index": idx,
        "type": problem["type"],
        "matched": matched,
        "total": total,
        "solution": solution,
        "error": error,
        "retry_count": retry_count,
    }


async def run_evaluation(
    limit: Optional[int] = None,
    problem_type: Optional[str] = None,
    concurrency: int = 2,
) -> None:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        dataset: list[dict] = json.load(f)

    if problem_type:
        dataset = [p for p in dataset if p["type"] == problem_type]

    if limit:
        dataset = dataset[:limit]

    total_problems = len(dataset)
    print(f"\nRunning evaluation on {total_problems} problems (concurrency={concurrency})...\n")

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [_run_problem(p, semaphore) for p in dataset]
    results = await asyncio.gather(*tasks)

    # --- Aggregate ---
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    overall_matched = 0
    overall_total = 0

    for ptype, items in sorted(by_type.items()):
        type_matched = sum(r["matched"] for r in items)
        type_total = sum(r["total"] for r in items)
        problems_solved = sum(1 for r in items if r["matched"] == r["total"])
        acc = type_matched / type_total * 100 if type_total > 0 else 0.0
        problem_acc = problems_solved / len(items) * 100 if items else 0.0
        print(
            f"  {ptype:<25} key-acc={acc:5.1f}%  problem-acc={problem_acc:5.1f}%"
            f"  ({problems_solved}/{len(items)} problems fully solved)"
        )
        overall_matched += type_matched
        overall_total += type_total

    print("-" * 60)
    overall_key_acc = overall_matched / overall_total * 100 if overall_total > 0 else 0.0
    overall_prob_acc = sum(1 for r in results if r["matched"] == r["total"]) / total_problems * 100
    print(f"  {'OVERALL':<25} key-acc={overall_key_acc:5.1f}%  problem-acc={overall_prob_acc:5.1f}%")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AgenOptimic on OptiBench.")
    parser.add_argument("--limit", type=int, default=None, help="Max number of problems to run.")
    parser.add_argument("--type", type=str, default=None, dest="problem_type",
                        help="Filter by problem type (e.g. linear-notable).")
    parser.add_argument("--concurrency", type=int, default=2,
                        help="Number of problems to solve concurrently.")
    args = parser.parse_args()

    asyncio.run(run_evaluation(
        limit=args.limit,
        problem_type=args.problem_type,
        concurrency=args.concurrency,
    ))


if __name__ == "__main__":
    main()
