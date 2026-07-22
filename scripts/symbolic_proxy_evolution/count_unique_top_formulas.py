#!/usr/bin/env python3
"""Count unique top-k LaTeX formulas across one dataset's seeded EA runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from summarize_formula_duplicates import extract_formula_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count normalized formulas in archive_latex.tex for a contiguous seed range. "
            "Exactly one named run directory is inspected per seed."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True, help="Run directory name, including the run_ prefix.")
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-end", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when a seed is missing archive_latex.tex or does not contain exactly top-k rows.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.seed_end < args.seed_start:
        raise ValueError("--seed-end must be >= --seed-start")
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    if "/" in args.run_name or not args.run_name.startswith("run_"):
        raise ValueError("--run-name must be a directory name beginning with 'run_'")

    formulas: set[str] = set()
    archive_file_count = 0
    total_formula_rows = 0
    warnings: list[str] = []

    for seed in range(args.seed_start, args.seed_end + 1):
        latex_path = (
            args.dataset_root
            / f"seed_{seed}"
            / args.run_name
            / "visualizations"
            / "archive_latex.tex"
        )
        if not latex_path.is_file():
            warnings.append(f"missing: {latex_path}")
            continue

        rows = extract_formula_rows(latex_path, args.top_k)
        if len(rows) != args.top_k:
            warnings.append(
                f"expected {args.top_k} ranked formulas, found {len(rows)}: {latex_path}"
            )
        archive_file_count += 1
        total_formula_rows += len(rows)
        formulas.update(formula for _rank, formula in rows)

    if args.strict and warnings:
        preview = "\n".join(warnings[:20])
        suffix = f"\n... and {len(warnings) - 20} more" if len(warnings) > 20 else ""
        raise RuntimeError(f"Incomplete formula archives:\n{preview}{suffix}")

    # A compact, stable format that Bash can parse without an extra dependency.
    print(archive_file_count, total_formula_rows, len(formulas))
    for warning in warnings:
        print(f"WARNING: {warning}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
