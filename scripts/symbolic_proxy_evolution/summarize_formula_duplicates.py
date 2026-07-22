#!/usr/bin/env python3
"""Summarize duplicate top-k LaTeX formulas in symbolic proxy archives."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


ROW_RE = re.compile(r"^\s*(\d+)\s*&")
MATH_RE = re.compile(r"\$(.*?)\$")
SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class FormulaOccurrence:
    seed: str
    run: str
    rank: int
    path: Path


def normalize_formula(raw_formula: str) -> str:
    return SPACE_RE.sub(" ", raw_formula.strip())


def extract_formula_rows(path: Path, top_k: int) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row_match = ROW_RE.match(line)
        if not row_match:
            continue

        rank = int(row_match.group(1))
        if rank > top_k:
            continue

        formula_cell = line.rsplit("&", 1)[-1].strip()
        if formula_cell.endswith(r"\\"):
            formula_cell = formula_cell[:-2].strip()

        math_match = MATH_RE.search(formula_cell)
        formula = math_match.group(1) if math_match else formula_cell
        rows.append((rank, normalize_formula(formula)))
    return rows


def summarize(
    archive_root: Path, top_k: int, seed_start: int, seed_end: int
) -> dict[str, dict[str, object]]:
    by_dataset: dict[str, dict[str, object]] = {}
    archive_files = sorted(archive_root.glob("*/seed_*/run_*/visualizations/archive_latex.tex"))

    for path in archive_files:
        rel = path.relative_to(archive_root)
        dataset, seed, run = rel.parts[:3]
        seed_number = int(seed.removeprefix("seed_"))
        if not seed_start <= seed_number <= seed_end:
            continue

        dataset_summary = by_dataset.setdefault(
            dataset,
            {
                "files": set(),
                "seeds": set(),
                "formula_counts": Counter(),
                "occurrences": defaultdict(list),
                "warnings": [],
            },
        )
        dataset_summary["files"].add(path)
        dataset_summary["seeds"].add(seed_number)

        rows = extract_formula_rows(path, top_k)
        if len(rows) != top_k:
            dataset_summary["warnings"].append(
                f"{path}: extracted {len(rows)} formula rows, expected {top_k}"
            )

        for rank, formula in rows:
            dataset_summary["formula_counts"][formula] += 1
            dataset_summary["occurrences"][formula].append(
                FormulaOccurrence(seed=seed, run=run, rank=rank, path=path)
            )

    return by_dataset


def duplicate_metrics(formula_counts: Counter[str]) -> dict[str, int]:
    total_formula_rows = sum(formula_counts.values())
    duplicate_counts = [count for count in formula_counts.values() if count > 1]
    return {
        "total_formula_rows": total_formula_rows,
        "unique_formula_count": len(formula_counts),
        "duplicate_formula_count": len(duplicate_counts),
        "duplicate_occurrence_count": sum(duplicate_counts),
        "redundant_duplicate_count": total_formula_rows - len(formula_counts),
    }


def occurrence_refs(occurrences: list[FormulaOccurrence]) -> str:
    sorted_occurrences = sorted(occurrences, key=lambda item: (item.seed, item.run, item.rank))
    return ";".join(f"{item.seed}/{item.run}/rank_{item.rank}" for item in sorted_occurrences)


def write_csv(summary: dict[str, dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "dataset",
        "total_formula_rows",
        "unique_formula_count",
        "duplicate_formula_count",
        "duplicate_occurrence_count",
        "redundant_duplicate_count",
        "formula_frequency",
        "formula",
        "occurrences",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for dataset in sorted(summary):
            formula_counts: Counter[str] = summary[dataset]["formula_counts"]
            occurrences: dict[str, list[FormulaOccurrence]] = summary[dataset]["occurrences"]
            duplicate_formulas = {
                formula: count for formula, count in formula_counts.items() if count > 1
            }
            metrics = duplicate_metrics(formula_counts)

            for formula, count in sorted(
                duplicate_formulas.items(), key=lambda item: (-item[1], item[0])
            ):
                writer.writerow(
                    {
                        "dataset": dataset,
                        "total_formula_rows": metrics["total_formula_rows"],
                        "unique_formula_count": metrics["unique_formula_count"],
                        "duplicate_formula_count": len(duplicate_formulas),
                        "duplicate_occurrence_count": metrics["duplicate_occurrence_count"],
                        "redundant_duplicate_count": metrics["redundant_duplicate_count"],
                        "formula_frequency": count,
                        "formula": formula,
                        "occurrences": occurrence_refs(occurrences[formula]),
                    }
                )


def write_summary_csv(
    summary: dict[str, dict[str, object]],
    output_path: Path,
    top_k: int,
    seed_start: int,
    seed_end: int,
) -> None:
    fieldnames = [
        "dataset",
        "seed_start",
        "seed_end",
        "seed_count",
        "archive_file_count",
        "expected_formula_rows",
        "total_formula_rows",
        "unique_formula_count",
        "duplicate_formula_count",
        "duplicate_occurrence_count",
        "redundant_duplicate_count",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dataset in sorted(summary):
            metrics = duplicate_metrics(summary[dataset]["formula_counts"])
            writer.writerow(
                {
                    "dataset": dataset,
                    "seed_start": seed_start,
                    "seed_end": seed_end,
                    "seed_count": len(summary[dataset]["seeds"]),
                    "archive_file_count": len(summary[dataset]["files"]),
                    "expected_formula_rows": (seed_end - seed_start + 1) * top_k,
                    **metrics,
                }
            )


def print_summary(summary: dict[str, dict[str, object]], output_path: Path) -> None:
    print(f"Wrote duplicate formula CSV: {output_path}")
    for dataset in sorted(summary):
        formula_counts: Counter[str] = summary[dataset]["formula_counts"]
        metrics = duplicate_metrics(formula_counts)
        file_count = len(summary[dataset]["files"])
        print(
            f"{dataset}: files={file_count}, total={metrics['total_formula_rows']}, "
            f"unique={metrics['unique_formula_count']}, "
            f"duplicate_formulas={metrics['duplicate_formula_count']}, "
            f"duplicate_occurrences={metrics['duplicate_occurrence_count']}, "
            f"redundant_duplicates={metrics['redundant_duplicate_count']}"
        )
        for warning in summary[dataset]["warnings"]:
            print(f"WARNING: {warning}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a CSV of duplicate formulas from archive_latex.tex files."
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("archive/symbolic_proxy_evolution/autoformer"),
        help="Root directory containing dataset/seed/run archive folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("archive/symbolic_proxy_evolution/autoformer/formula_duplicates.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("archive/symbolic_proxy_evolution/autoformer/formula_summary.csv"),
        help="Output CSV path for the per-dataset summary.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of ranked formulas per file.")
    parser.add_argument("--seed-start", type=int, default=2027)
    parser.add_argument("--seed-end", type=int, default=2226)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize(args.archive_root, args.top_k, args.seed_start, args.seed_end)
    write_csv(summary, args.output)
    write_summary_csv(
        summary,
        args.summary_output,
        args.top_k,
        args.seed_start,
        args.seed_end,
    )
    print_summary(summary, args.output)


if __name__ == "__main__":
    main()
