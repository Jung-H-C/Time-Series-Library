#!/usr/bin/env python3
"""Aggregate token counts and frequent top-k symbolic proxy formulas."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from summarize_formula_duplicates import extract_formula_rows, normalize_formula
from symbolic_tree import ALL_BINARY_TOKENS, EOS_TOKEN, PROXY_TOKENS, UNARY_TOKENS


FORMULA_TOKENS = PROXY_TOKENS + UNARY_TOKENS + ALL_BINARY_TOKENS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate top-k archive_latex.tex formulas across every available seed. "
            "Token counts are read from the matching archive.csv RPN representation."
        )
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("archive/symbolic_proxy_evolution/autoformer"),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--global-k", type=int, default=50)
    parser.add_argument(
        "--dataset-output",
        type=Path,
        default=Path(
            "archive/symbolic_proxy_evolution/autoformer/"
            "dataset_top10_token_counts.csv"
        ),
    )
    parser.add_argument(
        "--formula-output",
        type=Path,
        default=Path(
            "archive/symbolic_proxy_evolution/autoformer/"
            "global_top50_proxy_formulas.csv"
        ),
    )
    parser.add_argument(
        "--dataset-formula-output",
        type=Path,
        default=Path(
            "archive/symbolic_proxy_evolution/autoformer/"
            "dataset_unique_top10_formulas.csv"
        ),
        help=(
            "Output CSV containing every dataset-wise unique top-k formula, "
            "using canonical RPN as the uniqueness key."
        ),
    )
    return parser.parse_args()


def read_archive_rows(path: Path, top_k: int) -> dict[int, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = {
            int(row["rank"]): row
            for row in csv.DictReader(handle)
            if int(row["rank"]) <= top_k
        }
    return rows


def write_dataset_counts(
    output_path: Path,
    dataset_files: dict[str, set[Path]],
    dataset_seeds: dict[str, set[str]],
    dataset_formula_counts: Counter[str],
    dataset_token_frequencies: dict[str, Counter[str]],
    top_k: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "seed_count",
        "archive_file_count",
        "top_k",
        "formula_count",
        "total_token_count",
        *FORMULA_TOKENS,
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dataset in sorted(dataset_token_frequencies):
            frequencies = dataset_token_frequencies[dataset]
            writer.writerow(
                {
                    "dataset": dataset,
                    "seed_count": len(dataset_seeds[dataset]),
                    "archive_file_count": len(dataset_files[dataset]),
                    "top_k": top_k,
                    "formula_count": dataset_formula_counts[dataset],
                    "total_token_count": sum(frequencies.values()),
                    **{token: frequencies[token] for token in FORMULA_TOKENS},
                }
            )


def write_global_top_formulas(
    output_path: Path,
    formula_counts: Counter[str],
    formula_datasets: dict[str, set[str]],
    formula_seeds: dict[str, set[tuple[str, str]]],
    formula_token_counts: dict[str, set[int]],
    formula_rpn: dict[str, str],
    total_formula_rows: int,
    global_k: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "global_rank",
        "occurrence_count",
        "occurrence_percentage",
        "dataset_count",
        "seed_count",
        "token_count",
        "rpn_tokens",
        "latex_formula",
    ]
    ranked = sorted(formula_counts.items(), key=lambda item: (-item[1], item[0]))
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for global_rank, (formula, count) in enumerate(ranked[:global_k], start=1):
            token_counts = formula_token_counts[formula]
            if len(token_counts) != 1:
                raise ValueError(
                    f"Formula maps to inconsistent token counts {sorted(token_counts)}: {formula}"
                )
            writer.writerow(
                {
                    "global_rank": global_rank,
                    "occurrence_count": count,
                    "occurrence_percentage": f"{100 * count / total_formula_rows:.6f}",
                    "dataset_count": len(formula_datasets[formula]),
                    "seed_count": len(formula_seeds[formula]),
                    "token_count": next(iter(token_counts)),
                    "rpn_tokens": formula_rpn[formula],
                    "latex_formula": formula,
                }
            )


def write_dataset_unique_formulas(
    output_path: Path,
    dataset_files: dict[str, set[Path]],
    dataset_seeds: dict[str, set[str]],
    formula_counts: dict[str, Counter[str]],
    formula_seeds: dict[str, dict[str, set[str]]],
    formula_ranks: dict[str, dict[str, list[int]]],
    formula_token_counts: dict[str, dict[str, set[int]]],
    formula_latex: dict[str, dict[str, set[str]]],
    formula_examples: dict[str, dict[str, str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "dataset_seed_count",
        "dataset_archive_file_count",
        "dataset_top10_formula_rows",
        "dataset_unique_formula_count",
        "frequency_rank",
        "occurrence_count",
        "occurrence_percentage",
        "seed_count",
        "best_top10_rank",
        "mean_top10_rank",
        "token_count",
        "rpn_tokens",
        "latex_formula",
        "example_source",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dataset in sorted(formula_counts):
            counts = formula_counts[dataset]
            total_rows = sum(counts.values())
            ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            for frequency_rank, (rpn_tokens, count) in enumerate(ranked, start=1):
                token_counts = formula_token_counts[dataset][rpn_tokens]
                latex_formulas = formula_latex[dataset][rpn_tokens]
                if len(token_counts) != 1:
                    raise ValueError(
                        f"Inconsistent token counts for {dataset}/{rpn_tokens}: "
                        f"{sorted(token_counts)}"
                    )
                if len(latex_formulas) != 1:
                    raise ValueError(
                        f"Inconsistent LaTeX formulas for {dataset}/{rpn_tokens}: "
                        f"{sorted(latex_formulas)}"
                    )
                ranks = formula_ranks[dataset][rpn_tokens]
                writer.writerow(
                    {
                        "dataset": dataset,
                        "dataset_seed_count": len(dataset_seeds[dataset]),
                        "dataset_archive_file_count": len(dataset_files[dataset]),
                        "dataset_top10_formula_rows": total_rows,
                        "dataset_unique_formula_count": len(counts),
                        "frequency_rank": frequency_rank,
                        "occurrence_count": count,
                        "occurrence_percentage": f"{100 * count / total_rows:.6f}",
                        "seed_count": len(formula_seeds[dataset][rpn_tokens]),
                        "best_top10_rank": min(ranks),
                        "mean_top10_rank": f"{sum(ranks) / len(ranks):.6f}",
                        "token_count": next(iter(token_counts)),
                        "rpn_tokens": rpn_tokens,
                        "latex_formula": next(iter(latex_formulas)),
                        "example_source": formula_examples[dataset][rpn_tokens],
                    }
                )


def main() -> int:
    args = parse_args()
    if args.top_k < 1 or args.global_k < 1:
        raise ValueError("--top-k and --global-k must be positive")

    latex_paths = sorted(
        args.archive_root.glob("*/seed_*/run_*/visualizations/archive_latex.tex")
    )
    if not latex_paths:
        raise FileNotFoundError(f"No archive_latex.tex files found under {args.archive_root}")

    dataset_files: dict[str, set[Path]] = defaultdict(set)
    dataset_seeds: dict[str, set[str]] = defaultdict(set)
    dataset_formula_counts: Counter[str] = Counter()
    dataset_token_frequencies: dict[str, Counter[str]] = defaultdict(Counter)
    formula_counts: Counter[str] = Counter()
    formula_datasets: dict[str, set[str]] = defaultdict(set)
    formula_seeds: dict[str, set[tuple[str, str]]] = defaultdict(set)
    formula_token_counts: dict[str, set[int]] = defaultdict(set)
    formula_rpn: dict[str, str] = {}
    dataset_unique_counts: dict[str, Counter[str]] = defaultdict(Counter)
    dataset_unique_seeds: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    dataset_unique_ranks: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    dataset_unique_token_counts: dict[str, dict[str, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    dataset_unique_latex: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    dataset_unique_examples: dict[str, dict[str, str]] = defaultdict(dict)

    for latex_path in latex_paths:
        dataset, seed, run = latex_path.relative_to(args.archive_root).parts[:3]
        latex_rows = extract_formula_rows(latex_path, args.top_k)
        if len(latex_rows) != args.top_k:
            raise ValueError(
                f"Expected {args.top_k} LaTeX rows, found {len(latex_rows)}: {latex_path}"
            )

        archive_csv = latex_path.parent.parent / "archive.csv"
        if not archive_csv.is_file():
            raise FileNotFoundError(f"Missing matching archive.csv: {archive_csv}")
        csv_rows = read_archive_rows(archive_csv, args.top_k)
        if len(csv_rows) != args.top_k:
            raise ValueError(
                f"Expected {args.top_k} CSV rows, found {len(csv_rows)}: {archive_csv}"
            )

        dataset_files[dataset].add(latex_path)
        dataset_seeds[dataset].add(seed)
        for rank, latex_formula in latex_rows:
            csv_row = csv_rows[rank]
            csv_formula = normalize_formula(csv_row["latex"])
            if latex_formula != csv_formula:
                raise ValueError(
                    f"Formula mismatch at rank {rank}: {latex_path}\n"
                    f"LaTeX: {latex_formula}\nCSV: {csv_formula}"
                )
            token_count = int(csv_row["token_count"])
            rpn_tokens = csv_row["rpn_tokens"].split()
            formula_tokens = [token for token in rpn_tokens if token != EOS_TOKEN]
            unknown_tokens = sorted(set(formula_tokens) - set(FORMULA_TOKENS))
            if unknown_tokens:
                raise ValueError(f"Unknown RPN tokens {unknown_tokens}: {archive_csv}")
            if len(formula_tokens) != token_count:
                raise ValueError(
                    f"RPN/token_count mismatch at rank {rank}: {archive_csv}; "
                    f"RPN has {len(formula_tokens)}, token_count is {token_count}"
                )
            dataset_formula_counts[dataset] += 1
            dataset_token_frequencies[dataset].update(formula_tokens)
            canonical_rpn = " ".join(rpn_tokens)
            dataset_unique_counts[dataset][canonical_rpn] += 1
            dataset_unique_seeds[dataset][canonical_rpn].add(seed)
            dataset_unique_ranks[dataset][canonical_rpn].append(rank)
            dataset_unique_token_counts[dataset][canonical_rpn].add(token_count)
            dataset_unique_latex[dataset][canonical_rpn].add(latex_formula)
            dataset_unique_examples[dataset].setdefault(
                canonical_rpn,
                f"{seed}/{run}/rank_{rank}",
            )
            formula_counts[latex_formula] += 1
            formula_datasets[latex_formula].add(dataset)
            formula_seeds[latex_formula].add((dataset, seed))
            formula_token_counts[latex_formula].add(token_count)
            formula_rpn.setdefault(latex_formula, csv_row["rpn_tokens"])

    write_dataset_counts(
        args.dataset_output,
        dataset_files,
        dataset_seeds,
        dataset_formula_counts,
        dataset_token_frequencies,
        args.top_k,
    )
    total_formula_rows = sum(formula_counts.values())
    write_global_top_formulas(
        args.formula_output,
        formula_counts,
        formula_datasets,
        formula_seeds,
        formula_token_counts,
        formula_rpn,
        total_formula_rows,
        args.global_k,
    )
    write_dataset_unique_formulas(
        args.dataset_formula_output,
        dataset_files,
        dataset_seeds,
        dataset_unique_counts,
        dataset_unique_seeds,
        dataset_unique_ranks,
        dataset_unique_token_counts,
        dataset_unique_latex,
        dataset_unique_examples,
    )

    print(f"Datasets: {len(dataset_token_frequencies)}")
    print(f"Archive files: {len(latex_paths)}")
    print(f"Top-{args.top_k} formula rows: {total_formula_rows}")
    print(f"Unique formulas: {len(formula_counts)}")
    print(
        "Dataset-wise unique formula rows: "
        f"{sum(len(counts) for counts in dataset_unique_counts.values())}"
    )
    print(f"Wrote: {args.dataset_output}")
    print(f"Wrote: {args.formula_output}")
    print(f"Wrote: {args.dataset_formula_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
