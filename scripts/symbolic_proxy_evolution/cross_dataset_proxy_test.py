#!/usr/bin/env python3
"""Evaluate one seed's top symbolic formulas on every dataset's proxy-test split."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

try:
    from evolve_symbolic_proxy import (
        find_groundtruth_csv,
        normalize_proxy_scores,
        spearman_correlation,
    )
    from score_archive_proxy_test import format_score, load_split_values
    from symbolic_tree import parse_rpn
except ImportError:  # pragma: no cover - supports package-style execution.
    from .evolve_symbolic_proxy import (
        find_groundtruth_csv,
        normalize_proxy_scores,
        spearman_correlation,
    )
    from .score_archive_proxy_test import format_score, load_split_values
    from .symbolic_tree import parse_rpn


@dataclass(frozen=True)
class SourceFormula:
    dataset: str
    seed: int
    rank: int
    generation: str
    search_score: str
    train_fitness: str
    rpn_tokens: str
    infix: str
    latex: str
    archive_csv: Path


@dataclass(frozen=True)
class TargetBenchmark:
    dataset: str
    groundtruth_csv: Path
    proxy_values: dict[str, list[float]]
    directed_target: list[float]
    raw_target: list[float]
    split_count: int


def dataset_names(archive_root: Path, seed: int) -> list[str]:
    names = {
        path.parent.name
        for path in archive_root.glob(f"*/seed_{seed}")
        if path.is_dir()
    }
    if not names:
        raise FileNotFoundError(f"No dataset directories found for seed_{seed} under {archive_root}")
    return sorted(names)


def load_source_formulas(
    archive_root: Path,
    datasets: list[str],
    seed: int,
    top_k: int,
) -> list[SourceFormula]:
    formulas: list[SourceFormula] = []
    expected_ranks = list(range(1, top_k + 1))

    for dataset in datasets:
        candidates = sorted((archive_root / dataset / f"seed_{seed}").glob("run_*/archive.csv"))
        if len(candidates) != 1:
            joined = ", ".join(str(path) for path in candidates) if candidates else "none"
            raise FileNotFoundError(
                f"Expected one archive.csv for dataset={dataset}, seed={seed}; found {joined}"
            )
        archive_csv = candidates[0]
        with archive_csv.open(newline="", encoding="utf-8") as handle:
            rows = [
                row
                for row in csv.DictReader(handle)
                if int(row["rank"]) <= top_k
            ]
        rows.sort(key=lambda row: int(row["rank"]))
        ranks = [int(row["rank"]) for row in rows]
        if ranks != expected_ranks:
            raise ValueError(
                f"{archive_csv} has top-k ranks {ranks}; expected {expected_ranks}"
            )

        for row in rows:
            formulas.append(
                SourceFormula(
                    dataset=dataset,
                    seed=seed,
                    rank=int(row["rank"]),
                    generation=row.get("generation", ""),
                    search_score=row.get("Score", ""),
                    train_fitness=row.get("fitness", row.get("objective_fitness", "")),
                    rpn_tokens=row["rpn_tokens"],
                    infix=row.get("inflix", row.get("infix", "")),
                    latex=row.get("latex", ""),
                    archive_csv=archive_csv,
                )
            )
    return formulas


def load_target_benchmarks(
    groundtruth_dir: Path,
    backbone: str,
    datasets: list[str],
    split: str,
    target_metric: str,
) -> list[TargetBenchmark]:
    benchmarks: list[TargetBenchmark] = []
    for dataset in datasets:
        groundtruth_csv = find_groundtruth_csv(groundtruth_dir, backbone, dataset)
        values, directed_target, raw_target, split_count = load_split_values(
            groundtruth_csv=groundtruth_csv,
            split=split,
            target_metric=target_metric,
            target_direction="minimize",
        )
        benchmarks.append(
            TargetBenchmark(
                dataset=dataset,
                groundtruth_csv=groundtruth_csv,
                proxy_values=values,
                directed_target=directed_target,
                raw_target=raw_target,
                split_count=split_count,
            )
        )
    return benchmarks


def score_formulas(
    formulas: list[SourceFormula],
    benchmarks: list[TargetBenchmark],
    split: str,
    target_metric: str,
    proxy_score_decimals: int | None,
    max_abs_proxy_score: float,
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for formula in formulas:
        tree = parse_rpn(formula.rpn_tokens.split())
        for benchmark in benchmarks:
            invalid_reason = ""
            directed_corr = float("nan")
            raw_corr = float("nan")
            try:
                scores, invalid_reason = normalize_proxy_scores(
                    tree.evaluate(benchmark.proxy_values),
                    proxy_score_decimals=proxy_score_decimals,
                    max_abs_proxy_score=max_abs_proxy_score,
                )
                if not invalid_reason:
                    directed_corr = spearman_correlation(scores, benchmark.directed_target)
                    raw_corr = spearman_correlation(scores, benchmark.raw_target)
                    if not math.isfinite(directed_corr):
                        invalid_reason = "nonfinite_spearman_correlation"
            except Exception as exc:  # Keep all formula-target pairs in the output.
                invalid_reason = f"{type(exc).__name__}: {exc}"

            rows.append(
                {
                    "seed": formula.seed,
                    "source_dataset": formula.dataset,
                    "source_rank": formula.rank,
                    "source_generation": formula.generation,
                    "source_search_score": formula.search_score,
                    "source_train_fitness": formula.train_fitness,
                    "rpn_tokens": formula.rpn_tokens,
                    "infix": formula.infix,
                    "latex": formula.latex,
                    "target_dataset": benchmark.dataset,
                    "is_source_dataset": str(formula.dataset == benchmark.dataset).lower(),
                    "target_groundtruth_csv": str(benchmark.groundtruth_csv),
                    "split": split,
                    "split_count": benchmark.split_count,
                    f"spearman_neg_{target_metric}": format_score(directed_corr),
                    f"spearman_raw_{target_metric}": format_score(raw_corr),
                    "invalid_reason": invalid_reason,
                }
            )
    return rows


def write_long_csv(rows: list[dict[str, str | int]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_matrix_csv(
    rows: list[dict[str, str | int]],
    datasets: list[str],
    target_metric: str,
    output_csv: Path,
) -> None:
    identity_fields = [
        "seed",
        "source_dataset",
        "source_rank",
        "source_generation",
        "source_search_score",
        "source_train_fitness",
        "rpn_tokens",
        "infix",
        "latex",
    ]
    score_field = f"spearman_neg_{target_metric}"
    grouped: dict[tuple[str | int, ...], dict[str, str | int]] = {}
    for row in rows:
        key = tuple(row[field] for field in identity_fields)
        matrix_row = grouped.setdefault(key, {field: row[field] for field in identity_fields})
        target = str(row["target_dataset"])
        matrix_row[f"{target}_{score_field}"] = row[score_field]
        matrix_row[f"{target}_invalid_reason"] = row["invalid_reason"]

    fieldnames = identity_fields + [
        field
        for dataset in datasets
        for field in (f"{dataset}_{score_field}", f"{dataset}_invalid_reason")
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(grouped.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-evaluate one seed's top-k symbolic formulas on all proxy-test datasets."
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("archive/symbolic_proxy_evolution/autoformer"),
    )
    parser.add_argument("--groundtruth-dir", type=Path, default=Path("GroundTruth"))
    parser.add_argument("--backbone", default="autoformer")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--split", default="proxy_test")
    parser.add_argument("--target-metric", default="mse")
    parser.add_argument("--proxy-score-decimals", type=int, default=12)
    parser.add_argument("--max-abs-proxy-score", type=float, default=1e12)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "archive/symbolic_proxy_evolution/autoformer/"
            "seed_2027_cross_dataset_proxy_test_spearman.csv"
        ),
    )
    parser.add_argument(
        "--matrix-output-csv",
        type=Path,
        default=Path(
            "archive/symbolic_proxy_evolution/autoformer/"
            "seed_2027_cross_dataset_proxy_test_spearman_matrix.csv"
        ),
    )
    args = parser.parse_args()
    args.proxy_score_decimals = (
        None if args.proxy_score_decimals < 0 else args.proxy_score_decimals
    )
    return args


def main() -> None:
    args = parse_args()
    datasets = dataset_names(args.archive_root, args.seed)
    formulas = load_source_formulas(args.archive_root, datasets, args.seed, args.top_k)
    benchmarks = load_target_benchmarks(
        args.groundtruth_dir,
        args.backbone,
        datasets,
        args.split,
        args.target_metric,
    )
    rows = score_formulas(
        formulas,
        benchmarks,
        args.split,
        args.target_metric,
        args.proxy_score_decimals,
        args.max_abs_proxy_score,
    )
    write_long_csv(rows, args.output_csv)
    write_matrix_csv(rows, datasets, args.target_metric, args.matrix_output_csv)

    invalid_count = sum(bool(row["invalid_reason"]) for row in rows)
    print(f"datasets={len(datasets)} formulas={len(formulas)} evaluations={len(rows)}")
    print(f"valid={len(rows) - invalid_count} invalid={invalid_count}")
    print(f"output_csv={args.output_csv}")
    print(f"matrix_output_csv={args.matrix_output_csv}")


if __name__ == "__main__":
    main()
