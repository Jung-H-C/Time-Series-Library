#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def extract_display_name(csv_path: Path) -> str:
    stem = csv_path.stem
    stem = stem.removeprefix("DSPBuilder_")
    suffixes = ("_Benchmark", "_benchmark", "_zscore", "_ZScore")
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                changed = True
    return stem


def benchmark_file_priority(csv_path: Path) -> tuple[int, int, str]:
    stem = csv_path.stem
    has_zscore_suffix = stem.lower().endswith("_zscore")
    is_legacy_named = stem.startswith("DSPBuilder_") or stem.endswith("_Benchmark") or stem.endswith("_benchmark")
    return (
        0 if has_zscore_suffix else 1,
        1 if is_legacy_named else 0,
        str(csv_path),
    )


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)

    start = 0
    while start < sorted_values.shape[0]:
        end = start + 1
        while end < sorted_values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = ((start + end - 1) / 2.0) + 1.0
        ranks[order[start:end]] = average_rank
        start = end

    return ranks


def compute_spearman_correlation(lhs: np.ndarray, rhs: np.ndarray) -> float:
    if lhs.shape != rhs.shape:
        raise ValueError("Spearman correlation inputs must have the same shape.")
    if lhs.size < 2:
        raise ValueError("Spearman correlation requires at least 2 values.")

    lhs_ranks = average_ranks(lhs.astype(np.float64, copy=False))
    rhs_ranks = average_ranks(rhs.astype(np.float64, copy=False))

    lhs_centered = lhs_ranks - lhs_ranks.mean()
    rhs_centered = rhs_ranks - rhs_ranks.mean()
    denominator = np.sqrt(
        float(np.dot(lhs_centered, lhs_centered))
        * float(np.dot(rhs_centered, rhs_centered))
    )
    if denominator == 0.0:
        return 0.0
    return float(np.dot(lhs_centered, rhs_centered) / denominator)


def compute_proxy_signature(metrics: np.ndarray, proxies: np.ndarray) -> np.ndarray:
    return np.asarray(
        [-compute_spearman_correlation(proxies[:, index], metrics) for index in range(proxies.shape[1])],
        dtype=np.float64,
    )


def load_benchmark_csv(csv_path: Path) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = [str(column).replace("\ufeff", "").strip() for column in next(reader)]
        rows = [[float(value) for value in row] for row in reader if row]

    if len(header) < 2:
        raise ValueError(f"Benchmark CSV must contain a metric column and at least one proxy column: {csv_path}")
    if not rows:
        raise ValueError(f"Benchmark CSV is empty: {csv_path}")

    row_array = np.asarray(rows, dtype=np.float64)
    proxy_names = tuple(str(column).strip() for column in header[1:])
    metrics = row_array[:, 0]
    proxies = row_array[:, 1:]
    return proxy_names, metrics, proxies


def build_lookup_rows(benchmark_dir: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    selected_csv_paths: dict[str, Path] = {}
    for csv_path in sorted(benchmark_dir.glob("*.csv")):
        dataset_name = extract_display_name(csv_path)
        dataset_key = dataset_name.lower()
        existing_path = selected_csv_paths.get(dataset_key)
        if existing_path is None or benchmark_file_priority(csv_path) < benchmark_file_priority(existing_path):
            selected_csv_paths[dataset_key] = csv_path
        elif benchmark_file_priority(csv_path) == benchmark_file_priority(existing_path):
            raise ValueError(
                f"Duplicate benchmark CSVs detected for dataset '{dataset_name}': "
                f"{existing_path.name}, {csv_path.name}"
            )

    csv_paths = [selected_csv_paths[key] for key in sorted(selected_csv_paths)]
    if not csv_paths:
        raise FileNotFoundError(f"No benchmark CSV files found under {benchmark_dir}")

    expected_proxy_names: tuple[str, ...] | None = None
    rows: list[dict[str, str]] = []

    for csv_path in csv_paths:
        proxy_names, metrics, proxies = load_benchmark_csv(csv_path)
        if expected_proxy_names is None:
            expected_proxy_names = proxy_names
        elif proxy_names != expected_proxy_names:
            raise ValueError(
                f"Inconsistent proxy columns in {csv_path.name}: "
                f"expected {expected_proxy_names}, got {proxy_names}"
            )

        signature = compute_proxy_signature(metrics, proxies)
        row = {"dataset": extract_display_name(csv_path)}
        row.update(
            {
                proxy_name: f"{float(signature[index]):.6f}"
                for index, proxy_name in enumerate(proxy_names)
            }
        )
        rows.append(row)

    if expected_proxy_names is None:
        raise RuntimeError("Failed to determine proxy names from benchmark CSV files.")
    return expected_proxy_names, rows


def write_lookup_csv(output_path: Path, proxy_names: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", *proxy_names]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Generate benchmark/lookup/proxy_signature_lookup.csv from benchmark CSV files. "
            "Each dataset row stores the raw negative Spearman vector over proxy columns."
        )
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=repo_root / "benchmark",
        help="Directory containing benchmark CSV files such as ECL_zscore.csv or DSPBuilder_ECL_Benchmark.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "benchmark" / "lookup" / "proxy_signature_lookup.csv",
        help="Destination CSV path for the generated proxy signature lookup table.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    proxy_names, rows = build_lookup_rows(args.benchmark_dir)
    write_lookup_csv(args.output, proxy_names, rows)
    print(f"Saved {len(rows)} dataset proxy signatures to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
