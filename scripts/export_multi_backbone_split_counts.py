#!/usr/bin/env python3
"""Export train/val/test sample counts for every run_candidates dataset."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
PROXY_DIR = SCRIPT_PATH.parent / "multi_backbone_proxy"
sys.path.insert(0, str(SCRIPT_PATH.parent))
sys.path.insert(0, str(PROXY_DIR))

from add_split_sample_counts import row_lengths  # noqa: E402
from proxy_experiment_config import DATASETS, DatasetSpec  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "dataset" / "multi_backbone_dataset_split_counts.csv"
FIELDNAMES = [
    "dataset_name",
    "family",
    "source_format",
    "input_len",
    "horizon",
    "train_num_samples",
    "val_num_samples",
    "test_num_samples",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def context_overlap_counts(lengths, seq_len: int, pred_len: int) -> tuple[int, int, int]:
    totals = [0, 0, 0]
    for length in lengths:
        train_end = int(length * 0.7)
        valid_end = int(length * 0.8)
        totals[0] += max(0, train_end - seq_len - pred_len + 1)
        if train_end >= seq_len:
            totals[1] += max(0, valid_end - train_end - pred_len + 1)
        if valid_end >= seq_len:
            totals[2] += max(0, length - valid_end - pred_len + 1)
    return tuple(totals)  # type: ignore[return-value]


def csv_length(dataset: DatasetSpec) -> int:
    path = REPO_ROOT / dataset.root_path / dataset.data_path
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def benchmark_counts(dataset: DatasetSpec, seq_len: int, pred_len: int) -> tuple[int, int, int]:
    length = csv_length(dataset)
    if dataset.name == "ETTh1":
        # Dataset_ETT_hour uses the canonical 12/4/4 month boundaries rather
        # than ratios over every row in the CSV.
        train_length = 12 * 30 * 24
        valid_length = 4 * 30 * 24
        test_length = 4 * 30 * 24
    else:
        train_length = int(length * 0.7)
        test_length = int(length * 0.2)
        valid_length = length - train_length - test_length
    return (
        max(0, train_length - seq_len - pred_len + 1),
        max(0, valid_length - pred_len + 1),
        max(0, test_length - pred_len + 1),
    )


def summary_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {row["dataset_name"]: row for row in csv.DictReader(handle)}


def main() -> int:
    args = parse_args()
    monash_rows = summary_rows(REPO_ROOT / "dataset/monash_dataset_summary.csv")
    time_rows = summary_rows(REPO_ROOT / "dataset/time_dataset_summary.csv")
    output_rows = []

    for registry_name, dataset in DATASETS.items():
        seq_len = 36 if dataset.name == "ILI" else 96
        pred_len = 36 if dataset.name == "ILI" else 96
        if registry_name.startswith("Monash__"):
            family = "Monash"
            raw_name = registry_name.removeprefix("Monash__")
            summary = monash_rows[raw_name]
            root = REPO_ROOT / "dataset/Monash_Dataset"
            counts = context_overlap_counts(row_lengths(summary, root), seq_len, pred_len)
            source_format = summary["source_format"]
        elif registry_name.startswith("TIME__"):
            family = "TIME"
            raw_name = registry_name.removeprefix("TIME__")
            summary = time_rows[raw_name]
            root = REPO_ROOT / "dataset/Time_Dataset"
            counts = context_overlap_counts(row_lengths(summary, root), seq_len, pred_len)
            source_format = summary["source_format"]
        else:
            family = "Benchmark"
            counts = benchmark_counts(dataset, seq_len, pred_len)
            source_format = "csv"

        output_rows.append(
            {
                "dataset_name": registry_name,
                "family": family,
                "source_format": source_format,
                "input_len": seq_len,
                "horizon": pred_len,
                "train_num_samples": counts[0],
                "val_num_samples": counts[1],
                "test_num_samples": counts[2],
            }
        )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote={output} datasets={len(output_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
