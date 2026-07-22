#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from calculate_spearman import spearman_correlation


PROXY_COLUMNS = (
    "MParams",
    "l2_norm",
    "GFLOPs",
    "grad_norm",
    "zico",
    "fisher",
    "grasp",
    "jacob_cov",
    "jacob_fro",
    "plain",
    "snip",
    "GSynFlow",
)
TARGET_COLUMN = "mse"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Calculate one negative-MSE Spearman signature per Monash/TIME dataset. "
            "The 12 zero-cost proxy columns are correlated with MSE and multiplied by -1."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=repo_root / "proxy_scores" / "monash_time",
        help="Directory containing one 300-candidate proxy-score CSV per dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "proxy_scores" / "monash_time_spearman_neg_mse.csv",
        help="Output matrix with one dataset per row and one proxy per column.",
    )
    return parser.parse_args()


def read_dataset_signature(csv_path: Path) -> dict[str, str]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    expected_tail = [*PROXY_COLUMNS, TARGET_COLUMN]
    if fieldnames[11:24] != expected_tail:
        raise ValueError(
            f"{csv_path}: expected Excel columns L:X to be {expected_tail}, "
            f"got {fieldnames[11:24]}"
        )
    if len(rows) != 300:
        raise ValueError(f"{csv_path}: expected 300 model rows, got {len(rows)}")

    dataset_names = {row.get("dataset_name", "").strip() for row in rows}
    if len(dataset_names) != 1 or not next(iter(dataset_names)):
        raise ValueError(f"{csv_path}: expected exactly one non-empty dataset_name")
    dataset_name = next(iter(dataset_names))

    target_values = [float(row[TARGET_COLUMN]) for row in rows]
    if not all(math.isfinite(value) for value in target_values):
        raise ValueError(f"{csv_path}: target column {TARGET_COLUMN!r} contains non-finite values")

    output_row = {"dataset_name": dataset_name}
    for proxy_name in PROXY_COLUMNS:
        proxy_values = [float(row[proxy_name]) for row in rows]
        if not all(math.isfinite(value) for value in proxy_values):
            raise ValueError(f"{csv_path}: proxy column {proxy_name!r} contains non-finite values")
        raw_coefficient = spearman_correlation(proxy_values, target_values)
        if raw_coefficient is None or not math.isfinite(raw_coefficient):
            raise ValueError(f"{csv_path}: Spearman correlation is undefined for {proxy_name!r}")
        output_row[proxy_name] = f"{-raw_coefficient:.10f}"
    return output_row


def main() -> int:
    args = parse_args()
    csv_paths = sorted(args.input_dir.resolve().glob("*.csv"))
    if len(csv_paths) != 47:
        raise ValueError(f"Expected 47 input CSV files under {args.input_dir}, got {len(csv_paths)}")

    rows = [read_dataset_signature(csv_path) for csv_path in csv_paths]
    dataset_names = [row["dataset_name"] for row in rows]
    if len(set(dataset_names)) != len(dataset_names):
        raise ValueError("Duplicate dataset_name values found across input CSV files")

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset_name", *PROXY_COLUMNS],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Saved {len(rows)} datasets x {len(PROXY_COLUMNS)} negative-MSE "
        f"Spearman coefficients to {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
