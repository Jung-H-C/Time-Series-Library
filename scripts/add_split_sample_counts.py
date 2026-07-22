#!/usr/bin/env python3
"""Add non-crossing train/validation/test window counts to a dataset summary CSV.

Each time series is split independently at int(0.7 * L) and int(0.8 * L).
The number of stride-1 windows in a split of length S is
max(0, S - input_len - pred_len + 1).
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import pyarrow.ipc as ipc

from summarize_monash_datasets import summarize_rds_file


COUNT_COLUMNS = ("train_num_samples", "valid_num_samples", "test_num_samples")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, help="Dataset summary CSV to process.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV (default: <input_stem>_with_sample_counts.csv).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help=(
            "Root against which dataset_path/source_files are resolved. If omitted, "
            "the script searches beside the input CSV for a matching dataset root."
        ),
    )
    parser.add_argument("--input-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop at the first unreadable dataset instead of reporting and skipping it.",
    )
    return parser.parse_args()


def has_numeric_horizon(value: object) -> bool:
    try:
        return math.isfinite(float(str(value).strip()))
    except (TypeError, ValueError):
        return False


def windows(split_length: int, input_len: int, pred_len: int, stride: int) -> int:
    available = split_length - input_len - pred_len
    return max(0, available // stride + 1)


def counts_for_lengths(
    lengths: Iterable[int], input_len: int, pred_len: int, stride: int
) -> tuple[int, int, int]:
    totals = [0, 0, 0]
    for length in lengths:
        train_end = int(length * 0.7)
        valid_end = int(length * 0.8)
        split_lengths = (train_end, valid_end - train_end, length - valid_end)
        for index, split_length in enumerate(split_lengths):
            totals[index] += windows(split_length, input_len, pred_len, stride)
    return tuple(totals)  # type: ignore[return-value]


def tsf_lengths(path: Path) -> Iterable[int]:
    in_data = False
    attribute_count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if not in_data:
                lowered = line.lower()
                if lowered.startswith("@attribute"):
                    attribute_count += 1
                elif lowered == "@data":
                    in_data = True
                continue
            parts = line.split(":", attribute_count)
            if len(parts) != attribute_count + 1:
                raise ValueError(f"Malformed TSF data row in {path}")
            values = parts[-1]
            yield 0 if not values else values.count(",") + 1
    if not in_data:
        raise ValueError(f"Missing @data section in {path}")


def arrow_lengths(paths: Iterable[Path]) -> Iterable[int]:
    for path in paths:
        with path.open("rb") as handle:
            reader = ipc.open_stream(handle)
            if "target" not in reader.schema.names:
                raise ValueError(f"Missing target column in {path}")
            target_index = reader.schema.get_field_index("target")
            for batch in reader:
                for target in batch.column(target_index).to_pylist():
                    if not isinstance(target, list):
                        raise ValueError(f"Invalid target value in {path}")
                    # A nested target is one multivariate series, not one series per channel.
                    if target and isinstance(target[0], list):
                        channel_lengths = [len(channel) for channel in target]
                        if len(set(channel_lengths)) != 1:
                            raise ValueError(f"Unequal channel lengths within a row in {path}")
                        yield channel_lengths[0]
                    else:
                        yield len(target)


def source_paths(row: dict[str, str], root: Path) -> list[Path]:
    pattern = row.get("source_files", "").strip()
    if not pattern:
        raise ValueError("source_files is empty")
    matches = sorted(root.glob(pattern))
    if not matches:
        direct = root / pattern
        if direct.is_file():
            matches = [direct]
    if not matches:
        raise FileNotFoundError(f"No files match {pattern!r} under {root}")
    return matches


def row_lengths(row: dict[str, str], root: Path) -> Iterable[int]:
    source_format = row.get("source_format", "").strip().lower()
    paths = source_paths(row, root)
    if source_format == "tsf":
        for path in paths:
            yield from tsf_lengths(path)
    elif source_format == "arrow":
        yield from arrow_lengths(paths)
    elif source_format == "rds":
        for path in paths:
            yield from summarize_rds_file(path, max_unique_values=100_000).row_lengths
    else:
        raise ValueError(f"Unsupported source_format: {source_format!r}")


def infer_dataset_root(input_csv: Path, rows: list[dict[str, str]]) -> Path:
    parent = input_csv.resolve().parent
    candidates = [parent, parent / "Monash_Dataset", parent / "Time_Dataset", parent / "TIME"]
    example = next((row for row in rows if row.get("source_files", "").strip()), None)
    if example is None:
        raise ValueError("CSV contains no source_files value")
    pattern = example["source_files"].strip()
    for candidate in candidates:
        if candidate.is_dir() and (list(candidate.glob(pattern)) or (candidate / pattern).is_file()):
            return candidate
    raise FileNotFoundError("Could not infer dataset root; pass --dataset-root explicitly")


def main() -> int:
    args = parse_args()
    if args.input_len <= 0 or args.pred_len <= 0 or args.stride <= 0:
        raise ValueError("input-len, pred-len, and stride must be positive")

    input_csv = args.input_csv.resolve()
    output = args.output or input_csv.with_name(f"{input_csv.stem}_with_sample_counts.csv")
    with input_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {input_csv}")
        rows = list(reader)
        original_fields = list(reader.fieldnames)
    if "horizon" not in original_fields:
        raise ValueError("Input CSV must contain a horizon column")

    root = args.dataset_root.resolve() if args.dataset_root else infer_dataset_root(input_csv, rows)
    kept: list[dict[str, str]] = []
    invalid_horizon = 0
    errors = 0
    for row in rows:
        if not has_numeric_horizon(row.get("horizon")):
            invalid_horizon += 1
            continue
        try:
            counts = counts_for_lengths(
                row_lengths(row, root), args.input_len, args.pred_len, args.stride
            )
        except Exception as exc:
            errors += 1
            message = f"[{row.get('dataset_name', '<unknown>')}] {exc}"
            if args.strict:
                raise RuntimeError(message) from exc
            print(f"warning: {message}")
            continue
        for name, count in zip(COUNT_COLUMNS, counts):
            row[name] = str(count)
        kept.append(row)

    fieldnames = [name for name in original_fields if name not in COUNT_COLUMNS] + list(COUNT_COLUMNS)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    print(f"dataset_root={root}")
    print(f"wrote={output} rows={len(kept)} invalid_horizon_removed={invalid_horizon} errors={errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
