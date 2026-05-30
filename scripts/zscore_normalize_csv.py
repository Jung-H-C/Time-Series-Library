#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


LOG1P_COLUMNS = {"synflow"}
ZSCORE_CLIP_MIN = -3.0
ZSCORE_CLIP_MAX = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Z-score normalize every column except the first column in a CSV file. "
            "The first column is copied as-is. The synflow column is log1p-transformed before normalization, "
            "and normalized values are clipped to [-3, 3]."
        )
    )
    parser.add_argument("input_csv", type=Path, help="Path to the input CSV file.")
    parser.add_argument("output_csv", type=Path, help="Path to the output CSV file.")
    return parser.parse_args()


def read_csv_rows(csv_path: Path) -> tuple[list[str], list[list[str]]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Input CSV is empty: {csv_path}")

    header = rows[0]
    data_rows = rows[1:]

    if len(header) < 2:
        raise ValueError("Expected at least 2 columns so the first column can be preserved.")

    expected_width = len(header)
    for row_index, row in enumerate(data_rows, start=2):
        if len(row) != expected_width:
            raise ValueError(
                f"Row {row_index} has {len(row)} columns, but header has {expected_width} columns."
            )

    return header, data_rows


def compute_mean(values: list[float]) -> float:
    return sum(values) / len(values)


def compute_population_std(values: list[float], mean: float) -> float:
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def clip_zscore(value: float) -> float:
    return min(max(value, ZSCORE_CLIP_MIN), ZSCORE_CLIP_MAX)


def transform_column_value(column_name: str, value: float, row_index: int) -> float:
    if column_name.strip().lower() not in LOG1P_COLUMNS:
        return value
    if value < 0.0:
        raise ValueError(
            f"Cannot log1p-transform negative value in column '{column_name}' at row {row_index}: {value}"
        )
    return math.log1p(value)


def zscore_normalize_rows(header: list[str], data_rows: list[list[str]]) -> list[list[str]]:
    if not data_rows:
        return []

    numeric_columns = header[1:]
    numeric_matrix: list[list[float]] = []
    for row_index, row in enumerate(data_rows, start=2):
        numeric_row: list[float] = []
        for column_name, raw_value in zip(numeric_columns, row[1:]):
            try:
                value = float(raw_value)
            except ValueError as exc:
                raise ValueError(
                    f"Failed to parse numeric value in column '{column_name}' at row {row_index}: {raw_value}"
                ) from exc
            numeric_row.append(transform_column_value(column_name, value, row_index))
        numeric_matrix.append(numeric_row)

    num_numeric_columns = len(numeric_matrix[0])
    normalized_columns: list[list[float]] = []
    for column_index in range(num_numeric_columns):
        column_values = [row[column_index] for row in numeric_matrix]
        column_mean = compute_mean(column_values)
        column_std = compute_population_std(column_values, column_mean)
        if column_std == 0.0:
            normalized_columns.append([0.0] * len(column_values))
        else:
            normalized_columns.append(
                [clip_zscore((value - column_mean) / column_std) for value in column_values]
            )

    normalized_rows: list[list[str]] = []
    for row_index, original_row in enumerate(data_rows):
        normalized_values = [
            str(normalized_columns[column_index][row_index])
            for column_index in range(num_numeric_columns)
        ]
        normalized_rows.append([original_row[0], *normalized_values])
    return normalized_rows


def write_csv_rows(csv_path: Path, header: list[str], data_rows: list[list[str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(data_rows)


def main() -> int:
    args = parse_args()
    header, data_rows = read_csv_rows(args.input_csv)
    normalized_rows = zscore_normalize_rows(header, data_rows)
    write_csv_rows(args.output_csv, header, normalized_rows)
    print(f"Saved z-score normalized CSV to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
