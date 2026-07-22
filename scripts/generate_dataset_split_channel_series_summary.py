#!/usr/bin/env python3
"""Measure channels, series, and usable forecast windows from dataset files.

The selected datasets are the six benchmark CSVs plus every row in
``monash_dataset_summary_with_sample_counts.csv`` and
``time_dataset_summary_with_sample_counts.csv``.  Existing channel/count
columns in those summaries are used only to select datasets and locate files;
all reported measurements are recomputed from the source data.

Window counts follow the repository's long-term forecast loaders:

* Every window has stride 1.
* Monash/TIME series use per-series 70/10/20 target ranges.
* Validation and test may use observations before their target range as input
  context, but their prediction targets never cross split boundaries.
* ETTh1 uses its canonical 12/4/4-month boundaries.
* The six benchmark CSVs are one multivariate time series each.
* Each TSF row and RDS metric group is an independent univariate series.
* Each Arrow row is one series and preserves the target's actual channel axis.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
DATASET_ROOT = REPO_ROOT / "dataset"
DEFAULT_MONASH_SUMMARY = DATASET_ROOT / "monash_dataset_summary_with_sample_counts.csv"
DEFAULT_TIME_SUMMARY = DATASET_ROOT / "time_dataset_summary_with_sample_counts.csv"
DEFAULT_OUTPUT = DATASET_ROOT / "dataset_split_channel_series_summary.csv"

FIELDNAMES = [
    "dataset_name",
    "family",
    "source_format",
    "source_path",
    "input_len",
    "pred_len",
    "split_policy",
    "series_type",
    "channel_count",
    "time_series_count",
    "train_num_samples",
    "valid_num_samples",
    "test_num_samples",
]


@dataclass(frozen=True)
class SourceMeasurement:
    lengths: tuple[int, ...]
    channel_count: int

    @property
    def time_series_count(self) -> int:
        return len(self.lengths)


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    relative_path: str
    input_len: int
    pred_len: int
    split_policy: str


BENCHMARKS = (
    BenchmarkSpec(
        name="ECL",
        relative_path="electricity/electricity.csv",
        input_len=96,
        pred_len=96,
        split_policy="70_10_20_context_overlap",
    ),
    BenchmarkSpec(
        name="ETTh1",
        relative_path="ETT-small/ETTh1.csv",
        input_len=96,
        pred_len=96,
        split_policy="canonical_12_4_4_months_context_overlap",
    ),
    BenchmarkSpec(
        name="exchange_rate",
        relative_path="exchange_rate/exchange_rate.csv",
        input_len=96,
        pred_len=96,
        split_policy="70_10_20_context_overlap",
    ),
    BenchmarkSpec(
        name="illness",
        relative_path="illness/national_illness.csv",
        input_len=36,
        pred_len=36,
        split_policy="70_10_20_context_overlap",
    ),
    BenchmarkSpec(
        name="traffic",
        relative_path="traffic/traffic.csv",
        input_len=96,
        pred_len=96,
        split_policy="70_10_20_context_overlap",
    ),
    BenchmarkSpec(
        name="weather",
        relative_path="weather/weather.csv",
        input_len=96,
        pred_len=96,
        split_policy="70_10_20_context_overlap",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monash-summary", type=Path, default=DEFAULT_MONASH_SUMMARY)
    parser.add_argument("--time-summary", type=Path, default=DEFAULT_TIME_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        required = {"dataset_name", "source_format", "source_files", "horizon"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        return list(reader)


def resolve_source_paths(root: Path, pattern: str) -> list[Path]:
    matches = sorted(root.glob(pattern))
    if not matches and (root / pattern).is_file():
        matches = [root / pattern]
    if not matches:
        raise FileNotFoundError(f"No files match {pattern!r} under {root}")
    return matches


def measure_csv(path: Path) -> SourceMeasurement:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty CSV: {path}") from exc
        if not header:
            raise ValueError(f"Missing CSV columns: {path}")
        date_columns = [index for index, name in enumerate(header) if name.strip().lower() == "date"]
        if len(date_columns) != 1:
            raise ValueError(f"Expected exactly one date column in {path}, found {len(date_columns)}")
        channel_count = len(header) - 1
        length = sum(1 for row in reader if row)
    if channel_count <= 0 or length <= 0:
        raise ValueError(f"CSV has no usable target data: {path}")
    return SourceMeasurement(lengths=(length,), channel_count=channel_count)


def iter_tsf_lengths(path: Path) -> Iterator[int]:
    attribute_count = 0
    in_data = False
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
                raise ValueError(f"Malformed TSF row in {path}")
            values = parts[-1]
            length = 0 if not values else values.count(",") + 1
            if length > 0:
                yield length
    if not in_data:
        raise ValueError(f"Missing @data section in {path}")


def measure_tsf(paths: Iterable[Path]) -> SourceMeasurement:
    lengths = tuple(length for path in paths for length in iter_tsf_lengths(path))
    if not lengths:
        raise ValueError("TSF source contains no non-empty series")
    # TSF stores one scalar target sequence per data row. Attribute values such
    # as series_name are identifiers, not a target channel axis.
    return SourceMeasurement(lengths=lengths, channel_count=1)


def target_shape(target: object, path: Path) -> tuple[int, int]:
    if not isinstance(target, list) or not target:
        raise ValueError(f"Invalid or empty Arrow target in {path}")
    if isinstance(target[0], list):
        channels = len(target)
        lengths = [len(values) for values in target if isinstance(values, list)]
        if len(lengths) != channels or len(set(lengths)) != 1:
            raise ValueError(f"Unequal or invalid Arrow channel lengths in {path}")
        return lengths[0], channels
    return len(target), 1


def measure_arrow(paths: Iterable[Path]) -> SourceMeasurement:
    try:
        import pyarrow.ipc as ipc
    except ImportError as exc:
        raise ImportError("pyarrow is required to measure TIME Arrow files") from exc

    lengths: list[int] = []
    channel_counts: set[int] = set()
    for path in paths:
        with path.open("rb") as handle:
            reader = ipc.open_stream(handle)
            if "target" not in reader.schema.names:
                raise ValueError(f"Missing target column in {path}")
            target_index = reader.schema.get_field_index("target")
            for batch in reader:
                for target in batch.column(target_index).to_pylist():
                    length, channels = target_shape(target, path)
                    lengths.append(length)
                    channel_counts.add(channels)
    if not lengths:
        raise ValueError("Arrow source contains no series")
    if len(channel_counts) != 1:
        raise ValueError(f"Mixed Arrow channel counts: {sorted(channel_counts)}")
    return SourceMeasurement(lengths=tuple(lengths), channel_count=next(iter(channel_counts)))


def measure_rds(paths: Iterable[Path]) -> SourceMeasurement:
    try:
        import pyreadr
    except ImportError as exc:
        raise ImportError("pyreadr is required to measure Monash RDS files") from exc

    lengths: list[int] = []
    for path in paths:
        result = pyreadr.read_r(str(path))
        if not result:
            raise ValueError(f"pyreadr returned no objects for {path}")
        frame = next(iter(result.values()))
        if not hasattr(frame, "columns"):
            raise ValueError(f"RDS object is not dataframe-like: {path}")
        if "metric" in frame.columns:
            lengths.extend(int(length) for length in frame.groupby("metric", dropna=False).size())
        else:
            lengths.append(int(len(frame)))
    lengths = [length for length in lengths if length > 0]
    if not lengths:
        raise ValueError("RDS source contains no non-empty series")
    # The repository loader turns every RDS metric group into one scalar
    # sequence, so columns such as sum/min/max are not simultaneous channels.
    return SourceMeasurement(lengths=tuple(lengths), channel_count=1)


def measure_sources(source_format: str, paths: list[Path]) -> SourceMeasurement:
    normalized = source_format.strip().lower()
    if normalized == "tsf":
        return measure_tsf(paths)
    if normalized == "arrow":
        return measure_arrow(paths)
    if normalized == "rds":
        return measure_rds(paths)
    raise ValueError(f"Unsupported source format: {source_format!r}")


def usable_windows(target_length: int, pred_len: int, has_input_context: bool) -> int:
    if not has_input_context:
        return 0
    return max(0, target_length - pred_len + 1)


def per_series_split_counts(
    lengths: Iterable[int], input_len: int, pred_len: int
) -> tuple[int, int, int]:
    totals = [0, 0, 0]
    for length in lengths:
        train_end = int(length * 0.7)
        valid_end = int(length * 0.8)
        totals[0] += usable_windows(
            train_end - input_len,
            pred_len,
            has_input_context=train_end >= input_len,
        )
        totals[1] += usable_windows(
            valid_end - train_end,
            pred_len,
            has_input_context=train_end >= input_len,
        )
        totals[2] += usable_windows(
            length - valid_end,
            pred_len,
            has_input_context=valid_end >= input_len,
        )
    return totals[0], totals[1], totals[2]


def custom_csv_split_counts(length: int, input_len: int, pred_len: int) -> tuple[int, int, int]:
    train_length = int(length * 0.7)
    test_length = int(length * 0.2)
    valid_length = length - train_length - test_length
    return (
        max(0, train_length - input_len - pred_len + 1),
        max(0, valid_length - pred_len + 1) if train_length >= input_len else 0,
        max(0, test_length - pred_len + 1) if train_length + valid_length >= input_len else 0,
    )


def etth1_split_counts(input_len: int, pred_len: int) -> tuple[int, int, int]:
    train_length = 12 * 30 * 24
    valid_length = 4 * 30 * 24
    test_length = 4 * 30 * 24
    return (
        max(0, train_length - input_len - pred_len + 1),
        max(0, valid_length - pred_len + 1),
        max(0, test_length - pred_len + 1),
    )


def output_row(
    *,
    dataset_name: str,
    family: str,
    source_format: str,
    source_path: str,
    input_len: int,
    pred_len: int,
    split_policy: str,
    measurement: SourceMeasurement,
    counts: tuple[int, int, int],
) -> dict[str, object]:
    return {
        "dataset_name": dataset_name,
        "family": family,
        "source_format": source_format,
        "source_path": source_path,
        "input_len": input_len,
        "pred_len": pred_len,
        "split_policy": split_policy,
        "series_type": "univariate" if measurement.channel_count == 1 else "multivariate",
        "channel_count": measurement.channel_count,
        "time_series_count": measurement.time_series_count,
        "train_num_samples": counts[0],
        "valid_num_samples": counts[1],
        "test_num_samples": counts[2],
    }


def benchmark_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in BENCHMARKS:
        path = DATASET_ROOT / spec.relative_path
        measurement = measure_csv(path)
        if spec.name == "ETTh1":
            counts = etth1_split_counts(spec.input_len, spec.pred_len)
        else:
            counts = custom_csv_split_counts(
                measurement.lengths[0], spec.input_len, spec.pred_len
            )
        rows.append(
            output_row(
                dataset_name=spec.name,
                family="Benchmark",
                source_format="csv",
                source_path=path.relative_to(REPO_ROOT).as_posix(),
                input_len=spec.input_len,
                pred_len=spec.pred_len,
                split_policy=spec.split_policy,
                measurement=measurement,
                counts=counts,
            )
        )
    return rows


def summary_dataset_rows(
    summary_path: Path, source_root: Path, family: str
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for summary_row in read_summary(summary_path):
        name = summary_row["dataset_name"].strip()
        source_format = summary_row["source_format"].strip().lower()
        source_pattern = summary_row["source_files"].strip()
        pred_len = int(float(summary_row["horizon"]))
        input_len = 96
        paths = resolve_source_paths(source_root, source_pattern)
        measurement = measure_sources(source_format, paths)
        counts = per_series_split_counts(measurement.lengths, input_len, pred_len)
        output.append(
            output_row(
                dataset_name=name,
                family=family,
                source_format=source_format,
                source_path=f"{source_root.relative_to(REPO_ROOT).as_posix()}/{source_pattern}",
                input_len=input_len,
                pred_len=pred_len,
                split_policy="70_10_20_per_series_context_overlap",
                measurement=measurement,
                counts=counts,
            )
        )
    return output


def write_output(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows = benchmark_rows()
    rows.extend(
        summary_dataset_rows(
            args.monash_summary.resolve(),
            DATASET_ROOT / "Monash_Dataset",
            "Monash",
        )
    )
    rows.extend(
        summary_dataset_rows(
            args.time_summary.resolve(),
            DATASET_ROOT / "Time_Dataset",
            "Time",
        )
    )
    write_output(rows, args.output.resolve())

    family_counts = {
        family: sum(row["family"] == family for row in rows)
        for family in ("Benchmark", "Monash", "Time")
    }
    print(
        f"wrote={args.output.resolve()} datasets={len(rows)} "
        f"benchmark={family_counts['Benchmark']} monash={family_counts['Monash']} "
        f"time={family_counts['Time']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
