from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.ipc as ipc


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "TIME"
DEFAULT_OUTPUT = DEFAULT_DATASET_ROOT / "time_dataset_summary.csv"

FIELDNAMES = [
    "dataset_name",
    "source_format",
    "dataset_path",
    "source_files",
    "file_count",
    "status",
    "metadata",
    "feature_count",
    "feature_names",
    "feature_types",
    "channel_attribute",
    "channel_count",
    "channel_names",
    "time_series_count",
    "length_mean",
    "length_min",
    "length_max",
    "length_std",
    "frequency",
    "horizon",
    "missing",
    "equallength",
    "notes",
]


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_value: int | None = None
    max_value: int | None = None

    def add(self, value: int) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.min_value = value if self.min_value is None else min(self.min_value, value)
        self.max_value = value if self.max_value is None else max(self.max_value, value)

    @property
    def std(self) -> float:
        if self.count <= 0:
            return math.nan
        return math.sqrt(self.m2 / self.count)


@dataclass
class TargetSummary:
    length_stats: RunningStats
    channel_counts: set[int]
    channel_names: set[str]
    frequencies: set[str]
    has_missing: bool
    rows: int
    notes: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize TIME/*/* Arrow datasets with the same CSV schema used by "
            "Monash_Dataset/monash_dataset_summary.csv."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Root directory containing TIME datasets. Default: {DEFAULT_DATASET_ROOT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--max-channel-names",
        type=int,
        default=500,
        help="Maximum channel names to retain in the channel_names CSV column.",
    )
    return parser.parse_args()


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def stable_join(values: Any, sep: str = ";") -> str:
    if values is None:
        return ""
    if isinstance(values, dict):
        values = values.keys()
    return sep.join(str(value) for value in sorted(values, key=lambda item: str(item)))


def stats_to_columns(stats: RunningStats) -> dict[str, Any]:
    if stats.count == 0:
        return {
            "length_mean": "",
            "length_min": "",
            "length_max": "",
            "length_std": "",
        }
    return {
        "length_mean": f"{stats.mean:.6f}",
        "length_min": stats.min_value,
        "length_max": stats.max_value,
        "length_std": f"{stats.std:.6f}",
    }


def compact_metadata(dataset_info: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "builder_name",
        "config_name",
        "dataset_name",
        "dataset_size",
        "download_size",
        "size_in_bytes",
        "splits",
        "version",
    ]
    return {key: dataset_info[key] for key in keys if key in dataset_info}


def discover_dataset_dirs(dataset_root: Path) -> list[Path]:
    return sorted(
        path
        for path in dataset_root.glob("*/*")
        if path.is_dir() and list(path.glob("data-*.arrow"))
    )


def feature_types_from_schema(schema: Any) -> dict[str, str]:
    return {field.name: str(field.type).replace("\n", " ") for field in schema}


def target_length_from_dataset_info(dataset_info: dict[str, Any]) -> int:
    target_info = dataset_info.get("features", {}).get("target", {})
    length = target_info.get("length")
    if isinstance(length, int) and length > 0:
        return length
    return 1


def target_dtype_from_dataset_info(dataset_info: dict[str, Any]) -> str:
    node: Any = dataset_info.get("features", {}).get("target", {})
    while isinstance(node, dict):
        dtype = node.get("dtype")
        if isinstance(dtype, str):
            return dtype
        node = node.get("feature")
    return ""


def train_num_examples_from_dataset_info(dataset_info: dict[str, Any]) -> int:
    splits = dataset_info.get("splits", {})
    if isinstance(splits, dict):
        train = splits.get("train")
        if isinstance(train, dict) and isinstance(train.get("num_examples"), int):
            return int(train["num_examples"])
        split_counts = [
            int(split["num_examples"])
            for split in splits.values()
            if isinstance(split, dict) and isinstance(split.get("num_examples"), int)
        ]
        if split_counts:
            return sum(split_counts)
    return 0


def generated_feature_names(feature_count: int) -> list[str]:
    if feature_count <= 1:
        return ["target"]
    return [f"feature_{index}" for index in range(feature_count)]


def sequence_has_missing(values: list[Any]) -> bool:
    for value in values:
        if value is None:
            return True
        if isinstance(value, float) and math.isnan(value):
            return True
    return False


def target_to_sequences(target: Any) -> list[list[Any]]:
    if target is None:
        return []
    if not isinstance(target, list):
        return []
    if target and isinstance(target[0], list):
        return [values if isinstance(values, list) else [] for values in target]
    return [target]


def row_target_length(target: Any, notes: list[str]) -> int | None:
    sequences = target_to_sequences(target)
    if not sequences:
        return None

    lengths = [len(values) for values in sequences]
    unique_lengths = set(lengths)
    if len(unique_lengths) > 1:
        notes.append("mixed target lengths within a multivariate row")
    return lengths[0]


def summarize_arrow_targets(
    arrow_paths: list[Path],
    max_channel_names: int,
) -> TargetSummary:
    length_stats = RunningStats()
    channel_counts: set[int] = set()
    channel_names: set[str] = set()
    frequencies: set[str] = set()
    has_missing = False
    rows = 0
    notes: list[str] = []
    channel_names_truncated = False

    for arrow_path in arrow_paths:
        with arrow_path.open("rb") as handle:
            reader = ipc.open_stream(handle)
            for batch in reader:
                column_names = set(batch.schema.names)
                if "target" not in column_names:
                    notes.append(f"{arrow_path.name}: missing target column")
                    continue

                targets = batch.column(batch.schema.get_field_index("target")).to_pylist()
                freq_values = (
                    batch.column(batch.schema.get_field_index("freq")).to_pylist()
                    if "freq" in column_names
                    else [None for _ in range(batch.num_rows)]
                )
                variate_name_rows = (
                    batch.column(batch.schema.get_field_index("variate_names")).to_pylist()
                    if "variate_names" in column_names
                    else [None for _ in range(batch.num_rows)]
                )

                for target, freq_value, variate_names in zip(targets, freq_values, variate_name_rows):
                    rows += 1
                    if freq_value is not None:
                        frequencies.add(str(freq_value))

                    sequences = target_to_sequences(target)
                    channel_counts.add(len(sequences))
                    length = row_target_length(target, notes)
                    if length is not None:
                        length_stats.add(length)
                    if isinstance(variate_names, list):
                        for name in variate_names:
                            if name is None:
                                continue
                            if len(channel_names) < max_channel_names:
                                channel_names.add(str(name))
                            else:
                                channel_names_truncated = True

                    for values in sequences:
                        if not has_missing and sequence_has_missing(values):
                            has_missing = True

    if channel_names_truncated:
        notes.append(f"channel_names truncated at {max_channel_names}")

    return TargetSummary(
        length_stats=length_stats,
        channel_counts=channel_counts,
        channel_names=channel_names,
        frequencies=frequencies,
        has_missing=has_missing,
        rows=rows,
        notes=notes,
    )


def format_channel_count(values: set[int]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return str(next(iter(values)))
    return f"{min(values)}-{max(values)}"


def summarize_dataset_dir(dataset_dir: Path, dataset_root: Path, max_channel_names: int) -> dict[str, Any]:
    arrow_paths = sorted(dataset_dir.glob("data-*.arrow"))
    dataset_info_path = dataset_dir / "dataset_info.json"
    dataset_info: dict[str, Any] = {}
    if dataset_info_path.exists():
        dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8"))

    schema = None
    for arrow_path in arrow_paths:
        with arrow_path.open("rb") as handle:
            reader = ipc.open_stream(handle)
            schema = reader.schema
            break
    if schema is None:
        raise ValueError(f"No Arrow schema found in {dataset_dir}")

    arrow_feature_names = [field.name for field in schema]
    arrow_feature_types = feature_types_from_schema(schema)
    target_summary = summarize_arrow_targets(arrow_paths, max_channel_names=max_channel_names)
    feature_count = target_length_from_dataset_info(dataset_info)
    time_series_count = train_num_examples_from_dataset_info(dataset_info)
    target_dtype = target_dtype_from_dataset_info(dataset_info)

    folder_frequency = dataset_dir.name
    frequency = stable_join(target_summary.frequencies) or folder_frequency
    channel_count = str(feature_count)
    if target_summary.channel_names:
        feature_names = sorted(target_summary.channel_names, key=lambda item: str(item))
        channel_names = stable_join(feature_names)
        channel_attribute = "variate_names"
    else:
        feature_names = generated_feature_names(feature_count)
        channel_names = "value" if channel_count == "1" else ""
        channel_attribute = ""
    feature_types = stable_join(f"{name}:{target_dtype}" for name in feature_names)

    notes = list(target_summary.notes)
    if len(target_summary.channel_counts) > 1:
        notes.append("mixed channel counts across rows")
    if target_summary.channel_counts and feature_count not in target_summary.channel_counts:
        notes.append(
            "dataset_info target.length differs from arrow target channel count "
            f"({feature_count} vs {format_channel_count(target_summary.channel_counts)})"
        )
    if time_series_count and target_summary.rows != time_series_count:
        notes.append("arrow row count differs from dataset_info train.num_examples")

    dataset_path = relative_path(dataset_dir, dataset_root)
    if len(arrow_paths) == 1:
        source_files = relative_path(arrow_paths[0], dataset_root)
    else:
        source_files = f"{dataset_path}/data-*.arrow"

    row = {
        "dataset_name": f"{dataset_dir.parent.name}__{dataset_dir.name}",
        "source_format": "arrow",
        "dataset_path": dataset_path,
        "source_files": source_files,
        "file_count": len(arrow_paths),
        "status": "ok" if not target_summary.notes else "partial",
        "metadata": json.dumps(compact_metadata(dataset_info), ensure_ascii=False, sort_keys=True),
        "feature_count": feature_count,
        "feature_names": stable_join(feature_names),
        "feature_types": feature_types,
        "channel_attribute": channel_attribute,
        "channel_count": channel_count,
        "channel_names": channel_names,
        "time_series_count": time_series_count or target_summary.rows,
        "frequency": frequency,
        "horizon": "",
        "missing": str(target_summary.has_missing).lower(),
        "equallength": str(
            target_summary.length_stats.count > 0
            and target_summary.length_stats.min_value == target_summary.length_stats.max_value
        ).lower(),
        "notes": " | ".join(notes),
    }
    row.update(stats_to_columns(target_summary.length_stats))
    return row


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDNAMES})


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    rows = [
        summarize_dataset_dir(
            dataset_dir=dataset_dir,
            dataset_root=dataset_root,
            max_channel_names=args.max_channel_names,
        )
        for dataset_dir in discover_dataset_dirs(dataset_root)
    ]
    rows.sort(key=lambda row: str(row["dataset_name"]))
    write_csv(rows, args.output.resolve())
    print(f"Wrote {len(rows)} TIME dataset summaries to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
