from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "Monash_Dataset"
DEFAULT_OUTPUT = DEFAULT_DATASET_ROOT / "monash_dataset_summary.csv"


CHANNEL_ATTRIBUTE_NAMES = {
    "channel",
    "channels",
    "metric",
    "metrics",
    "series_type",
    "type",
    "variable",
    "variables",
    "target",
}
TIME_COLUMN_TOKENS = ("time", "date", "timestamp", "utc")
ID_COLUMN_NAMES = {"id", "series_id", "series_name", "unit"}
NON_VALUE_COLUMN_NAMES = ID_COLUMN_NAMES | {"count"}


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

    def add_many(self, values: list[int]) -> None:
        for value in values:
            self.add(int(value))

    @property
    def std(self) -> float:
        if self.count <= 0:
            return math.nan
        return math.sqrt(self.m2 / self.count)


@dataclass
class RObject:
    type_name: str
    length: int | None = None
    attrs: dict[str, "RObject"] = field(default_factory=dict)
    tag: "RObject | None" = None
    values: list[Any] | None = None
    elements: list["RObject"] | None = None
    unique_counts: Counter[str] = field(default_factory=Counter)
    unique_truncated: bool = False
    ref_index: int | None = None


class RdsReader:
    """Small RDS reader for Monash .rds data frames.

    It intentionally implements only the R serialization types needed for
    lightweight metadata extraction. Numeric payloads are skipped instead of
    materialized; string vectors keep small values and bounded unique counts.
    """

    TYPE_NAMES = {
        1: "SYMSXP",
        2: "LISTSXP",
        6: "LANGSXP",
        9: "CHARSXP",
        10: "LGLSXP",
        13: "INTSXP",
        14: "REALSXP",
        15: "CPLXSXP",
        16: "STRSXP",
        19: "VECSXP",
        24: "RAWSXP",
        254: "NILVALUE",
        255: "REFSXP",
    }
    HAS_ATTR = 1 << 9
    HAS_TAG = 1 << 10

    def __init__(self, data: bytes, max_unique_values: int = 50) -> None:
        self.data = data
        self.index = 0
        self.refs: list[RObject] = []
        self.max_unique_values = max_unique_values

    def read(self) -> RObject:
        self._read_header()
        return self._read_item()

    def _read(self, nbytes: int) -> bytes:
        if self.index + nbytes > len(self.data):
            raise ValueError("Unexpected end of RDS payload")
        value = self.data[self.index : self.index + nbytes]
        self.index += nbytes
        return value

    def _read_int(self) -> int:
        return struct.unpack(">i", self._read(4))[0]

    def _read_double(self) -> float:
        return struct.unpack(">d", self._read(8))[0]

    def _skip(self, nbytes: int) -> None:
        self._read(nbytes)

    def _read_header(self) -> None:
        magic = self._read(2)
        if magic != b"X\n":
            raise ValueError(f"Unsupported RDS encoding magic {magic!r}; expected XDR binary RDS")

        format_version = self._read_int()
        self._read_int()  # writer version
        self._read_int()  # minimum reader version
        if format_version >= 3:
            encoding_length = self._read_int()
            if encoding_length > 0:
                self._skip(encoding_length)

    def _read_item(self) -> RObject:
        flags = self._read_int()
        sexp_type = flags & 0xFF

        if sexp_type == 254:
            return RObject(type_name="NILVALUE")

        if sexp_type == 255:
            ref_index = flags >> 8
            ref_obj = self._resolve_ref(ref_index)
            if ref_obj is not None:
                return ref_obj
            return RObject(type_name="REFSXP", ref_index=ref_index)

        tag = self._read_item() if flags & self.HAS_TAG else None
        type_name = self.TYPE_NAMES.get(sexp_type, f"SEXP_{sexp_type}")

        if sexp_type == 1:
            obj = RObject(type_name=type_name, values=[self._read_item()], tag=tag)
            self.refs.append(obj)
        elif sexp_type == 2 or sexp_type == 6:
            car = self._read_item()
            cdr = self._read_item()
            obj = RObject(type_name=type_name, values=[car, cdr], tag=tag)
        elif sexp_type == 9:
            length = self._read_int()
            value = None if length < 0 else self._read(length).decode("utf-8", errors="replace")
            obj = RObject(type_name=type_name, length=length, values=[value], tag=tag)
        elif sexp_type == 10 or sexp_type == 13:
            length = self._read_int()
            obj = RObject(type_name=type_name, length=length, tag=tag)
            self._skip(4 * length)
        elif sexp_type == 14:
            length = self._read_int()
            obj = RObject(type_name=type_name, length=length, tag=tag)
            self._skip(8 * length)
        elif sexp_type == 15:
            length = self._read_int()
            obj = RObject(type_name=type_name, length=length, tag=tag)
            self._skip(16 * length)
        elif sexp_type == 16:
            obj = self._read_string_vector(tag=tag)
        elif sexp_type == 19:
            length = self._read_int()
            elements = [self._read_item() for _ in range(length)]
            obj = RObject(type_name=type_name, length=length, elements=elements, tag=tag)
        elif sexp_type == 24:
            length = self._read_int()
            obj = RObject(type_name=type_name, length=length, tag=tag)
            self._skip(length)
        else:
            raise ValueError(f"Unsupported RDS SEXP type {sexp_type} at byte {self.index}")

        if flags & self.HAS_ATTR:
            obj.attrs = self._read_pairlist_attrs(self._read_item())
        return obj

    def _read_string_vector(self, tag: RObject | None) -> RObject:
        length = self._read_int()
        values: list[str | None] | None = [] if length <= self.max_unique_values else None
        unique_counts: Counter[str] = Counter()
        unique_truncated = False

        for _ in range(length):
            item = self._read_item()
            value = self._string_value(item)
            if values is not None:
                values.append(value)
            if value is None:
                continue
            if value in unique_counts:
                unique_counts[value] += 1
            elif len(unique_counts) < self.max_unique_values:
                unique_counts[value] = 1
            else:
                unique_truncated = True

        return RObject(
            type_name="STRSXP",
            length=length,
            values=values,
            unique_counts=unique_counts,
            unique_truncated=unique_truncated,
            tag=tag,
        )

    def _read_pairlist_attrs(self, pairlist: RObject) -> dict[str, RObject]:
        attrs: dict[str, RObject] = {}
        current = pairlist
        while current.type_name != "NILVALUE":
            if current.type_name not in {"LISTSXP", "LANGSXP"} or not current.values:
                break
            car, cdr = current.values
            tag_name = self._symbol_name(current.tag)
            if tag_name is not None:
                attrs[tag_name] = car
            current = cdr
        return attrs

    def _resolve_ref(self, ref_index: int) -> RObject | None:
        if ref_index <= 0:
            return None
        position = ref_index - 1
        if position >= len(self.refs):
            return None
        return self.refs[position]

    def _string_value(self, obj: RObject | None) -> str | None:
        if obj is None:
            return None
        if obj.type_name == "CHARSXP" and obj.values:
            return obj.values[0]
        if obj.type_name == "SYMSXP" and obj.values:
            return self._string_value(obj.values[0])
        return None

    def _symbol_name(self, obj: RObject | None) -> str | None:
        return self._string_value(obj)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Monash .tsf and per-unit .rds time-series datasets into a CSV file."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Root directory containing Monash datasets. Default: {DEFAULT_DATASET_ROOT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--skip-rds",
        action="store_true",
        help="Only summarize .tsf datasets.",
    )
    parser.add_argument(
        "--max-unique-values",
        type=int,
        default=50,
        help="Maximum unique categorical values to record per metadata/channel field.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately if any dataset cannot be parsed.",
    )
    parser.add_argument(
        "--print-series-name-keys",
        action="store_true",
        help=(
            "Print all unique series_name values for datasets whose feature_types "
            "include series_name. The values are printed to stdout and are not "
            "stored in the output CSV."
        ),
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
    if isinstance(values, Counter):
        values = values.keys()
    if isinstance(values, dict):
        values = values.keys()
    return sep.join(str(value) for value in sorted(values, key=lambda item: str(item)))


def stats_to_columns(stats: RunningStats) -> dict[str, Any]:
    if stats.count == 0:
        return {
            "time_series_count": 0,
            "length_mean": "",
            "length_min": "",
            "length_max": "",
            "length_std": "",
        }
    return {
        "time_series_count": stats.count,
        "length_mean": f"{stats.mean:.6f}",
        "length_min": stats.min_value,
        "length_max": stats.max_value,
        "length_std": f"{stats.std:.6f}",
    }


def read_tsf_header(path: Path) -> tuple[dict[str, str], list[tuple[str, str]]]:
    metadata: dict[str, str] = {}
    attributes: list[tuple[str, str]] = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            lowered = line.lower()
            if lowered == "@data":
                return metadata, attributes
            if lowered.startswith("@attribute"):
                parts = line.split(maxsplit=2)
                if len(parts) < 3:
                    raise ValueError(f"Malformed @attribute line in {path}: {line}")
                attributes.append((parts[1], parts[2]))
            elif lowered.startswith("@"):
                parts = line.split(maxsplit=1)
                key = parts[0][1:].lower()
                metadata[key] = parts[1].strip() if len(parts) > 1 else ""

    raise ValueError(f"Missing @data section in {path}")


def summarize_tsf_file(path: Path, dataset_root: Path, max_unique_values: int) -> dict[str, Any]:
    metadata, attributes = read_tsf_header(path)
    attribute_names = [name for name, _ in attributes]
    attribute_types = [attr_type for _, attr_type in attributes]
    channel_candidates: dict[str, Counter[str]] = {
        name: Counter()
        for name in attribute_names
        if name.lower() in CHANNEL_ATTRIBUTE_NAMES
    }
    channel_truncated: set[str] = set()
    length_stats = RunningStats()
    in_data = False

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if not in_data:
                if line.lower() == "@data":
                    in_data = True
                continue

            parts = line.split(":", len(attribute_names))
            if len(parts) != len(attribute_names) + 1:
                raise ValueError(
                    f"Malformed TSF data row in {path}: expected {len(attribute_names)} attributes"
                )

            values = parts[-1]
            length_stats.add(0 if values == "" else values.count(",") + 1)

            for attr_name, attr_value in zip(attribute_names, parts[:-1]):
                if attr_name not in channel_candidates:
                    continue
                counter = channel_candidates[attr_name]
                if attr_value in counter:
                    counter[attr_value] += 1
                elif len(counter) < max_unique_values:
                    counter[attr_value] = 1
                else:
                    channel_truncated.add(attr_name)

    channel_attribute = ""
    channel_names: list[str] = []
    for attr_name, counter in channel_candidates.items():
        if counter:
            channel_attribute = attr_name
            channel_names = sorted(counter.keys())
            break

    channel_count = len(channel_names) if channel_names else 1
    if channel_attribute in channel_truncated:
        channel_count = f">={channel_count}"

    row = {
        "dataset_name": path.parent.name,
        "source_format": "tsf",
        "dataset_path": relative_path(path.parent, dataset_root),
        "source_files": relative_path(path, dataset_root),
        "file_count": 1,
        "status": "ok",
        "metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        "feature_count": len(attributes),
        "feature_names": stable_join(attribute_names),
        "feature_types": stable_join(
            f"{name}:{attr_type}" for name, attr_type in zip(attribute_names, attribute_types)
        ),
        "channel_attribute": channel_attribute,
        "channel_count": channel_count,
        "channel_names": stable_join(channel_names) if channel_names else "value",
        "frequency": metadata.get("frequency", ""),
        "horizon": metadata.get("horizon", ""),
        "missing": metadata.get("missing", ""),
        "equallength": metadata.get("equallength", ""),
        "notes": "",
    }
    row.update(stats_to_columns(length_stats))
    return row


def read_rds_bytes(path: Path) -> bytes:
    with path.open("rb") as handle:
        magic = handle.read(2)
    if magic == b"\x1f\x8b":
        with gzip.open(path, "rb") as handle:
            return handle.read()
    return path.read_bytes()


def r_attrs_strings(obj: RObject, key: str) -> list[str]:
    attr = obj.attrs.get(key)
    if attr is None:
        return []
    if attr.values is not None:
        return [str(value) for value in attr.values if value is not None]
    return sorted(attr.unique_counts.keys())


def r_object_type(obj: RObject) -> str:
    classes = r_attrs_strings(obj, "class")
    if classes:
        return f"{obj.type_name}<{','.join(classes)}>"
    return obj.type_name


def is_time_column(name: str, obj: RObject) -> bool:
    lowered = name.lower()
    if any(token in lowered for token in TIME_COLUMN_TOKENS):
        return True
    classes = {value.lower() for value in r_attrs_strings(obj, "class")}
    return bool(classes & {"posixct", "posixt", "date"})


def is_numeric_r_object(obj: RObject) -> bool:
    return obj.type_name in {"REALSXP", "INTSXP", "LGLSXP"}


@dataclass
class RdsFileSummary:
    row_lengths: list[int]
    feature_names: list[str]
    feature_types: dict[str, str]
    channel_names: set[str]
    channel_attribute: str
    object_classes: set[str]


def summarize_rds_file(path: Path, max_unique_values: int) -> RdsFileSummary:
    try:
        return summarize_rds_file_with_pyreadr(path)
    except ImportError:
        pass
    except Exception:
        # Keep the script usable for simple Monash RDS files even when pyreadr
        # fails on a specific file.
        pass

    data = read_rds_bytes(path)
    root = RdsReader(data, max_unique_values=max_unique_values).read()
    object_classes = set(r_attrs_strings(root, "class"))

    if root.type_name == "VECSXP" and root.elements:
        names_attr = root.attrs.get("names")
        names = names_attr.values if names_attr is not None and names_attr.values else []
        feature_names = [
            str(name) if name is not None else f"field_{index}"
            for index, name in enumerate(names[: len(root.elements)])
        ]
        if len(feature_names) < len(root.elements):
            feature_names.extend(
                f"field_{index}" for index in range(len(feature_names), len(root.elements))
            )

        feature_types = {
            name: r_object_type(element)
            for name, element in zip(feature_names, root.elements)
        }
        candidate_channels = extract_rds_metric_channels(feature_names, root.elements)
        if candidate_channels:
            channel_attribute, channel_lengths, channel_names = candidate_channels
        else:
            channel_attribute = ""
            channel_names = {
                name
                for name, element in zip(feature_names, root.elements)
                if is_numeric_r_object(element)
                and not is_time_column(name, element)
                and name.lower() not in NON_VALUE_COLUMN_NAMES
            }
            row_length = next(
                (element.length for element in root.elements if element.length is not None),
                root.length or 0,
            )
            channel_lengths = [int(row_length)] if row_length else []

        return RdsFileSummary(
            row_lengths=channel_lengths,
            feature_names=feature_names,
            feature_types=feature_types,
            channel_names=channel_names,
            channel_attribute=channel_attribute,
            object_classes=object_classes,
        )

    return RdsFileSummary(
        row_lengths=[root.length] if root.length is not None else [],
        feature_names=["value"],
        feature_types={"value": r_object_type(root)},
        channel_names={"value"},
        channel_attribute="",
        object_classes=object_classes,
    )


def summarize_rds_file_with_pyreadr(path: Path) -> RdsFileSummary:
    import pyreadr

    result = pyreadr.read_r(str(path))
    if not result:
        raise ValueError(f"pyreadr returned no objects for {path}")

    first_object = next(iter(result.values()))
    if not hasattr(first_object, "columns"):
        raise ValueError(f"pyreadr object is not a dataframe-like object: {type(first_object)!r}")

    df = first_object
    feature_names = [str(column) for column in df.columns]
    feature_types = {str(column): str(dtype) for column, dtype in df.dtypes.items()}

    candidate = extract_pyreadr_metric_channels(df)
    if candidate is not None:
        channel_attribute, row_lengths, channel_names = candidate
    else:
        channel_attribute = ""
        channel_names = {
            name
            for name in feature_names
            if is_pyreadr_value_column(name=name, dtype=str(df.dtypes[name]))
        }
        row_count = int(len(df))
        row_lengths = [row_count for _ in channel_names] if row_count else []

    return RdsFileSummary(
        row_lengths=row_lengths,
        feature_names=feature_names,
        feature_types=feature_types,
        channel_names=channel_names,
        channel_attribute=channel_attribute,
        object_classes={type(df).__name__, "pyreadr"},
    )


def extract_pyreadr_metric_channels(df: Any) -> tuple[str, list[int], set[str]] | None:
    for column in df.columns:
        name = str(column)
        if name.lower() not in CHANNEL_ATTRIBUTE_NAMES:
            continue
        counts = df[column].value_counts(dropna=False)
        if counts.empty:
            continue
        channel_names = {str(index) for index in counts.index}
        row_lengths = [int(value) for value in counts.to_list()]
        return name, row_lengths, channel_names
    return None


def is_pyreadr_value_column(name: str, dtype: str) -> bool:
    lowered = name.lower()
    if lowered in NON_VALUE_COLUMN_NAMES:
        return False
    if any(token in lowered for token in TIME_COLUMN_TOKENS):
        return False
    return any(token in dtype for token in ("int", "float", "bool"))


def extract_rds_metric_channels(
    feature_names: list[str],
    elements: list[RObject],
) -> tuple[str, list[int], set[str]] | None:
    for name, element in zip(feature_names, elements):
        if name.lower() not in CHANNEL_ATTRIBUTE_NAMES:
            continue
        if not element.unique_counts:
            continue
        return name, [int(count) for count in element.unique_counts.values()], set(element.unique_counts.keys())
    return None


def summarize_rds_group(
    dataset_dir: Path,
    rds_files: list[Path],
    dataset_root: Path,
    max_unique_values: int,
    strict: bool,
) -> dict[str, Any]:
    length_stats = RunningStats()
    feature_names: set[str] = set()
    feature_types: dict[str, set[str]] = defaultdict(set)
    channel_names: set[str] = set()
    channel_attributes: set[str] = set()
    object_classes: set[str] = set()
    errors: list[str] = []

    for path in sorted(rds_files):
        try:
            summary = summarize_rds_file(path, max_unique_values=max_unique_values)
        except Exception as exc:
            message = f"{path.name}: {exc}"
            if strict:
                raise ValueError(message) from exc
            errors.append(message)
            continue

        length_stats.add_many(summary.row_lengths)
        feature_names.update(summary.feature_names)
        for name, type_name in summary.feature_types.items():
            feature_types[name].add(type_name)
        channel_names.update(summary.channel_names)
        if summary.channel_attribute:
            channel_attributes.add(summary.channel_attribute)
        object_classes.update(summary.object_classes)

    row = {
        "dataset_name": dataset_dir.name,
        "source_format": "rds",
        "dataset_path": relative_path(dataset_dir, dataset_root),
        "source_files": f"{relative_path(dataset_dir, dataset_root)}/*.rds",
        "file_count": len(rds_files),
        "status": "ok" if not errors else "partial",
        "metadata": json.dumps({"object_classes": sorted(object_classes)}, ensure_ascii=False),
        "feature_count": len(feature_names),
        "feature_names": stable_join(feature_names),
        "feature_types": stable_join(
            f"{name}:{stable_join(types, '|')}" for name, types in feature_types.items()
        ),
        "channel_attribute": stable_join(channel_attributes),
        "channel_count": len(channel_names) if channel_names else "",
        "channel_names": stable_join(channel_names),
        "frequency": "",
        "horizon": "",
        "missing": "",
        "equallength": "",
        "notes": " | ".join(errors[:5]),
    }
    if len(errors) > 5:
        row["notes"] += f" | ... {len(errors) - 5} more errors"
    row.update(stats_to_columns(length_stats))
    return row


def discover_tsf_files(dataset_root: Path) -> list[Path]:
    return sorted(dataset_root.rglob("*.tsf"))


def discover_rds_groups(dataset_root: Path) -> dict[Path, list[Path]]:
    groups: dict[Path, list[Path]] = defaultdict(list)
    for path in sorted(dataset_root.rglob("*.rds")):
        groups[path.parent].append(path)
    return dict(groups)


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def row_has_series_name_feature(row: dict[str, Any]) -> bool:
    feature_types = str(row.get("feature_types", ""))
    return any(
        item.split(":", 1)[0] == "series_name"
        for item in feature_types.split(";")
        if item
    )


def collect_tsf_series_name_keys(path: Path) -> set[str]:
    _, attributes = read_tsf_header(path)
    attribute_names = [name for name, _ in attributes]
    if "series_name" not in attribute_names:
        return set()

    series_name_index = attribute_names.index("series_name")
    series_names: set[str] = set()
    in_data = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if not in_data:
                if line.lower() == "@data":
                    in_data = True
                continue

            parts = line.split(":", len(attribute_names))
            if len(parts) != len(attribute_names) + 1:
                raise ValueError(
                    f"Malformed TSF data row in {path}: expected {len(attribute_names)} attributes"
                )
            series_names.add(parts[series_name_index])
    return series_names


def collect_rds_series_name_keys(dataset_dir: Path) -> set[str]:
    import pyreadr

    series_names: set[str] = set()
    for path in sorted(dataset_dir.glob("*.rds")):
        result = pyreadr.read_r(str(path))
        if not result:
            continue
        first_object = next(iter(result.values()))
        if not hasattr(first_object, "columns") or "series_name" not in first_object.columns:
            continue
        series_names.update(str(value) for value in first_object["series_name"].dropna().unique())
    return series_names


def collect_series_name_keys(row: dict[str, Any], dataset_root: Path) -> set[str]:
    source_format = str(row.get("source_format", ""))
    if source_format == "tsf":
        source_path = dataset_root / str(row["source_files"])
        return collect_tsf_series_name_keys(source_path)
    if source_format == "rds":
        dataset_dir = dataset_root / str(row["dataset_path"])
        return collect_rds_series_name_keys(dataset_dir)
    return set()


def print_series_name_keys(rows: list[dict[str, Any]], dataset_root: Path) -> None:
    target_rows = [row for row in rows if row_has_series_name_feature(row)]
    if not target_rows:
        print("\n[series_name unique keys]")
        print("No datasets contain a series_name feature.")
        return

    print("\n[series_name unique keys]")
    for row in target_rows:
        dataset_name = str(row["dataset_name"])
        series_names = collect_series_name_keys(row, dataset_root)
        print(
            f"\n## {dataset_name} "
            f"(source_format={row['source_format']}, unique_count={len(series_names)})"
        )
        for value in sorted(series_names):
            print(value)


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    rows: list[dict[str, Any]] = []
    for tsf_path in discover_tsf_files(dataset_root):
        rows.append(
            summarize_tsf_file(
                tsf_path,
                dataset_root=dataset_root,
                max_unique_values=args.max_unique_values,
            )
        )

    if not args.skip_rds:
        for dataset_dir, rds_files in discover_rds_groups(dataset_root).items():
            rows.append(
                summarize_rds_group(
                    dataset_dir=dataset_dir,
                    rds_files=rds_files,
                    dataset_root=dataset_root,
                    max_unique_values=args.max_unique_values,
                    strict=args.strict,
                )
            )

    rows.sort(key=lambda row: (str(row["source_format"]), str(row["dataset_name"])))
    write_csv(rows, args.output.resolve())
    print(f"Wrote {len(rows)} dataset summaries to {args.output.resolve()}")
    if args.print_series_name_keys:
        print_series_name_keys(rows, dataset_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
