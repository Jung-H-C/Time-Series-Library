from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset"


@dataclass(frozen=True)
class SeriesRecord:
    series_index: int
    series_id: str
    attributes: dict[str, str]
    values: np.ndarray


@dataclass(frozen=True)
class TsfDataset:
    path: Path
    metadata: dict[str, str]
    attribute_names: list[str]
    records: list[SeriesRecord]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize one or more univariate time series slices from a TSF dataset.",
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset name (for example: sunspot, weather_tsf, m1_monthly) or a direct .tsf path.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory containing TSF datasets. Defaults to Time-Series-Library/dataset.",
    )
    parser.add_argument(
        "--series-index",
        type=int,
        default=None,
        help="Specific series index to plot. If omitted, the script plots the first few series.",
    )
    parser.add_argument(
        "--num-series",
        type=int,
        default=4,
        help="Number of series to plot when --series-index is not provided.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Randomly select 10 series to plot. Ignored if --series-index is provided.",
    )
    parser.add_argument(
        "--random-count",
        type=int,
        default=10,
        help="Number of random series to plot when --random is provided.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible --random selections.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index of the visible slice. Negative values count from the end of each series.",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=300,
        help="Number of points to visualize from each selected series.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output figure DPI.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to <tsf_file_stem>_preview.png next to the dataset file.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the matplotlib window in addition to saving the figure.",
    )
    parser.add_argument(
        "--terminal",
        action="store_true",
        help="Print an ASCII plot in the terminal. Skips image saving unless --output or --show is also provided.",
    )
    parser.add_argument(
        "--terminal-width",
        type=int,
        default=80,
        help="Width of each terminal plot in characters.",
    )
    parser.add_argument(
        "--terminal-height",
        type=int,
        default=15,
        help="Height of each terminal plot in rows.",
    )
    return parser.parse_args()


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _tokenize_name(value: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", value.lower()) if token]


def _score_candidate(path: Path, dataset_name: str, dataset_root: Path) -> tuple[int, int, str]:
    target = _normalize_name(dataset_name)
    tokens = _tokenize_name(dataset_name)
    relative = path.relative_to(dataset_root).as_posix().lower()
    normalized_relative = _normalize_name(relative)
    normalized_stem = _normalize_name(path.stem)
    normalized_parent = _normalize_name(path.parent.name)

    score = 0
    if normalized_parent == target:
        score += 100
    if normalized_stem == target:
        score += 95
    if target and target in normalized_stem:
        score += 70
    if target and target in normalized_parent:
        score += 60
    if target and target in normalized_relative:
        score += 30
    if tokens and all(token in relative for token in tokens):
        score += 20
    if "_downloads" not in path.parts:
        score += 5

    return score, len(relative), relative


def resolve_tsf_path(dataset: str, dataset_root: Path) -> Path:
    dataset_root = dataset_root.resolve()
    dataset_path = Path(dataset).expanduser()

    direct_candidates = [
        dataset_path,
        dataset_root / dataset,
    ]
    for candidate in direct_candidates:
        candidate = candidate.resolve()
        if candidate.is_file() and candidate.suffix.lower() == ".tsf":
            return candidate
        if candidate.is_dir():
            tsf_files = sorted(candidate.rglob("*.tsf"))
            if len(tsf_files) == 1:
                return tsf_files[0].resolve()

    all_tsf_files = sorted(dataset_root.rglob("*.tsf"))
    if not all_tsf_files:
        raise FileNotFoundError(f"No .tsf files were found under {dataset_root}")

    scored_candidates: list[tuple[tuple[int, int, str], Path]] = []
    for path in all_tsf_files:
        score_info = _score_candidate(path, dataset, dataset_root)
        if score_info[0] > 0:
            scored_candidates.append((score_info, path.resolve()))

    if not scored_candidates:
        raise FileNotFoundError(
            f"Could not resolve dataset `{dataset}` under {dataset_root}. "
            "Pass a more specific dataset name or a direct .tsf path."
        )

    scored_candidates.sort(key=lambda item: (-item[0][0], item[0][1], item[0][2]))
    return scored_candidates[0][1]


def read_tsf_header(filepath: Path) -> tuple[dict[str, str], list[str]]:
    metadata: dict[str, str] = {}
    attribute_names: list[str] = []

    with filepath.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            lowered = line.lower()
            if lowered == "@data":
                if not attribute_names:
                    raise ValueError(f"TSF file is missing `@attribute` definitions: {filepath}")
                return metadata, attribute_names

            if lowered.startswith("@attribute"):
                parts = re.split(r"\s+", line, maxsplit=2)
                if len(parts) < 3:
                    raise ValueError(f"Malformed TSF attribute line: `{line}`")
                attribute_names.append(parts[1])
                continue

            for key in ("@relation", "@frequency", "@horizon", "@missing", "@equallength"):
                if lowered.startswith(key):
                    metadata[key[1:]] = line.split(None, 1)[1].strip() if " " in line else ""
                    break

    raise ValueError(f"TSF file is missing an `@data` section: {filepath}")


def iter_tsf_records(filepath: Path, attribute_names: list[str]) -> list[SeriesRecord]:
    records: list[SeriesRecord] = []
    in_data_section = False

    with filepath.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            lowered = line.lower()
            if lowered == "@data":
                in_data_section = True
                continue
            if not in_data_section:
                continue

            parts = line.split(":", len(attribute_names))
            if len(parts) != len(attribute_names) + 1:
                raise ValueError(
                    f"Malformed TSF record in {filepath}. Expected {len(attribute_names)} attributes "
                    f"before the value field, got line: `{line[:200]}`"
                )

            attributes = {name: value for name, value in zip(attribute_names, parts[:-1])}
            values = np.asarray(
                [np.nan if token == "?" else float(token) for token in parts[-1].split(",")],
                dtype=np.float32,
            )
            series_id = attributes.get("series_name") or attributes.get(attribute_names[0]) or f"series_{len(records)}"
            records.append(
                SeriesRecord(
                    series_index=len(records),
                    series_id=series_id,
                    attributes=attributes,
                    values=values,
                )
            )

    if not records:
        raise ValueError(f"No time series records were found in {filepath}")
    return records


def load_tsf_dataset(filepath: Path) -> TsfDataset:
    metadata, attribute_names = read_tsf_header(filepath)
    records = iter_tsf_records(filepath, attribute_names)
    return TsfDataset(
        path=filepath,
        metadata=metadata,
        attribute_names=attribute_names,
        records=records,
    )


def select_records(
    dataset: TsfDataset,
    series_index: int | None,
    num_series: int,
    random: bool,
    random_count: int,
    random_seed: int | None,
) -> list[SeriesRecord]:
    if series_index is not None:
        if series_index < 0 or series_index >= len(dataset.records):
            raise IndexError(
                f"--series-index must be in [0, {len(dataset.records) - 1}], got {series_index}."
            )
        return [dataset.records[series_index]]

    if random:
        if random_count <= 0:
            raise ValueError(f"--random-count must be positive, got {random_count}.")
        rng = np.random.default_rng(random_seed)
        count = min(random_count, len(dataset.records))
        indices = rng.choice(len(dataset.records), size=count, replace=False)
        return [dataset.records[int(index)] for index in indices]

    if num_series <= 0:
        raise ValueError(f"--num-series must be positive, got {num_series}.")
    return dataset.records[: min(num_series, len(dataset.records))]


def resolve_slice_bounds(series_length: int, start: int, length: int) -> tuple[int, int]:
    if length <= 0:
        raise ValueError(f"--length must be positive, got {length}.")

    resolved_start = start if start >= 0 else max(0, series_length + start)
    if resolved_start >= series_length:
        raise ValueError(
            f"Requested start index {resolved_start} is outside a series of length {series_length}."
        )

    resolved_end = min(series_length, resolved_start + length)
    return resolved_start, resolved_end


def truncate_label(value: str, limit: int = 80) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def summarize_dataset(dataset: TsfDataset) -> dict[str, object]:
    lengths = np.asarray([len(record.values) for record in dataset.records], dtype=np.int64)
    nan_counts = np.asarray([int(np.isnan(record.values).sum()) for record in dataset.records], dtype=np.int64)
    return {
        "num_series": len(dataset.records),
        "min_length": int(lengths.min()),
        "median_length": float(np.median(lengths)),
        "max_length": int(lengths.max()),
        "total_nan_values": int(nan_counts.sum()),
    }


def downsample_for_terminal(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        raise ValueError(f"--terminal-width must be greater than 1, got {width}.")
    if len(values) <= width:
        return values.astype(np.float64, copy=False)

    edges = np.linspace(0, len(values), num=width + 1, dtype=np.int64)
    samples = np.empty(width, dtype=np.float64)
    for column, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        window = values[start:max(start + 1, end)]
        finite_window = window[np.isfinite(window)]
        samples[column] = np.nan if len(finite_window) == 0 else float(np.mean(finite_window))
    return samples


def render_terminal_plot(values: np.ndarray, width: int, height: int) -> str:
    if height <= 1:
        raise ValueError(f"--terminal-height must be greater than 1, got {height}.")

    samples = downsample_for_terminal(values, width)
    finite_samples = samples[np.isfinite(samples)]
    if len(finite_samples) == 0:
        return "\n".join([" " * width for _ in range(height)])

    min_value = float(finite_samples.min())
    max_value = float(finite_samples.max())
    if min_value == max_value:
        rows = np.full(len(samples), height // 2, dtype=np.int64)
    else:
        rows = np.full(len(samples), -1, dtype=np.int64)
        finite_mask = np.isfinite(samples)
        scaled = (samples[finite_mask] - min_value) / (max_value - min_value)
        rows[finite_mask] = np.rint((1.0 - scaled) * (height - 1)).astype(np.int64)

    canvas = [[" " for _ in range(len(samples))] for _ in range(height)]
    for column, row in enumerate(rows):
        if np.isfinite(samples[column]):
            canvas[int(np.clip(row, 0, height - 1))][column] = "*"

    lines: list[str] = []
    for row_index, row_values in enumerate(canvas):
        if height == 1:
            label_value = max_value
        else:
            label_value = max_value - (max_value - min_value) * row_index / (height - 1)
        lines.append(f"{label_value:>10.4g} |{''.join(row_values)}")
    lines.append(f"{'':>10} +{'-' * len(samples)}")
    return "\n".join(lines)


def print_terminal_records(
    dataset_name: str,
    dataset: TsfDataset,
    selected_records: list[SeriesRecord],
    start: int,
    length: int,
    width: int,
    height: int,
) -> list[tuple[SeriesRecord, int, int]]:
    relation = dataset.metadata.get("relation", dataset.path.stem)
    frequency = dataset.metadata.get("frequency", "unknown")
    plotted_windows: list[tuple[SeriesRecord, int, int]] = []

    print(f"{dataset_name} | relation={relation} | frequency={frequency}")
    for record in selected_records:
        slice_start, slice_end = resolve_slice_bounds(len(record.values), start, length)
        window = record.values[slice_start:slice_end]
        start_timestamp = record.attributes.get("start_timestamp")
        subtitle = (
            f"[series_index={record.series_index}] id={truncate_label(str(record.series_id), 55)} | "
            f"slice=[{slice_start}:{slice_end}) | total_len={len(record.values)}"
        )
        if start_timestamp:
            subtitle += f" | start={start_timestamp}"

        print()
        print(subtitle)
        print(render_terminal_plot(window, width=width, height=height))
        plotted_windows.append((record, slice_start, slice_end))

    return plotted_windows


def plot_records(
    dataset_name: str,
    dataset: TsfDataset,
    selected_records: list[SeriesRecord],
    start: int,
    length: int,
    output_path: Path,
    dpi: int,
    show: bool,
) -> list[tuple[SeriesRecord, int, int]]:
    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plotted_windows: list[tuple[SeriesRecord, int, int]] = []
    fig, axes = plt.subplots(
        nrows=len(selected_records),
        ncols=1,
        figsize=(14, max(3.5 * len(selected_records), 4.5)),
        squeeze=False,
    )
    axes_flat = axes[:, 0]

    relation = dataset.metadata.get("relation", dataset.path.stem)
    frequency = dataset.metadata.get("frequency", "unknown")
    fig.suptitle(f"{dataset_name} | relation={relation} | frequency={frequency}", fontsize=14)

    for axis, record in zip(axes_flat, selected_records):
        slice_start, slice_end = resolve_slice_bounds(len(record.values), start, length)
        window = record.values[slice_start:slice_end]
        x_axis = np.arange(slice_start, slice_end)

        axis.plot(x_axis, window, color="#1f77b4", linewidth=1.7)
        axis.set_xlim(slice_start, max(slice_start + 1, slice_end - 1))
        axis.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
        axis.set_ylabel("value")

        start_timestamp = record.attributes.get("start_timestamp")
        subtitle = (
            f"[series_index={record.series_index}] id={truncate_label(str(record.series_id), 55)} | "
            f"slice=[{slice_start}:{slice_end}) | total_len={len(record.values)}"
        )
        if start_timestamp:
            subtitle += f" | start={start_timestamp}"
        axis.set_title(subtitle, fontsize=10)
        plotted_windows.append((record, slice_start, slice_end))

    axes_flat[-1].set_xlabel("time index")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
    return plotted_windows


def default_output_path(tsf_path: Path) -> Path:
    return tsf_path.with_name(f"{tsf_path.stem}_preview.png")


def main() -> int:
    args = parse_args()

    tsf_path = resolve_tsf_path(args.dataset, args.dataset_root)
    dataset = load_tsf_dataset(tsf_path)
    selected_records = select_records(
        dataset,
        args.series_index,
        args.num_series,
        args.random,
        args.random_count,
        args.random_seed,
    )
    should_save_image = not args.terminal or args.output is not None or args.show
    output_path = args.output.resolve() if args.output is not None else default_output_path(tsf_path)

    stats = summarize_dataset(dataset)
    if args.terminal:
        plotted_windows = print_terminal_records(
            dataset_name=args.dataset,
            dataset=dataset,
            selected_records=selected_records,
            start=args.start,
            length=args.length,
            width=args.terminal_width,
            height=args.terminal_height,
        )
    else:
        plotted_windows = []

    if should_save_image:
        plotted_windows = plot_records(
            dataset_name=args.dataset,
            dataset=dataset,
            selected_records=selected_records,
            start=args.start,
            length=args.length,
            output_path=output_path,
            dpi=args.dpi,
            show=args.show,
        )

    print(f"Resolved TSF path: {tsf_path}")
    print(f"Frequency: {dataset.metadata.get('frequency', 'unknown')}")
    print(f"Total series: {stats['num_series']}")
    print(
        "Series length stats: "
        f"min={stats['min_length']} median={stats['median_length']} max={stats['max_length']}"
    )
    print(f"Total NaN values: {stats['total_nan_values']}")
    for record, slice_start, slice_end in plotted_windows:
        print(
            f"Plotted series_index={record.series_index} "
            f"series_id={record.series_id} slice=[{slice_start}:{slice_end}) length={len(record.values)}"
        )
    if should_save_image:
        print(f"Saved figure to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
