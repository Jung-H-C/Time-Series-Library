from __future__ import annotations

import argparse
import csv
import math
import os
import re
from pathlib import Path



# python ./calculate_spearman.py --input-glob ./mamba_100_sl96_*.csv --proxy-columns I:T --target-columns V 
EXCEL_COLUMN_PATTERN = re.compile(r"^[A-Za-z]+$")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_MPLCONFIG_DIR = Path("/tmp/tslib_matplotlib")
KNOWN_DATASET_LABELS = ("Exchange", "Traffic", "Weather", "ETTh1", "ETTh2", "ETTm1", "ETTm2", "ECL", "ILI")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Spearman correlations between proxy columns and target metric columns. "
            "The exported spearman_coefficient is multiplied by -1 because lower target metrics are better."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "One or more input CSV file paths. Relative paths are resolved under "
            f"{DEFAULT_RESULTS_DIR}. When multiple CSVs are passed, Spearman rows are combined."
        ),
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default=None,
        help=(
            "Glob pattern for input CSVs under the results directory, or an absolute/relative glob path. "
            "Example: 'autoformer_100_sl96_*_proxy_scores_*.csv'. Use either --input-csv or --input-glob."
        ),
    )
    parser.add_argument(
        "--proxy-columns",
        nargs="+",
        required=True,
        help=(
            "Proxy columns as comma-separated names/Excel letters/ranges. "
            "Examples: I,J,K,L,M,N,O,P,Q or I:T or params:synflow."
        ),
    )
    parser.add_argument(
        "--target-columns",
        nargs="+",
        required=True,
        help=(
            "Target columns as comma-separated names/Excel letters/ranges. "
            "Examples: S,T,U,V,W,X or V or ecl:weather."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Output CSV path. Relative paths are resolved under "
            f"{DEFAULT_RESULTS_DIR}. Defaults to <input stem>_spearman.csv for one input, "
            "or multi_input_spearman.csv for multiple inputs."
        ),
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=None,
        help=(
            "Output PNG path for the polygon/radar plot. Relative paths are resolved under "
            f"{DEFAULT_RESULTS_DIR}. Defaults to the output CSV path with a .png suffix."
        ),
    )
    parser.add_argument(
        "--target-labels",
        nargs="+",
        default=None,
        help=(
            "Optional labels for targets in the polygon plot. For multiple input CSVs, pass one label per CSV "
            "when each file has one target column. Defaults to dataset labels inferred from filenames."
        ),
    )
    parser.add_argument(
        "--split",
        default="proxy_train",
        help=(
            "Filter input rows by the CSV split column before computing Spearman. "
            "Use 'all' to disable split filtering. Default: proxy_train."
        ),
    )
    return parser.parse_args()


def split_column_specs(values: list[str]) -> list[str]:
    specs: list[str] = []
    for value in values:
        for spec in value.split(","):
            stripped = spec.strip()
            if stripped:
                specs.append(stripped)
    return specs


def excel_column_to_index(column: str) -> int:
    index = 0
    for char in column.upper():
        if char < "A" or char > "Z":
            raise ValueError(f"Invalid Excel column letter: {column}")
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def index_to_excel_column(index: int) -> str:
    if index < 0:
        raise ValueError(f"Column index must be non-negative: {index}")

    letters = []
    value = index + 1
    while value:
        value, remainder = divmod(value - 1, 26)
        letters.append(chr(ord("A") + remainder))
    return "".join(reversed(letters))


def resolve_header_name(spec: str, header: list[str]) -> int | None:
    exact_matches = [index for index, name in enumerate(header) if name == spec]
    if len(exact_matches) > 1:
        raise ValueError(f"Column name appears multiple times in header: {spec}")
    if exact_matches:
        return exact_matches[0]

    normalized_spec = spec.casefold()
    casefold_matches = [index for index, name in enumerate(header) if name.casefold() == normalized_spec]
    if len(casefold_matches) > 1:
        raise ValueError(f"Column name appears multiple times in header ignoring case: {spec}")
    if casefold_matches:
        return casefold_matches[0]

    return None


def resolve_column_spec(spec: str, header: list[str]) -> int:
    header_index = resolve_header_name(spec, header)
    if header_index is not None:
        return header_index

    if EXCEL_COLUMN_PATTERN.fullmatch(spec):
        index = excel_column_to_index(spec)
        if index >= len(header):
            raise ValueError(
                f"Excel column {spec} resolves to index {index}, but the CSV only has {len(header)} columns."
            )
        return index

    raise ValueError(f"Could not resolve column spec {spec!r} as a header name or Excel column letter.")


def resolve_columns(raw_specs: list[str], header: list[str]) -> list[int]:
    indices: list[int] = []
    seen: set[int] = set()

    for spec in split_column_specs(raw_specs):
        if ":" in spec:
            start_spec, end_spec = [part.strip() for part in spec.split(":", 1)]
            if not start_spec or not end_spec:
                raise ValueError(f"Invalid column range: {spec}")
            start = resolve_column_spec(start_spec, header)
            end = resolve_column_spec(end_spec, header)
            if start > end:
                raise ValueError(f"Column range must be increasing: {spec}")
            range_indices = range(start, end + 1)
        else:
            range_indices = [resolve_column_spec(spec, header)]

        for index in range_indices:
            if index not in seen:
                indices.append(index)
                seen.add(index)

    if not indices:
        raise ValueError("No columns were resolved.")
    return indices


def resolve_results_path(path: Path, default_dir: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (default_dir / path).resolve()


def resolve_input_csv_path(path: Path, default_dir: Path) -> Path:
    if path.is_absolute():
        return path.resolve()

    cwd_path = path.resolve()
    if cwd_path.is_file():
        return cwd_path

    return (default_dir / path).resolve()


def resolve_input_glob(pattern: str, default_dir: Path) -> list[Path]:
    pattern_path = Path(pattern)
    if pattern_path.is_absolute():
        matches = sorted(Path("/").glob(pattern_path.as_posix().lstrip("/")))
    else:
        matches = sorted(default_dir.glob(pattern))
        if not matches:
            matches = sorted(Path.cwd().glob(pattern))
    return [path.resolve() for path in matches if path.is_file()]


def infer_target_label_from_path(path: Path) -> str:
    stem = path.stem
    for label in KNOWN_DATASET_LABELS:
        if re.search(rf"(^|[_\-]){re.escape(label)}([_\-]|$)", stem, flags=re.IGNORECASE):
            return label
    return stem


def parse_float(value: str) -> float | None:
    stripped = value.strip()
    if not stripped:
        return None

    try:
        parsed = float(stripped)
    except ValueError:
        return None

    if not math.isfinite(parsed):
        return None
    return parsed


def rank_values(values: list[float]) -> list[float]:
    ranked = [0.0] * len(values)
    sorted_pairs = sorted((value, index) for index, value in enumerate(values))

    position = 0
    while position < len(sorted_pairs):
        next_position = position + 1
        while next_position < len(sorted_pairs) and sorted_pairs[next_position][0] == sorted_pairs[position][0]:
            next_position += 1

        average_rank = (position + 1 + next_position) / 2.0
        for _, original_index in sorted_pairs[position:next_position]:
            ranked[original_index] = average_rank
        position = next_position

    return ranked


def pearson_correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right):
        raise ValueError("Correlation inputs must have the same length.")
    if len(left) < 2:
        return None

    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = 0.0
    left_square_sum = 0.0
    right_square_sum = 0.0

    for left_value, right_value in zip(left, right):
        left_diff = left_value - left_mean
        right_diff = right_value - right_mean
        numerator += left_diff * right_diff
        left_square_sum += left_diff * left_diff
        right_square_sum += right_diff * right_diff

    denominator = math.sqrt(left_square_sum * right_square_sum)
    if denominator == 0.0:
        return None
    return numerator / denominator


def spearman_correlation(left: list[float], right: list[float]) -> float | None:
    return pearson_correlation(rank_values(left), rank_values(right))


def get_cell(row: list[str], index: int) -> str:
    if index >= len(row):
        return ""
    return row[index]


def filter_rows_by_split(header: list[str], data_rows: list[list[str]], split: str) -> list[list[str]]:
    requested_split = split.strip()
    if not requested_split:
        raise ValueError("--split must be a non-empty value or 'all'.")
    if requested_split.casefold() == "all":
        return data_rows

    split_index = resolve_header_name("split", header)
    if split_index is None:
        raise ValueError("Input CSV is missing required 'split' column. Use --split all to disable filtering.")

    filtered_rows = [
        row
        for row in data_rows
        if get_cell(row, split_index).strip() == requested_split
    ]
    if not filtered_rows:
        raise ValueError(f"No input rows matched --split {requested_split!r}.")
    return filtered_rows


def build_result_rows(
    header: list[str],
    data_rows: list[list[str]],
    proxy_indices: list[int],
    target_indices: list[int],
    *,
    input_csv: Path,
    split: str,
    target_label: str | None = None,
) -> list[dict[str, object]]:
    result_rows: list[dict[str, object]] = []

    for target_index in target_indices:
        target_name = header[target_index]
        if target_label is not None:
            target_name = target_label if len(target_indices) == 1 else f"{target_label}:{target_name}"

        for proxy_index in proxy_indices:
            proxy_values: list[float] = []
            target_values: list[float] = []

            for row in data_rows:
                proxy_value = parse_float(get_cell(row, proxy_index))
                target_value = parse_float(get_cell(row, target_index))
                if proxy_value is None or target_value is None:
                    continue
                proxy_values.append(proxy_value)
                target_values.append(target_value)

            raw_spearman = spearman_correlation(proxy_values, target_values)
            adjusted_spearman = None if raw_spearman is None else -raw_spearman

            result_rows.append(
                {
                    "input_csv": input_csv.name,
                    "split": split,
                    "target_column": index_to_excel_column(target_index),
                    "target_name": target_name,
                    "proxy_column": index_to_excel_column(proxy_index),
                    "proxy_name": header[proxy_index],
                    "num_pairs": len(proxy_values),
                    "num_input_rows": len(data_rows),
                    "skipped_rows": len(data_rows) - len(proxy_values),
                    "spearman_coefficient": adjusted_spearman,
                    "raw_spearman_coefficient": raw_spearman,
                }
            )

    return result_rows


def format_float(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.10g}"
    return value


def write_results(output_csv: Path, rows: list[dict[str, object]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "input_csv",
        "split",
        "target_column",
        "target_name",
        "proxy_column",
        "proxy_name",
        "num_pairs",
        "num_input_rows",
        "skipped_rows",
        "spearman_coefficient",
        "raw_spearman_coefficient",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_float(value) for key, value in row.items()})


def unique_values(rows: list[dict[str, object]], key: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for row in rows:
        value = str(row[key])
        if value not in seen:
            values.append(value)
            seen.add(value)
    return values


def coefficient_to_radius(coefficient: float) -> float:
    return (coefficient + 1.0) / 2.0


def write_polygon_plot(output_png: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot draw polygon plot because there are no Spearman result rows.")

    os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPLCONFIG_DIR))
    DEFAULT_MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    targets = unique_values(rows, "target_name")
    proxies = unique_values(rows, "proxy_name")
    target_angles = np.linspace(0.0, 2.0 * np.pi, len(targets), endpoint=False)
    closed_angles = np.concatenate([target_angles, target_angles[:1]])

    cmap = plt.get_cmap("tab10" if len(proxies) <= 10 else "tab20")
    proxy_colors = {proxy: cmap(index % cmap.N) for index, proxy in enumerate(proxies)}

    figure_size = max(7.5, min(12.0, 6.5 + len(targets) * 0.25 + len(proxies) * 0.08))
    fig, ax = plt.subplots(figsize=(figure_size, figure_size), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)

    coefficient_by_proxy_target: dict[tuple[str, str], float] = {}
    for row in rows:
        coefficient = row["spearman_coefficient"]
        if coefficient is None:
            continue
        coefficient_by_proxy_target[(str(row["proxy_name"]), str(row["target_name"]))] = float(coefficient)

    for proxy in proxies:
        radii = [
            coefficient_to_radius(coefficient_by_proxy_target[(proxy, target)])
            if (proxy, target) in coefficient_by_proxy_target
            else float("nan")
            for target in targets
        ]
        closed_radii = radii + radii[:1]
        has_complete_polygon = all(math.isfinite(radius) for radius in closed_radii)

        ax.plot(
            closed_angles,
            closed_radii,
            label=proxy,
            color=proxy_colors[proxy],
            linewidth=1.8,
            marker="o",
            markersize=4.2,
            alpha=0.9,
        )
        if has_complete_polygon:
            ax.fill(closed_angles, closed_radii, color=proxy_colors[proxy], alpha=0.08)

    ax.set_xticks(target_angles)
    ax.set_xticklabels(targets)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["-1.0", "-0.5", "0.0", "0.5", "1.0"])
    ax.set_rlabel_position(90)
    ax.grid(color="#dddddd", linewidth=0.8)
    ax.spines["polar"].set_color("#777777")
    ax.spines["polar"].set_linewidth(0.9)
    ax.set_title("Proxy-target Spearman polygon plot", pad=24)
    ax.legend(title="proxy_name", bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False)
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if bool(args.input_csv) == bool(args.input_glob):
        raise ValueError("Use exactly one of --input-csv or --input-glob.")

    if args.input_glob:
        input_csvs = resolve_input_glob(args.input_glob, DEFAULT_RESULTS_DIR)
        if not input_csvs:
            raise FileNotFoundError(f"No input CSVs matched --input-glob pattern: {args.input_glob}")
    else:
        input_csvs = [resolve_input_csv_path(input_path, DEFAULT_RESULTS_DIR) for input_path in args.input_csv]

    for input_csv in input_csvs:
        if not input_csv.is_file():
            raise FileNotFoundError(f"Input CSV does not exist: {input_csv}")

    target_labels = args.target_labels
    if target_labels is not None and len(target_labels) != len(input_csvs):
        raise ValueError(
            f"--target-labels must provide exactly one label per input CSV; "
            f"got {len(target_labels)} labels for {len(input_csvs)} input CSVs."
        )

    output_csv = (
        resolve_results_path(args.output_csv, DEFAULT_RESULTS_DIR)
        if args.output_csv is not None
        else DEFAULT_RESULTS_DIR
        / (f"{input_csvs[0].stem}_spearman.csv" if len(input_csvs) == 1 else "multi_input_spearman.csv")
    )
    output_png = (
        resolve_results_path(args.output_png, DEFAULT_RESULTS_DIR)
        if args.output_png is not None
        else output_csv.with_suffix(".png")
    )

    result_rows: list[dict[str, object]] = []
    for input_index, input_csv in enumerate(input_csvs):
        with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ValueError(f"Input CSV is empty: {input_csv}") from exc
            data_rows = list(reader)

        filtered_rows = filter_rows_by_split(header, data_rows, args.split)
        proxy_indices = resolve_columns(args.proxy_columns, header)
        target_indices = resolve_columns(args.target_columns, header)
        inferred_label = target_labels[input_index] if target_labels is not None else infer_target_label_from_path(input_csv)
        target_label = inferred_label if len(input_csvs) > 1 or target_labels is not None else None
        result_rows.extend(
            build_result_rows(
                header,
                filtered_rows,
                proxy_indices,
                target_indices,
                input_csv=input_csv,
                split=args.split.strip(),
                target_label=target_label,
            )
        )

    write_results(output_csv, result_rows)
    write_polygon_plot(output_png, result_rows)

    print(
        f"Wrote {len(result_rows)} proxy-target Spearman rows from {len(input_csvs)} input CSV(s) "
        f"to {output_csv} and plot to {output_png}. "
        f"split={args.split.strip()!r}. spearman_coefficient is -1 * raw_spearman_coefficient."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
