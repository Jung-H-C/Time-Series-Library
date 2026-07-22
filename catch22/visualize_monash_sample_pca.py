from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tslib_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycatch22
import pyreadr


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from summarize_monash_datasets import read_tsf_header  # noqa: E402


META_COLUMNS = [
    "dataset",
    "source_format",
    "source_path",
    "sample_rank",
    "sample_index",
    "series_name",
    "series_length_original",
    "series_length_used",
    "downsampled",
    "value_source",
]


@dataclass(frozen=True)
class TimeSeriesSample:
    dataset: str
    source_format: str
    source_path: Path
    sample_rank: int
    sample_index: int
    series_name: str
    values: np.ndarray
    value_source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample time series from each Monash dataset, extract catch22 features, "
            "project them into 2D with PCA, and save a visualization."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "Monash_Dataset",
        help="Root directory containing Monash .tsf and .rds datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "catch22" / "monash_sample_pca",
        help="Directory for output CSV and PNG files.",
    )
    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=5,
        help="Number of time-series samples per dataset. If fewer exist, all are used.",
    )
    parser.add_argument("--seed", type=int, default=20260703, help="Random seed for sampling.")
    parser.add_argument(
        "--max-series-length",
        type=int,
        default=50000,
        help=(
            "Maximum length passed to pycatch22 after uniform downsampling. "
            "Use 0 or a negative value to disable downsampling."
        ),
    )
    parser.add_argument(
        "--rds-value-column",
        choices=("mean", "sum", "min", "max"),
        default="mean",
        help=(
            "Value series to extract from per-unit .rds files. "
            "`mean` uses sum/count when both columns are available."
        ),
    )
    parser.add_argument("--dpi", type=int, default=220, help="Output figure DPI.")
    parser.add_argument(
        "--legend",
        action="store_true",
        help="Add a full dataset legend to the PCA plot. This can be crowded for all Monash datasets.",
    )
    return parser.parse_args()


def resolve_path(path: Path, base: Path = REPO_ROOT) -> Path:
    return path if path.is_absolute() else base / path


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def discover_tsf_files(dataset_root: Path) -> list[Path]:
    return sorted(dataset_root.rglob("*.tsf"))


def discover_rds_groups(dataset_root: Path) -> dict[Path, list[Path]]:
    groups: dict[Path, list[Path]] = {}
    for path in sorted(dataset_root.rglob("*.rds")):
        groups.setdefault(path.parent, []).append(path)
    return groups


def reservoir_sample_tsf_lines(
    path: Path,
    n_samples: int,
    rng: np.random.Generator,
) -> list[tuple[int, str]]:
    reservoir: list[tuple[int, str]] = []
    in_data = False
    data_index = 0

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if not in_data:
                if line.lower() == "@data":
                    in_data = True
                continue

            data_index += 1
            if len(reservoir) < n_samples:
                reservoir.append((data_index, line))
            else:
                replace_at = int(rng.integers(0, data_index))
                if replace_at < n_samples:
                    reservoir[replace_at] = (data_index, line)

    return sorted(reservoir, key=lambda item: item[0])


def parse_tsf_values(values_text: str) -> np.ndarray:
    values = []
    for token in values_text.split(","):
        token = token.strip()
        values.append(np.nan if token == "?" or token == "" else float(token))
    return np.asarray(values, dtype=np.float64)


def collect_tsf_samples(
    path: Path,
    n_samples: int,
    rng: np.random.Generator,
) -> list[TimeSeriesSample]:
    _, attributes = read_tsf_header(path)
    attribute_names = [name for name, _ in attributes]
    sampled_lines = reservoir_sample_tsf_lines(path, n_samples, rng)
    samples: list[TimeSeriesSample] = []

    for sample_rank, (data_index, line) in enumerate(sampled_lines, start=1):
        parts = line.split(":", len(attribute_names))
        if len(parts) != len(attribute_names) + 1:
            raise ValueError(f"Malformed TSF row in {path}: {line[:200]}")

        attrs = dict(zip(attribute_names, parts[:-1]))
        series_name = attrs.get("series_name", f"series_{data_index}")
        samples.append(
            TimeSeriesSample(
                dataset=path.parent.name,
                source_format="tsf",
                source_path=path,
                sample_rank=sample_rank,
                sample_index=data_index,
                series_name=series_name,
                values=parse_tsf_values(parts[-1]),
                value_source="value",
            )
        )

    return samples


def sample_without_replacement(length: int, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    if length <= 0:
        return np.asarray([], dtype=np.int64)
    count = min(length, n_samples)
    return np.sort(rng.choice(length, size=count, replace=False))


def rds_value_series(df: pd.DataFrame, value_column: str) -> tuple[np.ndarray, str]:
    if "utc" in df.columns:
        df = df.sort_values("utc")

    if value_column == "mean" and {"sum", "count"}.issubset(df.columns):
        count = pd.to_numeric(df["count"], errors="coerce").replace(0, np.nan)
        values = pd.to_numeric(df["sum"], errors="coerce") / count
        return values.to_numpy(dtype=np.float64), "sum/count"

    if value_column not in df.columns:
        raise ValueError(f"RDS dataframe does not contain `{value_column}` column.")
    values = pd.to_numeric(df[value_column], errors="coerce")
    return values.to_numpy(dtype=np.float64), value_column


def collect_rds_samples(
    dataset_dir: Path,
    rds_files: list[Path],
    n_samples: int,
    rng: np.random.Generator,
    value_column: str,
) -> list[TimeSeriesSample]:
    selected_indices = sample_without_replacement(len(rds_files), n_samples, rng)
    samples: list[TimeSeriesSample] = []

    for sample_rank, file_index in enumerate(selected_indices, start=1):
        path = rds_files[int(file_index)]
        result = pyreadr.read_r(str(path))
        if not result:
            raise ValueError(f"pyreadr returned no objects for {path}")
        df = next(iter(result.values()))
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"RDS object is not a DataFrame: {path}")

        values, value_source = rds_value_series(df, value_column)
        metric = ""
        if "metric" in df.columns and len(df["metric"].dropna()) > 0:
            metric = f":{df['metric'].dropna().astype(str).iloc[0]}"
        samples.append(
            TimeSeriesSample(
                dataset=dataset_dir.name,
                source_format="rds",
                source_path=path,
                sample_rank=sample_rank,
                sample_index=int(file_index) + 1,
                series_name=f"{path.stem}{metric}",
                values=values,
                value_source=value_source,
            )
        )

    return samples


def interpolate_missing(values: np.ndarray) -> np.ndarray:
    series = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(series)
    if finite.all():
        return series
    if finite.sum() == 0:
        raise ValueError("time series contains no finite values")
    if finite.sum() == 1:
        return np.full_like(series, float(series[finite][0]), dtype=np.float64)
    indices = np.arange(len(series), dtype=np.float64)
    return np.interp(indices, indices[finite], series[finite])


def maybe_downsample(values: np.ndarray, max_length: int) -> tuple[np.ndarray, bool]:
    if max_length <= 0 or len(values) <= max_length:
        return values, False
    indices = np.linspace(0, len(values) - 1, num=max_length, dtype=np.int64)
    return values[indices], True


def prepare_series(values: np.ndarray, max_length: int) -> tuple[np.ndarray, bool]:
    if len(values) < 3:
        raise ValueError(f"time series is too short for catch22: length={len(values)}")
    cleaned = interpolate_missing(values)
    sampled, downsampled = maybe_downsample(cleaned, max_length=max_length)
    if len(sampled) < 3:
        raise ValueError(f"downsampled time series is too short for catch22: length={len(sampled)}")
    return sampled.astype(np.float64, copy=False), downsampled


def catch22_features(values: np.ndarray) -> tuple[list[str], np.ndarray]:
    result = pycatch22.catch22_all(values, catch24=False, short_names=True)
    return list(result["short_names"]), np.asarray(result["values"], dtype=np.float64)


def collect_all_samples(args: argparse.Namespace) -> tuple[list[TimeSeriesSample], list[dict[str, object]]]:
    rng = np.random.default_rng(args.seed)
    dataset_root = resolve_path(args.dataset_root)
    samples: list[TimeSeriesSample] = []
    errors: list[dict[str, object]] = []

    for tsf_path in discover_tsf_files(dataset_root):
        try:
            samples.extend(collect_tsf_samples(tsf_path, args.samples_per_dataset, rng))
        except Exception as exc:
            errors.append({"dataset": tsf_path.parent.name, "source_path": str(tsf_path), "error": str(exc)})

    for dataset_dir, rds_files in discover_rds_groups(dataset_root).items():
        try:
            samples.extend(
                collect_rds_samples(
                    dataset_dir,
                    sorted(rds_files),
                    args.samples_per_dataset,
                    rng,
                    args.rds_value_column,
                )
            )
        except Exception as exc:
            errors.append({"dataset": dataset_dir.name, "source_path": str(dataset_dir), "error": str(exc)})

    return samples, errors


def build_feature_frame(
    samples: list[TimeSeriesSample],
    dataset_root: Path,
    max_series_length: int,
) -> tuple[pd.DataFrame, list[str], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    feature_names: list[str] | None = None

    for sample in samples:
        try:
            prepared, downsampled = prepare_series(sample.values, max_length=max_series_length)
            short_names, features = catch22_features(prepared)
        except Exception as exc:
            errors.append(
                {
                    "dataset": sample.dataset,
                    "source_format": sample.source_format,
                    "source_path": relative_path(sample.source_path, dataset_root),
                    "sample_rank": sample.sample_rank,
                    "sample_index": sample.sample_index,
                    "series_name": sample.series_name,
                    "error": str(exc),
                }
            )
            continue

        if feature_names is None:
            feature_names = short_names
        elif feature_names != short_names:
            raise RuntimeError("catch22 feature order changed across samples")

        row: dict[str, object] = {
            "dataset": sample.dataset,
            "source_format": sample.source_format,
            "source_path": relative_path(sample.source_path, dataset_root),
            "sample_rank": sample.sample_rank,
            "sample_index": sample.sample_index,
            "series_name": sample.series_name,
            "series_length_original": len(sample.values),
            "series_length_used": len(prepared),
            "downsampled": downsampled,
            "value_source": sample.value_source,
        }
        for name, value in zip(short_names, features):
            row[name] = float(value)
        rows.append(row)

    if feature_names is None:
        raise RuntimeError("No catch22 feature rows were collected.")
    return pd.DataFrame(rows), feature_names, errors


def numeric_feature_frame(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    features = df[feature_names].apply(pd.to_numeric, errors="coerce")
    return features.replace([np.inf, -np.inf], np.nan)


def zscore_and_impute(features: pd.DataFrame) -> pd.DataFrame:
    imputed = features.copy()
    means = imputed.mean(axis=0, skipna=True)
    imputed = imputed.fillna(means).fillna(0.0)
    stds = imputed.std(axis=0, ddof=0).replace(0.0, np.nan)
    return ((imputed - imputed.mean(axis=0)) / stds).fillna(0.0)


def add_pca_columns(sample_df: pd.DataFrame, z_features: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    matrix = z_features[feature_names].to_numpy(dtype=float)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    n_components = min(2, vt.shape[0])

    coords = np.zeros((len(centered), 2), dtype=float)
    explained = np.zeros(2, dtype=float)
    if n_components > 0:
        components = vt[:n_components].T
        coords[:, :n_components] = centered @ components
        variances = singular_values**2
        total_variance = variances.sum()
        if total_variance > 0:
            explained[:n_components] = variances[:n_components] / total_variance

    pca_df = sample_df[META_COLUMNS].copy()
    pca_df["pc1"] = coords[:, 0]
    pca_df["pc2"] = coords[:, 1]
    pca_df["pc1_explained_ratio"] = float(explained[0])
    pca_df["pc2_explained_ratio"] = float(explained[1])
    return pca_df


def save_pca_plot(pca_df: pd.DataFrame, output_path: Path, dpi: int, legend: bool) -> None:
    dataset_names = list(dict.fromkeys(pca_df["dataset"].tolist()))
    cmap = plt.get_cmap("nipy_spectral")
    colors = {
        dataset: cmap(index / max(1, len(dataset_names) - 1))
        for index, dataset in enumerate(dataset_names)
    }

    fig, ax = plt.subplots(figsize=(14, 10), constrained_layout=True)
    for dataset in dataset_names:
        group = pca_df[pca_df["dataset"] == dataset]
        ax.scatter(
            group["pc1"],
            group["pc2"],
            s=42,
            alpha=0.78,
            color=colors[dataset],
            label=dataset,
            edgecolors="white",
            linewidths=0.35,
        )
        ax.text(
            float(group["pc1"].mean()),
            float(group["pc2"].mean()),
            dataset,
            fontsize=6,
            color=colors[dataset],
            weight="bold",
        )

    pc1_ratio = float(pca_df["pc1_explained_ratio"].iloc[0])
    pc2_ratio = float(pca_df["pc2_explained_ratio"].iloc[0])
    ax.axhline(0.0, color="#bbbbbb", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="#bbbbbb", linewidth=0.8, zorder=0)
    ax.set_xlabel(f"PC1 ({pc1_ratio:.2%})")
    ax.set_ylabel(f"PC2 ({pc2_ratio:.2%})")
    ax.set_title("Monash dataset samples in catch22 feature space")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.55)
    if legend:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6, ncol=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_error_csv(errors: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "source_format", "source_path", "sample_rank", "sample_index", "series_name", "error"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for error in errors:
            writer.writerow({key: error.get(key, "") for key in fieldnames})


def main() -> int:
    args = parse_args()
    dataset_root = resolve_path(args.dataset_root).resolve()
    output_dir = resolve_path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, collection_errors = collect_all_samples(args)
    feature_df, feature_names, feature_errors = build_feature_frame(
        samples,
        dataset_root=dataset_root,
        max_series_length=args.max_series_length,
    )
    features = numeric_feature_frame(feature_df, feature_names)
    z_features = zscore_and_impute(features)
    z_features.columns = feature_names
    pca_df = add_pca_columns(feature_df, z_features, feature_names)

    feature_path = output_dir / "sample_catch22_features.csv"
    z_feature_path = output_dir / "sample_catch22_features_zscored.csv"
    pca_path = output_dir / "sample_catch22_pca_2d.csv"
    plot_path = output_dir / "sample_catch22_pca_2d.png"
    error_path = output_dir / "skipped_samples.csv"

    feature_df.to_csv(feature_path, index=False, float_format="%.9f")
    pd.concat([feature_df[META_COLUMNS], z_features], axis=1).to_csv(
        z_feature_path,
        index=False,
        float_format="%.9f",
    )
    pca_df.to_csv(pca_path, index=False, float_format="%.9f")
    save_pca_plot(pca_df, plot_path, dpi=args.dpi, legend=args.legend)
    write_error_csv([*collection_errors, *feature_errors], error_path)

    print(f"Collected {len(samples)} raw samples from {dataset_root}")
    print(f"Computed catch22 features for {len(feature_df)} samples across {feature_df['dataset'].nunique()} datasets")
    print(f"Skipped samples/errors: {len(collection_errors) + len(feature_errors)}")
    print(f"Feature CSV: {feature_path}")
    print(f"Z-scored feature CSV: {z_feature_path}")
    print(f"PCA CSV: {pca_path}")
    print(f"PCA plot: {plot_path}")
    print(f"Error CSV: {error_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
