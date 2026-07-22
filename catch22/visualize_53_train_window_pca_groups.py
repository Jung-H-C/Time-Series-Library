#!/usr/bin/env python3
"""Create one global catch22 PCA cluster plot for 53 datasets.

The default registry consists of the six benchmark datasets used by the local
catch22 scripts, the 18 runnable Monash datasets, and the 29 runnable TIME
datasets.  Twenty windows are sampled without replacement from each
Time-Series-Library training split.  Input length is 96 except for illness,
which uses 36.  For multivariate windows, catch22 is
computed per channel and pooled feature-wise, producing one 22-dimensional
vector per window.

All 1,060 raw catch22 vectors are feature-wise z-score normalized together,
then one shared 2D PCA is fitted.  The plot shows all 20 sample coordinates per
dataset and a circle centered at their mean.  Each radius is the maximum
distance from the center, so every circle contains all 20 dataset samples.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import csv
from dataclasses import dataclass
import gc
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tslib_matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
import pycatch22


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_provider.data_loader import (  # noqa: E402
    Dataset_Custom,
    Dataset_ETT_hour,
    Dataset_MultiSeriesForecast,
)


EXPECTED_DATASET_COUNT = 53
EXPECTED_FAMILY_COUNTS = {"Benchmark": 6, "Monash": 18, "TIME": 29}

META_COLUMNS = [
    "group_id",
    "group_rank",
    "dataset",
    "family",
    "source_format",
    "source_path",
    "split",
    "sample_rank",
    "window_index",
    "series_index",
    "window_start",
    "seq_len",
    "n_channels",
    "pooling",
    "scale",
]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    family: str
    loader: str
    root_path: Path
    data_path: str
    source_format: str
    enc_in: int
    target: str = "OT"
    freq: str = "h"

    @property
    def source_path(self) -> str:
        return (self.root_path / self.data_path).as_posix()


def benchmark_specs(repo_root: Path) -> list[DatasetSpec]:
    """Return the same six benchmark units used by existing catch22 scripts."""
    definitions = (
        ("electricity", "custom", "dataset/electricity", "electricity.csv", 321),
        ("ETT-small", "ETT_hour", "dataset/ETT-small", "ETTh1.csv", 7),
        ("exchange_rate", "custom", "dataset/exchange_rate", "exchange_rate.csv", 8),
        ("illness", "custom", "dataset/illness", "national_illness.csv", 7),
        ("traffic", "custom", "dataset/traffic", "traffic.csv", 862),
        ("weather", "custom", "dataset/weather", "weather.csv", 21),
    )
    return [
        DatasetSpec(
            name=name,
            family="Benchmark",
            loader=loader,
            root_path=repo_root / root_path,
            data_path=data_path,
            source_format="csv",
            enc_in=enc_in,
        )
        for name, loader, root_path, data_path, enc_in in definitions
    ]


def read_summary_specs(
    summary_path: Path,
    dataset_root: Path,
    family: str,
) -> list[DatasetSpec]:
    """Read the curated horizon-96 rows from a Monash or TIME summary."""
    if not summary_path.is_file():
        raise FileNotFoundError(f"Dataset summary does not exist: {summary_path}")

    specs: list[DatasetSpec] = []
    with summary_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset_name", "source_format", "source_files", "channel_count", "horizon"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{summary_path} is missing columns: {sorted(missing)}")

        for row in reader:
            horizon = float(str(row["horizon"]).strip())
            if not np.isfinite(horizon) or horizon != 96:
                raise ValueError(
                    "Curated summary contains a non-96 horizon for "
                    f"{row['dataset_name']!r}: {horizon}"
                )
            source_format = str(row["source_format"]).strip().lower()
            if source_format not in {"tsf", "rds", "arrow"}:
                raise ValueError(
                    f"Unsupported source format for {row['dataset_name']!r}: {source_format!r}"
                )
            raw_name = str(row["dataset_name"]).strip()
            if not raw_name:
                raise ValueError(f"Empty dataset_name in {summary_path}")
            source_files = str(row["source_files"]).strip()
            if not source_files:
                raise ValueError(f"Empty source_files for {raw_name!r}")

            # TSF/RDS records are independent univariate series.  TIME Arrow
            # rows retain the multivariate target stored in each record.
            enc_in = int(row["channel_count"]) if source_format == "arrow" else 1
            specs.append(
                DatasetSpec(
                    name=f"{family}__{raw_name}",
                    family=family,
                    loader="multi_series",
                    root_path=dataset_root,
                    data_path=source_files,
                    source_format=source_format,
                    enc_in=enc_in,
                )
            )
    return specs


def build_registry(args: argparse.Namespace, repo_root: Path) -> list[DatasetSpec]:
    monash_summary = resolve_path(args.monash_summary, repo_root)
    time_summary = resolve_path(args.time_summary, repo_root)
    monash_root = resolve_path(args.monash_root, repo_root)
    time_root = resolve_path(args.time_root, repo_root)

    specs = benchmark_specs(repo_root)
    specs.extend(read_summary_specs(monash_summary, monash_root, "Monash"))
    specs.extend(read_summary_specs(time_summary, time_root, "TIME"))

    names = [spec.name for spec in specs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate dataset names in registry: {duplicates}")
    if len(specs) != EXPECTED_DATASET_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_DATASET_COUNT} datasets, but discovered {len(specs)}. "
            "Check the two curated *_with_sample_counts.csv summaries."
        )

    family_counts = pd.Series([spec.family for spec in specs]).value_counts().to_dict()
    if family_counts != EXPECTED_FAMILY_COUNTS:
        raise ValueError(
            f"Expected family counts {EXPECTED_FAMILY_COUNTS}, but found {family_counts}."
        )

    for spec in specs:
        if spec.source_format == "csv":
            exists = (spec.root_path / spec.data_path).is_file()
        else:
            exists = bool(list(spec.root_path.glob(spec.data_path)))
        if not exists:
            raise FileNotFoundError(f"No source files found for {spec.name}: {spec.source_path}")
    return specs


def shuffled_groups(
    specs: list[DatasetSpec],
    datasets_per_group: int,
    rng: np.random.Generator,
) -> list[list[DatasetSpec]]:
    if datasets_per_group <= 0:
        raise ValueError(f"datasets-per-group must be positive, got {datasets_per_group}")
    shuffled = [specs[int(index)] for index in rng.permutation(len(specs))]
    groups = [
        shuffled[start : start + datasets_per_group]
        for start in range(0, len(shuffled), datasets_per_group)
    ]
    if len(specs) == 53 and datasets_per_group == 8:
        sizes = [len(group) for group in groups]
        if sizes != [8, 8, 8, 8, 8, 8, 5]:
            raise RuntimeError(f"Unexpected default group sizes: {sizes}")
    return groups


def assignment_frame(groups: list[list[DatasetSpec]], repo_root: Path) -> pd.DataFrame:
    rows = []
    for group_index, group in enumerate(groups, start=1):
        group_id = f"group_{group_index:03d}"
        for group_rank, spec in enumerate(group, start=1):
            rows.append(
                {
                    "group_id": group_id,
                    "group_rank": group_rank,
                    "dataset": spec.name,
                    "family": spec.family,
                    "source_format": spec.source_format,
                    "source_path": relative_path(spec.root_path / spec.data_path, repo_root),
                }
            )
    return pd.DataFrame(rows)


def sample_lengths(spec: DatasetSpec, args: argparse.Namespace) -> tuple[int, int, int]:
    if spec.name == "illness":
        return args.illness_seq_len, args.illness_label_len, args.illness_pred_len
    return args.seq_len, args.label_len, args.pred_len


def build_train_dataset(spec: DatasetSpec, args: argparse.Namespace):
    size = list(sample_lengths(spec, args))
    loader_args = SimpleNamespace(
        augmentation_ratio=0,
        enc_in=spec.enc_in,
        multi_series_lru_size=args.multi_series_lru_size,
        # Zero disables the loader's optional candidate cap, so sampling is
        # performed over the complete train-window population.
        long_term_train_sample_limit=0,
        candidate_sample_seed=args.seed,
    )
    scale = not args.no_scale

    if spec.loader == "multi_series":
        dataset_class = Dataset_MultiSeriesForecast
    elif spec.loader == "ETT_hour":
        dataset_class = Dataset_ETT_hour
    elif spec.loader == "custom":
        dataset_class = Dataset_Custom
    else:
        raise ValueError(f"Unknown loader {spec.loader!r} for {spec.name}")

    return dataset_class(
        args=loader_args,
        root_path=str(spec.root_path),
        flag="train",
        size=size,
        features="M",
        data_path=spec.data_path,
        target=spec.target,
        scale=scale,
        timeenc=1,
        freq=spec.freq,
    )


def sample_indices(length: int, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    if n_samples <= 0:
        raise ValueError(f"samples-per-dataset must be positive, got {n_samples}")
    if length < n_samples:
        raise ValueError(f"Train split has {length} windows, but {n_samples} are required")
    return rng.choice(length, size=n_samples, replace=False).astype(np.int64, copy=False)


def interpolate_nonfinite(series: np.ndarray) -> np.ndarray:
    values = np.asarray(series, dtype=np.float64)
    finite = np.isfinite(values)
    if finite.all():
        return values
    if not finite.any():
        return np.zeros_like(values)
    if finite.sum() == 1:
        return np.full_like(values, float(values[finite][0]))
    positions = np.arange(len(values), dtype=np.float64)
    return np.interp(positions, positions[finite], values[finite])


def catch22_by_channel(sample: np.ndarray) -> tuple[list[str], list[str], np.ndarray]:
    values = np.asarray(sample, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(f"Expected a [seq_len, channels] window, got {values.shape}")

    feature_names: list[str] | None = None
    short_names: list[str] | None = None
    channel_features = []
    for channel_index in range(values.shape[1]):
        series = interpolate_nonfinite(values[:, channel_index])
        result = pycatch22.catch22_all(series, catch24=False, short_names=True)
        current_names = list(result["names"])
        current_short_names = list(result["short_names"])
        if feature_names is None:
            feature_names = current_names
            short_names = current_short_names
        elif short_names != current_short_names:
            raise RuntimeError("catch22 feature order changed between channels")
        channel_features.append(np.asarray(result["values"], dtype=np.float64))

    if feature_names is None or short_names is None:
        raise RuntimeError("No channel was available for catch22 extraction")
    matrix = np.vstack(channel_features)
    if matrix.shape[1] != 22:
        raise RuntimeError(f"Expected 22 catch22 features, got shape={matrix.shape}")
    return feature_names, short_names, matrix


def pool_channel_features(features: np.ndarray, pooling: str) -> np.ndarray:
    finite_features = np.where(np.isfinite(features), features, np.nan)
    if pooling == "mean":
        counts = np.sum(np.isfinite(finite_features), axis=0)
        pooled = np.full(features.shape[1], np.nan, dtype=np.float64)
        np.divide(
            np.nansum(finite_features, axis=0),
            counts,
            out=pooled,
            where=counts > 0,
        )
        return pooled
    if pooling == "median":
        with np.errstate(all="ignore"):
            return np.nanmedian(finite_features, axis=0)
    raise ValueError(f"Unsupported pooling method: {pooling}")


def window_location(dataset: object, window_index: int) -> tuple[int, int]:
    """Return source-series index and within-series start when available."""
    if isinstance(dataset, Dataset_MultiSeriesForecast):
        global_index = int(window_index)
        series_index = bisect_right(dataset.cumulative_windows, global_index)
        previous = 0 if series_index == 0 else int(dataset.cumulative_windows[series_index - 1])
        start = int(dataset.series_start_offsets[series_index] + global_index - previous)
        return series_index, start
    return 0, int(window_index)


def collect_features(
    groups: list[list[DatasetSpec]],
    args: argparse.Namespace,
    repo_root: Path,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    rng = np.random.default_rng(args.seed + 1)
    rows: list[dict[str, object]] = []
    feature_names: list[str] | None = None
    feature_short_names: list[str] | None = None

    dataset_total = sum(len(group) for group in groups)
    dataset_counter = 0
    for group_index, group in enumerate(groups, start=1):
        group_id = f"group_{group_index:03d}"
        for group_rank, spec in enumerate(group, start=1):
            dataset_counter += 1
            dataset = build_train_dataset(spec, args)
            indices = sample_indices(len(dataset), args.samples_per_dataset, rng)
            print(
                f"[{dataset_counter:02d}/{dataset_total}] {spec.name}: "
                f"sampling {len(indices)} of {len(dataset)} train windows",
                flush=True,
            )

            for sample_rank, window_index in enumerate(indices, start=1):
                seq_x, _, _, _ = dataset[int(window_index)]
                sample = np.asarray(seq_x, dtype=np.float64)
                expected_seq_len = int(dataset.seq_len)
                if sample.shape[0] != expected_seq_len:
                    raise RuntimeError(
                        f"{spec.name} returned window length {sample.shape[0]}, "
                        f"expected {expected_seq_len}"
                    )

                names, short_names, channel_features = catch22_by_channel(sample)
                pooled = pool_channel_features(channel_features, args.pooling)
                if feature_short_names is None:
                    feature_names = names
                    feature_short_names = short_names
                elif feature_short_names != short_names:
                    raise RuntimeError(f"catch22 feature order changed for {spec.name}")

                series_index, window_start = window_location(dataset, int(window_index))
                row: dict[str, object] = {
                    "group_id": group_id,
                    "group_rank": group_rank,
                    "dataset": spec.name,
                    "family": spec.family,
                    "source_format": spec.source_format,
                    "source_path": relative_path(spec.root_path / spec.data_path, repo_root),
                    "split": "train",
                    "sample_rank": sample_rank,
                    "window_index": int(window_index),
                    "series_index": series_index,
                    "window_start": window_start,
                    "seq_len": expected_seq_len,
                    "n_channels": int(sample.shape[1]) if sample.ndim == 2 else 1,
                    "pooling": args.pooling,
                    "scale": "train_split_standard" if not args.no_scale else "none",
                }
                for short_name, value in zip(short_names, pooled):
                    row[short_name] = float(value)
                rows.append(row)

            del dataset
            gc.collect()

    if feature_names is None or feature_short_names is None:
        raise RuntimeError("No catch22 features were collected")
    feature_df = pd.DataFrame(rows, columns=[*META_COLUMNS, *feature_short_names])
    expected_rows = dataset_total * args.samples_per_dataset
    if len(feature_df) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} feature rows, collected {len(feature_df)}")
    return feature_df, feature_names, feature_short_names


def zscore_features(group: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    features = group[feature_columns].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    imputed = features.fillna(features.mean(axis=0, skipna=True)).fillna(0.0)
    stds = imputed.std(axis=0, ddof=0).replace(0.0, np.nan)
    return ((imputed - imputed.mean(axis=0)) / stds).fillna(0.0)


def pca_2d(z_features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    matrix = z_features.to_numpy(dtype=np.float64)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    component_count = min(2, vt.shape[0])

    coordinates = np.zeros((len(centered), 2), dtype=np.float64)
    explained = np.zeros(2, dtype=np.float64)
    if component_count:
        coordinates[:, :component_count] = centered @ vt[:component_count].T
        variances = singular_values**2
        total_variance = float(variances.sum())
        if total_variance > 0.0:
            explained[:component_count] = variances[:component_count] / total_variance
    return coordinates, explained


def add_groupwise_pca(
    feature_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    z_frames = []
    pca_frames = []
    for _, group in feature_df.groupby("group_id", sort=False):
        group = group.reset_index(drop=True)
        z_features = zscore_features(group, feature_columns)
        z_features.columns = feature_columns
        z_frames.append(pd.concat([group[META_COLUMNS], z_features], axis=1))

        coordinates, explained = pca_2d(z_features)
        pca_output = group[META_COLUMNS].copy()
        pca_output["pc1"] = coordinates[:, 0]
        pca_output["pc2"] = coordinates[:, 1]
        pca_output["pc1_explained_ratio"] = float(explained[0])
        pca_output["pc2_explained_ratio"] = float(explained[1])
        pca_frames.append(pca_output)
    return pd.concat(z_frames, ignore_index=True), pd.concat(pca_frames, ignore_index=True)


def add_global_sample_pca(
    feature_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Globally z-score all sample vectors, then fit one shared 2D PCA."""
    z_features = zscore_features(feature_df, feature_columns)
    z_features.columns = feature_columns
    coordinates, explained = pca_2d(z_features)

    z_feature_df = pd.concat(
        [feature_df[META_COLUMNS].reset_index(drop=True), z_features.reset_index(drop=True)],
        axis=1,
    )
    pca_df = feature_df[META_COLUMNS].copy()
    pca_df["pc1"] = coordinates[:, 0]
    pca_df["pc2"] = coordinates[:, 1]
    pca_df["pc1_explained_ratio"] = float(explained[0])
    pca_df["pc2_explained_ratio"] = float(explained[1])
    return z_feature_df, pca_df


def cluster_circle_summary(
    pca_df: pd.DataFrame,
    min_visible_radius_fraction: float,
) -> pd.DataFrame:
    x_span = float(pca_df["pc1"].max() - pca_df["pc1"].min())
    y_span = float(pca_df["pc2"].max() - pca_df["pc2"].min())
    minimum_radius = max(x_span, y_span, 1.0) * max(0.0, min_visible_radius_fraction)

    records: list[dict[str, object]] = []
    for dataset, group in pca_df.groupby("dataset", sort=False):
        coordinates = group[["pc1", "pc2"]].to_numpy(dtype=np.float64)
        center = coordinates.mean(axis=0)
        distances = np.linalg.norm(coordinates - center[None, :], axis=1)
        true_radius = float(np.max(distances))
        records.append(
            {
                "dataset": dataset,
                "family": str(group["family"].iloc[0]),
                "n_samples": len(group),
                "center_pc1": float(center[0]),
                "center_pc2": float(center[1]),
                "true_radius": true_radius,
                "mean_sample_distance": float(np.mean(distances)),
                "rms_sample_distance": float(np.sqrt(np.mean(distances * distances))),
                "display_radius": max(true_radius, minimum_radius),
                "radius_was_expanded_for_visibility": true_radius < minimum_radius,
            }
        )
    return pd.DataFrame(records)


def save_global_sample_cluster_plot(
    pca_df: pd.DataFrame,
    circle_df: pd.DataFrame,
    output_path: Path,
    dpi: int,
) -> None:
    datasets = circle_df["dataset"].tolist()
    cmap = plt.get_cmap("turbo")
    color_denominator = max(1, len(datasets) - 1)
    colors = {
        dataset: cmap(index / color_denominator)
        for index, dataset in enumerate(datasets)
    }

    fig, ax = plt.subplots(figsize=(15.0, 11.0), constrained_layout=True)
    for row in circle_df.itertuples(index=False):
        dataset = str(row.dataset)
        rgba = colors[dataset]
        points = pca_df[pca_df["dataset"] == dataset]
        ax.add_patch(
            Circle(
                (float(row.center_pc1), float(row.center_pc2)),
                float(row.display_radius),
                facecolor=(rgba[0], rgba[1], rgba[2], 0.045),
                edgecolor=(rgba[0], rgba[1], rgba[2], 0.72),
                linewidth=1.15,
                linestyle=":" if bool(row.radius_was_expanded_for_visibility) else "-",
                zorder=1,
            )
        )
        ax.scatter(
            points["pc1"],
            points["pc2"],
            s=18,
            color=rgba,
            alpha=0.72,
            edgecolors="white",
            linewidths=0.25,
            zorder=2,
        )

    left = float((circle_df["center_pc1"] - circle_df["display_radius"]).min())
    right = float((circle_df["center_pc1"] + circle_df["display_radius"]).max())
    bottom = float((circle_df["center_pc2"] - circle_df["display_radius"]).min())
    top = float((circle_df["center_pc2"] + circle_df["display_radius"]).max())
    span = max(right - left, top - bottom, 1.0)
    padding = span * 0.04
    ax.set_xlim(left - padding, right + padding)
    ax.set_ylim(bottom - padding, top + padding)

    pc1_ratio = float(pca_df["pc1_explained_ratio"].iloc[0])
    pc2_ratio = float(pca_df["pc2_explained_ratio"].iloc[0])
    ax.set_xlabel(f"PC1 ({pc1_ratio:.2%})")
    ax.set_ylabel(f"PC2 ({pc2_ratio:.2%})")
    ax.set_title(
        f"Global z-scored catch22 PCA: {len(circle_df)} datasets × "
        f"{int(circle_df['n_samples'].iloc[0])} train samples"
    )
    ax.grid(True, linestyle=":", linewidth=0.65, alpha=0.45)
    ax.set_aspect("equal", adjustable="box")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def add_global_centroid_pca(
    feature_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate each dataset in 22D, then fit one PCA over all centroids."""
    numeric_features = feature_df[feature_columns].apply(pd.to_numeric, errors="coerce")
    numeric_features = numeric_features.replace([np.inf, -np.inf], np.nan)
    # Impute at sample level using a global feature mean before aggregation, so
    # every dataset contributes a complete 22-dimensional centroid.
    imputed_features = numeric_features.fillna(
        numeric_features.mean(axis=0, skipna=True)
    ).fillna(0.0)

    feature_input = imputed_features.copy()
    feature_input.insert(0, "dataset", feature_df["dataset"].to_numpy())
    centroid_features = feature_input.groupby("dataset", sort=False)[feature_columns].mean()

    metadata = (
        feature_df.groupby("dataset", sort=False)
        .agg(
            family=("family", "first"),
            source_format=("source_format", "first"),
            source_path=("source_path", "first"),
            sample_count=("sample_rank", "size"),
        )
        .reset_index()
    )
    metadata.insert(
        0,
        "centroid_id",
        [f"D{index:02d}" for index in range(1, len(metadata) + 1)],
    )
    centroid_feature_df = pd.concat(
        [metadata.reset_index(drop=True), centroid_features.reset_index(drop=True)],
        axis=1,
    )

    z_features = zscore_features(centroid_feature_df, feature_columns)
    z_features.columns = feature_columns
    centroid_z_df = pd.concat([metadata, z_features], axis=1)

    coordinates, explained = pca_2d(z_features)
    centroid_pca_df = metadata.copy()
    centroid_pca_df["pc1"] = coordinates[:, 0]
    centroid_pca_df["pc2"] = coordinates[:, 1]
    centroid_pca_df["pc1_explained_ratio"] = float(explained[0])
    centroid_pca_df["pc2_explained_ratio"] = float(explained[1])
    return centroid_feature_df, centroid_z_df, centroid_pca_df


def save_group_plot(pca_df: pd.DataFrame, output_path: Path, dpi: int) -> None:
    datasets = list(dict.fromkeys(pca_df["dataset"].tolist()))
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(11.5, 7.5), constrained_layout=True)
    for dataset_index, dataset_name in enumerate(datasets):
        points = pca_df[pca_df["dataset"] == dataset_name]
        ax.scatter(
            points["pc1"],
            points["pc2"],
            s=62,
            alpha=0.84,
            color=cmap(dataset_index % 10),
            label=dataset_name,
            edgecolors="white",
            linewidths=0.55,
        )

    pc1_ratio = float(pca_df["pc1_explained_ratio"].iloc[0])
    pc2_ratio = float(pca_df["pc2_explained_ratio"].iloc[0])
    group_id = str(pca_df["group_id"].iloc[0])
    ax.axhline(0.0, color="#b8b8b8", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="#b8b8b8", linewidth=0.8, zorder=0)
    ax.set_xlabel(f"PC1 ({pc1_ratio:.2%})")
    ax.set_ylabel(f"PC2 ({pc2_ratio:.2%})")
    ax.set_title(
        f"Train-window catch22 PCA: {group_id} "
        f"({len(datasets)} datasets x {len(pca_df) // len(datasets)} samples)"
    )
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.55)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def save_global_centroid_plot(
    pca_df: pd.DataFrame,
    output_path: Path,
    dpi: int,
    label_mode: str,
) -> None:
    family_colors = {
        "Benchmark": "#d62728",
        "Monash": "#1f77b4",
        "TIME": "#2ca02c",
    }
    fig, ax = plt.subplots(figsize=(13.0, 9.5), constrained_layout=True)
    for family, points in pca_df.groupby("family", sort=False):
        ax.scatter(
            points["pc1"],
            points["pc2"],
            s=72,
            alpha=0.88,
            color=family_colors.get(str(family), "#7f7f7f"),
            label=f"{family} ({len(points)})",
            edgecolors="white",
            linewidths=0.6,
        )

    if label_mode != "none":
        label_column = "centroid_id" if label_mode == "index" else "dataset"
        font_size = 7.0 if label_mode == "index" else 5.2
        for row in pca_df.itertuples(index=False):
            ax.annotate(
                str(getattr(row, label_column)),
                (float(row.pc1), float(row.pc2)),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=font_size,
                alpha=0.9,
            )

    pc1_ratio = float(pca_df["pc1_explained_ratio"].iloc[0])
    pc2_ratio = float(pca_df["pc2_explained_ratio"].iloc[0])
    ax.axhline(0.0, color="#b8b8b8", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="#b8b8b8", linewidth=0.8, zorder=0)
    ax.set_xlabel(f"PC1 ({pc1_ratio:.2%})")
    ax.set_ylabel(f"PC2 ({pc2_ratio:.2%})")
    ax.set_title(f"Global catch22 PCA of {len(pca_df)} dataset centroids")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.55)
    ax.legend(loc="best", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def write_feature_manifest(
    output_path: Path,
    feature_names: list[str],
    feature_short_names: list[str],
) -> None:
    pd.DataFrame(
        {
            "feature_index": np.arange(1, len(feature_names) + 1),
            "feature_name": feature_names,
            "feature_short_name": feature_short_names,
        }
    ).to_csv(output_path, index=False)


def write_summary(
    output_path: Path,
    args: argparse.Namespace,
    groups: list[list[DatasetSpec]],
    feature_df: pd.DataFrame,
    circle_df: pd.DataFrame,
) -> None:
    group_sizes = [len(group) for group in groups]
    lines = [
        "53-dataset global train-window catch22 PCA with cluster circles",
        "",
        "visualization_mode: global_sample_coordinates_with_cluster_circles",
        f"seed: {args.seed}",
        f"dataset_count: {sum(group_sizes)}",
        f"family_counts: {EXPECTED_FAMILY_COUNTS}",
        f"input_len_default: {args.seq_len}",
        f"input_len_illness: {args.illness_seq_len}",
        f"samples_per_dataset: {args.samples_per_dataset}",
        f"feature_rows: {len(feature_df)}",
        f"channel_pooling: {args.pooling}",
        "input_scaling: "
        + ("none" if args.no_scale else "Time-Series-Library train-split standardization"),
        "pca_preprocessing: global feature-mean imputation and feature-wise z-score",
        "pca_scope: one PCA fitted over all z-scored sample vectors",
        "circle_center: mean of each dataset's sample PCA coordinates",
        "circle_radius: maximum distance from center (contains every dataset sample)",
        f"visibility_expanded_circle_count: {int(circle_df['radius_was_expanded_for_visibility'].sum())}",
        "",
        "Files",
        "- dataset_assignments.csv: the 53 datasets and source paths.",
        "- sample_catch22_features.csv: raw pooled catch22 vectors.",
        "- sample_catch22_features_zscored_global.csv: normalized PCA inputs.",
        "- sample_catch22_pca_2d.csv: global two-dimensional PCA coordinates.",
        "- dataset_cluster_circles.csv: circle centers and radii.",
        "- catch22_feature_manifest.csv: full and short catch22 feature names.",
        "- plots/all_53_datasets_20_samples_cluster_circles.png: combined visualization.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--monash-root",
        type=Path,
        default=Path("dataset/Monash_Dataset"),
        help="Monash root; relative paths are resolved under --repo-root.",
    )
    parser.add_argument(
        "--time-root",
        type=Path,
        default=Path("dataset/Time_Dataset"),
        help="TIME root; relative paths are resolved under --repo-root.",
    )
    parser.add_argument(
        "--monash-summary",
        type=Path,
        default=Path("dataset/monash_dataset_summary_with_sample_counts.csv"),
    )
    parser.add_argument(
        "--time-summary",
        type=Path,
        default=Path("dataset/time_dataset_summary_with_sample_counts.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("catch22/train_window_pca_clusters_53"),
    )
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--label-len", type=int, default=48)
    parser.add_argument(
        "--pred-len",
        type=int,
        default=96,
        help="Forecast horizon used only to determine valid train windows.",
    )
    parser.add_argument("--samples-per-dataset", type=int, default=20)
    parser.add_argument("--illness-seq-len", type=int, default=36)
    parser.add_argument("--illness-label-len", type=int, default=18)
    parser.add_argument("--illness-pred-len", type=int, default=36)
    parser.add_argument("--pooling", choices=("mean", "median"), default="mean")
    parser.add_argument("--no-scale", action="store_true")
    parser.add_argument("--multi-series-lru-size", type=int, default=8)
    parser.add_argument("--min-visible-radius-fraction", type=float, default=0.004)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_values = {
        "seq-len": args.seq_len,
        "label-len": args.label_len,
        "pred-len": args.pred_len,
        "illness-seq-len": args.illness_seq_len,
        "illness-label-len": args.illness_label_len,
        "illness-pred-len": args.illness_pred_len,
        "samples-per-dataset": args.samples_per_dataset,
        "multi-series-lru-size": args.multi_series_lru_size,
        "dpi": args.dpi,
    }
    invalid = {name: value for name, value in positive_values.items() if value <= 0}
    if invalid:
        raise ValueError(f"These arguments must be positive: {invalid}")
    if args.label_len > args.seq_len:
        raise ValueError("label-len cannot exceed seq-len")
    if args.illness_label_len > args.illness_seq_len:
        raise ValueError("illness-label-len cannot exceed illness-seq-len")
    if args.min_visible_radius_fraction < 0.0:
        raise ValueError("min-visible-radius-fraction cannot be negative")


def main() -> int:
    args = parse_args()
    validate_args(args)
    repo_root = args.repo_root.resolve()
    output_dir = resolve_path(args.output_dir, repo_root).resolve()

    specs = build_registry(args, repo_root)
    # One global group makes feature collection deterministic while PCA is fit
    # once across all 53 datasets rather than independently by small groups.
    groups = [specs]
    feature_df, feature_names, feature_short_names = collect_features(groups, args, repo_root)
    z_feature_df, pca_df = add_global_sample_pca(feature_df, feature_short_names)
    circle_df = cluster_circle_summary(
        pca_df, min_visible_radius_fraction=args.min_visible_radius_fraction
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    assignment_frame(groups, repo_root).to_csv(
        output_dir / "dataset_assignments.csv", index=False
    )
    feature_df.to_csv(
        output_dir / "sample_catch22_features.csv", index=False, float_format="%.9f"
    )
    z_feature_df.to_csv(
        output_dir / "sample_catch22_features_zscored_global.csv",
        index=False,
        float_format="%.9f",
    )
    pca_df.to_csv(
        output_dir / "sample_catch22_pca_2d.csv", index=False, float_format="%.9f"
    )
    circle_df.to_csv(
        output_dir / "dataset_cluster_circles.csv", index=False, float_format="%.9f"
    )
    write_feature_manifest(
        output_dir / "catch22_feature_manifest.csv", feature_names, feature_short_names
    )

    plots_dir = output_dir / "plots"
    plot_path = plots_dir / "all_53_datasets_20_samples_cluster_circles.png"
    save_global_sample_cluster_plot(pca_df, circle_df, plot_path, args.dpi)
    write_summary(output_dir / "summary.txt", args, groups, feature_df, circle_df)

    print(f"Datasets: {len(specs)} ({EXPECTED_FAMILY_COUNTS})")
    print(f"Feature rows: {len(feature_df)} x {len(feature_short_names)} catch22 features")
    print(f"Global z-scored PCA rows: {len(pca_df)}")
    print(f"Cluster circles: {len(circle_df)}")
    print(f"PCA cluster plot: {plot_path}")
    print(f"Output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
