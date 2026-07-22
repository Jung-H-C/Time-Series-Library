#!/usr/bin/env python3
"""Plot only 53 dataset cluster circles in a global catch22 PCA plane.

For each of the six Benchmark, 18 Monash, and 29 TIME datasets, this script
samples 20 stride-1 windows without replacement from the Time-Series-Library
train split.  Input windows have length 96 except for illness, whose input
length is 36.  catch22 is evaluated independently on every channel and pooled
feature-wise, yielding one 22-dimensional vector per sampled window.

All 1,060 pooled vectors are globally imputed, z-scored, and projected by one
shared 2D PCA.  Each dataset circle is centered at the mean of its 20 PCA
coordinates.  Its radius is the maximum Euclidean distance from that center,
so the circle contains all 20 samples.  The figure deliberately draws no
sample points, center markers, coordinate annotations, or dataset labels: only
the 53 cluster circles are rendered on the PCA axes.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
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


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_provider.data_loader import (  # noqa: E402
    Dataset_Custom,
    Dataset_ETT_hour,
    Dataset_MultiSeriesForecast,
)
from visualize_53_train_window_pca_groups import (  # noqa: E402
    DatasetSpec,
    EXPECTED_DATASET_COUNT,
    build_registry,
    catch22_by_channel,
    pca_2d,
    pool_channel_features,
    relative_path,
    resolve_path,
    sample_indices,
    window_location,
    write_feature_manifest,
    zscore_features,
)


DEFAULT_OUTPUT_DIR = Path("catch22/train_window_cluster_circles_53")
FEATURE_META_COLUMNS = [
    "dataset",
    "registry_name",
    "family",
    "source_format",
    "source_path",
    "split",
    "sample_rank",
    "window_index",
    "series_index",
    "window_start",
    "input_len",
    "pred_len",
    "n_channels",
    "pooling",
    "scale",
]
PCA_META_COLUMNS = FEATURE_META_COLUMNS


def canonical_dataset_name(spec: DatasetSpec) -> str:
    benchmark_aliases = {"electricity": "ECL", "ETT-small": "ETTh1"}
    if spec.family == "Benchmark":
        return benchmark_aliases.get(spec.name, spec.name)
    if spec.family == "Monash":
        return spec.name.removeprefix("Monash__")
    if spec.family == "TIME":
        return spec.name.removeprefix("TIME__")
    return spec.name


def canonicalize_specs(specs: list[DatasetSpec]) -> list[DatasetSpec]:
    canonical = [replace(spec, name=canonical_dataset_name(spec)) for spec in specs]
    names = [spec.name for spec in canonical]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Canonical dataset names are not unique: {duplicates}")
    return canonical


def validate_against_dataset_summary(specs: list[DatasetSpec], repo_root: Path) -> None:
    """Ensure the plotting registry is exactly the current 53-row summary."""
    summary_path = repo_root / "dataset/dataset_split_channel_series_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Missing 53-dataset summary: {summary_path}. "
            "Run scripts/generate_dataset_split_channel_series_summary.py first."
        )
    summary = pd.read_csv(summary_path)
    if "dataset_name" not in summary.columns:
        raise ValueError(f"Missing dataset_name column in {summary_path}")
    expected = set(summary["dataset_name"].astype(str))
    actual = {spec.name for spec in specs}
    if len(summary) != EXPECTED_DATASET_COUNT or len(expected) != EXPECTED_DATASET_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_DATASET_COUNT} unique summary rows, found "
            f"{len(summary)} rows and {len(expected)} unique names"
        )
    if actual != expected:
        raise ValueError(
            "Plot registry differs from dataset summary; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def sample_lengths(spec: DatasetSpec, args: argparse.Namespace) -> tuple[int, int, int]:
    if spec.name == "illness":
        return args.illness_input_len, args.illness_label_len, args.illness_pred_len
    return args.input_len, args.label_len, args.pred_len


def build_train_dataset(
    spec: DatasetSpec,
    args: argparse.Namespace,
):
    input_len, label_len, pred_len = sample_lengths(spec, args)
    loader_args = SimpleNamespace(
        augmentation_ratio=0,
        enc_in=spec.enc_in,
        multi_series_lru_size=args.multi_series_lru_size,
        # Sample over every valid train window, not the experimental cap.
        long_term_train_sample_limit=0,
        candidate_sample_seed=args.seed,
    )
    if spec.loader == "multi_series":
        dataset_class = Dataset_MultiSeriesForecast
    elif spec.loader == "ETT_hour":
        dataset_class = Dataset_ETT_hour
    elif spec.loader == "custom":
        dataset_class = Dataset_Custom
    else:
        raise ValueError(f"Unknown loader {spec.loader!r} for {spec.name}")

    dataset = dataset_class(
        args=loader_args,
        root_path=str(spec.root_path),
        flag="train",
        size=[input_len, label_len, pred_len],
        features="M",
        data_path=spec.data_path,
        target=spec.target,
        scale=not args.no_scale,
        timeenc=1,
        freq=spec.freq,
    )
    return dataset, input_len, pred_len


def select_specs(
    specs: list[DatasetSpec], selected_names: list[str] | None
) -> list[DatasetSpec]:
    if not selected_names:
        return specs
    requested = set(selected_names)
    lookup = {spec.name: spec for spec in specs}
    unknown = sorted(requested.difference(lookup))
    if unknown:
        raise ValueError(f"Unknown --dataset values: {unknown}")
    return [spec for spec in specs if spec.name in requested]


def collect_sample_features(
    specs: list[DatasetSpec],
    args: argparse.Namespace,
    repo_root: Path,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    feature_names: list[str] | None = None
    feature_short_names: list[str] | None = None

    for dataset_rank, spec in enumerate(specs, start=1):
        dataset, input_len, pred_len = build_train_dataset(spec, args)
        indices = sample_indices(len(dataset), args.samples_per_dataset, rng)
        print(
            f"[{dataset_rank:02d}/{len(specs):02d}] {spec.name}: sampling "
            f"{len(indices)} of {len(dataset)} train windows "
            f"(input_len={input_len}, channels={spec.enc_in})",
            flush=True,
        )

        for sample_rank, window_index in enumerate(indices, start=1):
            seq_x, _, _, _ = dataset[int(window_index)]
            sample = np.asarray(seq_x, dtype=np.float64)
            if sample.ndim == 1:
                sample = sample[:, None]
            if sample.ndim != 2 or sample.shape[0] != input_len:
                raise RuntimeError(
                    f"{spec.name} returned shape {sample.shape}; "
                    f"expected [{input_len}, channels]"
                )

            names, short_names, channel_features = catch22_by_channel(sample)
            pooled = pool_channel_features(channel_features, args.pooling)
            if pooled.shape != (22,):
                raise RuntimeError(
                    f"Expected one 22D pooled vector for {spec.name}, got {pooled.shape}"
                )
            if feature_short_names is None:
                feature_names = names
                feature_short_names = short_names
            elif feature_short_names != short_names:
                raise RuntimeError(f"catch22 feature order changed for {spec.name}")

            series_index, window_start = window_location(dataset, int(window_index))
            row: dict[str, object] = {
                "dataset": spec.name,
                "registry_name": spec.name,
                "family": spec.family,
                "source_format": spec.source_format,
                "source_path": relative_path(spec.root_path / spec.data_path, repo_root),
                "split": "train",
                "sample_rank": sample_rank,
                "window_index": int(window_index),
                "series_index": series_index,
                "window_start": window_start,
                "input_len": input_len,
                "pred_len": pred_len,
                "n_channels": int(sample.shape[1]),
                "pooling": args.pooling,
                "scale": "none" if args.no_scale else "train_split_standard",
            }
            for short_name, value in zip(short_names, pooled):
                row[short_name] = float(value)
            rows.append(row)

        del dataset
        gc.collect()

    if feature_names is None or feature_short_names is None:
        raise RuntimeError("No catch22 features were collected")
    feature_df = pd.DataFrame(rows, columns=[*FEATURE_META_COLUMNS, *feature_short_names])
    expected_rows = len(specs) * args.samples_per_dataset
    if len(feature_df) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} feature rows, got {len(feature_df)}")
    per_dataset = feature_df.groupby("dataset", sort=False).size()
    if not (per_dataset == args.samples_per_dataset).all():
        raise RuntimeError(f"Unexpected per-dataset sample counts: {per_dataset.to_dict()}")
    return feature_df, feature_names, feature_short_names


def global_sample_pca(
    feature_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    z_features = zscore_features(feature_df, feature_columns)
    z_features.columns = feature_columns
    coordinates, explained = pca_2d(z_features)

    z_feature_df = pd.concat(
        [feature_df[PCA_META_COLUMNS].reset_index(drop=True), z_features.reset_index(drop=True)],
        axis=1,
    )
    pca_df = feature_df[PCA_META_COLUMNS].copy()
    pca_df["pc1"] = coordinates[:, 0]
    pca_df["pc2"] = coordinates[:, 1]
    pca_df["pc1_explained_ratio"] = float(explained[0])
    pca_df["pc2_explained_ratio"] = float(explained[1])
    return z_feature_df, pca_df


def cluster_circle_summary(pca_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for dataset, group in pca_df.groupby("dataset", sort=False):
        coordinates = group[["pc1", "pc2"]].to_numpy(dtype=np.float64)
        center = coordinates.mean(axis=0)
        distances = np.linalg.norm(coordinates - center[None, :], axis=1)
        records.append(
            {
                "dataset": dataset,
                "family": str(group["family"].iloc[0]),
                "n_samples": len(group),
                "center_pc1": float(center[0]),
                "center_pc2": float(center[1]),
                "true_radius": float(np.max(distances)),
                "mean_sample_distance": float(np.mean(distances)),
                "rms_sample_distance": float(np.sqrt(np.mean(distances * distances))),
            }
        )
    return pd.DataFrame(records)


def pca_span(pca_df: pd.DataFrame) -> float:
    x_span = float(pca_df["pc1"].max() - pca_df["pc1"].min())
    y_span = float(pca_df["pc2"].max() - pca_df["pc2"].min())
    return max(x_span, y_span, 1.0)


def add_display_radii(
    summary_df: pd.DataFrame,
    pca_df: pd.DataFrame,
    min_visible_radius_fraction: float,
) -> pd.DataFrame:
    output = summary_df.copy()
    minimum = pca_span(pca_df) * max(0.0, min_visible_radius_fraction)
    output["display_radius"] = np.maximum(output["true_radius"].to_numpy(float), minimum)
    output["radius_was_expanded_for_visibility"] = (
        output["display_radius"] > output["true_radius"]
    )
    return output


def save_circles_only_plot(
    pca_df: pd.DataFrame,
    circle_df: pd.DataFrame,
    output_path: Path,
    dpi: int,
) -> None:
    cmap = plt.get_cmap("turbo")
    color_denominator = max(1, len(circle_df) - 1)

    fig, ax = plt.subplots(figsize=(14.0, 11.0), constrained_layout=True)
    for circle_index, row in enumerate(circle_df.itertuples(index=False)):
        rgba = cmap(circle_index / color_denominator)
        adjusted = bool(row.radius_was_expanded_for_visibility)
        ax.add_patch(
            Circle(
                (float(row.center_pc1), float(row.center_pc2)),
                float(row.display_radius),
                facecolor=(rgba[0], rgba[1], rgba[2], 0.055),
                edgecolor=(rgba[0], rgba[1], rgba[2], 0.82),
                linewidth=1.25,
                linestyle=":" if adjusted else "-",
            )
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
        f"Global catch22 PCA: {len(circle_df)} train-window dataset cluster circles"
    )
    ax.grid(True, linestyle=":", linewidth=0.65, alpha=0.45)
    ax.set_aspect("equal", adjustable="box")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_summary(
    output_path: Path,
    args: argparse.Namespace,
    specs: list[DatasetSpec],
    feature_df: pd.DataFrame,
    circle_df: pd.DataFrame,
) -> None:
    input_lengths = feature_df.groupby("dataset", sort=False)["input_len"].first().to_dict()
    lines = [
        "53-dataset train-window catch22 PCA cluster circles",
        "",
        f"seed: {args.seed}",
        f"dataset_count: {len(specs)}",
        f"family_counts: {pd.Series([spec.family for spec in specs]).value_counts().to_dict()}",
        f"samples_per_dataset: {args.samples_per_dataset}",
        f"feature_rows: {len(feature_df)}",
        "catch22_dimensions_per_sample: 22",
        f"channel_pooling: {args.pooling}",
        f"input_scaling: {'none' if args.no_scale else 'Time-Series-Library train standardization'}",
        f"input_lengths_by_dataset: {input_lengths}",
        "pca_preprocessing: global feature-mean imputation and z-score over all samples",
        "pca_scope: one two-component PCA fitted over all sample-level 22D vectors",
        f"circle_center: mean of each dataset's {args.samples_per_dataset} 2D PCA coordinates",
        "circle_true_radius: maximum 2D distance from center (contains all sampled points)",
        f"minimum_visible_radius_fraction: {args.min_visible_radius_fraction}",
        f"visibility_expanded_circle_count: {int(circle_df['radius_was_expanded_for_visibility'].sum())}",
        "plot_contents: cluster circles only; sample points, centers, labels, and annotations omitted",
        "",
        "Files",
        "- sample_catch22_features.csv: raw channel-pooled 22D vectors.",
        "- sample_catch22_features_zscored.csv: global PCA inputs.",
        "- sample_catch22_pca_2d.csv: PCA coordinates retained for reproducibility, not plotted.",
        "- dataset_cluster_circles.csv: circle centers and true/display radii.",
        "- catch22_feature_manifest.csv: full and short catch22 feature names.",
        "- train_window_cluster_circles_only.png: circle-only 2D PCA plot.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--monash-root", type=Path, default=Path("dataset/Monash_Dataset"))
    parser.add_argument("--time-root", type=Path, default=Path("dataset/Time_Dataset"))
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--samples-per-dataset", type=int, default=20)
    parser.add_argument("--input-len", type=int, default=96)
    parser.add_argument("--label-len", type=int, default=48)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--illness-input-len", type=int, default=36)
    parser.add_argument("--illness-label-len", type=int, default=18)
    parser.add_argument("--illness-pred-len", type=int, default=36)
    parser.add_argument("--pooling", choices=("mean", "median"), default="mean")
    parser.add_argument("--no-scale", action="store_true")
    parser.add_argument("--multi-series-lru-size", type=int, default=8)
    parser.add_argument("--min-visible-radius-fraction", type=float, default=0.004)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument(
        "--dataset",
        action="append",
        help="Optional canonical dataset name to run; repeat to select a smoke-test subset.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "samples-per-dataset": args.samples_per_dataset,
        "input-len": args.input_len,
        "label-len": args.label_len,
        "pred-len": args.pred_len,
        "illness-input-len": args.illness_input_len,
        "illness-label-len": args.illness_label_len,
        "illness-pred-len": args.illness_pred_len,
        "multi-series-lru-size": args.multi_series_lru_size,
        "dpi": args.dpi,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"Arguments must be positive: {invalid}")
    if args.label_len > args.input_len:
        raise ValueError("label-len cannot exceed input-len")
    if args.illness_label_len > args.illness_input_len:
        raise ValueError("illness-label-len cannot exceed illness-input-len")
    if args.min_visible_radius_fraction < 0.0:
        raise ValueError("min-visible-radius-fraction cannot be negative")


def main() -> int:
    args = parse_args()
    validate_args(args)
    repo_root = args.repo_root.resolve()
    output_dir = resolve_path(args.output_dir, repo_root).resolve()

    registry_specs = canonicalize_specs(build_registry(args, repo_root))
    validate_against_dataset_summary(registry_specs, repo_root)
    specs = select_specs(registry_specs, args.dataset)
    feature_df, feature_names, feature_short_names = collect_sample_features(
        specs, args, repo_root
    )
    z_feature_df, pca_df = global_sample_pca(feature_df, feature_short_names)
    circle_df = add_display_radii(
        cluster_circle_summary(pca_df),
        pca_df,
        args.min_visible_radius_fraction,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(
        output_dir / "sample_catch22_features.csv", index=False, float_format="%.9f"
    )
    z_feature_df.to_csv(
        output_dir / "sample_catch22_features_zscored.csv",
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
    plot_path = output_dir / "train_window_cluster_circles_only.png"
    save_circles_only_plot(pca_df, circle_df, plot_path, args.dpi)
    write_summary(output_dir / "summary.txt", args, specs, feature_df, circle_df)

    print(f"Datasets: {len(specs)}")
    print(f"Feature matrix: {len(feature_df)} x {len(feature_short_names)}")
    print(f"Cluster circles: {len(circle_df)}")
    print(f"Circle-only plot: {plot_path}")
    print(f"Output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
