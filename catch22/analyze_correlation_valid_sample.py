from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tslib_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycatch22

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from extract_valid_sample_catch24 import (  # noqa: E402
    DATASETS,
    build_dataset,
    pool_features,
)


META_COLUMNS = [
    "dataset",
    "data_path",
    "split",
    "sample_rank",
    "sample_index",
    "seq_len",
    "pred_len",
    "n_channels",
    "pooling",
    "scale",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample validation sequences, extract channel-pooled catch22 features, "
            "and analyze within-dataset and between-dataset feature differences."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root. Default: inferred from this script path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("catch22/analysis_valid_samples/22 features"),
        help="Output directory. Relative paths are resolved under --repo-root.",
    )
    parser.add_argument("--seed", type=int, default=20260603, help="Random seed for validation sampling.")
    parser.add_argument("--samples-per-dataset", type=int, default=10, help="Number of val samples per dataset.")
    parser.add_argument(
        "--pooling",
        choices=("mean", "median"),
        default="mean",
        help="Feature-wise pooling across channels before sample-level analysis.",
    )
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Use raw values instead of Time-Series-Library train-split scaling.",
    )
    return parser.parse_args()


def catch22_by_channel(sample: np.ndarray) -> tuple[list[str], list[str], np.ndarray]:
    feature_names = None
    short_names = None
    channel_features = []
    for channel_index in range(sample.shape[1]):
        series = np.asarray(sample[:, channel_index], dtype=np.float64)
        result = pycatch22.catch22_all(series, catch24=False, short_names=True)
        if feature_names is None:
            feature_names = list(result["names"])
            short_names = list(result["short_names"])
        channel_features.append(np.asarray(result["values"], dtype=np.float64))

    return feature_names, short_names, np.vstack(channel_features)


def sample_indices(length: int, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    if length <= 0:
        raise ValueError("Dataset has no validation samples.")
    replace = n_samples > length
    return rng.choice(length, size=n_samples, replace=replace)


def collect_sample_features(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    feature_names: list[str] | None = None
    scale = not args.no_scale

    for config in DATASETS:
        dataset = build_dataset(config, scale=scale)
        indices = sample_indices(len(dataset), args.samples_per_dataset, rng)
        for sample_rank, sample_index in enumerate(indices, start=1):
            seq_x, _, _, _ = dataset[int(sample_index)]
            sample = np.asarray(seq_x, dtype=np.float64)
            _, short_names, channel_features = catch22_by_channel(sample)
            pooled = pool_features(channel_features, args.pooling)

            if feature_names is None:
                feature_names = short_names
            elif feature_names != short_names:
                raise RuntimeError(f"Feature order changed for dataset: {config.name}")

            row = {
                "dataset": config.name,
                "data_path": str(Path(config.root_path) / config.data_path),
                "split": "val",
                "sample_rank": sample_rank,
                "sample_index": int(sample_index),
                "seq_len": config.seq_len,
                "pred_len": config.pred_len,
                "n_channels": sample.shape[1],
                "pooling": args.pooling,
                "scale": "train_split_standard" if scale else "none",
            }
            for short_name, value in zip(short_names, pooled):
                row[short_name] = float(value)
            rows.append(row)

    if feature_names is None:
        raise RuntimeError("No feature rows were collected.")
    return pd.DataFrame(rows), feature_names


def numeric_feature_frame(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    features = df[feature_names].apply(pd.to_numeric, errors="coerce")
    return features.replace([np.inf, -np.inf], np.nan)


def zscore_frame(features: pd.DataFrame) -> pd.DataFrame:
    means = features.mean(axis=0, skipna=True)
    stds = features.std(axis=0, ddof=0, skipna=True).replace(0.0, np.nan)
    z = (features - means) / stds
    return z.fillna(0.0)


def pairwise_euclidean(matrix: np.ndarray) -> np.ndarray:
    diff = matrix[:, None, :] - matrix[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def pairwise_cosine_distance(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1)
    denom = np.outer(norms, norms)
    sim = np.divide(matrix @ matrix.T, denom, out=np.zeros((len(matrix), len(matrix))), where=denom > 0)
    return 1.0 - np.clip(sim, -1.0, 1.0)


def pairwise_correlation_distance(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    return pairwise_cosine_distance(centered)


def upper_triangle_mean(distance_matrix: np.ndarray) -> float:
    if distance_matrix.shape[0] < 2:
        return float("nan")
    indices = np.triu_indices(distance_matrix.shape[0], k=1)
    return float(np.mean(distance_matrix[indices]))


def within_feature_variation(
    sample_df: pd.DataFrame,
    z_features: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for dataset_name, group in sample_df.groupby("dataset", sort=False):
        idx = group.index
        raw_values = numeric_feature_frame(sample_df.loc[idx], feature_names)
        z_values = z_features.loc[idx, feature_names]
        for feature_name in feature_names:
            vals = raw_values[feature_name].to_numpy(dtype=float)
            z_vals = z_values[feature_name].to_numpy(dtype=float)
            mean = float(np.nanmean(vals))
            std = float(np.nanstd(vals, ddof=0))
            min_value = float(np.nanmin(vals))
            max_value = float(np.nanmax(vals))
            cv_abs = std / abs(mean) if abs(mean) > 1e-12 else float("nan")
            records.append(
                {
                    "dataset": dataset_name,
                    "feature": feature_name,
                    "mean": mean,
                    "std": std,
                    "min": min_value,
                    "max": max_value,
                    "range": max_value - min_value,
                    "cv_abs": cv_abs,
                    "std_z": float(np.nanstd(z_vals, ddof=0)),
                    "range_z": float(np.nanmax(z_vals) - np.nanmin(z_vals)),
                }
            )
    return pd.DataFrame(records)


def within_dataset_summary(sample_df: pd.DataFrame, z_features: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    variation_df = within_feature_variation(sample_df, z_features, feature_names)
    for dataset_name, group in sample_df.groupby("dataset", sort=False):
        idx = group.index
        matrix = z_features.loc[idx, feature_names].to_numpy(dtype=float)
        feature_variation = variation_df[variation_df["dataset"] == dataset_name]
        max_row = feature_variation.sort_values("std_z", ascending=False).iloc[0]
        records.append(
            {
                "dataset": dataset_name,
                "n_samples": len(group),
                "mean_feature_std_z": float(feature_variation["std_z"].mean()),
                "max_feature_std_z": float(max_row["std_z"]),
                "max_varying_feature": str(max_row["feature"]),
                "mean_pairwise_euclidean_z": upper_triangle_mean(pairwise_euclidean(matrix)),
                "mean_pairwise_cosine_distance_z": upper_triangle_mean(pairwise_cosine_distance(matrix)),
                "mean_pairwise_corr_distance_z": upper_triangle_mean(pairwise_correlation_distance(matrix)),
            }
        )
    return pd.DataFrame(records)


def representative_features(sample_df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for dataset_name, group in sample_df.groupby("dataset", sort=False):
        feature_values = numeric_feature_frame(group, feature_names)
        row = {
            "dataset": dataset_name,
            "n_samples": len(group),
            "pooling": str(group["pooling"].iloc[0]),
            "scale": str(group["scale"].iloc[0]),
        }
        for feature_name in feature_names:
            row[feature_name] = float(feature_values[feature_name].mean(skipna=True))
        records.append(row)
    return pd.DataFrame(records)


def representative_distance_matrices(
    representative_df: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reps = numeric_feature_frame(representative_df, feature_names)
    reps_z = zscore_frame(reps)
    matrix = reps_z.to_numpy(dtype=float)
    datasets = representative_df["dataset"].tolist()
    euclidean = pd.DataFrame(pairwise_euclidean(matrix), index=datasets, columns=datasets)
    cosine = pd.DataFrame(pairwise_cosine_distance(matrix), index=datasets, columns=datasets)
    corr = pd.DataFrame(pairwise_correlation_distance(matrix), index=datasets, columns=datasets)
    return euclidean, cosine, corr


def distance_long(distance_df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    records = []
    datasets = list(distance_df.index)
    for i, left in enumerate(datasets):
        for j, right in enumerate(datasets):
            if j <= i:
                continue
            records.append(
                {
                    "dataset_a": left,
                    "dataset_b": right,
                    "metric": metric_name,
                    "distance": float(distance_df.loc[left, right]),
                }
            )
    return pd.DataFrame(records)


def pca_projection(z_features: pd.DataFrame, sample_df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    matrix = z_features[feature_names].to_numpy(dtype=float)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    n_components = min(3, vt.shape[0])
    components = vt[:n_components].T
    coords = centered @ components
    variances = singular_values**2
    explained = variances / variances.sum() if variances.sum() > 0 else np.zeros_like(variances)
    result = sample_df[META_COLUMNS].copy()
    result["pc1"] = coords[:, 0]
    result["pc2"] = coords[:, 1] if coords.shape[1] > 1 else 0.0
    result["pc3"] = coords[:, 2] if coords.shape[1] > 2 else 0.0
    result["pc1_explained_ratio"] = float(explained[0]) if len(explained) > 0 else 0.0
    result["pc2_explained_ratio"] = float(explained[1]) if len(explained) > 1 else 0.0
    result["pc3_explained_ratio"] = float(explained[2]) if len(explained) > 2 else 0.0
    return result


def write_dataframe(df: pd.DataFrame, path: Path, float_format: str = "%.9f") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True if df.index.name is not None else False, float_format=float_format)


def save_heatmap(distance_df: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    im = ax.imshow(distance_df.to_numpy(dtype=float), cmap="viridis")
    ax.set_xticks(np.arange(len(distance_df.columns)), labels=distance_df.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(distance_df.index)), labels=distance_df.index)
    ax.set_title(title)
    for i in range(len(distance_df.index)):
        for j in range(len(distance_df.columns)):
            ax.text(j, i, f"{distance_df.iloc[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_within_variation_bar(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    x = np.arange(len(summary_df))
    ax.bar(x, summary_df["mean_pairwise_euclidean_z"].to_numpy(dtype=float), color="#4C78A8")
    ax.set_xticks(x, labels=summary_df["dataset"].tolist(), rotation=30, ha="right")
    ax.set_ylabel("Mean pairwise Euclidean distance (z-feature)")
    ax.set_title("Within-dataset variation across 10 validation samples")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_pca_plot(pca_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    for dataset_name, group in pca_df.groupby("dataset", sort=False):
        ax.scatter(group["pc1"], group["pc2"], label=dataset_name, s=42, alpha=0.85)
        ax.text(group["pc1"].mean(), group["pc2"].mean(), dataset_name, fontsize=9, weight="bold")
    pc1_ratio = float(pca_df["pc1_explained_ratio"].iloc[0])
    pc2_ratio = float(pca_df["pc2_explained_ratio"].iloc[0])
    ax.set_xlabel(f"PC1 ({pc1_ratio:.1%})")
    ax.set_ylabel(f"PC2 ({pc2_ratio:.1%})")
    ax.set_title("Validation sample catch22 feature distribution")
    ax.legend(loc="best", fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_pca_3d_plot(pca_df: pd.DataFrame, output_path: Path) -> None:
    fig = plt.figure(figsize=(8.0, 6.5), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    for dataset_name, group in pca_df.groupby("dataset", sort=False):
        ax.scatter(group["pc1"], group["pc2"], group["pc3"], label=dataset_name, s=42, alpha=0.85)
        ax.text(
            group["pc1"].mean(),
            group["pc2"].mean(),
            group["pc3"].mean(),
            dataset_name,
            fontsize=8,
            weight="bold",
        )
    pc1_ratio = float(pca_df["pc1_explained_ratio"].iloc[0])
    pc2_ratio = float(pca_df["pc2_explained_ratio"].iloc[0])
    pc3_ratio = float(pca_df["pc3_explained_ratio"].iloc[0])
    ax.set_xlabel(f"PC1 ({pc1_ratio:.1%})")
    ax.set_ylabel(f"PC2 ({pc2_ratio:.1%})")
    ax.set_zlabel(f"PC3 ({pc3_ratio:.1%})")
    ax.set_title("Validation sample catch22 feature distribution (3D PCA)")
    ax.view_init(elev=24, azim=38)
    ax.legend(loc="best", fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary_text(
    output_path: Path,
    args: argparse.Namespace,
    within_df: pd.DataFrame,
    distance_long_df: pd.DataFrame,
    pca_df: pd.DataFrame,
) -> None:
    most_variable = within_df.sort_values("mean_pairwise_euclidean_z", ascending=False).iloc[0]
    least_variable = within_df.sort_values("mean_pairwise_euclidean_z", ascending=True).iloc[0]
    farthest = distance_long_df[distance_long_df["metric"] == "representative_euclidean_z"].sort_values(
        "distance", ascending=False
    ).iloc[0]
    closest = distance_long_df[distance_long_df["metric"] == "representative_euclidean_z"].sort_values(
        "distance", ascending=True
    ).iloc[0]
    pc1_ratio = float(pca_df["pc1_explained_ratio"].iloc[0])
    pc2_ratio = float(pca_df["pc2_explained_ratio"].iloc[0])
    pc3_ratio = float(pca_df["pc3_explained_ratio"].iloc[0])

    lines = [
        "catch22 valid-sample analysis summary",
        "",
        f"samples_per_dataset: {args.samples_per_dataset}",
        f"seed: {args.seed}",
        f"pooling: {args.pooling}",
        f"scale: {'train_split_standard' if not args.no_scale else 'none'}",
        "",
        "Within-dataset variation",
        (
            f"- Largest mean pairwise Euclidean distance: {most_variable['dataset']} "
            f"({most_variable['mean_pairwise_euclidean_z']:.6f}); "
            f"most varying feature: {most_variable['max_varying_feature']}."
        ),
        (
            f"- Smallest mean pairwise Euclidean distance: {least_variable['dataset']} "
            f"({least_variable['mean_pairwise_euclidean_z']:.6f}); "
            f"most varying feature: {least_variable['max_varying_feature']}."
        ),
        "",
        "Between-dataset representative distance",
        (
            f"- Farthest representative pair: {farthest['dataset_a']} vs {farthest['dataset_b']} "
            f"({farthest['distance']:.6f})."
        ),
        (
            f"- Closest representative pair: {closest['dataset_a']} vs {closest['dataset_b']} "
            f"({closest['distance']:.6f})."
        ),
        "",
        "PCA view",
        f"- PC1 + PC2 explained ratio: {pc1_ratio + pc2_ratio:.2%} ({pc1_ratio:.2%}, {pc2_ratio:.2%}).",
        (
            f"- PC1 + PC2 + PC3 explained ratio: {pc1_ratio + pc2_ratio + pc3_ratio:.2%} "
            f"({pc1_ratio:.2%}, {pc2_ratio:.2%}, {pc3_ratio:.2%})."
        ),
        "",
        "Files",
        "- sample_features.csv: 10 sampled catch22 feature vectors per dataset.",
        "- dataset_representative_features.csv: dataset-wise mean feature vectors.",
        "- within_dataset_feature_variation.csv: per-feature variation within each dataset.",
        "- within_dataset_summary.csv: compact within-dataset variation metrics.",
        "- between_dataset_*_distance.csv: representative-feature distance matrices.",
        "- plots/*.png: heatmap/bar/2D PCA/3D PCA visualizations.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir

    original_cwd = Path.cwd()
    os.chdir(repo_root)
    try:
        sample_df, feature_names = collect_sample_features(args)
    finally:
        os.chdir(original_cwd)

    features = numeric_feature_frame(sample_df, feature_names)
    z_features = zscore_frame(features)
    z_features.columns = feature_names

    representative_df = representative_features(sample_df, feature_names)
    within_feature_df = within_feature_variation(sample_df, z_features, feature_names)
    within_summary_df = within_dataset_summary(sample_df, z_features, feature_names)
    rep_euclidean_df, rep_cosine_df, rep_corr_df = representative_distance_matrices(representative_df, feature_names)
    pca_df = pca_projection(z_features, sample_df, feature_names)

    distance_long_df = pd.concat(
        [
            distance_long(rep_euclidean_df, "representative_euclidean_z"),
            distance_long(rep_cosine_df, "representative_cosine_distance_z"),
            distance_long(rep_corr_df, "representative_corr_distance_z"),
        ],
        ignore_index=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(output_dir / "sample_features.csv", index=False, float_format="%.9f")
    z_feature_output = pd.concat([sample_df[META_COLUMNS], z_features], axis=1)
    z_feature_output.to_csv(output_dir / "sample_features_zscored_for_analysis.csv", index=False, float_format="%.9f")
    representative_df.to_csv(output_dir / "dataset_representative_features.csv", index=False, float_format="%.9f")
    within_feature_df.to_csv(output_dir / "within_dataset_feature_variation.csv", index=False, float_format="%.9f")
    within_summary_df.to_csv(output_dir / "within_dataset_summary.csv", index=False, float_format="%.9f")
    rep_euclidean_df.to_csv(output_dir / "between_dataset_euclidean_distance.csv", float_format="%.9f")
    rep_cosine_df.to_csv(output_dir / "between_dataset_cosine_distance.csv", float_format="%.9f")
    rep_corr_df.to_csv(output_dir / "between_dataset_correlation_distance.csv", float_format="%.9f")
    distance_long_df.to_csv(output_dir / "between_dataset_distance_long.csv", index=False, float_format="%.9f")
    pca_df.to_csv(output_dir / "sample_features_pca.csv", index=False, float_format="%.9f")

    plots_dir = output_dir / "plots"
    save_within_variation_bar(within_summary_df, plots_dir / "within_dataset_variation_bar.png")
    save_heatmap(
        rep_euclidean_df,
        plots_dir / "between_dataset_euclidean_heatmap.png",
        "Between-dataset representative Euclidean distance",
    )
    save_heatmap(
        rep_corr_df,
        plots_dir / "between_dataset_correlation_heatmap.png",
        "Between-dataset representative correlation distance",
    )
    save_pca_plot(pca_df, plots_dir / "sample_feature_pca.png")
    save_pca_3d_plot(pca_df, plots_dir / "sample_feature_pca_3d.png")
    write_summary_text(output_dir / "analysis_summary.txt", args, within_summary_df, distance_long_df, pca_df)

    print(f"Wrote analysis outputs to {output_dir}")
    print(within_summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
