#!/usr/bin/env python3
"""Cluster 47 non-Benchmark datasets by catch22 PCA-space centroids.

The six Benchmark datasets (ECL/electricity, Weather, Illness, Traffic,
Exchange, and ETTh1/ETT-small) are excluded, leaving 18 Monash and 29 TIME
datasets.  For each dataset, 20 train windows are sampled without replacement.
catch22 is computed per channel and pooled feature-wise into one 22-dimensional
vector per window.

All 940 sample vectors are globally z-score normalized.  PCA is fitted to the
normalized sample matrix and the first 11 principal components are retained;
these components have already been verified to explain at least 90% of the
variance.  A dataset centroid is the mean of its 20 coordinates in this PCA
space.  Pairwise Euclidean distances between the 47 centroids are clustered
with k=2, 3, 4, 5, 6, 7, and 8 by average-linkage agglomerative clustering.

No visualization is produced.  The primary output summarizes the number of
datasets in every cluster for each value of k from 2 through 8.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


os.environ.setdefault("MPLCONFIGDIR", "/tmp/tslib_matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visualize_53_train_window_pca_groups import (  # noqa: E402
    EXPECTED_DATASET_COUNT,
    META_COLUMNS,
    DatasetSpec,
    build_registry,
    collect_features,
    relative_path,
    resolve_path,
    write_feature_manifest,
    zscore_features,
)


EXPECTED_RETAINED_DATASET_COUNT = 47
EXPECTED_RETAINED_FAMILY_COUNTS = {"Monash": 18, "TIME": 29}
EXCLUDED_BENCHMARK_NAMES = {
    "electricity",
    "ETT-small",
    "exchange_rate",
    "illness",
    "traffic",
    "weather",
}
DEFAULT_OUTPUT_DIR = Path("catch22/dataset_centroid_clusters_47_pca90_k8")
PCA_COMPONENT_COUNT = 11
CLUSTER_COUNTS = tuple(range(2, 9))


def canonical_dataset_name(dataset_id: str, family: str) -> str:
    if family == "Monash":
        return dataset_id.removeprefix("Monash__")
    if family == "TIME":
        return dataset_id.removeprefix("TIME__")
    return dataset_id


def retained_specs(args: argparse.Namespace, repo_root: Path) -> list[DatasetSpec]:
    all_specs = build_registry(args, repo_root)
    if len(all_specs) != EXPECTED_DATASET_COUNT:
        raise ValueError(f"Expected 53 registry entries, found {len(all_specs)}")

    actual_benchmark_names = {spec.name for spec in all_specs if spec.family == "Benchmark"}
    if actual_benchmark_names != EXCLUDED_BENCHMARK_NAMES:
        raise ValueError(
            "Benchmark exclusion set differs from the registry: "
            f"expected={sorted(EXCLUDED_BENCHMARK_NAMES)}, "
            f"actual={sorted(actual_benchmark_names)}"
        )
    specs = [spec for spec in all_specs if spec.family != "Benchmark"]
    family_counts = pd.Series([spec.family for spec in specs]).value_counts().to_dict()
    if len(specs) != EXPECTED_RETAINED_DATASET_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_RETAINED_DATASET_COUNT} retained datasets, found {len(specs)}"
        )
    if family_counts != EXPECTED_RETAINED_FAMILY_COUNTS:
        raise ValueError(
            f"Expected family counts {EXPECTED_RETAINED_FAMILY_COUNTS}, found {family_counts}"
        )
    return specs


def pca_fixed_components(
    z_features: pd.DataFrame,
    component_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return coordinates, components, and ratios for a fixed PC count."""
    matrix = z_features.to_numpy(dtype=np.float64)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    variances = singular_values**2
    total_variance = float(variances.sum())
    if total_variance <= 0.0:
        raise ValueError("PCA input has zero total variance")
    if component_count > vt.shape[0]:
        raise ValueError(
            f"Requested {component_count} PCA components, but only {vt.shape[0]} are available"
        )
    explained_ratios = variances / total_variance
    coordinates = centered @ vt[:component_count].T
    return coordinates, vt[:component_count], explained_ratios[:component_count]


def sample_pca_frame(
    feature_df: pd.DataFrame,
    coordinates: np.ndarray,
    explained_ratios: np.ndarray,
) -> pd.DataFrame:
    output = feature_df[META_COLUMNS].copy()
    for index in range(coordinates.shape[1]):
        output[f"pc{index + 1}"] = coordinates[:, index]
    output["retained_pc_count"] = coordinates.shape[1]
    output["retained_explained_variance_ratio"] = float(explained_ratios.sum())
    return output


def dataset_centroids(
    pca_df: pd.DataFrame,
    component_count: int,
) -> pd.DataFrame:
    pc_columns = [f"pc{index}" for index in range(1, component_count + 1)]
    centroids = (
        pca_df.groupby(["dataset", "family"], sort=False)[pc_columns]
        .mean()
        .reset_index()
        .rename(columns={"dataset": "dataset_id"})
    )
    centroids.insert(
        1,
        "dataset_name",
        [
            canonical_dataset_name(str(row.dataset_id), str(row.family))
            for row in centroids.itertuples(index=False)
        ],
    )
    if len(centroids) != EXPECTED_RETAINED_DATASET_COUNT:
        raise RuntimeError(f"Expected 47 dataset centroids, got {len(centroids)}")
    return centroids


def pairwise_euclidean(matrix: np.ndarray) -> np.ndarray:
    differences = matrix[:, None, :] - matrix[None, :, :]
    squared = np.einsum("ijk,ijk->ij", differences, differences)
    distances = np.sqrt(np.maximum(squared, 0.0))
    np.fill_diagonal(distances, 0.0)
    return distances


def agglomerative_labels(distance_matrix: np.ndarray, cluster_count: int) -> np.ndarray:
    kwargs = {
        "n_clusters": cluster_count,
        "linkage": "average",
    }
    try:
        model = AgglomerativeClustering(metric="precomputed", **kwargs)
    except TypeError:
        # Compatibility with older scikit-learn releases.
        model = AgglomerativeClustering(affinity="precomputed", **kwargs)
    return model.fit_predict(distance_matrix).astype(np.int64, copy=False)


def stable_cluster_ids(
    raw_labels: np.ndarray,
    dataset_names: list[str],
) -> np.ndarray:
    """Remap arbitrary sklearn labels by each cluster's first dataset name."""
    members: dict[int, list[str]] = {}
    for label, name in zip(raw_labels, dataset_names):
        members.setdefault(int(label), []).append(name)
    ordered_labels = sorted(members, key=lambda label: min(members[label]).lower())
    mapping = {label: index for index, label in enumerate(ordered_labels, start=1)}
    return np.asarray([mapping[int(label)] for label in raw_labels], dtype=np.int64)


def cluster_membership_frame(
    centroids: pd.DataFrame,
    cluster_ids: np.ndarray,
    distance_matrix: np.ndarray,
) -> pd.DataFrame:
    output = centroids[["dataset_id", "dataset_name", "family"]].copy()
    output.insert(0, "cluster_id", cluster_ids)
    sizes = output.groupby("cluster_id")["dataset_id"].transform("size")
    output.insert(1, "cluster_size", sizes.astype(int))

    nearest_names: list[str] = []
    nearest_distances: list[float] = []
    names = output["dataset_name"].tolist()
    for row_index in range(len(output)):
        candidate_indices = np.flatnonzero(cluster_ids == cluster_ids[row_index])
        candidate_indices = candidate_indices[candidate_indices != row_index]
        if len(candidate_indices) == 0:
            nearest_names.append("")
            nearest_distances.append(float("nan"))
            continue
        nearest_index = int(
            candidate_indices[
                np.argmin(distance_matrix[row_index, candidate_indices])
            ]
        )
        nearest_names.append(names[nearest_index])
        nearest_distances.append(float(distance_matrix[row_index, nearest_index]))
    output["nearest_dataset_in_cluster"] = nearest_names
    output["nearest_centroid_distance"] = nearest_distances
    return output.sort_values(
        ["cluster_id", "dataset_name"], key=lambda column: column.astype(str).str.lower()
    ).reset_index(drop=True)


def cluster_summary_frame(membership: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cluster_id, group in membership.groupby("cluster_id", sort=True):
        names = sorted(group["dataset_name"].astype(str), key=str.lower)
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_size": len(names),
                "dataset_names": ";".join(names),
            }
        )
    return pd.DataFrame(rows)


def cluster_size_summary_frame(
    summaries: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Combine per-k cluster dataset counts into one tidy table."""
    rows: list[dict[str, int]] = []
    for cluster_count, summary in summaries.items():
        for row in summary.itertuples(index=False):
            rows.append(
                {
                    "cluster_count": cluster_count,
                    "cluster_id": int(row.cluster_id),
                    "dataset_count": int(row.cluster_size),
                }
            )
    return pd.DataFrame(rows)


def component_manifest(
    components: np.ndarray,
    explained_ratios: np.ndarray,
) -> pd.DataFrame:
    cumulative = np.cumsum(explained_ratios)
    return pd.DataFrame(
        {
            "component": np.arange(1, len(explained_ratios) + 1),
            "explained_variance_ratio": explained_ratios,
            "cumulative_explained_variance_ratio": cumulative,
        }
    )


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
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--label-len", type=int, default=48)
    parser.add_argument("--pred-len", type=int, default=96)
    # Required by the shared loader helper, although illness is excluded.
    parser.add_argument("--illness-seq-len", type=int, default=36)
    parser.add_argument("--illness-label-len", type=int, default=18)
    parser.add_argument("--illness-pred-len", type=int, default=36)
    parser.add_argument("--pooling", choices=("mean", "median"), default="mean")
    parser.add_argument("--no-scale", action="store_true")
    parser.add_argument("--multi-series-lru-size", type=int, default=8)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "samples-per-dataset": args.samples_per_dataset,
        "seq-len": args.seq_len,
        "label-len": args.label_len,
        "pred-len": args.pred_len,
        "multi-series-lru-size": args.multi_series_lru_size,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"Arguments must be positive: {invalid}")
    if args.label_len > args.seq_len:
        raise ValueError("label-len cannot exceed seq-len")


def write_summary(
    path: Path,
    args: argparse.Namespace,
    feature_rows: int,
    component_count: int,
    explained_ratio: float,
    cluster_summaries: dict[int, pd.DataFrame],
) -> None:
    lines = [
        "47-dataset centroid clustering from z-scored catch22 PCA",
        "",
        f"excluded_benchmarks: {sorted(EXCLUDED_BENCHMARK_NAMES)}",
        f"retained_dataset_count: {EXPECTED_RETAINED_DATASET_COUNT}",
        f"retained_family_counts: {EXPECTED_RETAINED_FAMILY_COUNTS}",
        f"seed: {args.seed}",
        f"samples_per_dataset: {args.samples_per_dataset}",
        f"feature_rows: {feature_rows}",
        "catch22_dimension: 22",
        f"channel_pooling: {args.pooling}",
        f"input_scaling: {'none' if args.no_scale else 'Time-Series-Library train standardization'}",
        "normalization: global feature-wise z-score over all sampled vectors",
        "pca_component_selection: fixed",
        f"retained_pc_count: {component_count}",
        f"retained_cumulative_explained_variance_ratio: {explained_ratio:.12f}",
        f"dataset_centroid: mean of {args.samples_per_dataset} sample coordinates "
        "in retained PCA space",
        "distance: Euclidean between dataset centroids",
        "clustering: agglomerative, average linkage, precomputed distance matrix",
        f"cluster_counts: {list(CLUSTER_COUNTS)}",
        *[
            f"cluster_sizes_k{cluster_count}: "
            f"{cluster_summaries[cluster_count]['cluster_size'].tolist()}"
            for cluster_count in CLUSTER_COUNTS
        ],
        "visualization: none",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    validate_args(args)
    repo_root = args.repo_root.resolve()
    output_dir = resolve_path(args.output_dir, repo_root).resolve()
    specs = retained_specs(args, repo_root)

    feature_df, feature_names, feature_short_names = collect_features(
        [specs], args, repo_root
    )
    expected_feature_rows = EXPECTED_RETAINED_DATASET_COUNT * args.samples_per_dataset
    if len(feature_df) != expected_feature_rows:
        raise RuntimeError(
            f"Expected {expected_feature_rows} sampled feature rows, got {len(feature_df)}"
        )
    z_feature_df = zscore_features(feature_df, feature_short_names)
    z_feature_df.columns = feature_short_names
    component_count = PCA_COMPONENT_COUNT
    coordinates, components, explained_ratios = pca_fixed_components(
        z_feature_df, component_count
    )
    retained_ratio = float(explained_ratios.sum())
    pca_df = sample_pca_frame(feature_df, coordinates, explained_ratios)
    centroids = dataset_centroids(pca_df, component_count)
    pc_columns = [f"pc{index}" for index in range(1, component_count + 1)]
    centroid_matrix = centroids[pc_columns].to_numpy(dtype=np.float64)
    distance_matrix = pairwise_euclidean(centroid_matrix)
    memberships: dict[int, pd.DataFrame] = {}
    cluster_summaries: dict[int, pd.DataFrame] = {}
    for cluster_count in CLUSTER_COUNTS:
        raw_labels = agglomerative_labels(distance_matrix, cluster_count)
        cluster_ids = stable_cluster_ids(raw_labels, centroids["dataset_name"].tolist())
        membership = cluster_membership_frame(centroids, cluster_ids, distance_matrix)
        memberships[cluster_count] = membership
        cluster_summaries[cluster_count] = cluster_summary_frame(membership)
    cluster_size_summary = cluster_size_summary_frame(cluster_summaries)

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(
        output_dir / "sample_catch22_features.csv", index=False, float_format="%.9f"
    )
    pd.concat(
        [feature_df[META_COLUMNS].reset_index(drop=True), z_feature_df.reset_index(drop=True)],
        axis=1,
    ).to_csv(
        output_dir / "sample_catch22_features_zscored.csv",
        index=False,
        float_format="%.9f",
    )
    pca_df.to_csv(
        output_dir / "sample_catch22_pca_retained.csv", index=False, float_format="%.9f"
    )
    centroids.to_csv(
        output_dir / "dataset_centroids_pca_retained.csv", index=False, float_format="%.9f"
    )
    pd.DataFrame(
        distance_matrix,
        index=centroids["dataset_name"],
        columns=centroids["dataset_name"],
    ).to_csv(output_dir / "dataset_centroid_distance_matrix.csv", float_format="%.9f")
    for cluster_count in CLUSTER_COUNTS:
        memberships[cluster_count].to_csv(
            output_dir / f"dataset_clusters_k{cluster_count}.csv",
            index=False,
            float_format="%.9f",
        )
        cluster_summaries[cluster_count].to_csv(
            output_dir / f"cluster_summary_k{cluster_count}.csv", index=False
        )
    cluster_size_summary.to_csv(
        output_dir / "cluster_size_summary_k2_to_k8.csv", index=False
    )
    component_manifest(components, explained_ratios).to_csv(
        output_dir / "pca_component_summary.csv", index=False, float_format="%.12f"
    )
    write_feature_manifest(
        output_dir / "catch22_feature_manifest.csv", feature_names, feature_short_names
    )
    write_summary(
        output_dir / "summary.txt",
        args,
        len(feature_df),
        component_count,
        retained_ratio,
        cluster_summaries,
    )

    print(f"Datasets: {len(specs)} ({EXPECTED_RETAINED_FAMILY_COUNTS})")
    print(f"Feature rows: {len(feature_df)} x {len(feature_short_names)}")
    print(
        f"Retained PCs: {component_count}, cumulative explained variance: "
        f"{retained_ratio:.6%}"
    )
    for cluster_count in CLUSTER_COUNTS:
        sizes = cluster_summaries[cluster_count]["cluster_size"].tolist()
        print(f"Clusters k={cluster_count}: sizes={sizes}")
    print(f"Cluster size CSV: {output_dir / 'cluster_size_summary_k2_to_k8.csv'}")
    print(f"Output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
