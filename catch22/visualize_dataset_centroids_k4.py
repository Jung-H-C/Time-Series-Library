from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tslib_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = (
    REPO_ROOT / "catch22" / "dataset_centroid_clusters_47_pca90_k8"
)
DEFAULT_CLUSTER_SUMMARIES = tuple(
    DEFAULT_INPUT_DIR / f"cluster_summary_k{cluster_count}.csv"
    for cluster_count in (5, 7, 8)
)
CLUSTER_COLORS = {
    1: "#4C78A8",
    2: "#F58518",
    3: "#54A24B",
    4: "#E45756",
    5: "#B279A2",
    6: "#72B7B2",
    7: "#FF9DA6",
    8: "#9D755D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate PC1-PC2 plots of the 47 dataset centroids for the K=5, "
            "K=7, and K=8 cluster summaries."
        )
    )
    parser.add_argument(
        "--centroids-csv",
        type=Path,
        default=DEFAULT_INPUT_DIR / "dataset_centroids_pca_retained.csv",
        help="CSV containing dataset_name and retained PCA coordinates.",
    )
    parser.add_argument(
        "--cluster-summary-csvs",
        type=Path,
        nargs="+",
        default=list(DEFAULT_CLUSTER_SUMMARIES),
        help="One or more cluster_summary_kN.csv files.",
    )
    parser.add_argument(
        "--pca-summary-csv",
        type=Path,
        default=DEFAULT_INPUT_DIR / "pca_component_summary.csv",
        help="PCA explained-variance summary used for axis labels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory for dataset_centroids_pca_pc1_pc2_kN.png outputs.",
    )
    parser.add_argument("--dpi", type=int, default=240, help="Output image DPI.")
    parser.add_argument(
        "--label-mode",
        choices=("index", "name", "none"),
        default="index",
        help=(
            "Point-label style. 'index' puts compact numeric IDs on points and a "
            "cluster-grouped dataset key beside the plot."
        ),
    )
    parser.add_argument(
        "--show-cluster-centroids",
        action="store_true",
        help="Also draw the arithmetic mean of the displayed points in each cluster.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_cluster_assignments(path: Path) -> pd.DataFrame:
    summary = pd.read_csv(path)
    required = {"cluster_id", "cluster_size", "dataset_names"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    rows: list[dict[str, int | str]] = []
    for row in summary.itertuples(index=False):
        cluster_id = int(row.cluster_id)
        dataset_names = [
            name.strip() for name in str(row.dataset_names).split(";") if name.strip()
        ]
        if len(dataset_names) != int(row.cluster_size):
            raise ValueError(
                f"Cluster {cluster_id} declares size {row.cluster_size}, "
                f"but contains {len(dataset_names)} dataset names"
            )
        rows.extend(
            {"dataset_name": dataset_name, "cluster_id": cluster_id}
            for dataset_name in dataset_names
        )

    assignments = pd.DataFrame(rows)
    cluster_ids = tuple(sorted(assignments["cluster_id"].unique()))
    expected_cluster_ids = tuple(range(1, len(cluster_ids) + 1))
    if cluster_ids != expected_cluster_ids:
        raise ValueError(
            f"Expected consecutive cluster ids {expected_cluster_ids}, got {cluster_ids}"
        )
    if len(cluster_ids) > len(CLUSTER_COLORS):
        raise ValueError(
            f"At most {len(CLUSTER_COLORS)} clusters are supported, got {len(cluster_ids)}"
        )
    duplicates = assignments.loc[
        assignments["dataset_name"].duplicated(keep=False), "dataset_name"
    ].tolist()
    if duplicates:
        raise ValueError(f"Datasets assigned to multiple clusters: {sorted(set(duplicates))}")
    return assignments


def load_plot_frame(centroids_path: Path, cluster_path: Path) -> pd.DataFrame:
    centroids = pd.read_csv(centroids_path)
    required = {"dataset_name", "pc1", "pc2"}
    missing = required.difference(centroids.columns)
    if missing:
        raise ValueError(f"{centroids_path} is missing columns: {sorted(missing)}")
    if centroids["dataset_name"].duplicated().any():
        duplicates = centroids.loc[
            centroids["dataset_name"].duplicated(keep=False), "dataset_name"
        ].tolist()
        raise ValueError(f"Duplicate centroid rows: {sorted(set(duplicates))}")

    assignments = load_cluster_assignments(cluster_path)
    centroid_names = set(centroids["dataset_name"])
    assigned_names = set(assignments["dataset_name"])
    missing_assignments = sorted(centroid_names - assigned_names)
    missing_centroids = sorted(assigned_names - centroid_names)
    if missing_assignments or missing_centroids:
        raise ValueError(
            "Centroid/cluster dataset mismatch: "
            f"missing_assignments={missing_assignments}, "
            f"missing_centroids={missing_centroids}"
        )

    output = centroids.merge(assignments, on="dataset_name", validate="one_to_one")
    return output.sort_values(["cluster_id", "dataset_name"]).reset_index(drop=True)


def explained_variance_labels(path: Path) -> tuple[str, str, float]:
    summary = pd.read_csv(path)
    required = {"component", "explained_variance_ratio"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    ratios = summary.set_index("component")["explained_variance_ratio"]
    if 1 not in ratios or 2 not in ratios:
        raise ValueError(f"{path} must contain PCA components 1 and 2")
    pc1_ratio = float(ratios.loc[1])
    pc2_ratio = float(ratios.loc[2])
    return (
        f"PC1 ({pc1_ratio * 100:.2f}% explained variance)",
        f"PC2 ({pc2_ratio * 100:.2f}% explained variance)",
        pc1_ratio + pc2_ratio,
    )


def plot_centroids(
    frame: pd.DataFrame,
    x_label: str,
    y_label: str,
    displayed_variance: float,
    output_path: Path,
    cluster_count: int,
    dpi: int,
    label_mode: str,
    show_cluster_centroids: bool,
) -> None:
    if label_mode == "index":
        figure = plt.figure(figsize=(23, 13), constrained_layout=True)
        grid = figure.add_gridspec(1, 2, width_ratios=(4.2, 1.8))
        axis = figure.add_subplot(grid[0, 0])
        key_axis = figure.add_subplot(grid[0, 1])
        key_axis.set_axis_off()
    else:
        figure, axis = plt.subplots(figsize=(18, 13), constrained_layout=True)
        key_axis = None

    frame = frame.copy()
    frame["plot_id"] = range(1, len(frame) + 1)

    for cluster_id, group in frame.groupby("cluster_id", sort=True):
        cluster_id = int(cluster_id)
        color = CLUSTER_COLORS[cluster_id]
        axis.scatter(
            group["pc1"],
            group["pc2"],
            s=175 if label_mode == "index" else 105,
            color=color,
            edgecolor="white",
            linewidth=0.9,
            alpha=0.92,
            label=f"Cluster {cluster_id} (n={len(group)})",
            zorder=3,
        )

        if show_cluster_centroids:
            axis.scatter(
                group["pc1"].mean(),
                group["pc2"].mean(),
                marker="X",
                s=240,
                color=color,
                edgecolor="black",
                linewidth=1.0,
                zorder=5,
            )

        if label_mode == "index":
            for row in group.itertuples(index=False):
                axis.text(
                    row.pc1,
                    row.pc2,
                    str(row.plot_id),
                    ha="center",
                    va="center",
                    fontsize=6.3,
                    fontweight="bold",
                    color="white",
                    zorder=4,
                )
        elif label_mode == "name":
            offsets = ((5, 5), (5, -10), (-5, 5), (-5, -10))
            horizontal_alignments = ("left", "left", "right", "right")
            for label_index, row in enumerate(group.itertuples(index=False)):
                offset = offsets[label_index % len(offsets)]
                axis.annotate(
                    row.dataset_name,
                    xy=(row.pc1, row.pc2),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=7.0,
                    color="#202020",
                    ha=horizontal_alignments[label_index % len(offsets)],
                    va="bottom" if offset[1] > 0 else "top",
                    zorder=4,
                )

    if key_axis is not None:
        total_lines = len(frame) + frame["cluster_id"].nunique()
        line_step = 0.96 / total_lines
        y_position = 0.99
        for cluster_id, group in frame.groupby("cluster_id", sort=True):
            cluster_id = int(cluster_id)
            key_axis.text(
                0.01,
                y_position,
                f"Cluster {cluster_id} (n={len(group)})",
                transform=key_axis.transAxes,
                ha="left",
                va="top",
                fontsize=9.2,
                fontweight="bold",
                color=CLUSTER_COLORS[cluster_id],
            )
            y_position -= line_step
            for row in group.itertuples(index=False):
                key_axis.text(
                    0.03,
                    y_position,
                    f"{row.plot_id:02d}  {row.dataset_name}",
                    transform=key_axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7.0,
                    color="#202020",
                    family="monospace",
                )
                y_position -= line_step

    axis.axhline(0.0, color="#9A9A9A", linewidth=0.8, alpha=0.65, zorder=1)
    axis.axvline(0.0, color="#9A9A9A", linewidth=0.8, alpha=0.65, zorder=1)
    axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.28, zorder=0)
    axis.margins(x=0.08, y=0.08)
    axis.set_xlabel(x_label, fontsize=12)
    axis.set_ylabel(y_label, fontsize=12)
    axis.set_title(
        f"Dataset Centroids on the PC1-PC2 Plane, Colored by K={cluster_count} Cluster\n"
        f"K={cluster_count} clustering used the retained 11D PCA space; "
        "displayed variance: "
        f"{displayed_variance * 100:.2f}%",
        fontsize=15,
        pad=15,
    )
    axis.legend(title="Cluster assignment", loc="best", frameon=True, fontsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")

    centroids_path = resolve_path(args.centroids_csv)
    pca_summary_path = resolve_path(args.pca_summary_csv)
    output_dir = resolve_path(args.output_dir)

    x_label, y_label, displayed_variance = explained_variance_labels(pca_summary_path)
    generated_cluster_counts: set[int] = set()
    for raw_cluster_path in args.cluster_summary_csvs:
        cluster_path = resolve_path(raw_cluster_path)
        frame = load_plot_frame(centroids_path, cluster_path)
        cluster_count = int(frame["cluster_id"].nunique())
        if cluster_count in generated_cluster_counts:
            raise ValueError(f"Multiple input files describe K={cluster_count} clustering")
        generated_cluster_counts.add(cluster_count)
        output_path = output_dir / f"dataset_centroids_pca_pc1_pc2_k{cluster_count}.png"
        plot_centroids(
            frame=frame,
            x_label=x_label,
            y_label=y_label,
            displayed_variance=displayed_variance,
            output_path=output_path,
            cluster_count=cluster_count,
            dpi=args.dpi,
            label_mode=args.label_mode,
            show_cluster_centroids=args.show_cluster_centroids,
        )
        cluster_sizes = frame.groupby("cluster_id").size().to_dict()
        print(
            f"K={cluster_count}: plotted {len(frame)} dataset centroids; "
            f"cluster_sizes={cluster_sizes}"
        )
        print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
