from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tslib_matplotlib")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "catch22" / "monash_sample_pca" / "sample_catch22_pca_2d.csv"
DEFAULT_OUTPUT = REPO_ROOT / "catch22" / "monash_sample_pca" / "sample_catch22_pca_2d_cluster_circles.png"
DEFAULT_SUMMARY = REPO_ROOT / "catch22" / "monash_sample_pca" / "sample_catch22_pca_2d_cluster_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw per-dataset circular clusters on top of an existing Monash catch22 PCA result."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input PCA CSV produced by visualize_monash_sample_pca.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Output CSV containing cluster centroids and radii.",
    )
    parser.add_argument(
        "--radius",
        choices=("max", "rms", "mean"),
        default="max",
        help=(
            "Circle radius definition. `max` contains all sampled points in each dataset cluster."
        ),
    )
    parser.add_argument(
        "--min-visible-radius-frac",
        type=float,
        default=0.008,
        help=(
            "Minimum display radius as a fraction of the PCA span. Used only for singleton clusters "
            "or visually tiny clusters; true_radius is still recorded in the summary CSV."
        ),
    )
    parser.add_argument("--dpi", type=int, default=240, help="Output figure DPI.")
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Do not draw dataset names at cluster centroids.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def compute_radius(distances: np.ndarray, mode: str) -> float:
    if len(distances) == 0:
        return 0.0
    if mode == "max":
        return float(np.max(distances))
    if mode == "rms":
        return float(np.sqrt(np.mean(distances * distances)))
    if mode == "mean":
        return float(np.mean(distances))
    raise ValueError(f"Unsupported radius mode: {mode}")


def build_cluster_summary(df: pd.DataFrame, radius_mode: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for dataset, group in df.groupby("dataset", sort=True):
        pc = group[["pc1", "pc2"]].to_numpy(dtype=float)
        center = pc.mean(axis=0)
        distances = np.linalg.norm(pc - center[None, :], axis=1)
        records.append(
            {
                "dataset": dataset,
                "n_samples": len(group),
                "center_pc1": center[0],
                "center_pc2": center[1],
                "radius_mode": radius_mode,
                "true_radius": compute_radius(distances, radius_mode),
                "max_radius": float(np.max(distances)) if len(distances) else 0.0,
                "mean_radius": float(np.mean(distances)) if len(distances) else 0.0,
                "rms_radius": float(np.sqrt(np.mean(distances * distances))) if len(distances) else 0.0,
            }
        )
    return pd.DataFrame(records)


def pca_span(df: pd.DataFrame) -> float:
    x_span = float(df["pc1"].max() - df["pc1"].min())
    y_span = float(df["pc2"].max() - df["pc2"].min())
    return max(x_span, y_span, 1.0)


def save_cluster_plot(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_path: Path,
    dpi: int,
    min_visible_radius_frac: float,
    draw_labels: bool,
) -> None:
    dataset_names = summary_df["dataset"].tolist()
    cmap = plt.get_cmap("nipy_spectral")
    colors = {
        dataset: cmap(index / max(1, len(dataset_names) - 1))
        for index, dataset in enumerate(dataset_names)
    }
    min_visible_radius = pca_span(df) * max(0.0, min_visible_radius_frac)

    fig, ax = plt.subplots(figsize=(14, 10), constrained_layout=True)
    for _, row in summary_df.iterrows():
        dataset = str(row["dataset"])
        color = colors[dataset]
        group = df[df["dataset"] == dataset]
        true_radius = float(row["true_radius"])
        display_radius = max(true_radius, min_visible_radius)
        linestyle = "--" if true_radius == 0.0 else "-"

        circle = Circle(
            (float(row["center_pc1"]), float(row["center_pc2"])),
            display_radius,
            facecolor=color,
            edgecolor=color,
            alpha=0.09,
            linewidth=1.5,
            linestyle=linestyle,
        )
        ax.add_patch(circle)
        ax.scatter(
            group["pc1"],
            group["pc2"],
            s=35,
            color=color,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.35,
        )
        ax.scatter(
            [float(row["center_pc1"])],
            [float(row["center_pc2"])],
            marker="x",
            s=35,
            color=color,
            linewidths=1.2,
        )
        if draw_labels:
            ax.text(
                float(row["center_pc1"]),
                float(row["center_pc2"]),
                dataset,
                fontsize=6,
                color=color,
                weight="bold",
            )

    pc1_ratio = float(df["pc1_explained_ratio"].iloc[0]) if "pc1_explained_ratio" in df.columns else 0.0
    pc2_ratio = float(df["pc2_explained_ratio"].iloc[0]) if "pc2_explained_ratio" in df.columns else 0.0
    ax.axhline(0.0, color="#bbbbbb", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="#bbbbbb", linewidth=0.8, zorder=0)
    ax.set_xlabel(f"PC1 ({pc1_ratio:.2%})")
    ax.set_ylabel(f"PC2 ({pc2_ratio:.2%})")
    ax.set_title("Monash catch22 PCA with per-dataset circular clusters")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.55)
    ax.set_aspect("equal", adjustable="datalim")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    summary_path = resolve_path(args.summary_output)

    df = pd.read_csv(input_path)
    required_columns = {"dataset", "pc1", "pc2"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Input PCA CSV is missing required columns: {sorted(missing)}")

    summary_df = build_cluster_summary(df, radius_mode=args.radius)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False, float_format="%.9f")
    save_cluster_plot(
        df=df,
        summary_df=summary_df,
        output_path=output_path,
        dpi=args.dpi,
        min_visible_radius_frac=args.min_visible_radius_frac,
        draw_labels=not args.no_labels,
    )

    singleton_count = int((summary_df["n_samples"] == 1).sum())
    print(f"Read {len(df)} PCA sample rows from {input_path}")
    print(f"Drew {len(summary_df)} dataset cluster circles")
    print(f"Singleton clusters with dashed minimum-radius circles: {singleton_count}")
    print(f"Cluster summary CSV: {summary_path}")
    print(f"Cluster circle PCA plot: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
