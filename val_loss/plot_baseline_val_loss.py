#!/usr/bin/env python3
"""Plot baseline validation loss curves from baseline.csv."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path("/tmp/tslib_matplotlib")
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


SERIES = [
    ("Weather", "#d62728"),
    ("Traffic", "#ff7f0e"),
    ("ECL", "#f1c40f"),
    ("Etth1", "#2ca02c"),
    ("Exchange", "#1f77b4"),
    ("ILI", "#1f3a93"),
]


def load_baseline_losses(csv_path: Path) -> dict[str, list[tuple[int, float]]]:
    selected = {dataset for dataset, _ in SERIES}
    losses: dict[str, list[tuple[int, float]]] = {dataset: [] for dataset in selected}

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"dataset", "epoch", "val_loss"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required column(s) in {csv_path}: {missing}")

        for row in reader:
            dataset = row["dataset"]
            if dataset not in selected:
                continue
            losses[dataset].append((int(row["epoch"]), float(row["val_loss"])))

    empty_datasets = [dataset for dataset, points in losses.items() if not points]
    if empty_datasets:
        missing = ", ".join(sorted(empty_datasets))
        raise ValueError(f"No baseline rows found for dataset(s): {missing}")

    for points in losses.values():
        points.sort(key=lambda item: item[0])

    return losses


def plot_baseline_losses(csv_path: Path, output_path: Path) -> dict[str, tuple[int, float]]:
    losses_by_dataset = load_baseline_losses(csv_path)
    best_points: dict[str, tuple[int, float]] = {}

    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

    for dataset, color in SERIES:
        points = losses_by_dataset[dataset]
        epochs = [epoch for epoch, _ in points]
        losses = [loss for _, loss in points]
        best_index = min(range(len(losses)), key=losses.__getitem__)
        best_epoch = epochs[best_index]
        best_loss = losses[best_index]
        best_points[dataset] = (best_epoch, best_loss)

        ax.plot(epochs, losses, label=dataset, color=color, linewidth=2.0)
        ax.scatter(
            [best_epoch],
            [best_loss],
            s=110,
            facecolors="white",
            edgecolors=color,
            linewidths=2.5,
            zorder=5,
        )

    ax.set_title("Baseline Validation Loss by Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_xlim(0, 100)
    ax.set_ylim(0.45, 1.0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend(title="Val Dataset", frameon=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return best_points


def parse_args() -> argparse.Namespace:
    default_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Visualize baseline validation loss curves for six selected datasets."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=default_dir / "baseline.csv",
        help="Path to baseline.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_dir / "baseline_val_loss_curves_2.png",
        help="PNG output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    best_points = plot_baseline_losses(args.csv, args.output)
    print(f"Saved plot to {args.output}")
    for dataset, (epoch, loss) in best_points.items():
        print(f"{dataset}: best epoch={epoch}, val_loss={loss:.6f}")


if __name__ == "__main__":
    main()
