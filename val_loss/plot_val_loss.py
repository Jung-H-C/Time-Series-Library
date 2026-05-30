#!/usr/bin/env python3
"""Plot validation loss curves from the val_loss log files."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path("/tmp/tslib_matplotlib")
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


LOG_PATTERN = re.compile(r"epoch=(\d+)\s+val_loss=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")

SERIES = [
    ("Weather", "val_loss_Weather.txt", "#d62728"),
    ("Traffic", "val_loss_Traffic.txt", "#ff7f0e"),
    ("ECL", "val_loss_ECL.txt", "#f1c40f"),
    ("ETTh1", "val_loss_ETTh1.txt", "#2ca02c"),
    ("Exchange", "val_loss_Exchange.txt", "#1f77b4"),
    ("ILI", "val_loss_ILI.txt", "#1f3a93"),
]


def parse_val_loss_log(path: Path) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = LOG_PATTERN.search(line)
            if match:
                points.append((int(match.group(1)), float(match.group(2))))

    if not points:
        raise ValueError(f"No validation loss points found in {path}")
    return points


def plot_validation_losses(log_dir: Path, output_path: Path) -> dict[str, tuple[int, float]]:
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    best_points: dict[str, tuple[int, float]] = {}

    for label, filename, color in SERIES:
        points = parse_val_loss_log(log_dir / filename)
        epochs = [epoch for epoch, _ in points]
        losses = [loss for _, loss in points]
        best_index = min(range(len(losses)), key=losses.__getitem__)
        best_epoch = epochs[best_index]
        best_loss = losses[best_index]
        best_points[label] = (best_epoch, best_loss)

        ax.plot(epochs, losses, label=label, color=color, linewidth=2.0)
        ax.scatter(
            [best_epoch],
            [best_loss],
            s=110,
            facecolors="white",
            edgecolors=color,
            linewidths=2.5,
            zorder=5,
        )

    ax.set_title("Validation Loss by Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_xlim(0, 100)
    ax.set_ylim(0.45, 1.0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend(title="Left-One-Out", frameon=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return best_points


def parse_args() -> argparse.Namespace:
    default_log_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Visualize validation loss curves for the six benchmark logs."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=default_log_dir,
        help="Directory containing val_loss_*.txt log files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_log_dir / "val_loss_curves.png",
        help="PNG output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    best_points = plot_validation_losses(args.log_dir, args.output)
    print(f"Saved plot to {args.output}")
    for label, (epoch, loss) in best_points.items():
        print(f"{label}: best epoch={epoch}, val_loss={loss:.6f}")


if __name__ == "__main__":
    main()
