from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


@dataclass(frozen=True)
class ValidationLossPoint:
    dataset: str
    validation_dir: str
    run_dir: str
    epoch: int
    val_loss: float
    early_stopping_counter: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot validation loss trajectories for DSPBuilder meta leave-one-out runs."
    )
    parser.add_argument("--csv", type=Path, required=True, help="Input CSV exported from validation loss logs.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to loss.png in the input CSV directory.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="DSPBuilder Validation Loss",
        help="Figure title.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Image DPI.")
    return parser.parse_args()


def load_points(csv_path: Path) -> OrderedDict[str, list[ValidationLossPoint]]:
    grouped: OrderedDict[str, list[ValidationLossPoint]] = OrderedDict()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            point = ValidationLossPoint(
                dataset=row["dataset"],
                validation_dir=row["validation_dir"],
                run_dir=row["run_dir"],
                epoch=int(row["epoch"]),
                val_loss=float(row["val_loss"]),
                early_stopping_counter=int(row["early_stopping_counter"]),
            )
            grouped.setdefault(point.dataset, []).append(point)

    for dataset in grouped:
        grouped[dataset].sort(key=lambda point: point.epoch)
    return grouped


def build_color_cycle(num_series: int) -> list[tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab20")
    return [cmap(i) for i in range(num_series)]


def find_min_loss_point(points: list[ValidationLossPoint]) -> ValidationLossPoint:
    return min(points, key=lambda point: (point.val_loss, point.epoch))


def compute_y_limits(grouped: OrderedDict[str, list[ValidationLossPoint]]) -> tuple[float, float]:
    y_values = [point.val_loss for points in grouped.values() for point in points]
    y_min = min(y_values)
    y_max = max(y_values)
    if y_min == y_max:
        return y_min - 0.1, y_max + 0.1

    padding = (y_max - y_min) * 0.08
    return y_min - padding, y_max + padding


def plot_grouped_series(
    grouped: OrderedDict[str, list[ValidationLossPoint]],
    output_path: Path,
    title: str,
    dpi: int,
) -> None:
    colors = build_color_cycle(len(grouped))
    fig, ax = plt.subplots(figsize=(16, 9))

    for color, (dataset, points) in zip(colors, grouped.items()):
        epochs = [point.epoch for point in points]
        val_losses = [point.val_loss for point in points]

        ax.plot(
            epochs,
            val_losses,
            color=color,
            linestyle="-",
            linewidth=2.2,
            label=dataset,
            zorder=2,
        )

        min_point = find_min_loss_point(points)
        ax.scatter(
            min_point.epoch,
            min_point.val_loss,
            marker="o",
            s=85,
            color=color,
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
        )

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss (mean of pair-wise loss)")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(*compute_y_limits(grouped))

    all_epochs = [point.epoch for points in grouped.values() for point in points]
    if all_epochs:
        ax.set_xlim(min(all_epochs), max(all_epochs))

    dataset_legend = ax.legend(
        title="Dataset",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        ncol=1,
    )
    ax.add_artist(dataset_legend)

    style_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=2.2, label="Validation loss"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            linestyle="None",
            markeredgecolor="black",
            markerfacecolor="gray",
            markersize=8,
            label="Minimum loss",
        ),
    ]
    ax.legend(
        handles=style_handles,
        title="Markers",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        borderaxespad=0.0,
        frameon=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    csv_path = args.csv.resolve()
    output_path = args.output.resolve() if args.output is not None else csv_path.parent / "loss.png"

    grouped = load_points(csv_path)
    if not grouped:
        raise ValueError(f"No rows found in {csv_path}")
    plot_grouped_series(
        grouped=grouped,
        output_path=output_path,
        title=args.title,
        dpi=args.dpi,
    )
    print(f"Saved figure to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
