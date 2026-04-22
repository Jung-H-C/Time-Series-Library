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
class ValidationPoint:
    dataset: str
    run_name: str
    epoch: int
    spearman_mean: float
    best_epoch: int | None
    baseline_best_proxy: str
    baseline_coefficient: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot validation spearman trajectories for DSPBuilder meta leave-one-out runs."
    )
    parser.add_argument("--csv", type=Path, required=True, help="Input CSV exported from validation logs.")
    parser.add_argument("--output", type=Path, required=True, help="Output image path, e.g. .png")
    parser.add_argument(
        "--title",
        type=str,
        default="DSPBuilder Validation Result (Reg_loss = 0.5)",
        help="Figure title.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Image DPI.")
    return parser.parse_args()


def parse_optional_float(raw_value: str) -> float | None:
    value = raw_value.strip()
    if value == "" or value.lower() == "none":
        return None
    return float(value)


def load_points(csv_path: Path) -> OrderedDict[str, list[ValidationPoint]]:
    grouped: OrderedDict[str, list[ValidationPoint]] = OrderedDict()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            point = ValidationPoint(
                dataset=row["dataset"],
                run_name=row["run_name"],
                epoch=int(row["epoch"]),
                spearman_mean=float(row["spearman_mean"]),
                best_epoch=int(row["best_epoch"]) if row["best_epoch"] else None,
                baseline_best_proxy=row["baseline_best_proxy"],
                baseline_coefficient=parse_optional_float(row["baseline_coefficient"]),
            )
            grouped.setdefault(point.dataset, []).append(point)

    for dataset in grouped:
        grouped[dataset].sort(key=lambda point: point.epoch)
    return grouped


def build_color_cycle(num_series: int) -> list[tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab20")
    return [cmap(i) for i in range(num_series)]


def find_max_spearman_point(points: list[ValidationPoint]) -> ValidationPoint:
    return max(points, key=lambda point: (point.spearman_mean, -point.epoch))


def find_best_epoch_point(points: list[ValidationPoint]) -> ValidationPoint | None:
    if not points or points[0].best_epoch is None:
        return None
    best_epoch = points[0].best_epoch
    for point in points:
        if point.epoch == best_epoch:
            return point
    return None


def compute_y_limits(grouped: OrderedDict[str, list[ValidationPoint]]) -> tuple[float, float]:
    y_values: list[float] = []
    for points in grouped.values():
        y_values.extend(point.spearman_mean for point in points)
        y_values.extend(
            point.baseline_coefficient
            for point in points
            if point.baseline_coefficient is not None
        )

    y_min = min(y_values)
    y_max = max(y_values)
    if y_min == y_max:
        return y_min - 0.1, y_max + 0.1

    padding = (y_max - y_min) * 0.08
    return y_min - padding, y_max + padding


def plot_grouped_series(
    grouped: OrderedDict[str, list[ValidationPoint]],
    output_path: Path,
    title: str,
    dpi: int,
) -> None:
    colors = build_color_cycle(len(grouped))
    fig, ax = plt.subplots(figsize=(16, 9))

    for color, (dataset, points) in zip(colors, grouped.items()):
        epochs = [point.epoch for point in points]
        spearman_values = [point.spearman_mean for point in points]

        ax.plot(
            epochs,
            spearman_values,
            color=color,
            linestyle="--",
            linewidth=2.2,
            label=dataset,
            zorder=2,
        )

        baseline = points[0].baseline_coefficient
        if baseline is not None:
            ax.axhline(
                baseline,
                color=color,
                linestyle="-",
                linewidth=1.1,
                alpha=0.45,
                zorder=1,
            )

        max_point = find_max_spearman_point(points)
        ax.scatter(
            max_point.epoch,
            max_point.spearman_mean,
            marker="o",
            s=85,
            color=color,
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
        )

        best_epoch_point = find_best_epoch_point(points)
        if best_epoch_point is not None:
            ax.scatter(
                best_epoch_point.epoch,
                best_epoch_point.spearman_mean,
                marker="X",
                s=110,
                color=color,
                edgecolors="white",
                linewidths=0.8,
                zorder=5,
            )

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Spearman correlation")
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
        Line2D([0], [0], color="black", linestyle="--", linewidth=2.2, label="Spearman trajectory"),
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.1, alpha=0.7, label="Best Baseline"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            linestyle="None",
            markeredgecolor="black",
            markerfacecolor="gray",
            markersize=8,
            label="Best Spearman",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            color="black",
            linestyle="None",
            markeredgecolor="white",
            markerfacecolor="black",
            markersize=9,
            label="Best epoch",
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
    grouped = load_points(args.csv.resolve())
    if not grouped:
        raise ValueError(f"No rows found in {args.csv.resolve()}")
    plot_grouped_series(
        grouped=grouped,
        output_path=args.output.resolve(),
        title=args.title,
        dpi=args.dpi,
    )
    print(f"Saved figure to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
