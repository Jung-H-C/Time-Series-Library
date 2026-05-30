from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


DEFAULT_X_MIN = 0
DEFAULT_X_MAX = 70
DEFAULT_Y_MIN = 0.3
DEFAULT_Y_MAX = 1.1


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
        description="Plot DSPBuilder meta validation losses inside a fixed axis window."
    )
    parser.add_argument("--base-dir", type=Path, required=True, help="Root directory containing validation run folders.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to fixed_window_loss.png in the base directory.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="DSPBuilder Validation Loss",
        help="Figure title.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Image DPI.")
    parser.add_argument("--x-min", type=float, default=DEFAULT_X_MIN, help="Minimum epoch shown on the x-axis.")
    parser.add_argument("--x-max", type=float, default=DEFAULT_X_MAX, help="Maximum epoch shown on the x-axis.")
    parser.add_argument("--y-min", type=float, default=DEFAULT_Y_MIN, help="Minimum validation loss shown on the y-axis.")
    parser.add_argument("--y-max", type=float, default=DEFAULT_Y_MAX, help="Maximum validation loss shown on the y-axis.")
    return parser.parse_args()


def parse_key_value_line(line: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for token in line.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        values[key] = value
    return values


def load_points(base_dir: Path) -> OrderedDict[str, list[ValidationLossPoint]]:
    grouped: OrderedDict[str, list[ValidationLossPoint]] = OrderedDict()

    for valid_logs_dir in sorted(base_dir.rglob("valid_logs")):
        run_dir = valid_logs_dir.parent
        validation_dir = run_dir.parent

        for valid_log_path in sorted(valid_logs_dir.glob("*.txt")):
            dataset_name = valid_log_path.stem

            for raw_line in valid_log_path.read_text(encoding="utf-8").splitlines():
                if not raw_line.startswith("[EPOCH-SUMMARY]"):
                    continue

                parsed = parse_key_value_line(raw_line)
                point = ValidationLossPoint(
                    dataset=dataset_name,
                    validation_dir=validation_dir.name,
                    run_dir=run_dir.name,
                    epoch=int(parsed["epoch"]),
                    val_loss=float(parsed["val_loss"]),
                    early_stopping_counter=int(parsed.get("early_stopping_counter", 0)),
                )
                grouped.setdefault(point.dataset, []).append(point)

    for dataset in grouped:
        grouped[dataset].sort(
            key=lambda point: (
                point.validation_dir,
                point.run_dir,
                point.epoch,
            )
        )
    return grouped


def build_color_cycle(num_series: int) -> list[tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab20")
    return [cmap(i) for i in range(num_series)]


def find_min_loss_point(points: list[ValidationLossPoint]) -> ValidationLossPoint:
    return min(points, key=lambda point: (point.val_loss, point.epoch))


def validate_axis_limits(x_min: float, x_max: float, y_min: float, y_max: float) -> None:
    if x_min >= x_max:
        raise ValueError(f"x-min must be smaller than x-max, got {x_min} >= {x_max}.")
    if y_min >= y_max:
        raise ValueError(f"y-min must be smaller than y-max, got {y_min} >= {y_max}.")


def plot_grouped_series(
    grouped: OrderedDict[str, list[ValidationLossPoint]],
    output_path: Path,
    title: str,
    dpi: int,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
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
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)

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
    validate_axis_limits(args.x_min, args.x_max, args.y_min, args.y_max)

    base_dir = args.base_dir.resolve()
    output_path = args.output.resolve() if args.output is not None else base_dir / "fixed_window_loss.png"

    grouped = load_points(base_dir)
    if not grouped:
        raise ValueError(f"No validation loss rows found under {base_dir}")

    plot_grouped_series(
        grouped=grouped,
        output_path=output_path,
        title=args.title,
        dpi=args.dpi,
        x_limits=(args.x_min, args.x_max),
        y_limits=(args.y_min, args.y_max),
    )
    print(f"Saved figure to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
