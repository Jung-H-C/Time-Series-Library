from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SpearmanPoint:
    dataset: str
    epoch: int
    spearman_mean: float
    baseline_best_proxy: str
    baseline_coefficient: str


@dataclass(frozen=True)
class SpearmanSummary:
    dataset: str
    total_epochs: int
    best_epoch: int
    best_spearman: float
    baseline_best_proxy: str
    baseline_coefficient: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze DSPBuilder validation spearman CSV and export one summary row per dataset."
    )
    parser.add_argument("--base-dir", type=Path, required=True, help="Directory containing spearman.csv.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Input spearman CSV path. Defaults to spearman.csv in the base directory.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output summary CSV path. Defaults to spearman_summary.csv in the base directory.",
    )
    return parser.parse_args()


def resolve_path(path: Path | None, base_dir: Path, default_name: str) -> Path:
    if path is None:
        return base_dir / default_name
    if path.is_absolute():
        return path
    return base_dir / path


def load_points(input_csv: Path) -> OrderedDict[str, list[SpearmanPoint]]:
    grouped: OrderedDict[str, list[SpearmanPoint]] = OrderedDict()

    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_fields = {
            "dataset",
            "epoch",
            "spearman_mean",
            "baseline_best_proxy",
            "baseline_coefficient",
        }
        missing_fields = required_fields.difference(reader.fieldnames or [])
        if missing_fields:
            raise ValueError(f"Missing required columns in {input_csv}: {sorted(missing_fields)}")

        for row in reader:
            point = SpearmanPoint(
                dataset=row["dataset"],
                epoch=int(row["epoch"]),
                spearman_mean=float(row["spearman_mean"]),
                baseline_best_proxy=row["baseline_best_proxy"],
                baseline_coefficient=row["baseline_coefficient"],
            )
            grouped.setdefault(point.dataset, []).append(point)

    for dataset in grouped:
        grouped[dataset].sort(key=lambda point: point.epoch)
    return grouped


def summarize_dataset(dataset: str, points: list[SpearmanPoint]) -> SpearmanSummary:
    best_point = max(points, key=lambda point: (point.spearman_mean, -point.epoch))
    return SpearmanSummary(
        dataset=dataset,
        total_epochs=len(points),
        best_epoch=best_point.epoch,
        best_spearman=best_point.spearman_mean,
        baseline_best_proxy=best_point.baseline_best_proxy,
        baseline_coefficient=best_point.baseline_coefficient,
    )


def build_summaries(grouped: OrderedDict[str, list[SpearmanPoint]]) -> list[SpearmanSummary]:
    summaries: list[SpearmanSummary] = []
    for dataset, points in grouped.items():
        if not points:
            continue
        summaries.append(summarize_dataset(dataset, points))
    return summaries


def write_summary_csv(output_csv: Path, summaries: list[SpearmanSummary]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "total_epochs",
        "best_epoch",
        "best_spearman",
        "baseline_best_proxy",
        "baseline_coefficient",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "dataset": summary.dataset,
                    "total_epochs": summary.total_epochs,
                    "best_epoch": summary.best_epoch,
                    "best_spearman": f"{summary.best_spearman:.6f}",
                    "baseline_best_proxy": summary.baseline_best_proxy,
                    "baseline_coefficient": summary.baseline_coefficient,
                }
            )


def main() -> int:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    input_csv = resolve_path(args.input_csv, base_dir, "spearman.csv").resolve()
    output_csv = resolve_path(args.output_csv, base_dir, "spearman_summary.csv").resolve()

    grouped = load_points(input_csv)
    if not grouped:
        raise ValueError(f"No rows found in {input_csv}")

    summaries = build_summaries(grouped)
    write_summary_csv(output_csv, summaries)
    print(f"Wrote {len(summaries)} dataset summaries to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
