from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


TEST_SUMMARY_PATTERN = re.compile(
    r"\[TEST-SUMMARY\]\s+"
    r"epoch=(?P<epoch>\d+)\s+"
    r"dataset=(?P<dataset>.+?)\s+"
    r"metric=(?P<metric>.+?)\s+"
    r"spearman_mean=(?P<spearman_mean>[-+0-9.eE]+)\s+"
    r"spearman_std=(?P<spearman_std>[-+0-9.eE]+)\s+"
    r"support_sets=(?P<support_sets>\d+)\s+"
    r"num_candidates=(?P<num_candidates>\d+)"
    r"(?:\s+baseline_best_proxy=(?P<baseline_best_proxy>\S+))?"
    r"(?:\s+baseline_coefficient=(?P<baseline_coefficient>[-+0-9.eE]+))?"
)


@dataclass(frozen=True)
class TestSummaryPoint:
    dataset: str
    metric_name: str
    epoch: int
    spearman_mean: float
    spearman_std: float
    support_sets: int
    num_candidates: int
    baseline_best_proxy: str
    baseline_coefficient: str
    source_file: str


@dataclass(frozen=True)
class DatasetBestSummary:
    dataset: str
    metric_name: str
    total_epochs: int
    best_epoch: int
    best_spearman_mean: float
    best_spearman_std: float
    support_sets: int
    num_candidates: int
    baseline_best_proxy: str
    baseline_coefficient: str
    source_file: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read all test log .txt files under a test_logs directory and export "
            "the best epoch / best spearman_mean per dataset."
        )
    )
    parser.add_argument(
        "--test-log-dir",
        type=Path,
        required=True,
        help="Path to the test_logs directory containing per-dataset .txt files.",
    )
    return parser.parse_args()


def parse_test_summary_line(line: str, source_file: str) -> TestSummaryPoint | None:
    match = TEST_SUMMARY_PATTERN.search(line)
    if match is None:
        return None

    baseline_best_proxy = match.group("baseline_best_proxy") or ""
    baseline_coefficient = match.group("baseline_coefficient") or ""
    return TestSummaryPoint(
        dataset=match.group("dataset"),
        metric_name=match.group("metric"),
        epoch=int(match.group("epoch")),
        spearman_mean=float(match.group("spearman_mean")),
        spearman_std=float(match.group("spearman_std")),
        support_sets=int(match.group("support_sets")),
        num_candidates=int(match.group("num_candidates")),
        baseline_best_proxy=baseline_best_proxy,
        baseline_coefficient=baseline_coefficient,
        source_file=source_file,
    )


def load_summary_points(test_log_dir: Path) -> list[TestSummaryPoint]:
    points: list[TestSummaryPoint] = []
    for log_path in sorted(test_log_dir.glob("*.txt")):
        for line in log_path.read_text(encoding="utf-8").splitlines():
            point = parse_test_summary_line(line, source_file=log_path.name)
            if point is not None:
                points.append(point)
    return points


def summarize_points(points: list[TestSummaryPoint]) -> list[DatasetBestSummary]:
    grouped: dict[str, list[TestSummaryPoint]] = {}
    for point in points:
        grouped.setdefault(point.dataset, []).append(point)

    summaries: list[DatasetBestSummary] = []
    for dataset in sorted(grouped):
        dataset_points = sorted(grouped[dataset], key=lambda point: point.epoch)
        best_point = max(dataset_points, key=lambda point: (point.spearman_mean, -point.epoch))
        summaries.append(
            DatasetBestSummary(
                dataset=best_point.dataset,
                metric_name=best_point.metric_name,
                total_epochs=len(dataset_points),
                best_epoch=best_point.epoch,
                best_spearman_mean=best_point.spearman_mean,
                best_spearman_std=best_point.spearman_std,
                support_sets=best_point.support_sets,
                num_candidates=best_point.num_candidates,
                baseline_best_proxy=best_point.baseline_best_proxy,
                baseline_coefficient=best_point.baseline_coefficient,
                source_file=best_point.source_file,
            )
        )
    return summaries


def write_summary_csv(output_csv: Path, summaries: list[DatasetBestSummary]) -> None:
    fieldnames = [
        "dataset",
        "metric_name",
        "total_epochs",
        "best_epoch",
        "best_spearman_mean",
        "best_spearman_std",
        "support_sets",
        "num_candidates",
        "baseline_best_proxy",
        "baseline_coefficient",
        "source_file",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "dataset": summary.dataset,
                    "metric_name": summary.metric_name,
                    "total_epochs": summary.total_epochs,
                    "best_epoch": summary.best_epoch,
                    "best_spearman_mean": f"{summary.best_spearman_mean:.6f}",
                    "best_spearman_std": f"{summary.best_spearman_std:.6f}",
                    "support_sets": summary.support_sets,
                    "num_candidates": summary.num_candidates,
                    "baseline_best_proxy": summary.baseline_best_proxy,
                    "baseline_coefficient": summary.baseline_coefficient,
                    "source_file": summary.source_file,
                }
            )


def main() -> int:
    args = parse_args()
    test_log_dir = args.test_log_dir.resolve()
    if not test_log_dir.is_dir():
        raise FileNotFoundError(f"test_log_dir does not exist or is not a directory: {test_log_dir}")

    points = load_summary_points(test_log_dir)
    if not points:
        raise ValueError(f"No [TEST-SUMMARY] lines found under {test_log_dir}")

    summaries = summarize_points(points)
    output_csv = test_log_dir / "spearman_analysis.csv"
    write_summary_csv(output_csv, summaries)
    print(f"Wrote {len(summaries)} dataset summaries to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
