from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np


TASK_PREFIXES = (
    "zero_shot_forecast",
    "short_term_forecast",
    "long_term_forecast",
    "anomaly_detection",
    "classification",
    "imputation",
)

FORECAST_METRIC_NAMES = ("mae", "mse", "rmse", "mape", "mspe")
RUN_PATTERN = re.compile(
    r"^(?P<run_prefix>.*)"
    r"_dm(?P<d_model>\d+)"
    r"_df(?P<d_ff>\d+)"
    r"_expand(?P<expand>\d+)"
    r"_dc(?P<d_conv>\d+)"
    r"_(?P<candidate_id>\d+)$"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _detect_task_name(result_dir_name: str) -> str:
    for task_name in TASK_PREFIXES:
        prefix = f"{task_name}_"
        if result_dir_name.startswith(prefix):
            return task_name
    return ""


def _split_model_and_results_id(payload: str) -> tuple[str, str]:
    midpoint = len(payload) // 2
    if len(payload) % 2 == 1 and payload[midpoint] == "_" and payload[:midpoint] == payload[midpoint + 1 :]:
        repeated = payload[:midpoint]
        return repeated, repeated
    return payload, ""


def _parse_model_id(result_dir_name: str) -> str:
    task_name = _detect_task_name(result_dir_name)
    payload = result_dir_name[len(task_name) + 1 :] if task_name else result_dir_name
    model_id, _ = _split_model_and_results_id(payload)
    return model_id


def _candidate_sort_key(model_id: str) -> tuple[int, str]:
    run_match = RUN_PATTERN.match(model_id)
    if run_match is None:
        return 10**9, model_id
    return int(run_match.group("candidate_id")), model_id


def _load_row(result_dir: Path) -> dict[str, float | str]:
    metric_values = np.load(result_dir / "metrics.npy").reshape(-1).tolist()
    if len(metric_values) != len(FORECAST_METRIC_NAMES):
        raise ValueError(
            f"Expected {len(FORECAST_METRIC_NAMES)} metrics in '{result_dir / 'metrics.npy'}', "
            f"but found {len(metric_values)}."
        )

    row: dict[str, float | str] = {"model_id": _parse_model_id(result_dir.name)}
    for metric_name, metric_value in zip(FORECAST_METRIC_NAMES, metric_values):
        row[metric_name] = float(metric_value)
    return row


def export_subset(results_root: Path, prefix: str) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for result_dir in sorted(results_root.iterdir()):
        if not result_dir.is_dir() or not result_dir.name.startswith(prefix):
            continue
        metrics_path = result_dir / "metrics.npy"
        if not metrics_path.exists():
            continue
        rows.append(_load_row(result_dir))
    return sorted(rows, key=lambda row: _candidate_sort_key(str(row["model_id"])))


def write_csv(output_path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        raise ValueError("No matching result directories with metrics.npy were found.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model_id", *FORECAST_METRIC_NAMES]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(
        description="Export a filtered subset of forecast result metrics to a compact CSV."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=repo_root / "results",
        help="Directory containing per-run result folders.",
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help="Directory-name prefix used to select matching result folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination CSV path.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    rows = export_subset(args.results_root, args.prefix)
    write_csv(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
