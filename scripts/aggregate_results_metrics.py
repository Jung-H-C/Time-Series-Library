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


def _metric_names_for(task_name: str, metric_count: int) -> list[str]:
    if task_name in {
        "long_term_forecast",
        "short_term_forecast",
        "zero_shot_forecast",
        "imputation",
    } and metric_count == len(FORECAST_METRIC_NAMES):
        return list(FORECAST_METRIC_NAMES)
    return [f"metric_{index}" for index in range(metric_count)]


def _parse_result_row(result_dir: Path) -> dict[str, object]:
    metrics_path = result_dir / "metrics.npy"
    metric_values = np.load(metrics_path).reshape(-1).tolist()

    result_dir_name = result_dir.name
    task_name = _detect_task_name(result_dir_name)
    payload = result_dir_name[len(task_name) + 1 :] if task_name else result_dir_name
    model_id, results_id = _split_model_and_results_id(payload)

    row: dict[str, object] = {
        "result_dir": result_dir_name,
        "task_name": task_name,
        "model_id": model_id,
        "results_id": results_id,
        "metrics_path": str(metrics_path.relative_to(_repo_root())),
    }

    run_match = RUN_PATTERN.match(model_id)
    if run_match is not None:
        run_prefix = run_match.group("run_prefix")
        row.update(
            {
                "candidate_id": int(run_match.group("candidate_id")),
                "run_prefix": run_prefix,
                "model": run_prefix.split("_")[0] if run_prefix else "",
                "data": run_prefix.split("_")[-1] if run_prefix else "",
                "d_model": int(run_match.group("d_model")),
                "d_ff": int(run_match.group("d_ff")),
                "expand": int(run_match.group("expand")),
                "d_conv": int(run_match.group("d_conv")),
            }
        )
    else:
        row.update(
            {
                "candidate_id": "",
                "run_prefix": "",
                "model": "",
                "data": "",
                "d_model": "",
                "d_ff": "",
                "expand": "",
                "d_conv": "",
            }
        )

    for metric_name, metric_value in zip(_metric_names_for(task_name, len(metric_values)), metric_values):
        row[metric_name] = float(metric_value)

    return row


def _sort_key(row: dict[str, object]) -> tuple[int, str]:
    candidate_id = row.get("candidate_id")
    if isinstance(candidate_id, int):
        return candidate_id, str(row.get("result_dir", ""))
    return 10**9, str(row.get("result_dir", ""))


def aggregate_results(results_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        metrics_path = result_dir / "metrics.npy"
        if not metrics_path.exists():
            continue
        rows.append(_parse_result_row(result_dir))
    return sorted(rows, key=_sort_key)


def write_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No results with metrics.npy were found under '{output_path.parent}'.")

    preferred_columns = [
        "candidate_id",
        "model",
        "task_name",
        "data",
        "d_model",
        "d_ff",
        "expand",
        "d_conv",
        "mae",
        "mse",
        "rmse",
        "mape",
        "mspe",
        "run_prefix",
        "model_id",
        "results_id",
        "result_dir",
        "metrics_path",
    ]
    remaining_columns = [key for key in rows[0].keys() if key not in preferred_columns]
    fieldnames = [column for column in preferred_columns if column in rows[0]] + remaining_columns

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(
        description="Aggregate Time-Series-Library result directories with metrics.npy into a single CSV."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=repo_root / "results",
        help="Directory containing per-run result folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "results" / "results_metrics_summary.csv",
        help="Destination CSV path.",
    )
    args = parser.parse_args()

    rows = aggregate_results(args.results_root)
    write_csv(args.output, rows)

    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
