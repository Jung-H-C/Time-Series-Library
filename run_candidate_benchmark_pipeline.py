from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from benchmarking.candidate_sampler import (
    _candidate_recipe_runs,
    _prepare_candidate_run_args,
    _recipe_adjusted_run_args,
    _requested_uea_subset_names,
)


PROXY_COLUMNS = [
    "params",
    "flops",
    "grad_norm",
    "fisher",
    "grasp",
    "jacob_cov",
    "jacob_fro",
    "sfrd",
    "snip",
    "synflow",
]

FORECAST_METRIC_NAMES = ["mae", "mse", "rmse", "mape", "mspe"]
M4_METRIC_NAMES = ["smape", "owa", "mape", "mase"]
METRIC_COLUMN_PRIORITY = [
    "smape",
    "owa",
    "mae",
    "mse",
    "rmse",
    "mape",
    "mspe",
    "mase",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _timestamp_label() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def _parse_gpu_ids(raw_gpu_ids: list[str] | None) -> list[int]:
    if raw_gpu_ids is None:
        return []

    parsed: list[int] = []
    seen: set[int] = set()
    for raw_value in raw_gpu_ids:
        for token in str(raw_value).split(","):
            token = token.strip()
            if not token:
                continue
            gpu_id = int(token)
            if gpu_id < 0:
                raise ValueError(f"GPU ids must be non-negative integers, got {gpu_id}.")
            if gpu_id in seen:
                continue
            seen.add(gpu_id)
            parsed.append(gpu_id)
    return parsed


def _load_candidate_payload(candidate_path: Path) -> dict[str, Any]:
    with candidate_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"Candidate JSON must contain a non-empty 'candidates' list: {candidate_path}")
    return payload


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    capture_regex: re.Pattern[str] | None = None,
) -> str | None:
    print(f"[cmd] {' '.join(command)}")
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    captured: str | None = None
    assert process.stdout is not None
    for line in process.stdout:
        rendered = line.rstrip("\n")
        print(rendered, flush=True)
        if capture_regex is not None:
            match = capture_regex.search(rendered)
            if match is not None:
                captured = match.group(1).strip()

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(command)}")
    return captured


def _sanitize_for_results_component(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    for sep in filter(None, {os.sep, os.altsep}):
        text = text.replace(sep, "_")
    return text


def _result_dir_from_run_args(run_args: dict[str, Any], repo_root: Path) -> Path:
    task_name = _sanitize_for_results_component(run_args.get("task_name")) or "task"
    model_id = _sanitize_for_results_component(run_args.get("model_id")) or "model"
    results_id = _sanitize_for_results_component(run_args.get("results_id")) or model_id
    return repo_root / "results" / f"{task_name}_{model_id}_{results_id}"


def _metric_names_for_task(task_name: Any, metric_count: int) -> list[str]:
    normalized_task_name = str(task_name or "").strip()
    if normalized_task_name == "short_term_forecast" and metric_count == len(M4_METRIC_NAMES):
        return list(M4_METRIC_NAMES)
    if normalized_task_name in {"long_term_forecast", "zero_shot_forecast", "imputation"} and metric_count == len(
        FORECAST_METRIC_NAMES
    ):
        return list(FORECAST_METRIC_NAMES)
    return [f"metric_{index}" for index in range(metric_count)]


def _ordered_metric_columns(metric_names: set[str]) -> list[str]:
    ordered = [name for name in METRIC_COLUMN_PRIORITY if name in metric_names]
    ordered.extend(sorted(name for name in metric_names if name not in ordered))
    return ordered


def _candidate_run_args_list(
    candidate: dict[str, Any],
    *,
    requested_uea_subset_names: list[str] | None,
    repo_root: Path,
) -> list[dict[str, Any]]:
    if requested_uea_subset_names:
        recipe_selection = _candidate_recipe_runs(
            candidate,
            requested_subset_names=requested_uea_subset_names,
            repo_root=repo_root,
        )
        if recipe_selection is not None:
            _, recipe_runs = recipe_selection
            return [
                _recipe_adjusted_run_args(candidate, recipe_run, gpu_id=None)
                for recipe_run in recipe_runs
            ]
    return [_prepare_candidate_run_args(candidate, gpu_id=None)]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        numeric = float(text)
    except ValueError:
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _nanmean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=float)))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            normalized: dict[str, Any] = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, float) and math.isnan(value):
                    normalized[key] = ""
                elif value is None:
                    normalized[key] = ""
                else:
                    normalized[key] = value
            writer.writerow(normalized)


def _load_csv_rows(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _descending_average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx], reverse=True)
    ranks = [0.0] * len(values)

    i = 0
    while i < len(order):
        j = i + 1
        pivot = values[order[i]]
        while j < len(order) and np.isclose(values[order[j]], pivot, rtol=1e-12, atol=1e-12):
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def _rank_normalize_proxy_columns(rows: list[dict[str, Any]], proxy_columns: list[str]) -> list[dict[str, Any]]:
    normalized_rows = [dict(row) for row in rows]

    for proxy_name in proxy_columns:
        numeric_values: list[float] = []
        row_indices: list[int] = []
        for row_index, row in enumerate(rows):
            value = _safe_float(row.get(proxy_name))
            if value is None:
                continue
            numeric_values.append(value)
            row_indices.append(row_index)

        if not numeric_values:
            for row in normalized_rows:
                row[proxy_name] = float("nan")
            continue

        if len(numeric_values) == 1:
            normalized_rows[row_indices[0]][proxy_name] = 0.0
            for row_index, row in enumerate(normalized_rows):
                if row_index != row_indices[0]:
                    row[proxy_name] = float("nan")
            continue

        ranks = _descending_average_ranks(numeric_values)
        denom = len(numeric_values) - 1
        for local_index, row_index in enumerate(row_indices):
            rank = ranks[local_index]
            normalized_rows[row_index][proxy_name] = 2.0 * ((len(numeric_values) - rank) / denom) - 1.0

        for row_index, row in enumerate(normalized_rows):
            if row_index not in row_indices:
                row[proxy_name] = float("nan")

    return normalized_rows


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run candidate training/testing, score 10 zero-cost proxies, and export raw + "
            "rank-normalized benchmark CSV files."
        )
    )
    parser.add_argument("--candidates-file", type=str, required=True, help="Path to candidates.json.")
    parser.add_argument(
        "--gpu-id",
        nargs="+",
        default=None,
        help=(
            "One or more physical GPU ids (e.g., --gpu-id 0, --gpu-id 0 1 2, --gpu-id 0,1,2). "
            "Used for both run-candidates and proxy scoring."
        ),
    )
    parser.add_argument(
        "--run-workers-per-gpu",
        type=int,
        default=1,
        help="Workers per GPU for --run-candidates. Default: 1.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=5,
        help="Number of random mini-batches used in proxy scoring. Default: 5.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for proxy mini-batch sampling. Default: 2026.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save pipeline CSV outputs. Default: benchmark_results/",
    )
    parser.add_argument(
        "--proxy-deterministic",
        action="store_true",
        help="Enable deterministic mode for proxy scoring.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining candidates on run-candidates failures.",
    )
    parser.add_argument(
        "--skip-run-candidates",
        action="store_true",
        help=(
            "Skip step 1 (sample_candidates train/test). "
            "Use existing results/.../metrics.npy and continue from proxy scoring."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    repo_root = _repo_root()
    os.chdir(repo_root)

    candidate_path = Path(args.candidates_file)
    if not candidate_path.is_absolute():
        candidate_path = (repo_root / candidate_path).resolve()
    if not candidate_path.exists():
        parser.error(f"Candidate JSON not found: {candidate_path}")

    if args.run_workers_per_gpu < 1:
        parser.error("--run-workers-per-gpu must be >= 1.")

    gpu_ids = _parse_gpu_ids(args.gpu_id)
    gpu_tokens = [str(gpu_id) for gpu_id in gpu_ids]

    payload = _load_candidate_payload(candidate_path)
    candidate_stem = candidate_path.stem
    timestamp = _timestamp_label()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_run_candidates:
        print("[1/3] Skipping candidate training/testing (--skip-run-candidates).")
    else:
        print("[1/3] Running candidate training/testing...")
        run_command = [
            sys.executable,
            "sample_candidates.py",
            "--run-candidates-file",
            str(candidate_path),
        ]
        if gpu_tokens:
            run_command.extend(["--gpu-id", *gpu_tokens])
        if args.run_workers_per_gpu != 1:
            run_command.extend(["--workers-per-gpu", str(args.run_workers_per_gpu)])
        if args.continue_on_error:
            run_command.append("--continue-on-error")
        _run_command(run_command, cwd=repo_root)

    print("[2/3] Scoring all 10 zero-cost proxies...")
    proxy_csv_base_path = output_dir / f"{candidate_stem}_proxy_scores_raw.csv"
    score_command = [
        sys.executable,
        "score_candidates.py",
        "--candidates-file",
        str(candidate_path),
        "--num-batches",
        str(args.num_batches),
        "--seed",
        str(args.seed),
        "--csv-path",
        str(proxy_csv_base_path),
    ]
    if gpu_tokens:
        score_command.extend(["--gpu-id", *gpu_tokens])
    if args.proxy_deterministic:
        score_command.append("--deterministic")

    captured_proxy_csv_path = _run_command(
        score_command,
        cwd=repo_root,
        capture_regex=re.compile(r"^Saved proxy scores to (.+)$"),
    )
    if captured_proxy_csv_path is None:
        parser.error("Failed to detect proxy CSV output path from score_candidates.py logs.")
    proxy_csv_path = Path(captured_proxy_csv_path)
    if not proxy_csv_path.is_absolute():
        proxy_csv_path = (repo_root / proxy_csv_path).resolve()
    if not proxy_csv_path.exists():
        parser.error(f"Proxy score CSV not found: {proxy_csv_path}")

    print("[3/3] Building raw/normalized benchmark CSV files...")
    requested_uea_subset_names = _requested_uea_subset_names(payload)
    candidates = payload["candidates"]

    run_level_rows: list[dict[str, Any]] = []
    candidate_metric_map: dict[str, dict[str, Any]] = {}
    metric_names_seen: set[str] = set()
    for index, candidate in enumerate(candidates, start=1):
        candidate_id = str(
            candidate.get("candidate_id", candidate.get("candidate_name", f"candidate_{index:04d}"))
        )
        candidate_name = str(candidate.get("candidate_name", candidate_id))
        run_args_list = _candidate_run_args_list(
            candidate,
            requested_uea_subset_names=requested_uea_subset_names,
            repo_root=repo_root,
        )

        metric_acc: dict[str, list[float]] = {}
        success_runs = 0

        for run_index, run_args in enumerate(run_args_list, start=1):
            result_dir = _result_dir_from_run_args(run_args, repo_root)
            metrics_path = result_dir / "metrics.npy"
            row: dict[str, Any] = {
                "candidate_id": candidate_id,
                "candidate_name": candidate_name,
                "model": run_args.get("model", ""),
                "task_name": run_args.get("task_name", ""),
                "data": run_args.get("data", ""),
                "run_index": run_index,
                "run_name": run_args.get("model_id", ""),
                "result_dir": str(result_dir.relative_to(repo_root)),
                "metrics_path": str(metrics_path.relative_to(repo_root)),
                "status": "success",
                "error": "",
            }
            if not metrics_path.exists():
                row["status"] = "missing_metrics"
                row["error"] = "metrics.npy not found"
            else:
                try:
                    values = np.load(metrics_path).reshape(-1).astype(float).tolist()
                except Exception as exc:  # noqa: BLE001
                    row["status"] = "read_error"
                    row["error"] = str(exc)
                    values = []

                if values:
                    metric_names = _metric_names_for_task(run_args.get("task_name", ""), len(values))
                    for metric_name, metric_value in zip(metric_names, values):
                        row[metric_name] = metric_value
                        metric_acc.setdefault(metric_name, []).append(metric_value)
                        metric_names_seen.add(metric_name)
                    success_runs += 1
                else:
                    row["status"] = "invalid_metrics"
                    row["error"] = f"Expected >=1 value in metrics.npy, got {len(values)}"

            run_level_rows.append(row)

        aggregate_status = "success" if success_runs == len(run_args_list) else ("partial" if success_runs > 0 else "failed")
        aggregated_metrics = {
            metric_name: _nanmean(values)
            for metric_name, values in metric_acc.items()
        }
        metric_names_seen.update(aggregated_metrics)
        candidate_metric_map[candidate_id] = {
            "candidate_id": candidate_id,
            "candidate_name": candidate_name,
            "num_runs_total": len(run_args_list),
            "num_runs_success": success_runs,
            "training_status": aggregate_status,
            "metrics": aggregated_metrics,
        }

    metric_columns = _ordered_metric_columns(metric_names_seen)
    run_level_csv_path = output_dir / f"{candidate_stem}_training_metrics_raw_{timestamp}.csv"
    run_level_fieldnames = [
        "candidate_id",
        "candidate_name",
        "model",
        "task_name",
        "data",
        "run_index",
        "run_name",
        "result_dir",
        "metrics_path",
        "status",
        "error",
        *metric_columns,
    ]
    _write_csv(run_level_csv_path, run_level_rows, run_level_fieldnames)

    proxy_raw_copy_path = output_dir / f"{candidate_stem}_proxy_scores_raw_{timestamp}.csv"
    shutil.copy2(proxy_csv_path, proxy_raw_copy_path)
    proxy_rows = _load_csv_rows(proxy_raw_copy_path)
    proxy_row_by_candidate_id = {
        str(row.get("candidate_id", "")).strip(): row
        for row in proxy_rows
        if str(row.get("candidate_id", "")).strip()
    }

    missing_proxy_columns = [proxy_name for proxy_name in PROXY_COLUMNS if not proxy_rows or proxy_name not in proxy_rows[0]]
    if missing_proxy_columns:
        parser.error(
            "Proxy CSV is missing required columns: "
            + ", ".join(missing_proxy_columns)
            + f" (file: {proxy_raw_copy_path})"
        )

    combined_raw_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        candidate_id = str(
            candidate.get("candidate_id", candidate.get("candidate_name", f"candidate_{index:04d}"))
        )
        candidate_name = str(candidate.get("candidate_name", candidate_id))
        run_args = dict(candidate.get("run_args", {}))
        proxy_row = proxy_row_by_candidate_id.get(candidate_id, {})
        metric_row = candidate_metric_map.get(candidate_id, {})

        combined_row: dict[str, Any] = {
            "candidate_id": candidate_id,
            "candidate_name": candidate_name,
            "model": run_args.get("model", candidate.get("model", "")),
            "task_name": run_args.get("task_name", ""),
            "data": run_args.get("data", ""),
            "num_runs_total": metric_row.get("num_runs_total", 0),
            "num_runs_success": metric_row.get("num_runs_success", 0),
            "training_status": metric_row.get("training_status", "missing"),
            "proxy_status": proxy_row.get("status", ""),
            "proxy_error": proxy_row.get("error", ""),
        }
        for metric_name in metric_columns:
            combined_row[metric_name] = dict(metric_row.get("metrics", {})).get(metric_name)
        for proxy_name in PROXY_COLUMNS:
            combined_row[proxy_name] = _safe_float(proxy_row.get(proxy_name))
        combined_raw_rows.append(combined_row)

    combined_raw_csv_path = output_dir / f"{candidate_stem}_benchmark_raw_with_mse_{timestamp}.csv"
    combined_fieldnames = [
        "candidate_id",
        "candidate_name",
        "model",
        "task_name",
        "data",
        "num_runs_total",
        "num_runs_success",
        "training_status",
        "proxy_status",
        "proxy_error",
        *metric_columns,
        *PROXY_COLUMNS,
    ]
    _write_csv(combined_raw_csv_path, combined_raw_rows, combined_fieldnames)

    normalized_rows = _rank_normalize_proxy_columns(combined_raw_rows, PROXY_COLUMNS)
    normalized_csv_path = output_dir / f"{candidate_stem}_benchmark_ranknorm_with_mse_{timestamp}.csv"
    _write_csv(normalized_csv_path, normalized_rows, combined_fieldnames)

    print("Pipeline completed successfully.")
    print(f"- Step1 raw (training/test metrics): {run_level_csv_path}")
    print(f"- Step2 raw (proxy scores):         {proxy_raw_copy_path}")
    print(f"- Raw merged benchmark CSV:         {combined_raw_csv_path}")
    print(f"- Rank-normalized benchmark CSV:    {normalized_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
