from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any, Iterable

from evaluate_lodo_checkpoints import (
    DATASET_ORDER,
    checkpoint_step,
    csv_value,
    dataset_sort_key,
    discover_jobs as discover_checkpoint_jobs,
    parse_dataset_filter,
    resolve_path,
    run_checkpoint_test,
    truncate_error,
)


CSV_FIELDNAMES = [
    "status",
    "leave_out_dataset",
    "condition_dataset",
    "condition_ts_dataset",
    "test_dataset",
    "ts_dataset",
    "evaluation_dataset",
    "evaluation_ts_dataset",
    "checkpoint_step",
    "checkpoint_name",
    "checkpoint_path",
    "spearman_neg_corr",
    "spearman_neg_mse",
    "spearman_neg_correlation",
    "invalid_reason",
    "rpn_tokens",
    "infix",
    "latex",
    "benchmark_dataset",
    "benchmark_csv",
    "split",
    "split_count",
    "support_indices",
    "elapsed_sec",
    "returncode",
    "error_message",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every periodic LODO DCSPG checkpoint on every dataset's "
            "proxy_test split and collect Spearman(-MSE) results in one CSV."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--train-script", type=Path, default=Path("train_dcspg_framework.py")
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("DCSPG/checkpoints/lodo"),
        help=(
            "Checkpoint root. Accepts either one timestamped run directory "
            "or the parent directory containing run_*/leave_out_* folders."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Output CSV path. Default: lodo_checkpoint_all_dataset_test_results.csv, "
            "or lodo_checkpoint_condition_eval_matrix_results.csv with --matrix-test."
        ),
    )
    parser.add_argument("--ts-feature-dir", type=Path, default=Path("DCSPG/TS_dataset"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("DCSPG/Benchmark"))
    parser.add_argument(
        "--leave-out-datasets",
        "--datasets",
        dest="leave_out_datasets",
        type=str,
        default="",
        help="Optional comma-separated leave-out folds whose checkpoints should be evaluated.",
    )
    parser.add_argument(
        "--test-datasets",
        type=str,
        default="",
        help="Optional comma-separated evaluation datasets. Default: all six datasets.",
    )
    parser.add_argument(
        "--condition-datasets",
        type=str,
        default="",
        help=(
            "Optional comma-separated conditioning datasets for --matrix-test. "
            "Default: all six datasets."
        ),
    )
    parser.add_argument(
        "--matrix-test",
        action="store_true",
        help=(
            "For each checkpoint, evaluate every conditioning dataset x evaluation "
            "dataset pair. Without this flag, only condition_dataset == test_dataset "
            "is evaluated."
        ),
    )
    parser.add_argument(
        "--python",
        dest="python_executable",
        type=str,
        default=sys.executable,
        help="Python executable used to run train_dcspg_framework.py.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=1,
        help="CUDA GPU id to use by default. Set -1 to honor --device without passing --gpu-id.",
    )
    parser.add_argument("--k-samples", type=int, default=16)
    parser.add_argument("--test-seed", type=int, default=2026)
    parser.add_argument("--test-split", type=str, default="proxy_test")
    parser.add_argument("--test-max-len", type=int, default=None)
    parser.add_argument(
        "--max-checkpoints-per-dataset",
        type=int,
        default=None,
        help="Optional cap per leave-out fold for quick smoke tests.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing CSV and skip checkpoint/test-dataset pairs already present in it.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when a checkpoint subprocess fails or emits unparsable JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the planned checkpoint/test count without running tests.",
    )
    return parser


def parse_dataset_list(raw: str, default: tuple[str, ...]) -> tuple[str, ...]:
    if not raw.strip():
        return default

    datasets = []
    seen = set()
    for item in raw.split(","):
        dataset = item.strip()
        if dataset and dataset not in seen:
            datasets.append(dataset)
            seen.add(dataset)
    return tuple(datasets) or default


def expand_jobs(
    checkpoint_jobs: Iterable[tuple[str, Path]],
    test_datasets: Iterable[str],
) -> list[tuple[str, Path, str, str]]:
    test_dataset_list = tuple(test_datasets)
    return [
        (leave_out_dataset, checkpoint_path, test_dataset, test_dataset)
        for leave_out_dataset, checkpoint_path in checkpoint_jobs
        for test_dataset in test_dataset_list
    ]


def expand_matrix_jobs(
    checkpoint_jobs: Iterable[tuple[str, Path]],
    condition_datasets: Iterable[str],
    test_datasets: Iterable[str],
) -> list[tuple[str, Path, str, str]]:
    condition_dataset_list = tuple(condition_datasets)
    test_dataset_list = tuple(test_datasets)
    return [
        (leave_out_dataset, checkpoint_path, condition_dataset, test_dataset)
        for leave_out_dataset, checkpoint_path in checkpoint_jobs
        for condition_dataset in condition_dataset_list
        for test_dataset in test_dataset_list
    ]


def job_key(
    leave_out_dataset: str,
    checkpoint_path: Path,
    condition_dataset: str,
    test_dataset: str,
) -> tuple[str, str, str, str]:
    return leave_out_dataset, str(checkpoint_path), condition_dataset, test_dataset


def read_completed_job_keys(output_csv: Path) -> set[tuple[str, str, str, str]]:
    if not output_csv.exists():
        return set()

    completed = set()
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            leave_out_dataset = row.get("leave_out_dataset")
            checkpoint_path = row.get("checkpoint_path")
            test_dataset = row.get("test_dataset")
            condition_dataset = row.get("condition_dataset") or test_dataset
            if leave_out_dataset and checkpoint_path and condition_dataset and test_dataset:
                completed.add((leave_out_dataset, checkpoint_path, condition_dataset, test_dataset))
    return completed


def build_csv_row(
    *,
    leave_out_dataset: str,
    condition_dataset: str,
    test_dataset: str,
    checkpoint_path: Path,
    result: dict[str, Any] | None,
    returncode: int,
    error_message: str,
    elapsed_sec: float,
) -> dict[str, str]:
    step = checkpoint_step(checkpoint_path)
    base_row: dict[str, Any] = {
        "leave_out_dataset": leave_out_dataset,
        "condition_dataset": condition_dataset,
        "test_dataset": test_dataset,
        "evaluation_dataset": test_dataset,
        "checkpoint_step": step if step >= 0 else "",
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_path": str(checkpoint_path),
        "elapsed_sec": f"{elapsed_sec:.3f}",
        "returncode": returncode,
        "error_message": truncate_error(error_message),
    }

    if result is None:
        base_row.update(
            {
                "status": "error",
                "condition_ts_dataset": "",
                "ts_dataset": "",
                "evaluation_ts_dataset": "",
                "spearman_neg_corr": "",
                "spearman_neg_mse": "",
                "spearman_neg_correlation": "",
                "invalid_reason": "",
                "rpn_tokens": "",
                "infix": "",
                "latex": "",
                "benchmark_dataset": "",
                "benchmark_csv": "",
                "split": "",
                "split_count": "",
                "support_indices": "",
            }
        )
    else:
        spearman = result.get("spearman_neg_mse")
        invalid_reason = str(result.get("invalid_reason") or "")
        base_row.update(
            {
                "status": "invalid" if invalid_reason else "ok",
                "condition_dataset": result.get("condition_dataset", condition_dataset),
                "condition_ts_dataset": result.get("condition_ts_dataset", ""),
                "ts_dataset": result.get("ts_dataset", ""),
                "evaluation_dataset": result.get("evaluation_dataset", result.get("dataset", test_dataset)),
                "evaluation_ts_dataset": result.get("evaluation_ts_dataset", result.get("ts_dataset", "")),
                "spearman_neg_corr": spearman,
                "spearman_neg_mse": spearman,
                "spearman_neg_correlation": spearman,
                "invalid_reason": invalid_reason,
                "rpn_tokens": result.get("rpn_tokens", ""),
                "infix": result.get("infix", ""),
                "latex": result.get("latex", ""),
                "benchmark_dataset": result.get("benchmark_dataset", ""),
                "benchmark_csv": result.get("benchmark_csv", ""),
                "split": result.get("split", ""),
                "split_count": result.get("split_count", ""),
                "support_indices": result.get("support_indices", ()),
            }
        )

    return {fieldname: csv_value(base_row.get(fieldname, "")) for fieldname in CSV_FIELDNAMES}


def summarize_checkpoint_jobs(jobs: Iterable[tuple[str, Path]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for leave_out_dataset, _checkpoint in jobs:
        counts[leave_out_dataset] = counts.get(leave_out_dataset, 0) + 1
    return counts


def summarize_condition_jobs(jobs: Iterable[tuple[str, Path, str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _leave_out_dataset, _checkpoint, condition_dataset, _test_dataset in jobs:
        counts[condition_dataset] = counts.get(condition_dataset, 0) + 1
    return counts


def summarize_eval_jobs(jobs: Iterable[tuple[str, Path, str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _leave_out_dataset, _checkpoint, _condition_dataset, test_dataset in jobs:
        counts[test_dataset] = counts.get(test_dataset, 0) + 1
    return counts


def default_output_csv(matrix_test: bool) -> Path:
    if matrix_test:
        return Path("DCSPG/checkpoints/lodo/lodo_checkpoint_condition_eval_matrix_results.csv")
    return Path("DCSPG/checkpoints/lodo/lodo_checkpoint_all_dataset_test_results.csv")


def main() -> int:
    args = build_arg_parser().parse_args()
    repo_root = args.repo_root.resolve()
    train_script = resolve_path(args.train_script, repo_root).resolve()
    checkpoint_root = resolve_path(args.checkpoint_root, repo_root).resolve()
    output_csv = resolve_path(args.output_csv or default_output_csv(args.matrix_test), repo_root).resolve()
    ts_feature_dir = resolve_path(args.ts_feature_dir, repo_root).resolve()
    benchmark_dir = resolve_path(args.benchmark_dir, repo_root).resolve()
    test_datasets = parse_dataset_list(args.test_datasets, DATASET_ORDER)
    condition_datasets = parse_dataset_list(args.condition_datasets, DATASET_ORDER)

    if not train_script.is_file():
        raise FileNotFoundError(f"Missing train script: {train_script}")
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"Missing checkpoint root: {checkpoint_root}")

    checkpoint_jobs = discover_checkpoint_jobs(
        checkpoint_root=checkpoint_root,
        dataset_filter=parse_dataset_filter(args.leave_out_datasets),
        max_checkpoints_per_dataset=args.max_checkpoints_per_dataset,
    )
    jobs = (
        expand_matrix_jobs(checkpoint_jobs, condition_datasets, test_datasets)
        if args.matrix_test
        else expand_jobs(checkpoint_jobs, test_datasets)
    )

    checkpoint_counts = summarize_checkpoint_jobs(checkpoint_jobs)
    print(f"Discovered {len(checkpoint_jobs)} checkpoints under {checkpoint_root}")
    for dataset, count in sorted(checkpoint_counts.items(), key=lambda item: dataset_sort_key(item[0])):
        print(f"  leave_out={dataset}: {count}")
    if args.matrix_test:
        print(f"Condition datasets: {', '.join(condition_datasets)}")
    print(f"Evaluation datasets: {', '.join(test_datasets)}")
    print(f"Matrix test: {args.matrix_test}")
    print(f"Expanded to {len(jobs)} checkpoint/condition/evaluation jobs")

    condition_counts = summarize_condition_jobs(jobs)
    for dataset, count in sorted(condition_counts.items(), key=lambda item: dataset_sort_key(item[0])):
        print(f"  condition={dataset}: {count}")
    test_counts = summarize_eval_jobs(jobs)
    for dataset, count in sorted(test_counts.items(), key=lambda item: dataset_sort_key(item[0])):
        print(f"  evaluation={dataset}: {count}")

    if args.dry_run:
        return 0
    if not jobs:
        raise RuntimeError("No checkpoint/test-dataset jobs found.")

    completed_keys = read_completed_job_keys(output_csv) if args.resume else set()
    if completed_keys:
        before = len(jobs)
        jobs = [job for job in jobs if job_key(*job) not in completed_keys]
        print(f"Resume enabled: skipped {before - len(jobs)} already-recorded jobs.")
        if not jobs:
            print(f"No remaining jobs. Existing CSV: {output_csv}")
            return 0

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.resume or not output_csv.exists() or output_csv.stat().st_size == 0
    mode = "a" if args.resume else "w"
    failures = 0

    with output_csv.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()

        total = len(jobs)
        for index, (leave_out_dataset, checkpoint_path, condition_dataset, test_dataset) in enumerate(jobs, start=1):
            print(
                f"[{index}/{total}] leave_out={leave_out_dataset} "
                f"condition={condition_dataset} evaluation={test_dataset} {checkpoint_path.name}",
                flush=True,
            )
            result, returncode, error_message, elapsed_sec = run_checkpoint_test(
                python_executable=args.python_executable,
                train_script=train_script,
                repo_root=repo_root,
                checkpoint_path=checkpoint_path,
                dataset=test_dataset,
                condition_dataset=condition_dataset,
                ts_feature_dir=ts_feature_dir,
                benchmark_dir=benchmark_dir,
                device=args.device,
                gpu_id=args.gpu_id,
                k_samples=args.k_samples,
                test_seed=args.test_seed,
                test_split=args.test_split,
                test_max_len=args.test_max_len,
            )
            if result is None:
                failures += 1

            row = build_csv_row(
                leave_out_dataset=leave_out_dataset,
                condition_dataset=condition_dataset,
                test_dataset=test_dataset,
                checkpoint_path=checkpoint_path,
                result=result,
                returncode=returncode,
                error_message=error_message,
                elapsed_sec=elapsed_sec,
            )
            writer.writerow(row)
            handle.flush()

            status = row["status"]
            spearman = row["spearman_neg_corr"] or "NA"
            print(f"  status={status} spearman_neg_corr={spearman}", flush=True)

            if args.fail_fast and result is None:
                print(f"Stopping after failure. Partial CSV: {output_csv}")
                return 1

    print(f"Wrote {len(jobs)} rows to {output_csv}")
    if failures:
        print(f"Completed with {failures} subprocess failures. See error_message column.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
