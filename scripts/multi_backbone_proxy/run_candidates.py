from __future__ import annotations

import argparse
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from proxy_experiment_config import (
    CANONICAL_BACKBONES,
    DATASETS,
    DatasetSpec,
    normalize_backbone,
    normalize_dataset,
)


BOOL_FLAGS = {"individual", "inverse", "use_amp", "use_dtw", "no_use_gpu"}


def load_candidates(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError(f"{path} does not contain a top-level candidates list.")
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sampled multi-backbone candidates on long-term forecasting datasets."
    )
    parser.add_argument("--candidates", type=Path, required=True, help="Candidate JSON from sample_candidates.py.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Time-Series-Library repo root. Default: inferred from this script path.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help="Datasets to run: ECL ETTh1 Exchange ILI Traffic Weather, or all.",
    )
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=["all"],
        help="Backbone filter, or all.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["all"],
        choices=["all", "proxy_train", "proxy_eval"],
        help="Candidate split filter.",
    )
    parser.add_argument("--candidate-ids", nargs="+", default=None, help="Optional candidate_id filter.")
    parser.add_argument(
        "--pred-lens",
        type=int,
        nargs="+",
        default=None,
        help="Prediction horizons. Default: each dataset's canonical horizons.",
    )
    parser.add_argument(
        "--dataset-pred-lens",
        dest="dataset_pred_len_overrides",
        action="append",
        default=[],
        metavar="DATASET=PRED_LEN[,PRED_LEN...]",
        help=(
            "Dataset-specific prediction horizon override. Can be repeated. "
            "Example: --dataset-pred-lens Exchange=96 --dataset-pred-lens ILI=24"
        ),
    )
    parser.add_argument(
        "--fixed-seq-len",
        type=int,
        default=None,
        help="Override sampled candidate seq_len for all runs.",
    )
    parser.add_argument(
        "--dataset-seq-len",
        dest="dataset_seq_len_overrides",
        action="append",
        default=[],
        metavar="DATASET=SEQ_LEN",
        help=(
            "Dataset-specific seq_len override. Can be repeated. "
            "Example: --dataset-seq-len Exchange=96 --dataset-seq-len ILI=36"
        ),
    )
    parser.add_argument(
        "--label-len",
        type=int,
        default=None,
        help="Override decoder label_len. Default: min(dataset default, seq_len//2).",
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=None,
        help="Override run.py train_epochs. Default: run.py default.",
    )
    parser.add_argument("--patience", type=int, default=None, help="Override early stopping patience.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override dataset default batch size.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override candidate learning_rate.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override run.py num_workers.")
    parser.add_argument("--gpu", type=int, default=None, help="GPU id passed to run.py.")
    parser.add_argument("--gpu-type", default=None, choices=["cuda", "mps"], help="GPU type passed to run.py.")
    parser.add_argument("--no-use-gpu", action="store_true", help="Pass --no_use_gpu to run.py.")
    parser.add_argument(
        "--python-cmd",
        nargs="+",
        default=[sys.executable],
        help="Python command prefix. Example: --python-cmd conda run -n tslib python",
    )
    parser.add_argument("--run-group", default="mbproxy", help="Tag used in model_id/results_id/des.")
    parser.add_argument(
        "--checkpoints",
        default="./checkpoints/multi_backbone_proxy/",
        help="Checkpoint root passed to run.py.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/multi_backbone_proxy"),
        help="Directory for per-run stdout/stderr logs when --execute is set.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional JSONL manifest for planned/executed jobs.",
    )
    parser.add_argument(
        "--extra-run-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra run.py argument. Can be repeated.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Run/print only the first N expanded jobs.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip jobs with an existing metrics.npy.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed run.")
    parser.add_argument(
        "--n_jobs",
        "--n-jobs",
        dest="n_jobs",
        type=int,
        default=1,
        help="Number of parallel worker threads. Each worker launches one independent run.py process at a time.",
    )
    parser.add_argument("--execute", action="store_true", help="Actually launch run.py. Default is dry-run.")
    return parser.parse_args()


def selected_datasets(values: list[str]) -> list[DatasetSpec]:
    if values == ["all"] or "all" in values:
        return [DATASETS[name] for name in DATASETS]
    return [DATASETS[normalize_dataset(value)] for value in values]


def selected_backbones(values: list[str]) -> set[str]:
    if values == ["all"] or "all" in values:
        return set(CANONICAL_BACKBONES)
    return {normalize_backbone(value) for value in values}


def parse_extra_run_args(items: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--extra-run-arg must be KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip().lstrip("-").replace("-", "_")
        if not key:
            raise ValueError(f"Empty --extra-run-arg key in: {item}")
        result[key] = value
    return result


def split_dataset_override(item: str, option_name: str) -> tuple[str, str]:
    if "=" not in item:
        raise ValueError(f"{option_name} must be DATASET=VALUE, got: {item}")
    dataset_name, raw_value = item.split("=", 1)
    dataset_name = dataset_name.strip()
    raw_value = raw_value.strip()
    if not dataset_name or not raw_value:
        raise ValueError(f"{option_name} must be DATASET=VALUE, got: {item}")
    return normalize_dataset(dataset_name), raw_value


def parse_dataset_int_overrides(items: list[str], option_name: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in items:
        dataset_name, raw_value = split_dataset_override(item, option_name)
        if dataset_name in result:
            raise ValueError(f"Duplicate {option_name} override for dataset: {dataset_name}")
        value = int(raw_value)
        if value <= 0:
            raise ValueError(f"{option_name} value must be positive for {dataset_name}: {value}")
        result[dataset_name] = value
    return result


def parse_dataset_int_list_overrides(items: list[str], option_name: str) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for item in items:
        dataset_name, raw_value = split_dataset_override(item, option_name)
        if dataset_name in result:
            raise ValueError(f"Duplicate {option_name} override for dataset: {dataset_name}")
        values = [int(value.strip()) for value in raw_value.split(",") if value.strip()]
        if not values:
            raise ValueError(f"{option_name} needs at least one value for {dataset_name}")
        if any(value <= 0 for value in values):
            raise ValueError(f"{option_name} values must be positive for {dataset_name}: {raw_value}")
        result[dataset_name] = values
    return result


def add_arg(command: list[str], key: str, value: Any) -> None:
    flag = f"--{key}"
    if key in BOOL_FLAGS:
        if bool(value):
            command.append(flag)
        return
    if value is None:
        return
    command.extend([flag, str(value)])


def label_len_for(dataset: DatasetSpec, seq_len: int, override: int | None) -> int:
    if override is not None:
        return override
    return max(1, min(dataset.default_label_len, seq_len // 2))


def result_metrics_path(repo_root: Path, model_id: str, results_id: str) -> Path:
    folder_name = f"long_term_forecast_{model_id}_{results_id}"
    return repo_root / "results" / folder_name / "metrics.npy"


def build_job(
    candidate: dict[str, Any],
    dataset: DatasetSpec,
    pred_len: int,
    args: argparse.Namespace,
    extra_args: dict[str, str],
) -> dict[str, Any]:
    candidate_id = candidate["candidate_id"]
    backbone = candidate["backbone"]
    run_args = dict(candidate.get("run_args") or {})
    sampled_seq_len = run_args.pop("seq_len", dataset.default_seq_len)
    seq_len_override = args.dataset_seq_len_overrides.get(dataset.name)
    if seq_len_override is not None:
        seq_len_value = seq_len_override
    elif args.fixed_seq_len is not None:
        seq_len_value = args.fixed_seq_len
    else:
        seq_len_value = sampled_seq_len
    seq_len = int(seq_len_value)
    label_len = label_len_for(dataset, seq_len, args.label_len)
    model_id = f"{args.run_group}_{dataset.name}_{candidate_id}_sl{seq_len}_pl{pred_len}"
    results_id = f"{args.run_group}_{candidate.get('split', 'unsplit')}_{dataset.name}_{candidate_id}_pl{pred_len}"

    base_args: dict[str, Any] = {
        "task_name": "long_term_forecast",
        "is_training": 1,
        "root_path": dataset.root_path,
        "data_path": dataset.data_path,
        "model_id": model_id,
        "model": backbone,
        "data": dataset.data,
        "features": dataset.features,
        "target": dataset.target,
        "freq": dataset.freq,
        "seq_len": seq_len,
        "label_len": label_len,
        "pred_len": pred_len,
        "enc_in": dataset.enc_in,
        "dec_in": dataset.enc_in,
        "c_out": dataset.enc_in,
        "des": args.run_group,
        "itr": 1,
        "checkpoints": args.checkpoints,
        "results_id": results_id,
        "batch_size": args.batch_size if args.batch_size is not None else dataset.default_batch_size,
    }

    if args.train_epochs is not None:
        base_args["train_epochs"] = args.train_epochs
    if args.patience is not None:
        base_args["patience"] = args.patience
    if args.num_workers is not None:
        base_args["num_workers"] = args.num_workers
    if args.gpu is not None:
        base_args["gpu"] = args.gpu
    if args.gpu_type is not None:
        base_args["gpu_type"] = args.gpu_type
    if args.no_use_gpu:
        base_args["no_use_gpu"] = True
    if args.learning_rate is not None:
        run_args["learning_rate"] = args.learning_rate

    merged_args = {**base_args, **run_args, **extra_args}
    command = list(args.python_cmd) + ["-u", "run.py"]
    for key, value in merged_args.items():
        add_arg(command, key, value)

    return {
        "candidate_id": candidate_id,
        "backbone": backbone,
        "split": candidate.get("split"),
        "dataset": dataset.name,
        "pred_len": pred_len,
        "seq_len": seq_len,
        "label_len": label_len,
        "model_id": model_id,
        "results_id": results_id,
        "command": command,
        "metrics_path": str(result_metrics_path(args.repo_root, model_id, results_id)),
    }


def write_manifest(path: Path | None, record: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def execute_job(
    index: int,
    total: int,
    job: dict[str, Any],
    args: argparse.Namespace,
    repo_root: Path,
    io_lock: threading.Lock,
) -> dict[str, Any]:
    log_path = args.log_dir / (
        f"{index:05d}_{job['dataset']}_{job['candidate_id']}_pl{job['pred_len']}.log"
    )
    start_time = time.time()

    with io_lock:
        print(f"[{index}/{total}] {job['dataset']} {job['candidate_id']} pred_len={job['pred_len']}")
        print(f"  log: {log_path}")
        write_manifest(args.manifest, {**job, "status": "started", "log_path": str(log_path)})

    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(shlex.join(job["command"]) + "\n\n")
        log_handle.flush()
        completed = subprocess.run(
            job["command"],
            cwd=repo_root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
            env=os.environ.copy(),
        )

    elapsed = time.time() - start_time
    status = "completed" if completed.returncode == 0 else "failed"
    record = {
        **job,
        "status": status,
        "returncode": completed.returncode,
        "elapsed_sec": round(elapsed, 3),
        "log_path": str(log_path),
    }
    with io_lock:
        write_manifest(args.manifest, record)
        if completed.returncode == 0:
            print(f"[{index}/{total}] completed in {elapsed:.1f}s: {job['candidate_id']}")
        else:
            print(f"[{index}/{total}] failed rc={completed.returncode}: {job['candidate_id']}")
    return record


def execute_jobs_parallel(
    jobs: list[dict[str, Any]],
    args: argparse.Namespace,
    repo_root: Path,
) -> list[dict[str, Any]]:
    if args.n_jobs < 1:
        raise ValueError("--n_jobs must be a positive integer.")

    total = len(jobs)
    work_queue: queue.Queue[tuple[int, dict[str, Any]]] = queue.Queue()
    for index, job in enumerate(jobs, start=1):
        work_queue.put((index, job))

    io_lock = threading.Lock()
    stop_event = threading.Event()
    failures: list[dict[str, Any]] = []
    failure_lock = threading.Lock()

    def worker() -> None:
        while not stop_event.is_set():
            try:
                index, job = work_queue.get_nowait()
            except queue.Empty:
                return
            try:
                try:
                    record = execute_job(index, total, job, args, repo_root, io_lock)
                except Exception as exc:
                    record = {
                        **job,
                        "status": "failed",
                        "returncode": None,
                        "error": repr(exc),
                    }
                    with io_lock:
                        write_manifest(args.manifest, record)
                        print(f"[{index}/{total}] failed before completion: {job['candidate_id']} ({exc!r})")
                if record["returncode"] != 0:
                    with failure_lock:
                        failures.append(record)
                    if not args.keep_going:
                        stop_event.set()
            finally:
                work_queue.task_done()

    worker_count = min(args.n_jobs, total) if total > 0 else 0
    threads = [
        threading.Thread(target=worker, name=f"candidate-worker-{worker_index + 1}", daemon=False)
        for worker_index in range(worker_count)
    ]

    with io_lock:
        print(f"Launching {worker_count} worker thread(s) for {total} job(s).")
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    return failures


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    if not (repo_root / "run.py").exists():
        raise FileNotFoundError(f"run.py not found under repo root: {repo_root}")
    args.repo_root = repo_root

    datasets = selected_datasets(args.datasets)
    backbone_filter = selected_backbones(args.backbones)
    split_filter = None if "all" in args.splits else set(args.splits)
    id_filter = set(args.candidate_ids) if args.candidate_ids else None
    extra_args = parse_extra_run_args(args.extra_run_arg)
    args.dataset_seq_len_overrides = parse_dataset_int_overrides(
        args.dataset_seq_len_overrides, "--dataset-seq-len"
    )
    args.dataset_pred_len_overrides = parse_dataset_int_list_overrides(
        args.dataset_pred_len_overrides, "--dataset-pred-lens"
    )

    candidates = []
    for candidate in load_candidates(args.candidates):
        if candidate.get("backbone") not in backbone_filter:
            continue
        if split_filter is not None and candidate.get("split") not in split_filter:
            continue
        if id_filter is not None and candidate.get("candidate_id") not in id_filter:
            continue
        candidates.append(candidate)

    jobs: list[dict[str, Any]] = []
    for candidate in candidates:
        for dataset in datasets:
            pred_lens = args.dataset_pred_len_overrides.get(dataset.name)
            if pred_lens is None:
                pred_lens = args.pred_lens if args.pred_lens is not None else dataset.pred_lens
            for pred_len in pred_lens:
                job = build_job(candidate, dataset, int(pred_len), args, extra_args)
                if args.skip_existing and Path(job["metrics_path"]).exists():
                    job["status"] = "skipped_existing"
                    write_manifest(args.manifest, job)
                    continue
                jobs.append(job)

    if args.limit is not None:
        jobs = jobs[: args.limit]

    print(f"Expanded {len(jobs)} jobs from {len(candidates)} candidates.")
    if not args.execute:
        for job in jobs:
            print(shlex.join(job["command"]))
            write_manifest(args.manifest, {**job, "status": "planned"})
        print("Dry-run only. Add --execute to launch jobs.")
        return

    args.log_dir.mkdir(parents=True, exist_ok=True)
    failures = execute_jobs_parallel(jobs, args, repo_root)
    if failures and not args.keep_going:
        first = failures[0]
        raise SystemExit(
            f"Job failed with return code {first['returncode']}. See {first['log_path']}"
        )
    if failures:
        print(f"Finished with {len(failures)} failed job(s).")


if __name__ == "__main__":
    main()
