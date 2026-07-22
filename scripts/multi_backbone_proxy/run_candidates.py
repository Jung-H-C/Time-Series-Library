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
DEFAULT_FIXED_SEQ_LEN = 96
DEFAULT_FIXED_PRED_LEN = 96
ILI_FIXED_SEQ_LEN = 36
ILI_FIXED_PRED_LEN = 36


def fixed_seq_len_for(dataset: DatasetSpec) -> int:
    return ILI_FIXED_SEQ_LEN if dataset.name == "ILI" else DEFAULT_FIXED_SEQ_LEN


def fixed_pred_len_for(dataset: DatasetSpec) -> int:
    return ILI_FIXED_PRED_LEN if dataset.name == "ILI" else DEFAULT_FIXED_PRED_LEN


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
        help=(
            "Datasets to run. Use benchmarks, monash, time, or all for groups; "
            "individual horizon=96 datasets use Monash__<name> or TIME__<name>."
        ),
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
        choices=["all", "proxy_train", "proxy_valid", "proxy_test", "proxy_eval"],
        help="Candidate split filter.",
    )
    parser.add_argument("--candidate-ids", nargs="+", default=None, help="Optional candidate_id filter.")
    parser.add_argument(
        "--label-len",
        type=int,
        default=None,
        help="Override decoder label_len. Default: min(dataset default, fixed seq_len//2).",
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
    parser.add_argument("--train-sample-limit", type=int, default=5000)
    parser.add_argument("--val-sample-limit", type=int, default=600)
    parser.add_argument("--test-sample-limit", type=int, default=1400)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--multi-series-lru-size", type=int, default=8)
    parser.add_argument(
        "--gpu",
        "--gpu-ids",
        dest="gpu_ids",
        nargs="+",
        default=None,
        help=(
            "GPU id(s) passed to run.py. Supports space-separated or comma-separated values, "
            "e.g. --gpu 0 1 2 3 or --gpu 0,1,2,3."
        ),
    )
    parser.add_argument("--gpu-type", default=None, choices=["cuda", "mps"], help="GPU type passed to run.py.")
    parser.add_argument(
        "--gpu-memory-limit-mb",
        type=int,
        default=30000,
        help="Maximum aggregate memory budget used for scheduling on each GPU.",
    )
    parser.add_argument(
        "--job-gpu-memory-mb",
        type=int,
        default=2400,
        help="Memory reservation per job; default permits at most two jobs per 28000 MB GPU budget.",
    )
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
        default=16,
        help="Number of parallel worker threads. Each worker launches one independent run.py process at a time.",
    )
    parser.add_argument("--execute", action="store_true", help="Actually launch run.py. Default is dry-run.")
    return parser.parse_args()


def selected_datasets(values: list[str]) -> list[DatasetSpec]:
    tokens = [piece.strip() for value in values for piece in value.split(",") if piece.strip()]
    if tokens == ["all"] or "all" in tokens:
        return [DATASETS[name] for name in DATASETS]
    selected: list[DatasetSpec] = []
    for value in tokens:
        group = value.strip().lower()
        if group in {"benchmark", "benchmarks"}:
            selected.extend(dataset for dataset in DATASETS.values() if "__" not in dataset.name)
        elif group == "monash":
            selected.extend(dataset for dataset in DATASETS.values() if dataset.name.startswith("Monash__"))
        elif group == "time":
            selected.extend(dataset for dataset in DATASETS.values() if dataset.name.startswith("TIME__"))
        else:
            selected.append(DATASETS[normalize_dataset(value)])
    # Preserve registry/request order while removing duplicates from mixed groups.
    return list(dict.fromkeys(selected))


def selected_backbones(values: list[str]) -> set[str]:
    if values == ["all"] or "all" in values:
        return set(CANONICAL_BACKBONES)
    return {normalize_backbone(value) for value in values}


def parse_gpu_ids(values: list[str] | None) -> list[int]:
    if values is None:
        return []
    gpu_ids: list[int] = []
    for value in values:
        for piece in str(value).split(","):
            stripped = piece.strip()
            if not stripped:
                continue
            gpu_id = int(stripped)
            if gpu_id < 0:
                raise ValueError(f"GPU id must be non-negative, got: {gpu_id}")
            gpu_ids.append(gpu_id)
    if not gpu_ids:
        raise ValueError("--gpu/--gpu-ids requires at least one GPU id.")
    return gpu_ids


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
    run_args.pop("seq_len", None)
    seq_len = fixed_seq_len_for(dataset)
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
        "long_term_train_sample_limit": args.train_sample_limit,
        "long_term_val_sample_limit": args.val_sample_limit,
        "long_term_test_sample_limit": args.test_sample_limit,
        "candidate_sample_seed": args.sample_seed,
        "multi_series_lru_size": args.multi_series_lru_size,
        # Leave headroom for CUDA context/non-allocator memory within the
        # scheduler's reservation for this process.
        "gpu_memory_limit_mb": max(256, args.job_gpu_memory_mb - 512),
    }

    if args.train_epochs is not None:
        base_args["train_epochs"] = args.train_epochs
    if args.patience is not None:
        base_args["patience"] = args.patience
    if args.num_workers is not None:
        base_args["num_workers"] = args.num_workers
    elif dataset.data == "multi_series":
        # Avoid one independent bounded LRU cache per DataLoader worker.
        base_args["num_workers"] = 0
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


def current_gpu_memory_mb(gpu_ids: list[int]) -> dict[int, int]:
    if not gpu_ids:
        return {}
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        usage = {}
        for line in completed.stdout.splitlines():
            index_text, used_text = (piece.strip() for piece in line.split(",", 1))
            usage[int(index_text)] = int(used_text)
        return {gpu_id: usage.get(gpu_id, 0) for gpu_id in gpu_ids}
    except (OSError, subprocess.SubprocessError, ValueError):
        raise RuntimeError(
            "Could not query nvidia-smi; refusing to launch because the 15000 MB/GPU "
            "budget cannot be verified. Use --no-use-gpu for CPU execution."
        )


def worker_gpu_assignments(
    gpu_ids: list[int],
    worker_limit: int,
    memory_limit_mb: int,
    job_memory_mb: int,
) -> list[int | None]:
    if worker_limit <= 0:
        return []
    if not gpu_ids:
        return [None] * worker_limit
    if memory_limit_mb <= 0 or job_memory_mb <= 0:
        raise ValueError("GPU memory limits must be positive")
    usage = current_gpu_memory_mb(gpu_ids)
    slots = []
    slot_counts = {}
    for gpu_id in gpu_ids:
        available = max(0, memory_limit_mb - usage[gpu_id])
        count = available // job_memory_mb
        slot_counts[gpu_id] = count
        slots.extend([gpu_id] * count)
    if not slots:
        details = ", ".join(f"gpu {gpu}: {usage[gpu]} MB used" for gpu in gpu_ids)
        raise RuntimeError(
            f"No GPU reservation slot fits the {memory_limit_mb} MB budget "
            f"with {job_memory_mb} MB/job ({details})"
        )
    # Interleave GPUs so the first wave is balanced instead of filling GPU 0 first.
    balanced = []
    for slot_index in range(max(slot_counts.values())):
        for gpu_id in gpu_ids:
            if slot_index < slot_counts[gpu_id]:
                balanced.append(gpu_id)
    print(
        "GPU memory scheduler: "
        + ", ".join(
            f"gpu {gpu}: used={usage[gpu]} MB slots={slot_counts[gpu]}"
            for gpu in gpu_ids
        )
        + f"; budget={memory_limit_mb} MB, reservation={job_memory_mb} MB/job"
    )
    return balanced[:worker_limit]


def command_for_assigned_gpu(command: list[str], gpu_id: int | None) -> list[str]:
    assigned = list(command)
    if gpu_id is not None:
        assigned.extend(["--gpu", str(gpu_id)])
    return assigned


def gpu_assignment_summary(assignments: list[int | None]) -> str:
    if not assignments:
        return "no workers"
    counts: dict[int | None, int] = {}
    for gpu_id in assignments:
        counts[gpu_id] = counts.get(gpu_id, 0) + 1
    if set(counts) == {None}:
        return "no explicit GPU assignment"
    return ", ".join(
        f"gpu {gpu_id}: {count}"
        for gpu_id, count in sorted(
            ((gpu_id, count) for gpu_id, count in counts.items() if gpu_id is not None),
            key=lambda item: item[0],
        )
    )


def execute_job(
    index: int,
    total: int,
    job: dict[str, Any],
    args: argparse.Namespace,
    repo_root: Path,
    io_lock: threading.Lock,
    assigned_gpu: int | None,
) -> dict[str, Any]:
    log_path = args.log_dir / (
        f"{index:05d}_{job['dataset']}_{job['candidate_id']}_pl{job['pred_len']}.log"
    )
    start_time = time.time()
    command = command_for_assigned_gpu(job["command"], assigned_gpu)
    runtime_job = {**job, "command": command, "assigned_gpu": assigned_gpu}

    with io_lock:
        gpu_text = f" gpu={assigned_gpu}" if assigned_gpu is not None else ""
        print(
            f"[{index}/{total}] {job['dataset']} {job['candidate_id']} "
            f"pred_len={job['pred_len']}{gpu_text}"
        )
        print(f"  log: {log_path}")
        write_manifest(args.manifest, {**runtime_job, "status": "started", "log_path": str(log_path)})

    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(shlex.join(command) + "\n\n")
        log_handle.flush()
        completed = subprocess.run(
            command,
            cwd=repo_root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
            env=os.environ.copy(),
        )

    elapsed = time.time() - start_time
    status = "completed" if completed.returncode == 0 else "failed"
    record = {
        **runtime_job,
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

    def worker(worker_id: int, assigned_gpu: int | None) -> None:
        while not stop_event.is_set():
            try:
                index, job = work_queue.get_nowait()
            except queue.Empty:
                return
            try:
                try:
                    record = execute_job(
                        index,
                        total,
                        job,
                        args,
                        repo_root,
                        io_lock,
                        assigned_gpu,
                    )
                except Exception as exc:
                    record = {
                        **job,
                        "status": "failed",
                        "returncode": None,
                        "error": repr(exc),
                        "assigned_gpu": assigned_gpu,
                    }
                    with io_lock:
                        write_manifest(args.manifest, record)
                        print(
                            f"[{index}/{total}] worker={worker_id} failed before completion: "
                            f"{job['candidate_id']} ({exc!r})"
                        )
                if record["returncode"] != 0:
                    with failure_lock:
                        failures.append(record)
                    if not args.keep_going:
                        stop_event.set()
            finally:
                work_queue.task_done()

    worker_limit = min(args.n_jobs, total) if total > 0 else 0
    gpu_assignments = worker_gpu_assignments(
        args.gpu_ids,
        worker_limit,
        args.gpu_memory_limit_mb,
        args.job_gpu_memory_mb,
    )
    worker_count = len(gpu_assignments)
    threads = [
        threading.Thread(
            target=worker,
            args=(worker_index + 1, gpu_assignments[worker_index]),
            name=f"candidate-worker-{worker_index + 1}",
            daemon=False,
        )
        for worker_index in range(worker_count)
    ]

    with io_lock:
        print(f"Launching {worker_count} worker thread(s) for {total} job(s).")
        print(f"GPU worker assignment: {gpu_assignment_summary(gpu_assignments)}")
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    return failures


def main() -> None:
    args = parse_args()
    if args.gpu_ids is None and not args.no_use_gpu:
        args.gpu_ids = ["0", "1", "2", "3"]
    args.gpu_ids = parse_gpu_ids(args.gpu_ids)
    if args.no_use_gpu and args.gpu_ids:
        raise ValueError("--gpu/--gpu-ids cannot be used together with --no-use-gpu.")
    repo_root = args.repo_root.resolve()
    if not (repo_root / "run.py").exists():
        raise FileNotFoundError(f"run.py not found under repo root: {repo_root}")
    args.repo_root = repo_root

    datasets = selected_datasets(args.datasets)
    backbone_filter = selected_backbones(args.backbones)
    split_filter = None if "all" in args.splits else set(args.splits)
    id_filter = set(args.candidate_ids) if args.candidate_ids else None
    extra_args = parse_extra_run_args(args.extra_run_arg)

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
            pred_len = fixed_pred_len_for(dataset)
            job = build_job(candidate, dataset, pred_len, args, extra_args)
            if args.skip_existing and Path(job["metrics_path"]).exists():
                job["status"] = "skipped_existing"
                write_manifest(args.manifest, job)
                continue
            jobs.append(job)

    if args.limit is not None:
        jobs = jobs[: args.limit]

    print(f"Expanded {len(jobs)} jobs from {len(candidates)} candidates.")
    if not args.execute:
        worker_limit = min(args.n_jobs, len(jobs)) if jobs else 0
        gpu_assignments = worker_gpu_assignments(
            args.gpu_ids,
            worker_limit,
            args.gpu_memory_limit_mb,
            args.job_gpu_memory_mb,
        )
        worker_count = len(gpu_assignments)
        if worker_count:
            print(f"Dry-run GPU worker assignment: {gpu_assignment_summary(gpu_assignments)}")
        for index, job in enumerate(jobs, start=1):
            assigned_gpu = (
                gpu_assignments[(index - 1) % worker_count]
                if worker_count
                else None
            )
            command = command_for_assigned_gpu(job["command"], assigned_gpu)
            print(shlex.join(command))
            write_manifest(
                args.manifest,
                {
                    **job,
                    "command": command,
                    "assigned_gpu": assigned_gpu,
                    "status": "planned",
                },
            )
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
