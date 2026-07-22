from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Iterable


DATASET_ORDER = (
    "ETT-small",
    "electricity",
    "exchange_rate",
    "illness",
    "traffic",
    "weather",
)

CSV_FIELDNAMES = [
    "status",
    "ablation_mode",
    "leave_out_dataset",
    "test_dataset",
    "test_seed",
    "seed_index",
    "seed_count",
    "valid_seed_count",
    "spearman_mean",
    "spearman_std",
    "spearman_min",
    "spearman_max",
    "checkpoint_step",
    "checkpoint_name",
    "checkpoint_path",
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
    "beam_size",
    "beam_valid_count",
    "beam_rpn_tokens",
    "beam_infix",
    "beam_latex",
    "beam_log_probs",
    "beam_spearman_neg_mse",
    "beam_invalid_reasons",
    "elapsed_sec",
    "returncode",
    "error_message",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all periodic LODO DCSPG checkpoints on each fold's "
            "leave-out test dataset and collect Spearman(-MSE) results in one CSV."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--train-script", type=Path, default=Path("train_dcspg_framework.py")
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("DCSPG/checkpoints/"),
        help=(
            "Checkpoint root. Accepts either one timestamped run directory "
            "or the parent directory containing run_*/leave_out_* folders."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("DCSPG/checkpoints/lodo/lodo_checkpoint_test_results.csv"),
    )
    parser.add_argument("--ts-feature-dir", type=Path, default=Path("DCSPG/TS_dataset"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("DCSPG/Benchmark"))
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Optional comma-separated leave-out datasets to evaluate.",
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
    parser.add_argument(
        "--multiple-seeds",
        action="store_true",
        help=(
            "Run support-sampling sensitivity ablation. By default this evaluates "
            "--seed-count consecutive seeds starting at --test-seed."
        ),
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=5,
        help="Number of consecutive seeds used with --multiple-seeds.",
    )
    parser.add_argument(
        "--test-seeds",
        type=str,
        default="",
        help="Optional comma-separated seed list. Overrides --test-seed/--seed-count.",
    )
    parser.add_argument("--test-split", type=str, default="proxy_test")
    parser.add_argument("--test-max-len", type=int, default=None)
    parser.add_argument(
        "--beam-search",
        action="store_true",
        help="Use beam-search ensemble test mode for each checkpoint.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size used when --beam-search is enabled.",
    )
    parser.add_argument(
        "--max-checkpoints-per-dataset",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing CSV and skip checkpoint paths already present in it.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when a checkpoint subprocess fails or emits unparsable JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the planned checkpoint count without running tests.",
    )
    return parser


def resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def parse_dataset_filter(raw: str) -> set[str] | None:
    datasets = {item.strip() for item in raw.split(",") if item.strip()}
    return datasets or None


def parse_seed_list(raw: str) -> tuple[int, ...]:
    seeds = []
    seen = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        seed = int(item)
        if seed not in seen:
            seeds.append(seed)
            seen.add(seed)
    return tuple(seeds)


def resolve_test_seeds(test_seed: int, seed_count: int, test_seeds: str, multiple_seeds: bool) -> tuple[int, ...]:
    explicit_seeds = parse_seed_list(test_seeds)
    if explicit_seeds:
        return explicit_seeds
    if multiple_seeds:
        return tuple(range(test_seed, test_seed + seed_count))
    return (test_seed,)


def checkpoint_step(path: Path) -> int:
    match = re.search(r"step_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def dataset_sort_key(dataset: str) -> tuple[int, str]:
    try:
        return DATASET_ORDER.index(dataset), dataset
    except ValueError:
        return len(DATASET_ORDER), dataset


def discover_jobs(
    checkpoint_root: Path,
    dataset_filter: set[str] | None,
    max_checkpoints_per_dataset: int | None,
) -> list[tuple[str, Path]]:
    jobs: list[tuple[str, Path]] = []
    fold_dirs = []
    seen_fold_dirs: set[Path] = set()
    candidate_roots = [checkpoint_root]
    candidate_roots.extend(path for path in checkpoint_root.iterdir() if path.is_dir())

    for candidate_root in candidate_roots:
        for path in candidate_root.iterdir():
            if not path.is_dir() or not path.name.startswith("leave_out_"):
                continue
            if path in seen_fold_dirs:
                continue
            seen_fold_dirs.add(path)
            dataset = path.name.removeprefix("leave_out_")
            if dataset_filter is not None and dataset not in dataset_filter:
                continue
            checkpoint_dir = path / "checkpoints"
            if checkpoint_dir.is_dir():
                fold_dirs.append((dataset, checkpoint_dir))

    for dataset, checkpoint_dir in sorted(fold_dirs, key=lambda item: dataset_sort_key(item[0])):
        checkpoints = sorted(checkpoint_dir.glob("*.pt"), key=lambda path: (checkpoint_step(path), path.name))
        if max_checkpoints_per_dataset is not None:
            checkpoints = checkpoints[:max_checkpoints_per_dataset]
        jobs.extend((dataset, checkpoint) for checkpoint in checkpoints)
    return jobs


def read_completed_job_keys(output_csv: Path) -> set[tuple[str, str, str]]:
    if not output_csv.exists():
        return set()
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            (
                row["checkpoint_path"],
                row.get("test_seed", ""),
                row.get("beam_size", ""),
            )
            for row in reader
            if row.get("checkpoint_path")
        }


def extract_json_object(stdout: str) -> dict[str, Any]:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in train_dcspg_framework.py stdout.")
    return json.loads(stdout[start : end + 1])


def run_checkpoint_test(
    *,
    python_executable: str,
    train_script: Path,
    repo_root: Path,
    checkpoint_path: Path,
    dataset: str,
    condition_dataset: str | None = None,
    ts_feature_dir: Path,
    benchmark_dir: Path,
    device: str,
    gpu_id: int | None,
    k_samples: int,
    test_seed: int,
    test_split: str,
    test_max_len: int | None,
    beam_search: bool = False,
    beam_size: int = 5,
) -> tuple[dict[str, Any] | None, int, str, float]:
    command = [
        python_executable,
        str(train_script),
        "--test-checkpoint",
        str(checkpoint_path),
        "--test-dataset",
        dataset,
        "--ts-feature-dir",
        str(ts_feature_dir),
        "--benchmark-dir",
        str(benchmark_dir),
        "--device",
        device,
        "--k-samples",
        str(k_samples),
        "--test-seed",
        str(test_seed),
        "--test-split",
        test_split,
    ]
    if condition_dataset is not None:
        command.extend(["--condition-dataset", condition_dataset])
    if gpu_id is not None and gpu_id >= 0:
        command.extend(["--gpu-id", str(gpu_id)])
    if test_max_len is not None:
        command.extend(["--test-max-len", str(test_max_len)])
    if beam_search:
        command.extend(["--beam-search", "--beam-size", str(beam_size)])

    start_time = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed_sec = time.monotonic() - start_time
    if completed.returncode != 0:
        return None, completed.returncode, completed.stderr.strip() or completed.stdout.strip(), elapsed_sec
    try:
        return extract_json_object(completed.stdout), completed.returncode, completed.stderr.strip(), elapsed_sec
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        return None, completed.returncode, message.strip(), elapsed_sec


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value)
    if isinstance(value, (list, tuple)):
        if all(str(item) == "" for item in value):
            return ""
        return ";".join(str(item) for item in value)
    return str(value)


def truncate_error(message: str, max_chars: int = 2000) -> str:
    if len(message) <= max_chars:
        return message
    return message[: max_chars - 3] + "..."


def job_key(checkpoint_path: Path, test_seed: int, beam_size: int) -> tuple[str, str, str]:
    return str(checkpoint_path), str(test_seed), str(beam_size)


def spearman_summary(results: Iterable[dict[str, Any] | None]) -> dict[str, Any]:
    values = []
    for result in results:
        if result is None:
            continue
        try:
            value = float(result.get("spearman_neg_mse", float("nan")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)

    if not values:
        return {
            "valid_seed_count": 0,
            "spearman_mean": "",
            "spearman_std": "",
            "spearman_min": "",
            "spearman_max": "",
        }

    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "valid_seed_count": len(values),
        "spearman_mean": mean,
        "spearman_std": math.sqrt(variance),
        "spearman_min": min(values),
        "spearman_max": max(values),
    }


def build_csv_row(
    *,
    dataset: str,
    checkpoint_path: Path,
    result: dict[str, Any] | None,
    returncode: int,
    error_message: str,
    elapsed_sec: float,
    test_seed: int,
    seed_index: int,
    seed_count: int,
    seed_summary: dict[str, Any],
    beam_search: bool,
) -> dict[str, str]:
    step = checkpoint_step(checkpoint_path)
    if beam_search and seed_count > 1:
        ablation_mode = "beam_search_multiple_seeds"
    elif beam_search:
        ablation_mode = "beam_search"
    elif seed_count > 1:
        ablation_mode = "multiple_seeds"
    else:
        ablation_mode = "single_seed"
    base_row: dict[str, Any] = {
        "ablation_mode": ablation_mode,
        "leave_out_dataset": dataset,
        "test_dataset": dataset,
        "test_seed": test_seed,
        "seed_index": seed_index,
        "seed_count": seed_count,
        "valid_seed_count": seed_summary.get("valid_seed_count", ""),
        "spearman_mean": seed_summary.get("spearman_mean", ""),
        "spearman_std": seed_summary.get("spearman_std", ""),
        "spearman_min": seed_summary.get("spearman_min", ""),
        "spearman_max": seed_summary.get("spearman_max", ""),
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
                "beam_size": "",
                "beam_valid_count": "",
                "beam_rpn_tokens": "",
                "beam_infix": "",
                "beam_latex": "",
                "beam_log_probs": "",
                "beam_spearman_neg_mse": "",
                "beam_invalid_reasons": "",
            }
        )
    else:
        spearman = result.get("spearman_neg_mse")
        invalid_reason = str(result.get("invalid_reason") or "")
        base_row.update(
            {
                "status": "invalid" if invalid_reason else "ok",
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
                "beam_size": result.get("beam_size", ""),
                "beam_valid_count": result.get("beam_valid_count", ""),
                "beam_rpn_tokens": result.get("beam_rpn_tokens", ()),
                "beam_infix": result.get("beam_infix", ()),
                "beam_latex": result.get("beam_latex", ()),
                "beam_log_probs": result.get("beam_log_probs", ()),
                "beam_spearman_neg_mse": result.get("beam_spearman_neg_mse", ()),
                "beam_invalid_reasons": result.get("beam_invalid_reasons", ()),
            }
        )

    return {fieldname: csv_value(base_row.get(fieldname, "")) for fieldname in CSV_FIELDNAMES}


def summarize_jobs(jobs: Iterable[tuple[str, Path]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for dataset, _checkpoint in jobs:
        counts[dataset] = counts.get(dataset, 0) + 1
    return counts


def main() -> int:
    args = build_arg_parser().parse_args()
    repo_root = args.repo_root.resolve()
    train_script = resolve_path(args.train_script, repo_root).resolve()
    checkpoint_root = resolve_path(args.checkpoint_root, repo_root).resolve()
    output_csv = resolve_path(args.output_csv, repo_root).resolve()
    ts_feature_dir = resolve_path(args.ts_feature_dir, repo_root).resolve()
    benchmark_dir = resolve_path(args.benchmark_dir, repo_root).resolve()
    if args.seed_count <= 0:
        raise ValueError("--seed-count must be positive.")
    if args.beam_size <= 0:
        raise ValueError("--beam-size must be positive.")
    test_seeds = resolve_test_seeds(
        test_seed=args.test_seed,
        seed_count=args.seed_count,
        test_seeds=args.test_seeds,
        multiple_seeds=args.multiple_seeds,
    )
    effective_beam_size = args.beam_size if args.beam_search else 1

    if not train_script.is_file():
        raise FileNotFoundError(f"Missing train script: {train_script}")
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"Missing checkpoint root: {checkpoint_root}")

    jobs = discover_jobs(
        checkpoint_root=checkpoint_root,
        dataset_filter=parse_dataset_filter(args.datasets),
        max_checkpoints_per_dataset=args.max_checkpoints_per_dataset,
    )
    counts = summarize_jobs(jobs)
    print(f"Discovered {len(jobs)} checkpoints under {checkpoint_root}")
    for dataset, count in sorted(counts.items(), key=lambda item: dataset_sort_key(item[0])):
        print(f"  {dataset}: {count}")
    print(f"Seeds: {', '.join(str(seed) for seed in test_seeds)}")
    print(f"Beam search: {args.beam_search} beam_size={effective_beam_size}")
    print(f"Expanded to {len(jobs) * len(test_seeds)} checkpoint/seed jobs")

    if args.dry_run:
        return 0
    if not jobs:
        raise RuntimeError("No checkpoint jobs found.")

    completed_keys = read_completed_job_keys(output_csv) if args.resume else set()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.resume or not output_csv.exists() or output_csv.stat().st_size == 0
    mode = "a" if args.resume else "w"
    failures = 0
    written_rows = 0

    with output_csv.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()

        total = len(jobs)
        for index, (dataset, checkpoint_path) in enumerate(jobs, start=1):
            print(f"[{index}/{total}] {dataset} {checkpoint_path.name}", flush=True)
            seed_outputs: list[tuple[int, int, dict[str, Any] | None, int, str, float]] = []
            skipped = 0
            for seed_index, test_seed in enumerate(test_seeds, start=1):
                if job_key(checkpoint_path, test_seed, effective_beam_size) in completed_keys:
                    skipped += 1
                    continue
                print(f"  seed={test_seed} [{seed_index}/{len(test_seeds)}]", flush=True)
                result, returncode, error_message, elapsed_sec = run_checkpoint_test(
                    python_executable=args.python_executable,
                    train_script=train_script,
                    repo_root=repo_root,
                    checkpoint_path=checkpoint_path,
                    dataset=dataset,
                    ts_feature_dir=ts_feature_dir,
                    benchmark_dir=benchmark_dir,
                    device=args.device,
                    gpu_id=args.gpu_id,
                    k_samples=args.k_samples,
                    test_seed=test_seed,
                    test_split=args.test_split,
                    test_max_len=args.test_max_len,
                    beam_search=args.beam_search,
                    beam_size=args.beam_size,
                )
                if result is None:
                    failures += 1
                seed_outputs.append((seed_index, test_seed, result, returncode, error_message, elapsed_sec))

                if args.fail_fast and result is None:
                    seed_summary = spearman_summary([result])
                    row = build_csv_row(
                        dataset=dataset,
                        checkpoint_path=checkpoint_path,
                        result=result,
                        returncode=returncode,
                        error_message=error_message,
                        elapsed_sec=elapsed_sec,
                        test_seed=test_seed,
                        seed_index=seed_index,
                        seed_count=len(test_seeds),
                        seed_summary=seed_summary,
                        beam_search=args.beam_search,
                    )
                    writer.writerow(row)
                    written_rows += 1
                    handle.flush()
                    print(f"Stopping after failure. Partial CSV: {output_csv}")
                    return 1

            if skipped:
                print(f"  resume skipped {skipped} already-recorded seed jobs", flush=True)
            if not seed_outputs:
                continue

            seed_summary = spearman_summary(result for _idx, _seed, result, _rc, _err, _elapsed in seed_outputs)
            for seed_index, test_seed, result, returncode, error_message, elapsed_sec in seed_outputs:
                row = build_csv_row(
                    dataset=dataset,
                    checkpoint_path=checkpoint_path,
                    result=result,
                    returncode=returncode,
                    error_message=error_message,
                    elapsed_sec=elapsed_sec,
                    test_seed=test_seed,
                    seed_index=seed_index,
                    seed_count=len(test_seeds),
                    seed_summary=seed_summary,
                    beam_search=args.beam_search,
                )
                writer.writerow(row)
                written_rows += 1
                handle.flush()

                status = row["status"]
                spearman = row["spearman_neg_mse"] or "NA"
                print(f"    status={status} spearman_neg_mse={spearman}", flush=True)

    print(f"Wrote {written_rows} rows to {output_csv}")
    if failures:
        print(f"Completed with {failures} subprocess failures. See error_message column.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
