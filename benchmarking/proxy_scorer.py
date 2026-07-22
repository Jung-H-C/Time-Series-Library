from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
import os
import queue
import random
import re
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader

from benchmarking.candidate_sampler import (
    _prepare_candidate_run_args,
    discover_run_argument_defaults,
)

try:
    from scripts.multi_backbone_proxy.proxy_experiment_config import (
        DATASETS as MB_PROXY_DATASETS,
        normalize_dataset as _normalize_mb_proxy_dataset,
    )
except Exception:
    MB_PROXY_DATASETS = {}
    _normalize_mb_proxy_dataset = None


BN_TYPES = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
)

DROPOUT_TYPES = (
    torch.nn.Dropout,
    torch.nn.Dropout1d,
    torch.nn.Dropout2d,
    torch.nn.Dropout3d,
    torch.nn.AlphaDropout,
    torch.nn.FeatureAlphaDropout,
)

ALL_PROXY_COLUMNS = [
    "params",
    "l2_norm",
    "flops",
    "grad_norm",
    "zico",
    "fisher",
    "grasp",
    "jacob_cov",
    "jacob_fro",
    "plain",
    "snip",
    "synflow",
]

DATA_INDEPENDENT_PROXY_COLUMNS = {"params", "l2_norm"}

CSV_PROXY_COLUMN_LABELS = {
    "params": "MParams",
    "flops": "GFLOPs",
    "synflow": "GSynFlow",
}

CSV_PROXY_COLUMN_SCALES = {
    "params": 1e6,
    "flops": 1e9,
    "synflow": 1e9,
}

META_COLUMNS = [
    "candidate_id",
    "candidate_name",
    "model",
    "split",
    "task_name",
    "data",
    "dataset_name",
    "dataset_family",
    "num_batches",
    "status",
    "error",
]

SEPARATE_META_COLUMNS = [
    "candidate_id",
    "candidate_name",
    "model",
    "split",
    "task_name",
    "data",
    "dataset_name",
    "dataset_family",
    "run_index",
    "run_name",
    "batch_index",
    "num_batches",
    "status",
    "error",
]

CANDIDATE_SUFFIX_PATTERN = re.compile(r"_(\d+)$")
SUPPORTED_TASK_NAME = "long_term_forecast"
DATASET_FAMILY_NAMES = ("Benchmark", "Monash", "TIME")
EXPECTED_DATASET_FAMILY_COUNTS = {"Benchmark": 6, "Monash": 18, "TIME": 29}
BENCHMARK_DATASET_ALIASES = {
    "ecl": "ECL",
    "electricity": "ECL",
    "etth1": "ETTh1",
    "ett": "ETTh1",
    "exchange": "Exchange",
    "exchangerate": "Exchange",
    "ili": "ILI",
    "illness": "ILI",
    "traffic": "Traffic",
    "weather": "Weather",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _slugify(value: str) -> str:
    return value.replace("\\", "/").split("/")[-1].replace(".json", "").replace("_candidates", "").lower()


def _case_insensitive_child(directory: Path, filename: str) -> Path | None:
    if not directory.exists():
        return None
    wanted = filename.casefold()
    for child in directory.iterdir():
        if child.name.casefold() == wanted:
            return child
    return None


def _resolve_candidates_path(candidate_name: str, repo_root: Path | None = None) -> Path:
    repo_root = repo_root or _repo_root()
    candidates_dir = repo_root / "candidates"
    raw_path = Path(candidate_name)
    normalized = _slugify(candidate_name)
    path_options: list[Path] = []

    if raw_path.is_absolute():
        path_options.append(raw_path)
    else:
        path_options.append(repo_root / raw_path)
        path_options.append(candidates_dir / raw_path.name)

    if raw_path.suffix != ".json":
        path_options.append(candidates_dir / f"{raw_path.name}.json")
    path_options.append(candidates_dir / f"{normalized}.json")
    path_options.append(candidates_dir / f"{normalized}_candidates.json")

    seen: set[Path] = set()
    for candidate_path in path_options:
        if candidate_path in seen:
            continue
        seen.add(candidate_path)
        if candidate_path.exists():
            return candidate_path.resolve()
        if candidate_path.parent == candidates_dir:
            matched_path = _case_insensitive_child(candidates_dir, candidate_path.name)
            if matched_path is not None:
                return matched_path.resolve()

    raise ValueError(
        f"Candidate JSON not found for '{candidate_name}'. Tried direct paths and candidates/<name>.json."
    )


def _is_compact_mbproxy_candidate(candidate: dict[str, Any]) -> bool:
    if not isinstance(candidate, dict) or "backbone" not in candidate:
        return False
    run_args = candidate.get("run_args")
    if not isinstance(run_args, dict):
        return True
    return not {"model", "task_name", "data"}.issubset(run_args)


def _uses_compact_mbproxy_schema(payload: dict[str, Any]) -> bool:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return False
    return any(_is_compact_mbproxy_candidate(candidate) for candidate in candidates)


def _require_mbproxy_dataset(dataset_name: str):
    if _normalize_mb_proxy_dataset is None or not MB_PROXY_DATASETS:
        raise ValueError(
            "Compact multi-backbone candidate JSONs require "
            "scripts/multi_backbone_proxy/proxy_experiment_config.py."
        )
    canonical_name = _normalize_mb_proxy_dataset(dataset_name)
    return MB_PROXY_DATASETS[canonical_name]


def _dataset_family(dataset_name: str) -> str:
    if dataset_name.startswith("Monash__"):
        return "Monash"
    if dataset_name.startswith("TIME__"):
        return "TIME"
    return "Benchmark"


def _registry_family_names(family: str) -> list[str]:
    names = [name for name in MB_PROXY_DATASETS if _dataset_family(name) == family]
    expected = EXPECTED_DATASET_FAMILY_COUNTS[family]
    if len(names) != expected:
        raise ValueError(
            f"Expected {expected} {family} datasets in the proxy registry, found {len(names)}."
        )
    return names


def _split_dataset_selectors(raw_selectors: list[str] | str | None) -> list[str]:
    values = [raw_selectors] if isinstance(raw_selectors, str) else list(raw_selectors or [])
    selectors = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if token:
                selectors.append(token)
    return selectors or ["ECL"]


def _resolve_scoring_dataset_names(raw_selectors: list[str] | str | None) -> list[str]:
    """Expand explicit dataset names and Benchmark/Monash/TIME/all selectors."""
    if _normalize_mb_proxy_dataset is None or not MB_PROXY_DATASETS:
        raise ValueError(
            "Dataset selectors require scripts/multi_backbone_proxy/proxy_experiment_config.py."
        )

    for family in DATASET_FAMILY_NAMES:
        _registry_family_names(family)
    expected_total = sum(EXPECTED_DATASET_FAMILY_COUNTS.values())
    if len(MB_PROXY_DATASETS) != expected_total:
        raise ValueError(
            f"Expected {expected_total} total datasets in the proxy registry, "
            f"found {len(MB_PROXY_DATASETS)}."
        )

    selectors = _split_dataset_selectors(raw_selectors)
    selected: list[str] = []
    for selector in selectors:
        compact = re.sub(r"[^a-z0-9]", "", selector.lower())
        if compact == "all":
            expanded = list(MB_PROXY_DATASETS)
        elif compact in {"benchmark", "benchmarks"}:
            expanded = _registry_family_names("Benchmark")
        elif compact == "monash":
            expanded = _registry_family_names("Monash")
        elif compact == "time":
            expanded = _registry_family_names("TIME")
        else:
            alias = BENCHMARK_DATASET_ALIASES.get(compact, selector)
            try:
                expanded = [_normalize_mb_proxy_dataset(alias)]
            except ValueError as exc:
                family_choices = ", ".join((*DATASET_FAMILY_NAMES, "all"))
                raise ValueError(
                    f"Unknown dataset/family selector {selector!r}. "
                    f"Use a registry dataset name or one of: {family_choices}."
                ) from exc
        for dataset_name in expanded:
            if dataset_name not in selected:
                selected.append(dataset_name)

    if not selected:
        raise ValueError("Dataset selection resolved to zero datasets.")
    return selected


def _default_label_len(dataset: Any, seq_len: int, override: int | None) -> int:
    if override is not None:
        return int(override)
    dataset_label_len = int(getattr(dataset, "default_label_len", 48))
    return max(1, min(dataset_label_len, seq_len // 2))


def _default_pred_len(dataset: Any, override: int | None) -> int:
    if override is not None:
        return int(override)
    pred_lens = tuple(getattr(dataset, "pred_lens", (96,)))
    if not pred_lens:
        return 96
    return int(pred_lens[0])


def _adapt_compact_mbproxy_payload(
    payload: dict[str, Any],
    *,
    dataset_name: str,
    pred_len: int | None,
    fixed_seq_len: int | None,
    label_len: int | None,
    batch_size: int | None,
    num_workers: int | None,
    run_group: str,
) -> dict[str, Any]:
    if not _uses_compact_mbproxy_schema(payload):
        return payload

    task_name = str(payload.get("task_name") or SUPPORTED_TASK_NAME)
    if task_name != SUPPORTED_TASK_NAME:
        raise ValueError(
            "Compact multi-backbone candidate JSON adaptation currently supports "
            f"only {SUPPORTED_TASK_NAME}, got '{task_name}'."
        )

    dataset = _require_mbproxy_dataset(dataset_name)
    resolved_pred_len = _default_pred_len(dataset, pred_len)
    resolved_batch_size = (
        int(batch_size) if batch_size is not None else int(getattr(dataset, "default_batch_size", 32))
    )
    adapted_candidates: list[dict[str, Any]] = []

    for index, candidate in enumerate(payload["candidates"]):
        if not isinstance(candidate, dict):
            raise ValueError(f"Candidate at index {index} must be an object.")

        raw_run_args = candidate.get("run_args") or {}
        if not isinstance(raw_run_args, dict):
            raise ValueError(
                f"Candidate '{candidate.get('candidate_id', index)}' has invalid run_args; expected an object."
            )

        backbone = str(candidate.get("backbone") or candidate.get("model") or raw_run_args.get("model") or "").strip()
        if not backbone:
            raise ValueError(f"Candidate at index {index} is missing a backbone/model name.")

        candidate_id = str(
            candidate.get("candidate_id")
            or candidate.get("candidate_name")
            or raw_run_args.get("model_id")
            or f"{backbone}_{index:03d}"
        )
        seq_len = int(
            fixed_seq_len
            if fixed_seq_len is not None
            else raw_run_args.get("seq_len", getattr(dataset, "default_seq_len", 96))
        )
        resolved_label_len = _default_label_len(dataset, seq_len, label_len)
        model_id = f"{run_group}_{dataset.name}_{candidate_id}_sl{seq_len}_pl{resolved_pred_len}"
        results_id = (
            f"{run_group}_{candidate.get('split', 'unsplit')}_"
            f"{dataset.name}_{candidate_id}_pl{resolved_pred_len}"
        )

        base_run_args: dict[str, Any] = {
            "task_name": task_name,
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
            "label_len": resolved_label_len,
            "pred_len": resolved_pred_len,
            "enc_in": dataset.enc_in,
            "dec_in": dataset.enc_in,
            "c_out": dataset.enc_in,
            "des": run_group,
            "itr": 1,
            "checkpoints": "./checkpoints/multi_backbone_proxy/",
            "results_id": results_id,
            "batch_size": resolved_batch_size,
        }
        merged_run_args = {**base_run_args, **raw_run_args}
        merged_run_args.update(
            {
                "task_name": task_name,
                "model": backbone,
                "data": dataset.data,
                "root_path": dataset.root_path,
                "data_path": dataset.data_path,
                "features": dataset.features,
                "target": dataset.target,
                "freq": dataset.freq,
                "seq_len": seq_len,
                "label_len": resolved_label_len,
                "pred_len": resolved_pred_len,
                "enc_in": dataset.enc_in,
                "dec_in": dataset.enc_in,
                "c_out": dataset.enc_in,
                "batch_size": resolved_batch_size,
            }
        )
        if num_workers is not None:
            merged_run_args["num_workers"] = int(num_workers)
        elif dataset.data == "multi_series":
            # Arrow/RDS/TSF sources are lazily loaded and keep small local
            # caches. Avoid duplicating those readers across worker processes.
            merged_run_args["num_workers"] = 0
        merged_run_args.setdefault("model_id", model_id)
        merged_run_args.setdefault("results_id", results_id)
        merged_run_args.setdefault("des", run_group)

        adapted_candidate = dict(candidate)
        adapted_candidate["candidate_id"] = candidate_id
        adapted_candidate.setdefault("candidate_name", candidate_id)
        adapted_candidate["model"] = backbone
        adapted_candidate["dataset_name"] = dataset.name
        adapted_candidate["dataset_family"] = _dataset_family(dataset.name)
        adapted_candidate["hyperparameters"] = dict(candidate.get("hyperparameters") or raw_run_args)
        adapted_candidate["run_args"] = merged_run_args
        adapted_candidates.append(adapted_candidate)

    adapted_payload = dict(payload)
    metadata = dict(adapted_payload.get("metadata") or {})
    metadata.update(
        {
            "source_schema": "multi_backbone_proxy_compact",
            "scoring_dataset": dataset.name,
            "scoring_dataset_family": _dataset_family(dataset.name),
            "scoring_pred_len": resolved_pred_len,
            "scoring_batch_size": resolved_batch_size,
            "scoring_run_group": run_group,
        }
    )
    adapted_payload["metadata"] = metadata
    adapted_payload["candidates"] = adapted_candidates
    return adapted_payload


def _load_candidate_payload(candidate_path: Path) -> dict[str, Any]:
    with candidate_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"Candidate JSON must contain a non-empty 'candidates' list: {candidate_path}")
    return payload


def _default_csv_path(
    candidate_path: Path,
    repo_root: Path | None = None,
    proxy_columns: list[str] | None = None,
    proxy_filename_labels: list[str] | None = None,
    *,
    separate: bool = False,
) -> Path:
    repo_root = repo_root or _repo_root()
    proxy_columns = proxy_columns or list(ALL_PROXY_COLUMNS)
    proxy_filename_labels = proxy_filename_labels or list(proxy_columns)
    filename_suffix = "separate_proxy_scores.csv" if separate else "proxy_scores.csv"
    if proxy_filename_labels == list(ALL_PROXY_COLUMNS):
        filename = f"{candidate_path.stem}_{filename_suffix}"
    else:
        proxy_suffix = "_".join(proxy_filename_labels)
        filename = f"{candidate_path.stem}_{proxy_suffix}_{filename_suffix}"
    return repo_root / "proxy_scores" / filename


def _timestamp_label() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _append_timestamp_to_csv_path(csv_path: Path, timestamp_label: str) -> Path:
    suffix = csv_path.suffix or ".csv"
    stem = csv_path.stem if csv_path.suffix else csv_path.name
    return csv_path.with_name(f"{stem}_{timestamp_label}{suffix}")


def _safe_dataset_filename(dataset_name: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_name).strip("._")
    return safe_name or "dataset"


def _insert_dataset_into_csv_path(
    csv_path: Path,
    dataset_name: str,
    *,
    candidate_stem: str | None = None,
) -> Path:
    """Make one collision-free CSV path per dataset in a multi-dataset run."""
    dataset_label = _safe_dataset_filename(dataset_name)
    suffix = csv_path.suffix or ".csv"
    stem = csv_path.stem if csv_path.suffix else csv_path.name
    if candidate_stem and stem.startswith(candidate_stem):
        stem = f"{candidate_stem}_{dataset_label}{stem[len(candidate_stem):]}"
    else:
        stem = f"{stem}_{dataset_label}"
    return csv_path.with_name(f"{stem}{suffix}")


def _dataset_csv_path(
    *,
    explicit_csv_path: str | None,
    candidate_path: Path,
    repo_root: Path,
    dataset_name: str,
    multi_dataset: bool,
    proxy_columns: list[str],
    proxy_filename_labels: list[str],
    separate: bool,
    timestamp_label: str,
) -> Path:
    if explicit_csv_path:
        requested_path = Path(explicit_csv_path)
        if not requested_path.is_absolute():
            requested_path = (repo_root / requested_path).resolve()
        if multi_dataset and requested_path.suffix.lower() != ".csv":
            base_path = _default_csv_path(
                candidate_path,
                repo_root,
                proxy_columns=proxy_columns,
                proxy_filename_labels=proxy_filename_labels,
                separate=separate,
            )
            requested_path = requested_path / base_path.name
        if multi_dataset:
            requested_path = _insert_dataset_into_csv_path(
                requested_path,
                dataset_name,
                candidate_stem=candidate_path.stem,
            )
    else:
        requested_path = _default_csv_path(
            candidate_path,
            repo_root,
            proxy_columns=proxy_columns,
            proxy_filename_labels=proxy_filename_labels,
            separate=separate,
        )
        if multi_dataset:
            requested_path = _insert_dataset_into_csv_path(
                requested_path,
                dataset_name,
                candidate_stem=candidate_path.stem,
            )
    return _append_timestamp_to_csv_path(requested_path, timestamp_label)


def _set_global_seed(seed: int, deterministic: bool = False) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            torch.backends.cudnn.deterministic = False

    torch.use_deterministic_algorithms(deterministic)


def _proxy_loader_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _seed_proxy_dataloader_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _parse_gpu_ids(raw_gpu_ids: list[str] | None) -> list[int]:
    if raw_gpu_ids is None:
        return []

    parsed_gpu_ids: list[int] = []
    seen_gpu_ids: set[int] = set()
    for raw_value in raw_gpu_ids:
        for token in str(raw_value).split(","):
            token = token.strip()
            if not token:
                continue
            gpu_id = int(token)
            if gpu_id < 0:
                raise ValueError(f"GPU ids must be non-negative integers, got {gpu_id}.")
            if gpu_id in seen_gpu_ids:
                continue
            seen_gpu_ids.add(gpu_id)
            parsed_gpu_ids.append(gpu_id)

    return parsed_gpu_ids


def _build_args(run_args: dict[str, Any], gpu_id: int | None, repo_root: Path | None = None) -> SimpleNamespace:
    repo_root = repo_root or _repo_root()
    defaults = discover_run_argument_defaults(repo_root)
    merged = dict(defaults)
    merged.update(run_args)

    if "model" not in merged or "task_name" not in merged or "data" not in merged:
        raise ValueError("Each candidate run_args must include at least model, task_name, and data.")

    if merged["task_name"] != SUPPORTED_TASK_NAME:
        raise ValueError(
            f"proxy_scorer.py now supports only task_name='{SUPPORTED_TASK_NAME}', "
            f"got '{merged['task_name']}'."
        )

    if gpu_id is not None:
        merged["use_gpu"] = True
        merged["gpu_type"] = "cuda"
        merged["gpu"] = gpu_id
        merged["use_multi_gpu"] = False

    if merged.get("use_gpu") and merged.get("use_multi_gpu"):
        devices = str(merged.get("devices", "")).replace(" ", "")
        device_ids = [int(device_id) for device_id in devices.split(",") if device_id]
        merged["device_ids"] = device_ids
        if device_ids:
            merged["gpu"] = device_ids[0]

    if merged.get("use_gpu") and merged.get("gpu_type") == "cuda" and torch.cuda.is_available():
        merged["device"] = torch.device(f"cuda:{merged.get('gpu', 0)}")
    elif (
        merged.get("use_gpu")
        and merged.get("gpu_type") == "mps"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        merged["device"] = torch.device("mps")
    else:
        # Exp_Basic selects its device from use_gpu rather than args.device.
        # Keep those fields consistent so CPU-only scoring does not attempt to
        # construct a CUDA model when no accelerator is available.
        merged["use_gpu"] = False
        merged["use_multi_gpu"] = False
        merged["device"] = torch.device("cpu")

    return SimpleNamespace(**merged)


def _candidate_run_args_list(
    candidate: dict[str, Any],
    *,
    gpu_id: int | None,
) -> list[dict[str, Any]]:
    return [_prepare_candidate_run_args(candidate, gpu_id=gpu_id)]


def _select_exp_class(task_name: str):
    if task_name != SUPPORTED_TASK_NAME:
        raise ValueError(f"Unsupported task_name: {task_name}. Only {SUPPORTED_TASK_NAME} is supported.")

    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

    return Exp_Long_Term_Forecast


def _set_proxy_stochastic_layers_mode(
    model: nn.Module,
    *,
    batch_norm_mode: str = "eval",
) -> list[tuple[nn.Module, bool]]:
    states: list[tuple[nn.Module, bool]] = []
    for module in model.modules():
        if isinstance(module, BN_TYPES):
            states.append((module, module.training))
            if batch_norm_mode == "train":
                module.train()
            else:
                module.eval()
        elif isinstance(module, DROPOUT_TYPES):
            states.append((module, module.training))
            module.eval()
    return states


def _restore_module_training_states(states: list[tuple[nn.Module, bool]]) -> None:
    for module, was_training in states:
        module.train(was_training)


def _nanmean(values: list[float]) -> float:
    vals = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        vals.append(value)
    if not vals:
        return float("nan")
    return float(torch.tensor(vals, dtype=torch.float64).mean().item())


def _load_existing_rows(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _write_rows(
    csv_path: Path,
    rows: list[dict[str, Any]],
    proxy_columns: list[str],
    meta_columns: list[str],
) -> None:
    proxy_fieldnames = [CSV_PROXY_COLUMN_LABELS.get(column, column) for column in proxy_columns]
    fieldnames = meta_columns + proxy_fieldnames
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            normalized = {key: row.get(key, "") for key in meta_columns}
            for proxy_name, fieldname in zip(proxy_columns, proxy_fieldnames):
                value = row.get(proxy_name, "")
                scale = CSV_PROXY_COLUMN_SCALES.get(proxy_name)
                if scale is not None and value != "":
                    value = f"{float(value) / scale:.9f}"
                normalized[fieldname] = value
            writer.writerow(normalized)


def _candidate_sort_key(candidate_id: str) -> tuple[int, int, str]:
    match = CANDIDATE_SUFFIX_PATTERN.search(candidate_id)
    if match is not None:
        return (0, int(match.group(1)), candidate_id)
    return (1, 0, candidate_id)


def _output_meta_columns(separate: bool) -> list[str]:
    return list(SEPARATE_META_COLUMNS if separate else META_COLUMNS)


def _optional_sort_index(value: Any) -> int:
    if value is None:
        return 0
    try:
        text = str(value).strip()
    except Exception:
        return 0
    if not text:
        return 0
    try:
        return int(text)
    except ValueError:
        return 0


def _row_storage_key(row: dict[str, Any]) -> str:
    explicit_key = str(row.get("_row_key", "")).strip()
    if explicit_key:
        return explicit_key

    candidate_id = str(row.get("candidate_id", "")).strip()
    run_index_text = str(row.get("run_index", "")).strip()
    run_name = str(row.get("run_name", "")).strip()
    batch_index_text = str(row.get("batch_index", "")).strip()
    if not run_index_text and not run_name and not batch_index_text:
        return candidate_id

    parts = [candidate_id]
    if run_index_text:
        parts.append(f"run{_optional_sort_index(run_index_text):06d}")
    elif run_name:
        parts.append(run_name)
    if batch_index_text:
        parts.append(f"batch{_optional_sort_index(batch_index_text):06d}")
    return "::".join(parts)


def _row_sort_key(row: dict[str, Any]) -> tuple[int, int, str, int, int, str]:
    candidate_sort = _candidate_sort_key(str(row.get("candidate_id", "")))
    return (
        candidate_sort[0],
        candidate_sort[1],
        candidate_sort[2],
        _optional_sort_index(row.get("run_index")),
        _optional_sort_index(row.get("batch_index")),
        _row_storage_key(row),
    )


def _run_name_from_run_args(run_args: dict[str, Any], *, fallback_index: int | None = None) -> str:
    for key in ("model_id", "results_id", "des"):
        value = str(run_args.get(key, "")).strip()
        if value:
            return value
    if fallback_index is not None:
        return f"run_{fallback_index}"
    return ""


def _build_failed_row(
    candidate: dict[str, Any],
    *,
    candidate_id: str,
    proxy_columns: list[str],
    error: str,
    num_batches: int,
    run_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_args = run_args or candidate.get("run_args", {})
    split = candidate.get("split", run_args.get("split", ""))
    row = {
        "candidate_id": candidate_id,
        "candidate_name": candidate.get("candidate_name", candidate_id),
        "model": run_args.get("model", candidate.get("model", "")),
        "split": split,
        "task_name": run_args.get("task_name", ""),
        "data": run_args.get("data", ""),
        "dataset_name": candidate.get("dataset_name", ""),
        "dataset_family": candidate.get("dataset_family", ""),
        "num_batches": num_batches,
        "status": "failed",
        "error": error,
    }
    for proxy_name in proxy_columns:
        row.setdefault(proxy_name, float("nan"))
    return row


def _build_separate_rows(
    candidate: dict[str, Any],
    *,
    candidate_id: str,
    summary: dict[str, Any],
    proxy_columns: list[str],
    run_index: int,
    run_name: str,
) -> list[dict[str, Any]]:
    num_batches = int(summary["num_batches"])
    split = candidate.get("split", "")
    rows: list[dict[str, Any]] = []
    row_count = max(1, num_batches)
    for batch_offset in range(row_count):
        batch_index = batch_offset + 1
        row = {
            "_row_key": f"{candidate_id}::run{run_index:06d}::batch{batch_index:06d}",
            "candidate_id": candidate_id,
            "candidate_name": candidate.get("candidate_name", candidate_id),
            "model": summary["model"],
            "split": split,
            "task_name": summary["task_name"],
            "data": summary["data"],
            "dataset_name": candidate.get("dataset_name", ""),
            "dataset_family": candidate.get("dataset_family", ""),
            "run_index": run_index,
            "run_name": run_name,
            "batch_index": batch_index,
            "num_batches": num_batches,
            "status": "success",
            "error": "",
        }
        if "params" in proxy_columns:
            row["params"] = summary["params"] if summary["params"] is not None else float("nan")
        if "l2_norm" in proxy_columns:
            row["l2_norm"] = summary["l2_norm"] if summary["l2_norm"] is not None else float("nan")
        for proxy_name in proxy_columns:
            if proxy_name in DATA_INDEPENDENT_PROXY_COLUMNS:
                continue
            values = summary["proxy_accumulators"].get(proxy_name, [])
            row[proxy_name] = values[batch_offset] if batch_offset < len(values) else float("nan")
        rows.append(row)
    return rows


def _normalize_proxy_selection(raw_values: list[str] | None) -> list[str]:
    if not raw_values:
        return list(ALL_PROXY_COLUMNS)

    tokens: list[str] = []
    for raw_value in raw_values:
        for token in raw_value.split(","):
            token = token.strip()
            if token:
                tokens.append(token)

    if not tokens or any(token.lower() == "all" for token in tokens):
        return list(ALL_PROXY_COLUMNS)

    unknown = [token for token in tokens if token not in ALL_PROXY_COLUMNS]
    if unknown:
        raise ValueError(
            f"Unknown proxy names: {', '.join(sorted(set(unknown)))}. "
            f"Available proxies: {', '.join(ALL_PROXY_COLUMNS)}"
        )

    selected: list[str] = []
    for token in tokens:
        if token not in selected:
            selected.append(token)
    return selected


def _count_total_params(model: nn.Module) -> float:
    return float(sum(p.numel() for p in model.parameters()))


@torch.no_grad()
def _count_weight_l2_norm(model: nn.Module) -> float:
    total_sq = None
    for name, param in model.named_parameters():
        if not param.requires_grad or name.rsplit(".", 1)[-1] != "weight":
            continue
        param_sq = torch.sum(param.detach() * param.detach())
        total_sq = param_sq if total_sq is None else total_sq + param_sq
    if total_sq is None:
        return float("nan")
    return float(torch.sqrt(total_sq).item())


def _extract_activation_tensor(output: Any) -> torch.Tensor | None:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output:
        return _extract_activation_tensor(output[0])
    if isinstance(output, dict):
        for value in output.values():
            tensor = _extract_activation_tensor(value)
            if tensor is not None:
                return tensor
    return None


def _run_model_forward_raw(exp, prepared_batch: dict[str, Any], input_override: torch.Tensor | None = None):
    args = exp.args
    model = exp.model

    batch_x = input_override if input_override is not None else prepared_batch["batch_x"]
    batch_y = prepared_batch["batch_y"]
    batch_x_mark = prepared_batch["batch_x_mark"]
    batch_y_mark = prepared_batch["batch_y_mark"]
    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float()
    dec_inp = torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1).float().to(exp.device)
    return model(batch_x, batch_x_mark, dec_inp, batch_y_mark), {
        "batch_x": batch_x,
        "batch_y": batch_y,
        "batch_x_mark": batch_x_mark,
        "batch_y_mark": batch_y_mark,
    }


def _randomized_train_loader(exp, *, seed: int):
    args = exp.args
    train_data, _ = exp._get_data(flag="train")

    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "drop_last": False,
        "generator": _proxy_loader_generator(seed),
        "worker_init_fn": _seed_proxy_dataloader_worker,
    }

    return DataLoader(train_data, **loader_kwargs)


def _training_batch_cache_key(
    run_args: dict[str, Any],
    *,
    num_batches: int,
    seed: int,
) -> str:
    """Return a stable key for every setting that can change sampled train batches."""
    args = _build_args(run_args, gpu_id=None)
    fields = (
        "task_name",
        "data",
        "root_path",
        "data_path",
        "features",
        "target",
        "freq",
        "embed",
        "seasonal_patterns",
        "seq_len",
        "label_len",
        "pred_len",
        "batch_size",
        "num_workers",
        "augmentation_ratio",
        "long_term_train_sample_limit",
        "candidate_sample_seed",
    )
    key_payload = {field: getattr(args, field, None) for field in fields}
    key_payload.update({"num_batches": int(num_batches), "proxy_batch_seed": int(seed)})
    return json.dumps(key_payload, sort_keys=True, default=str, separators=(",", ":"))


def _cpu_raw_batch(raw_batch: Any) -> tuple[Any, ...]:
    """Detach a DataLoader batch into CPU-owned tensors suitable for reuse."""
    if not isinstance(raw_batch, (tuple, list)):
        raise TypeError(f"Expected a tuple/list DataLoader batch, got {type(raw_batch).__name__}.")
    return tuple(
        None if value is None else value.detach().to(device="cpu").contiguous()
        if torch.is_tensor(value)
        else value
        for value in raw_batch
    )


def _raw_batches_nbytes(raw_batches: list[Any]) -> int:
    return sum(
        int(value.numel() * value.element_size())
        for raw_batch in raw_batches
        for value in raw_batch
        if torch.is_tensor(value)
    )


def _raw_batch_shapes(raw_batch: Any) -> str:
    names = ("x", "y", "x_mark", "y_mark")
    parts = []
    for index, value in enumerate(raw_batch):
        name = names[index] if index < len(names) else f"item{index}"
        shape = "None" if value is None else "x".join(str(dim) for dim in value.shape)
        parts.append(f"{name}={shape}")
    return ", ".join(parts)


def _load_cpu_training_batches(
    run_args: dict[str, Any],
    *,
    num_batches: int,
    seed: int,
    deterministic: bool,
    dataset_name: str,
    config_index: int,
    config_count: int,
) -> list[Any]:
    """Load randomized train batches once without constructing a candidate model."""
    from data_provider.data_factory import data_provider

    started_at = time.perf_counter()
    _set_global_seed(seed, deterministic=deterministic)
    args = _build_args(run_args, gpu_id=None)
    print(
        f"[batch-cache][{dataset_name}][config {config_index}/{config_count}] "
        f"Loading train dataset (data={args.data}, seq_len={args.seq_len}, "
        f"pred_len={args.pred_len}, batch_size={args.batch_size}, "
        f"requested_batches={num_batches}, workers={args.num_workers}) ...",
        flush=True,
    )
    train_data, _ = data_provider(args, "train")
    print(
        f"[batch-cache][{dataset_name}][config {config_index}/{config_count}] "
        f"Dataset ready: train_windows={len(train_data):,}; sampling batches ...",
        flush=True,
    )
    loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        generator=_proxy_loader_generator(seed),
        worker_init_fn=_seed_proxy_dataloader_worker,
    )
    raw_batches: list[Any] = []
    iterator = iter(loader)
    for batch_index in range(1, num_batches + 1):
        batch_started_at = time.perf_counter()
        try:
            raw_batch = next(iterator)
        except StopIteration:
            print(
                f"[batch-cache][{dataset_name}][config {config_index}/{config_count}] "
                f"Train loader exhausted after {len(raw_batches)}/{num_batches} batches.",
                flush=True,
            )
            break
        cached_batch = _cpu_raw_batch(raw_batch)
        raw_batches.append(cached_batch)
        print(
            f"[batch-cache][{dataset_name}][config {config_index}/{config_count}] "
            f"Cached batch {batch_index}/{num_batches} on CPU "
            f"({_raw_batch_shapes(cached_batch)}; "
            f"{time.perf_counter() - batch_started_at:.2f}s)",
            flush=True,
        )
    if not raw_batches:
        raise RuntimeError("Failed to collect any training minibatch for proxy scoring.")
    size_mib = _raw_batches_nbytes(raw_batches) / (1024**2)
    print(
        f"[batch-cache][{dataset_name}][config {config_index}/{config_count}] "
        f"Ready: {len(raw_batches)} batches, {size_mib:.2f} MiB CPU RAM, "
        f"elapsed={time.perf_counter() - started_at:.2f}s. "
        "These batches will be reused for every candidate in this dataset.",
        flush=True,
    )
    return raw_batches


def _prepare_batches(exp, num_batches: int, *, seed: int) -> list[dict[str, Any]]:
    train_loader = _randomized_train_loader(exp, seed=seed)
    iterator = iter(train_loader)
    prepared = []
    for _ in range(num_batches):
        try:
            raw_batch = next(iterator)
        except StopIteration:
            break
        prepared.append(_prepare_single_batch(exp, raw_batch))
    if not prepared:
        raise RuntimeError("Failed to collect any training minibatch for proxy scoring.")
    return prepared


def _prepare_single_batch(exp, raw_batch: Any) -> dict[str, Any]:
    device = exp.device
    batch_x, batch_y, batch_x_mark, batch_y_mark = raw_batch
    return {
        "task_name": SUPPORTED_TASK_NAME,
        "batch_x": batch_x.float().to(device),
        "batch_y": batch_y.float().to(device),
        "batch_x_mark": None if batch_x_mark is None else batch_x_mark.float().to(device),
        "batch_y_mark": None if batch_y_mark is None else batch_y_mark.float().to(device),
    }


def _build_proxy_criterion(exp):
    criterion = exp._select_criterion() if hasattr(exp, "_select_criterion") else None
    if isinstance(criterion, nn.Module):
        criterion = criterion.to(exp.device)
    return criterion


def _forward_task_outputs(exp, prepared_batch: dict[str, Any], input_override: torch.Tensor | None = None):
    args = exp.args
    f_dim = -1 if getattr(args, "features", "M") == "MS" else 0
    raw_outputs, raw_context = _run_model_forward_raw(exp, prepared_batch, input_override=input_override)
    outputs = _extract_activation_tensor(raw_outputs)
    if outputs is None:
        raise RuntimeError(f"Model forward returned no tensor output for task='{SUPPORTED_TASK_NAME}'.")

    batch_x = raw_context["batch_x"]
    batch_y = raw_context["batch_y"]
    outputs = outputs[:, -args.pred_len :, f_dim:]
    return outputs, {
        "target": batch_y[:, -args.pred_len :, f_dim:],
        "primary_input": batch_x,
    }


def _compute_task_loss(exp, outputs: torch.Tensor, context: dict[str, Any], criterion) -> torch.Tensor:
    del exp
    return criterion(outputs, context["target"])


def _compute_task_loss_and_outputs(exp, prepared_batch: dict[str, Any], criterion, input_override=None):
    outputs, context = _forward_task_outputs(exp, prepared_batch, input_override=input_override)
    loss = _compute_task_loss(exp, outputs, context, criterion)
    return loss, outputs, context


CHANNEL_FIRST_ACTIVATION_TYPES = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.GroupNorm,
)


def _activation_feature_axis(module: nn.Module, activation: torch.Tensor) -> int:
    if activation.ndim <= 1:
        return 0
    if isinstance(module, CHANNEL_FIRST_ACTIVATION_TYPES):
        return 1
    return activation.ndim - 1


def _channelwise_fisher_score(module: nn.Module, activation: torch.Tensor) -> torch.Tensor:
    if activation.grad is None:
        return activation.new_zeros(())

    fisher_term = activation.detach() * activation.grad.detach()
    feature_axis = _activation_feature_axis(module, fisher_term)
    reduce_dims = tuple(dim for dim in range(fisher_term.ndim) if dim != feature_axis)
    feature_sums = fisher_term.sum(dim=reduce_dims) if reduce_dims else fisher_term
    return torch.sum(feature_sums * feature_sums)


def _register_activation_hooks(model: nn.Module, activations: list[tuple[nn.Module, torch.Tensor]]):
    def hook(module, __, output):
        activation = _extract_activation_tensor(output)
        if activation is None or not activation.requires_grad:
            return
        activation.retain_grad()
        activations.append((module, activation))

    handles = []
    for module in model.modules():
        if module is model:
            continue
        has_own_params = any(p.requires_grad for p in module.parameters(recurse=False))
        if not has_own_params:
            continue
        if any(True for _ in module.children()):
            continue
        handles.append(module.register_forward_hook(hook))
    return handles


def _single_batch_real_grad_metrics(
    exp,
    prepared_batch: dict[str, Any],
    criterion,
    *,
    batch_norm_mode: str = "eval",
):
    model = exp.model
    stochastic_states = _set_proxy_stochastic_layers_mode(model, batch_norm_mode=batch_norm_mode)
    model.zero_grad(set_to_none=True)
    try:
        loss, _, _ = _compute_task_loss_and_outputs(exp, prepared_batch, criterion)
        loss.backward()

        device = next(model.parameters()).device
        grad_norm_sq = torch.zeros(1, device=device)
        grad_sum = torch.zeros(1, device=device)
        grad_abs_sum = torch.zeros(1, device=device)
        grad_sq_sum = torch.zeros(1, device=device)
        grad_count = 0
        plain = torch.zeros(1, device=device)
        snip = torch.zeros(1, device=device)
        for param in model.parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            grad_sq = grad * grad
            grad_norm_sq += torch.sum(grad_sq)
            grad_sum += torch.sum(grad)
            grad_abs_sum += torch.sum(torch.abs(grad))
            grad_sq_sum += torch.sum(grad_sq)
            grad_count += grad.numel()
            if param.requires_grad:
                plain_term = param.detach() * grad
                plain += torch.sum(plain_term)
                snip += torch.sum(torch.abs(plain_term))
        grad_norm = float(torch.sqrt(grad_norm_sq).item())
        if grad_count == 0:
            zico = float("nan")
        else:
            mean_grad = grad_sum / grad_count
            mean_abs_grad = grad_abs_sum / grad_count
            grad_var = torch.clamp(grad_sq_sum / grad_count - mean_grad * mean_grad, min=0.0)
            grad_std = torch.sqrt(grad_var)
            zico = float((mean_abs_grad / grad_std).item()) if float(grad_std.item()) > 0.0 else float("nan")
        return grad_norm, zico, float(plain.item()), float(snip.item())
    finally:
        model.zero_grad(set_to_none=True)
        _restore_module_training_states(stochastic_states)


def _single_batch_fisher(
    exp,
    prepared_batch: dict[str, Any],
    criterion,
    *,
    batch_norm_mode: str = "eval",
):
    model = exp.model
    stochastic_states = _set_proxy_stochastic_layers_mode(model, batch_norm_mode=batch_norm_mode)
    model.zero_grad(set_to_none=True)
    activations: list[tuple[nn.Module, torch.Tensor]] = []
    handles = _register_activation_hooks(model, activations)
    try:
        loss, _, _ = _compute_task_loss_and_outputs(exp, prepared_batch, criterion)
        loss.backward()
        device = next(model.parameters()).device
        score = torch.zeros(1, device=device)
        for module, activation in activations:
            score += _channelwise_fisher_score(module, activation)
        return float(score.item())
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
        _restore_module_training_states(stochastic_states)


def _compute_loss_grads(exp, prepared_batch: dict[str, Any], criterion, params: list[torch.Tensor]):
    exp.model.zero_grad(set_to_none=True)
    loss, _, _ = _compute_task_loss_and_outputs(exp, prepared_batch, criterion)
    raw_grads = torch.autograd.grad(loss, params, create_graph=False, allow_unused=True)
    grads = []
    for param, grad in zip(params, raw_grads):
        grads.append(torch.zeros_like(param) if grad is None else grad.detach())
    return grads


def _perturb_params(params: list[torch.Tensor], directions: list[torch.Tensor], scale: float) -> None:
    with torch.no_grad():
        for param, direction in zip(params, directions):
            param.add_(direction, alpha=scale)


def _single_batch_grasp(
    exp,
    prepared_batch: dict[str, Any],
    criterion,
    fd_eps: float,
    *,
    batch_norm_mode: str = "eval",
):
    model = exp.model
    stochastic_states = _set_proxy_stochastic_layers_mode(model, batch_norm_mode=batch_norm_mode)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return float("nan")
    try:
        grads = _compute_loss_grads(exp, prepared_batch, criterion, params)
        device = next(model.parameters()).device
        grad_norm_sq = torch.zeros(1, device=device)
        for grad in grads:
            grad_norm_sq += torch.sum(grad * grad)
        grad_norm = torch.sqrt(grad_norm_sq)
        if float(grad_norm.item()) == 0.0:
            return 0.0

        directions = [grad / grad_norm for grad in grads]
        current_offset = 0.0
        try:
            _perturb_params(params, directions, fd_eps)
            current_offset += fd_eps
            grads_pos = _compute_loss_grads(exp, prepared_batch, criterion, params)

            _perturb_params(params, directions, -2.0 * fd_eps)
            current_offset -= 2.0 * fd_eps
            grads_neg = _compute_loss_grads(exp, prepared_batch, criterion, params)
        finally:
            if current_offset != 0.0:
                _perturb_params(params, directions, -current_offset)

        hv_scale = grad_norm / (2.0 * fd_eps)
        score = torch.zeros(1, device=device)
        for param, grad_pos, grad_neg in zip(params, grads_pos, grads_neg):
            hg = (grad_pos - grad_neg) * hv_scale
            score += torch.sum(-param.detach() * hg)
        return float(score.item())
    finally:
        model.zero_grad(set_to_none=True)
        _restore_module_training_states(stochastic_states)


def _single_batch_jacob_cov(exp, prepared_batch: dict[str, Any], *, batch_norm_mode: str = "eval"):
    model = exp.model
    stochastic_states = _set_proxy_stochastic_layers_mode(model, batch_norm_mode=batch_norm_mode)
    model.zero_grad(set_to_none=True)
    try:
        primary_input = prepared_batch["batch_x"].detach().clone().requires_grad_(True)
        outputs, _ = _forward_task_outputs(exp, prepared_batch, input_override=primary_input)
        pseudo_loss = outputs.sum()
        grads = torch.autograd.grad(
            outputs=pseudo_loss,
            inputs=primary_input,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        jacobs = grads.reshape(grads.size(0), -1).detach().cpu().numpy()
        if jacobs.shape[0] < 2:
            return float("nan")
        try:
            corrs = np.corrcoef(jacobs)
            eigvals, _ = np.linalg.eig(corrs)
            k = 1e-5
            score = -np.sum(np.log(eigvals + k) + 1.0 / (eigvals + k))
            score = np.real_if_close(score)
            if np.iscomplexobj(score):
                score = np.real(score)
            return float(score)
        except Exception:
            return float("nan")
    finally:
        model.zero_grad(set_to_none=True)
        _restore_module_training_states(stochastic_states)


def _single_batch_jacob_fro(exp, prepared_batch: dict[str, Any], *, batch_norm_mode: str = "eval"):
    model = exp.model
    stochastic_states = _set_proxy_stochastic_layers_mode(model, batch_norm_mode=batch_norm_mode)
    model.zero_grad(set_to_none=True)
    try:
        primary_input = prepared_batch["batch_x"].detach().clone().requires_grad_(True)
        outputs, _ = _forward_task_outputs(exp, prepared_batch, input_override=primary_input)
        pseudo_loss = outputs.sum()
        grads = torch.autograd.grad(
            outputs=pseudo_loss,
            inputs=primary_input,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        per_sample = []
        for idx in range(grads.size(0)):
            per_sample.append(torch.linalg.norm(grads[idx], ord="fro"))
        if not per_sample:
            return float("nan")
        return float(torch.stack(per_sample).mean().item())
    finally:
        model.zero_grad(set_to_none=True)
        _restore_module_training_states(stochastic_states)


def _ones_like_or_none(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return torch.ones_like(value)
    return None


def _synflow_decoder_input(
    batch_x: torch.Tensor,
    batch_y: torch.Tensor | None,
    args,
) -> torch.Tensor:
    label_len = int(getattr(args, "label_len", 0) or 0)
    pred_len = int(getattr(args, "pred_len", 0) or 0)
    total_len = label_len + pred_len
    if total_len < 1:
        total_len = 1

    if torch.is_tensor(batch_y) and batch_y.ndim >= 3:
        channels = int(batch_y.size(-1))
    else:
        channels = int(batch_x.size(-1))

    return batch_x.new_ones((int(batch_x.size(0)), total_len, channels))


def _run_model_forward_synflow(exp, prepared_batch: dict[str, Any]):
    args = exp.args
    model = exp.model

    batch_x = torch.ones_like(prepared_batch["batch_x"])
    batch_y = prepared_batch.get("batch_y")
    dec_inp = _synflow_decoder_input(batch_x, batch_y, args)
    batch_x_mark = _ones_like_or_none(prepared_batch.get("batch_x_mark"))
    batch_y_mark = _ones_like_or_none(prepared_batch.get("batch_y_mark"))
    return model(batch_x, batch_x_mark, dec_inp, batch_y_mark)


@torch.no_grad()
def _linearize_model(model: nn.Module):
    signs = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            signs[name] = torch.sign(param.data)
            param.data = param.data.abs()
    return signs


@torch.no_grad()
def _nonlinearize_model(model: nn.Module, signs: dict[str, torch.Tensor]):
    for name, param in model.named_parameters():
        if name in signs:
            param.data = param.data * signs[name]


def _single_batch_synflow(exp, prepared_batch: dict[str, Any], *, batch_norm_mode: str = "eval"):
    model = exp.model
    stochastic_states = _set_proxy_stochastic_layers_mode(model, batch_norm_mode=batch_norm_mode)
    model.zero_grad(set_to_none=True)
    signs = _linearize_model(model)
    try:
        raw_outputs = _run_model_forward_synflow(exp, prepared_batch)
        outputs = _extract_activation_tensor(raw_outputs)
        if outputs is None:
            raise RuntimeError(f"Model forward returned no tensor output for SynFlow task='{exp.args.task_name}'.")
        pseudo_loss = outputs.sum()
        pseudo_loss.backward()
        device = next(model.parameters()).device
        score = torch.zeros(1, device=device)
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                score += torch.sum(torch.abs(param.detach() * param.grad.detach()))
        return float(score.item())
    finally:
        _nonlinearize_model(model, signs)
        model.zero_grad(set_to_none=True)
        _restore_module_training_states(stochastic_states)


def _single_batch_flops(exp, prepared_batch: dict[str, Any], *, batch_norm_mode: str = "eval"):
    model = exp.model
    activities = [ProfilerActivity.CPU]
    try:
        first_param = next(model.parameters())
    except StopIteration:
        return float("nan")

    if first_param.device.type == "cuda" and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    stochastic_states = _set_proxy_stochastic_layers_mode(model, batch_norm_mode=batch_norm_mode)
    try:
        with torch.no_grad():
            with profile(activities=activities, with_flops=True, profile_memory=False) as prof:
                outputs, _ = _forward_task_outputs(exp, prepared_batch)
                if torch.is_tensor(outputs):
                    _ = outputs.sum()
        total_flops = sum(event.flops for event in prof.key_averages() if getattr(event, "flops", 0))
        return float(total_flops) if total_flops > 0 else float("nan")
    except Exception:
        return float("nan")
    finally:
        _restore_module_training_states(stochastic_states)


def _score_prepared_batches(
    exp,
    *,
    criterion,
    prepared_batches: list[dict[str, Any]],
    batch_norm_mode: str,
    proxy_columns: list[str],
) -> dict[str, Any]:
    proxy_accumulators: dict[str, list[float]] = {name: [] for name in proxy_columns}
    params_score = _count_total_params(exp.model) if "params" in proxy_columns else None
    l2_norm_score = _count_weight_l2_norm(exp.model) if "l2_norm" in proxy_columns else None

    for prepared_batch in prepared_batches:
        if "flops" in proxy_accumulators:
            proxy_accumulators["flops"].append(
                _single_batch_flops(exp, prepared_batch, batch_norm_mode=batch_norm_mode)
            )

        if (
            "grad_norm" in proxy_accumulators
            or "zico" in proxy_accumulators
            or "plain" in proxy_accumulators
            or "snip" in proxy_accumulators
        ):
            grad_norm, zico, plain, snip = _single_batch_real_grad_metrics(
                exp,
                prepared_batch,
                criterion,
                batch_norm_mode=batch_norm_mode,
            )
            if "grad_norm" in proxy_accumulators:
                proxy_accumulators["grad_norm"].append(grad_norm)
            if "zico" in proxy_accumulators:
                proxy_accumulators["zico"].append(zico)
            if "plain" in proxy_accumulators:
                proxy_accumulators["plain"].append(plain)
            if "snip" in proxy_accumulators:
                proxy_accumulators["snip"].append(snip)

        if "fisher" in proxy_accumulators:
            proxy_accumulators["fisher"].append(
                _single_batch_fisher(exp, prepared_batch, criterion, batch_norm_mode=batch_norm_mode)
            )
        if "grasp" in proxy_accumulators:
            proxy_accumulators["grasp"].append(
                _single_batch_grasp(
                    exp,
                    prepared_batch,
                    criterion,
                    fd_eps=1e-3,
                    batch_norm_mode=batch_norm_mode,
                )
            )
        if "jacob_cov" in proxy_accumulators:
            proxy_accumulators["jacob_cov"].append(
                _single_batch_jacob_cov(exp, prepared_batch, batch_norm_mode=batch_norm_mode)
            )
        if "jacob_fro" in proxy_accumulators:
            proxy_accumulators["jacob_fro"].append(
                _single_batch_jacob_fro(exp, prepared_batch, batch_norm_mode=batch_norm_mode)
            )
        if "synflow" in proxy_accumulators:
            proxy_accumulators["synflow"].append(
                _single_batch_synflow(exp, prepared_batch, batch_norm_mode=batch_norm_mode)
            )

    result = {
        "model": exp.args.model,
        "task_name": exp.args.task_name,
        "data": exp.args.data,
        "num_batches": len(prepared_batches),
        "params": params_score,
        "l2_norm": l2_norm_score,
        "proxy_accumulators": proxy_accumulators,
    }
    return result


def _score_single_run_args(
    run_args: dict[str, Any],
    *,
    gpu_id: int | None,
    num_batches: int,
    seed: int,
    deterministic: bool,
    batch_norm_mode: str,
    proxy_columns: list[str],
    cpu_batch_cache: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    _set_global_seed(seed, deterministic=deterministic)
    args = _build_args(run_args, gpu_id=gpu_id)
    Exp = _select_exp_class(args.task_name)
    exp = Exp(args)
    needs_batches = any(proxy_name not in DATA_INDEPENDENT_PROXY_COLUMNS for proxy_name in proxy_columns)
    criterion = _build_proxy_criterion(exp) if needs_batches else None
    if needs_batches and cpu_batch_cache is not None:
        cache_key = _training_batch_cache_key(run_args, num_batches=num_batches, seed=seed)
        if cache_key not in cpu_batch_cache:
            raise KeyError("No preloaded CPU training batches match this candidate's data settings.")
        batches = [_prepare_single_batch(exp, raw_batch) for raw_batch in cpu_batch_cache[cache_key]]
    else:
        batches = _prepare_batches(exp, num_batches, seed=seed) if needs_batches else []
    try:
        return _score_prepared_batches(
            exp,
            criterion=criterion,
            prepared_batches=batches,
            batch_norm_mode=batch_norm_mode,
            proxy_columns=proxy_columns,
        )
    finally:
        del batches
        del criterion
        del exp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _score_candidate_rows(
    candidate: dict[str, Any],
    *,
    gpu_id: int | None,
    num_batches: int,
    seed: int,
    deterministic: bool,
    batch_norm_mode: str,
    proxy_columns: list[str],
    separate: bool = False,
    cpu_batch_cache: dict[str, list[Any]] | None = None,
) -> list[dict[str, Any]]:
    run_arg_sets = _candidate_run_args_list(
        candidate,
        gpu_id=gpu_id,
    )
    if not run_arg_sets:
        raise ValueError("Candidate is missing run_args.")

    candidate_id = str(candidate.get("candidate_id", run_arg_sets[0].get("model_id", "")))
    candidate_name = str(candidate.get("candidate_name", candidate_id))
    candidate_split = candidate.get("split", run_arg_sets[0].get("split", ""))
    aggregate_proxy_accumulators: dict[str, list[float]] = {name: [] for name in proxy_columns}
    separate_rows: list[dict[str, Any]] = []
    params_score = None
    l2_norm_score = None
    total_num_batches = 0
    run_summary: dict[str, Any] | None = None

    for run_index, run_args in enumerate(run_arg_sets, start=1):
        summary = _score_single_run_args(
            run_args,
            gpu_id=gpu_id,
            num_batches=num_batches,
            seed=seed,
            deterministic=deterministic,
            batch_norm_mode=batch_norm_mode,
            proxy_columns=proxy_columns,
            cpu_batch_cache=cpu_batch_cache,
        )
        if run_summary is None:
            run_summary = summary
        if params_score is None and summary["params"] is not None:
            params_score = summary["params"]
        if l2_norm_score is None and summary["l2_norm"] is not None:
            l2_norm_score = summary["l2_norm"]
        total_num_batches += int(summary["num_batches"])
        if separate:
            separate_rows.extend(
                _build_separate_rows(
                    candidate,
                    candidate_id=candidate_id,
                    summary=summary,
                    proxy_columns=proxy_columns,
                    run_index=run_index,
                    run_name=_run_name_from_run_args(run_args, fallback_index=run_index),
                )
            )
        else:
            for proxy_name, values in summary["proxy_accumulators"].items():
                aggregate_proxy_accumulators[proxy_name].extend(values)

    if run_summary is None:
        raise RuntimeError("Failed to collect any proxy-scoring run summary.")

    if separate:
        return separate_rows

    row = {
        "candidate_id": candidate_id,
        "candidate_name": candidate_name,
        "model": run_summary["model"],
        "split": candidate_split,
        "task_name": run_summary["task_name"],
        "data": run_summary["data"],
        "dataset_name": candidate.get("dataset_name", ""),
        "dataset_family": candidate.get("dataset_family", ""),
        "num_batches": total_num_batches,
        "status": "success",
        "error": "",
    }
    if params_score is not None:
        row["params"] = params_score
    if l2_norm_score is not None:
        row["l2_norm"] = l2_norm_score
    for proxy_name, values in aggregate_proxy_accumulators.items():
        if proxy_name in DATA_INDEPENDENT_PROXY_COLUMNS:
            continue
        row[proxy_name] = _nanmean(values)

    return [row]


def _score_candidates_to_csv(
    candidates: list[dict[str, Any]],
    *,
    csv_path: Path,
    args: argparse.Namespace,
    gpu_ids: list[int],
    output_proxy_columns: list[str],
    output_meta_columns: list[str],
    dataset_index: int,
    dataset_count: int,
    dataset_name: str,
) -> None:
    """Score one adapted dataset payload and persist its candidates incrementally."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _load_existing_rows(csv_path)
    row_by_key: dict[str, dict[str, Any]] = {}
    existing_candidate_ids: set[str] = set()
    for existing_row in rows:
        row_key = _row_storage_key(existing_row)
        candidate_id = str(existing_row.get("candidate_id", ""))
        if row_key:
            row_by_key[row_key] = dict(existing_row)
        if candidate_id:
            existing_candidate_ids.add(candidate_id)

    candidate_jobs: list[tuple[int, dict[str, Any], str]] = []
    for index, candidate in enumerate(candidates, start=1):
        candidate_id = str(
            candidate.get("candidate_id", candidate.get("candidate_name", f"candidate_{index:04d}"))
        )
        if args.skip_existing and candidate_id in existing_candidate_ids:
            print(
                f"[dataset {dataset_index}/{dataset_count}][{index}/{len(candidates)}] "
                f"Skipping {dataset_name}/{candidate_id} (already scored)"
            )
            continue
        candidate_jobs.append((index, candidate, candidate_id))

    needs_batches = any(
        proxy_name not in DATA_INDEPENDENT_PROXY_COLUMNS for proxy_name in output_proxy_columns
    )
    cpu_batch_cache: dict[str, list[Any]] = {}
    if candidate_jobs and needs_batches:
        cache_sources: dict[str, dict[str, Any]] = {}
        for _, candidate, _ in candidate_jobs:
            for run_args in _candidate_run_args_list(candidate, gpu_id=None):
                cache_key = _training_batch_cache_key(
                    run_args,
                    num_batches=args.num_batches,
                    seed=args.seed,
                )
                cache_sources.setdefault(cache_key, run_args)

        print(
            f"[batch-cache][{dataset_name}] Preloading {args.num_batches} train batches "
            f"for {len(cache_sources)} unique data configuration(s) before candidate scoring.",
            flush=True,
        )
        for config_index, (cache_key, run_args) in enumerate(cache_sources.items(), start=1):
            cpu_batch_cache[cache_key] = _load_cpu_training_batches(
                run_args,
                num_batches=args.num_batches,
                seed=args.seed,
                deterministic=args.deterministic,
                dataset_name=dataset_name,
                config_index=config_index,
                config_count=len(cache_sources),
            )
    elif candidate_jobs:
        print(
            f"[batch-cache][{dataset_name}] Skipped: selected proxies are data-independent.",
            flush=True,
        )

    write_lock = threading.Lock()

    def persist_row(row: dict[str, Any]) -> None:
        with write_lock:
            row_by_key[_row_storage_key(row)] = row
            existing_candidate_ids.add(str(row["candidate_id"]))
            ordered_rows = sorted(row_by_key.values(), key=_row_sort_key)
            _write_rows(csv_path, ordered_rows, output_proxy_columns, output_meta_columns)

    def score_one(
        index: int,
        candidate: dict[str, Any],
        candidate_id: str,
        gpu_id: int | None,
    ) -> None:
        prefix = f"[dataset {dataset_index}/{dataset_count}][{index}/{len(candidates)}]"
        gpu_suffix = f"[gpu:{gpu_id}]" if gpu_id is not None else ""
        print(f"{prefix}{gpu_suffix} Scoring {dataset_name}/{candidate_id}")
        try:
            rows_to_persist = _score_candidate_rows(
                candidate,
                gpu_id=gpu_id,
                num_batches=args.num_batches,
                seed=args.seed,
                deterministic=args.deterministic,
                batch_norm_mode=args.proxy_bn_mode,
                proxy_columns=output_proxy_columns,
                separate=args.separate,
                cpu_batch_cache=cpu_batch_cache,
            )
        except Exception as exc:
            rows_to_persist = [
                _build_failed_row(
                    candidate,
                    candidate_id=candidate_id,
                    proxy_columns=output_proxy_columns,
                    error=str(exc),
                    num_batches=args.num_batches,
                )
            ]
            print(f"{prefix}{gpu_suffix} failed: {exc}")

        for row in rows_to_persist:
            persist_row(row)

    if len(gpu_ids) <= 1:
        assigned_gpu_id = gpu_ids[0] if gpu_ids else None
        for index, candidate, candidate_id in candidate_jobs:
            score_one(index, candidate, candidate_id, assigned_gpu_id)
    else:
        print(
            f"Launching {len(gpu_ids)} parallel GPU workers for {len(candidate_jobs)} "
            f"candidates on {dataset_name}: "
            + ", ".join(f"cuda:{gpu_id}" for gpu_id in gpu_ids)
        )
        job_queue: queue.Queue[tuple[int, dict[str, Any], str]] = queue.Queue()
        for job in candidate_jobs:
            job_queue.put(job)

        def worker(gpu_id: int) -> None:
            while True:
                try:
                    index, candidate, candidate_id = job_queue.get_nowait()
                except queue.Empty:
                    return
                score_one(index, candidate, candidate_id, gpu_id)

        threads = [
            threading.Thread(target=worker, name=f"proxy-scorer-gpu-{gpu_id}", args=(gpu_id,))
            for gpu_id in gpu_ids
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    # When there are no jobs (for example after --skip-existing), preserve an
    # existing CSV and otherwise create a header-only output for this dataset.
    if not candidate_jobs and not csv_path.exists():
        _write_rows(csv_path, [], output_proxy_columns, output_meta_columns)
    cached_mib = sum(_raw_batches_nbytes(batches) for batches in cpu_batch_cache.values()) / (1024**2)
    cpu_batch_cache.clear()
    if cached_mib:
        print(
            f"[batch-cache][{dataset_name}] Released {cached_mib:.2f} MiB CPU batch cache.",
            flush=True,
        )
    print(f"Saved {dataset_name} proxy scores to {csv_path}", flush=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score zero-cost proxies for every candidate in a candidates.json file."
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default=None,
        help="Load a candidate JSON by name, accepting candidates/<name>.json or candidates/<name>_candidates.json.",
    )
    parser.add_argument(
        "--candidates-file",
        type=str,
        default=None,
        help="Path to a specific candidates.json file.",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["ECL"],
        help=(
            "One or more scoring datasets/families used when a compact multi-backbone JSON "
            "lacks full run_args. Accepts explicit registry names, comma-separated values, "
            "or the selectors Benchmark (6), Monash (18), TIME (29), and all (53). "
            "Examples: --dataset ECL; --dataset Monash TIME; --dataset Monash,TIME; "
            "--dataset all. Default: ECL."
        ),
    )
    parser.add_argument(
        "--pred-len",
        type=int,
        default=None,
        help=(
            "Prediction length used when adapting compact multi-backbone JSONs. "
            "Default: the selected dataset's first canonical horizon."
        ),
    )
    parser.add_argument(
        "--fixed-seq-len",
        type=int,
        default=None,
        help="Override sampled seq_len when adapting compact multi-backbone JSONs.",
    )
    parser.add_argument(
        "--label-len",
        type=int,
        default=None,
        help="Override label_len when adapting compact multi-backbone JSONs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch_size when adapting compact multi-backbone JSONs.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=(
            "Override DataLoader workers for adapted candidates. Monash/TIME multi-series "
            "datasets default to 0 to keep lazy Arrow/RDS/TSF readers process-local."
        ),
    )
    parser.add_argument(
        "--scoring-run-group",
        type=str,
        default="mbproxy_score",
        help="model_id/results_id prefix used for adapted compact multi-backbone JSONs.",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help=(
            "Where to store proxy score CSVs. For multiple datasets, a .csv path gets a "
            "dataset suffix; a path without .csv is treated as an output directory."
        ),
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=5,
        help="How many minibatches to sample for each proxy. Default output averages them unless --separate is set.",
    )
    parser.add_argument(
        "--separate",
        action="store_true",
        help="Store one CSV row per sampled minibatch instead of averaging proxy values across --num-batches.",
    )
    parser.add_argument(
        "--gpu-id",
        nargs="+",
        default=None,
        help=(
            "One or more physical GPU ids to use. Examples: --gpu-id 0, --gpu-id 0 1 2, "
            "--gpu-id 0,1,2. When multiple ids are given, proxy scoring runs in parallel "
            "with one worker per GPU."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed controlling proxy minibatch shuffling and other proxy-time stochasticity.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Force PyTorch deterministic algorithms and disable cuDNN autotuning. "
            "This improves reproducibility, but can be slower or raise if a proxy uses a non-deterministic CUDA op."
        ),
    )
    parser.add_argument(
        "--proxy-bn-mode",
        type=str,
        default="eval",
        choices=("eval", "train"),
        help=(
            "BatchNorm mode to use during proxy scoring. Default keeps BatchNorm layers in eval mode; "
            "set to 'train' to preserve train-mode BatchNorm behavior during proxy computation."
        ),
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip candidates already present in the CSV.")
    parser.add_argument("--max-candidates", type=int, default=-1, help="Only process the first N candidates.")
    parser.add_argument(
        "--max-datasets",
        type=int,
        default=-1,
        help="Debug option: process only the first N resolved datasets.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Print resolved dataset names/families and exit without scoring.",
    )
    parser.add_argument(
        "--proxies",
        nargs="+",
        default=None,
        help=(
            "Only compute the selected proxies. Use names like 'l2_norm', 'zico', 'plain', 'jacob_cov', "
            "or a comma-separated list. Use 'all' to score every proxy."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    # Redirected stdout (for example nohup > run.out) is block-buffered by
    # default. Line buffering makes dataset-loading and scoring progress
    # visible in the log immediately without requiring `python -u`.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if bool(args.candidates) == bool(args.candidates_file):
        parser.error("Use exactly one of --candidates or --candidates-file.")
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("--num-workers must be non-negative.")

    repo_root = _repo_root()
    invocation_cwd = Path.cwd()
    os.chdir(repo_root)
    _set_global_seed(args.seed, deterministic=args.deterministic)
    gpu_ids = _parse_gpu_ids(args.gpu_id)

    if args.candidates:
        candidate_path = _resolve_candidates_path(args.candidates, repo_root)
    else:
        raw_candidate_path = Path(args.candidates_file)
        if raw_candidate_path.is_absolute():
            candidate_path = raw_candidate_path
        else:
            repo_relative_path = (repo_root / raw_candidate_path).resolve()
            cwd_relative_path = (invocation_cwd / raw_candidate_path).resolve()
            candidate_path = cwd_relative_path if cwd_relative_path.exists() else repo_relative_path
        if not candidate_path.exists():
            parser.error(f"Candidate JSON not found: {candidate_path}")

    base_payload = _load_candidate_payload(candidate_path)
    try:
        dataset_names = _resolve_scoring_dataset_names(args.dataset)
    except ValueError as exc:
        parser.error(str(exc))
    if args.max_datasets > 0:
        dataset_names = dataset_names[: args.max_datasets]
    if args.list_datasets:
        family_counts = {family: 0 for family in DATASET_FAMILY_NAMES}
        for index, dataset_name in enumerate(dataset_names, start=1):
            family = _dataset_family(dataset_name)
            family_counts[family] += 1
            print(f"{index:02d}\t{family}\t{dataset_name}")
        print(f"Resolved {len(dataset_names)} datasets: {family_counts}")
        return 0

    compact_payload = _uses_compact_mbproxy_schema(base_payload)
    if not compact_payload and len(dataset_names) > 1:
        parser.error(
            "Multiple --dataset selectors require a compact multi-backbone candidate JSON. "
            "The supplied candidates already contain full dataset-specific run_args."
        )

    requested_proxy_columns = _normalize_proxy_selection(args.proxies)
    output_proxy_columns = list(requested_proxy_columns)
    proxy_filename_labels = list(requested_proxy_columns)
    output_meta_columns = _output_meta_columns(args.separate)
    timestamp_label = _timestamp_label()
    multi_dataset = len(dataset_names) > 1
    csv_paths: list[Path] = []
    for dataset_index, dataset_name in enumerate(dataset_names, start=1):
        if compact_payload:
            try:
                payload = _adapt_compact_mbproxy_payload(
                    base_payload,
                    dataset_name=dataset_name,
                    pred_len=args.pred_len,
                    fixed_seq_len=args.fixed_seq_len,
                    label_len=args.label_len,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    run_group=args.scoring_run_group,
                )
            except ValueError as exc:
                parser.error(str(exc))
        else:
            payload = base_payload

        candidates = payload["candidates"]
        if args.max_candidates > 0:
            candidates = candidates[: args.max_candidates]
        csv_path = _dataset_csv_path(
            explicit_csv_path=args.csv_path,
            candidate_path=candidate_path,
            repo_root=repo_root,
            dataset_name=dataset_name,
            multi_dataset=multi_dataset,
            proxy_columns=output_proxy_columns,
            proxy_filename_labels=proxy_filename_labels,
            separate=args.separate,
            timestamp_label=timestamp_label,
        )
        family = _dataset_family(dataset_name)
        print(
            f"=== Dataset {dataset_index}/{len(dataset_names)}: "
            f"{dataset_name} ({family}); candidates={len(candidates)} ==="
        )
        _score_candidates_to_csv(
            candidates,
            csv_path=csv_path,
            args=args,
            gpu_ids=gpu_ids,
            output_proxy_columns=output_proxy_columns,
            output_meta_columns=output_meta_columns,
            dataset_index=dataset_index,
            dataset_count=len(dataset_names),
            dataset_name=dataset_name,
        )
        csv_paths.append(csv_path)

    print(f"Completed proxy scoring for {len(dataset_names)} datasets.")
    if len(csv_paths) > 1:
        print(f"CSV output directory: {csv_paths[0].parent}")
    return 0
