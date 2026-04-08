from __future__ import annotations

import argparse
import csv
from datetime import datetime
import inspect
import json
import math
import os
import queue
import random
import re
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader

from benchmarking.candidate_sampler import (
    _candidate_recipe_runs,
    _prepare_candidate_run_args,
    _recipe_adjusted_run_args,
    _requested_uea_subset_names,
    discover_run_argument_defaults,
)


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

META_COLUMNS = [
    "candidate_id",
    "candidate_name",
    "model",
    "task_name",
    "data",
    "num_batches",
    "status",
    "error",
]

SEPARATE_META_COLUMNS = [
    "candidate_id",
    "candidate_name",
    "model",
    "task_name",
    "data",
    "run_index",
    "run_name",
    "batch_index",
    "num_batches",
    "status",
    "error",
]

CANDIDATE_SUFFIX_PATTERN = re.compile(r"_(\d+)$")

SFRD_HEAD_MODULE_TOKENS = ("projection", "classifier", "head", "output_layer", "fc", "flatten")
SFRD_ENCODER_MODULE_TOKENS = ("encoder", "encoders")
SFRD_EMBED_MODULE_TOKENS = ("enc_embedding", "patch_embedding")
SFRD_DECODER_MODULE_TOKENS = ("decoder", "dec_embedding")
SFRD_POST_ENCODER_TOKENS = ("dropout", "act")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _slugify(value: str) -> str:
    return value.replace("\\", "/").split("/")[-1].replace(".json", "").replace("_candidates", "").lower()


def _resolve_candidates_path(candidate_name: str, repo_root: Path | None = None) -> Path:
    repo_root = repo_root or _repo_root()
    candidates_dir = repo_root / "candidates"
    normalized = _slugify(candidate_name)
    candidate_path = candidates_dir / f"{normalized}_candidates.json"
    if not candidate_path.exists():
        raise ValueError(f"Candidate JSON not found for '{candidate_name}'.")
    return candidate_path


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
        merged["device"] = torch.device("cpu")

    return SimpleNamespace(**merged)


def _candidate_run_args_list(
    candidate: dict[str, Any],
    *,
    gpu_id: int | None,
    requested_uea_subset_names: list[str] | None = None,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    repo_root = repo_root or _repo_root()

    if requested_uea_subset_names:
        recipe_selection = _candidate_recipe_runs(
            candidate,
            requested_subset_names=requested_uea_subset_names,
            repo_root=repo_root,
        )
        if recipe_selection is not None:
            _, recipe_runs = recipe_selection
            return [
                _recipe_adjusted_run_args(candidate, recipe_run, gpu_id=gpu_id)
                for recipe_run in recipe_runs
            ]

    return [_prepare_candidate_run_args(candidate, gpu_id=gpu_id)]


def _select_exp_class(task_name: str):
    if task_name == "long_term_forecast":
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

        return Exp_Long_Term_Forecast
    if task_name == "short_term_forecast":
        from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast

        return Exp_Short_Term_Forecast
    if task_name == "imputation":
        from exp.exp_imputation import Exp_Imputation

        return Exp_Imputation
    if task_name == "anomaly_detection":
        from exp.exp_anomaly_detection import Exp_Anomaly_Detection

        return Exp_Anomaly_Detection
    if task_name == "classification":
        from exp.exp_classification import Exp_Classification

        return Exp_Classification
    if task_name == "zero_shot_forecast":
        from exp.exp_zero_shot_forecasting import Exp_Zero_Shot_Forecast

        return Exp_Zero_Shot_Forecast
    raise ValueError(f"Unsupported task_name: {task_name}")


_SHARED_MODEL_DICT = None


def _shared_model_dict(repo_root: Path | None = None):
    global _SHARED_MODEL_DICT
    if _SHARED_MODEL_DICT is not None:
        return _SHARED_MODEL_DICT

    from exp.exp_basic import LazyModelDict

    repo_root = repo_root or _repo_root()
    models_dir = repo_root / "models"
    model_map = {
        path.stem: f"models.{path.stem}"
        for path in models_dir.glob("*.py")
        if path.name != "__init__.py"
    }
    _SHARED_MODEL_DICT = LazyModelDict(model_map)
    return _SHARED_MODEL_DICT


def _build_model_from_args(args, *, repo_root: Path | None = None) -> nn.Module:
    model_class = _shared_model_dict(repo_root=repo_root)[args.model]
    model = model_class(args).float()
    if getattr(args, "use_multi_gpu", False) and getattr(args, "use_gpu", False):
        model = nn.DataParallel(model, device_ids=getattr(args, "device_ids", None))
    return model.to(args.device)


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
    fieldnames = meta_columns + proxy_columns
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            normalized = {key: row.get(key, "") for key in fieldnames}
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
    row = {
        "candidate_id": candidate_id,
        "candidate_name": candidate.get("candidate_name", candidate_id),
        "model": run_args.get("model", candidate.get("model", "")),
        "task_name": run_args.get("task_name", ""),
        "data": run_args.get("data", ""),
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
    rows: list[dict[str, Any]] = []
    for batch_offset in range(num_batches):
        batch_index = batch_offset + 1
        row = {
            "_row_key": f"{candidate_id}::run{run_index:06d}::batch{batch_index:06d}",
            "candidate_id": candidate_id,
            "candidate_name": candidate.get("candidate_name", candidate_id),
            "model": summary["model"],
            "task_name": summary["task_name"],
            "data": summary["data"],
            "run_index": run_index,
            "run_name": run_name,
            "batch_index": batch_index,
            "num_batches": num_batches,
            "status": "success",
            "error": "",
        }
        if "params" in proxy_columns:
            row["params"] = summary["params"] if summary["params"] is not None else float("nan")
        for proxy_name in proxy_columns:
            if proxy_name == "params":
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


def _unwrap_model(model: nn.Module) -> nn.Module:
    wrapped = getattr(model, "module", None)
    if isinstance(wrapped, nn.Module):
        return wrapped
    return model


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


def _extract_activation_tensors(output: Any) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    if torch.is_tensor(output):
        tensors.append(output)
        return tensors
    if isinstance(output, (tuple, list)):
        for item in output:
            tensors.extend(_extract_activation_tensors(item))
        return tensors
    if isinstance(output, dict):
        for value in output.values():
            tensors.extend(_extract_activation_tensors(value))
    return tensors


def _expected_sfrd_time_sizes(exp, prepared_batch: dict[str, Any]) -> list[int]:
    sizes = []
    for key in ("batch_x", "batch_y", "batch_x_mark", "batch_y_mark"):
        value = prepared_batch.get(key)
        if torch.is_tensor(value) and value.ndim >= 2:
            sizes.append(int(value.size(1)))
    for attr_name in ("seq_len", "pred_len", "label_len"):
        value = int(getattr(exp.args, attr_name, 0) or 0)
        if value > 1:
            sizes.append(value)
    return sorted(set(sizes))


def _canonicalize_sfrd_tensor(
    tensor: torch.Tensor,
    *,
    batch_size: int,
    expected_time_sizes: list[int],
    prefer_last_axis: bool,
) -> torch.Tensor | None:
    if not torch.is_tensor(tensor) or tensor.ndim < 3 or tensor.size(0) != batch_size:
        return None

    if tensor.ndim == 3:
        axis_1_exact = tensor.size(1) in expected_time_sizes
        axis_2_exact = tensor.size(2) in expected_time_sizes
        if axis_1_exact != axis_2_exact:
            time_axis = 1 if axis_1_exact else 2
        elif tensor.size(1) > 1:
            time_axis = 1
        elif tensor.size(2) > 1:
            time_axis = 2
        else:
            return None
    else:
        exact_axes = [axis for axis in range(1, tensor.ndim) if tensor.size(axis) in expected_time_sizes]
        if exact_axes:
            time_axis = exact_axes[-1] if prefer_last_axis else exact_axes[0]
        else:
            candidate_axes = [axis for axis in range(1, tensor.ndim) if tensor.size(axis) > 1]
            if not candidate_axes:
                return None
            time_axis = candidate_axes[-1] if prefer_last_axis else candidate_axes[0]

    ordered_axes = [0, time_axis] + [axis for axis in range(1, tensor.ndim) if axis != time_axis]
    canonical = tensor.detach().permute(*ordered_axes).contiguous()
    canonical = canonical.reshape(canonical.size(0), canonical.size(1), -1)
    if canonical.size(1) < 2 or canonical.size(2) < 1:
        return None
    return canonical


def _sfrd_module_layer_rank(module_name: str) -> tuple[int, ...] | None:
    lower_name = module_name.lower()
    if not any(token in lower_name for token in SFRD_ENCODER_MODULE_TOKENS):
        return None

    indices = tuple(int(match) for match in re.findall(r"(?:^|\.)(\d+)(?=\.|$)", module_name))
    if not indices:
        return None
    return indices


def _deepest_encoder_layer_rank(model: nn.Module) -> tuple[int, ...] | None:
    deepest_rank: tuple[int, ...] | None = None
    for module_name, _ in model.named_modules():
        layer_rank = _sfrd_module_layer_rank(module_name)
        if layer_rank is None:
            continue
        if deepest_rank is None or layer_rank > deepest_rank:
            deepest_rank = layer_rank
    return deepest_rank


def _sfrd_module_priority(module_name: str, hook_kind: str) -> int:
    lower_name = module_name.lower()
    priority = 50

    if any(token in lower_name for token in SFRD_ENCODER_MODULE_TOKENS):
        priority += 45
    elif any(token in lower_name for token in SFRD_EMBED_MODULE_TOKENS):
        priority -= 20

    if any(token in lower_name for token in SFRD_DECODER_MODULE_TOKENS):
        priority -= 40

    if any(token in lower_name for token in SFRD_HEAD_MODULE_TOKENS):
        priority += 35 if hook_kind == "pre" else -60

    if any(token in lower_name for token in SFRD_POST_ENCODER_TOKENS):
        priority -= 15

    return priority


def _run_model_forward_raw(exp, prepared_batch: dict[str, Any], input_override: torch.Tensor | None = None):
    args = exp.args
    task = args.task_name
    model = exp.model

    if task in {"long_term_forecast", "zero_shot_forecast"}:
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

    if task == "short_term_forecast":
        batch_x = input_override if input_override is not None else prepared_batch["batch_x"]
        batch_y = prepared_batch["batch_y"]
        batch_y_mark = prepared_batch["batch_y_mark"]
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float()
        dec_inp = torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1).float().to(exp.device)
        return model(batch_x, None, dec_inp, None), {
            "batch_x": batch_x,
            "batch_y": batch_y,
            "batch_y_mark": batch_y_mark,
        }

    if task == "imputation":
        batch_x = input_override if input_override is not None else prepared_batch["batch_x"]
        batch_x_mark = prepared_batch["batch_x_mark"]
        mask = prepared_batch["mask"]
        inp = batch_x.masked_fill(mask == 0, 0)
        return model(inp, batch_x_mark, None, None, mask), {
            "batch_x": batch_x,
            "batch_x_mark": batch_x_mark,
            "mask": mask,
        }

    if task == "anomaly_detection":
        batch_x = input_override if input_override is not None else prepared_batch["batch_x"]
        return model(batch_x, None, None, None), {
            "batch_x": batch_x,
        }

    if task == "classification":
        batch_x = input_override if input_override is not None else prepared_batch["batch_x"]
        padding_mask = prepared_batch["padding_mask"]
        return model(batch_x, padding_mask, None, None), {
            "batch_x": batch_x,
            "padding_mask": padding_mask,
        }

    raise ValueError(f"Unsupported task for proxy scoring: {task}")


def _locate_last_layer_module(
    container: nn.Module,
    *,
    container_name: str,
    stack_attr_names: tuple[str, ...],
) -> tuple[str, nn.Module] | None:
    for attr_name in stack_attr_names:
        layer_stack = getattr(container, attr_name, None)
        if isinstance(layer_stack, nn.ModuleList) and len(layer_stack) > 0:
            return f"{container_name}.{attr_name}.{len(layer_stack) - 1}", layer_stack[-1]
    return None


def _capture_sfrd_representation_from_module(
    exp,
    prepared_batch: dict[str, Any],
    *,
    module: nn.Module,
    module_name: str,
    source_kind: str,
    batch_size: int,
    expected_time_sizes: list[int],
) -> dict[str, Any] | None:
    captured: dict[str, Any] | None = None

    def hook(_module, _inputs, output):
        nonlocal captured
        if captured is not None:
            return
        for tensor in _extract_activation_tensors(output):
            representation = _canonicalize_sfrd_tensor(
                tensor,
                batch_size=batch_size,
                expected_time_sizes=expected_time_sizes,
                prefer_last_axis=False,
            )
            if representation is None:
                continue
            captured = {
                "representation": representation,
                "source_kind": source_kind,
                "module_name": module_name,
                "raw_shape": tuple(int(dim) for dim in tensor.shape),
                "canonical_shape": tuple(int(dim) for dim in representation.shape),
            }
            break

    handle = module.register_forward_hook(hook)
    try:
        _run_model_forward_raw(exp, prepared_batch)
    except Exception:
        return None
    finally:
        handle.remove()

    return captured


def _extract_direct_encoder_method_representation(
    exp,
    prepared_batch: dict[str, Any],
    *,
    batch_size: int,
    expected_time_sizes: list[int],
) -> dict[str, Any] | None:
    encoder_method = getattr(exp.model, "encoder", None)
    if encoder_method is None or isinstance(encoder_method, nn.Module) or not callable(encoder_method):
        return None

    try:
        signature = inspect.signature(encoder_method)
        positional = [
            param
            for param in signature.parameters.values()
            if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
    except (TypeError, ValueError):
        positional = []

    if len(positional) != 1:
        return None

    try:
        output = encoder_method(prepared_batch["batch_x"])
    except Exception:
        return None

    for tensor in _extract_activation_tensors(output):
        representation = _canonicalize_sfrd_tensor(
            tensor,
            batch_size=batch_size,
            expected_time_sizes=expected_time_sizes,
            prefer_last_axis=False,
        )
        if representation is None:
            continue
        return {
            "representation": representation,
            "source_kind": "encoder_method",
            "module_name": "encoder()",
            "raw_shape": tuple(int(dim) for dim in tensor.shape),
            "canonical_shape": tuple(int(dim) for dim in representation.shape),
        }
    return None


def _extract_mambasinglelayer_out_layer_representation(
    exp,
    prepared_batch: dict[str, Any],
    *,
    batch_size: int,
    expected_time_sizes: list[int],
) -> dict[str, Any] | None:
    if getattr(exp.args, "model", None) != "MambaSingleLayer":
        return None

    model = _unwrap_model(exp.model)
    out_layer = getattr(model, "out_layer", None)
    if not isinstance(out_layer, nn.Module):
        return None

    return _capture_sfrd_representation_from_module(
        exp,
        prepared_batch,
        module=out_layer,
        module_name="out_layer",
        source_kind="mambasinglelayer_out_layer",
        batch_size=batch_size,
        expected_time_sizes=expected_time_sizes,
    )


def _extract_decoder_last_layer_representation(
    exp,
    prepared_batch: dict[str, Any],
    *,
    batch_size: int,
    expected_time_sizes: list[int],
) -> dict[str, Any] | None:
    decoder = getattr(exp.model, "decoder", None)
    if not isinstance(decoder, nn.Module):
        return None

    located = _locate_last_layer_module(
        decoder,
        container_name="decoder",
        stack_attr_names=("layers", "decode_layers"),
    )
    if located is None:
        return None

    module_name, last_layer = located
    return _capture_sfrd_representation_from_module(
        exp,
        prepared_batch,
        module=last_layer,
        module_name=module_name,
        source_kind="decoder_last_layer",
        batch_size=batch_size,
        expected_time_sizes=expected_time_sizes,
    )


def _extract_encoder_last_layer_representation(
    exp,
    prepared_batch: dict[str, Any],
    *,
    batch_size: int,
    expected_time_sizes: list[int],
) -> dict[str, Any] | None:
    encoder = getattr(exp.model, "encoder", None)
    if not isinstance(encoder, nn.Module):
        return None

    located = _locate_last_layer_module(
        encoder,
        container_name="encoder",
        stack_attr_names=("attn_layers", "layers", "encode_layers", "encode_blocks"),
    )
    if located is None:
        return None

    module_name, last_layer = located
    return _capture_sfrd_representation_from_module(
        exp,
        prepared_batch,
        module=last_layer,
        module_name=module_name,
        source_kind="encoder_last_layer",
        batch_size=batch_size,
        expected_time_sizes=expected_time_sizes,
    )


def _extract_sfrd_sequence_representation_details(exp, prepared_batch: dict[str, Any]) -> dict[str, Any] | None:
    batch_x = prepared_batch.get("batch_x")
    if not torch.is_tensor(batch_x) or batch_x.ndim != 3:
        return None

    batch_size = int(batch_x.size(0))
    expected_time_sizes = _expected_sfrd_time_sizes(exp, prepared_batch)
    first_representation = _canonicalize_sfrd_tensor(
        batch_x,
        batch_size=batch_size,
        expected_time_sizes=expected_time_sizes,
        prefer_last_axis=False,
    )
    if first_representation is None:
        return None

    first_info = {
        "representation": first_representation,
        "source_kind": "input_signal",
        "module_name": "batch_x",
        "raw_shape": tuple(int(dim) for dim in batch_x.shape),
        "canonical_shape": tuple(int(dim) for dim in first_representation.shape),
    }

    for extractor in (
        _extract_mambasinglelayer_out_layer_representation,
        _extract_decoder_last_layer_representation,
        _extract_encoder_last_layer_representation,
        _extract_direct_encoder_method_representation,
    ):
        second_info = extractor(
            exp,
            prepared_batch,
            batch_size=batch_size,
            expected_time_sizes=expected_time_sizes,
        )
        if second_info is not None:
            return {
                "first": first_info,
                "second": second_info,
                "expected_time_sizes": expected_time_sizes,
            }

    model = _unwrap_model(exp.model)
    candidates: list[tuple[int, tuple[int, ...], int, dict[str, Any]]] = []
    call_order = 0
    deepest_encoder_rank = _deepest_encoder_layer_rank(model)

    def record_candidate(module_name: str, hook_kind: str, tensor: torch.Tensor, prefer_last_axis: bool) -> None:
        nonlocal call_order
        representation = _canonicalize_sfrd_tensor(
            tensor,
            batch_size=batch_size,
            expected_time_sizes=expected_time_sizes,
            prefer_last_axis=prefer_last_axis,
        )
        if representation is None:
            return
        layer_rank = _sfrd_module_layer_rank(module_name)
        selection_rank = layer_rank if layer_rank is not None else (-1,)
        candidates.append(
            (
                _sfrd_module_priority(module_name, hook_kind),
                selection_rank,
                call_order,
                {
                    "representation": representation,
                    "source_kind": "generic_forward_activation",
                    "module_name": module_name,
                    "raw_shape": tuple(int(dim) for dim in tensor.shape),
                    "canonical_shape": tuple(int(dim) for dim in representation.shape),
                },
            )
        )
        call_order += 1

    handles = []
    try:
        for module_name, module in model.named_modules():
            if module is model:
                continue

            module_layer_rank = _sfrd_module_layer_rank(module_name)
            if (
                deepest_encoder_rank is not None
                and module_layer_rank is not None
                and module_layer_rank < deepest_encoder_rank
            ):
                continue

            def forward_hook_factory(name: str):
                def hook(_module, _inputs, output):
                    for tensor in _extract_activation_tensors(output):
                        record_candidate(name, "forward", tensor, prefer_last_axis=False)

                return hook

            handles.append(module.register_forward_hook(forward_hook_factory(module_name)))

        _run_model_forward_raw(exp, prepared_batch)
    except Exception:
        return None
    finally:
        for handle in handles:
            handle.remove()

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return {
        "first": first_info,
        "second": candidates[-1][3],
        "expected_time_sizes": expected_time_sizes,
    }


def _extract_sfrd_sequence_representation(exp, prepared_batch: dict[str, Any]) -> torch.Tensor | None:
    details = _extract_sfrd_sequence_representation_details(exp, prepared_batch)
    if details is None:
        return None
    return details["second"]["representation"]


def _sample_valid_length_from_mask(
    prepared_batch: dict[str, Any],
    sample_index: int,
    max_len: int,
) -> int:
    padding_mask = prepared_batch.get("padding_mask")
    if padding_mask is None or padding_mask.ndim != 2:
        return max_len

    valid_length = int(round(float(padding_mask[sample_index].sum().item())))
    valid_length = max(0, min(valid_length, max_len))
    return valid_length


def _train_flag(task_name: str) -> str:
    return "TRAIN" if task_name == "classification" else "train"


def _randomized_train_loader(exp, *, seed: int):
    args = exp.args
    task_name = args.task_name
    train_data, _ = exp._get_data(flag=_train_flag(task_name))

    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "drop_last": False,
        "generator": _proxy_loader_generator(seed),
        "worker_init_fn": _seed_proxy_dataloader_worker,
    }
    if task_name == "classification":
        from data_provider.uea import collate_fn

        loader_kwargs["collate_fn"] = lambda batch: collate_fn(batch, max_len=args.seq_len)

    return DataLoader(train_data, **loader_kwargs)


def _prepare_batches(exp, num_batches: int, *, seed: int) -> list[dict[str, Any]]:
    iterator = iter(_randomized_train_loader(exp, seed=seed))
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
    args = exp.args
    device = exp.device
    task = args.task_name

    if task in {"long_term_forecast", "short_term_forecast", "zero_shot_forecast"}:
        batch_x, batch_y, batch_x_mark, batch_y_mark = raw_batch
        return {
            "task_name": task,
            "batch_x": batch_x.float().to(device),
            "batch_y": batch_y.float().to(device),
            "batch_x_mark": None if batch_x_mark is None else batch_x_mark.float().to(device),
            "batch_y_mark": None if batch_y_mark is None else batch_y_mark.float().to(device),
        }

    if task == "imputation":
        batch_x, _, batch_x_mark, _ = raw_batch
        batch_x = batch_x.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        mask = torch.rand(batch_x.shape, device=device)
        mask = (mask > args.mask_rate).float()
        return {
            "task_name": task,
            "batch_x": batch_x,
            "batch_x_mark": batch_x_mark,
            "mask": mask,
        }

    if task == "anomaly_detection":
        batch_x, batch_y = raw_batch
        return {
            "task_name": task,
            "batch_x": batch_x.float().to(device),
            "batch_y": batch_y,
        }

    if task == "classification":
        batch_x, label, padding_mask = raw_batch
        return {
            "task_name": task,
            "batch_x": batch_x.float().to(device),
            "label": label.to(device),
            "padding_mask": padding_mask.float().to(device),
        }

    raise ValueError(f"Unsupported task for proxy scoring: {task}")


def _build_proxy_criterion(exp):
    if exp.args.task_name == "short_term_forecast":
        criterion = exp._select_criterion(exp.args.loss)
    else:
        criterion = exp._select_criterion() if hasattr(exp, "_select_criterion") else None
    if isinstance(criterion, nn.Module):
        criterion = criterion.to(exp.device)
    return criterion


def _forward_task_outputs(exp, prepared_batch: dict[str, Any], input_override: torch.Tensor | None = None):
    args = exp.args
    task = args.task_name
    f_dim = -1 if getattr(args, "features", "M") == "MS" else 0
    raw_outputs, raw_context = _run_model_forward_raw(exp, prepared_batch, input_override=input_override)
    outputs = _extract_activation_tensor(raw_outputs)
    if outputs is None:
        raise RuntimeError(f"Model forward returned no tensor output for task='{task}'.")

    if task in {"long_term_forecast", "zero_shot_forecast"}:
        batch_x = raw_context["batch_x"]
        batch_y = raw_context["batch_y"]
        outputs = outputs[:, -args.pred_len :, f_dim:]
        return outputs, {
            "target": batch_y[:, -args.pred_len :, f_dim:],
            "primary_input": batch_x,
        }

    if task == "short_term_forecast":
        batch_x = raw_context["batch_x"]
        batch_y = raw_context["batch_y"]
        batch_y_mark = raw_context["batch_y_mark"]
        outputs = outputs[:, -args.pred_len :, f_dim:]
        return outputs, {
            "target": batch_y[:, -args.pred_len :, f_dim:],
            "batch_y_mark": batch_y_mark[:, -args.pred_len :, f_dim:],
            "batch_x_for_loss": batch_x,
            "primary_input": batch_x,
        }

    if task == "imputation":
        batch_x = raw_context["batch_x"]
        mask = raw_context["mask"]
        outputs = outputs[:, :, f_dim:]
        return outputs, {
            "target": batch_x[:, :, f_dim:],
            "mask": mask[:, :, f_dim:],
            "primary_input": batch_x,
        }

    if task == "anomaly_detection":
        batch_x = raw_context["batch_x"]
        outputs = outputs[:, :, f_dim:]
        return outputs, {
            "target": batch_x[:, :, f_dim:],
            "primary_input": batch_x,
        }

    if task == "classification":
        batch_x = raw_context["batch_x"]
        return outputs, {
            "target": prepared_batch["label"].long().squeeze(-1),
            "primary_input": batch_x,
        }

    raise ValueError(f"Unsupported task for proxy scoring: {task}")


def _compute_task_loss(exp, outputs: torch.Tensor, context: dict[str, Any], criterion) -> torch.Tensor:
    task = exp.args.task_name

    if task in {"long_term_forecast", "zero_shot_forecast", "anomaly_detection"}:
        return criterion(outputs, context["target"])

    if task == "short_term_forecast":
        return criterion(
            context["batch_x_for_loss"],
            exp.args.frequency_map,
            outputs,
            context["target"],
            context["batch_y_mark"],
        )

    if task == "imputation":
        mask = context["mask"]
        target = context["target"]
        if torch.sum(mask == 0).item() == 0:
            return criterion(outputs, target)
        return criterion(outputs[mask == 0], target[mask == 0])

    if task == "classification":
        return criterion(outputs, context["target"])

    raise ValueError(f"Unsupported task for proxy scoring: {task}")


def _compute_task_loss_and_outputs(exp, prepared_batch: dict[str, Any], criterion, input_override=None):
    outputs, context = _forward_task_outputs(exp, prepared_batch, input_override=input_override)
    loss = _compute_task_loss(exp, outputs, context, criterion)
    return loss, outputs, context


def _register_activation_hooks(model: nn.Module, activations: list[torch.Tensor]):
    def hook(_, __, output):
        activation = _extract_activation_tensor(output)
        if activation is None or not activation.requires_grad:
            return
        activation.retain_grad()
        activations.append(activation)

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
        snip = torch.zeros(1, device=device)
        for param in model.parameters():
            if param.grad is None:
                continue
            grad_norm_sq += torch.sum(param.grad.detach() * param.grad.detach())
            if param.requires_grad:
                snip += torch.sum(torch.abs(param.detach() * param.grad.detach()))
        grad_norm = float(torch.sqrt(grad_norm_sq).item())
        return grad_norm, float(snip.item())
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
    activations: list[torch.Tensor] = []
    handles = _register_activation_hooks(model, activations)
    try:
        loss, _, _ = _compute_task_loss_and_outputs(exp, prepared_batch, criterion)
        loss.backward()
        device = next(model.parameters()).device
        score = torch.zeros(1, device=device)
        for activation in activations:
            if activation.grad is None:
                continue
            fisher_term = activation.detach() * activation.grad.detach()
            score += torch.sum(fisher_term * fisher_term)
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


def _extract_sfrd_inputs_and_representation(
    exp,
    prepared_batch: dict[str, Any],
    *,
    batch_norm_mode: str = "eval",
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    model = exp.model
    stochastic_states = _set_proxy_stochastic_layers_mode(model, batch_norm_mode=batch_norm_mode)
    try:
        # DSPBuilder setting: SFRD is defined only for long-term forecasting.
        # d_t: input sequence delta, r_t: model prediction delta.
        if getattr(exp.args, "task_name", None) != "long_term_forecast":
            return None, None

        outputs, context = _forward_task_outputs(exp, prepared_batch)
        primary_input = context.get("primary_input", prepared_batch.get("batch_x"))
        return primary_input, outputs
    finally:
        _restore_module_training_states(stochastic_states)


def _single_batch_sfrd_from_sequences(
    prepared_batch: dict[str, Any],
    primary_input: torch.Tensor | None,
    outputs: torch.Tensor | None,
    eps: float = 1e-8,
):
    if not torch.is_tensor(primary_input) or not torch.is_tensor(outputs):
        return float("nan")

    if primary_input.ndim != 3 or outputs.ndim != 3:
        return float("nan")

    disc_scores = []
    for b in range(primary_input.size(0)):
        input_valid_len = _sample_valid_length_from_mask(prepared_batch, b, primary_input.size(1))
        if input_valid_len < 2:
            continue

        repr_valid_len = outputs.size(1)
        if prepared_batch.get("padding_mask") is not None and primary_input.size(1) > 0:
            repr_valid_len = int(round(input_valid_len * outputs.size(1) / primary_input.size(1)))
            repr_valid_len = max(0, min(repr_valid_len, outputs.size(1)))
        if repr_valid_len < 2:
            continue

        input_seq = primary_input[b, :input_valid_len, :]
        repr_seq = outputs[b, :repr_valid_len, :]

        # Frame-wise change vectors for input/prediction output.
        d_b = input_seq[1:, :] - input_seq[:-1, :]
        r_b = repr_seq[1:, :] - repr_seq[:-1, :]

        d_len = d_b.size(0)
        r_len = r_b.size(0)
        if d_len < 1 or r_len < 1:
            continue

        if d_len != r_len:
            return float("nan")

        if d_b.size(1) != r_b.size(1):
            return float("nan")
        if d_b.size(1) < 1:
            continue

        frame_cos = F.cosine_similarity(d_b, r_b, dim=-1, eps=eps)
        finite_cos = frame_cos[torch.isfinite(frame_cos)]
        if finite_cos.numel() > 0:
            disc_scores.append(finite_cos.mean())
    if not disc_scores:
        return float("nan")
    score = torch.stack(disc_scores).mean()
    return float(torch.clamp(score, min=-1.0, max=1.0).item())


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
        ones_input = torch.ones_like(prepared_batch["batch_x"])
        outputs, _ = _forward_task_outputs(exp, prepared_batch, input_override=ones_input)
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

    for prepared_batch in prepared_batches:
        if "flops" in proxy_accumulators:
            proxy_accumulators["flops"].append(
                _single_batch_flops(exp, prepared_batch, batch_norm_mode=batch_norm_mode)
            )

        if "grad_norm" in proxy_accumulators or "snip" in proxy_accumulators:
            grad_norm, snip = _single_batch_real_grad_metrics(
                exp,
                prepared_batch,
                criterion,
                batch_norm_mode=batch_norm_mode,
            )
            if "grad_norm" in proxy_accumulators:
                proxy_accumulators["grad_norm"].append(grad_norm)
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
        needs_sfrd = "sfrd" in proxy_accumulators
        sfrd_primary_input = None
        sfrd_outputs = None
        if needs_sfrd:
            sfrd_primary_input, sfrd_outputs = _extract_sfrd_inputs_and_representation(
                exp,
                prepared_batch,
                batch_norm_mode=batch_norm_mode,
            )
        if "sfrd" in proxy_accumulators:
            proxy_accumulators["sfrd"].append(
                _single_batch_sfrd_from_sequences(
                    prepared_batch,
                    sfrd_primary_input,
                    sfrd_outputs,
                )
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
        "proxy_accumulators": proxy_accumulators,
    }
    return result


def _prepare_uea_classification_subset_context(
    run_args: dict[str, Any],
    *,
    gpu_id: int | None,
    num_batches: int,
    seed: int,
    deterministic: bool,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    _set_global_seed(seed, deterministic=deterministic)
    args = _build_args(run_args, gpu_id=gpu_id, repo_root=repo_root)
    if args.task_name != "classification" or args.data != "UEA":
        raise ValueError("Subset-first caching is only supported for classification/UEA runs.")

    from data_provider.data_factory import data_dict
    from data_provider.uea import collate_fn

    Data = data_dict["UEA"]
    train_data = Data(args=args, root_path=args.root_path, flag="TRAIN")
    test_data = Data(args=args, root_path=args.root_path, flag="TEST")

    args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
    args.pred_len = 0
    args.enc_in = train_data.feature_df.shape[1]
    args.num_class = len(train_data.class_names)
    args.c_out = args.num_class

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        generator=_proxy_loader_generator(seed),
        worker_init_fn=_seed_proxy_dataloader_worker,
        collate_fn=lambda batch: collate_fn(batch, max_len=args.seq_len),
    )

    loader_exp = SimpleNamespace(args=args, device=args.device)
    prepared_batches: list[dict[str, Any]] = []
    iterator = iter(train_loader)
    for _ in range(num_batches):
        try:
            raw_batch = next(iterator)
        except StopIteration:
            break
        prepared_batches.append(_prepare_single_batch(loader_exp, raw_batch))

    if not prepared_batches:
        raise RuntimeError("Failed to collect any training minibatch for proxy scoring.")

    return {
        "args_template": args,
        "prepared_batches": prepared_batches,
        "criterion": nn.CrossEntropyLoss().to(args.device),
    }


def _score_single_uea_subset_run_with_context(
    run_args: dict[str, Any],
    subset_context: dict[str, Any],
    *,
    gpu_id: int | None,
    batch_norm_mode: str,
    proxy_columns: list[str],
    repo_root: Path | None = None,
) -> dict[str, Any]:
    args = _build_args(run_args, gpu_id=gpu_id, repo_root=repo_root)
    template_args = subset_context["args_template"]
    args.seq_len = template_args.seq_len
    args.pred_len = 0
    args.enc_in = template_args.enc_in
    args.num_class = template_args.num_class
    args.c_out = template_args.c_out
    args.device = template_args.device

    model = _build_model_from_args(args, repo_root=repo_root)
    exp = SimpleNamespace(args=args, model=model, device=args.device)
    try:
        return _score_prepared_batches(
            exp,
            criterion=subset_context["criterion"],
            prepared_batches=subset_context["prepared_batches"],
            batch_norm_mode=batch_norm_mode,
            proxy_columns=proxy_columns,
        )
    finally:
        del exp
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _score_single_run_args(
    run_args: dict[str, Any],
    *,
    gpu_id: int | None,
    num_batches: int,
    seed: int,
    deterministic: bool,
    batch_norm_mode: str,
    proxy_columns: list[str],
) -> dict[str, Any]:
    _set_global_seed(seed, deterministic=deterministic)
    args = _build_args(run_args, gpu_id=gpu_id)
    Exp = _select_exp_class(args.task_name)
    exp = Exp(args)
    criterion = _build_proxy_criterion(exp)
    batches = _prepare_batches(exp, num_batches, seed=seed)
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


def _score_candidates_subset_first(
    candidate_jobs: list[tuple[int, dict[str, Any], str]],
    *,
    total_candidates: int,
    gpu_id: int | None,
    num_batches: int,
    seed: int,
    deterministic: bool,
    batch_norm_mode: str,
    proxy_columns: list[str],
    separate: bool = False,
    requested_uea_subset_names: list[str],
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    repo_root = repo_root or _repo_root()
    if not candidate_jobs:
        return []

    subset_slugs = [_slugify(subset_name) for subset_name in requested_uea_subset_names]
    gpu_suffix = f"[gpu:{gpu_id}]" if gpu_id is not None else ""

    candidate_states: dict[str, dict[str, Any]] = {}
    ordered_candidate_ids: list[str] = []

    for index, candidate, candidate_id in candidate_jobs:
        run_arg_sets = _candidate_run_args_list(
            candidate,
            gpu_id=gpu_id,
            requested_uea_subset_names=requested_uea_subset_names,
            repo_root=repo_root,
        )
        if not run_arg_sets:
            raise ValueError(f"Candidate {candidate_id} is missing subset-specific run_args.")

        run_args_by_subset: dict[str, dict[str, Any]] = {}
        for run_args in run_arg_sets:
            subset_slug = _slugify(str(run_args.get("model_id", "")).strip())
            if subset_slug:
                run_args_by_subset[subset_slug] = run_args

        missing_subset_slugs = [subset_slug for subset_slug in subset_slugs if subset_slug not in run_args_by_subset]
        if missing_subset_slugs:
            raise ValueError(
                f"Candidate {candidate_id} is missing subset run_args for: {', '.join(missing_subset_slugs)}"
            )

        candidate_states[candidate_id] = {
            "index": index,
            "candidate": candidate,
            "run_args_by_subset": run_args_by_subset,
            "proxy_accumulators": {name: [] for name in proxy_columns},
            "separate_rows": [],
            "params": None,
            "num_batches": 0,
            "summary": None,
            "status": "success",
            "error": "",
            "failed_run_args": None,
        }
        ordered_candidate_ids.append(candidate_id)

    for subset_index, subset_name in enumerate(requested_uea_subset_names, start=1):
        subset_slug = subset_slugs[subset_index - 1]
        sample_run_args = candidate_states[ordered_candidate_ids[0]]["run_args_by_subset"][subset_slug]
        print(
            f"[subset:{subset_index}/{len(requested_uea_subset_names)}]{gpu_suffix} "
            f"Loading {subset_name} for {len(candidate_jobs)} candidates"
        )
        try:
            subset_context = _prepare_uea_classification_subset_context(
                sample_run_args,
                gpu_id=gpu_id,
                num_batches=num_batches,
                seed=seed,
                deterministic=deterministic,
                repo_root=repo_root,
            )
        except Exception as exc:
            error_message = f"subset '{subset_name}' load failed: {exc}"
            print(
                f"[subset:{subset_index}/{len(requested_uea_subset_names)}]{gpu_suffix} failed: {error_message}"
            )
            for candidate_id in ordered_candidate_ids:
                state = candidate_states[candidate_id]
                if state["status"] != "success":
                    continue
                state["status"] = "failed"
                state["error"] = error_message
                state["failed_run_args"] = state["run_args_by_subset"][subset_slug]
            break

        try:
            for candidate_id in ordered_candidate_ids:
                state = candidate_states[candidate_id]
                if state["status"] != "success":
                    continue

                run_args = state["run_args_by_subset"][subset_slug]
                index = state["index"]
                print(
                    f"[subset:{subset_index}/{len(requested_uea_subset_names)}]"
                    f"[{index}/{total_candidates}]{gpu_suffix} Scoring {candidate_id}"
                )
                try:
                    summary = _score_single_uea_subset_run_with_context(
                        run_args,
                        subset_context,
                        gpu_id=gpu_id,
                        batch_norm_mode=batch_norm_mode,
                        proxy_columns=proxy_columns,
                        repo_root=repo_root,
                    )
                except Exception as exc:
                    state["status"] = "failed"
                    state["error"] = str(exc)
                    state["failed_run_args"] = run_args
                    print(
                        f"[subset:{subset_index}/{len(requested_uea_subset_names)}]"
                        f"[{index}/{total_candidates}]{gpu_suffix} failed: {exc}"
                    )
                    continue

                if state["summary"] is None:
                    state["summary"] = summary
                if state["params"] is None and summary["params"] is not None:
                    state["params"] = summary["params"]
                state["num_batches"] += int(summary["num_batches"])
                if separate:
                    state["separate_rows"].extend(
                        _build_separate_rows(
                            state["candidate"],
                            candidate_id=candidate_id,
                            summary=summary,
                            proxy_columns=proxy_columns,
                            run_index=subset_index,
                            run_name=_run_name_from_run_args(run_args, fallback_index=subset_index),
                        )
                    )
                else:
                    for proxy_name, values in summary["proxy_accumulators"].items():
                        state["proxy_accumulators"][proxy_name].extend(values)
        finally:
            del subset_context
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    for candidate_id in ordered_candidate_ids:
        state = candidate_states[candidate_id]
        candidate = state["candidate"]
        failed_run_args = state["failed_run_args"] or candidate.get("run_args", {})

        if state["status"] != "success" or state["summary"] is None:
            rows.append(
                _build_failed_row(
                    candidate,
                    candidate_id=candidate_id,
                    proxy_columns=proxy_columns,
                    error=state["error"],
                    num_batches=state["num_batches"],
                    run_args=failed_run_args,
                )
            )
            continue

        if separate:
            rows.extend(state["separate_rows"])
            continue

        summary = state["summary"]
        row = {
            "candidate_id": candidate_id,
            "candidate_name": candidate.get("candidate_name", candidate_id),
            "model": summary["model"],
            "task_name": summary["task_name"],
            "data": summary["data"],
            "num_batches": state["num_batches"],
            "status": "success",
            "error": "",
        }
        if state["params"] is not None:
            row["params"] = state["params"]
        for proxy_name, values in state["proxy_accumulators"].items():
            if proxy_name == "params":
                continue
            row[proxy_name] = _nanmean(values)
        rows.append(row)

    return rows


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
    requested_uea_subset_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    run_arg_sets = _candidate_run_args_list(
        candidate,
        gpu_id=gpu_id,
        requested_uea_subset_names=requested_uea_subset_names,
    )
    if not run_arg_sets:
        raise ValueError("Candidate is missing run_args.")

    candidate_id = str(candidate.get("candidate_id", run_arg_sets[0].get("model_id", "")))
    candidate_name = str(candidate.get("candidate_name", candidate_id))
    aggregate_proxy_accumulators: dict[str, list[float]] = {name: [] for name in proxy_columns}
    separate_rows: list[dict[str, Any]] = []
    params_score = None
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
        )
        if run_summary is None:
            run_summary = summary
        if params_score is None and summary["params"] is not None:
            params_score = summary["params"]
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
        "task_name": run_summary["task_name"],
        "data": run_summary["data"],
        "num_batches": total_num_batches,
        "status": "success",
        "error": "",
    }
    if params_score is not None:
        row["params"] = params_score
    for proxy_name, values in aggregate_proxy_accumulators.items():
        if proxy_name == "params":
            continue
        row[proxy_name] = _nanmean(values)

    return [row]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score zero-cost proxies for every candidate in a candidates.json file."
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default=None,
        help="Load candidates/<name>_candidates.json by name.",
    )
    parser.add_argument(
        "--candidates-file",
        type=str,
        default=None,
        help="Path to a specific candidates.json file.",
    )
    parser.add_argument("--csv-path", type=str, default=None, help="Where to store the proxy score CSV.")
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
        "--proxies",
        nargs="+",
        default=None,
        help=(
            "Only compute the selected proxies. Use names like 'sfrd', 'jacob_cov', "
            "or a comma-separated list. Use 'all' to score every proxy."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if bool(args.candidates) == bool(args.candidates_file):
        parser.error("Use exactly one of --candidates or --candidates-file.")

    repo_root = _repo_root()
    os.chdir(repo_root)
    _set_global_seed(args.seed, deterministic=args.deterministic)
    gpu_ids = _parse_gpu_ids(args.gpu_id)

    if args.candidates:
        candidate_path = _resolve_candidates_path(args.candidates, repo_root)
    else:
        candidate_path = Path(args.candidates_file)
        if not candidate_path.is_absolute():
            candidate_path = (repo_root / candidate_path).resolve()
        if not candidate_path.exists():
            parser.error(f"Candidate JSON not found: {candidate_path}")

    payload = _load_candidate_payload(candidate_path)
    candidates = payload["candidates"]
    requested_uea_subset_names = _requested_uea_subset_names(payload)
    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]

    requested_proxy_columns = _normalize_proxy_selection(args.proxies)
    output_proxy_columns = list(requested_proxy_columns)
    proxy_filename_labels = list(requested_proxy_columns)
    output_meta_columns = _output_meta_columns(args.separate)
    timestamp_label = _timestamp_label()

    csv_path = (
        Path(args.csv_path)
        if args.csv_path
        else _default_csv_path(
            candidate_path,
            repo_root,
            proxy_columns=output_proxy_columns,
            proxy_filename_labels=proxy_filename_labels,
            separate=args.separate,
        )
    )
    if not csv_path.is_absolute():
        csv_path = (repo_root / csv_path).resolve()
    csv_path = _append_timestamp_to_csv_path(csv_path, timestamp_label)
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
        candidate_id = str(candidate.get("candidate_id", candidate.get("candidate_name", f"candidate_{index:04d}")))
        if args.skip_existing and candidate_id in existing_candidate_ids:
            print(f"[{index}/{len(candidates)}] Skipping {candidate_id} (already scored)")
            continue
        candidate_jobs.append((index, candidate, candidate_id))

    write_lock = threading.Lock()

    def persist_row(row: dict[str, Any]) -> None:
        with write_lock:
            row_by_key[_row_storage_key(row)] = row
            existing_candidate_ids.add(str(row["candidate_id"]))
            ordered_rows = sorted(
                row_by_key.values(),
                key=_row_sort_key,
            )
            _write_rows(csv_path, ordered_rows, output_proxy_columns, output_meta_columns)

    def score_one(index: int, candidate: dict[str, Any], candidate_id: str, gpu_id: int | None) -> None:
        prefix = f"[{index}/{len(candidates)}]"
        gpu_suffix = f"[gpu:{gpu_id}]" if gpu_id is not None else ""
        print(f"{prefix}{gpu_suffix} Scoring {candidate_id}")
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
                requested_uea_subset_names=requested_uea_subset_names,
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

    use_subset_first_uea_scoring = bool(requested_uea_subset_names) and len(gpu_ids) <= 1

    if use_subset_first_uea_scoring:
        assigned_gpu_id = gpu_ids[0] if gpu_ids else None
        subset_first_rows = _score_candidates_subset_first(
            candidate_jobs,
            total_candidates=len(candidates),
            gpu_id=assigned_gpu_id,
            num_batches=args.num_batches,
            seed=args.seed,
            deterministic=args.deterministic,
            batch_norm_mode=args.proxy_bn_mode,
            proxy_columns=output_proxy_columns,
            separate=args.separate,
            requested_uea_subset_names=requested_uea_subset_names,
            repo_root=repo_root,
        )
        for row in subset_first_rows:
            persist_row(row)
    elif len(gpu_ids) <= 1:
        assigned_gpu_id = gpu_ids[0] if gpu_ids else None
        for index, candidate, candidate_id in candidate_jobs:
            score_one(index, candidate, candidate_id, assigned_gpu_id)
    else:
        print(
            f"Launching {len(gpu_ids)} parallel GPU workers for {len(candidate_jobs)} candidates: "
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

    print(f"Saved proxy scores to {csv_path}")
    return 0
