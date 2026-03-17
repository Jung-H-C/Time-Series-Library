from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import random
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from benchmarking.candidate_sampler import discover_run_argument_defaults


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

SFRD_Q_SWEEP_VALUES = [round(step * 0.05, 2) for step in range(1, 11)]

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

CANDIDATE_SUFFIX_PATTERN = re.compile(r"_(\d+)$")


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
) -> Path:
    repo_root = repo_root or _repo_root()
    proxy_columns = proxy_columns or list(ALL_PROXY_COLUMNS)
    proxy_filename_labels = proxy_filename_labels or list(proxy_columns)
    if proxy_filename_labels == list(ALL_PROXY_COLUMNS):
        filename = f"{candidate_path.stem}_proxy_scores.csv"
    else:
        proxy_suffix = "_".join(proxy_filename_labels)
        filename = f"{candidate_path.stem}_{proxy_suffix}_proxy_scores.csv"
    return repo_root / "proxy_scores" / filename


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    return SimpleNamespace(**merged)


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


def _set_batch_norm_eval(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, BN_TYPES):
            module.eval()


def _set_proxy_stochastic_layers_eval(model: nn.Module) -> list[tuple[nn.Module, bool]]:
    states: list[tuple[nn.Module, bool]] = []
    for module in model.modules():
        if isinstance(module, BN_TYPES + DROPOUT_TYPES):
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


def _write_rows(csv_path: Path, rows: list[dict[str, Any]], proxy_columns: list[str]) -> None:
    fieldnames = META_COLUMNS + proxy_columns
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


def _sfrd_q_column_name(q_value: float) -> str:
    q_suffix = int(round(q_value * 100))
    return f"sfrd_q{q_suffix:03d}"


def _resolve_proxy_output_config(
    proxy_columns: list[str],
    sfrd_q_sweep: bool,
) -> tuple[list[str], list[str], list[float]]:
    output_columns: list[str] = []
    filename_labels: list[str] = []
    sfrd_q_values: list[float] = []

    for proxy_name in proxy_columns:
        if proxy_name == "sfrd" and sfrd_q_sweep:
            sfrd_q_values = list(SFRD_Q_SWEEP_VALUES)
            output_columns.extend(_sfrd_q_column_name(q_value) for q_value in sfrd_q_values)
            filename_labels.append("sfrd_qsweep")
            continue
        output_columns.append(proxy_name)
        filename_labels.append(proxy_name)

    return output_columns, filename_labels, sfrd_q_values


def _count_total_params(model: nn.Module) -> float:
    return float(sum(p.numel() for p in model.parameters()))


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


def _find_classification_head(model: nn.Module) -> nn.Module | None:
    for attr_name in ("projection", "classifier", "head", "output_layer", "fc"):
        module = getattr(model, attr_name, None)
        if isinstance(module, nn.Module):
            return module
    return None


def _extract_classification_sequence_representation(
    exp,
    prepared_batch: dict[str, Any],
    input_override: torch.Tensor | None = None,
) -> torch.Tensor | None:
    model = exp.model
    head = _find_classification_head(model)
    if head is None:
        return None

    batch_x = input_override if input_override is not None else prepared_batch["batch_x"]
    padding_mask = prepared_batch["padding_mask"]
    captured: dict[str, torch.Tensor] = {}

    def pre_hook(_module, inputs):
        if not inputs:
            return
        tensor = _extract_activation_tensor(inputs[0])
        if tensor is not None:
            captured["head_input"] = tensor

    handle = head.register_forward_pre_hook(pre_hook)
    try:
        _ = model(batch_x, padding_mask, None, None)
    finally:
        handle.remove()

    head_input = captured.get("head_input")
    if head_input is None:
        return None

    if head_input.ndim == 3:
        return head_input

    if head_input.ndim == 2:
        seq_len = int(getattr(exp.args, "seq_len", 0) or 0)
        if seq_len > 0 and head_input.size(1) % seq_len == 0:
            hidden_dim = head_input.size(1) // seq_len
            return head_input.reshape(head_input.size(0), seq_len, hidden_dim)

    return None


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


def _build_decoder_input(exp, prepared_batch: dict[str, Any]) -> torch.Tensor:
    batch_y = prepared_batch["batch_y"]
    dec_inp = torch.zeros_like(batch_y[:, -exp.args.pred_len :, :]).float()
    dec_inp = torch.cat([batch_y[:, : exp.args.label_len, :], dec_inp], dim=1).float().to(exp.device)
    return dec_inp


def _extract_raw_forecast_sequence_representation(exp, prepared_batch: dict[str, Any]) -> torch.Tensor | None:
    forecast_method = getattr(exp.model, "forecast", None)
    if not callable(forecast_method):
        return None

    batch_x = prepared_batch["batch_x"]
    batch_x_mark = prepared_batch.get("batch_x_mark")
    batch_y_mark = prepared_batch.get("batch_y_mark")
    dec_inp = _build_decoder_input(exp, prepared_batch)

    try:
        signature = inspect.signature(forecast_method)
        positional = [
            param
            for param in signature.parameters.values()
            if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        num_positional = len(positional)
    except (TypeError, ValueError):
        num_positional = 4

    try:
        if num_positional <= 1:
            outputs = forecast_method(batch_x)
        elif num_positional == 2:
            outputs = forecast_method(batch_x, batch_x_mark)
        else:
            outputs = forecast_method(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    except Exception:
        return None

    outputs = _extract_activation_tensor(outputs)
    if outputs is None or not torch.is_tensor(outputs):
        return None

    if outputs.ndim != 3:
        return None

    f_dim = -1 if getattr(exp.args, "features", "M") == "MS" else 0
    outputs = outputs[:, :, f_dim:]
    return outputs


def _train_flag(task_name: str) -> str:
    return "TRAIN" if task_name == "classification" else "train"


def _prepare_batches(exp, num_batches: int) -> list[dict[str, Any]]:
    _, train_loader = exp._get_data(flag=_train_flag(exp.args.task_name))
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
    model = exp.model
    task = args.task_name
    f_dim = -1 if getattr(args, "features", "M") == "MS" else 0

    if task in {"long_term_forecast", "zero_shot_forecast"}:
        batch_x = input_override if input_override is not None else prepared_batch["batch_x"]
        batch_y = prepared_batch["batch_y"]
        batch_x_mark = prepared_batch["batch_x_mark"]
        batch_y_mark = prepared_batch["batch_y_mark"]
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float()
        dec_inp = torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1).float().to(exp.device)
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        outputs = _extract_activation_tensor(outputs)
        outputs = outputs[:, -args.pred_len :, f_dim:]
        return outputs, {
            "target": batch_y[:, -args.pred_len :, f_dim:],
            "primary_input": batch_x,
        }

    if task == "short_term_forecast":
        batch_x = input_override if input_override is not None else prepared_batch["batch_x"]
        batch_y = prepared_batch["batch_y"]
        batch_y_mark = prepared_batch["batch_y_mark"]
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float()
        dec_inp = torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1).float().to(exp.device)
        outputs = model(batch_x, None, dec_inp, None)
        outputs = _extract_activation_tensor(outputs)
        outputs = outputs[:, -args.pred_len :, f_dim:]
        return outputs, {
            "target": batch_y[:, -args.pred_len :, f_dim:],
            "batch_y_mark": batch_y_mark[:, -args.pred_len :, f_dim:],
            "batch_x_for_loss": batch_x,
            "primary_input": batch_x,
        }

    if task == "imputation":
        batch_x = input_override if input_override is not None else prepared_batch["batch_x"]
        batch_x_mark = prepared_batch["batch_x_mark"]
        mask = prepared_batch["mask"]
        inp = batch_x.masked_fill(mask == 0, 0)
        outputs = model(inp, batch_x_mark, None, None, mask)
        outputs = _extract_activation_tensor(outputs)
        outputs = outputs[:, :, f_dim:]
        return outputs, {
            "target": batch_x[:, :, f_dim:],
            "mask": mask[:, :, f_dim:],
            "primary_input": batch_x,
        }

    if task == "anomaly_detection":
        batch_x = input_override if input_override is not None else prepared_batch["batch_x"]
        outputs = model(batch_x, None, None, None)
        outputs = _extract_activation_tensor(outputs)
        outputs = outputs[:, :, f_dim:]
        return outputs, {
            "target": batch_x[:, :, f_dim:],
            "primary_input": batch_x,
        }

    if task == "classification":
        batch_x = input_override if input_override is not None else prepared_batch["batch_x"]
        outputs = model(batch_x, prepared_batch["padding_mask"], None, None)
        outputs = _extract_activation_tensor(outputs)
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


def _single_batch_real_grad_metrics(exp, prepared_batch: dict[str, Any], criterion):
    model = exp.model
    stochastic_states = _set_proxy_stochastic_layers_eval(model)
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


def _single_batch_fisher(exp, prepared_batch: dict[str, Any], criterion):
    model = exp.model
    stochastic_states = _set_proxy_stochastic_layers_eval(model)
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


def _single_batch_grasp(exp, prepared_batch: dict[str, Any], criterion, fd_eps: float):
    model = exp.model
    stochastic_states = _set_proxy_stochastic_layers_eval(model)
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


def _single_batch_jacob_cov(exp, prepared_batch: dict[str, Any]):
    model = exp.model
    stochastic_states = _set_proxy_stochastic_layers_eval(model)
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


def _single_batch_jacob_fro(exp, prepared_batch: dict[str, Any]):
    model = exp.model
    stochastic_states = _set_proxy_stochastic_layers_eval(model)
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


def _single_batch_sfrd(exp, prepared_batch: dict[str, Any], q: float, normalize_repr: bool, eps: float = 1e-8):
    model = exp.model
    stochastic_states = _set_proxy_stochastic_layers_eval(model)
    try:
        if exp.args.task_name == "classification":
            primary_input = prepared_batch["batch_x"]
            outputs = _extract_classification_sequence_representation(exp, prepared_batch)
            if outputs is None:
                outputs, context = _forward_task_outputs(exp, prepared_batch)
                primary_input = context["primary_input"]
        elif exp.args.task_name in {"long_term_forecast", "short_term_forecast", "zero_shot_forecast"}:
            primary_input = prepared_batch["batch_x"]
            outputs = _extract_raw_forecast_sequence_representation(exp, prepared_batch)
            if outputs is None:
                outputs, context = _forward_task_outputs(exp, prepared_batch)
                primary_input = context["primary_input"]
        else:
            outputs, context = _forward_task_outputs(exp, prepared_batch)
            primary_input = context["primary_input"]
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
            d_b = torch.norm(input_seq[1:, :] - input_seq[:-1, :], p=2, dim=-1)

            if normalize_repr:
                repr_seq = F.normalize(repr_seq, p=2, dim=-1, eps=eps)
            r_l2_b = torch.norm(repr_seq[1:, :] - repr_seq[:-1, :], p=2, dim=-1)

            d_len = d_b.numel()
            r_len = r_l2_b.numel()
            if d_len < 1 or r_len < 1:
                continue

            if d_len != r_len:
                d_b = F.interpolate(
                    d_b.unsqueeze(0).unsqueeze(0),
                    size=r_len,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
            n = r_l2_b.numel()
            k = max(1, int(n * q))
            if 2 * k > n:
                k = max(1, n // 2)
            top_idx = torch.topk(d_b, k=k, largest=True).indices
            bottom_idx = torch.topk(d_b, k=k, largest=False).indices
            disc = torch.log((r_l2_b[top_idx].mean() + eps) / (r_l2_b[bottom_idx].mean() + eps))
            if torch.isfinite(disc):
                disc_scores.append(disc)
        if not disc_scores:
            return float("nan")
        return float(torch.stack(disc_scores).mean().item())
    finally:
        _restore_module_training_states(stochastic_states)


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


def _single_batch_synflow(exp, prepared_batch: dict[str, Any]):
    model = exp.model
    stochastic_states = _set_proxy_stochastic_layers_eval(model)
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


def _single_batch_flops(exp, prepared_batch: dict[str, Any]):
    model = exp.model
    activities = [ProfilerActivity.CPU]
    try:
        first_param = next(model.parameters())
    except StopIteration:
        return float("nan")

    if first_param.device.type == "cuda" and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    stochastic_states = _set_proxy_stochastic_layers_eval(model)
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


def _score_candidate(
    candidate: dict[str, Any],
    *,
    gpu_id: int | None,
    num_batches: int,
    seed: int,
    proxy_columns: list[str],
    sfrd_q_values: list[float] | None = None,
) -> dict[str, Any]:
    run_args = dict(candidate.get("run_args", {}))
    if not isinstance(run_args, dict) or not run_args:
        raise ValueError("Candidate is missing run_args.")

    args = _build_args(run_args, gpu_id=gpu_id)
    Exp = _select_exp_class(args.task_name)
    exp = Exp(args)
    criterion = _build_proxy_criterion(exp)
    batches = _prepare_batches(exp, num_batches)

    sfrd_q_values = sfrd_q_values or []
    proxy_accumulators: dict[str, list[float]] = {name: [] for name in proxy_columns}
    params_score = _count_total_params(exp.model) if "params" in proxy_columns else None

    for prepared_batch in batches:
        if "flops" in proxy_accumulators:
            proxy_accumulators["flops"].append(_single_batch_flops(exp, prepared_batch))

        if "grad_norm" in proxy_accumulators or "snip" in proxy_accumulators:
            grad_norm, snip = _single_batch_real_grad_metrics(exp, prepared_batch, criterion)
            if "grad_norm" in proxy_accumulators:
                proxy_accumulators["grad_norm"].append(grad_norm)
            if "snip" in proxy_accumulators:
                proxy_accumulators["snip"].append(snip)

        if "fisher" in proxy_accumulators:
            proxy_accumulators["fisher"].append(_single_batch_fisher(exp, prepared_batch, criterion))
        if "grasp" in proxy_accumulators:
            proxy_accumulators["grasp"].append(_single_batch_grasp(exp, prepared_batch, criterion, fd_eps=1e-3))
        if "jacob_cov" in proxy_accumulators:
            proxy_accumulators["jacob_cov"].append(_single_batch_jacob_cov(exp, prepared_batch))
        if "jacob_fro" in proxy_accumulators:
            proxy_accumulators["jacob_fro"].append(_single_batch_jacob_fro(exp, prepared_batch))
        if "sfrd" in proxy_accumulators:
            proxy_accumulators["sfrd"].append(_single_batch_sfrd(exp, prepared_batch, q=0.25, normalize_repr=False))
        for q_value in sfrd_q_values:
            column_name = _sfrd_q_column_name(q_value)
            proxy_accumulators[column_name].append(
                _single_batch_sfrd(exp, prepared_batch, q=q_value, normalize_repr=False)
            )
        if "synflow" in proxy_accumulators:
            proxy_accumulators["synflow"].append(_single_batch_synflow(exp, prepared_batch))

    row = {
        "candidate_id": candidate.get("candidate_id", run_args.get("model_id")),
        "candidate_name": candidate.get("candidate_name", run_args.get("model_id")),
        "model": args.model,
        "task_name": args.task_name,
        "data": args.data,
        "num_batches": len(batches),
        "status": "success",
        "error": "",
    }
    if params_score is not None:
        row["params"] = params_score
    for proxy_name, values in proxy_accumulators.items():
        if proxy_name == "params":
            continue
        row[proxy_name] = _nanmean(values)

    del batches
    del criterion
    del exp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row


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
    parser.add_argument("--num-batches", type=int, default=5, help="How many minibatches to average for each proxy.")
    parser.add_argument("--gpu-id", type=int, default=None, help="Physical GPU id to use.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for deterministic batch sampling.")
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
    parser.add_argument(
        "--sfrd-q-sweep",
        action="store_true",
        help=(
            "When 'sfrd' is selected, compute 10 SFRD variants for q=0.05, 0.10, ..., 0.50 "
            "and store them as separate columns."
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
    _set_global_seed(args.seed)

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
    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]

    requested_proxy_columns = _normalize_proxy_selection(args.proxies)
    output_proxy_columns, proxy_filename_labels, sfrd_q_values = _resolve_proxy_output_config(
        requested_proxy_columns,
        args.sfrd_q_sweep,
    )

    csv_path = (
        Path(args.csv_path)
        if args.csv_path
        else _default_csv_path(
            candidate_path,
            repo_root,
            proxy_columns=output_proxy_columns,
            proxy_filename_labels=proxy_filename_labels,
        )
    )
    if not csv_path.is_absolute():
        csv_path = (repo_root / csv_path).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_existing_rows(csv_path)
    row_by_id = {}
    for existing_row in rows:
        candidate_id = str(existing_row.get("candidate_id", ""))
        if candidate_id:
            row_by_id[candidate_id] = dict(existing_row)

    existing_ids = set(row_by_id)

    for index, candidate in enumerate(candidates, start=1):
        candidate_id = str(candidate.get("candidate_id", candidate.get("candidate_name", f"candidate_{index:04d}")))
        if args.skip_existing and candidate_id in existing_ids:
            print(f"[{index}/{len(candidates)}] Skipping {candidate_id} (already scored)")
            continue

        print(f"[{index}/{len(candidates)}] Scoring {candidate_id}")
        try:
            row = _score_candidate(
                candidate,
                gpu_id=args.gpu_id,
                num_batches=args.num_batches,
                seed=args.seed,
                proxy_columns=output_proxy_columns,
                sfrd_q_values=sfrd_q_values,
            )
        except Exception as exc:
            row = {
                "candidate_id": candidate_id,
                "candidate_name": candidate.get("candidate_name", candidate_id),
                "model": candidate.get("model", ""),
                "task_name": candidate.get("run_args", {}).get("task_name", ""),
                "data": candidate.get("run_args", {}).get("data", ""),
                "num_batches": args.num_batches,
                "status": "failed",
                "error": str(exc),
            }
            for proxy_name in output_proxy_columns:
                row.setdefault(proxy_name, float("nan"))
            print(f"  failed: {exc}")

        row_by_id[str(row["candidate_id"])] = row
        existing_ids.add(str(row["candidate_id"]))
        rows = sorted(
            row_by_id.values(),
            key=lambda row_item: _candidate_sort_key(str(row_item.get("candidate_id", ""))),
        )
        _write_rows(csv_path, rows, output_proxy_columns)

    print(f"Saved proxy scores to {csv_path}")
    return 0
