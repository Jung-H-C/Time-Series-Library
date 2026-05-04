from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from .data_loader import TaskContext, extract_input_sequence
from .model import DSPBuilderMetaModel


@dataclass(frozen=True)
class FixedEvaluationPlan:
    loss_support_indices: tuple[int, ...]
    spearman_support_indices_sets: tuple[tuple[int, ...], ...]
    query_batches: tuple[tuple[int, ...], ...]


def sample_query_indices(num_candidates: int, query_size: int, rng: random.Random) -> list[int]:
    if num_candidates < 2:
        raise ValueError("Each benchmark CSV must contain at least 2 candidate rows.")
    if query_size < 2:
        raise ValueError("query_size must be at least 2 for pairwise ranking.")

    population = list(range(num_candidates))
    if num_candidates >= query_size:
        return rng.sample(population, query_size)
    return [rng.choice(population) for _ in range(query_size)]


def compute_pairwise_loss(
    scores: torch.Tensor,
    metrics: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    pair_index = torch.triu_indices(scores.shape[0], scores.shape[0], offset=1, device=scores.device)
    left = pair_index[0]
    right = pair_index[1]

    score_diff = scores[left] - scores[right]
    metric_diff = metrics[right] - metrics[left]
    target_sign = torch.sign(metric_diff)

    valid_mask = target_sign != 0
    if valid_mask.sum().item() == 0:
        zero_loss = scores.new_zeros(())
        return zero_loss, {"pair_acc": 0.0, "num_pairs": 0, "pair_loss_mean": 0.0}

    score_diff = score_diff[valid_mask]
    target_sign = target_sign[valid_mask]

    pair_losses = torch.nn.functional.softplus(-target_sign * score_diff)
    pair_loss_mean = pair_losses.mean()

    predicted_sign = torch.sign(score_diff)
    correct = (predicted_sign == target_sign).float().mean().item()
    return pair_loss_mean, {
        "pair_acc": correct,
        "num_pairs": int(valid_mask.sum().item()),
        "pair_loss_mean": float(pair_loss_mean.item()),
    }


def compute_dataset_classification_loss(
    dataset_logits: torch.Tensor,
    dataset_class_id: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = dataset_logits.unsqueeze(0)
    target = torch.tensor([dataset_class_id], device=dataset_logits.device, dtype=torch.long)
    cls_loss = torch.nn.functional.cross_entropy(logits, target)
    predicted_class = int(logits.argmax(dim=-1).item())
    return cls_loss, {
        "dataset_acc": 1.0 if predicted_class == dataset_class_id else 0.0,
    }


def compute_proxy_signature_regression_loss(
    predicted_signature: torch.Tensor,
    target_signature: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss = torch.nn.functional.mse_loss(predicted_signature, target_signature)
    pred = torch.nn.functional.normalize(predicted_signature, dim=0)
    target = torch.nn.functional.normalize(target_signature, dim=0)
    cosine = torch.sum(pred * target)
    return loss, {
        "signature_cosine": float(cosine.item()),
    }


def load_support_samples_from_indices(
    task: TaskContext,
    indices: Iterable[int],
    device: torch.device,
) -> list[torch.Tensor]:
    support_samples: list[torch.Tensor] = []
    for index in indices:
        sample = task.train_dataset[int(index)]
        support_samples.append(extract_input_sequence(sample).to(device))
    return support_samples


def sample_support_samples(
    task: TaskContext,
    support_size: int,
    rng: random.Random,
    device: torch.device,
) -> list[torch.Tensor]:
    if support_size <= 0:
        raise ValueError("support_size must be positive.")

    population = list(range(len(task.train_dataset)))
    if not population:
        raise ValueError(f"Train dataset is empty for task: {task.benchmark.display_name}")

    if len(population) >= support_size:
        indices = rng.sample(population, support_size)
    else:
        indices = [rng.choice(population) for _ in range(support_size)]

    return load_support_samples_from_indices(task, indices=indices, device=device)


def format_weight_vector(weight_vector: list[float]) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in weight_vector) + "]"


def write_iteration_log(
    log_dir: Path,
    stage_name: str,
    epoch: int,
    dataset_name: str,
    iteration_index: int,
    stats: dict[str, float | list[float] | str],
) -> None:
    weight_vector = stats["weight_vector"]
    assert isinstance(weight_vector, list)
    signature_suffix = ""
    aux_metric_name = stats.get("aux_metric_name")
    aux_loss_label = "cls_loss"
    if aux_metric_name == "signature_cosine":
        aux_loss_label = "reg_loss"
        signature_suffix = f" signature_cosine={float(stats['signature_cosine']):.6f}"
    log_line = (
        f"[{stage_name.upper()}] "
        f"epoch={epoch:03d} "
        f"dataset={dataset_name} "
        f"iteration={iteration_index:03d} "
        f"loss={float(stats['loss']):.6f} "
        f"pair_acc={float(stats['pair_acc']):.4f} "
        f"pair_loss_mean={float(stats['pair_loss_mean']):.6f} "
        f"{aux_loss_label}={float(stats['cls_loss']):.6f} "
        f"dataset_acc={float(stats['dataset_acc']):.4f} "
        f"num_pairs={int(float(stats['num_pairs']))} "
        f"weight_norm={float(stats['weight_norm']):.6f} "
        f"weight_vector={format_weight_vector(weight_vector)}"
        f"{signature_suffix}"
    )
    dataset_log_path = log_dir / f"{dataset_name}.txt"
    with dataset_log_path.open("a", encoding="utf-8") as handle:
        handle.write(log_line + "\n")


def run_task_iteration(
    model: DSPBuilderMetaModel,
    task: TaskContext,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    rng: random.Random,
    support_size: int,
    query_size: int,
    support_samples: list[torch.Tensor] | None = None,
    support_indices: Iterable[int] | None = None,
    query_indices: Iterable[int] | None = None,
    cls_loss_weight: float = 0.1,
    use_proxy_signature_regression: bool = False,
) -> dict[str, float | list[float] | str]:
    if support_samples is not None:
        prepared_support_samples = support_samples
    elif support_indices is None:
        prepared_support_samples = sample_support_samples(
            task,
            support_size=support_size,
            rng=rng,
            device=device,
        )
    else:
        prepared_support_samples = load_support_samples_from_indices(task, indices=support_indices, device=device)

    if query_indices is None:
        selected_query_indices = sample_query_indices(
            task.benchmark.num_candidates,
            query_size=query_size,
            rng=rng,
        )
    else:
        selected_query_indices = [int(index) for index in query_indices]
        if len(selected_query_indices) < 2:
            raise ValueError("query_indices must contain at least 2 candidate indices.")

    query_proxies = task.benchmark.proxies[selected_query_indices].to(device)
    query_metrics = task.benchmark.metrics[selected_query_indices].to(device)

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    if optimizer is not None and not use_proxy_signature_regression and task.dataset_class_id is None:
        raise ValueError(f"Training task is missing dataset_class_id: {task.benchmark.display_name}")

    weight_vector, _task_embedding, dataset_logits, predicted_signature = model(prepared_support_samples)
    query_scores = torch.matmul(query_proxies, weight_vector)
    pair_loss_mean, pair_stats = compute_pairwise_loss(query_scores, query_metrics)
    cls_loss = query_scores.new_zeros(())
    dataset_acc = 0.0
    signature_cosine = 0.0
    aux_metric_name = "dataset_acc"
    total_loss = pair_loss_mean

    if optimizer is not None:
        if use_proxy_signature_regression:
            if task.benchmark.proxy_signature is None:
                raise ValueError(
                    f"Training task is missing proxy_signature: {task.benchmark.display_name}"
                )
            cls_loss, cls_stats = compute_proxy_signature_regression_loss(
                predicted_signature,
                task.benchmark.proxy_signature.to(device),
            )
            signature_cosine = float(cls_stats["signature_cosine"])
            aux_metric_name = "signature_cosine"
        else:
            assert task.dataset_class_id is not None
            cls_loss, cls_stats = compute_dataset_classification_loss(dataset_logits, task.dataset_class_id)
            dataset_acc = float(cls_stats["dataset_acc"])
        total_loss = pair_loss_mean + (cls_loss_weight * cls_loss)

    if optimizer is not None:
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return {
        "loss": float(total_loss.detach().cpu().item()),
        "pair_acc": float(pair_stats["pair_acc"]),
        "num_pairs": float(pair_stats["num_pairs"]),
        "pair_loss_mean": float(pair_stats["pair_loss_mean"]),
        "cls_loss": float(cls_loss.detach().cpu().item()),
        "dataset_acc": dataset_acc,
        "signature_cosine": signature_cosine,
        "aux_metric_name": aux_metric_name,
        "weight_norm": float(weight_vector.detach().norm().cpu().item()),
        "weight_vector": [float(value) for value in weight_vector.detach().cpu().tolist()],
    }


def run_split_epoch(
    model: DSPBuilderMetaModel,
    tasks: list[TaskContext],
    device: torch.device,
    rng: random.Random,
    iterations_per_dataset: int,
    support_size: int,
    query_size: int,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    stage_name: str,
    log_dir: Path,
    fixed_plans: dict[str, FixedEvaluationPlan] | None = None,
    cls_loss_weight: float = 0.1,
    use_proxy_signature_regression: bool = False,
) -> dict[str, float]:
    aggregate = {
        "loss_sum": 0.0,
        "pair_acc_sum": 0.0,
        "pair_loss_mean_sum": 0.0,
        "cls_loss_sum": 0.0,
        "dataset_acc_sum": 0.0,
        "signature_cosine_sum": 0.0,
        "weight_norm_sum": 0.0,
        "steps": 0,
        "pairs": 0.0,
    }

    is_training = optimizer is not None
    model.train(is_training)

    with torch.set_grad_enabled(is_training):
        for task in tasks:
            task_plan = fixed_plans.get(task.benchmark.key) if fixed_plans is not None else None
            effective_iterations = (
                len(task_plan.query_batches) if task_plan is not None else iterations_per_dataset
            )
            if task_plan is not None:
                shared_support_samples = load_support_samples_from_indices(
                    task,
                    indices=task_plan.loss_support_indices,
                    device=device,
                )
            elif is_training:
                shared_support_samples = sample_support_samples(
                    task,
                    support_size=support_size,
                    rng=rng,
                    device=device,
                )
            else:
                shared_support_samples = None

            task_loss_sum = 0.0
            task_cls_loss_sum = 0.0
            task_signature_cosine_sum = 0.0
            for iteration_index in range(1, effective_iterations + 1):
                stats = run_task_iteration(
                    model=model,
                    task=task,
                    optimizer=optimizer,
                    device=device,
                    rng=rng,
                    support_size=support_size,
                    query_size=query_size,
                    support_samples=shared_support_samples,
                    support_indices=task_plan.loss_support_indices if task_plan is not None else None,
                    query_indices=task_plan.query_batches[iteration_index - 1] if task_plan is not None else None,
                    cls_loss_weight=cls_loss_weight,
                    use_proxy_signature_regression=use_proxy_signature_regression,
                )
                write_iteration_log(
                    log_dir=log_dir,
                    stage_name=stage_name,
                    epoch=epoch,
                    dataset_name=task.benchmark.display_name,
                    iteration_index=iteration_index,
                    stats=stats,
                )
                task_loss_sum += float(stats["loss"])
                task_cls_loss_sum += float(stats["cls_loss"])
                task_signature_cosine_sum += float(stats["signature_cosine"])
                aggregate["loss_sum"] += float(stats["loss"])
                aggregate["pair_acc_sum"] += float(stats["pair_acc"])
                aggregate["pair_loss_mean_sum"] += float(stats["pair_loss_mean"])
                aggregate["cls_loss_sum"] += float(stats["cls_loss"])
                aggregate["dataset_acc_sum"] += float(stats["dataset_acc"])
                aggregate["signature_cosine_sum"] += float(stats["signature_cosine"])
                aggregate["weight_norm_sum"] += float(stats["weight_norm"])
                aggregate["steps"] += 1
                aggregate["pairs"] += float(stats["num_pairs"])
            if stage_name == "train":
                task_avg_loss = task_loss_sum / max(effective_iterations, 1)
                task_avg_cls_loss = task_cls_loss_sum / max(effective_iterations, 1)
                aux_loss_label = "avg_reg_loss" if use_proxy_signature_regression else "avg_cls_loss"
                print_line = (
                    f"[TRAIN] epoch={epoch:03d} "
                    f"dataset={task.benchmark.display_name} "
                    f"avg_loss_over_{effective_iterations}_iters={task_avg_loss:.6f} "
                    f"{aux_loss_label}_over_{effective_iterations}_iters={task_avg_cls_loss:.6f}"
                )
                if use_proxy_signature_regression:
                    task_avg_signature_cosine = task_signature_cosine_sum / max(effective_iterations, 1)
                    print_line += f" avg_signature_cosine_so_far={task_avg_signature_cosine:.6f}"
                print(print_line, flush=True)

    if aggregate["steps"] == 0:
        return {
            "loss": 0.0,
            "pair_acc": 0.0,
            "pair_loss_mean": 0.0,
            "cls_loss": 0.0,
            "dataset_acc": 0.0,
            "signature_cosine": 0.0,
            "weight_norm": 0.0,
            "num_steps": 0.0,
            "num_pairs": 0.0,
        }

    return {
        "loss": aggregate["loss_sum"] / aggregate["steps"],
        "pair_acc": aggregate["pair_acc_sum"] / aggregate["steps"],
        "pair_loss_mean": aggregate["pair_loss_mean_sum"] / aggregate["steps"],
        "cls_loss": aggregate["cls_loss_sum"] / aggregate["steps"],
        "dataset_acc": aggregate["dataset_acc_sum"] / aggregate["steps"],
        "signature_cosine": aggregate["signature_cosine_sum"] / aggregate["steps"],
        "weight_norm": aggregate["weight_norm_sum"] / aggregate["steps"],
        "num_steps": float(aggregate["steps"]),
        "num_pairs": aggregate["pairs"],
    }


def run_supervised_train_epoch(
    model: DSPBuilderMetaModel,
    tasks: list[TaskContext],
    device: torch.device,
    rng: random.Random,
    iterations_per_epoch: int,
    batch_size: int,
    support_size: int,
    query_size: int,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    log_dir: Path,
    cls_loss_weight: float = 0.1,
    use_proxy_signature_regression: bool = False,
) -> dict[str, float]:
    if not tasks:
        raise ValueError("tasks must contain at least one training dataset.")
    if iterations_per_epoch <= 0:
        raise ValueError("iterations_per_epoch must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not use_proxy_signature_regression:
        missing_class_ids = [
            task.benchmark.display_name
            for task in tasks
            if task.dataset_class_id is None
        ]
        if missing_class_ids:
            raise ValueError(
                "Training tasks are missing dataset_class_id: "
                + ", ".join(missing_class_ids)
            )

    aggregate = {
        "loss_sum": 0.0,
        "pair_acc_sum": 0.0,
        "pair_loss_mean_sum": 0.0,
        "cls_loss_sum": 0.0,
        "dataset_acc_sum": 0.0,
        "signature_cosine_sum": 0.0,
        "weight_norm_sum": 0.0,
        "task_losses": 0,
        "pairs": 0.0,
    }

    model.train(True)
    progress_interval = max(1, iterations_per_epoch // 10)

    for iteration_index in range(1, iterations_per_epoch + 1):
        optimizer.zero_grad(set_to_none=True)
        batch_losses: list[torch.Tensor] = []
        batch_stats: list[tuple[TaskContext, dict[str, float | list[float] | str]]] = []

        for _ in range(batch_size):
            task = rng.choice(tasks)
            support_samples = sample_support_samples(
                task,
                support_size=support_size,
                rng=rng,
                device=device,
            )
            query_indices = sample_query_indices(
                task.benchmark.num_candidates,
                query_size=query_size,
                rng=rng,
            )
            query_proxies = task.benchmark.proxies[query_indices].to(device)
            query_metrics = task.benchmark.metrics[query_indices].to(device)

            weight_vector, _task_embedding, dataset_logits, predicted_signature = model(support_samples)
            query_scores = torch.matmul(query_proxies, weight_vector)
            pair_loss_mean, pair_stats = compute_pairwise_loss(query_scores, query_metrics)

            cls_loss = query_scores.new_zeros(())
            dataset_acc = 0.0
            signature_cosine = 0.0
            aux_metric_name = "dataset_acc"
            if use_proxy_signature_regression:
                if task.benchmark.proxy_signature is None:
                    raise ValueError(
                        f"Training task is missing proxy_signature: {task.benchmark.display_name}"
                    )
                cls_loss, cls_stats = compute_proxy_signature_regression_loss(
                    predicted_signature,
                    task.benchmark.proxy_signature.to(device),
                )
                signature_cosine = float(cls_stats["signature_cosine"])
                aux_metric_name = "signature_cosine"
            else:
                assert task.dataset_class_id is not None
                cls_loss, cls_stats = compute_dataset_classification_loss(
                    dataset_logits,
                    task.dataset_class_id,
                )
                dataset_acc = float(cls_stats["dataset_acc"])

            task_loss = pair_loss_mean + (cls_loss_weight * cls_loss)
            batch_losses.append(task_loss)
            stats: dict[str, float | list[float] | str] = {
                "loss": float(task_loss.detach().cpu().item()),
                "pair_acc": float(pair_stats["pair_acc"]),
                "num_pairs": float(pair_stats["num_pairs"]),
                "pair_loss_mean": float(pair_stats["pair_loss_mean"]),
                "cls_loss": float(cls_loss.detach().cpu().item()),
                "dataset_acc": dataset_acc,
                "signature_cosine": signature_cosine,
                "aux_metric_name": aux_metric_name,
                "weight_norm": float(weight_vector.detach().norm().cpu().item()),
                "weight_vector": [float(value) for value in weight_vector.detach().cpu().tolist()],
            }
            batch_stats.append((task, stats))

        batch_loss = torch.stack(batch_losses).mean()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for task, stats in batch_stats:
            write_iteration_log(
                log_dir=log_dir,
                stage_name="train",
                epoch=epoch,
                dataset_name=task.benchmark.display_name,
                iteration_index=iteration_index,
                stats=stats,
            )
            aggregate["loss_sum"] += float(stats["loss"])
            aggregate["pair_acc_sum"] += float(stats["pair_acc"])
            aggregate["pair_loss_mean_sum"] += float(stats["pair_loss_mean"])
            aggregate["cls_loss_sum"] += float(stats["cls_loss"])
            aggregate["dataset_acc_sum"] += float(stats["dataset_acc"])
            aggregate["signature_cosine_sum"] += float(stats["signature_cosine"])
            aggregate["weight_norm_sum"] += float(stats["weight_norm"])
            aggregate["task_losses"] += 1
            aggregate["pairs"] += float(stats["num_pairs"])

        if iteration_index % progress_interval == 0 or iteration_index == iterations_per_epoch:
            completed_task_losses = max(aggregate["task_losses"], 1)
            aux_loss_label = "avg_reg_loss" if use_proxy_signature_regression else "avg_cls_loss"
            print_line = (
                f"[TRAIN] epoch={epoch:03d} "
                f"iteration={iteration_index:03d}/{iterations_per_epoch:03d} "
                f"batch_loss={float(batch_loss.detach().cpu().item()):.6f} "
                f"avg_loss_so_far={aggregate['loss_sum'] / completed_task_losses:.6f} "
                f"{aux_loss_label}_so_far={aggregate['cls_loss_sum'] / completed_task_losses:.6f}"
            )
            if use_proxy_signature_regression:
                print_line += (
                    f" avg_signature_cosine_so_far="
                    f"{aggregate['signature_cosine_sum'] / completed_task_losses:.6f}"
                )
            print(print_line, flush=True)

    if aggregate["task_losses"] == 0:
        return {
            "loss": 0.0,
            "pair_acc": 0.0,
            "pair_loss_mean": 0.0,
            "cls_loss": 0.0,
            "dataset_acc": 0.0,
            "signature_cosine": 0.0,
            "weight_norm": 0.0,
            "num_steps": 0.0,
            "num_pairs": 0.0,
        }

    task_losses = aggregate["task_losses"]
    return {
        "loss": aggregate["loss_sum"] / task_losses,
        "pair_acc": aggregate["pair_acc_sum"] / task_losses,
        "pair_loss_mean": aggregate["pair_loss_mean_sum"] / task_losses,
        "cls_loss": aggregate["cls_loss_sum"] / task_losses,
        "dataset_acc": aggregate["dataset_acc_sum"] / task_losses,
        "signature_cosine": aggregate["signature_cosine_sum"] / task_losses,
        "weight_norm": aggregate["weight_norm_sum"] / task_losses,
        "num_steps": float(iterations_per_epoch),
        "num_task_losses": float(task_losses),
        "num_pairs": aggregate["pairs"],
    }
