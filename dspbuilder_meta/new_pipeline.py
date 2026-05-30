from __future__ import annotations

import json
import random
import re
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from .baseline import SpearmanBaselineEntry, load_proxy_signature_lookup, load_spearman_baselines
from .data_loader import (
    BenchmarkTask,
    TaskContext,
    build_task_contexts,
    discover_benchmark_tasks,
    discover_candidate_configs,
    prompt_dataset_names,
    resolve_dataset_names,
    split_dataset_input,
)
from .engine import (
    compute_dataset_classification_loss,
    compute_pairwise_loss,
    compute_proxy_signature_regression_loss,
    format_weight_vector,
    load_support_samples_from_indices,
    sample_support_samples,
    write_iteration_log,
)
from .model import DSPBuilderMetaModel
from .pipeline import (
    prepare_run_dir,
    select_device,
    set_seed,
    task_id_map_for_logging,
    task_names_for_logging,
    write_summary,
)
from .test import compute_spearman_correlation, flip_spearman_for_lower_is_better_metric
from .valid import write_validation_epoch_summary_logs


@dataclass(frozen=True)
class CandidateRowSplit:
    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]


@dataclass(frozen=True)
class FixedSupportPlan:
    support_indices_sets: tuple[tuple[int, ...], ...]


VALID_ITERATION_LOSS_PATTERN = re.compile(
    r"\[VALID\]\s+epoch=(?P<epoch>\d+).*?\sloss=(?P<loss>[-+0-9.eE]+)\b"
)
VAL_LOSS_LOG_PATTERN = re.compile(
    r"\[VAL-LOSS\]\s+epoch=(?P<epoch>\d+)\s+val_loss=(?P<loss>[-+0-9.eE]+)\b"
)
VAL_NORMALIZED_SCORE_LOG_PATTERN = re.compile(
    r"\[VAL-NORMALIZED-SCORE\]\s+epoch=(?P<epoch>\d+)\s+normalized_score=(?P<score>[-+0-9.eE]+)\b"
)
TEST_SUMMARY_PATTERN = re.compile(
    r"\[TEST-SUMMARY\]\s+epoch=(?P<epoch>\d+).*?\sspearman_mean=(?P<spearman>[-+0-9.eE]+)\b"
)

TRAIN_BATCH_UNIFORM_RATIO = 0.8
TRAIN_BATCH_ADAPTIVE_RATIO = 0.2


def _task_names(task_keys: list[str], available_tasks: dict[str, BenchmarkTask]) -> str:
    return ", ".join(available_tasks[key].display_name for key in task_keys)


def _sample_support_indices(task: TaskContext, support_size: int, rng: random.Random) -> tuple[int, ...]:
    if support_size <= 0:
        raise ValueError("support_size must be positive.")

    population_size = len(task.train_dataset)
    if population_size <= 0:
        raise ValueError(f"Train dataset is empty for task: {task.benchmark.display_name}")

    if population_size >= support_size:
        return tuple(sorted(rng.sample(range(population_size), support_size)))
    return tuple(sorted(rng.randrange(population_size) for _ in range(support_size)))


def _sample_distinct_support_sets(
    task: TaskContext,
    support_size: int,
    num_support_sets: int,
    rng: random.Random,
) -> tuple[tuple[int, ...], ...]:
    if num_support_sets <= 0:
        raise ValueError("num_support_sets must be positive.")

    support_sets: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    max_attempts = max(num_support_sets * 20, 100)
    attempts = 0

    while len(support_sets) < num_support_sets and attempts < max_attempts:
        support_indices = _sample_support_indices(task, support_size=support_size, rng=rng)
        attempts += 1
        if support_indices in seen:
            continue
        seen.add(support_indices)
        support_sets.append(support_indices)

    if len(support_sets) < num_support_sets:
        raise ValueError(
            f"Could not generate {num_support_sets} distinct support sets for "
            f"{task.benchmark.display_name}."
        )
    return tuple(support_sets)


def build_fixed_support_plans(
    tasks: list[TaskContext],
    support_size: int,
    num_support_sets: int,
    rng: random.Random,
) -> dict[str, FixedSupportPlan]:
    return {
        task.benchmark.key: FixedSupportPlan(
            support_indices_sets=_sample_distinct_support_sets(
                task,
                support_size=support_size,
                num_support_sets=num_support_sets,
                rng=rng,
            )
        )
        for task in tasks
    }


def build_candidate_row_splits(
    tasks: list[TaskContext],
    train_count: int,
    val_count: int,
    stratified: bool = False,
    rng: random.Random | None = None,
) -> dict[str, CandidateRowSplit]:
    if train_count < 2:
        raise ValueError("candidate_train_count must be at least 2.")
    if val_count < 2:
        raise ValueError("candidate_val_count must be at least 2.")

    expected_count = train_count + val_count
    splits: dict[str, CandidateRowSplit] = {}
    if stratified:
        num_groups = 10
        if expected_count % num_groups != 0:
            raise ValueError(
                "Stratified candidate split requires "
                "(candidate_train_count + candidate_val_count) to be divisible by 10."
            )
        if train_count % num_groups != 0 or val_count % num_groups != 0:
            raise ValueError(
                "Stratified candidate split requires candidate_train_count and "
                "candidate_val_count to each be divisible by 10."
            )
        group_size = expected_count // num_groups
        train_per_group = train_count // num_groups
        val_per_group = val_count // num_groups
        if train_per_group + val_per_group != group_size:
            raise ValueError(
                "Stratified candidate split requires each of the 10 groups to use "
                "all rows exactly once."
            )
        if rng is None:
            raise ValueError("Stratified candidate split requires an RNG instance.")

    for task in tasks:
        num_candidates = task.benchmark.num_candidates
        if num_candidates != expected_count:
            raise ValueError(
                f"{task.benchmark.display_name} has {num_candidates} candidate rows, "
                f"but this pipeline expects exactly {expected_count} rows "
                f"({train_count} train + {val_count} validation)."
            )
        if not stratified:
            splits[task.benchmark.key] = CandidateRowSplit(
                train_indices=tuple(range(train_count)),
                val_indices=tuple(range(train_count, expected_count)),
            )
            continue

        ranked_indices = sorted(
            range(num_candidates),
            key=lambda candidate_index: (
                float(task.benchmark.metrics[candidate_index]),
                candidate_index,
            ),
        )
        train_indices: list[int] = []
        val_indices: list[int] = []
        for group_start in range(0, expected_count, group_size):
            group_indices = ranked_indices[group_start : group_start + group_size]
            val_group_indices = set(rng.sample(group_indices, val_per_group))
            for candidate_index in group_indices:
                if candidate_index in val_group_indices:
                    val_indices.append(candidate_index)
                else:
                    train_indices.append(candidate_index)
        splits[task.benchmark.key] = CandidateRowSplit(
            train_indices=tuple(train_indices),
            val_indices=tuple(val_indices),
        )
    return splits


def describe_candidate_split_rule(
    train_count: int,
    val_count: int,
    stratified: bool,
) -> str:
    if not stratified:
        return (
            "first candidate_train_count rows for train; "
            "remaining candidate_val_count rows for validation"
        )
    num_groups = 10
    expected_count = train_count + val_count
    group_size = expected_count // num_groups
    train_per_group = train_count // num_groups
    val_per_group = val_count // num_groups
    return (
        "sort candidate rows by ascending metric (lower is better), "
        f"split the ranked {expected_count} rows into {num_groups} groups of {group_size}, "
        f"and sample a fixed {train_per_group}:{val_per_group} train:validation split "
        "inside every group"
    )


def _sample_candidate_indices(
    candidate_pool: tuple[int, ...],
    query_size: int,
    rng: random.Random,
) -> list[int]:
    if len(candidate_pool) < 2:
        raise ValueError("candidate_pool must contain at least 2 rows.")
    if query_size < 2:
        raise ValueError("query_size must be at least 2 for pairwise ranking.")
    if len(candidate_pool) >= query_size:
        return rng.sample(candidate_pool, query_size)
    return [rng.choice(candidate_pool) for _ in range(query_size)]


def _build_training_batch_plan(
    tasks: list[TaskContext],
    adaptive_sampling_tasks: list[TaskContext],
    batch_size: int,
    rng: random.Random,
) -> list[tuple[TaskContext, bool]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not tasks:
        raise ValueError("tasks must contain at least one training dataset.")
    if not adaptive_sampling_tasks:
        return [(rng.choice(tasks), False) for _ in range(batch_size)]

    exact_adaptive_count = batch_size * TRAIN_BATCH_ADAPTIVE_RATIO
    adaptive_count = int(exact_adaptive_count)
    if rng.random() < (exact_adaptive_count - adaptive_count):
        adaptive_count += 1
    adaptive_count = min(batch_size, adaptive_count)

    sampling_modes = [True] * adaptive_count + [False] * (batch_size - adaptive_count)
    rng.shuffle(sampling_modes)
    batch_plan: list[tuple[TaskContext, bool]] = []
    for use_adaptive_sampling in sampling_modes:
        source_tasks = adaptive_sampling_tasks if use_adaptive_sampling else tasks
        batch_plan.append((rng.choice(source_tasks), use_adaptive_sampling))
    return batch_plan


def _build_adaptive_sampling_candidates(
    tasks: list[TaskContext],
    validation_loss_history: dict[str, list[float]],
    window_size: int,
) -> tuple[list[TaskContext], dict[str, dict[str, float | str]]]:
    if window_size <= 0:
        raise ValueError("adaptive_sampling_window must be positive.")

    adaptive_tasks: list[TaskContext] = []
    candidate_details: dict[str, dict[str, float | str]] = {}
    required_history = window_size * 2
    for task in tasks:
        losses = validation_loss_history.get(task.benchmark.key, [])
        if len(losses) < required_history:
            continue

        past_window = losses[-required_history:-window_size]
        recent_window = losses[-window_size:]
        past_mean = float(np.mean(past_window))
        recent_mean = float(np.mean(recent_window))
        if recent_mean <= past_mean:
            continue

        adaptive_tasks.append(task)
        candidate_details[task.benchmark.key] = {
            "dataset": task.benchmark.display_name,
            "past_mean": past_mean,
            "recent_mean": recent_mean,
            "delta": recent_mean - past_mean,
        }
    return adaptive_tasks, candidate_details


def _format_adaptive_sampling_candidates(
    adaptive_sampling_tasks: list[TaskContext],
    adaptive_sampling_details: dict[str, dict[str, float | str]],
) -> str:
    if not adaptive_sampling_tasks:
        return "none"

    formatted_candidates: list[str] = []
    for task in adaptive_sampling_tasks:
        details = adaptive_sampling_details.get(task.benchmark.key)
        if details is None:
            formatted_candidates.append(task.benchmark.display_name)
            continue
        formatted_candidates.append(
            f"{task.benchmark.display_name}(past={float(details['past_mean']):.6f},"
            f"recent={float(details['recent_mean']):.6f},"
            f"delta={float(details['delta']):.6f})"
        )
    return ", ".join(formatted_candidates)


def _stats_from_task_loss(
    task_loss: torch.Tensor,
    pair_stats: dict[str, float],
    cls_loss: torch.Tensor,
    dataset_acc: float,
    signature_cosine: float,
    aux_metric_name: str,
    weight_vector: torch.Tensor,
) -> dict[str, float | list[float] | str]:
    return {
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


def run_candidate_split_train_epoch(
    model: DSPBuilderMetaModel,
    tasks: list[TaskContext],
    candidate_splits: dict[str, CandidateRowSplit],
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
    adaptive_sampling_tasks: list[TaskContext] | None = None,
) -> dict[str, float | int]:
    if not tasks:
        raise ValueError("tasks must contain at least one training dataset.")
    if iterations_per_epoch <= 0:
        raise ValueError("iterations_per_epoch must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not use_proxy_signature_regression:
        missing_class_ids = [
            task.benchmark.display_name for task in tasks if task.dataset_class_id is None
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
        "uniform_task_samples": 0,
        "adaptive_task_samples": 0,
    }

    model.train(True)
    progress_interval = max(1, iterations_per_epoch // 10)
    adaptive_sampling_tasks = adaptive_sampling_tasks or []

    for iteration_index in range(1, iterations_per_epoch + 1):
        optimizer.zero_grad(set_to_none=True)
        batch_losses: list[torch.Tensor] = []
        batch_stats: list[tuple[TaskContext, dict[str, float | list[float] | str]]] = []
        batch_plan = _build_training_batch_plan(
            tasks=tasks,
            adaptive_sampling_tasks=adaptive_sampling_tasks,
            batch_size=batch_size,
            rng=rng,
        )

        for task, used_adaptive_sampling in batch_plan:
            split = candidate_splits[task.benchmark.key]
            support_samples = sample_support_samples(
                task,
                support_size=support_size,
                rng=rng,
                device=device,
            )
            query_indices = _sample_candidate_indices(
                split.train_indices,
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
            batch_stats.append(
                (
                    task,
                    _stats_from_task_loss(
                        task_loss=task_loss,
                        pair_stats=pair_stats,
                        cls_loss=cls_loss,
                        dataset_acc=dataset_acc,
                        signature_cosine=signature_cosine,
                        aux_metric_name=aux_metric_name,
                        weight_vector=weight_vector,
                    ),
                )
            )
            if used_adaptive_sampling:
                aggregate["adaptive_task_samples"] += 1
            else:
                aggregate["uniform_task_samples"] += 1

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
            if adaptive_sampling_tasks:
                print_line += (
                    f" uniform_samples_so_far={aggregate['uniform_task_samples']} "
                    f"adaptive_samples_so_far={aggregate['adaptive_task_samples']}"
                )
            print(print_line, flush=True)

    task_losses = max(aggregate["task_losses"], 1)
    return {
        "loss": aggregate["loss_sum"] / task_losses,
        "pair_acc": aggregate["pair_acc_sum"] / task_losses,
        "pair_loss_mean": aggregate["pair_loss_mean_sum"] / task_losses,
        "cls_loss": aggregate["cls_loss_sum"] / task_losses,
        "dataset_acc": aggregate["dataset_acc_sum"] / task_losses,
        "signature_cosine": aggregate["signature_cosine_sum"] / task_losses,
        "weight_norm": aggregate["weight_norm_sum"] / task_losses,
        "num_steps": float(iterations_per_epoch),
        "num_task_losses": float(aggregate["task_losses"]),
        "num_pairs": aggregate["pairs"],
        "uniform_task_samples": aggregate["uniform_task_samples"],
        "adaptive_task_samples": aggregate["adaptive_task_samples"],
    }


def run_candidate_split_validation_epoch(
    model: DSPBuilderMetaModel,
    tasks: list[TaskContext],
    candidate_splits: dict[str, CandidateRowSplit],
    fixed_support_plans: dict[str, FixedSupportPlan],
    device: torch.device,
    epoch: int,
    log_dir: Path,
) -> dict[str, float | dict[str, float]]:
    aggregate = {
        "loss_sum": 0.0,
        "pair_acc_sum": 0.0,
        "pair_loss_mean_sum": 0.0,
        "weight_norm_sum": 0.0,
        "steps": 0,
        "pairs": 0.0,
    }
    dataset_losses: dict[str, float] = {}

    model.eval()
    with torch.no_grad():
        for task in tasks:
            split = candidate_splits[task.benchmark.key]
            support_indices = fixed_support_plans[task.benchmark.key].support_indices_sets[0]
            support_samples = load_support_samples_from_indices(
                task,
                indices=support_indices,
                device=device,
            )
            query_proxies = task.benchmark.proxies[list(split.val_indices)].to(device)
            query_metrics = task.benchmark.metrics[list(split.val_indices)].to(device)

            weight_vector, _task_embedding, _dataset_logits, _predicted_signature = model(support_samples)
            query_scores = torch.matmul(query_proxies, weight_vector)
            pair_loss_mean, pair_stats = compute_pairwise_loss(query_scores, query_metrics)
            zero_aux_loss = query_scores.new_zeros(())
            stats = _stats_from_task_loss(
                task_loss=pair_loss_mean,
                pair_stats=pair_stats,
                cls_loss=zero_aux_loss,
                dataset_acc=0.0,
                signature_cosine=0.0,
                aux_metric_name="dataset_acc",
                weight_vector=weight_vector,
            )
            write_iteration_log(
                log_dir=log_dir,
                stage_name="valid",
                epoch=epoch,
                dataset_name=task.benchmark.display_name,
                iteration_index=1,
                stats=stats,
            )

            aggregate["loss_sum"] += float(stats["loss"])
            aggregate["pair_acc_sum"] += float(stats["pair_acc"])
            aggregate["pair_loss_mean_sum"] += float(stats["pair_loss_mean"])
            aggregate["weight_norm_sum"] += float(stats["weight_norm"])
            aggregate["steps"] += 1
            aggregate["pairs"] += float(stats["num_pairs"])
            dataset_losses[task.benchmark.key] = float(stats["loss"])

    steps = max(aggregate["steps"], 1)
    return {
        "loss": aggregate["loss_sum"] / steps,
        "pair_acc": aggregate["pair_acc_sum"] / steps,
        "pair_loss_mean": aggregate["pair_loss_mean_sum"] / steps,
        "cls_loss": 0.0,
        "dataset_acc": 0.0,
        "signature_cosine": 0.0,
        "weight_norm": aggregate["weight_norm_sum"] / steps,
        "num_steps": float(aggregate["steps"]),
        "num_pairs": aggregate["pairs"],
        "dataset_losses": dataset_losses,
    }


def _compute_normalized_validation_score(
    tasks: list[TaskContext],
    validation_loss_history: dict[str, list[float]],
) -> tuple[float, dict[str, float]]:
    normalized_scores: dict[str, float] = {}
    for task in tasks:
        task_key = task.benchmark.key
        losses = validation_loss_history.get(task_key, [])
        if not losses:
            raise ValueError(
                f"Missing validation loss history for task: {task.benchmark.display_name}"
            )

        baseline_loss = float(losses[0])
        current_loss = float(losses[-1])
        # Avoid dividing by zero if the very first validation loss is already zero.
        if np.isclose(baseline_loss, 0.0):
            normalized_score = 0.0
        else:
            normalized_score = (baseline_loss - current_loss) / baseline_loss
        normalized_scores[task_key] = float(normalized_score)

    if not normalized_scores:
        return 0.0, normalized_scores

    aggregate_score = float(
        np.mean([normalized_scores[task.benchmark.key] for task in tasks])
    )
    return aggregate_score, normalized_scores


def initialize_new_log_files(
    train_log_dir: Path,
    val_log_dir: Path,
    test_log_dir: Path,
    train_tasks: list[TaskContext],
    test_tasks: list[TaskContext],
    candidate_splits: dict[str, CandidateRowSplit],
    candidate_split_rule: str,
    fixed_val_support_plans: dict[str, FixedSupportPlan],
    fixed_test_support_plans: dict[str, FixedSupportPlan],
    train_only: bool,
) -> None:
    train_log_dir.mkdir(parents=True, exist_ok=True)
    val_log_dir.mkdir(parents=True, exist_ok=True)

    for task in train_tasks:
        split = candidate_splits[task.benchmark.key]
        (train_log_dir / f"{task.benchmark.display_name}.txt").write_text(
            "\n".join(
                [
                    "# Train iteration logs",
                    f"# candidate_split_rule={candidate_split_rule}",
                    f"# train_candidate_indices={list(split.train_indices)}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        val_support = fixed_val_support_plans[task.benchmark.key].support_indices_sets[0]
        (val_log_dir / f"{task.benchmark.display_name}.txt").write_text(
            "\n".join(
                [
                    "# Validation iteration logs",
                    f"# candidate_split_rule={candidate_split_rule}",
                    f"# val_candidate_indices={list(split.val_indices)}",
                    f"# fixed_support_indices={list(val_support)}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    (val_log_dir / "val_loss.txt").write_text(
        "# Validation mean loss logs\n",
        encoding="utf-8",
    )
    (val_log_dir / "normalized_score.txt").write_text(
        "\n".join(
            [
                "# Validation normalized score logs",
                "# normalized score = (val_loss(first_epoch) - val_loss(current_epoch)) / val_loss(first_epoch)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    if train_only:
        return

    test_log_dir.mkdir(parents=True, exist_ok=True)
    for task in test_tasks:
        support_sets = [
            list(indices)
            for indices in fixed_test_support_plans[task.benchmark.key].support_indices_sets
        ]
        (test_log_dir / f"{task.benchmark.display_name}.txt").write_text(
            "\n".join(
                [
                    "# Checkpoint test Spearman correlation logs",
                    "# candidate_indices=all",
                    f"# fixed_support_indices_sets={support_sets}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )


def write_validation_loss_log(
    log_dir: Path,
    epoch: int,
    val_loss: float,
    early_stopping_counter: int,
    best_val_loss: float,
) -> None:
    best_val_label = f"{best_val_loss:.6f}" if best_val_loss != float("inf") else "inf"
    log_line = (
        f"[VAL-LOSS] "
        f"epoch={epoch:03d} "
        f"val_loss={val_loss:.6f} "
        f"best_val_loss={best_val_label} "
        f"early_stopping_counter={early_stopping_counter}"
    )
    with (log_dir / "val_loss.txt").open("a", encoding="utf-8") as handle:
        handle.write(log_line + "\n")


def write_validation_normalized_score_log(
    log_dir: Path,
    epoch: int,
    normalized_score: float,
    early_stopping_counter: int,
    best_normalized_score: float,
) -> None:
    best_score_label = (
        f"{best_normalized_score:.6f}" if best_normalized_score != float("-inf") else "-inf"
    )
    log_line = (
        f"[VAL-NORMALIZED-SCORE] "
        f"epoch={epoch:03d} "
        f"normalized_score={normalized_score:.6f} "
        f"best_normalized_score={best_score_label} "
        f"early_stopping_counter={early_stopping_counter}"
    )
    with (log_dir / "normalized_score.txt").open("a", encoding="utf-8") as handle:
        handle.write(log_line + "\n")


def _extract_epoch_loss_series(log_path: Path) -> tuple[list[int], list[float]]:
    if log_path.name == "val_loss.txt":
        pattern = VAL_LOSS_LOG_PATTERN
    else:
        pattern = VALID_ITERATION_LOSS_PATTERN

    losses_by_epoch: dict[int, list[float]] = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match is None:
            continue
        epoch = int(match.group("epoch"))
        loss = float(match.group("loss"))
        losses_by_epoch.setdefault(epoch, []).append(loss)

    epochs = sorted(losses_by_epoch)
    averaged_losses = [
        float(np.mean(losses_by_epoch[epoch]))
        for epoch in epochs
    ]
    return epochs, averaged_losses


def _extract_epoch_normalized_score_series(log_path: Path) -> tuple[list[int], list[float]]:
    scores_by_epoch: dict[int, list[float]] = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = VAL_NORMALIZED_SCORE_LOG_PATTERN.search(line)
        if match is None:
            continue
        epoch = int(match.group("epoch"))
        score = float(match.group("score"))
        scores_by_epoch.setdefault(epoch, []).append(score)

    epochs = sorted(scores_by_epoch)
    averaged_scores = [
        float(np.mean(scores_by_epoch[epoch]))
        for epoch in epochs
    ]
    return epochs, averaged_scores


def plot_validation_loss_logs(log_dir: Path) -> list[str]:
    saved_plot_paths: list[str] = []
    for log_path in sorted(log_dir.glob("*.txt")):
        if log_path.name == "normalized_score.txt":
            epochs, values = _extract_epoch_normalized_score_series(log_path)
            y_label = "Normalized Score"
            title = "normalized_score"
        else:
            epochs, values = _extract_epoch_loss_series(log_path)
            y_label = "Loss"
            title = f"{log_path.stem} loss"
        if not epochs:
            continue

        plt.figure(figsize=(8, 4.5))
        plt.plot(epochs, values, marker="o", linewidth=1.8, markersize=4)
        plt.xlabel("Epoch")
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = log_dir / f"{log_path.stem}.png"
        plt.savefig(output_path, dpi=200)
        plt.close()
        saved_plot_paths.append(str(output_path))

    return saved_plot_paths


def _extract_epoch_spearman_series(log_path: Path) -> tuple[list[int], list[float]]:
    spearman_by_epoch: dict[int, list[float]] = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = TEST_SUMMARY_PATTERN.search(line)
        if match is None:
            continue
        epoch = int(match.group("epoch"))
        spearman = float(match.group("spearman"))
        spearman_by_epoch.setdefault(epoch, []).append(spearman)

    epochs = sorted(spearman_by_epoch)
    averaged_spearman = [
        float(np.mean(spearman_by_epoch[epoch]))
        for epoch in epochs
    ]
    return epochs, averaged_spearman


def plot_test_spearman_logs(log_dir: Path) -> list[str]:
    saved_plot_paths: list[str] = []
    for log_path in sorted(log_dir.glob("*.txt")):
        epochs, spearman_values = _extract_epoch_spearman_series(log_path)
        if not epochs:
            continue

        plt.figure(figsize=(8, 4.5))
        plt.plot(epochs, spearman_values, marker="o", linewidth=1.8, markersize=4)
        plt.xlabel("Epoch")
        plt.ylabel("Spearman")
        plt.title(f"{log_path.stem} Spearman")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = log_dir / f"{log_path.stem}.png"
        plt.savefig(output_path, dpi=200)
        plt.close()
        saved_plot_paths.append(str(output_path))

    return saved_plot_paths


def prune_epoch_checkpoints(
    checkpoint_paths_by_epoch: dict[int, Path],
    best_epoch: int,
    final_epoch: int,
) -> tuple[dict[int, Path], dict[int, Path]]:
    if not checkpoint_paths_by_epoch:
        return {}, {}
    if best_epoch <= 0:
        raise ValueError("best_epoch must be positive when pruning checkpoints.")
    if final_epoch <= 0:
        raise ValueError("final_epoch must be positive when pruning checkpoints.")

    retained_epochs = {best_epoch, final_epoch}
    retained_checkpoints: dict[int, Path] = {}
    removed_checkpoints: dict[int, Path] = {}
    for epoch, checkpoint_path in sorted(checkpoint_paths_by_epoch.items()):
        if epoch in retained_epochs:
            retained_checkpoints[epoch] = checkpoint_path
            continue
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        removed_checkpoints[epoch] = checkpoint_path

    return retained_checkpoints, removed_checkpoints


def print_new_run_overview(
    available_tasks: dict[str, BenchmarkTask],
    train_keys: list[str],
    test_keys: list[str],
    train_tasks: list[TaskContext],
    test_tasks: list[TaskContext],
    candidate_splits: dict[str, CandidateRowSplit],
    candidate_split_rule: str,
    fixed_val_support_plans: dict[str, FixedSupportPlan],
    fixed_test_support_plans: dict[str, FixedSupportPlan],
    train_dataset_class_ids: dict[str, int],
    device: torch.device,
    model: DSPBuilderMetaModel,
    run_dir: Path,
    train_only: bool,
    train_batch_size: int,
    train_iterations_per_epoch: int,
    adaptive_sampling_window: int,
) -> None:
    print("Available tasks:", ", ".join(task.display_name for task in available_tasks.values()))
    print("Train/Val tasks:", _task_names(train_keys, available_tasks))
    if train_only:
        print("Test tasks: skipped (--train-only)")
    else:
        print("Test tasks:", _task_names(test_keys, available_tasks))
    print(f"Using device: {device}")
    print(
        f"raw_stat_emb={model.support_encoder.raw_stat_emb} "
        f"sample_embedding_dim={model.sample_embedding_dim} "
        f"number_of_conv1d_layer={model.support_encoder.number_of_conv1d_layer} "
        f"sample_encoder_norm={model.support_encoder.sample_encoder_norm} "
        f"number_of_setencoder_mlp_layers={model.set_encoder.number_of_setencoder_mlp_layers} "
        f"set_encoder_norm={model.set_encoder.set_encoder_norm} "
        f"dataset_description_dim={model.dataset_description_dim} "
        f"weight_head_layers={model.weight_head_layers} "
        f"mlp_norm={model.mlp_norm}"
    )
    print(
        "Auxiliary mode:",
        "proxy_signature_regression" if model.proxy_signature_regression else "dataset_classification",
    )
    print(
        f"Train schedule: batch_size={train_batch_size} "
        f"iterations_per_epoch={train_iterations_per_epoch}"
    )
    print(
        f"Adaptive task sampling: uniform={TRAIN_BATCH_UNIFORM_RATIO:.0%} "
        f"adaptive={TRAIN_BATCH_ADAPTIVE_RATIO:.0%} "
        f"window={adaptive_sampling_window}"
    )
    print(f"Candidate split rule: {candidate_split_rule}")
    print(
        "Train dataset ids:",
        ", ".join(
            f"{dataset_id}:{available_tasks[key].display_name}"
            for key, dataset_id in train_dataset_class_ids.items()
        ),
    )
    print(f"Run directory: {run_dir}")

    for task in train_tasks:
        split = candidate_splits[task.benchmark.key]
        val_support = fixed_val_support_plans[task.benchmark.key].support_indices_sets[0]
        print(
            f"[train/valid] {task.benchmark.display_name}: "
            f"metric={task.benchmark.metric_name}, "
            f"sample_shape={task.sample_shape}, "
            f"train_examples={len(task.train_dataset)}, "
            f"train_candidates={list(split.train_indices)}, "
            f"val_candidates={list(split.val_indices)}, "
            f"fixed_val_support={list(val_support)}"
        )

    if not train_only:
        for task in test_tasks:
            support_sets = fixed_test_support_plans[task.benchmark.key].support_indices_sets
            print(
                f"[test-plan] {task.benchmark.display_name}: "
                f"metric={task.benchmark.metric_name}, "
                f"num_candidates={task.benchmark.num_candidates}, "
                f"fixed_support_sets={[list(indices) for indices in support_sets]}"
            )


def _write_checkpoint_test_iteration_log(
    log_dir: Path,
    epoch: int,
    dataset_name: str,
    metric_name: str,
    support_set_index: int,
    total_support_sets: int,
    support_indices: list[int],
    weight_vector: list[float],
    spearman_corr: float,
    num_candidates: int,
) -> None:
    log_line = (
        f"[TEST] "
        f"epoch={epoch:03d} "
        f"dataset={dataset_name} "
        f"metric={metric_name} "
        f"support_set={support_set_index:02d}/{total_support_sets:02d} "
        f"spearman_corr={spearman_corr:.6f} "
        f"num_candidates={num_candidates} "
        f"support_indices={support_indices} "
        f"weight_norm={float(np.linalg.norm(weight_vector)):.6f} "
        f"weight_vector={format_weight_vector(weight_vector)}"
    )
    with (log_dir / f"{dataset_name}.txt").open("a", encoding="utf-8") as handle:
        handle.write(log_line + "\n")


def _write_checkpoint_test_summary_log(
    log_dir: Path,
    epoch: int,
    dataset_name: str,
    metric_name: str,
    spearman_values: list[float],
    num_candidates: int,
    baseline_entry: SpearmanBaselineEntry | None = None,
) -> dict[str, float | int | str | None]:
    spearman_mean = float(np.mean(spearman_values)) if spearman_values else 0.0
    spearman_std = float(np.std(spearman_values)) if spearman_values else 0.0
    baseline_suffix = ""
    if baseline_entry is not None:
        baseline_suffix = (
            f" baseline_best_proxy={baseline_entry['best_proxy']} "
            f"baseline_coefficient={float(baseline_entry['coefficient']):.6f}"
        )
    log_line = (
        f"[TEST-SUMMARY] "
        f"epoch={epoch:03d} "
        f"dataset={dataset_name} "
        f"metric={metric_name} "
        f"spearman_mean={spearman_mean:.6f} "
        f"spearman_std={spearman_std:.6f} "
        f"support_sets={len(spearman_values)} "
        f"num_candidates={num_candidates}"
        f"{baseline_suffix}"
    )
    with (log_dir / f"{dataset_name}.txt").open("a", encoding="utf-8") as handle:
        handle.write(log_line + "\n")
    return {
        "dataset": dataset_name,
        "metric_name": metric_name,
        "spearman_mean": spearman_mean,
        "spearman_std": spearman_std,
        "support_sets": len(spearman_values),
        "num_candidates": num_candidates,
        "baseline_best_proxy": baseline_entry["best_proxy"] if baseline_entry is not None else None,
        "baseline_coefficient": float(baseline_entry["coefficient"]) if baseline_entry is not None else None,
    }


def run_checkpoint_test_sweep(
    model: DSPBuilderMetaModel,
    tasks: list[TaskContext],
    device: torch.device,
    checkpoint_paths_by_epoch: dict[int, Path],
    fixed_support_plans: dict[str, FixedSupportPlan],
    log_dir: Path,
    output_txt_path: Path,
    baseline_lookup: dict[str, SpearmanBaselineEntry] | None = None,
) -> dict[str, object]:
    epoch_records: list[dict[str, object]] = []
    dataset_names = [task.benchmark.display_name for task in tasks]
    header = ["epoch", "checkpoint", "spearman_mean", "spearman_std"]
    for dataset_name in dataset_names:
        header.extend([f"{dataset_name}_spearman_mean", f"{dataset_name}_spearman_std"])
    output_lines = ["\t".join(header)]

    for epoch, checkpoint_path in sorted(checkpoint_paths_by_epoch.items()):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        all_spearman_values: list[float] = []
        dataset_results: dict[str, dict[str, float | int | str | None]] = {}

        with torch.no_grad():
            for task in tasks:
                dataset_spearman_values: list[float] = []
                all_candidate_proxies = task.benchmark.proxies.to(device)
                all_candidate_metrics = task.benchmark.metrics.to(device)
                support_sets = fixed_support_plans[task.benchmark.key].support_indices_sets
                baseline_entry = (
                    baseline_lookup.get(task.benchmark.key) if baseline_lookup is not None else None
                )

                for support_set_index, support_indices_tuple in enumerate(support_sets, start=1):
                    support_indices = list(support_indices_tuple)
                    support_samples = load_support_samples_from_indices(
                        task,
                        indices=support_indices,
                        device=device,
                    )
                    weight_vector, _task_embedding, _dataset_logits, _predicted_signature = model(support_samples)
                    predicted_proxy_scores = torch.matmul(all_candidate_proxies, weight_vector)
                    spearman_corr = flip_spearman_for_lower_is_better_metric(
                        compute_spearman_correlation(predicted_proxy_scores, all_candidate_metrics)
                    )
                    weight_vector_list = [float(value) for value in weight_vector.detach().cpu().tolist()]
                    _write_checkpoint_test_iteration_log(
                        log_dir=log_dir,
                        epoch=epoch,
                        dataset_name=task.benchmark.display_name,
                        metric_name=task.benchmark.metric_name,
                        support_set_index=support_set_index,
                        total_support_sets=len(support_sets),
                        support_indices=support_indices,
                        weight_vector=weight_vector_list,
                        spearman_corr=spearman_corr,
                        num_candidates=task.benchmark.num_candidates,
                    )
                    dataset_spearman_values.append(spearman_corr)
                    all_spearman_values.append(spearman_corr)

                dataset_results[task.benchmark.display_name] = _write_checkpoint_test_summary_log(
                    log_dir=log_dir,
                    epoch=epoch,
                    dataset_name=task.benchmark.display_name,
                    metric_name=task.benchmark.metric_name,
                    spearman_values=dataset_spearman_values,
                    num_candidates=task.benchmark.num_candidates,
                    baseline_entry=baseline_entry,
                )

        overall_mean = float(np.mean(all_spearman_values)) if all_spearman_values else 0.0
        overall_std = float(np.std(all_spearman_values)) if all_spearman_values else 0.0
        record = {
            "epoch": epoch,
            "checkpoint": str(checkpoint_path),
            "spearman_mean": overall_mean,
            "spearman_std": overall_std,
            "dataset_results": dataset_results,
        }
        epoch_records.append(record)

        row = [str(epoch), str(checkpoint_path), f"{overall_mean:.6f}", f"{overall_std:.6f}"]
        for dataset_name in dataset_names:
            dataset_record = dataset_results[dataset_name]
            row.extend(
                [
                    f"{float(dataset_record['spearman_mean']):.6f}",
                    f"{float(dataset_record['spearman_std']):.6f}",
                ]
            )
        output_lines.append("\t".join(row))
        output_txt_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
        print(
            f"[TEST-CHECKPOINT] epoch={epoch:03d} "
            f"spearman_mean={overall_mean:.6f} spearman_std={overall_std:.6f} "
            f"checkpoint={checkpoint_path}",
            flush=True,
        )

    output_txt_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    output_json_path = output_txt_path.with_suffix(".json")
    output_json_path.write_text(json.dumps(epoch_records, indent=2), encoding="utf-8")

    best_record = max(epoch_records, key=lambda item: float(item["spearman_mean"])) if epoch_records else None
    return {
        "spearman_by_epoch_path": str(output_txt_path),
        "spearman_by_epoch_json_path": str(output_json_path),
        "epoch_results": epoch_records,
        "best_test_epoch": int(best_record["epoch"]) if best_record is not None else None,
        "best_test_spearman_mean": (
            float(best_record["spearman_mean"]) if best_record is not None else None
        ),
    }


def run_new_pipeline(args: Namespace) -> int:
    if args.adaptive_sampling_window <= 0:
        raise ValueError("adaptive_sampling_window must be positive.")

    set_seed(args.seed)

    repo_root = Path(__file__).resolve().parent.parent
    benchmark_dir = args.benchmark_dir.resolve()
    candidate_dir = args.candidate_dir.resolve()
    proxy_signature_lookup_path = repo_root / "benchmark" / "lookup" / "proxy_signature_lookup.csv"
    proxy_signature_lookup = (
        load_proxy_signature_lookup(proxy_signature_lookup_path)
        if args.proxy_signature_regression
        else None
    )
    available_tasks = discover_benchmark_tasks(
        benchmark_dir,
        proxy_signature_lookup=proxy_signature_lookup,
    )
    candidate_configs = discover_candidate_configs(candidate_dir)
    baseline_lookup = load_spearman_baselines(repo_root / "benchmark" / "lookup" / "spearman_baseline.csv")

    train_keys = (
        resolve_dataset_names(split_dataset_input(args.train_datasets), available_tasks, "train")
        if args.train_datasets.strip()
        else prompt_dataset_names("train/val", available_tasks)
    )
    if args.val_datasets.strip():
        requested_val_keys = resolve_dataset_names(
            split_dataset_input(args.val_datasets),
            available_tasks,
            "val",
        )
        if set(requested_val_keys) != set(train_keys):
            raise ValueError(
                "train_new_dspbuilder_meta.py uses the same dataset subset for training and validation. "
                "Please omit --val-datasets or pass the same datasets as --train-datasets."
            )
        if requested_val_keys != train_keys:
            print("[Info] --val-datasets contains the same set; using --train-datasets order for validation.")
    val_keys = list(train_keys)

    if args.train_only:
        if args.test_datasets.strip():
            print("[Info] --train-only enabled: ignoring --test-datasets.")
        test_keys: list[str] = []
    else:
        test_keys = (
            resolve_dataset_names(split_dataset_input(args.test_datasets), available_tasks, "test")
            if args.test_datasets.strip()
            else prompt_dataset_names("test", available_tasks)
        )

    train_dataset_class_ids = {key: dataset_id for dataset_id, key in enumerate(train_keys)}
    train_tasks = build_task_contexts(
        train_keys,
        available_tasks,
        candidate_configs,
        repo_root,
        dataset_class_ids=train_dataset_class_ids,
    )
    if not train_tasks:
        raise ValueError("At least one training dataset is required.")

    val_tasks = train_tasks
    test_tasks = build_task_contexts(test_keys, available_tasks, candidate_configs, repo_root)

    proxy_dim = len(train_tasks[0].benchmark.proxy_names)
    for task in train_tasks + test_tasks:
        if len(task.benchmark.proxy_names) != proxy_dim:
            raise ValueError("All benchmark CSVs must share the same proxy dimension.")

    candidate_splits = build_candidate_row_splits(
        val_tasks,
        train_count=args.candidate_train_count,
        val_count=args.candidate_val_count,
        stratified=args.stratified,
        rng=random.Random(args.seed + 30_000) if args.stratified else None,
    )
    candidate_split_rule = describe_candidate_split_rule(
        train_count=args.candidate_train_count,
        val_count=args.candidate_val_count,
        stratified=args.stratified,
    )

    device = select_device(args.device)
    model = DSPBuilderMetaModel(
        proxy_dim=proxy_dim,
        num_dataset_classes=len(train_dataset_class_ids),
        encoder_hidden_dim=args.encoder_hidden_dim,
        head_hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        raw_stat_emb=args.raw_stat_emb,
        number_of_conv1d_layer=args.number_of_conv1d_layer,
        sample_encoder_norm=args.sample_encoder_norm,
        number_of_setencoder_mlp_layers=args.number_of_setencoder_mlp_layers,
        set_encoder_norm=args.set_encoder_norm,
        weight_head_layers=args.weight_head_layers,
        mlp_norm=args.mlp_norm,
    ).to(device)
    model.proxy_signature_regression = bool(args.proxy_signature_regression)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    run_dir = prepare_run_dir(args.output_dir.resolve())
    checkpoint_dir = run_dir / "epoch_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = run_dir / "best_checkpoint.pth"
    train_log_dir = run_dir / "train_logs"
    val_log_dir = run_dir / "valid_logs"
    test_log_dir = run_dir / "test_logs"

    fixed_val_support_plans = build_fixed_support_plans(
        val_tasks,
        support_size=args.support_size,
        num_support_sets=1,
        rng=random.Random(args.seed + 10_000),
    )
    fixed_test_support_plans = (
        build_fixed_support_plans(
            test_tasks,
            support_size=args.support_size,
            num_support_sets=args.eval_iterations_per_dataset,
            rng=random.Random(args.seed + 20_000),
        )
        if not args.train_only
        else {}
    )

    config_payload = {
        "benchmark_dir": str(benchmark_dir),
        "candidate_dir": str(candidate_dir),
        "train_datasets": task_names_for_logging(train_keys, available_tasks),
        "val_datasets": task_names_for_logging(val_keys, available_tasks),
        "test_datasets": task_names_for_logging(test_keys, available_tasks),
        "candidate_train_count": args.candidate_train_count,
        "candidate_val_count": args.candidate_val_count,
        "stratified": args.stratified,
        "candidate_split_rule": candidate_split_rule,
        "epochs": args.epochs,
        "train_iterations_per_epoch": args.iterations_per_epoch,
        "train_batch_size": args.train_batch_size,
        "test_iterations_per_dataset": args.eval_iterations_per_dataset,
        "support_size": args.support_size,
        "train_query_size": args.train_query_size,
        "val_query_size": args.candidate_val_count,
        "test_query_size": "all_candidates",
        "encoder_hidden_dim": args.encoder_hidden_dim,
        "number_of_conv1d_layer": args.number_of_conv1d_layer,
        "sample_encoder_norm": args.sample_encoder_norm,
        "number_of_setencoder_mlp_layers": args.number_of_setencoder_mlp_layers,
        "set_encoder_norm": args.set_encoder_norm,
        "hidden_dim": args.hidden_dim,
        "weight_head_layers": args.weight_head_layers,
        "mlp_norm": args.mlp_norm,
        "raw_stat_emb": args.raw_stat_emb,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "cls_loss_weight": args.cls_loss_weight,
        "proxy_signature_regression": args.proxy_signature_regression,
        "proxy_signature_lookup": str(proxy_signature_lookup_path) if args.proxy_signature_regression else None,
        "adaptive_sampling_window": args.adaptive_sampling_window,
        "train_uniform_sampling_ratio": TRAIN_BATCH_UNIFORM_RATIO,
        "train_adaptive_sampling_ratio": TRAIN_BATCH_ADAPTIVE_RATIO,
        "patience": args.patience,
        "seed": args.seed,
        "device": str(device),
        "train_only": args.train_only,
        "train_dataset_class_ids": task_id_map_for_logging(train_dataset_class_ids, available_tasks),
        "fixed_val_support_indices": {
            task.benchmark.display_name: [
                list(indices)
                for indices in fixed_val_support_plans[task.benchmark.key].support_indices_sets
            ]
            for task in val_tasks
        },
        "candidate_split_indices": {
            task.benchmark.display_name: {
                "train": list(candidate_splits[task.benchmark.key].train_indices),
                "val": list(candidate_splits[task.benchmark.key].val_indices),
            }
            for task in val_tasks
        },
        "fixed_test_support_indices": {
            task.benchmark.display_name: [
                list(indices)
                for indices in fixed_test_support_plans[task.benchmark.key].support_indices_sets
            ]
            for task in test_tasks
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    initialize_new_log_files(
        train_log_dir=train_log_dir,
        val_log_dir=val_log_dir,
        test_log_dir=test_log_dir,
        train_tasks=train_tasks,
        test_tasks=test_tasks,
        candidate_splits=candidate_splits,
        candidate_split_rule=candidate_split_rule,
        fixed_val_support_plans=fixed_val_support_plans,
        fixed_test_support_plans=fixed_test_support_plans,
        train_only=args.train_only,
    )

    print_new_run_overview(
        available_tasks=available_tasks,
        train_keys=train_keys,
        test_keys=test_keys,
        train_tasks=train_tasks,
        test_tasks=test_tasks,
        candidate_splits=candidate_splits,
        candidate_split_rule=candidate_split_rule,
        fixed_val_support_plans=fixed_val_support_plans,
        fixed_test_support_plans=fixed_test_support_plans,
        train_dataset_class_ids=train_dataset_class_ids,
        device=device,
        model=model,
        run_dir=run_dir,
        train_only=args.train_only,
        train_batch_size=args.train_batch_size,
        train_iterations_per_epoch=args.iterations_per_epoch,
        adaptive_sampling_window=args.adaptive_sampling_window,
    )

    train_rng = random.Random(args.seed)
    history: list[dict[str, object]] = []
    checkpoint_paths_by_epoch: dict[int, Path] = {}
    best_val_loss = float("inf")
    best_val_normalized_score = float("-inf")
    best_epoch = 0
    early_stopping_counter = 0
    lowest_raw_val_loss = float("inf")
    lowest_raw_val_loss_epoch = 0
    validation_loss_history: dict[str, list[float]] = {task.benchmark.key: [] for task in val_tasks}
    adaptive_sampling_tasks: list[TaskContext] = []
    adaptive_sampling_details: dict[str, dict[str, float | str]] = {}

    for epoch in range(1, args.epochs + 1):
        epoch_adaptive_sampling_tasks = list(adaptive_sampling_tasks)
        epoch_adaptive_sampling_details = {
            task_key: dict(details)
            for task_key, details in adaptive_sampling_details.items()
        }
        epoch_sampling_mode = "uniform_only" if not epoch_adaptive_sampling_tasks else "mixed"
        epoch_adaptive_ratio = 0.0 if not epoch_adaptive_sampling_tasks else TRAIN_BATCH_ADAPTIVE_RATIO
        print(
            f"[ADAPTIVE-SAMPLING] epoch={epoch:03d} "
            f"mode={epoch_sampling_mode} "
            f"uniform_ratio={1.0 - epoch_adaptive_ratio:.2f} "
            f"adaptive_ratio={epoch_adaptive_ratio:.2f} "
            f"window={args.adaptive_sampling_window} "
            f"candidates={_format_adaptive_sampling_candidates(epoch_adaptive_sampling_tasks, epoch_adaptive_sampling_details)}",
            flush=True,
        )
        train_stats = run_candidate_split_train_epoch(
            model=model,
            tasks=train_tasks,
            candidate_splits=candidate_splits,
            device=device,
            rng=train_rng,
            iterations_per_epoch=args.iterations_per_epoch,
            batch_size=args.train_batch_size,
            support_size=args.support_size,
            query_size=args.train_query_size,
            optimizer=optimizer,
            epoch=epoch,
            log_dir=train_log_dir,
            cls_loss_weight=args.cls_loss_weight,
            use_proxy_signature_regression=args.proxy_signature_regression,
            adaptive_sampling_tasks=epoch_adaptive_sampling_tasks,
        )
        val_stats = run_candidate_split_validation_epoch(
            model=model,
            tasks=val_tasks,
            candidate_splits=candidate_splits,
            fixed_support_plans=fixed_val_support_plans,
            device=device,
            epoch=epoch,
            log_dir=val_log_dir,
        )
        dataset_losses = val_stats["dataset_losses"]
        if not isinstance(dataset_losses, dict):
            raise TypeError("run_candidate_split_validation_epoch must return dataset_losses as a dict.")
        for task_key, loss in dataset_losses.items():
            validation_loss_history.setdefault(task_key, []).append(float(loss))
        adaptive_sampling_tasks, adaptive_sampling_details = _build_adaptive_sampling_candidates(
            val_tasks,
            validation_loss_history=validation_loss_history,
            window_size=args.adaptive_sampling_window,
        )
        next_epoch_adaptive_ratio = 0.0 if not adaptive_sampling_tasks else TRAIN_BATCH_ADAPTIVE_RATIO
        print(
            f"[ADAPTIVE-SAMPLING-UPDATE] next_epoch={epoch + 1:03d} "
            f"uniform_ratio={1.0 - next_epoch_adaptive_ratio:.2f} "
            f"adaptive_ratio={next_epoch_adaptive_ratio:.2f} "
            f"candidates={_format_adaptive_sampling_candidates(adaptive_sampling_tasks, adaptive_sampling_details)}",
            flush=True,
        )

        current_val_loss = float(val_stats["loss"])
        current_val_normalized_score, dataset_normalized_scores = _compute_normalized_validation_score(
            val_tasks,
            validation_loss_history=validation_loss_history,
        )
        if current_val_loss < lowest_raw_val_loss:
            lowest_raw_val_loss = current_val_loss
            lowest_raw_val_loss_epoch = epoch

        previous_best_val_normalized_score = best_val_normalized_score
        previous_best_score_label = (
            f"{previous_best_val_normalized_score:.6f}"
            if previous_best_val_normalized_score != float("-inf")
            else "-inf"
        )
        print(
            f"[VALID] epoch={epoch:03d} "
            f"val_loss={current_val_loss:.6f} "
            f"val_normalized_score={current_val_normalized_score:.6f} "
            f"best_val_normalized_score_so_far={previous_best_score_label}",
            flush=True,
        )

        checkpoint_path = checkpoint_dir / f"epoch_{epoch:03d}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        checkpoint_paths_by_epoch[epoch] = checkpoint_path

        improved = current_val_normalized_score > best_val_normalized_score
        if improved:
            best_val_normalized_score = current_val_normalized_score
            best_val_loss = current_val_loss
            best_epoch = epoch
            early_stopping_counter = 0
            torch.save(model.state_dict(), best_checkpoint_path)
            print(
                f"[BEST] epoch={epoch:03d} "
                f"best_val_normalized_score: "
                f"{previous_best_score_label} -> {best_val_normalized_score:.6f} "
                f"val_loss_at_best_score={best_val_loss:.6f} "
                f"saved_checkpoint={best_checkpoint_path}"
            )
        else:
            early_stopping_counter += 1
            print(
                f"EarlyStopping counter: {early_stopping_counter} out of {args.patience} "
                f"(normalized score)"
            )

        epoch_record: dict[str, object] = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_pair_acc": train_stats["pair_acc"],
            "train_pair_loss_mean": train_stats["pair_loss_mean"],
            "train_dataset_acc": train_stats["dataset_acc"],
            "train_signature_cosine": train_stats["signature_cosine"],
            "val_loss": val_stats["loss"],
            "val_normalized_score": current_val_normalized_score,
            "val_pair_acc": val_stats["pair_acc"],
            "val_pair_loss_mean": val_stats["pair_loss_mean"],
            "train_weight_norm": train_stats["weight_norm"],
            "val_weight_norm": val_stats["weight_norm"],
            "checkpoint": str(checkpoint_path),
            "early_stopping_counter": early_stopping_counter,
            "best_val_normalized_score": best_val_normalized_score,
            "best_val_loss_at_best_score": best_val_loss,
            "lowest_raw_val_loss": lowest_raw_val_loss,
            "lowest_raw_val_loss_epoch": lowest_raw_val_loss_epoch,
            "val_dataset_normalized_scores": {
                available_tasks[task_key].display_name: score
                for task_key, score in dataset_normalized_scores.items()
            },
            "adaptive_sampling_candidate_count": len(epoch_adaptive_sampling_tasks),
            "adaptive_sampling_candidate_datasets": [
                task.benchmark.display_name for task in epoch_adaptive_sampling_tasks
            ],
            "adaptive_sampling_candidate_details": [
                epoch_adaptive_sampling_details[task.benchmark.key]
                for task in epoch_adaptive_sampling_tasks
                if task.benchmark.key in epoch_adaptive_sampling_details
            ],
            "next_adaptive_sampling_candidate_datasets": [
                task.benchmark.display_name for task in adaptive_sampling_tasks
            ],
            "train_uniform_task_samples": train_stats["uniform_task_samples"],
            "train_adaptive_task_samples": train_stats["adaptive_task_samples"],
        }
        epoch_record["train_reg_loss" if args.proxy_signature_regression else "train_cls_loss"] = train_stats[
            "cls_loss"
        ]
        history.append(epoch_record)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        write_validation_epoch_summary_logs(
            log_dir=val_log_dir,
            tasks=val_tasks,
            epoch=epoch,
            val_loss=float(val_stats["loss"]),
            early_stopping_counter=early_stopping_counter,
        )
        write_validation_loss_log(
            log_dir=val_log_dir,
            epoch=epoch,
            val_loss=current_val_loss,
            early_stopping_counter=early_stopping_counter,
            best_val_loss=lowest_raw_val_loss,
        )
        write_validation_normalized_score_log(
            log_dir=val_log_dir,
            epoch=epoch,
            normalized_score=current_val_normalized_score,
            early_stopping_counter=early_stopping_counter,
            best_normalized_score=best_val_normalized_score,
        )

        if early_stopping_counter >= args.patience:
            print("Early stopping triggered.")
            break

    if not checkpoint_paths_by_epoch:
        raise RuntimeError("No checkpoint was saved.")

    if args.train_only:
        valid_log_plot_paths = plot_validation_loss_logs(val_log_dir)
        summary = {
            "best_checkpoint": str(best_checkpoint_path),
            "all_checkpoints": {
                str(epoch): str(path)
                for epoch, path in sorted(checkpoint_paths_by_epoch.items())
            },
            "num_epochs_ran": len(history),
            "best_val_loss": best_val_loss,
            "best_val_normalized_score": best_val_normalized_score,
            "best_epoch": best_epoch,
            "lowest_raw_val_loss": lowest_raw_val_loss,
            "lowest_raw_val_loss_epoch": lowest_raw_val_loss_epoch,
            "test_spearman_by_epoch_path": None,
            "test_spearman_by_epoch_json_path": None,
            "test_epoch_results": None,
            "test_log_plot_paths": None,
            "valid_loss_log_path": str(val_log_dir / "val_loss.txt"),
            "normalized_score_log_path": str(val_log_dir / "normalized_score.txt"),
            "valid_log_plot_paths": valid_log_plot_paths,
            "train_log_dir": str(train_log_dir),
            "valid_log_dir": str(val_log_dir),
            "test_log_dir": None,
            "train_only": True,
        }
        write_summary(run_dir, summary)
        print("Train-only mode enabled: skipped checkpoint test sweep.")
        print(f"Saved summary to {run_dir / 'summary.json'}")
        return 0

    test_sweep_stats = run_checkpoint_test_sweep(
        model=model,
        tasks=test_tasks,
        device=device,
        checkpoint_paths_by_epoch=checkpoint_paths_by_epoch,
        fixed_support_plans=fixed_test_support_plans,
        log_dir=test_log_dir,
        output_txt_path=run_dir / "test_spearman_by_epoch.txt",
        baseline_lookup=baseline_lookup,
    )
    valid_log_plot_paths = plot_validation_loss_logs(val_log_dir)
    test_log_plot_paths = plot_test_spearman_logs(test_log_dir)
    final_epoch = max(checkpoint_paths_by_epoch)
    retained_checkpoint_paths_by_epoch, removed_checkpoint_paths_by_epoch = prune_epoch_checkpoints(
        checkpoint_paths_by_epoch=checkpoint_paths_by_epoch,
        best_epoch=best_epoch,
        final_epoch=final_epoch,
    )
    retained_epoch_labels = ", ".join(f"{epoch:03d}" for epoch in sorted(retained_checkpoint_paths_by_epoch))
    removed_epoch_labels = ", ".join(f"{epoch:03d}" for epoch in sorted(removed_checkpoint_paths_by_epoch))
    print(
        f"[CHECKPOINT-CLEANUP] retained_epochs={retained_epoch_labels or 'none'} "
        f"removed_epochs={removed_epoch_labels or 'none'}"
    )

    summary = {
        "best_checkpoint": str(best_checkpoint_path),
        "all_checkpoints": {
            str(epoch): str(path)
            for epoch, path in sorted(retained_checkpoint_paths_by_epoch.items())
        },
        "removed_epoch_checkpoints": {
            str(epoch): str(path)
            for epoch, path in sorted(removed_checkpoint_paths_by_epoch.items())
        },
        "num_epochs_ran": len(history),
        "best_val_loss": best_val_loss,
        "best_val_normalized_score": best_val_normalized_score,
        "best_epoch": best_epoch,
        "final_epoch": final_epoch,
        "lowest_raw_val_loss": lowest_raw_val_loss,
        "lowest_raw_val_loss_epoch": lowest_raw_val_loss_epoch,
        "test_spearman_by_epoch_path": test_sweep_stats["spearman_by_epoch_path"],
        "test_spearman_by_epoch_json_path": test_sweep_stats["spearman_by_epoch_json_path"],
        "test_epoch_results": test_sweep_stats["epoch_results"],
        "best_test_epoch": test_sweep_stats["best_test_epoch"],
        "best_test_spearman_mean": test_sweep_stats["best_test_spearman_mean"],
        "test_log_plot_paths": test_log_plot_paths,
        "valid_loss_log_path": str(val_log_dir / "val_loss.txt"),
        "normalized_score_log_path": str(val_log_dir / "normalized_score.txt"),
        "valid_log_plot_paths": valid_log_plot_paths,
        "train_log_dir": str(train_log_dir),
        "valid_log_dir": str(val_log_dir),
        "test_log_dir": str(test_log_dir),
        "train_only": False,
    }
    write_summary(run_dir, summary)

    print(
        f"[Test sweep] result_path={test_sweep_stats['spearman_by_epoch_path']} "
        f"best_test_epoch={test_sweep_stats['best_test_epoch']} "
        f"best_test_spearman_mean={test_sweep_stats['best_test_spearman_mean']}"
    )
    print(f"Saved summary to {run_dir / 'summary.json'}")
    return 0
