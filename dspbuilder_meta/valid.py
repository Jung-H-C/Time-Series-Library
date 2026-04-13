from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from .baseline import SpearmanBaselineEntry
from .data_loader import TaskContext
from .engine import FixedEvaluationPlan, format_weight_vector, load_support_samples_from_indices, run_split_epoch
from .model import DSPBuilderMetaModel
from .test import compute_spearman_correlation, flip_spearman_for_lower_is_better_metric


def _sample_support_indices(
    train_population: list[int],
    support_size: int,
    rng: random.Random,
) -> tuple[int, ...]:
    if len(train_population) >= support_size:
        return tuple(sorted(rng.sample(train_population, support_size)))
    return tuple(sorted(rng.choice(train_population) for _ in range(support_size)))


def _sample_distinct_support_sets(
    train_population: list[int],
    support_size: int,
    num_support_sets: int,
    rng: random.Random,
) -> tuple[tuple[int, ...], ...]:
    support_sets: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    max_attempts = max(num_support_sets * 20, 100)
    attempts = 0

    while len(support_sets) < num_support_sets and attempts < max_attempts:
        sampled_indices = _sample_support_indices(train_population, support_size=support_size, rng=rng)
        attempts += 1
        if sampled_indices in seen:
            continue
        seen.add(sampled_indices)
        support_sets.append(sampled_indices)

    if len(support_sets) < num_support_sets:
        raise ValueError(
            "Could not generate the requested number of distinct fixed support index sets "
            f"(requested={num_support_sets}, generated={len(support_sets)})."
        )

    return tuple(support_sets)


def build_fixed_evaluation_plans(
    tasks: list[TaskContext],
    support_size: int,
    query_size: int,
    iterations_per_dataset: int,
    rng: random.Random,
    num_support_sets: int = 5,
) -> dict[str, FixedEvaluationPlan]:
    if support_size <= 0:
        raise ValueError("support_size must be positive.")
    if query_size < 2:
        raise ValueError("query_size must be at least 2 for pairwise ranking.")
    if iterations_per_dataset <= 0:
        raise ValueError("iterations_per_dataset must be positive.")
    if num_support_sets <= 0:
        raise ValueError("num_support_sets must be positive.")

    plans: dict[str, FixedEvaluationPlan] = {}
    required_candidates = query_size * iterations_per_dataset
    for task in tasks:
        train_population = list(range(len(task.train_dataset)))
        if not train_population:
            raise ValueError(f"Train dataset is empty for task: {task.benchmark.display_name}")

        support_index_sets = _sample_distinct_support_sets(
            train_population,
            support_size=support_size,
            num_support_sets=num_support_sets,
            rng=rng,
        )

        if task.benchmark.num_candidates < required_candidates:
            raise ValueError(
                f"Validation for '{task.benchmark.display_name}' requires at least "
                f"{required_candidates} benchmark candidates, found {task.benchmark.num_candidates}."
            )

        candidate_indices = list(range(required_candidates))
        query_batches = tuple(
            tuple(candidate_indices[start : start + query_size])
            for start in range(0, required_candidates, query_size)
        )
        plans[task.benchmark.key] = FixedEvaluationPlan(
            loss_support_indices=support_index_sets[0],
            spearman_support_indices_sets=support_index_sets,
            query_batches=query_batches,
        )
    return plans


def run_validation_epoch(
    model: DSPBuilderMetaModel,
    tasks: list[TaskContext],
    device: torch.device,
    rng: random.Random,
    iterations_per_dataset: int,
    support_size: int,
    query_size: int,
    epoch: int,
    log_dir: Path,
    fixed_plans: dict[str, FixedEvaluationPlan] | None = None,
    baseline_lookup: dict[str, SpearmanBaselineEntry] | None = None,
) -> dict[str, object]:
    loss_stats = run_split_epoch(
        model=model,
        tasks=tasks,
        device=device,
        rng=rng,
        iterations_per_dataset=iterations_per_dataset,
        support_size=support_size,
        query_size=query_size,
        optimizer=None,
        epoch=epoch,
        stage_name="valid",
        log_dir=log_dir,
        fixed_plans=fixed_plans,
    )
    spearman_results = run_validation_spearman_analysis(
        model=model,
        tasks=tasks,
        device=device,
        rng=rng,
        support_size=support_size,
        epoch=epoch,
        log_dir=log_dir,
        fixed_plans=fixed_plans,
        baseline_lookup=baseline_lookup,
    )
    return {
        **loss_stats,
        "spearman_mean": spearman_results["spearman_mean"],
        "dataset_spearman_results": spearman_results["dataset_results"],
    }


def _write_validation_spearman_log(
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
        f"[VALID-SPEARMAN] "
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
    dataset_log_path = log_dir / f"{dataset_name}.txt"
    with dataset_log_path.open("a", encoding="utf-8") as handle:
        handle.write(log_line + "\n")


def _write_validation_spearman_summary_log(
    log_dir: Path,
    epoch: int,
    dataset_name: str,
    metric_name: str,
    spearman_values: list[float],
    num_candidates: int,
    baseline_entry: SpearmanBaselineEntry | None = None,
) -> None:
    spearman_mean = float(np.mean(spearman_values)) if spearman_values else 0.0
    spearman_std = float(np.std(spearman_values)) if spearman_values else 0.0
    baseline_suffix = ""
    if baseline_entry is not None:
        baseline_suffix = (
            f" baseline_best_proxy={baseline_entry['best_proxy']} "
            f"baseline_coefficient={float(baseline_entry['coefficient']):.6f}"
        )
    log_line = (
        f"[VALID-SPEARMAN-SUMMARY] "
        f"epoch={epoch:03d} "
        f"dataset={dataset_name} "
        f"metric={metric_name} "
        f"spearman_mean={spearman_mean:.6f} "
        f"spearman_std={spearman_std:.6f} "
        f"support_sets={len(spearman_values)} "
        f"num_candidates={num_candidates}"
        f"{baseline_suffix}"
    )
    dataset_log_path = log_dir / f"{dataset_name}.txt"
    with dataset_log_path.open("a", encoding="utf-8") as handle:
        handle.write(log_line + "\n")


def run_validation_spearman_analysis(
    model: DSPBuilderMetaModel,
    tasks: list[TaskContext],
    device: torch.device,
    rng: random.Random,
    support_size: int,
    epoch: int,
    log_dir: Path,
    fixed_plans: dict[str, FixedEvaluationPlan] | None = None,
    baseline_lookup: dict[str, SpearmanBaselineEntry] | None = None,
) -> dict[str, object]:
    del rng
    del support_size

    model.eval()
    dataset_results: dict[str, dict[str, float | int | str]] = {}
    dataset_mean_spearman_values: list[float] = []

    with torch.no_grad():
        for task in tasks:
            if fixed_plans is None or task.benchmark.key not in fixed_plans:
                raise ValueError("Validation Spearman analysis requires fixed_plans for reproducible support sampling.")

            plan = fixed_plans[task.benchmark.key]
            baseline_entry = baseline_lookup.get(task.benchmark.key) if baseline_lookup is not None else None
            dataset_spearman_values: list[float] = []
            total_support_sets = len(plan.spearman_support_indices_sets)
            for support_set_index, support_indices_tuple in enumerate(plan.spearman_support_indices_sets, start=1):
                support_indices = list(support_indices_tuple)
                support_samples = load_support_samples_from_indices(task, indices=support_indices, device=device)
                weight_vector, _task_embedding, _dataset_logits = model(support_samples)
                predicted_proxy_scores = torch.matmul(task.benchmark.proxies.to(device), weight_vector)
                spearman_corr = flip_spearman_for_lower_is_better_metric(
                    compute_spearman_correlation(
                        predicted_proxy_scores,
                        task.benchmark.metrics.to(device),
                    )
                )
                weight_vector_list = [float(value) for value in weight_vector.detach().cpu().tolist()]
                _write_validation_spearman_log(
                    log_dir=log_dir,
                    epoch=epoch,
                    dataset_name=task.benchmark.display_name,
                    metric_name=task.benchmark.metric_name,
                    support_set_index=support_set_index,
                    total_support_sets=total_support_sets,
                    support_indices=support_indices,
                    weight_vector=weight_vector_list,
                    spearman_corr=spearman_corr,
                    num_candidates=task.benchmark.num_candidates,
                )
                dataset_spearman_values.append(spearman_corr)

            dataset_spearman_mean = float(np.mean(dataset_spearman_values)) if dataset_spearman_values else 0.0
            dataset_spearman_std = float(np.std(dataset_spearman_values)) if dataset_spearman_values else 0.0
            _write_validation_spearman_summary_log(
                log_dir=log_dir,
                epoch=epoch,
                dataset_name=task.benchmark.display_name,
                metric_name=task.benchmark.metric_name,
                spearman_values=dataset_spearman_values,
                num_candidates=task.benchmark.num_candidates,
                baseline_entry=baseline_entry,
            )
            dataset_results[task.benchmark.display_name] = {
                "dataset": task.benchmark.display_name,
                "metric_name": task.benchmark.metric_name,
                "spearman_corr": dataset_spearman_mean,
                "spearman_corr_std": dataset_spearman_std,
                "support_sets": total_support_sets,
                "num_candidates": task.benchmark.num_candidates,
                "baseline_best_proxy": baseline_entry["best_proxy"] if baseline_entry is not None else None,
                "baseline_coefficient": (
                    float(baseline_entry["coefficient"]) if baseline_entry is not None else None
                ),
            }
            baseline_suffix = ""
            if baseline_entry is not None:
                baseline_suffix = (
                    f" baseline_best_proxy={baseline_entry['best_proxy']} "
                    f"baseline_coefficient={float(baseline_entry['coefficient']):.6f}"
                )
            dataset_mean_spearman_values.append(dataset_spearman_mean)
            print(
                f"[VALID-SPEARMAN] dataset={task.benchmark.display_name} "
                f"metric={task.benchmark.metric_name} "
                f"spearman_mean={dataset_spearman_mean:.6f} "
                f"spearman_std={dataset_spearman_std:.6f} "
                f"support_sets={total_support_sets} "
                f"num_candidates={task.benchmark.num_candidates}"
                f"{baseline_suffix}",
                flush=True,
            )

    spearman_mean = float(np.mean(dataset_mean_spearman_values)) if dataset_mean_spearman_values else 0.0
    return {
        "spearman_mean": spearman_mean,
        "dataset_results": dataset_results,
    }
