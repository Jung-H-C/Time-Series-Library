from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from .baseline import SpearmanBaselineEntry
from .data_loader import TaskContext
from .engine import format_weight_vector, load_support_samples_from_indices
from .model import DSPBuilderMetaModel


def _sample_support_indices(task: TaskContext, support_size: int, rng: random.Random) -> list[int]:
    if support_size <= 0:
        raise ValueError("support_size must be positive.")

    population = list(range(len(task.train_dataset)))
    if not population:
        raise ValueError(f"Test dataset is empty for task: {task.benchmark.display_name}")

    if len(population) >= support_size:
        return rng.sample(population, support_size)
    return [rng.choice(population) for _ in range(support_size)]


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)

    start = 0
    while start < sorted_values.shape[0]:
        end = start + 1
        while end < sorted_values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = ((start + end - 1) / 2.0) + 1.0
        ranks[order[start:end]] = average_rank
        start = end

    return ranks


def compute_spearman_correlation(predicted_scores: torch.Tensor, metrics: torch.Tensor) -> float:
    if predicted_scores.numel() != metrics.numel():
        raise ValueError("predicted_scores and metrics must have the same number of elements.")
    if predicted_scores.numel() < 2:
        raise ValueError("Spearman correlation requires at least 2 candidates.")

    predicted_np = predicted_scores.detach().cpu().numpy().astype(np.float64)
    metrics_np = metrics.detach().cpu().numpy().astype(np.float64)

    predicted_ranks = _average_ranks(predicted_np)
    metric_ranks = _average_ranks(metrics_np)

    predicted_centered = predicted_ranks - predicted_ranks.mean()
    metric_centered = metric_ranks - metric_ranks.mean()
    denominator = np.sqrt(
        float(np.dot(predicted_centered, predicted_centered))
        * float(np.dot(metric_centered, metric_centered))
    )
    if denominator == 0.0:
        return 0.0

    return float(np.dot(predicted_centered, metric_centered) / denominator)


def flip_spearman_for_lower_is_better_metric(spearman_corr: float) -> float:
    return -float(spearman_corr)


def _write_test_iteration_log(
    log_dir: Path,
    epoch: int,
    dataset_name: str,
    metric_name: str,
    iteration_index: int,
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
        f"iteration={iteration_index:03d} "
        f"spearman_corr={spearman_corr:.6f} "
        f"num_candidates={num_candidates} "
        f"support_indices={support_indices} "
        f"weight_norm={float(np.linalg.norm(weight_vector)):.6f} "
        f"weight_vector={format_weight_vector(weight_vector)}"
    )
    dataset_log_path = log_dir / f"{dataset_name}.txt"
    with dataset_log_path.open("a", encoding="utf-8") as handle:
        handle.write(log_line + "\n")


def _write_test_summary_log(
    log_dir: Path,
    epoch: int,
    dataset_name: str,
    metric_name: str,
    spearman_values: list[float],
    num_candidates: int,
    baseline_entry: SpearmanBaselineEntry | None = None,
) -> dict[str, float | int | str]:
    spearman_mean = float(np.mean(spearman_values)) if spearman_values else 0.0
    spearman_std = float(np.std(spearman_values)) if spearman_values else 0.0
    summary = {
        "dataset": dataset_name,
        "metric_name": metric_name,
        "spearman_mean": spearman_mean,
        "spearman_std": spearman_std,
        "iterations": len(spearman_values),
        "num_candidates": num_candidates,
        "baseline_best_proxy": baseline_entry["best_proxy"] if baseline_entry is not None else None,
        "baseline_coefficient": float(baseline_entry["coefficient"]) if baseline_entry is not None else None,
    }
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
        f"iterations={len(spearman_values)} "
        f"num_candidates={num_candidates}"
        f"{baseline_suffix}"
    )
    dataset_log_path = log_dir / f"{dataset_name}.txt"
    with dataset_log_path.open("a", encoding="utf-8") as handle:
        handle.write(log_line + "\n")
    return summary


def run_test_epoch(
    model: DSPBuilderMetaModel,
    tasks: list[TaskContext],
    device: torch.device,
    rng: random.Random,
    iterations_per_dataset: int,
    support_size: int,
    query_size: int,
    epoch: int,
    log_dir: Path,
    baseline_lookup: dict[str, SpearmanBaselineEntry] | None = None,
) -> dict[str, object]:
    del query_size

    model.eval()
    all_spearman_values: list[float] = []
    dataset_results: dict[str, dict[str, float | int | str]] = {}

    with torch.no_grad():
        for task in tasks:
            dataset_spearman_values: list[float] = []
            all_candidate_proxies = task.benchmark.proxies.to(device)
            all_candidate_metrics = task.benchmark.metrics.to(device)
            num_candidates = task.benchmark.num_candidates
            baseline_entry = baseline_lookup.get(task.benchmark.key) if baseline_lookup is not None else None

            for iteration_index in range(1, iterations_per_dataset + 1):
                support_indices = _sample_support_indices(task, support_size=support_size, rng=rng)
                support_samples = load_support_samples_from_indices(task, indices=support_indices, device=device)

                weight_vector, _task_embedding, _dataset_logits = model(support_samples)
                predicted_proxy_scores = torch.matmul(all_candidate_proxies, weight_vector)
                spearman_corr = flip_spearman_for_lower_is_better_metric(
                    compute_spearman_correlation(predicted_proxy_scores, all_candidate_metrics)
                )

                weight_vector_list = [float(value) for value in weight_vector.detach().cpu().tolist()]
                _write_test_iteration_log(
                    log_dir=log_dir,
                    epoch=epoch,
                    dataset_name=task.benchmark.display_name,
                    metric_name=task.benchmark.metric_name,
                    iteration_index=iteration_index,
                    support_indices=support_indices,
                    weight_vector=weight_vector_list,
                    spearman_corr=spearman_corr,
                    num_candidates=num_candidates,
                )
                dataset_spearman_values.append(spearman_corr)
                all_spearman_values.append(spearman_corr)

            dataset_summary = _write_test_summary_log(
                log_dir=log_dir,
                epoch=epoch,
                dataset_name=task.benchmark.display_name,
                metric_name=task.benchmark.metric_name,
                spearman_values=dataset_spearman_values,
                num_candidates=num_candidates,
                baseline_entry=baseline_entry,
            )
            dataset_results[task.benchmark.display_name] = dataset_summary
            baseline_suffix = ""
            if baseline_entry is not None:
                baseline_suffix = (
                    f" baseline_best_proxy={baseline_entry['best_proxy']} "
                    f"baseline_coefficient={float(baseline_entry['coefficient']):.6f}"
                )
            print(
                f"[TEST] dataset={task.benchmark.display_name} "
                f"metric={task.benchmark.metric_name} "
                f"spearman_mean={float(dataset_summary['spearman_mean']):.6f} "
                f"spearman_std={float(dataset_summary['spearman_std']):.6f} "
                f"iterations={int(dataset_summary['iterations'])} "
                f"num_candidates={int(dataset_summary['num_candidates'])}"
                f"{baseline_suffix}",
                flush=True,
            )

    overall_mean = float(np.mean(all_spearman_values)) if all_spearman_values else 0.0
    overall_std = float(np.std(all_spearman_values)) if all_spearman_values else 0.0
    return {
        "spearman_mean": overall_mean,
        "spearman_std": overall_std,
        "num_steps": len(all_spearman_values),
        "dataset_results": dataset_results,
    }
