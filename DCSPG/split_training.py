from __future__ import annotations

import csv
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from DCSPG.config import DCSPGConfig, MetaBatchConfig
from DCSPG.data import (
    Catch22FeatureStore,
    ClusterBalancedMetaBatchSampler,
    MetaBatch,
    sample_support_episode,
)
from DCSPG.dataset_partition import DatasetPartition, resolve_proxy_score_csv
from DCSPG.evaluate import (
    evaluate_rpn_tokens,
    generated_ids_to_rpn,
    load_proxy_test_values,
)
from DCSPG.experiment import build_training_components
from DCSPG.lodo_training import OptimizerConfig
from DCSPG.objectives import (
    autoregressive_cross_entropy_per_sequence,
    make_decoder_inputs,
)
from DCSPG.targets import GroundTruthFormulaTargetProvider
from DCSPG.trainer import DCSPGTrainer


@dataclass(frozen=True)
class SplitTrainingConfig:
    max_epochs: int = 100
    iterations_per_epoch: int = 100
    patience: int = 5
    validation_split: str = "proxy_test"
    test_split: str = "proxy_test"
    test_repeats: int = 10
    target_metric: str = "mse"
    invalid_spearman_penalty: float = -1.0
    proxy_score_decimals: int | None = None
    max_abs_proxy_score: float = 1e18
    checkpoint_keep: int = 5
    averaged_checkpoint_count: int = 3
    iteration_log_interval: int = 20
    validation_episodes_per_dataset: int = 5
    validation_ce_teacher_batch_size: int = 128
    early_stopping_criterion: str = "celoss"
    checkpoint_ranking_criterion: str = "spearman_corr"


@dataclass(frozen=True)
class SplitTrainResult:
    epochs: int
    steps: int
    best_epoch: int
    best_step: int
    early_stopping_criterion: str
    best_validation_criterion: float
    best_validation_spearman: float
    best_validation_weighted_ce: float
    final_train_loss: float
    stop_reason: str
    output_dir: str
    top_checkpoint_paths: tuple[str, ...]
    averaged_checkpoint_path: str
    averaged_checkpoint_count: int


@dataclass(frozen=True)
class RankedCheckpoint:
    criterion_name: str
    criterion_value: float
    selection_score: float
    validation_spearman: float
    validation_weighted_ce: float
    epoch: int
    step: int
    path: Path


@dataclass(frozen=True)
class ProxyBenchmark:
    dataset_name: str
    proxy_dataset_name: str
    csv_path: Path
    values: dict[str, list[float]]
    directed_target: list[float]
    split_count: int


BENCHMARK_PROXY_DATASETS = frozenset(
    {"ECL", "ETTh1", "Exchange", "ILI", "Traffic", "Weather"}
)

LEGACY_TOP_CHECKPOINT_FILENAMES = (
    "best_checkpoint.pth",
    "second_best_checkpoint.pth",
    "third_best_checkpoint.pth",
    "fourth_best_checkpoint.pth",
    "fifth_best_checkpoint.pth",
)


def top_checkpoint_filename(rank: int) -> str:
    if rank <= 0:
        raise ValueError("checkpoint rank must be positive")
    if rank <= len(LEGACY_TOP_CHECKPOINT_FILENAMES):
        return LEGACY_TOP_CHECKPOINT_FILENAMES[rank - 1]
    return f"rank_{rank:03d}_checkpoint.pth"


def validation_criterion_value(
    criterion_name: str,
    validation_spearman: float,
    validation_weighted_ce: float,
) -> tuple[float, float]:
    """Return (criterion value, higher-is-better selection score)."""
    if criterion_name == "spearman_corr":
        value = float(validation_spearman)
        return value, value
    if criterion_name == "celoss":
        value = float(validation_weighted_ce)
        return value, -value
    raise ValueError(
        "criterion_name must be one of "
        f"('spearman_corr', 'celoss'), got {criterion_name!r}"
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_fixed_support_batch(
    store: Catch22FeatureStore,
    dataset_names: tuple[str, ...],
    episodes_per_dataset: int,
    k_samples: int,
    stage: str,
) -> MetaBatch:
    """Build support episodes that are stable across epochs and training seeds."""
    if episodes_per_dataset <= 0:
        raise ValueError("episodes per dataset must be positive")
    if not stage:
        raise ValueError("fixed support stage must not be empty")

    inputs = []
    scheduled_names = []
    support_indices = []
    for dataset_name in dataset_names:
        for episode in range(1, episodes_per_dataset + 1):
            seed_material = (
                f"dcspg-fixed-support-v1:{stage}:{dataset_name}:{episode}"
            ).encode("utf-8")
            episode_seed = int.from_bytes(
                hashlib.sha256(seed_material).digest()[:8],
                byteorder="little",
                signed=False,
            )
            stats, indices = sample_support_episode(
                store,
                dataset_name,
                k_samples,
                np.random.default_rng(episode_seed),
            )
            inputs.append(stats)
            scheduled_names.append(dataset_name)
            support_indices.append(indices)

    return MetaBatch(
        inputs=torch.from_numpy(np.stack(inputs, axis=0)).float(),
        dataset_names=tuple(scheduled_names),
        support_indices=tuple(support_indices),
    )


def load_proxy_benchmarks(
    dataset_names: tuple[str, ...],
    partition: DatasetPartition,
    proxy_score_dir: Path | str,
    benchmark_dir: Path | str,
    split: str,
    target_metric: str,
) -> dict[str, ProxyBenchmark]:
    benchmarks = {}
    for dataset_name in dataset_names:
        proxy_dataset_name = partition.proxy_dataset_names[dataset_name]
        search_dir = (
            benchmark_dir
            if proxy_dataset_name in BENCHMARK_PROXY_DATASETS
            else proxy_score_dir
        )
        csv_path = resolve_proxy_score_csv(search_dir, proxy_dataset_name)
        values, directed_target, split_count = load_proxy_test_values(
            csv_path,
            split=split,
            target_metric=target_metric,
        )
        benchmarks[dataset_name] = ProxyBenchmark(
            dataset_name=dataset_name,
            proxy_dataset_name=proxy_dataset_name,
            csv_path=csv_path,
            values=values,
            directed_target=directed_target,
            split_count=split_count,
        )
    return benchmarks


def generate_rpn_batch(
    model: torch.nn.Module,
    batch: MetaBatch,
    vocabulary,
    device: torch.device,
    max_len: int,
) -> list[str]:
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            generated = model.generate(
                batch.inputs.to(device),
                bos_id=vocabulary.bos_id,
                eos_id=vocabulary.eos_id,
                pad_id=vocabulary.pad_id,
                max_len=max_len,
                greedy=True,
            )
    finally:
        model.train(was_training)
    return [
        generated_ids_to_rpn(token_ids.detach().cpu(), vocabulary)
        for token_ids in generated
    ]


def evaluate_generated_rpn(
    rpn_tokens: str,
    benchmark: ProxyBenchmark,
    config: SplitTrainingConfig,
) -> dict[str, object]:
    _scores, spearman, invalid_reason, infix, latex = evaluate_rpn_tokens(
        rpn_tokens,
        benchmark.values,
        benchmark.directed_target,
        config.proxy_score_decimals,
        config.max_abs_proxy_score,
    )
    score_for_mean = (
        float(spearman)
        if math.isfinite(spearman) and not invalid_reason
        else config.invalid_spearman_penalty
    )
    return {
        "rpn_tokens": rpn_tokens,
        "infix": infix,
        "latex": latex,
        "spearman_neg_mse": spearman,
        "score_for_mean": score_for_mean,
        "invalid_reason": invalid_reason,
    }


def evaluate_validation(
    model: torch.nn.Module,
    vocabulary,
    batch: MetaBatch,
    validation_datasets: tuple[str, ...],
    benchmarks: dict[str, ProxyBenchmark],
    device: torch.device,
    model_config: DCSPGConfig,
    training_config: SplitTrainingConfig,
    epoch: int,
    step: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], float]:
    generated_rpns = generate_rpn_batch(
        model,
        batch,
        vocabulary,
        device,
        model_config.max_formula_len,
    )
    formula_rows = []
    dataset_rows = []
    episode_counter: Counter[str] = Counter()
    for dataset_name, support_indices, rpn_tokens in zip(
        batch.dataset_names,
        batch.support_indices,
        generated_rpns,
    ):
        episode_counter[dataset_name] += 1
        benchmark = benchmarks[dataset_name]
        result = evaluate_generated_rpn(rpn_tokens, benchmark, training_config)
        formula_rows.append(
            {
                "epoch": epoch,
                "step": step,
                "dataset": dataset_name,
                "proxy_dataset": benchmark.proxy_dataset_name,
                "benchmark_csv": str(benchmark.csv_path),
                "split": training_config.validation_split,
                "split_count": benchmark.split_count,
                "episode": episode_counter[dataset_name],
                "support_indices": ";".join(
                    str(int(index)) for index in support_indices
                ),
                "generation_strategy": "greedy",
                **result,
            }
        )

    for dataset_name in validation_datasets:
        current_formula_rows = [
            row for row in formula_rows if row["dataset"] == dataset_name
        ]
        if len(current_formula_rows) != training_config.validation_episodes_per_dataset:
            raise RuntimeError(
                f"Expected {training_config.validation_episodes_per_dataset} validation "
                f"episodes for {dataset_name!r}, got {len(current_formula_rows)}"
            )
        dataset_mean = float(
            np.mean([float(row["score_for_mean"]) for row in current_formula_rows])
        )
        valid_formula_count = sum(
            not str(row["invalid_reason"]) for row in current_formula_rows
        )
        for row in current_formula_rows:
            row["dataset_formula_count"] = len(current_formula_rows)
            row["dataset_valid_formula_count"] = valid_formula_count
            row["dataset_mean_spearman_neg_mse"] = dataset_mean
        dataset_rows.append(
            {
                "epoch": epoch,
                "step": step,
                "dataset": dataset_name,
                "formula_count": len(current_formula_rows),
                "valid_formula_count": valid_formula_count,
                "mean_spearman_neg_mse": dataset_mean,
            }
        )
    mean_spearman = float(
        np.mean([float(row["mean_spearman_neg_mse"]) for row in dataset_rows])
    )
    return formula_rows, dataset_rows, mean_spearman


def evaluate_validation_weighted_ce(
    model: torch.nn.Module,
    vocabulary,
    grammar,
    target_provider: GroundTruthFormulaTargetProvider,
    batch: MetaBatch,
    validation_datasets: tuple[str, ...],
    device: torch.device,
    max_formula_len: int,
    teacher_batch_size: int,
) -> tuple[dict[str, float], float]:
    """Compute all-teacher weighted CE on fixed validation episodes.

    Every teacher formula is evaluated with teacher forcing against every
    fixed-support episode of its dataset. Teachers are chunked only to limit
    memory use; the final reduction is the exact normalized weighted mean.
    """
    if teacher_batch_size <= 0:
        raise ValueError("validation CE teacher batch size must be positive")

    was_training = model.training
    model.eval()
    dataset_losses: dict[str, float] = {}
    try:
        with torch.no_grad():
            for dataset_name in validation_datasets:
                episode_indices = [
                    index
                    for index, name in enumerate(batch.dataset_names)
                    if name == dataset_name
                ]
                if not episode_indices:
                    raise RuntimeError(
                        f"No fixed validation episodes found for {dataset_name!r}"
                    )

                episode_inputs = batch.inputs[episode_indices].to(device)
                encoder_output = model.encode_full(episode_inputs)
                episode_count = len(episode_indices)

                ground_truth_name = target_provider.resolve_dataset_name(dataset_name)
                ground_truth = target_provider.store[ground_truth_name]
                teacher_count = len(ground_truth.rpn_tokens)
                if teacher_count == 0:
                    raise RuntimeError(
                        f"Validation dataset {ground_truth_name!r} has no teacher formulas"
                    )
                total_weight = float(sum(ground_truth.weights))
                if not math.isfinite(total_weight) or total_weight <= 0.0:
                    raise ValueError(
                        f"Validation dataset {ground_truth_name!r} has invalid total "
                        f"teacher weight {total_weight}"
                    )

                weighted_loss_sums = torch.zeros(
                    episode_count,
                    dtype=torch.float64,
                    device=device,
                )
                for start in range(0, teacher_count, teacher_batch_size):
                    stop = min(start + teacher_batch_size, teacher_count)
                    encoded_teachers = []
                    for rpn_tokens in ground_truth.rpn_tokens[start:stop]:
                        token_ids = vocabulary.encode_rpn(rpn_tokens, strict=True)
                        if len(token_ids) > max_formula_len:
                            raise ValueError(
                                f"Validation target length {len(token_ids)} exceeds "
                                f"max_formula_len={max_formula_len}: {rpn_tokens}"
                            )
                        if grammar is not None and not grammar.is_valid_sequence(token_ids):
                            raise ValueError(
                                f"Invalid validation RPN target for "
                                f"{ground_truth_name}: {rpn_tokens}"
                            )
                        encoded_teachers.append(token_ids)

                    chunk_size = len(encoded_teachers)
                    target_len = max(len(token_ids) for token_ids in encoded_teachers)
                    teacher_targets = torch.full(
                        (chunk_size, target_len),
                        fill_value=vocabulary.pad_id,
                        dtype=torch.long,
                        device=device,
                    )
                    for teacher_index, token_ids in enumerate(encoded_teachers):
                        teacher_targets[teacher_index, : len(token_ids)] = torch.tensor(
                            token_ids,
                            dtype=torch.long,
                            device=device,
                        )

                    # Cartesian product: every fixed episode x every teacher in
                    # this chunk. Flattening keeps teachers contiguous per episode.
                    flat_targets = teacher_targets.unsqueeze(0).expand(
                        episode_count, -1, -1
                    ).reshape(episode_count * chunk_size, target_len)
                    decoder_inputs = make_decoder_inputs(
                        flat_targets,
                        bos_id=vocabulary.bos_id,
                    )
                    flat_context = encoder_output.context[:, None, :].expand(
                        -1, chunk_size, -1
                    ).reshape(episode_count * chunk_size, -1)
                    flat_memory = encoder_output.memory[:, None, :, :].expand(
                        -1, chunk_size, -1, -1
                    ).reshape(
                        episode_count * chunk_size,
                        encoder_output.memory.shape[1],
                        encoder_output.memory.shape[2],
                    )
                    logits = model.decoder(
                        decoder_input_ids=decoder_inputs,
                        context=flat_context,
                        memory=flat_memory,
                    )
                    if grammar is not None:
                        logits = grammar.mask_logits(logits, decoder_inputs)
                    sequence_losses = autoregressive_cross_entropy_per_sequence(
                        logits,
                        flat_targets,
                        pad_id=vocabulary.pad_id,
                    ).reshape(episode_count, chunk_size)
                    teacher_weights = torch.tensor(
                        ground_truth.weights[start:stop],
                        dtype=torch.float64,
                        device=device,
                    )
                    weighted_loss_sums += (
                        sequence_losses.to(torch.float64)
                        * teacher_weights.unsqueeze(0)
                    ).sum(dim=1)

                episode_weighted_means = weighted_loss_sums / total_weight
                dataset_loss = float(episode_weighted_means.mean().cpu())
                if not math.isfinite(dataset_loss):
                    raise RuntimeError(
                        f"Non-finite validation weighted CE for {dataset_name!r}"
                    )
                dataset_losses[dataset_name] = dataset_loss
    finally:
        model.train(was_training)

    macro_average = float(
        np.mean([dataset_losses[name] for name in validation_datasets])
    )
    return dataset_losses, macro_average


VALIDATION_FIELDNAMES = [
    "epoch",
    "step",
    "dataset",
    "proxy_dataset",
    "benchmark_csv",
    "split",
    "split_count",
    "episode",
    "support_indices",
    "generation_strategy",
    "rpn_tokens",
    "infix",
    "latex",
    "spearman_neg_mse",
    "score_for_mean",
    "invalid_reason",
    "dataset_formula_count",
    "dataset_valid_formula_count",
    "dataset_mean_spearman_neg_mse",
]

VALIDATION_DATASET_FIELDNAMES = [
    "epoch",
    "step",
    "dataset",
    "formula_count",
    "valid_formula_count",
    "mean_spearman_neg_mse",
]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_support_manifest(path: Path, batch: MetaBatch) -> None:
    episode_counter: Counter[str] = Counter()
    rows = []
    for dataset_name, support_indices in zip(
        batch.dataset_names, batch.support_indices
    ):
        episode_counter[dataset_name] += 1
        rows.append(
            {
                "dataset": dataset_name,
                "episode": episode_counter[dataset_name],
                "support_indices": ";".join(
                    str(int(index)) for index in support_indices
                ),
            }
        )
    write_csv(path, ["dataset", "episode", "support_indices"], rows)


def plot_epoch_metrics(
    output_path: Path,
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        return
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/tslib_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in rows]
    train_losses = [float(row["mean_train_loss"]) for row in rows]
    validation = [
        float(row["mean_validation_spearman_neg_mse"]) for row in rows
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, train_axis = plt.subplots(figsize=(10, 5.5))
    validation_axis = train_axis.twinx()
    train_line = train_axis.plot(
        epochs,
        train_losses,
        color="tab:blue",
        marker="o",
        linewidth=1.8,
        markersize=3.5,
        label="Mean train loss",
    )[0]
    validation_line = validation_axis.plot(
        epochs,
        validation,
        color="tab:orange",
        marker="s",
        linewidth=1.8,
        markersize=3.5,
        label="Validation Spearman",
    )[0]
    train_axis.set_xlabel("Epoch")
    train_axis.set_ylabel("Mean train loss", color="tab:blue")
    validation_axis.set_ylabel(
        "Mean validation Spearman (negative MSE)", color="tab:orange"
    )
    train_axis.grid(True, alpha=0.3)
    train_axis.legend(
        handles=[train_line, validation_line],
        loc="best",
    )
    fig.suptitle("DCSPG training and validation trajectory")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_validation_dataset_spearman(
    output_path: Path,
    rows: list[dict[str, object]],
    dataset_order: tuple[str, ...],
) -> None:
    """Plot the per-epoch validation Spearman history for every dataset."""
    if not rows:
        return
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/tslib_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(12, 6.5))
    for dataset_name in dataset_order:
        dataset_rows = [row for row in rows if row["dataset"] == dataset_name]
        axis.plot(
            [int(row["epoch"]) for row in dataset_rows],
            [float(row["mean_spearman_neg_mse"]) for row in dataset_rows],
            marker="o",
            linewidth=1.8,
            markersize=3.5,
            label=dataset_name,
        )
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean Spearman correlation (negative MSE)")
    axis.set_ylim(-1.05, 1.05)
    axis.set_title("Validation Spearman correlation by dataset")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_validation_weighted_ce(
    output_path: Path,
    rows: list[dict[str, object]],
    dataset_order: tuple[str, ...],
) -> None:
    """Plot dataset-wise and macro-average weighted validation CE."""
    if not rows:
        return
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/tslib_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in rows]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(12, 6.5))
    for dataset_name in dataset_order:
        axis.plot(
            epochs,
            [float(row[dataset_name]) for row in rows],
            marker="o",
            linewidth=1.5,
            markersize=3.0,
            label=dataset_name,
        )
    axis.plot(
        epochs,
        [float(row["macro_avg"]) for row in rows],
        color="black",
        linestyle="--",
        marker="s",
        linewidth=3.0,
        markersize=4.0,
        label="Macro Average",
        zorder=10,
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Weighted validation CE")
    axis.set_title("Dataset-wise Weighted Validation CE")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_epoch_checkpoint_and_prune(
    checkpoint_dir: Path,
    keep: int,
    **checkpoint_kwargs,
) -> Path:
    if keep <= 0:
        raise ValueError("checkpoint keep count must be positive")
    epoch = int(checkpoint_kwargs["epoch"])
    checkpoint_path = checkpoint_dir / f"epoch_{epoch:04d}.pth"
    save_checkpoint(checkpoint_path, **checkpoint_kwargs)
    epoch_paths = sorted(checkpoint_dir.glob("epoch_*.pth"))
    for stale_path in epoch_paths[:-keep]:
        stale_path.unlink()
    return checkpoint_path


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    vocabulary_tokens: tuple[str, ...],
    model_config: DCSPGConfig,
    meta_config: MetaBatchConfig,
    training_config: SplitTrainingConfig,
    optimizer_config: OptimizerConfig,
    partition: DatasetPartition,
    epoch: int,
    step: int,
    validation_spearman: float,
    validation_weighted_ce: float,
    validation_criterion_name: str,
    validation_criterion_value: float,
    train_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "train_datasets": partition.train_datasets,
            "validation_datasets": partition.validation_datasets,
            "test_datasets": partition.test_datasets,
            "cluster_datasets": partition.cluster_datasets,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocabulary_tokens": vocabulary_tokens,
            "model_config": asdict(model_config),
            "meta_config": asdict(meta_config),
            "training_config": asdict(training_config),
            "optimizer_config": asdict(optimizer_config),
            "validation_spearman": validation_spearman,
            "validation_weighted_ce": validation_weighted_ce,
            "validation_criterion_name": validation_criterion_name,
            "validation_criterion_value": validation_criterion_value,
            "train_loss": train_loss,
        },
        path,
    )


def _ranked_checkpoint_sort_key(record: RankedCheckpoint) -> tuple[float, int, int]:
    return (-record.selection_score, record.epoch, record.step)


def update_top_checkpoints(
    records: list[RankedCheckpoint],
    candidate_dir: Path,
    limit: int = 5,
    ranking_criterion_name: str = "spearman_corr",
    **checkpoint_kwargs,
) -> list[RankedCheckpoint]:
    if limit <= 0:
        raise ValueError("top checkpoint limit must be positive")
    validation_spearman = float(checkpoint_kwargs["validation_spearman"])
    validation_weighted_ce = float(checkpoint_kwargs["validation_weighted_ce"])
    early_stopping_criterion_name = str(
        checkpoint_kwargs["validation_criterion_name"]
    )
    early_stopping_criterion_value = float(
        checkpoint_kwargs["validation_criterion_value"]
    )
    expected_early_stopping_value, _ = validation_criterion_value(
        early_stopping_criterion_name,
        validation_spearman,
        validation_weighted_ce,
    )
    criterion_value, selection_score = validation_criterion_value(
        ranking_criterion_name,
        validation_spearman,
        validation_weighted_ce,
    )
    if not all(
        math.isfinite(value)
        for value in (
            validation_spearman,
            validation_weighted_ce,
            early_stopping_criterion_value,
            criterion_value,
            selection_score,
        )
    ):
        raise ValueError("top checkpoint validation metrics must be finite")
    if not math.isclose(
        early_stopping_criterion_value,
        expected_early_stopping_value,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Early-stopping validation criterion mismatch: "
            f"stored={early_stopping_criterion_value}, "
            f"expected={expected_early_stopping_value} for "
            f"{early_stopping_criterion_name}"
        )
    epoch = int(checkpoint_kwargs["epoch"])
    step = int(checkpoint_kwargs["step"])
    candidate = RankedCheckpoint(
        criterion_name=ranking_criterion_name,
        criterion_value=criterion_value,
        selection_score=selection_score,
        validation_spearman=validation_spearman,
        validation_weighted_ce=validation_weighted_ce,
        epoch=epoch,
        step=step,
        path=candidate_dir / f"epoch_{epoch:04d}_step_{step:08d}.pth",
    )
    ranked = sorted(records, key=_ranked_checkpoint_sort_key)
    if len(ranked) >= limit:
        worst_key = _ranked_checkpoint_sort_key(ranked[-1])
        if _ranked_checkpoint_sort_key(candidate) >= worst_key:
            return ranked

    save_checkpoint(candidate.path, **checkpoint_kwargs)
    ranked = sorted([*ranked, candidate], key=_ranked_checkpoint_sort_key)
    for dropped in ranked[limit:]:
        if dropped.path.exists():
            dropped.path.unlink()
    return ranked[:limit]


def finalize_top_checkpoints(
    records: list[RankedCheckpoint],
    output_dir: Path,
) -> list[RankedCheckpoint]:
    ranked = sorted(records, key=_ranked_checkpoint_sort_key)
    if not ranked:
        raise RuntimeError("No validation checkpoints were saved")
    finalized = []
    for rank, record in enumerate(ranked, start=1):
        filename = top_checkpoint_filename(rank)
        destination = output_dir / filename
        record.path.replace(destination)
        finalized.append(
            RankedCheckpoint(
                criterion_name=record.criterion_name,
                criterion_value=record.criterion_value,
                selection_score=record.selection_score,
                validation_spearman=record.validation_spearman,
                validation_weighted_ce=record.validation_weighted_ce,
                epoch=record.epoch,
                step=record.step,
                path=destination,
            )
        )
    candidate_dir = ranked[0].path.parent
    if candidate_dir.exists() and not any(candidate_dir.iterdir()):
        candidate_dir.rmdir()
    write_csv(
        output_dir / "top_checkpoints.csv",
        [
            "rank",
            "checkpoint_path",
            "epoch",
            "step",
            "criterion_name",
            "criterion_value",
            "validation_spearman",
            "validation_weighted_ce",
        ],
        [
            {
                "rank": rank,
                "checkpoint_path": str(record.path),
                "epoch": record.epoch,
                "step": record.step,
                "criterion_name": record.criterion_name,
                "criterion_value": record.criterion_value,
                "validation_spearman": record.validation_spearman,
                "validation_weighted_ce": record.validation_weighted_ce,
            }
            for rank, record in enumerate(finalized, start=1)
        ],
    )
    return finalized


def average_ranked_checkpoints(
    records: list[RankedCheckpoint],
    output_path: Path,
) -> dict[str, object]:
    if not records:
        raise ValueError("At least one checkpoint is required for weight averaging")
    checkpoint_count = len(records)
    first_checkpoint = torch.load(records[0].path, map_location="cpu")
    source_state = first_checkpoint["model_state_dict"]
    original_dtypes = {key: tensor.dtype for key, tensor in source_state.items()}
    averaged_state = {}
    for key, tensor in source_state.items():
        if torch.is_floating_point(tensor):
            averaged_state[key] = tensor.to(dtype=torch.float64)
        elif torch.is_complex(tensor):
            averaged_state[key] = tensor.to(dtype=torch.complex128)
        else:
            averaged_state[key] = tensor.clone()
    first_checkpoint.pop("optimizer_state_dict", None)

    expected_keys = tuple(source_state.keys())
    for record in records[1:]:
        checkpoint = torch.load(record.path, map_location="cpu")
        state = checkpoint["model_state_dict"]
        if tuple(state.keys()) != expected_keys:
            raise ValueError(f"Model state keys differ in {record.path}")
        for key, tensor in state.items():
            if torch.is_floating_point(tensor):
                averaged_state[key].add_(tensor.to(dtype=torch.float64))
            elif torch.is_complex(tensor):
                averaged_state[key].add_(tensor.to(dtype=torch.complex128))
            elif not torch.equal(averaged_state[key], tensor):
                raise ValueError(
                    f"Non-floating model state {key!r} differs in {record.path}"
                )

    for key, dtype in original_dtypes.items():
        if torch.is_floating_point(averaged_state[key]) or torch.is_complex(
            averaged_state[key]
        ):
            averaged_state[key].div_(checkpoint_count)
        averaged_state[key] = averaged_state[key].to(dtype=dtype)
    first_checkpoint["model_state_dict"] = averaged_state
    first_checkpoint["checkpoint_type"] = "uniform_weight_average"
    first_checkpoint["epoch"] = None
    first_checkpoint["step"] = None
    first_checkpoint["validation_spearman"] = None
    first_checkpoint["validation_weighted_ce"] = None
    first_checkpoint["validation_criterion_value"] = None
    first_checkpoint["train_loss"] = None
    first_checkpoint["averaged_from"] = tuple(str(record.path) for record in records)
    first_checkpoint["averaged_epochs"] = tuple(record.epoch for record in records)
    first_checkpoint["averaged_steps"] = tuple(record.step for record in records)
    first_checkpoint["averaged_validation_spearman"] = tuple(
        record.validation_spearman for record in records
    )
    first_checkpoint["averaged_validation_weighted_ce"] = tuple(
        record.validation_weighted_ce for record in records
    )
    first_checkpoint["averaged_validation_criterion_values"] = tuple(
        record.criterion_value for record in records
    )
    first_checkpoint["checkpoint_ranking_criterion"] = records[0].criterion_name
    first_checkpoint["averaging_weights"] = tuple(
        [1.0 / checkpoint_count] * checkpoint_count
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(first_checkpoint, output_path)
    return first_checkpoint


def evaluate_test_formula_average_ce(
    model: torch.nn.Module,
    vocabulary,
    batch: MetaBatch,
    test_datasets: tuple[str, ...],
    generated_rpns: list[str],
    device: torch.device,
    max_formula_len: int,
) -> list[float]:
    """Average each generated formula's CE over all dataset support episodes.

    For a dataset with E fixed test episodes and E generated formulas, this
    evaluates the E x E Cartesian product with teacher forcing. The returned
    list stays aligned with ``batch`` and ``generated_rpns``.
    """
    if len(generated_rpns) != len(batch.dataset_names):
        raise ValueError(
            f"Generated formula count {len(generated_rpns)} does not match "
            f"test batch size {len(batch.dataset_names)}"
        )

    grammar = getattr(model, "grammar", None)
    formula_average_ce = [float("nan")] * len(generated_rpns)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for dataset_name in test_datasets:
                dataset_indices = [
                    index
                    for index, name in enumerate(batch.dataset_names)
                    if name == dataset_name
                ]
                if not dataset_indices:
                    raise RuntimeError(
                        f"No fixed test episodes found for {dataset_name!r}"
                    )
                episode_count = len(dataset_indices)
                dataset_rpns = [generated_rpns[index] for index in dataset_indices]
                if len(dataset_rpns) != episode_count:
                    raise RuntimeError(
                        f"Expected one generated formula per test episode for "
                        f"{dataset_name!r}"
                    )

                encoded_formulas = []
                for rpn_tokens in dataset_rpns:
                    token_ids = vocabulary.encode_rpn(rpn_tokens, strict=True)
                    if len(token_ids) > max_formula_len:
                        raise ValueError(
                            f"Generated test target length {len(token_ids)} exceeds "
                            f"max_formula_len={max_formula_len}: {rpn_tokens}"
                        )
                    if grammar is not None and not grammar.is_valid_sequence(token_ids):
                        raise ValueError(
                            f"Invalid generated test RPN target for "
                            f"{dataset_name}: {rpn_tokens}"
                        )
                    encoded_formulas.append(token_ids)

                formula_count = len(encoded_formulas)
                target_len = max(len(token_ids) for token_ids in encoded_formulas)
                formula_targets = torch.full(
                    (formula_count, target_len),
                    fill_value=vocabulary.pad_id,
                    dtype=torch.long,
                    device=device,
                )
                for formula_index, token_ids in enumerate(encoded_formulas):
                    formula_targets[formula_index, : len(token_ids)] = torch.tensor(
                        token_ids,
                        dtype=torch.long,
                        device=device,
                    )

                # Cartesian product: every fixed test episode x every generated
                # formula from the same dataset. With defaults this is 10 x 10.
                flat_targets = formula_targets.unsqueeze(0).expand(
                    episode_count, -1, -1
                ).reshape(episode_count * formula_count, target_len)
                decoder_inputs = make_decoder_inputs(
                    flat_targets,
                    bos_id=vocabulary.bos_id,
                )
                encoder_output = model.encode_full(
                    batch.inputs[dataset_indices].to(device)
                )
                flat_context = encoder_output.context[:, None, :].expand(
                    -1, formula_count, -1
                ).reshape(episode_count * formula_count, -1)
                flat_memory = encoder_output.memory[:, None, :, :].expand(
                    -1, formula_count, -1, -1
                ).reshape(
                    episode_count * formula_count,
                    encoder_output.memory.shape[1],
                    encoder_output.memory.shape[2],
                )
                logits = model.decoder(
                    decoder_input_ids=decoder_inputs,
                    context=flat_context,
                    memory=flat_memory,
                )
                if grammar is not None:
                    logits = grammar.mask_logits(logits, decoder_inputs)
                sequence_losses = autoregressive_cross_entropy_per_sequence(
                    logits,
                    flat_targets,
                    pad_id=vocabulary.pad_id,
                ).reshape(episode_count, formula_count)
                dataset_formula_average_ce = sequence_losses.mean(dim=0).cpu()
                for result_index, average_ce in zip(
                    dataset_indices,
                    dataset_formula_average_ce.tolist(),
                ):
                    if not math.isfinite(average_ce):
                        raise RuntimeError(
                            f"Non-finite test Avg. CE for {dataset_name!r} "
                            f"at result index {result_index}"
                        )
                    formula_average_ce[result_index] = float(average_ce)
    finally:
        model.train(was_training)

    if any(not math.isfinite(value) for value in formula_average_ce):
        raise RuntimeError("Some generated test formulas are missing Avg. CE")
    return formula_average_ce


def run_test_stage(
    model: torch.nn.Module,
    vocabulary,
    batch: MetaBatch,
    partition: DatasetPartition,
    benchmarks: dict[str, ProxyBenchmark],
    output_dir: Path,
    device: torch.device,
    model_config: DCSPGConfig,
    training_config: SplitTrainingConfig,
    checkpoint_label: str,
    checkpoint_path: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    episode_counts = Counter(batch.dataset_names)
    unexpected_counts = {
        dataset_name: episode_counts[dataset_name]
        for dataset_name in partition.test_datasets
        if episode_counts[dataset_name] != training_config.test_repeats
    }
    if unexpected_counts:
        raise RuntimeError(
            f"Fixed test batch has unexpected episode counts: {unexpected_counts}"
        )
    generated_rpns = generate_rpn_batch(
        model,
        batch,
        vocabulary,
        device,
        model_config.max_formula_len,
    )
    formula_average_ce = evaluate_test_formula_average_ce(
        model=model,
        vocabulary=vocabulary,
        batch=batch,
        test_datasets=partition.test_datasets,
        generated_rpns=generated_rpns,
        device=device,
        max_formula_len=model_config.max_formula_len,
    )

    repeat_counter = {dataset_name: 0 for dataset_name in partition.test_datasets}
    result_rows = []
    for dataset_name, support_indices, rpn_tokens, average_ce in zip(
        batch.dataset_names,
        batch.support_indices,
        generated_rpns,
        formula_average_ce,
    ):
        repeat_counter[dataset_name] += 1
        benchmark = benchmarks[dataset_name]
        result = evaluate_generated_rpn(rpn_tokens, benchmark, training_config)
        result_rows.append(
            {
                "checkpoint": checkpoint_label,
                "checkpoint_path": str(checkpoint_path),
                "dataset": dataset_name,
                "proxy_dataset": benchmark.proxy_dataset_name,
                "repeat": repeat_counter[dataset_name],
                "benchmark_csv": str(benchmark.csv_path),
                "split": training_config.test_split,
                "split_count": benchmark.split_count,
                "support_indices": ";".join(str(int(index)) for index in support_indices),
                "Avg. CE": average_ce,
                **result,
            }
        )

    summary_rows = []
    for dataset_name in partition.test_datasets:
        dataset_rows = [row for row in result_rows if row["dataset"] == dataset_name]
        valid_count = sum(not str(row["invalid_reason"]) for row in dataset_rows)
        summary_rows.append(
            {
                "checkpoint": checkpoint_label,
                "checkpoint_path": str(checkpoint_path),
                "scope": "dataset",
                "dataset": dataset_name,
                "formula_count": len(dataset_rows),
                "valid_formula_count": valid_count,
                "mean_spearman_neg_mse": float(
                    np.mean([float(row["score_for_mean"]) for row in dataset_rows])
                ),
            }
        )
    summary_rows.append(
        {
            "checkpoint": checkpoint_label,
            "checkpoint_path": str(checkpoint_path),
            "scope": "overall_macro",
            "dataset": "ALL_TEST_DATASETS",
            "formula_count": len(result_rows),
            "valid_formula_count": sum(not str(row["invalid_reason"]) for row in result_rows),
            "mean_spearman_neg_mse": float(
                np.mean(
                    [
                        float(row["mean_spearman_neg_mse"])
                        for row in summary_rows
                    ]
                )
            ),
        }
    )

    write_csv(
        output_dir / f"test_results_{checkpoint_label}.csv",
        [
            "checkpoint",
            "checkpoint_path",
            "dataset",
            "proxy_dataset",
            "repeat",
            "benchmark_csv",
            "split",
            "split_count",
            "support_indices",
            "Avg. CE",
            "rpn_tokens",
            "infix",
            "latex",
            "spearman_neg_mse",
            "score_for_mean",
            "invalid_reason",
        ],
        result_rows,
    )
    write_csv(
        output_dir / f"test_summary_{checkpoint_label}.csv",
        [
            "checkpoint",
            "checkpoint_path",
            "scope",
            "dataset",
            "formula_count",
            "valid_formula_count",
            "mean_spearman_neg_mse",
        ],
        summary_rows,
    )
    return result_rows, summary_rows


def train_fixed_split(
    ts_feature_dir: Path | str,
    ground_truth_dir: Path | str,
    proxy_score_dir: Path | str,
    benchmark_dir: Path | str,
    output_dir: Path | str,
    partition: DatasetPartition,
    model_config: DCSPGConfig,
    meta_config: MetaBatchConfig,
    training_config: SplitTrainingConfig,
    optimizer_config: OptimizerConfig,
    seed: int,
    device: torch.device,
    target_sampling_strategy: str = "random",
) -> SplitTrainResult:
    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    components = build_training_components(
        ts_feature_dir=ts_feature_dir,
        ground_truth_dir=ground_truth_dir,
        model_config=model_config,
        seed=seed,
        target_sampling_strategy=target_sampling_strategy,
        targets_per_episode=meta_config.teachers_per_episode,
    )
    sampler = ClusterBalancedMetaBatchSampler(
        components.store,
        partition.cluster_datasets,
        config=meta_config,
        seed=seed + 1,
    )
    if set(sampler.train_dataset_names) != set(partition.train_datasets):
        raise RuntimeError("Sampler training datasets do not match the configured partition")

    validation_benchmarks = load_proxy_benchmarks(
        partition.validation_datasets,
        partition,
        proxy_score_dir,
        benchmark_dir,
        training_config.validation_split,
        training_config.target_metric,
    )
    test_benchmarks = load_proxy_benchmarks(
        partition.test_datasets,
        partition,
        proxy_score_dir,
        benchmark_dir,
        training_config.test_split,
        training_config.target_metric,
    )
    validation_batch = build_fixed_support_batch(
        components.store,
        partition.validation_datasets,
        training_config.validation_episodes_per_dataset,
        meta_config.k_samples,
        stage="validation",
    )
    test_batch = build_fixed_support_batch(
        components.store,
        partition.test_datasets,
        training_config.test_repeats,
        meta_config.k_samples,
        stage="test",
    )
    write_support_manifest(
        output_dir / "validation_support_samples.csv", validation_batch
    )
    write_support_manifest(output_dir / "test_support_samples.csv", test_batch)

    model = components.model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config.learning_rate,
        weight_decay=optimizer_config.weight_decay,
    )
    trainer = DCSPGTrainer(
        model=model,
        vocabulary=components.vocabulary,
        optimizer=optimizer,
        target_provider=components.target_provider,
        device=device,
        grammar=components.grammar,
        grad_clip=optimizer_config.grad_clip,
    )
    train_history = []
    validation_rows = []
    validation_dataset_history = []
    validation_summary = []
    validation_ce_history = []
    criterion_name = training_config.early_stopping_criterion
    checkpoint_ranking_criterion = training_config.checkpoint_ranking_criterion
    # Validate direct programmatic construction of SplitTrainingConfig as well
    # as CLI-created configurations.
    validation_criterion_value(criterion_name, 0.0, 0.0)
    validation_criterion_value(checkpoint_ranking_criterion, 0.0, 0.0)
    best_selection_score = float("-inf")
    best_criterion = float("nan")
    best_validation_spearman = float("-inf")
    best_validation_ce = float("inf")
    best_epoch = 0
    best_step = 0
    epochs_without_improvement = 0
    step = 0
    final_loss = float("nan")
    stop_reason = "max_epochs_reached"
    start_time = time.time()
    iteration_window_losses: list[float] = []
    iteration_window_dataset_counts: Counter[str] = Counter()
    top_checkpoint_records: list[RankedCheckpoint] = []
    top_checkpoint_candidate_dir = output_dir / ".top_checkpoint_candidates"

    for epoch in range(1, training_config.max_epochs + 1):
        epoch_losses = []
        for iteration in range(1, training_config.iterations_per_epoch + 1):
            step += 1
            train_batch = sampler.sample_train_batch()
            metrics = trainer.train_step(train_batch)
            final_loss = float(metrics["loss"])
            epoch_losses.append(final_loss)
            iteration_window_losses.append(final_loss)
            iteration_window_dataset_counts.update(train_batch.dataset_names)
            train_history.append(
                {
                    "epoch": epoch,
                    "iteration": iteration,
                    "step": step,
                    "loss": final_loss,
                    "unweighted_loss": metrics["unweighted_loss"],
                    "mean_teacher_weight": metrics["mean_teacher_weight"],
                    "target_len": metrics["target_len"],
                    "teachers_per_episode": metrics["teachers_per_episode"],
                    "elapsed_sec": time.time() - start_time,
                }
            )
            if step % training_config.iteration_log_interval == 0:
                selected_count = sum(iteration_window_dataset_counts.values())
                expected_count = (
                    training_config.iteration_log_interval * meta_config.batch_size
                )
                if selected_count != expected_count:
                    raise RuntimeError(
                        f"Expected {expected_count} dataset selections in the recent "
                        f"iteration window, got {selected_count}"
                    )
                counts_text = ", ".join(
                    f"{name}={count}"
                    for name, count in sorted(iteration_window_dataset_counts.items())
                )
                print(
                    f"step={step} recent_{training_config.iteration_log_interval}_iteration_"
                    f"mean_train_loss={np.mean(iteration_window_losses):.6f} "
                    f"dataset_counts=[{counts_text}]",
                    flush=True,
                )
                iteration_window_losses.clear()
                iteration_window_dataset_counts.clear()

        current_formula_rows, current_dataset_rows, validation_mean = evaluate_validation(
            model,
            components.vocabulary,
            validation_batch,
            partition.validation_datasets,
            validation_benchmarks,
            device,
            model_config,
            training_config,
            epoch,
            step,
        )
        validation_ce_by_dataset, validation_ce_macro = evaluate_validation_weighted_ce(
            model=model,
            vocabulary=components.vocabulary,
            grammar=components.grammar,
            target_provider=components.target_provider,
            batch=validation_batch,
            validation_datasets=partition.validation_datasets,
            device=device,
            max_formula_len=model_config.max_formula_len,
            teacher_batch_size=training_config.validation_ce_teacher_batch_size,
        )
        validation_ce_history.append(
            {
                "epoch": epoch,
                **validation_ce_by_dataset,
                "macro_avg": validation_ce_macro,
            }
        )
        validation_rows.extend(current_formula_rows)
        validation_dataset_history.extend(current_dataset_rows)
        best_validation_spearman = max(best_validation_spearman, validation_mean)
        best_validation_ce = min(best_validation_ce, validation_ce_macro)
        criterion_value, selection_score = validation_criterion_value(
            criterion_name,
            validation_mean,
            validation_ce_macro,
        )
        improved = selection_score > best_selection_score
        if improved:
            best_selection_score = selection_score
            best_criterion = criterion_value
            best_epoch = epoch
            best_step = step
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        top_checkpoint_records = update_top_checkpoints(
            top_checkpoint_records,
            top_checkpoint_candidate_dir,
            limit=training_config.averaged_checkpoint_count,
            ranking_criterion_name=checkpoint_ranking_criterion,
            model=model,
            optimizer=optimizer,
            vocabulary_tokens=components.vocabulary.tokens,
            model_config=model_config,
            meta_config=meta_config,
            training_config=training_config,
            optimizer_config=optimizer_config,
            partition=partition,
            epoch=epoch,
            step=step,
            validation_spearman=validation_mean,
            validation_weighted_ce=validation_ce_macro,
            validation_criterion_name=criterion_name,
            validation_criterion_value=criterion_value,
            train_loss=final_loss,
        )
        validation_summary.append(
            {
                "epoch": epoch,
                "step": step,
                "mean_validation_spearman_neg_mse": validation_mean,
                "best_validation_spearman_neg_mse": best_validation_spearman,
                "mean_validation_weighted_ce": validation_ce_macro,
                "best_validation_weighted_ce": best_validation_ce,
                "early_stopping_criterion": criterion_name,
                "validation_criterion_value": criterion_value,
                "best_validation_criterion_value": best_criterion,
                "improved": int(improved),
                "epochs_without_improvement": epochs_without_improvement,
                "mean_train_loss": float(np.mean(epoch_losses)),
            }
        )
        save_epoch_checkpoint_and_prune(
            output_dir / "checkpoints",
            training_config.checkpoint_keep,
            model=model,
            optimizer=optimizer,
            vocabulary_tokens=components.vocabulary.tokens,
            model_config=model_config,
            meta_config=meta_config,
            training_config=training_config,
            optimizer_config=optimizer_config,
            partition=partition,
            epoch=epoch,
            step=step,
            validation_spearman=validation_mean,
            validation_weighted_ce=validation_ce_macro,
            validation_criterion_name=criterion_name,
            validation_criterion_value=criterion_value,
            train_loss=final_loss,
        )
        write_csv(
            output_dir / "log" / "epoch_metrics.csv",
            [
                "epoch",
                "step",
                "mean_train_loss",
                "mean_validation_spearman_neg_mse",
                "best_validation_spearman_neg_mse",
                "mean_validation_weighted_ce",
                "best_validation_weighted_ce",
                "early_stopping_criterion",
                "validation_criterion_value",
                "best_validation_criterion_value",
                "improved",
                "epochs_without_improvement",
            ],
            validation_summary,
        )
        plot_epoch_metrics(
            output_dir / "log" / "train_validation_curve.png",
            validation_summary,
        )
        write_csv(
            output_dir / "validation_dataset_summary.csv",
            VALIDATION_DATASET_FIELDNAMES,
            validation_dataset_history,
        )
        plot_validation_dataset_spearman(
            output_dir / "log" / "validation_dataset_spearman_curve.png",
            validation_dataset_history,
            partition.validation_datasets,
        )
        write_csv(
            output_dir / "log" / "validation_weighted_ce.csv",
            ["epoch", *partition.validation_datasets, "macro_avg"],
            validation_ce_history,
        )
        plot_validation_weighted_ce(
            output_dir / "log" / "validation_weighted_ce_curve.png",
            validation_ce_history,
            partition.validation_datasets,
        )

        print(
            f"epoch={epoch} step={step} mean_train_loss={np.mean(epoch_losses):.6f}",
            flush=True,
        )
        for row in current_dataset_rows:
            print(
                f"  validation_dataset={row['dataset']} "
                f"greedy_episodes={int(row['formula_count'])} "
                f"valid_formulas={int(row['valid_formula_count'])} "
                f"mean_spearman_neg_mse={float(row['mean_spearman_neg_mse']):.6f}",
                flush=True,
            )
        print(
            f"  validation_spearman_macro={validation_mean:.6f} "
            f"best_spearman={best_validation_spearman:.6f}",
            flush=True,
        )
        print(f"[Validation CE] Epoch {epoch:03d}", flush=True)
        print(
            " | ".join(
                f"{dataset_name}: {validation_ce_by_dataset[dataset_name]:.4f}"
                for dataset_name in partition.validation_datasets
            ),
            flush=True,
        )
        print(f"Macro Average: {validation_ce_macro:.4f}", flush=True)
        print(
            f"  early_stopping_criterion={criterion_name} "
            f"current={criterion_value:.6f} best={best_criterion:.6f} "
            f"improved={int(improved)} "
            f"patience={epochs_without_improvement}/{training_config.patience}",
            flush=True,
        )
        if epochs_without_improvement >= training_config.patience:
            stop_reason = "early_stopping_patience_exhausted"
            break

    epochs_completed = validation_summary[-1]["epoch"] if validation_summary else 0
    final_validation_row = validation_summary[-1]
    save_checkpoint(
        path=output_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        vocabulary_tokens=components.vocabulary.tokens,
        model_config=model_config,
        meta_config=meta_config,
        training_config=training_config,
        optimizer_config=optimizer_config,
        partition=partition,
        epoch=int(epochs_completed),
        step=step,
        validation_spearman=float(
            final_validation_row["mean_validation_spearman_neg_mse"]
        ),
        validation_weighted_ce=float(
            final_validation_row["mean_validation_weighted_ce"]
        ),
        validation_criterion_name=criterion_name,
        validation_criterion_value=float(
            final_validation_row["validation_criterion_value"]
        ),
        train_loss=final_loss,
    )
    write_csv(
        output_dir / "train_history.csv",
        [
            "epoch",
            "iteration",
            "step",
            "loss",
            "unweighted_loss",
            "mean_teacher_weight",
            "target_len",
            "teachers_per_episode",
            "elapsed_sec",
        ],
        train_history,
    )
    write_csv(output_dir / "validation_results.csv", VALIDATION_FIELDNAMES, validation_rows)
    write_csv(
        output_dir / "validation_dataset_summary.csv",
        VALIDATION_DATASET_FIELDNAMES,
        validation_dataset_history,
    )
    write_csv(
        output_dir / "validation_summary.csv",
        [
            "epoch",
            "step",
            "mean_validation_spearman_neg_mse",
            "best_validation_spearman_neg_mse",
            "mean_validation_weighted_ce",
            "best_validation_weighted_ce",
            "early_stopping_criterion",
            "validation_criterion_value",
            "best_validation_criterion_value",
            "improved",
            "epochs_without_improvement",
            "mean_train_loss",
        ],
        validation_summary,
    )

    finalized_top_checkpoints = finalize_top_checkpoints(
        top_checkpoint_records,
        output_dir,
    )
    averaged_checkpoint_path = output_dir / "averaged_checkpoint.pth"
    averaged_checkpoint = average_ranked_checkpoints(
        finalized_top_checkpoints,
        averaged_checkpoint_path,
    )
    best_checkpoint_path = finalized_top_checkpoints[0].path
    best_checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
    model.load_state_dict(best_checkpoint["model_state_dict"])
    print(
        f"Testing with best single checkpoint: {best_checkpoint_path}",
        flush=True,
    )
    run_test_stage(
        model,
        components.vocabulary,
        test_batch,
        partition,
        test_benchmarks,
        output_dir,
        device,
        model_config,
        training_config,
        checkpoint_label="best_checkpoint",
        checkpoint_path=best_checkpoint_path,
    )

    model.load_state_dict(averaged_checkpoint["model_state_dict"])
    print(
        f"Testing with uniformly averaged checkpoint: {averaged_checkpoint_path} "
        f"sources={len(finalized_top_checkpoints)}",
        flush=True,
    )
    run_test_stage(
        model,
        components.vocabulary,
        test_batch,
        partition,
        test_benchmarks,
        output_dir,
        device,
        model_config,
        training_config,
        checkpoint_label="averaged_checkpoint",
        checkpoint_path=averaged_checkpoint_path,
    )

    result = SplitTrainResult(
        epochs=int(epochs_completed),
        steps=step,
        best_epoch=best_epoch,
        best_step=best_step,
        early_stopping_criterion=criterion_name,
        best_validation_criterion=best_criterion,
        best_validation_spearman=best_validation_spearman,
        best_validation_weighted_ce=best_validation_ce,
        final_train_loss=final_loss,
        stop_reason=stop_reason,
        output_dir=str(output_dir),
        top_checkpoint_paths=tuple(
            str(record.path) for record in finalized_top_checkpoints
        ),
        averaged_checkpoint_path=str(averaged_checkpoint_path),
        averaged_checkpoint_count=len(finalized_top_checkpoints),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(asdict(result), indent=2) + "\n",
        encoding="utf-8",
    )
    return result
