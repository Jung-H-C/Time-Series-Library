from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from DCSPG.config import DCSPGConfig, MetaBatchConfig
from DCSPG.data import LODOMetaBatchSampler
from DCSPG.experiment import build_training_components
from DCSPG.trainer import DCSPGTrainer


@dataclass(frozen=True)
class TrainingBudgetConfig:
    max_steps: int = 5000
    check_every: int = 100
    checkpoint_interval: int = 100
    loss_window: int = 100
    loss_log_interval: int = 100


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0


@dataclass(frozen=True)
class FoldTrainResult:
    leave_out_dataset: str
    train_datasets: tuple[str, ...]
    steps: int
    best_step: int
    best_smoothed_loss: float
    final_loss: float
    stop_reason: str
    fold_dir: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_device_from_gpu_id(device: str, gpu_id: int | None) -> torch.device:
    if gpu_id is None:
        return resolve_device(device)
    if gpu_id < 0:
        raise ValueError("--gpu-id must be non-negative.")
    if not torch.cuda.is_available():
        raise RuntimeError("--gpu-id was provided, but CUDA is not available.")
    device_count = torch.cuda.device_count()
    if gpu_id >= device_count:
        raise ValueError(f"--gpu-id {gpu_id} is out of range; available CUDA devices: 0..{device_count - 1}.")
    return torch.device(f"cuda:{gpu_id}")


def sanitize_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


class BestLossTracker:
    def __init__(self, config: TrainingBudgetConfig) -> None:
        self.config = config
        self.best_smoothed_loss = float("inf")
        self.best_step = 0

    def update(self, step: int, losses: list[float]) -> tuple[float, bool]:
        smoothed_loss = float(np.mean(losses[-self.config.loss_window :]))
        improved = smoothed_loss < self.best_smoothed_loss
        if improved:
            self.best_smoothed_loss = smoothed_loss
            self.best_step = step
        return smoothed_loss, improved


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    leave_out_dataset: str,
    train_datasets: tuple[str, ...],
    vocabulary_tokens: tuple[str, ...],
    model_config: DCSPGConfig,
    meta_config: MetaBatchConfig,
    budget_config: TrainingBudgetConfig,
    optimizer_config: OptimizerConfig,
    target_sampling_strategy: str,
    smoothed_loss: float,
    raw_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "leave_out_dataset": leave_out_dataset,
            "train_datasets": train_datasets,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocabulary_tokens": vocabulary_tokens,
            "model_config": asdict(model_config),
            "meta_config": asdict(meta_config),
            "budget_config": asdict(budget_config),
            "optimizer_config": asdict(optimizer_config),
            "target_sampling_strategy": target_sampling_strategy,
            "smoothed_loss": smoothed_loss,
            "raw_loss": raw_loss,
        },
        path,
    )


def checkpoint_name(step: int) -> str:
    return f"step_{step:06d}.pt"


def write_loss_history(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["step", "loss", "smoothed_loss", "target_len", "elapsed_sec"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_interval_loss_history(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["step", "interval_mean_loss", "raw_loss", "elapsed_sec"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_interval_loss(path: Path, rows: list[dict[str, object]], title: str) -> None:
    if not rows:
        return

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/tslib_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [int(row["step"]) for row in rows]
    losses = [float(row["interval_mean_loss"]) for row in rows]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, losses, marker="o", linewidth=1.8, markersize=3.5)
    ax.set_title(title)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean train loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def train_one_lodo_fold(
    leave_out_dataset: str,
    ts_feature_dir: Path | str,
    ground_truth_dir: Path | str,
    output_dir: Path | str,
    model_config: DCSPGConfig,
    meta_config: MetaBatchConfig,
    budget_config: TrainingBudgetConfig,
    optimizer_config: OptimizerConfig,
    seed: int,
    device: torch.device,
    target_sampling_strategy: str = "cycle",
    log_every: int = 50,
) -> FoldTrainResult:
    set_seed(seed)
    components = build_training_components(
        ts_feature_dir=ts_feature_dir,
        ground_truth_dir=ground_truth_dir,
        model_config=model_config,
        seed=seed,
        target_sampling_strategy=target_sampling_strategy,
    )
    train_datasets = tuple(name for name in components.store.dataset_names if name != leave_out_dataset)
    sampler = LODOMetaBatchSampler(
        components.store,
        leave_out_dataset=leave_out_dataset,
        config=meta_config,
        seed=seed,
    )
    components.model.to(device)
    optimizer = torch.optim.AdamW(
        components.model.parameters(),
        lr=optimizer_config.learning_rate,
        weight_decay=optimizer_config.weight_decay,
    )
    trainer = DCSPGTrainer(
        model=components.model,
        vocabulary=components.vocabulary,
        optimizer=optimizer,
        target_provider=components.target_provider,
        device=device,
        grammar=components.grammar,
        grad_clip=optimizer_config.grad_clip,
    )

    fold_dir = Path(output_dir) / f"leave_out_{sanitize_name(leave_out_dataset)}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    best_tracker = BestLossTracker(budget_config)
    losses: list[float] = []
    history: list[dict[str, object]] = []
    interval_history: list[dict[str, object]] = []
    start_time = time.time()
    stop_reason = "max_steps_reached"
    smoothed_loss = float("inf")

    print(
        f"[{leave_out_dataset}] train_datasets={train_datasets} "
        f"batch_size={meta_config.batch_size} K={meta_config.k_samples}"
    )

    for step in range(1, budget_config.max_steps + 1):
        batch = sampler.sample_train_batch()
        metrics = trainer.train_step(batch)

        loss = float(metrics["loss"])
        losses.append(loss)
        if len(losses) >= budget_config.loss_window:
            smoothed_loss = float(np.mean(losses[-budget_config.loss_window :]))
        else:
            smoothed_loss = float(np.mean(losses))

        elapsed_sec = time.time() - start_time
        history.append(
            {
                "step": step,
                "loss": f"{loss:.8f}",
                "smoothed_loss": f"{smoothed_loss:.8f}",
                "target_len": f"{float(metrics.get('target_len', 0.0)):.1f}",
                "elapsed_sec": f"{elapsed_sec:.2f}",
            }
        )
        if step % budget_config.loss_log_interval == 0 or step == budget_config.max_steps:
            interval_losses = losses[-budget_config.loss_log_interval :]
            interval_mean_loss = float(np.mean(interval_losses))
            interval_history.append(
                {
                    "step": step,
                    "interval_mean_loss": f"{interval_mean_loss:.8f}",
                    "raw_loss": f"{loss:.8f}",
                    "elapsed_sec": f"{elapsed_sec:.2f}",
                }
            )

        if step % budget_config.checkpoint_interval == 0 or step == budget_config.max_steps:
            save_checkpoint(
                fold_dir / "checkpoints" / checkpoint_name(step),
                components.model,
                optimizer,
                step,
                leave_out_dataset,
                train_datasets,
                components.vocabulary.tokens,
                model_config,
                meta_config,
                budget_config,
                optimizer_config,
                target_sampling_strategy,
                smoothed_loss,
                loss,
            )

        should_check = step == 1 or step % budget_config.check_every == 0 or step == budget_config.max_steps
        if should_check:
            checked_loss, improved = best_tracker.update(step, losses)
            if improved:
                save_checkpoint(
                    fold_dir / "best.pt",
                    components.model,
                    optimizer,
                    step,
                    leave_out_dataset,
                    train_datasets,
                    components.vocabulary.tokens,
                    model_config,
                    meta_config,
                    budget_config,
                    optimizer_config,
                    target_sampling_strategy,
                    checked_loss,
                    loss,
                )
            if step % max(log_every, 1) == 0 or step == budget_config.max_steps:
                print(
                    f"[{leave_out_dataset}] step={step} loss={loss:.4f} "
                    f"smooth={checked_loss:.4f} best={best_tracker.best_smoothed_loss:.4f} "
                    f"reason={stop_reason if step == budget_config.max_steps else 'running'}"
                )
        elif step % max(log_every, 1) == 0:
            print(f"[{leave_out_dataset}] step={step} loss={loss:.4f} smooth={smoothed_loss:.4f}")

    final_step = len(losses)
    final_loss = losses[-1] if losses else float("nan")
    save_checkpoint(
        fold_dir / "last.pt",
        components.model,
        optimizer,
        final_step,
        leave_out_dataset,
        train_datasets,
        components.vocabulary.tokens,
        model_config,
        meta_config,
        budget_config,
        optimizer_config,
        target_sampling_strategy,
        smoothed_loss,
        final_loss,
    )
    write_loss_history(fold_dir / "loss_history.csv", history)
    write_interval_loss_history(fold_dir / "train_loss_interval.csv", interval_history)
    plot_interval_loss(
        fold_dir / "train_loss_interval.png",
        interval_history,
        title=f"DCSPG Train Loss - leave out {leave_out_dataset}",
    )

    result = FoldTrainResult(
        leave_out_dataset=leave_out_dataset,
        train_datasets=train_datasets,
        steps=final_step,
        best_step=best_tracker.best_step,
        best_smoothed_loss=float(best_tracker.best_smoothed_loss),
        final_loss=float(final_loss),
        stop_reason=stop_reason,
        fold_dir=str(fold_dir),
    )
    with (fold_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(result), handle, indent=2)
    return result


def write_lodo_summary(output_dir: Path | str, results: list[FoldTrainResult]) -> None:
    output_path = Path(output_dir) / "lodo_summary.csv"
    fieldnames = [
        "leave_out_dataset",
        "train_datasets",
        "steps",
        "best_step",
        "best_smoothed_loss",
        "final_loss",
        "stop_reason",
        "fold_dir",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            row["train_datasets"] = ";".join(result.train_datasets)
            writer.writerow(row)


def plot_lodo_interval_losses(output_dir: Path | str, results: list[FoldTrainResult]) -> None:
    rows_by_fold: dict[str, list[dict[str, str]]] = {}
    for result in results:
        csv_path = Path(result.fold_dir) / "train_loss_interval.csv"
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows_by_fold[result.leave_out_dataset] = list(csv.DictReader(handle))

    if not rows_by_fold:
        return

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/tslib_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    for leave_out_dataset, rows in rows_by_fold.items():
        if not rows:
            continue
        steps = [int(row["step"]) for row in rows]
        losses = [float(row["interval_mean_loss"]) for row in rows]
        ax.plot(steps, losses, marker="o", linewidth=1.6, markersize=3, label=leave_out_dataset)

    ax.set_title("DCSPG LODO Train Loss")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean train loss")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Leave-out dataset")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "lodo_train_loss_interval.png", dpi=160)
    plt.close(fig)
