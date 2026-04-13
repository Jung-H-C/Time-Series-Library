from __future__ import annotations

import random
from pathlib import Path

import torch

from .data_loader import TaskContext
from .engine import run_split_epoch
from .model import DSPBuilderMetaModel


def run_train_epoch(
    model: DSPBuilderMetaModel,
    tasks: list[TaskContext],
    device: torch.device,
    rng: random.Random,
    iterations_per_dataset: int,
    support_size: int,
    query_size: int,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    log_dir: Path,
    cls_loss_weight: float = 0.1,
) -> dict[str, float]:
    return run_split_epoch(
        model=model,
        tasks=tasks,
        device=device,
        rng=rng,
        iterations_per_dataset=iterations_per_dataset,
        support_size=support_size,
        query_size=query_size,
        optimizer=optimizer,
        epoch=epoch,
        stage_name="train",
        log_dir=log_dir,
        cls_loss_weight=cls_loss_weight,
    )
