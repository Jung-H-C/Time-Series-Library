from __future__ import annotations

import json
import random
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from utils.tools import EarlyStopping

from .baseline import load_spearman_baselines
from .data_loader import (
    BenchmarkTask,
    TaskContext,
    build_task_contexts,
    discover_benchmark_tasks,
    discover_candidate_configs,
    ensure_disjoint_splits,
    prompt_dataset_names,
    resolve_dataset_names,
    split_dataset_input,
)
from .engine import FixedEvaluationPlan
from .model import DSPBuilderMetaModel
from .test import run_test_epoch
from .train import run_train_epoch
from .valid import build_fixed_evaluation_plans, run_validation_epoch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_run_dir(base_dir: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"dspbuilder_meta_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def select_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw_device)


def task_names_for_logging(task_keys: list[str], available_tasks: dict[str, BenchmarkTask]) -> list[str]:
    return [available_tasks[key].display_name for key in task_keys]


def task_id_map_for_logging(
    dataset_class_ids: dict[str, int],
    available_tasks: dict[str, BenchmarkTask],
) -> dict[str, int]:
    return {available_tasks[key].display_name: dataset_class_ids[key] for key in dataset_class_ids}


def initialize_log_files(
    train_log_dir: Path,
    val_log_dir: Path,
    test_log_dir: Path,
    train_tasks: list[TaskContext],
    val_tasks: list[TaskContext],
    test_tasks: list[TaskContext],
    fixed_val_plans: dict[str, FixedEvaluationPlan],
    train_only: bool,
) -> None:
    train_log_dir.mkdir(parents=True, exist_ok=True)
    val_log_dir.mkdir(parents=True, exist_ok=True)

    log_specs = [
        (train_log_dir, train_tasks, "# Train iteration logs"),
        (val_log_dir, val_tasks, "# Validation iteration logs"),
    ]
    if not train_only:
        test_log_dir.mkdir(parents=True, exist_ok=True)
        log_specs.append((test_log_dir, test_tasks, "# Test Spearman correlation logs"))

    for log_dir, tasks, header in log_specs:
        for task in tasks:
            lines = [header]
            if log_dir == val_log_dir:
                plan = fixed_val_plans[task.benchmark.key]
                query_ranges = [f"{batch[0]}-{batch[-1]}" for batch in plan.query_batches]
                spearman_support_sets = [list(indices) for indices in plan.spearman_support_indices_sets]
                lines.append(f"# fixed_loss_support_indices={list(plan.loss_support_indices)}")
                lines.append(f"# fixed_spearman_support_indices={spearman_support_sets}")
                lines.append(f"# fixed_query_ranges={query_ranges}")
            (log_dir / f"{task.benchmark.display_name}.txt").write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8",
            )


def print_run_overview(
    available_tasks: dict[str, BenchmarkTask],
    train_tasks: list[TaskContext],
    val_tasks: list[TaskContext],
    test_tasks: list[TaskContext],
    fixed_val_plans: dict[str, FixedEvaluationPlan],
    train_dataset_class_ids: dict[str, int],
    device: torch.device,
    model: DSPBuilderMetaModel,
    run_dir: Path,
    train_only: bool,
) -> None:
    print("Available tasks:", ", ".join(task.display_name for task in available_tasks.values()))
    print("Train tasks:", ", ".join(task.benchmark.display_name for task in train_tasks))
    print("Val tasks:", ", ".join(task.benchmark.display_name for task in val_tasks))
    if train_only:
        print("Test tasks: skipped (--train-only)")
    else:
        print("Test tasks:", ", ".join(task.benchmark.display_name for task in test_tasks))
    print(f"Using device: {device}")
    print(
        f"raw_stat_emb={model.support_encoder.raw_stat_emb} "
        f"sample_embedding_dim={model.sample_embedding_dim} "
        f"task_embedding_dim={model.task_embedding_dim}"
    )
    print(
        "Train dataset ids:",
        ", ".join(
            f"{dataset_id}:{available_tasks[key].display_name}"
            for key, dataset_id in train_dataset_class_ids.items()
        ),
    )
    print(f"Run directory: {run_dir}")
    split_specs = [("train", train_tasks), ("val", val_tasks)]
    if not train_only:
        split_specs.append(("test", test_tasks))
    for split_name, tasks in split_specs:
        for task in tasks:
            print(
                f"[{split_name}] {task.benchmark.display_name}: "
                f"metric={task.benchmark.metric_name}, "
                f"sample_shape={task.sample_shape}, "
                f"train_examples={len(task.train_dataset)}"
            )
    for task in val_tasks:
        plan = fixed_val_plans[task.benchmark.key]
        query_ranges = ", ".join(f"{batch[0]}-{batch[-1]}" for batch in plan.query_batches)
        spearman_support_sets = ", ".join(str(list(indices)) for indices in plan.spearman_support_indices_sets)
        print(
            f"[valid-plan] {task.benchmark.display_name}: "
            f"loss_support_indices={list(plan.loss_support_indices)} "
            f"spearman_support_indices={spearman_support_sets} "
            f"query_ranges={query_ranges}"
        )


def write_summary(run_dir: Path, summary: dict[str, object]) -> None:
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_pipeline(args: Namespace) -> int:
    set_seed(args.seed)

    repo_root = Path(__file__).resolve().parent.parent
    benchmark_dir = args.benchmark_dir.resolve()
    candidate_dir = args.candidate_dir.resolve()
    available_tasks = discover_benchmark_tasks(benchmark_dir)
    candidate_configs = discover_candidate_configs(candidate_dir)
    baseline_lookup = load_spearman_baselines(repo_root / "benchmark" / "lookup" / "spearman_baseline.csv")

    train_keys = (
        resolve_dataset_names(split_dataset_input(args.train_datasets), available_tasks, "train")
        if args.train_datasets.strip()
        else prompt_dataset_names("train", available_tasks)
    )
    val_keys = (
        resolve_dataset_names(split_dataset_input(args.val_datasets), available_tasks, "val")
        if args.val_datasets.strip()
        else prompt_dataset_names("val", available_tasks)
    )
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
    ensure_disjoint_splits(train_keys, val_keys, test_keys)

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
    val_tasks = build_task_contexts(val_keys, available_tasks, candidate_configs, repo_root)
    test_tasks = build_task_contexts(test_keys, available_tasks, candidate_configs, repo_root)

    proxy_dim = len(train_tasks[0].benchmark.proxy_names)
    for task in train_tasks + val_tasks + test_tasks:
        if len(task.benchmark.proxy_names) != proxy_dim:
            raise ValueError("All benchmark CSVs must share the same proxy dimension.")

    device = select_device(args.device)
    model = DSPBuilderMetaModel(
        proxy_dim=proxy_dim,
        num_dataset_classes=len(train_dataset_class_ids),
        encoder_hidden_dim=args.encoder_hidden_dim,
        head_hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        raw_stat_emb=args.raw_stat_emb,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    run_dir = prepare_run_dir(args.output_dir.resolve())
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    best_checkpoint_path = run_dir / "best_checkpoint.pth"
    best_val_loss = float("inf")
    best_epoch = 0
    train_log_dir = run_dir / "train_logs"
    val_log_dir = run_dir / "valid_logs"
    test_log_dir = run_dir / "test_logs"
    fixed_val_plan_rng = random.Random(args.seed + 10_000)
    fixed_val_plans = build_fixed_evaluation_plans(
        val_tasks,
        support_size=args.support_size,
        query_size=args.val_query_size,
        iterations_per_dataset=args.val_iterations_per_dataset,
        rng=fixed_val_plan_rng,
    )

    config_payload = {
        "benchmark_dir": str(benchmark_dir),
        "candidate_dir": str(candidate_dir),
        "train_datasets": task_names_for_logging(train_keys, available_tasks),
        "val_datasets": task_names_for_logging(val_keys, available_tasks),
        "test_datasets": task_names_for_logging(test_keys, available_tasks),
        "epochs": args.epochs,
        "iterations_per_dataset": args.iterations_per_dataset,
        "val_iterations_per_dataset": args.val_iterations_per_dataset,
        "test_iterations_per_dataset": args.eval_iterations_per_dataset,
        "support_size": args.support_size,
        "train_query_size": args.train_query_size,
        "val_query_size": args.val_query_size,
        "test_query_size": args.test_query_size,
        "encoder_hidden_dim": args.encoder_hidden_dim,
        "hidden_dim": args.hidden_dim,
        "raw_stat_emb": args.raw_stat_emb,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "cls_loss_weight": args.cls_loss_weight,
        "patience": args.patience,
        "seed": args.seed,
        "device": str(device),
        "train_only": args.train_only,
        "train_dataset_class_ids": task_id_map_for_logging(train_dataset_class_ids, available_tasks),
    }
    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    initialize_log_files(
        train_log_dir=train_log_dir,
        val_log_dir=val_log_dir,
        test_log_dir=test_log_dir,
        train_tasks=train_tasks,
        val_tasks=val_tasks,
        test_tasks=test_tasks,
        fixed_val_plans=fixed_val_plans,
        train_only=args.train_only,
    )

    train_rng = random.Random(args.seed)
    eval_rng = random.Random(args.seed + 1)
    history: list[dict[str, float | int]] = []

    print_run_overview(
        available_tasks=available_tasks,
        train_tasks=train_tasks,
        val_tasks=val_tasks,
        test_tasks=test_tasks,
        fixed_val_plans=fixed_val_plans,
        train_dataset_class_ids=train_dataset_class_ids,
        device=device,
        model=model,
        run_dir=run_dir,
        train_only=args.train_only,
    )

    for epoch in range(1, args.epochs + 1):
        epoch_train_tasks = list(train_tasks)
        train_rng.shuffle(epoch_train_tasks)
        train_stats = run_train_epoch(
            model=model,
            tasks=epoch_train_tasks,
            device=device,
            rng=train_rng,
            iterations_per_dataset=args.iterations_per_dataset,
            support_size=args.support_size,
            query_size=args.train_query_size,
            optimizer=optimizer,
            epoch=epoch,
            log_dir=train_log_dir,
            cls_loss_weight=args.cls_loss_weight,
        )
        val_stats = run_validation_epoch(
            model=model,
            tasks=val_tasks,
            device=device,
            rng=eval_rng,
            iterations_per_dataset=args.val_iterations_per_dataset,
            support_size=args.support_size,
            query_size=args.val_query_size,
            epoch=epoch,
            log_dir=val_log_dir,
            fixed_plans=fixed_val_plans,
            baseline_lookup=baseline_lookup,
        )

        epoch_record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_pair_acc": train_stats["pair_acc"],
            "train_pair_loss_mean": train_stats["pair_loss_mean"],
            "train_cls_loss": train_stats["cls_loss"],
            "train_dataset_acc": train_stats["dataset_acc"],
            "val_loss": val_stats["loss"],
            "val_pair_acc": val_stats["pair_acc"],
            "val_pair_loss_mean": val_stats["pair_loss_mean"],
            "val_spearman_mean": float(val_stats["spearman_mean"]),
            "train_weight_norm": train_stats["weight_norm"],
            "val_weight_norm": val_stats["weight_norm"],
        }
        history.append(epoch_record)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_checkpoint_path)
            print(
                f"[BEST] epoch={epoch:03d} "
                f"val_loss={val_stats['loss']:.6f} "
                f"saved_checkpoint={best_checkpoint_path}"
            )

        early_stopping(val_stats["loss"], model, str(run_dir))
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    best_checkpoint = best_checkpoint_path if best_checkpoint_path.exists() else run_dir / "checkpoint.pth"
    if not best_checkpoint.exists():
        raise FileNotFoundError(f"Best checkpoint was not saved: {best_checkpoint}")

    if args.train_only:
        summary = {
            "best_checkpoint": str(best_checkpoint),
            "num_epochs_ran": len(history),
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "final_test_loss": None,
            "final_test_pair_acc": None,
            "final_test_pair_loss_mean": None,
            "final_test_spearman_mean": None,
            "final_test_spearman_std": None,
            "final_test_dataset_results": None,
            "train_log_dir": str(train_log_dir),
            "valid_log_dir": str(val_log_dir),
            "test_log_dir": None,
            "train_only": True,
        }
        write_summary(run_dir, summary)
        print("Train-only mode enabled: skipped test evaluation.")
        print(f"Saved summary to {run_dir / 'summary.json'}")
        return 0

    model.load_state_dict(torch.load(best_checkpoint, map_location=device))
    test_stats = run_test_epoch(
        model=model,
        tasks=test_tasks,
        device=device,
        rng=eval_rng,
        iterations_per_dataset=args.eval_iterations_per_dataset,
        support_size=args.support_size,
        query_size=args.test_query_size,
        epoch=best_epoch if best_epoch > 0 else len(history),
        log_dir=test_log_dir,
        baseline_lookup=baseline_lookup,
    )

    summary = {
        "best_checkpoint": str(best_checkpoint),
        "num_epochs_ran": len(history),
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "final_test_loss": None,
        "final_test_pair_acc": None,
        "final_test_pair_loss_mean": None,
        "final_test_spearman_mean": test_stats["spearman_mean"],
        "final_test_spearman_std": test_stats["spearman_std"],
        "final_test_dataset_results": test_stats["dataset_results"],
        "train_log_dir": str(train_log_dir),
        "valid_log_dir": str(val_log_dir),
        "test_log_dir": str(test_log_dir),
        "train_only": False,
    }
    write_summary(run_dir, summary)

    print(
        f"[Test] spearman_mean={test_stats['spearman_mean']:.6f} "
        f"spearman_std={test_stats['spearman_std']:.6f}"
    )
    print(f"Saved summary to {run_dir / 'summary.json'}")
    return 0
