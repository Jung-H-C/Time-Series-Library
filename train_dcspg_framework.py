from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

from DCSPG.config import DCSPGConfig, MetaBatchConfig
from DCSPG.data import Catch22FeatureStore
from DCSPG.dataset_partition import (
    DEFAULT_TEST_DATASETS,
    DEFAULT_VALIDATION_DATASETS,
    build_dataset_partition,
)
from DCSPG.lodo_training import OptimizerConfig, resolve_device_from_gpu_id
from DCSPG.split_training import SplitTrainingConfig, train_fixed_split


def create_run_output_dir(base_output_dir: Path, run_name: str | None = None) -> Path:
    base_output_dir.mkdir(parents=True, exist_ok=True)
    base_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if Path(base_name).name != base_name:
        raise ValueError("--run-name must be a single folder name, not a path")
    for suffix in range(1000):
        candidate_name = base_name if suffix == 0 else f"{base_name}_{suffix:03d}"
        candidate = base_output_dir / candidate_name
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create a unique run directory under {base_output_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one DCSPG model on a configurable 39/8/6 dataset split with "
            "cluster-balanced meta-batches, weighted multi-teacher CE, Spearman "
            "validation, early stopping, and repeated held-out testing."
        )
    )
    parser.add_argument("--ts-feature-dir", type=Path, default=Path("DCSPG/TS_dataset"))
    parser.add_argument("--ground-truth-dir", type=Path, default=Path("DCSPG/GroundTruth"))
    parser.add_argument(
        "--proxy-score-dir",
        type=Path,
        default=Path("proxy_scores/monash_time"),
        help="Proxy-score CSV directory for the 47 Monash/TIME datasets.",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("DCSPG/Benchmark"),
        help="MSE-enriched proxy-score CSV directory for the six benchmark datasets.",
    )
    parser.add_argument(
        "--cluster-file",
        type=Path,
        default=Path(
            "catch22/dataset_centroid_clusters_47_pca90_k8/cluster_summary_k4.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("DCSPG/checkpoints/fixed_split"),
    )
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument(
        "--validation-datasets",
        type=str,
        default=",".join(DEFAULT_VALIDATION_DATASETS),
        help=(
            "Comma-separated validation dataset names; exactly 8 are required "
            "and every cluster must be represented."
        ),
    )
    parser.add_argument(
        "--test-datasets",
        type=str,
        default=",".join(DEFAULT_TEST_DATASETS),
        help="Comma-separated test dataset names; exactly 6 are required.",
    )

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument(
        "--target-sampling-strategy",
        choices=("cycle", "random"),
        default="random",
        help=(
            "Single-target compatibility mode. With 16 teachers per episode, "
            "teachers are always sampled uniformly without replacement."
        ),
    )

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--base-episodes-per-cluster", type=int, default=64)
    parser.add_argument(
        "--extra-cluster-episodes",
        type=int,
        default=0,
        help=(
            "Number of distinct clusters sampled for one extra episode per batch. "
            "With four training clusters, the defaults use exactly 64 episodes "
            "from each cluster."
        ),
    )
    parser.add_argument("--k-samples", type=int, default=16)
    parser.add_argument("--teachers-per-episode", type=int, default=16)

    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=1)
    parser.add_argument("--encoder-layers", type=int, default=1)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-formula-len", type=int, default=12)
    parser.add_argument("--max-stack-depth", type=int, default=4)
    parser.add_argument(
        "--max-unary-chain",
        type=int,
        default=2,
        help=(
            "Maximum number of consecutive unary operators allowed in every "
            "training and generation RPN sequence."
        ),
    )

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--iterations-per-epoch", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--early-stopping-criterion",
        "--validation-criterion",
        dest="early_stopping_criterion",
        type=str.lower,
        choices=("spearman_corr", "celoss"),
        default="celoss",
        help=(
            "Validation criterion used only for early stopping. spearman_corr is "
            "maximized; celoss is minimized. Checkpoints used for weight averaging "
            "are ranked separately by validation Spearman correlation."
        ),
    )
    parser.add_argument("--validation-split", type=str, default="proxy_test")
    parser.add_argument("--test-split", type=str, default="proxy_test")
    parser.add_argument("--test-repeats", type=int, default=10)
    parser.add_argument("--invalid-spearman-penalty", type=float, default=-1.0)
    parser.add_argument(
        "--validation-episodes-per-dataset",
        type=int,
        default=5,
        help=(
            "Number of fixed-support greedy episodes generated for each "
            "validation dataset."
        ),
    )
    parser.add_argument(
        "--validation-ce-teacher-batch-size",
        type=int,
        default=256,
        help=(
            "Teacher-formula chunk size for all-teacher validation CE. "
            "This affects memory use only, not the weighted CE result."
        ),
    )
    parser.add_argument("--checkpoint-keep", type=int, default=5)
    parser.add_argument(
        "--averaged-checkpoint-count",
        type=int,
        default=3,
        help=(
            "Number of top validation-Spearman checkpoints to retain and uniformly "
            "weight-average for the test stage."
        ),
    )
    parser.add_argument("--iteration-log-interval", type=int, default=20)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "batch-size": args.batch_size,
        "base-episodes-per-cluster": args.base_episodes_per_cluster,
        "k-samples": args.k_samples,
        "teachers-per-episode": args.teachers_per_episode,
        "max-formula-len": args.max_formula_len,
        "max-stack-depth": args.max_stack_depth,
        "max-unary-chain": args.max_unary_chain,
        "learning-rate": args.learning_rate,
        "max-epochs": args.max_epochs,
        "iterations-per-epoch": args.iterations_per_epoch,
        "patience": args.patience,
        "test-repeats": args.test_repeats,
        "validation-episodes-per-dataset": args.validation_episodes_per_dataset,
        "validation-ce-teacher-batch-size": args.validation_ce_teacher_batch_size,
        "checkpoint-keep": args.checkpoint_keep,
        "averaged-checkpoint-count": args.averaged_checkpoint_count,
        "iteration-log-interval": args.iteration_log_interval,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"These arguments must be positive: {invalid}")
    if args.extra_cluster_episodes < 0:
        raise ValueError("--extra-cluster-episodes must be non-negative")
    if not -1.0 <= args.invalid_spearman_penalty <= 1.0:
        raise ValueError("--invalid-spearman-penalty must be in [-1, 1]")


def validate_batch_composition(
    args: argparse.Namespace,
    cluster_count: int,
) -> None:
    if args.extra_cluster_episodes > cluster_count:
        raise ValueError(
            "--extra-cluster-episodes must not exceed the number of non-empty "
            f"training clusters ({cluster_count})"
        )
    expected_batch = (
        cluster_count * args.base_episodes_per_cluster
        + args.extra_cluster_episodes
    )
    if args.batch_size != expected_batch:
        raise ValueError(
            "--batch-size must equal cluster_count * "
            "--base-episodes-per-cluster + --extra-cluster-episodes: "
            f"{cluster_count} * {args.base_episodes_per_cluster} + "
            f"{args.extra_cluster_episodes} = {expected_batch}"
        )


def main() -> int:
    args = build_arg_parser().parse_args()
    validate_args(args)
    device = resolve_device_from_gpu_id(args.device, args.gpu_id)

    store = Catch22FeatureStore(args.ts_feature_dir)
    partition = build_dataset_partition(
        available_ts_names=store.dataset_names,
        validation_datasets=args.validation_datasets,
        test_datasets=args.test_datasets,
        cluster_csv=args.cluster_file,
    )
    validate_batch_composition(args, len(partition.cluster_datasets))
    model_config = DCSPGConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_formula_len=args.max_formula_len,
        max_stack_depth=args.max_stack_depth,
        max_unary_chain=args.max_unary_chain,
    )
    meta_config = MetaBatchConfig(
        batch_size=args.batch_size,
        k_samples=args.k_samples,
        base_episodes_per_cluster=args.base_episodes_per_cluster,
        extra_cluster_episodes=args.extra_cluster_episodes,
        teachers_per_episode=args.teachers_per_episode,
    )
    training_config = SplitTrainingConfig(
        max_epochs=args.max_epochs,
        iterations_per_epoch=args.iterations_per_epoch,
        patience=args.patience,
        early_stopping_criterion=args.early_stopping_criterion,
        checkpoint_ranking_criterion="spearman_corr",
        validation_split=args.validation_split,
        test_split=args.test_split,
        test_repeats=args.test_repeats,
        invalid_spearman_penalty=args.invalid_spearman_penalty,
        validation_episodes_per_dataset=args.validation_episodes_per_dataset,
        validation_ce_teacher_batch_size=args.validation_ce_teacher_batch_size,
        checkpoint_keep=args.checkpoint_keep,
        averaged_checkpoint_count=args.averaged_checkpoint_count,
        iteration_log_interval=args.iteration_log_interval,
    )
    optimizer_config = OptimizerConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
    )

    run_output_dir = create_run_output_dir(args.output_dir, args.run_name.strip() or None)
    run_config = {
        "ts_feature_dir": str(args.ts_feature_dir),
        "ground_truth_dir": str(args.ground_truth_dir),
        "proxy_score_dir": str(args.proxy_score_dir),
        "benchmark_dir": str(args.benchmark_dir),
        "cluster_file": str(args.cluster_file),
        "run_output_dir": str(run_output_dir),
        "device": str(device),
        "gpu_id": args.gpu_id,
        "seed": args.seed,
        "target_sampling_strategy": args.target_sampling_strategy,
        "train_datasets": partition.train_datasets,
        "validation_datasets": partition.validation_datasets,
        "test_datasets": partition.test_datasets,
        "cluster_datasets": partition.cluster_datasets,
        "model_config": asdict(model_config),
        "meta_config": asdict(meta_config),
        "training_config": asdict(training_config),
        "optimizer_config": asdict(optimizer_config),
    }
    (run_output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"DCSPG fixed-split training on device={device}")
    print(
        f"datasets: train={len(partition.train_datasets)} "
        f"validation={len(partition.validation_datasets)} test={len(partition.test_datasets)}"
    )
    print(
        "train clusters: "
        + ", ".join(
            f"cluster_{cluster_id}={len(names)}"
            for cluster_id, names in partition.cluster_datasets.items()
        )
    )
    print(f"output: {run_output_dir}")

    result = train_fixed_split(
        ts_feature_dir=args.ts_feature_dir,
        ground_truth_dir=args.ground_truth_dir,
        proxy_score_dir=args.proxy_score_dir,
        benchmark_dir=args.benchmark_dir,
        output_dir=run_output_dir,
        partition=partition,
        model_config=model_config,
        meta_config=meta_config,
        training_config=training_config,
        optimizer_config=optimizer_config,
        seed=args.seed,
        device=device,
        target_sampling_strategy=args.target_sampling_strategy,
    )
    print(json.dumps(asdict(result), indent=2))
    print(f"Validation history: {run_output_dir / 'validation_summary.csv'}")
    print(
        "Validation dataset Spearman curve: "
        f"{run_output_dir / 'log' / 'validation_dataset_spearman_curve.png'}"
    )
    print(
        "Validation weighted CE history: "
        f"{run_output_dir / 'log' / 'validation_weighted_ce.csv'}"
    )
    print(
        "Validation weighted CE curve: "
        f"{run_output_dir / 'log' / 'validation_weighted_ce_curve.png'}"
    )
    print(
        "Best-checkpoint test results: "
        f"{run_output_dir / 'test_results_best_checkpoint.csv'}"
    )
    print(
        "Averaged-checkpoint test results: "
        f"{run_output_dir / 'test_results_averaged_checkpoint.csv'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
