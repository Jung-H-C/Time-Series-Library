from __future__ import annotations

import argparse
from pathlib import Path

from dspbuilder_meta.new_pipeline import run_new_pipeline

# conda run -n tslib python train_new_dspbuilder_meta.py \
#   --train-datasets "ECL,Traffic,Weather" \
#   --test-datasets "M1_Yearly" \
#   --epochs 100 \
#   --iterations-per-epoch 40 \
#   --train-batch-size 16 \
#   --support-size 5 \
#   --train-query-size 20 \
#   --device cuda:0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train DSPBuilder meta model with identical train/validation dataset subsets "
            "and a fixed candidate-row train/validation split inside each benchmark CSV."
        )
    )
    parser.add_argument("--benchmark-dir", type=Path, default=Path("./benchmark"))
    parser.add_argument("--candidate-dir", type=Path, default=Path("./candidates"))
    parser.add_argument("--train-datasets", type=str, default="", help="Comma-separated dataset names.")
    parser.add_argument(
        "--val-datasets",
        type=str,
        # 필수아님, 빈칸으로 두면 training으로 같이 돌아감
        default="",
        help="Optional compatibility argument. If provided, it must match --train-datasets.",
    )
    parser.add_argument("--test-datasets", type=str, default="", help="Comma-separated dataset names.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--iterations-per-epoch",
        type=int,
        default=40,
        help="Number of supervised mini-batch optimizer updates per training epoch.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=16,
        help="Number of dataset-level tasks sampled per training mini-batch.",
    )
    parser.add_argument(
        "--val-iterations-per-dataset",
        type=int,
        default=1,
        help="Kept for CLI compatibility; validation uses the fixed held-out candidate rows once per dataset.",
    )
    parser.add_argument(
        "--eval-iterations-per-dataset",
        type=int,
        default=5,
        help="Number of fixed support sets used for each checkpoint's final test Spearman sweep.",
    )
    parser.add_argument(
        "--candidate-train-count",
        type=int,
        default=80,
        help="Number of benchmark rows used as the train candidate pool.",
    )
    parser.add_argument(
        "--candidate-val-count",
        type=int,
        default=20,
        help="Number of benchmark rows used as the validation candidate pool.",
    )
    parser.add_argument(
        "--stratified",
        dest="stratified",
        action="store_true",
        default=False,
        help=(
            "Sort each benchmark CSV by ascending metric (lower is better), split the ranked rows "
            "into 10 groups, and draw a fixed train/validation split from every group."
        ),
    )
    parser.add_argument(
        "--no-stratified",
        dest="stratified",
        action="store_false",
        help="Disable grouped metric-rank stratified candidate splitting.",
    )
    parser.add_argument("--support-size", type=int, default=5)
    parser.add_argument("--train-query-size", type=int, default=20)
    parser.add_argument(
        "--val-query-size",
        type=int,
        default=20,
        help="Kept for CLI compatibility; validation uses --candidate-val-count rows.",
    )
    parser.add_argument(
        "--test-query-size",
        # 이 코드에서는 사용되지 않음, 모든 candidate를 test로 사용함
        type=int,
        default=10,
        help="Kept for CLI compatibility; checkpoint test Spearman uses all candidate rows.",
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument(
        "--weight-head-layers",
        type=int,
        default=1,
        help="Number of hidden Linear layers in the dataset-description-to-proxy-weight MLP.",
    )
    parser.add_argument(
        "--mlp_norm",
        "--mlp-norm",
        dest="mlp_norm",
        action="store_true",
        default=False,
        help="Insert LayerNorm between each weight-head Linear layer and ReLU.",
    )
    parser.add_argument(
        "--no-mlp_norm",
        "--no-mlp-norm",
        dest="mlp_norm",
        action="store_false",
        help="Disable LayerNorm layers in the weight-head MLP.",
    )
    parser.add_argument("--encoder-hidden-dim", type=int, default=64)
    parser.add_argument(
        "--number-of-conv1d-layer",
        "--number_of_conv1d_layer",
        dest="number_of_conv1d_layer",
        type=int,
        default=1,
        help="Number of additional sample-encoder Conv1d layers with kernel_size=5 after the first kernel_size=7 layer.",
    )
    parser.add_argument(
        "--sample_encoder_norm",
        "--sample-encoder-norm",
        dest="sample_encoder_norm",
        action="store_true",
        default=False,
        help="Insert GroupNorm(num_groups=8) between each sample-encoder Conv1d and GELU.",
    )
    parser.add_argument(
        "--no-sample_encoder_norm",
        "--no-sample-encoder-norm",
        dest="sample_encoder_norm",
        action="store_false",
        help="Disable GroupNorm layers in the sample encoder.",
    )
    parser.add_argument(
        "--number-of-setencoder-mlp-layers",
        "--number_of_setencoder_mlp_layers",
        dest="number_of_setencoder_mlp_layers",
        type=int,
        default=None,
        help=(
            "Number of Linear(output_dim, output_dim) layers in both SetEncoder MLPs. "
            "Default preserves the legacy shared=1/output=2 layout."
        ),
    )
    parser.add_argument(
        "--set_encoder_norm",
        "--set-encoder-norm",
        dest="set_encoder_norm",
        action="store_true",
        default=False,
        help="Insert LayerNorm between SetEncoder Linear and GELU layers.",
    )
    parser.add_argument(
        "--no-set_encoder_norm",
        "--no-set-encoder-norm",
        dest="set_encoder_norm",
        action="store_false",
        help="Disable LayerNorm layers in the SetEncoder.",
    )
    parser.add_argument(
        "--raw_stat_emb",
        dest="raw_stat_emb",
        action="store_true",
        default=False,
        help="Enable the 32-dim raw statistic embedding branch in the encoder (default: off).",
    )
    parser.add_argument(
        "--no-raw_stat_emb",
        dest="raw_stat_emb",
        action="store_false",
        help="Disable the raw statistic embedding branch so encoder output stays 64-dim.",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--cls-loss-weight",
        type=float,
        default=0,
        help="Weight for the auxiliary training loss (dataset classification or proxy signature regression).",
    )
    parser.add_argument(
        "--proxy-signature-regression",
        dest="proxy_signature_regression",
        action="store_true",
        default=False,
        help="Use proxy signature regression as the auxiliary loss instead of dataset-id classification.",
    )
    parser.add_argument(
        "--no-proxy-signature-regression",
        dest="proxy_signature_regression",
        action="store_false",
        help="Use the original dataset-id classification auxiliary loss.",
    )
    parser.add_argument(
        "--adaptive-sampling-window",
        type=int,
        default=3,
        help=(
            "Window size k for adaptive task sampling. A dataset becomes an adaptive candidate "
            "when the mean validation loss over the most recent k epochs exceeds the mean over "
            "the preceding k epochs."
        ),
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Run only train/validation and skip final checkpoint test sweep.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("./meta_checkpoints/new_dspbuilder_meta"))
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return run_new_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
