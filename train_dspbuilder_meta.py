from __future__ import annotations

import argparse
from pathlib import Path

from dspbuilder_meta.pipeline import run_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Skeleton training script for DSPBuilder: load benchmark CSV dictionaries, "
            "sample support/query candidates per dataset, and train a task-conditioned "
            "proxy-weight generator with pairwise ranking loss."
        )
    )
    parser.add_argument("--benchmark-dir", type=Path, default=Path("./benchmark"))
    parser.add_argument("--candidate-dir", type=Path, default=Path("./candidates"))
    parser.add_argument("--train-datasets", type=str, default="", help="Comma-separated dataset names.")
    parser.add_argument("--val-datasets", type=str, default="", help="Comma-separated dataset names.")
    parser.add_argument("--test-datasets", type=str, default="", help="Comma-separated dataset names.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--iterations-per-dataset", type=int, default=5)
    parser.add_argument("--val-iterations-per-dataset", type=int, default=10)
    parser.add_argument(
        "--eval-iterations-per-dataset",
        type=int,
        default=5,
        help="Number of test iterations per dataset.",
    )
    parser.add_argument("--support-size", type=int, default=5)
    parser.add_argument("--train-query-size", type=int, default=20)
    parser.add_argument("--val-query-size", type=int, default=10)
    parser.add_argument("--test-query-size", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=32) # proxy builder hidden dim: 128 > hidden-dim > proxy_dim
    parser.add_argument("--encoder-hidden-dim", type=int, default=16)
    parser.add_argument(
        "--raw_stat_emb",
        dest="raw_stat_emb",
        action="store_true",
        default=False,
        help="Enable the 32-dim raw statistic embedding branch in the encoder (default: on).",
    )
    parser.add_argument(
        "--no-raw_stat_emb",
        dest="raw_stat_emb",
        action="store_false",
        help="Disable the raw statistic embedding branch so encoder output stays 64-dim.",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--cls-loss-weight",
        type=float,
        default=0.5,
        help="Weight for the auxiliary dataset classification loss (train only).",
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Run only train/validation and skip final test evaluation.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("./meta_checkpoints/dspbuilder_meta"))
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
