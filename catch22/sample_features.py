from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pycatch22

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from extract_valid_sample_catch24 import (  # noqa: E402
    DATASETS,
    build_dataset,
    pool_features,
    write_csv,
)


SAMPLES_PER_DATASET = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample 50 validation sequences per dataset and export channel-pooled "
            "pycatch22 catch22 features."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root. Default: inferred from this script path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("catch22/sample_features.csv"),
        help="Output CSV path. Relative paths are resolved under --repo-root.",
    )
    parser.add_argument("--seed", type=int, default=20260603, help="Random seed for validation sample selection.")
    parser.add_argument(
        "--pooling",
        choices=("mean", "median"),
        default="mean",
        help="Feature-wise pooling across channels. Default: mean.",
    )
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Use raw values instead of Time-Series-Library train-split scaling.",
    )
    return parser.parse_args()


def catch22_by_channel(sample: np.ndarray) -> tuple[list[str], list[str], np.ndarray]:
    feature_names = None
    short_names = None
    channel_features = []
    for channel_index in range(sample.shape[1]):
        series = np.asarray(sample[:, channel_index], dtype=np.float64)
        result = pycatch22.catch22_all(series, catch24=False, short_names=True)
        if feature_names is None:
            feature_names = list(result["names"])
            short_names = list(result["short_names"])
        channel_features.append(np.asarray(result["values"], dtype=np.float64))

    return feature_names, short_names, np.vstack(channel_features)


def sample_indices(length: int, rng: np.random.Generator) -> np.ndarray:
    if length < SAMPLES_PER_DATASET:
        raise ValueError(
            f"Validation split has only {length} samples; "
            f"{SAMPLES_PER_DATASET} samples are required."
        )
    return rng.choice(length, size=SAMPLES_PER_DATASET, replace=False)


def make_rows(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[str]]:
    rng = np.random.default_rng(args.seed)
    rows = []
    feature_short_names = None
    feature_names = None
    scale = not args.no_scale

    for config in DATASETS:
        dataset = build_dataset(config, scale=scale)
        indices = sample_indices(len(dataset), rng)
        for sample_rank, sample_index in enumerate(indices, start=1):
            seq_x, _, _, _ = dataset[int(sample_index)]
            sample = np.asarray(seq_x, dtype=np.float64)

            names, short_names, channel_features = catch22_by_channel(sample)
            pooled = pool_features(channel_features, args.pooling)

            if feature_short_names is None:
                feature_short_names = short_names
                feature_names = names
            elif feature_short_names != short_names:
                raise RuntimeError(f"Feature short-name order changed for dataset: {config.name}")

            row = {
                "dataset": config.name,
                "data_path": str(Path(config.root_path) / config.data_path),
                "split": "val",
                "sample_rank": sample_rank,
                "sample_index": int(sample_index),
                "seq_len": config.seq_len,
                "pred_len": config.pred_len,
                "n_channels": sample.shape[1],
                "pooling": args.pooling,
                "scale": "train_split_standard" if scale else "none",
            }
            for short_name, value in zip(short_names, pooled):
                row[short_name] = f"{float(value):.9f}"
            rows.append(row)

    assert feature_short_names is not None
    assert feature_names is not None
    fieldnames = [
        "dataset",
        "data_path",
        "split",
        "sample_rank",
        "sample_index",
        "seq_len",
        "pred_len",
        "n_channels",
        "pooling",
        "scale",
        *feature_short_names,
    ]
    return rows, fieldnames


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_path = args.output if args.output.is_absolute() else repo_root / args.output

    original_cwd = Path.cwd()
    try:
        import os

        os.chdir(repo_root)
        rows, fieldnames = make_rows(args)
        write_csv(output_path, rows, fieldnames)
    finally:
        os.chdir(original_cwd)

    print(f"Wrote {len(rows)} sample rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
