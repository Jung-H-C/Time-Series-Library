from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pycatch22

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_provider.data_loader import Dataset_Custom, Dataset_ETT_hour


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    loader: str
    root_path: str
    data_path: str
    seq_len: int
    pred_len: int
    label_len: int
    features: str = "M"
    target: str = "OT"
    freq: str = "h"


DATASETS = (
    DatasetConfig(
        name="ETT-small",
        loader="ETT_hour",
        root_path="./dataset/ETT-small/",
        data_path="ETTh1.csv",
        seq_len=96,
        pred_len=96,
        label_len=48,
    ),
    DatasetConfig(
        name="exchange_rate",
        loader="custom",
        root_path="./dataset/exchange_rate/",
        data_path="exchange_rate.csv",
        seq_len=96,
        pred_len=96,
        label_len=48,
    ),
    DatasetConfig(
        name="illness",
        loader="custom",
        root_path="./dataset/illness/",
        data_path="national_illness.csv",
        seq_len=36,
        pred_len=36,
        label_len=18,
    ),
    DatasetConfig(
        name="traffic",
        loader="custom",
        root_path="./dataset/traffic/",
        data_path="traffic.csv",
        seq_len=96,
        pred_len=96,
        label_len=48,
    ),
    DatasetConfig(
        name="weather",
        loader="custom",
        root_path="./dataset/weather/",
        data_path="weather.csv",
        seq_len=96,
        pred_len=96,
        label_len=48,
    ),
    DatasetConfig(
        name="electricity",
        loader="custom",
        root_path="./dataset/electricity/",
        data_path="electricity.csv",
        seq_len=96,
        pred_len=96,
        label_len=48,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample one validation sequence per dataset and export channel-pooled "
            "pycatch22 catch24 features."
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
        default=Path("catch22/valid_sample_catch24_features.csv"),
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


def build_dataset(config: DatasetConfig, *, scale: bool):
    args = SimpleNamespace(augmentation_ratio=0)
    size = [config.seq_len, config.label_len, config.pred_len]
    dataset_cls = Dataset_ETT_hour if config.loader == "ETT_hour" else Dataset_Custom
    return dataset_cls(
        args=args,
        root_path=config.root_path,
        flag="val",
        size=size,
        features=config.features,
        data_path=config.data_path,
        target=config.target,
        scale=scale,
        timeenc=1,
        freq=config.freq,
    )


def catch24_by_channel(sample: np.ndarray) -> tuple[list[str], list[str], np.ndarray]:
    feature_names = None
    short_names = None
    channel_features = []
    for channel_index in range(sample.shape[1]):
        series = np.asarray(sample[:, channel_index], dtype=np.float64)
        result = pycatch22.catch22_all(series, catch24=True, short_names=True)
        if feature_names is None:
            feature_names = list(result["names"])
            short_names = list(result["short_names"])
        channel_features.append(np.asarray(result["values"], dtype=np.float64))

    return feature_names, short_names, np.vstack(channel_features)


def pool_features(channel_features: np.ndarray, pooling: str) -> np.ndarray:
    if pooling == "mean":
        return np.nanmean(channel_features, axis=0)
    if pooling == "median":
        return np.nanmedian(channel_features, axis=0)
    raise ValueError(f"Unsupported pooling: {pooling}")


def make_rows(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[str]]:
    rng = np.random.default_rng(args.seed)
    rows = []
    feature_short_names = None
    feature_names = None
    scale = not args.no_scale

    for config in DATASETS:
        dataset = build_dataset(config, scale=scale)
        sample_index = int(rng.integers(0, len(dataset)))
        seq_x, _, _, _ = dataset[sample_index]
        sample = np.asarray(seq_x, dtype=np.float64)

        names, short_names, channel_features = catch24_by_channel(sample)
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
            "sample_index": sample_index,
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
        "sample_index",
        "seq_len",
        "pred_len",
        "n_channels",
        "pooling",
        "scale",
        *feature_short_names,
    ]
    return rows, fieldnames


def write_csv(output_path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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

    print(f"Wrote {len(rows)} dataset rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
