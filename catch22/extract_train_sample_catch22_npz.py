from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pycatch22


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_provider.data_loader import (  # noqa: E402
    Dataset_Custom,
    Dataset_ETT_hour,
    Dataset_MultiSeriesForecast,
)
from scripts.multi_backbone_proxy.proxy_experiment_config import (  # noqa: E402
    DATASETS as DATASET_REGISTRY,
    DatasetSpec,
)


DEFAULT_SAMPLES_PER_DATASET = 500
EXPECTED_DATASET_COUNT = 53
EXPECTED_NORMALIZATION_FIT_COUNT = 39

# These validation/test datasets receive the transformation fitted on the
# other 39 datasets.  Their samples never contribute to the feature mean/std.
HELD_OUT_DATASETS = frozenset(
    {
        "electricity",
        "ETT-small",
        "exchange_rate",
        "illness",
        "traffic",
        "weather",
        "Coastal_T_S__H",
        "sunspot_dataset_without_missing_values",
        "Australia_Solar__H",
        "Water_Quality_Darwin__15T",
        "current_velocity__20T",
        "wind_4_seconds_dataset",
        "SG_Carpark__15T",
        "Port_Activity__D",
    }
)

BENCHMARK_OUTPUT_NAMES = {
    "ECL": "electricity",
    "ETTh1": "ETT-small",
    "Exchange": "exchange_rate",
    "ILI": "illness",
    "Traffic": "traffic",
    "Weather": "weather",
}


@dataclass(frozen=True)
class DatasetFeatureBatch:
    registry_name: str
    dataset_name: str
    family: str
    config: DatasetSpec
    sample_indices: np.ndarray
    sample_ranks: np.ndarray
    raw_features: np.ndarray
    n_channels: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample train sequences from all 53 forecasting datasets, extract "
            "channel-pooled catch22 features, fit feature-wise z-score statistics "
            "on the 39 normalization-fit datasets, and write one NPZ per dataset."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root. Default: inferred from this script path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("DCSPG/TS_dataset"),
        help="NPZ output directory. Relative paths are resolved under --repo-root.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260603,
        help="Random seed for train-window selection.",
    )
    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=DEFAULT_SAMPLES_PER_DATASET,
        help="Number of train windows drawn without replacement per dataset.",
    )
    parser.add_argument(
        "--pooling",
        choices=("mean", "median"),
        default="mean",
        help="Feature-wise pooling across channels.",
    )
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Use raw windows instead of Time-Series-Library train-split scaling.",
    )
    parser.add_argument(
        "--multi-series-lru-size",
        type=int,
        default=8,
        help="Number of decoded multi-series source records retained in memory.",
    )
    return parser.parse_args()


def output_dataset_name(registry_name: str) -> str:
    if registry_name in BENCHMARK_OUTPUT_NAMES:
        return BENCHMARK_OUTPUT_NAMES[registry_name]
    for prefix in ("Monash__", "TIME__"):
        if registry_name.startswith(prefix):
            return registry_name[len(prefix) :]
    raise ValueError(f"Unknown registry dataset name: {registry_name!r}")


def dataset_family(registry_name: str) -> str:
    if registry_name in BENCHMARK_OUTPUT_NAMES:
        return "Benchmark"
    return registry_name.split("__", maxsplit=1)[0]


def validated_registry() -> list[tuple[str, str, DatasetSpec]]:
    entries = [
        (registry_name, output_dataset_name(registry_name), config)
        for registry_name, config in DATASET_REGISTRY.items()
    ]
    if len(entries) != EXPECTED_DATASET_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_DATASET_COUNT} registry entries, got {len(entries)}."
        )

    output_names = [dataset_name for _, dataset_name, _ in entries]
    duplicates = sorted({name for name in output_names if output_names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate output dataset names: {duplicates}")

    missing_held_out = sorted(HELD_OUT_DATASETS.difference(output_names))
    if missing_held_out:
        raise ValueError(f"Held-out datasets are absent from the registry: {missing_held_out}")

    fit_count = len(set(output_names).difference(HELD_OUT_DATASETS))
    if fit_count != EXPECTED_NORMALIZATION_FIT_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_NORMALIZATION_FIT_COUNT} normalization-fit datasets, "
            f"got {fit_count}."
        )
    return entries


def sequence_lengths(dataset_name: str) -> tuple[int, int, int]:
    if dataset_name == "illness":
        return 36, 18, 36
    return 96, 48, 96


def build_dataset(
    config: DatasetSpec,
    dataset_name: str,
    args: argparse.Namespace,
):
    seq_len, label_len, pred_len = sequence_lengths(dataset_name)
    loader_args = SimpleNamespace(
        augmentation_ratio=0,
        enc_in=config.enc_in,
        multi_series_lru_size=args.multi_series_lru_size,
        # Sample from the complete train-window population.
        long_term_train_sample_limit=0,
        candidate_sample_seed=args.seed,
    )

    if config.data == "multi_series":
        dataset_cls = Dataset_MultiSeriesForecast
    elif config.data == "ETTh1":
        dataset_cls = Dataset_ETT_hour
    elif config.data == "custom":
        dataset_cls = Dataset_Custom
    else:
        raise ValueError(f"Unsupported loader {config.data!r} for {dataset_name}")

    return dataset_cls(
        args=loader_args,
        root_path=config.root_path,
        flag="train",
        size=[seq_len, label_len, pred_len],
        features=config.features,
        data_path=config.data_path,
        target=config.target,
        scale=not args.no_scale,
        timeenc=1,
        freq=config.freq,
    )


def interpolate_nonfinite(series: np.ndarray) -> np.ndarray:
    values = np.asarray(series, dtype=np.float64)
    finite = np.isfinite(values)
    if finite.all():
        return values
    if not finite.any():
        return np.zeros_like(values)
    if finite.sum() == 1:
        return np.full_like(values, float(values[finite][0]))
    positions = np.arange(len(values), dtype=np.float64)
    return np.interp(positions, positions[finite], values[finite])


def catch22_by_channel(sample: np.ndarray) -> tuple[list[str], list[str], np.ndarray]:
    values = np.asarray(sample, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(f"Expected a [seq_len, channels] window, got {values.shape}")

    feature_names = None
    short_names = None
    channel_features = []
    for channel_index in range(values.shape[1]):
        series = interpolate_nonfinite(values[:, channel_index])
        result = pycatch22.catch22_all(series, catch24=False, short_names=True)
        current_names = list(result["names"])
        current_short_names = list(result["short_names"])
        if feature_names is None:
            feature_names = current_names
            short_names = current_short_names
        elif short_names != current_short_names:
            raise RuntimeError("catch22 feature order changed between channels")
        channel_features.append(np.asarray(result["values"], dtype=np.float64))

    if feature_names is None or short_names is None:
        raise RuntimeError("No channel was available for catch22 extraction")
    matrix = np.vstack(channel_features)
    if matrix.shape[1] != 22:
        raise RuntimeError(f"Expected 22 catch22 features, got {matrix.shape}")
    return feature_names, short_names, matrix


def pool_features(channel_features: np.ndarray, pooling: str) -> np.ndarray:
    finite_features = np.where(np.isfinite(channel_features), channel_features, np.nan)
    if pooling == "mean":
        counts = np.sum(np.isfinite(finite_features), axis=0)
        pooled = np.full(channel_features.shape[1], np.nan, dtype=np.float64)
        np.divide(
            np.nansum(finite_features, axis=0),
            counts,
            out=pooled,
            where=counts > 0,
        )
        return pooled
    if pooling == "median":
        with np.errstate(all="ignore"):
            return np.nanmedian(finite_features, axis=0)
    raise ValueError(f"Unsupported pooling: {pooling}")


def sample_indices(length: int, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    if n_samples <= 0:
        raise ValueError(f"samples-per-dataset must be positive, got {n_samples}.")
    if length < n_samples:
        raise ValueError(
            f"Train split has only {length} windows; {n_samples} are required."
        )
    return rng.choice(length, size=n_samples, replace=False).astype(np.int64, copy=False)


def collect_features(
    args: argparse.Namespace,
) -> tuple[list[DatasetFeatureBatch], list[str], list[str]]:
    rng = np.random.default_rng(args.seed)
    entries = validated_registry()
    batches = []
    feature_names = None
    feature_short_names = None

    for dataset_number, (registry_name, dataset_name, config) in enumerate(
        entries, start=1
    ):
        dataset = build_dataset(config, dataset_name, args)
        indices = sample_indices(len(dataset), args.samples_per_dataset, rng)
        print(
            f"[{dataset_number:02d}/{len(entries)}] {dataset_name}: "
            f"sampling {len(indices)} of {len(dataset)} train windows",
            flush=True,
        )

        raw_features = []
        n_channels = []
        for sample_index in indices:
            seq_x, _, _, _ = dataset[int(sample_index)]
            sample = np.asarray(seq_x, dtype=np.float64)
            names, short_names, channel_features = catch22_by_channel(sample)
            pooled = pool_features(channel_features, args.pooling)

            if feature_short_names is None:
                feature_names = names
                feature_short_names = short_names
            elif feature_short_names != short_names:
                raise RuntimeError(f"Feature short-name order changed for {dataset_name}")

            raw_features.append(pooled.astype(np.float64, copy=False))
            n_channels.append(sample.shape[1] if sample.ndim == 2 else 1)

        batches.append(
            DatasetFeatureBatch(
                registry_name=registry_name,
                dataset_name=dataset_name,
                family=dataset_family(registry_name),
                config=config,
                sample_indices=indices,
                sample_ranks=np.arange(1, len(indices) + 1, dtype=np.int64),
                raw_features=np.vstack(raw_features).astype(np.float64, copy=False),
                n_channels=np.asarray(n_channels, dtype=np.int64),
            )
        )
        del dataset
        gc.collect()

    if feature_names is None or feature_short_names is None:
        raise RuntimeError("No catch22 features were collected.")
    return batches, feature_names, feature_short_names


def fit_and_apply_zscore(
    batches: list[DatasetFeatureBatch],
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, list[str]]:
    fit_batches = [
        batch for batch in batches if batch.dataset_name not in HELD_OUT_DATASETS
    ]
    if len(fit_batches) != EXPECTED_NORMALIZATION_FIT_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_NORMALIZATION_FIT_COUNT} fit batches, got {len(fit_batches)}."
        )

    fit_matrix = np.vstack([batch.raw_features for batch in fit_batches])
    finite_fit_matrix = np.where(np.isfinite(fit_matrix), fit_matrix, np.nan)
    means = np.nanmean(finite_fit_matrix, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)

    stds = np.nanstd(finite_fit_matrix, axis=0, ddof=0)
    safe_stds = np.where(np.isfinite(stds) & (stds > 0.0), stds, 1.0)

    normalized_by_dataset = []
    for batch in batches:
        clean_features = np.where(np.isfinite(batch.raw_features), batch.raw_features, means)
        normalized_by_dataset.append((clean_features - means) / safe_stds)

    fit_dataset_names = [batch.dataset_name for batch in fit_batches]
    return normalized_by_dataset, means, safe_stds, fit_dataset_names


def save_batches(
    output_dir: Path,
    batches: list[DatasetFeatureBatch],
    normalized_by_dataset: list[np.ndarray],
    feature_names: list[str],
    feature_short_names: list[str],
    means: np.ndarray,
    stds: np.ndarray,
    fit_dataset_names: list[str],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scale_label = "train_split_standard" if not args.no_scale else "none"
    fit_names_array = np.asarray(fit_dataset_names, dtype=str)

    expected_paths = set()
    for batch, normalized in zip(batches, normalized_by_dataset):
        output_path = output_dir / f"{batch.dataset_name}.npz"
        expected_paths.add(output_path.resolve())
        seq_len, _, pred_len = sequence_lengths(batch.dataset_name)
        normalization_role = (
            "held_out_transform_only"
            if batch.dataset_name in HELD_OUT_DATASETS
            else "normalization_fit"
        )
        # savez_compressed opens an existing path in write mode, so the old six
        # benchmark files are replaced rather than reused.
        np.savez_compressed(
            output_path,
            features=normalized.astype(np.float32),
            raw_features=batch.raw_features.astype(np.float32),
            sample_indices=batch.sample_indices,
            sample_ranks=batch.sample_ranks,
            feature_names=np.asarray(feature_names, dtype=str),
            feature_short_names=np.asarray(feature_short_names, dtype=str),
            global_feature_mean=means.astype(np.float64),
            global_feature_std=stds.astype(np.float64),
            dataset=np.asarray(batch.dataset_name, dtype=str),
            registry_name=np.asarray(batch.registry_name, dtype=str),
            family=np.asarray(batch.family, dtype=str),
            data_path=np.asarray(
                str(Path(batch.config.root_path) / batch.config.data_path), dtype=str
            ),
            split=np.asarray("train", dtype=str),
            seq_len=np.asarray(seq_len, dtype=np.int64),
            pred_len=np.asarray(pred_len, dtype=np.int64),
            n_channels=batch.n_channels,
            pooling=np.asarray(args.pooling, dtype=str),
            scale=np.asarray(scale_label, dtype=str),
            seed=np.asarray(args.seed, dtype=np.int64),
            samples_per_dataset=np.asarray(args.samples_per_dataset, dtype=np.int64),
            normalization_scope=np.asarray(
                "39_normalization_fit_datasets_sampled_train_features", dtype=str
            ),
            normalization_role=np.asarray(normalization_role, dtype=str),
            normalization_fit_dataset_count=np.asarray(
                EXPECTED_NORMALIZATION_FIT_COUNT, dtype=np.int64
            ),
            normalization_fit_datasets=fit_names_array,
        )

    actual_paths = {path.resolve() for path in output_dir.glob("*.npz")}
    missing_paths = sorted(path.name for path in expected_paths.difference(actual_paths))
    if missing_paths:
        raise RuntimeError(f"Failed to write expected NPZ files: {missing_paths}")


def validate_args(args: argparse.Namespace) -> None:
    if args.samples_per_dataset <= 0:
        raise ValueError("samples-per-dataset must be positive")
    if args.multi_series_lru_size <= 0:
        raise ValueError("multi-series-lru-size must be positive")


def main() -> int:
    args = parse_args()
    validate_args(args)
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir

    original_cwd = Path.cwd()
    try:
        import os

        os.chdir(repo_root)
        batches, feature_names, feature_short_names = collect_features(args)
        normalized_by_dataset, means, stds, fit_dataset_names = fit_and_apply_zscore(batches)
        save_batches(
            output_dir,
            batches,
            normalized_by_dataset,
            feature_names,
            feature_short_names,
            means,
            stds,
            fit_dataset_names,
            args,
        )
    finally:
        os.chdir(original_cwd)

    row_count = sum(len(batch.sample_indices) for batch in batches)
    print(f"Wrote {len(batches)} dataset NPZ files with {row_count} samples to {output_dir}")
    print(
        f"Z-score fit: {len(fit_dataset_names)} datasets; "
        f"held out from fit: {len(HELD_OUT_DATASETS)} datasets"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
