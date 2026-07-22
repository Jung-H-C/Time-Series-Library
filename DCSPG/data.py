from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from DCSPG.config import MetaBatchConfig


@dataclass(frozen=True)
class FeatureDataset:
    name: str
    features: np.ndarray
    feature_short_names: tuple[str, ...]
    path: Path


@dataclass(frozen=True)
class MetaBatch:
    inputs: torch.Tensor
    dataset_names: tuple[str, ...]
    support_indices: tuple[np.ndarray, ...]

    def to(self, device: torch.device | str) -> "MetaBatch":
        return MetaBatch(
            inputs=self.inputs.to(device),
            dataset_names=self.dataset_names,
            support_indices=self.support_indices,
        )


class Catch22FeatureStore:
    """Loads per-dataset NPZ files produced by catch22 extraction scripts."""

    def __init__(
        self,
        feature_dir: Path | str,
        dataset_names: Iterable[str] | None = None,
        features_key: str = "features",
    ) -> None:
        self.feature_dir = Path(feature_dir)
        self.features_key = features_key
        self._datasets: dict[str, FeatureDataset] = {}

        paths = self._resolve_paths(dataset_names)
        for path in paths:
            dataset = self._load_npz(path)
            self._datasets[dataset.name] = dataset

        if not self._datasets:
            raise ValueError(f"No NPZ feature files found under {self.feature_dir}")

    @property
    def dataset_names(self) -> tuple[str, ...]:
        return tuple(self._datasets.keys())

    def __getitem__(self, name: str) -> FeatureDataset:
        return self._datasets[name]

    def _resolve_paths(self, dataset_names: Iterable[str] | None) -> list[Path]:
        if dataset_names is None:
            return sorted(self.feature_dir.glob("*.npz"))
        return [self.feature_dir / f"{name}.npz" for name in dataset_names]

    def _load_npz(self, path: Path) -> FeatureDataset:
        if not path.exists():
            raise FileNotFoundError(path)

        with np.load(path) as data:
            if self.features_key not in data:
                raise KeyError(f"{path} does not contain key {self.features_key!r}")
            features = np.asarray(data[self.features_key], dtype=np.float32)
            if features.ndim != 2 or features.shape[1] != 22:
                raise ValueError(f"{path} features must have shape [N, 22], got {features.shape}")

            if "dataset" in data:
                name = str(data["dataset"].item())
            else:
                name = path.stem

            if "feature_short_names" in data:
                feature_short_names = tuple(str(x) for x in data["feature_short_names"].tolist())
            else:
                feature_short_names = tuple(f"feature_{i}" for i in range(features.shape[1]))

        return FeatureDataset(
            name=name,
            features=features,
            feature_short_names=feature_short_names,
            path=path,
        )


class LODOMetaBatchSampler:
    """Samples meta-batches for leave-one-dataset-out training."""

    def __init__(
        self,
        store: Catch22FeatureStore,
        leave_out_dataset: str,
        config: MetaBatchConfig = MetaBatchConfig(),
        seed: int = 2026,
    ) -> None:
        self.store = store
        self.leave_out_dataset = leave_out_dataset
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.train_dataset_names = tuple(name for name in store.dataset_names if name != leave_out_dataset)

        if leave_out_dataset not in store.dataset_names:
            raise ValueError(f"Unknown leave-out dataset: {leave_out_dataset}")
        if len(self.train_dataset_names) == 0:
            raise ValueError("LODO sampling requires at least one training dataset.")

        expected_batch = (
            len(self.train_dataset_names) * config.base_episodes_per_dataset
            + config.extra_episodes
        )
        if expected_batch != config.batch_size:
            raise ValueError(
                "Batch composition does not match batch_size: "
                f"{len(self.train_dataset_names)} * {config.base_episodes_per_dataset} "
                f"+ {config.extra_episodes} = {expected_batch}, "
                f"but batch_size={config.batch_size}."
            )
        if config.extra_episodes > len(self.train_dataset_names):
            raise ValueError("extra_episodes must be <= number of training datasets.")

    def sample_train_batch(self) -> MetaBatch:
        dataset_schedule = self._sample_dataset_schedule()
        inputs = []
        support_indices = []

        for dataset_name in dataset_schedule:
            stats, indices = self._sample_episode(dataset_name)
            inputs.append(stats)
            support_indices.append(indices)

        return MetaBatch(
            inputs=torch.from_numpy(np.stack(inputs, axis=0)).float(),
            dataset_names=tuple(dataset_schedule),
            support_indices=tuple(support_indices),
        )

    def sample_leaveout_batch(self, n_episodes: int) -> MetaBatch:
        inputs = []
        support_indices = []
        for _ in range(n_episodes):
            stats, indices = self._sample_episode(self.leave_out_dataset)
            inputs.append(stats)
            support_indices.append(indices)

        return MetaBatch(
            inputs=torch.from_numpy(np.stack(inputs, axis=0)).float(),
            dataset_names=tuple([self.leave_out_dataset] * n_episodes),
            support_indices=tuple(support_indices),
        )

    def _sample_dataset_schedule(self) -> list[str]:
        schedule: list[str] = []
        for name in self.train_dataset_names:
            schedule.extend([name] * self.config.base_episodes_per_dataset)

        extra_names = self.rng.choice(
            self.train_dataset_names,
            size=self.config.extra_episodes,
            replace=False,
        )
        schedule.extend(str(name) for name in extra_names)
        self.rng.shuffle(schedule)
        return schedule

    def _sample_episode(self, dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
        features = self.store[dataset_name].features
        replace = self.config.k_samples > len(features)
        indices = self.rng.choice(len(features), size=self.config.k_samples, replace=replace)
        support = features[indices]

        means = np.nanmean(support, axis=0)
        stds = np.nanstd(support, axis=0, ddof=0)
        means = np.where(np.isfinite(means), means, 0.0)
        stds = np.where(np.isfinite(stds), stds, 0.0)
        stats = np.stack([means, stds], axis=-1).astype(np.float32)
        return stats, indices.astype(np.int64, copy=False)


def sample_support_episode(
    store: Catch22FeatureStore,
    dataset_name: str,
    k_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one support set without replacement and return [22, 2] stats."""
    if k_samples <= 0:
        raise ValueError("k_samples must be positive")
    features = store[dataset_name].features
    if len(features) < k_samples:
        raise ValueError(
            f"Dataset {dataset_name!r} has {len(features)} feature samples, "
            f"but k_samples={k_samples} requires sampling without replacement"
        )
    indices = rng.choice(len(features), size=k_samples, replace=False)
    support = features[indices]
    means = np.nanmean(support, axis=0)
    stds = np.nanstd(support, axis=0, ddof=0)
    means = np.where(np.isfinite(means), means, 0.0)
    stds = np.where(np.isfinite(stds), stds, 0.0)
    stats = np.stack([means, stds], axis=-1).astype(np.float32)
    return stats, indices.astype(np.int64, copy=False)


class ClusterBalancedMetaBatchSampler:
    """Draw a near-equal episode count from every dataset cluster.

    Dataset ids are sampled uniformly with replacement within a cluster.  The
    support samples of each selected dataset are sampled without replacement.
    Every cluster receives ``base_episodes_per_cluster`` episodes, and
    ``extra_cluster_episodes`` distinct clusters are chosen uniformly to receive
    one additional episode. Thus a 256-episode batch over four K=4 training
    clusters with 64 base episodes and no extras has an equal
    [64, 64, 64, 64] allocation.
    """

    def __init__(
        self,
        store: Catch22FeatureStore,
        cluster_datasets: dict[int, tuple[str, ...]],
        config: MetaBatchConfig = MetaBatchConfig(),
        seed: int = 2026,
    ) -> None:
        self.store = store
        self.cluster_datasets = {
            int(cluster_id): tuple(dataset_names)
            for cluster_id, dataset_names in sorted(cluster_datasets.items())
        }
        self.config = config
        self.rng = np.random.default_rng(seed)

        if not self.cluster_datasets:
            raise ValueError("At least one training cluster is required")
        empty = [cluster_id for cluster_id, names in self.cluster_datasets.items() if not names]
        if empty:
            raise ValueError(f"Training clusters cannot be empty: {empty}")
        unknown = sorted(
            {
                name
                for names in self.cluster_datasets.values()
                for name in names
                if name not in store.dataset_names
            }
        )
        if unknown:
            raise ValueError(f"Cluster assignments contain unknown datasets: {unknown}")
        duplicates = [
            name
            for name in store.dataset_names
            if sum(name in names for names in self.cluster_datasets.values()) > 1
        ]
        if duplicates:
            raise ValueError(f"Datasets assigned to multiple clusters: {duplicates}")

        if config.base_episodes_per_cluster <= 0:
            raise ValueError("base_episodes_per_cluster must be positive")
        if not 0 <= config.extra_cluster_episodes <= len(self.cluster_datasets):
            raise ValueError(
                "extra_cluster_episodes must be between zero and the number of clusters"
            )
        expected_batch_size = (
            len(self.cluster_datasets) * config.base_episodes_per_cluster
            + config.extra_cluster_episodes
        )
        if config.batch_size != expected_batch_size:
            raise ValueError(
                "batch_size must equal cluster_count * base_episodes_per_cluster "
                "+ extra_cluster_episodes: "
                f"expected {expected_batch_size}, got {config.batch_size}"
            )

    @property
    def train_dataset_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for dataset_names in self.cluster_datasets.values()
            for name in dataset_names
        )

    def _sample_dataset_schedule(self) -> list[str]:
        dataset_schedule = []
        extra_cluster_ids = set(
            int(cluster_id)
            for cluster_id in self.rng.choice(
                tuple(self.cluster_datasets),
                size=self.config.extra_cluster_episodes,
                replace=False,
            )
        )
        for cluster_id, dataset_names in self.cluster_datasets.items():
            episode_count = self.config.base_episodes_per_cluster + int(
                cluster_id in extra_cluster_ids
            )
            sampled = self.rng.choice(
                dataset_names,
                size=episode_count,
                replace=True,
            )
            dataset_schedule.extend(str(name) for name in sampled)
        self.rng.shuffle(dataset_schedule)
        return dataset_schedule

    def sample_train_batch(self) -> MetaBatch:
        dataset_schedule = self._sample_dataset_schedule()
        inputs = []
        support_indices = []
        for dataset_name in dataset_schedule:
            stats, indices = sample_support_episode(
                self.store,
                dataset_name,
                self.config.k_samples,
                self.rng,
            )
            inputs.append(stats)
            support_indices.append(indices)
        return MetaBatch(
            inputs=torch.from_numpy(np.stack(inputs, axis=0)).float(),
            dataset_names=tuple(dataset_schedule),
            support_indices=tuple(support_indices),
        )

    def sample_dataset_batch(
        self,
        dataset_names: Iterable[str],
    ) -> MetaBatch:
        inputs = []
        support_indices = []
        names = tuple(dataset_names)
        for dataset_name in names:
            stats, indices = sample_support_episode(
                self.store,
                dataset_name,
                self.config.k_samples,
                self.rng,
            )
            inputs.append(stats)
            support_indices.append(indices)
        return MetaBatch(
            inputs=torch.from_numpy(np.stack(inputs, axis=0)).float(),
            dataset_names=names,
            support_indices=tuple(support_indices),
        )
