from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from DCSPG.data import MetaBatch
from DCSPG.grammar import RPNGrammar
from DCSPG.vocabulary import SymbolicVocabulary


DEFAULT_DATASET_NAME_MAP = {
    "ETT-small": "ETTh1",
    "electricity": "ECL",
    "exchange_rate": "Exchange",
    "illness": "ILI",
    "traffic": "Traffic",
    "weather": "Weather",
}


@dataclass(frozen=True)
class GroundTruthDataset:
    name: str
    rpn_tokens: tuple[str, ...]
    weights: tuple[float, ...]
    path: Path


@dataclass(frozen=True)
class SampledFormulaTargets:
    target_ids: torch.Tensor
    target_weights: torch.Tensor
    ground_truth_dataset_names: tuple[str, ...]
    formula_indices: tuple[tuple[int, ...], ...]
    rpn_tokens: tuple[tuple[str, ...], ...]
    sampling_strategy: str


class GroundTruthStore:
    def __init__(self, ground_truth_dir: Path | str, key: str = "rpn_tokens") -> None:
        self.ground_truth_dir = Path(ground_truth_dir)
        self.key = key
        self._datasets: dict[str, GroundTruthDataset] = {}

        for path in sorted(self.ground_truth_dir.glob("*.npz")):
            with np.load(path) as data:
                name = str(data["dataset"].item()) if "dataset" in data else path.stem
                rpn_tokens = tuple(str(item) for item in data[key].tolist())
                if "weight" in data:
                    weights = tuple(float(item) for item in data["weight"].tolist())
                else:
                    weights = tuple([1.0] * len(rpn_tokens))
            if len(weights) != len(rpn_tokens):
                raise ValueError(
                    f"{path} has {len(rpn_tokens)} formulas but {len(weights)} weights"
                )
            if not all(np.isfinite(weight) and weight >= 0.0 for weight in weights):
                raise ValueError(f"{path} contains invalid teacher weights")
            self._datasets[name] = GroundTruthDataset(
                name=name,
                rpn_tokens=rpn_tokens,
                weights=weights,
                path=path,
            )

        if not self._datasets:
            raise ValueError(f"No ground-truth NPZ files found under {self.ground_truth_dir}")

    @property
    def dataset_names(self) -> tuple[str, ...]:
        return tuple(self._datasets.keys())

    def __getitem__(self, dataset_name: str) -> GroundTruthDataset:
        return self._datasets[dataset_name]


class GroundTruthFormulaTargetProvider:
    """Samples weighted symbolic RPN targets for every meta-batch episode.

    The default ``cycle`` strategy uses a per-dataset shuffled permutation and
    reshuffles after every full pass through that dataset's formulas.
    """

    def __init__(
        self,
        ground_truth_dir: Path | str,
        vocabulary: SymbolicVocabulary,
        dataset_name_map: Mapping[str, str] | None = None,
        seed: int = 2026,
        grammar: RPNGrammar | None = None,
        max_target_len: int | None = None,
        sampling_strategy: str = "cycle",
        targets_per_episode: int = 1,
    ) -> None:
        self.store = GroundTruthStore(ground_truth_dir)
        self.vocabulary = vocabulary
        self.dataset_name_map = dict(DEFAULT_DATASET_NAME_MAP)
        if dataset_name_map is not None:
            self.dataset_name_map.update(dataset_name_map)
        self.rng = np.random.default_rng(seed)
        self.grammar = grammar
        self.max_target_len = max_target_len
        self.targets_per_episode = int(targets_per_episode)
        if self.targets_per_episode <= 0:
            raise ValueError("targets_per_episode must be positive")
        if sampling_strategy not in {"cycle", "random"}:
            raise ValueError(f"Unsupported target sampling strategy: {sampling_strategy}")
        self.sampling_strategy = sampling_strategy
        self._cycle_cursors = {name: 0 for name in self.store.dataset_names}
        self._cycle_orders: dict[str, np.ndarray] = {}
        self.last_sampled: SampledFormulaTargets | None = None

    def get_targets(self, batch: MetaBatch, device: torch.device | str) -> torch.Tensor:
        sampled = self.sample(batch)
        self.last_sampled = sampled
        return sampled.target_ids.to(device)

    def sample(self, batch: MetaBatch) -> SampledFormulaTargets:
        encoded_targets: list[list[list[int]]] = []
        target_weights: list[list[float]] = []
        gt_dataset_names = []
        formula_indices = []
        rpn_targets = []

        for dataset_name in batch.dataset_names:
            gt_name = self.resolve_dataset_name(dataset_name)
            gt_dataset = self.store[gt_name]
            indices = self._next_formula_indices(
                gt_name,
                len(gt_dataset.rpn_tokens),
                self.targets_per_episode,
            )
            episode_targets = []
            episode_rpns = []
            episode_weights = []
            for formula_index in indices:
                rpn = gt_dataset.rpn_tokens[formula_index]
                token_ids = self.vocabulary.encode_rpn(rpn, strict=True)
                if self.grammar is not None and not self.grammar.is_valid_sequence(token_ids):
                    raise ValueError(f"Invalid RPN target for {gt_name}: {rpn}")
                if self.max_target_len is not None and len(token_ids) > self.max_target_len:
                    raise ValueError(
                        f"Target length {len(token_ids)} exceeds "
                        f"max_target_len={self.max_target_len}: {rpn}"
                    )
                episode_targets.append(token_ids)
                episode_rpns.append(rpn)
                episode_weights.append(gt_dataset.weights[formula_index])

            encoded_targets.append(episode_targets)
            target_weights.append(episode_weights)
            gt_dataset_names.append(gt_name)
            formula_indices.append(tuple(indices))
            rpn_targets.append(tuple(episode_rpns))

        target_len = max(
            len(ids)
            for episode_targets in encoded_targets
            for ids in episode_targets
        )
        if self.max_target_len is not None:
            target_len = min(target_len, self.max_target_len)

        target_ids = torch.full(
            (len(encoded_targets), self.targets_per_episode, target_len),
            fill_value=self.vocabulary.pad_id,
            dtype=torch.long,
        )
        for row_idx, episode_targets in enumerate(encoded_targets):
            for teacher_idx, ids in enumerate(episode_targets):
                target_ids[row_idx, teacher_idx, : len(ids)] = torch.tensor(
                    ids, dtype=torch.long
                )

        return SampledFormulaTargets(
            target_ids=target_ids,
            target_weights=torch.tensor(target_weights, dtype=torch.float32),
            ground_truth_dataset_names=tuple(gt_dataset_names),
            formula_indices=tuple(formula_indices),
            rpn_tokens=tuple(rpn_targets),
            sampling_strategy=self.sampling_strategy,
        )

    def _next_formula_indices(
        self,
        dataset_name: str,
        n_formulas: int,
        count: int,
    ) -> list[int]:
        if count > n_formulas:
            raise ValueError(
                f"Ground-truth dataset {dataset_name} has {n_formulas} formulas, "
                f"but {count} targets per episode were requested without replacement"
            )
        if count == 1:
            return [self._next_formula_index(dataset_name, n_formulas)]
        # Every episode receives a fresh uniform sample without replacement.
        return [
            int(index)
            for index in self.rng.choice(n_formulas, size=count, replace=False)
        ]

    def _next_formula_index(self, dataset_name: str, n_formulas: int) -> int:
        if n_formulas <= 0:
            raise ValueError(f"Ground-truth dataset {dataset_name} has no formulas.")
        if self.sampling_strategy == "random":
            return int(self.rng.integers(0, n_formulas))

        cursor = self._cycle_cursors[dataset_name]
        order = self._cycle_orders.get(dataset_name)
        if order is None or len(order) != n_formulas or cursor >= n_formulas:
            order = self._reshuffle_cycle(dataset_name, n_formulas)
            self._cycle_orders[dataset_name] = order
            cursor = 0

        formula_index = int(order[cursor])
        self._cycle_cursors[dataset_name] = cursor + 1
        return formula_index

    def _reshuffle_cycle(self, dataset_name: str, n_formulas: int) -> np.ndarray:
        order = self.rng.permutation(n_formulas).astype(np.int64, copy=False)
        previous = self._cycle_orders.get(dataset_name)
        if previous is not None and n_formulas > 1 and np.array_equal(order, previous):
            order = np.roll(order, 1)
        return order

    def resolve_dataset_name(self, dataset_name: str) -> str:
        if dataset_name in self.store.dataset_names:
            return dataset_name
        mapped = self.dataset_name_map.get(dataset_name, dataset_name)
        if mapped in self.store.dataset_names:
            return mapped

        # Monash/TIME feature NPZ files use names without a family prefix,
        # while GroundTruth files retain their unambiguous registry names.
        suffix_matches = [
            name
            for name in self.store.dataset_names
            if "__" in name and name.split("__", maxsplit=1)[1] == dataset_name
        ]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        if len(suffix_matches) > 1:
            raise KeyError(
                f"Ambiguous GroundTruth suffix mapping for {dataset_name!r}: "
                f"{suffix_matches}"
            )
        raise KeyError(
            f"Could not map feature dataset {dataset_name!r} to a ground-truth dataset. "
            f"Available ground-truth datasets: {self.store.dataset_names}"
        )
