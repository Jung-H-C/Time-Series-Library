from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from DCSPG.config import DCSPGConfig, MetaBatchConfig
from DCSPG.data import Catch22FeatureStore, LODOMetaBatchSampler, MetaBatch
from DCSPG.grammar import RPNGrammar
from DCSPG.model import DCSPGModel
from DCSPG.targets import GroundTruthFormulaTargetProvider
from DCSPG.vocabulary import SymbolicVocabulary


@dataclass(frozen=True)
class LODOFold:
    leave_out_dataset: str
    train_datasets: tuple[str, ...]


class LODOExperiment:
    """Leave-one-dataset-out experiment scaffold."""

    def __init__(
        self,
        store: Catch22FeatureStore,
        meta_config: MetaBatchConfig = MetaBatchConfig(),
        seed: int = 2026,
    ) -> None:
        self.store = store
        self.meta_config = meta_config
        self.seed = seed

    def folds(self) -> tuple[LODOFold, ...]:
        folds = []
        for leave_out in self.store.dataset_names:
            train = tuple(name for name in self.store.dataset_names if name != leave_out)
            folds.append(LODOFold(leave_out_dataset=leave_out, train_datasets=train))
        return tuple(folds)

    def build_sampler(self, leave_out_dataset: str) -> LODOMetaBatchSampler:
        return LODOMetaBatchSampler(
            store=self.store,
            leave_out_dataset=leave_out_dataset,
            config=self.meta_config,
            seed=self.seed,
        )


def build_default_vocabulary(ground_truth_dir: Path | str = "DCSPG/GroundTruth") -> SymbolicVocabulary:
    return SymbolicVocabulary.from_ground_truth_dir(ground_truth_dir)


def build_default_grammar(
    vocabulary: SymbolicVocabulary,
    config: DCSPGConfig = DCSPGConfig(),
) -> RPNGrammar:
    return RPNGrammar.from_vocabulary(
        vocabulary,
        max_stack_depth=config.max_stack_depth,
        max_unary_chain=config.max_unary_chain,
    )


def build_default_model(
    vocabulary: SymbolicVocabulary | None = None,
    config: DCSPGConfig = DCSPGConfig(),
    grammar: RPNGrammar | None = None,
) -> DCSPGModel:
    vocab = vocabulary or SymbolicVocabulary.default()
    grammar = grammar or build_default_grammar(vocab, config)
    return DCSPGModel(config=config, vocab_size=len(vocab), grammar=grammar)


@dataclass(frozen=True)
class DCSPGComponents:
    store: Catch22FeatureStore
    vocabulary: SymbolicVocabulary
    grammar: RPNGrammar
    model: DCSPGModel
    target_provider: GroundTruthFormulaTargetProvider


def build_training_components(
    ts_feature_dir: Path | str = "DCSPG/TS_dataset",
    ground_truth_dir: Path | str = "DCSPG/GroundTruth",
    model_config: DCSPGConfig = DCSPGConfig(),
    seed: int = 2026,
    target_sampling_strategy: str = "cycle",
    targets_per_episode: int = 1,
) -> DCSPGComponents:
    store = Catch22FeatureStore(ts_feature_dir)
    vocabulary = build_default_vocabulary(ground_truth_dir)
    grammar = build_default_grammar(vocabulary, model_config)
    model = DCSPGModel(model_config, vocab_size=len(vocabulary), grammar=grammar)
    target_provider = GroundTruthFormulaTargetProvider(
        ground_truth_dir=ground_truth_dir,
        vocabulary=vocabulary,
        seed=seed,
        grammar=grammar,
        max_target_len=model_config.max_formula_len,
        sampling_strategy=target_sampling_strategy,
        targets_per_episode=targets_per_episode,
    )
    return DCSPGComponents(
        store=store,
        vocabulary=vocabulary,
        grammar=grammar,
        model=model,
        target_provider=target_provider,
    )


@torch.no_grad()
def infer_leaveout_batch(
    model: DCSPGModel,
    batch: MetaBatch,
    vocabulary: SymbolicVocabulary,
    device: torch.device | str = "cpu",
    max_len: int | None = None,
) -> list[list[str]]:
    model = model.to(device)
    batch = batch.to(device)
    token_ids = model.generate(
        batch.inputs,
        bos_id=vocabulary.bos_id,
        eos_id=vocabulary.eos_id,
        pad_id=vocabulary.pad_id,
        max_len=max_len,
        greedy=True,
    )
    return [vocabulary.decode(row.tolist()) for row in token_ids.cpu()]
