from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DCSPGConfig:
    n_features: int = 22
    stat_dim: int = 2
    d_model: int = 64
    n_heads: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    max_formula_len: int = 12
    max_stack_depth: int = 4
    max_unary_chain: int = 2


@dataclass(frozen=True)
class MetaBatchConfig:
    batch_size: int = 32
    k_samples: int = 16
    base_episodes_per_dataset: int = 6
    extra_episodes: int = 2
    base_episodes_per_cluster: int = 4
    extra_cluster_episodes: int = 4
    teachers_per_episode: int = 16
