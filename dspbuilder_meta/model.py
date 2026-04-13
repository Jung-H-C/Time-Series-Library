from __future__ import annotations

import torch
from torch import nn


class FeatureWiseSharedEncoder(nn.Module):
    def __init__(
        self,
        encoder_hidden_dim: int = 16,
        projected_dim: int = 32,
        raw_stat_dim: int = 32,
        raw_stat_emb: bool = True,
    ) -> None:
        super().__init__()
        self.raw_stat_emb = raw_stat_emb
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(1, encoder_hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(encoder_hidden_dim, encoder_hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.feature_projection = nn.Linear(encoder_hidden_dim, projected_dim)
        self.projected_dim = projected_dim
        self.raw_stat_dim = raw_stat_dim if raw_stat_emb else 0
        if raw_stat_emb:
            self.raw_stat_projection = nn.Sequential(
                nn.Linear(8, raw_stat_dim),
                nn.GELU(),
                nn.Linear(raw_stat_dim, raw_stat_dim),
            )
        else:
            self.raw_stat_projection = None

    @property
    def output_dim(self) -> int:
        return (2 * self.projected_dim) + self.raw_stat_dim

    def _extract_raw_stats(self, sample: torch.Tensor) -> torch.Tensor:
        feature_means = sample.mean(dim=0)
        feature_stds = sample.std(dim=0, unbiased=False)

        if sample.shape[0] > 1:
            temporal_diff = sample[1:] - sample[:-1]
            temporal_diff_mean = temporal_diff.mean()
            temporal_diff_std = temporal_diff.std(unbiased=False)
        else:
            temporal_diff_mean = sample.new_zeros(())
            temporal_diff_std = sample.new_zeros(())

        return torch.stack(
            [
                sample.mean(),
                sample.std(unbiased=False),
                feature_means.mean(),
                feature_means.std(unbiased=False),
                feature_stds.mean(),
                feature_stds.std(unbiased=False),
                temporal_diff_mean,
                temporal_diff_std,
            ],
            dim=0,
        )

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        # sample: [T, F]
        if sample.ndim != 2:
            raise ValueError(f"Expected [time, feature] sample tensor, got shape {tuple(sample.shape)}")

        sample = sample.float()  # [T, F]
        if self.raw_stat_emb:
            raw_stats = self._extract_raw_stats(sample)  # [8]
            raw_stat_embedding = self.raw_stat_projection(raw_stats)  # [R], R = raw_stat_dim

        feature_mean = sample.mean(dim=0, keepdim=True)  # [1, F]
        feature_std = sample.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)  # [1, F]
        sample = (sample - feature_mean) / feature_std  # [T, F]

        feature_series = sample.transpose(0, 1).unsqueeze(1)  # [F, 1, T]
        encoded = self.temporal_encoder(feature_series)  # [F, H, T], H = encoder_hidden_dim
        pooled_over_time = encoded.mean(dim=-1)  # [F, H]
        projected = self.feature_projection(pooled_over_time)  # [F, P], P = projected_dim

        pooled_mean = projected.mean(dim=0)  # [P]
        pooled_std = projected.std(dim=0, unbiased=False)  # [P]
        if self.raw_stat_emb:
            return torch.cat([pooled_mean, pooled_std, raw_stat_embedding], dim=0)  # [2P + R]
        return torch.cat([pooled_mean, pooled_std], dim=0)  # [2P]


class DSPBuilderMetaModel(nn.Module):
    def __init__(
        self,
        proxy_dim: int,
        num_dataset_classes: int,
        encoder_hidden_dim: int,
        head_hidden_dim: int,
        dropout: float,
        raw_stat_emb: bool,
    ) -> None:
        super().__init__()
        if num_dataset_classes <= 0:
            raise ValueError("num_dataset_classes must be positive.")
        self.support_encoder = FeatureWiseSharedEncoder(
            encoder_hidden_dim=encoder_hidden_dim,
            projected_dim=32,
            raw_stat_dim=32,
            raw_stat_emb=raw_stat_emb,
        )
        self.sample_embedding_dim = self.support_encoder.output_dim
        self.task_embedding_dim = self.sample_embedding_dim * 2
        self.weight_head = nn.Sequential(
            nn.Linear(self.task_embedding_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, proxy_dim),
        )
        self.dataset_classifier = nn.Sequential(
            nn.Linear(self.task_embedding_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, num_dataset_classes),
        )

    def forward(self, support_samples: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # support_samples: list of S tensors, each [T, F]
        if not support_samples:
            raise ValueError("support_samples must contain at least one mini sample.")

        sample_embeddings = [self.support_encoder(sample) for sample in support_samples]  # S x [E]
        stacked_embeddings = torch.stack(sample_embeddings, dim=0)  # [S, E]
        sample_mean = stacked_embeddings.mean(dim=0)  # [E]
        sample_std = stacked_embeddings.std(dim=0, unbiased=False)  # [E]
        task_embedding = torch.cat([sample_mean, sample_std], dim=0)  # [2E]
        weight_vector = torch.tanh(self.weight_head(task_embedding))  # [proxy_dim]
        dataset_logits = self.dataset_classifier(task_embedding)  # [num_dataset_classes]
        return weight_vector, task_embedding, dataset_logits  # ([proxy_dim], [2E], [num_dataset_classes])
