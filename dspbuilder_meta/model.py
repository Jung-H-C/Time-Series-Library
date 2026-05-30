from __future__ import annotations

import torch
from torch import nn


class FeatureWiseSharedEncoder(nn.Module):
    def __init__(
        self,
        encoder_hidden_dim: int = 64,
        number_of_conv1d_layer: int = 1,
        sample_encoder_norm: bool = False,
        raw_stat_dim: int = 32,
        raw_stat_emb: bool = True,
    ) -> None:
        super().__init__()
        if number_of_conv1d_layer < 0:
            raise ValueError("number_of_conv1d_layer must be non-negative.")
        if sample_encoder_norm and encoder_hidden_dim % 8 != 0:
            raise ValueError("encoder_hidden_dim must be divisible by 8 when sample_encoder_norm is enabled.")

        self.raw_stat_emb = raw_stat_emb
        self.encoder_hidden_dim = encoder_hidden_dim
        self.number_of_conv1d_layer = number_of_conv1d_layer
        self.sample_encoder_norm = sample_encoder_norm

        temporal_layers: list[nn.Module] = [
            nn.Conv1d(1, encoder_hidden_dim, kernel_size=7, padding=3),
        ]
        if sample_encoder_norm:
            temporal_layers.append(nn.GroupNorm(num_groups=8, num_channels=encoder_hidden_dim))
        temporal_layers.append(nn.GELU())
        for _ in range(number_of_conv1d_layer):
            temporal_layers.append(nn.Conv1d(encoder_hidden_dim, encoder_hidden_dim, kernel_size=5, padding=2))
            if sample_encoder_norm:
                temporal_layers.append(nn.GroupNorm(num_groups=8, num_channels=encoder_hidden_dim))
            temporal_layers.append(nn.GELU())
        self.temporal_encoder = nn.Sequential(*temporal_layers)

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
        return self.encoder_hidden_dim + self.raw_stat_dim

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

        # Normalize each feature across time
        feature_mean = sample.mean(dim=0, keepdim=True)  # [1, F]
        feature_std = sample.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)  # [1, F]
        sample = (sample - feature_mean) / feature_std  # [T, F]

        # Encode each feature independently using the same temporal encoder
        feature_series = sample.transpose(0, 1).unsqueeze(1)  # [F, 1, T]
        encoded = self.temporal_encoder(feature_series)  # [F, H, T], H = encoder_hidden_dim

        # Pool over time and feature dimensions to get a single embedding per sample
        pooled_over_time = encoded.mean(dim=-1)  # [F, H]

        pooled_mean = pooled_over_time.mean(dim=0)  # [H] ; mean over features
        sample_embedding = pooled_mean  # [H]
        if self.raw_stat_emb:
            return torch.cat([sample_embedding, raw_stat_embedding], dim=0)  # [H + R]
        return sample_embedding  # [H]


class SetEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 128,
        dropout: float = 0.1,
        number_of_setencoder_mlp_layers: int | None = None,
        set_encoder_norm: bool = False,
    ) -> None:
        super().__init__()
        if number_of_setencoder_mlp_layers is not None and number_of_setencoder_mlp_layers <= 0:
            raise ValueError("number_of_setencoder_mlp_layers must be positive when specified.")
        self.output_dim = output_dim
        self.number_of_setencoder_mlp_layers = number_of_setencoder_mlp_layers
        self.set_encoder_norm = set_encoder_norm

        shared_hidden_layers = number_of_setencoder_mlp_layers or 1
        output_hidden_layers = number_of_setencoder_mlp_layers or 2

        shared_layers: list[nn.Module] = [nn.Linear(input_dim, output_dim)]
        if set_encoder_norm:
            shared_layers.append(nn.LayerNorm(output_dim))
        shared_layers.extend([nn.GELU(), nn.Dropout(dropout)])
        for _ in range(shared_hidden_layers):
            shared_layers.append(nn.Linear(output_dim, output_dim))
            if set_encoder_norm:
                shared_layers.append(nn.LayerNorm(output_dim))
            shared_layers.append(nn.GELU())
        self.shared_mlp = nn.Sequential(*shared_layers)

        output_layers: list[nn.Module] = []
        for layer_index in range(output_hidden_layers):
            output_layers.append(nn.Linear(output_dim, output_dim))
            is_last_layer = layer_index == output_hidden_layers - 1
            if not is_last_layer:
                if set_encoder_norm:
                    output_layers.append(nn.LayerNorm(output_dim))
                output_layers.extend([nn.GELU(), nn.Dropout(dropout)])
        self.output_mlp = nn.Sequential(*output_layers)

    def forward(self, sample_embeddings: torch.Tensor) -> torch.Tensor:
        # sample_embeddings: [N, E]
        if sample_embeddings.ndim != 2:
            raise ValueError(
                f"Expected [num_samples, embedding_dim] tensor, got shape {tuple(sample_embeddings.shape)}"
            )
        encoded_samples = self.shared_mlp(sample_embeddings)  # [N, E] -> [N, 128]
        pooled = encoded_samples.mean(dim=0)  # [128]
        return self.output_mlp(pooled)  # [128]


def build_weight_head(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_hidden_layers: int,
    dropout: float,
    mlp_norm: bool = False,
) -> nn.Sequential:
    if num_hidden_layers <= 0:
        raise ValueError("weight_head_layers must be positive.")
    if hidden_dim <= 0:
        raise ValueError("head_hidden_dim must be positive.")

    layers: list[nn.Module] = []
    current_dim = input_dim
    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(current_dim, hidden_dim))
        if mlp_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.extend([nn.ReLU(), nn.Dropout(dropout)])
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class DSPBuilderMetaModel(nn.Module):
    def __init__(
        self,
        proxy_dim: int,
        num_dataset_classes: int,
        encoder_hidden_dim: int,
        head_hidden_dim: int,
        dropout: float,
        raw_stat_emb: bool,
        number_of_conv1d_layer: int = 1,
        sample_encoder_norm: bool = False,
        number_of_setencoder_mlp_layers: int | None = None,
        set_encoder_norm: bool = False,
        weight_head_layers: int = 1,
        mlp_norm: bool = False,
        dataset_description_dim: int = 128,
    ) -> None:
        super().__init__()
        if num_dataset_classes <= 0:
            raise ValueError("num_dataset_classes must be positive.")
        if proxy_dim <= 0:
            raise ValueError("proxy_dim must be positive.")
        self.support_encoder = FeatureWiseSharedEncoder(
            encoder_hidden_dim=encoder_hidden_dim,
            number_of_conv1d_layer=number_of_conv1d_layer,
            sample_encoder_norm=sample_encoder_norm,
            raw_stat_dim=32,
            raw_stat_emb=raw_stat_emb,
        )
        self.proxy_signature_regression = False
        self.sample_embedding_dim = self.support_encoder.output_dim
        self.dataset_description_dim = dataset_description_dim
        self.task_embedding_dim = dataset_description_dim
        self.weight_head_layers = weight_head_layers
        self.mlp_norm = mlp_norm
        self.set_encoder = SetEncoder(
            input_dim=self.sample_embedding_dim,
            output_dim=dataset_description_dim,
            dropout=dropout,
            number_of_setencoder_mlp_layers=number_of_setencoder_mlp_layers,
            set_encoder_norm=set_encoder_norm,
        )
        self.weight_head = build_weight_head(
            input_dim=dataset_description_dim,
            hidden_dim=head_hidden_dim,
            output_dim=proxy_dim,
            num_hidden_layers=weight_head_layers,
            dropout=dropout,
            mlp_norm=mlp_norm,
        )
        self.dataset_classifier = nn.Sequential(
            nn.Linear(dataset_description_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, num_dataset_classes),
        )
        self.signature_head = nn.Sequential(
            nn.LayerNorm(dataset_description_dim),
            nn.Linear(dataset_description_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, proxy_dim),
        )

    def forward(
        self,
        support_samples: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # support_samples: list of S tensors, each [T, F]
        if not support_samples:
            raise ValueError("support_samples must contain at least one mini sample.")

        sample_embeddings = [self.support_encoder(sample) for sample in support_samples]  # S x [E]
        stacked_embeddings = torch.stack(sample_embeddings, dim=0)  # [S, E]
        dataset_description = self.set_encoder(stacked_embeddings)  # [dataset_description_dim]
        weight_vector = torch.tanh(self.weight_head(dataset_description))  # [proxy_dim]
        dataset_logits = self.dataset_classifier(dataset_description)  # [num_dataset_classes]
        predicted_signature = self.signature_head(dataset_description)  # [proxy_dim]
        return weight_vector, dataset_description, dataset_logits, predicted_signature
