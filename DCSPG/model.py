from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from DCSPG.config import DCSPGConfig
from DCSPG.grammar import RPNGrammar


@dataclass(frozen=True)
class EncoderOutput:
    memory: torch.Tensor
    context: torch.Tensor


class ValueFeatureEmbedding(nn.Module):
    """Embeds [mean, std] statistics and catch22 feature ids into 64-d tokens."""

    def __init__(self, config: DCSPGConfig) -> None:
        super().__init__()
        self.n_features = config.n_features
        self.stat_dim = config.stat_dim
        self.value_embedding = nn.Linear(config.stat_dim, config.d_model)
        self.feature_embedding = nn.Embedding(config.n_features, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        if stats.ndim != 3:
            raise ValueError(f"stats must have shape [B, 22, 2], got {tuple(stats.shape)}")
        if stats.shape[1] != self.n_features or stats.shape[2] != self.stat_dim:
            raise ValueError(f"stats must have shape [B, {self.n_features}, {self.stat_dim}]")

        mean = stats[..., 0]
        std = torch.clamp(stats[..., 1], min=0.0)
        value_input = torch.stack([mean, torch.log1p(std)], dim=-1)

        value_tokens = self.value_embedding(value_input)
        feature_ids = torch.arange(self.n_features, device=stats.device)
        feature_tokens = self.feature_embedding(feature_ids).unsqueeze(0)
        return self.dropout(self.norm(value_tokens + feature_tokens))


class Catch22StatsEncoder(nn.Module):
    def __init__(self, config: DCSPGConfig) -> None:
        super().__init__()
        self.input_embedding = ValueFeatureEmbedding(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.encoder_layers)
        self.output_norm = nn.LayerNorm(config.d_model)

    def forward(self, stats: torch.Tensor) -> EncoderOutput:
        feature_tokens = self.input_embedding(stats)
        cls_token = self.cls_token.expand(feature_tokens.shape[0], -1, -1)
        tokens = torch.cat([cls_token, feature_tokens], dim=1)
        memory = self.output_norm(self.encoder(tokens))
        context = memory[:, 0, :]
        return EncoderOutput(memory=memory, context=context)


class SymbolicAutoregressiveDecoder(nn.Module):
    def __init__(self, config: DCSPGConfig, vocab_size: int) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_formula_len, config.d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=config.decoder_layers)
        self.output_norm = nn.LayerNorm(config.d_model)
        self.output_projection = nn.Linear(config.d_model, vocab_size)

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        context: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        if decoder_input_ids.ndim != 2:
            raise ValueError("decoder_input_ids must have shape [B, T]")
        if context.ndim != 2:
            raise ValueError("context must have shape [B, D]")

        batch_size, prefix_len = decoder_input_ids.shape
        decoder_len = prefix_len + 1
        if decoder_len > self.config.max_formula_len:
            raise ValueError(
                f"Decoder length {decoder_len} exceeds max_formula_len={self.config.max_formula_len}"
            )

        context_position = torch.zeros(batch_size, 1, dtype=torch.long, device=decoder_input_ids.device)
        context_token = context.unsqueeze(1) + self.position_embedding(context_position)
        if prefix_len > 0:
            positions = torch.arange(1, decoder_len, device=decoder_input_ids.device).unsqueeze(0)
            positions = positions.expand(batch_size, prefix_len)
            token_prefix = self.token_embedding(decoder_input_ids) + self.position_embedding(positions)
            tokens = torch.cat([context_token, token_prefix], dim=1)
        else:
            tokens = context_token

        causal_mask = torch.triu(
            torch.ones(decoder_len, decoder_len, device=decoder_input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        decoded = self.decoder(tgt=tokens, memory=memory, tgt_mask=causal_mask)
        return self.output_projection(self.output_norm(decoded))


class DCSPGModel(nn.Module):
    """End-to-end stats-conditioned symbolic proxy generator."""

    def __init__(
        self,
        config: DCSPGConfig,
        vocab_size: int,
        grammar: RPNGrammar | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.grammar = grammar
        self.encoder = Catch22StatsEncoder(config)
        self.decoder = SymbolicAutoregressiveDecoder(config, vocab_size=vocab_size)

    def encode(self, stats: torch.Tensor) -> torch.Tensor:
        return self.encoder(stats).context

    def encode_full(self, stats: torch.Tensor) -> EncoderOutput:
        return self.encoder(stats)

    def forward(
        self,
        stats: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        grammar: RPNGrammar | None = None,
    ) -> torch.Tensor:
        encoder_output = self.encode_full(stats)
        logits = self.decoder(
            decoder_input_ids=decoder_input_ids,
            context=encoder_output.context,
            memory=encoder_output.memory,
        )
        grammar = grammar or self.grammar
        if grammar is not None:
            logits = grammar.mask_logits(logits, decoder_input_ids)
        return logits

    @torch.no_grad()
    def generate(
        self,
        stats: torch.Tensor,
        bos_id: int,
        eos_id: int,
        pad_id: int | None = None,
        max_len: int | None = None,
        temperature: float = 1.0,
        greedy: bool = True,
        grammar: RPNGrammar | None = None,
    ) -> torch.Tensor:
        self.eval()
        max_len = max_len or self.config.max_formula_len
        if max_len < 1:
            raise ValueError("max_len must be positive.")

        batch_size = stats.shape[0]
        generated = torch.empty(
            (batch_size, 0),
            dtype=torch.long,
            device=stats.device,
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=stats.device)
        encoder_output = self.encode_full(stats)
        grammar = grammar or self.grammar
        pad_id = eos_id if pad_id is None else pad_id

        for step in range(max_len):
            logits = self.decoder(
                generated,
                context=encoder_output.context,
                memory=encoder_output.memory,
            )[:, -1, :]
            if grammar is not None:
                logits = grammar.mask_next_logits(
                    logits,
                    generated,
                    remaining_steps=max_len - step - 1,
                )
            logits = logits / max(temperature, 1e-6)
            if greedy:
                next_ids = torch.argmax(logits, dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1).squeeze(1)

            next_ids = torch.where(finished, torch.full_like(next_ids, pad_id), next_ids)
            generated = torch.cat([generated, next_ids.unsqueeze(1)], dim=1)
            finished = finished | (next_ids == eos_id)
            if torch.all(finished):
                break

        return generated
