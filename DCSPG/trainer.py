from __future__ import annotations

from typing import Protocol

import torch

from DCSPG.data import MetaBatch
from DCSPG.grammar import RPNGrammar
from DCSPG.model import DCSPGModel
from DCSPG.objectives import (
    autoregressive_cross_entropy_per_sequence,
    make_decoder_inputs,
)
from DCSPG.targets import GroundTruthFormulaTargetProvider
from DCSPG.vocabulary import SymbolicVocabulary


class FormulaTargetProvider(Protocol):
    """Maps a meta-batch to ground-truth symbolic token ids."""

    def get_targets(self, batch: MetaBatch, device: torch.device | str) -> torch.Tensor:
        ...


class NotConfiguredTargetProvider:
    def get_targets(self, batch: MetaBatch, device: torch.device | str) -> torch.Tensor:
        raise NotImplementedError(
            "Formula targets are not configured yet. Implement FormulaTargetProvider "
            "after the proxy formula benchmark is finalized."
        )


class DCSPGTrainer:
    def __init__(
        self,
        model: DCSPGModel,
        vocabulary: SymbolicVocabulary,
        optimizer: torch.optim.Optimizer,
        target_provider: FormulaTargetProvider | None = None,
        device: torch.device | str = "cpu",
        grammar: RPNGrammar | None = None,
        grad_clip: float = 0.0,
    ) -> None:
        self.model = model.to(device)
        self.vocabulary = vocabulary
        self.optimizer = optimizer
        self.target_provider = target_provider or NotConfiguredTargetProvider()
        self.device = device
        self.grammar = grammar
        self.grad_clip = grad_clip

    def train_step(self, batch: MetaBatch) -> dict[str, float]:
        self.model.train()
        batch = batch.to(self.device)
        if isinstance(self.target_provider, GroundTruthFormulaTargetProvider):
            sampled = self.target_provider.sample(batch)
            self.target_provider.last_sampled = sampled
            target_ids = sampled.target_ids.to(self.device)
            target_weights = sampled.target_weights.to(self.device)
        else:
            legacy_targets = self.target_provider.get_targets(batch, self.device)
            target_ids = legacy_targets.unsqueeze(1)
            target_weights = torch.ones(
                target_ids.shape[:2], dtype=torch.float32, device=self.device
            )

        if target_ids.ndim != 3:
            raise ValueError(
                f"Weighted target ids must have shape [B, teachers, T], "
                f"got {tuple(target_ids.shape)}"
            )
        batch_size, teachers_per_episode, target_len = target_ids.shape
        if target_weights.shape != (batch_size, teachers_per_episode):
            raise ValueError(
                f"target_weights must have shape {(batch_size, teachers_per_episode)}, "
                f"got {tuple(target_weights.shape)}"
            )

        flat_targets = target_ids.reshape(batch_size * teachers_per_episode, target_len)
        decoder_inputs = make_decoder_inputs(flat_targets, bos_id=self.vocabulary.bos_id)

        # Encode each support episode once, then reuse its context/memory for
        # all independently weighted teacher formulas from that episode.
        encoder_output = self.model.encode_full(batch.inputs)
        expanded_context = encoder_output.context[:, None, :].expand(
            -1, teachers_per_episode, -1
        )
        flat_context = expanded_context.reshape(
            batch_size * teachers_per_episode,
            encoder_output.context.shape[-1],
        )
        expanded_memory = encoder_output.memory[:, None, :, :].expand(
            -1, teachers_per_episode, -1, -1
        )
        flat_memory = expanded_memory.reshape(
            batch_size * teachers_per_episode,
            encoder_output.memory.shape[1],
            encoder_output.memory.shape[2],
        )
        logits = self.model.decoder(
            decoder_input_ids=decoder_inputs,
            context=flat_context,
            memory=flat_memory,
        )
        if self.grammar is not None:
            logits = self.grammar.mask_logits(logits, decoder_inputs)
        teacher_losses = autoregressive_cross_entropy_per_sequence(
            logits,
            flat_targets,
            pad_id=self.vocabulary.pad_id,
        ).reshape(batch_size, teachers_per_episode)
        normalized_weights = target_weights / (
            target_weights.sum(dim=1, keepdim=True) + 1e-8
        )
        episode_losses = (teacher_losses * normalized_weights).sum(dim=1)
        loss = episode_losses.mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        return {
            "loss": float(loss.detach().cpu()),
            "unweighted_loss": float(teacher_losses.mean().detach().cpu()),
            "mean_teacher_weight": float(target_weights.mean().detach().cpu()),
            "target_len": float(target_len),
            "teachers_per_episode": float(teachers_per_episode),
        }


__all__ = [
    "FormulaTargetProvider",
    "NotConfiguredTargetProvider",
    "GroundTruthFormulaTargetProvider",
    "DCSPGTrainer",
]
