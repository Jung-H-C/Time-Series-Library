from __future__ import annotations

import torch
from torch import nn


def make_decoder_inputs(target_ids: torch.Tensor, bos_id: int) -> torch.Tensor:
    if target_ids.ndim != 2:
        raise ValueError("target_ids must have shape [B, T]")
    if target_ids.shape[1] == 0:
        raise ValueError("target_ids must contain at least one token.")
    return target_ids[:, :-1]


def autoregressive_cross_entropy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    if logits.shape[:2] != target_ids.shape:
        raise ValueError(
            f"logits time shape {tuple(logits.shape[:2])} must match "
            f"target_ids shape {tuple(target_ids.shape)}"
        )
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_ids.reshape(-1),
        ignore_index=pad_id,
    )


def autoregressive_cross_entropy_per_sequence(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    """Return token-mean autoregressive CE independently for every sequence."""
    if logits.shape[:2] != target_ids.shape:
        raise ValueError(
            f"logits time shape {tuple(logits.shape[:2])} must match "
            f"target_ids shape {tuple(target_ids.shape)}"
        )
    token_losses = nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_ids.reshape(-1),
        ignore_index=pad_id,
        reduction="none",
    ).reshape(target_ids.shape)
    valid = target_ids.ne(pad_id)
    token_counts = valid.sum(dim=1).clamp_min(1)
    return (token_losses * valid).sum(dim=1) / token_counts
