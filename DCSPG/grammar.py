from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from DCSPG.vocabulary import SymbolicVocabulary


DEFAULT_UNARY_TOKENS = ("Square", "Sqrt", "Log", "Negative")
DEFAULT_BINARY_TOKENS = ("Add", "Sub", "Mul", "Div")


@dataclass(frozen=True)
class RPNGrammar:
    vocab_size: int
    pad_id: int
    eos_id: int
    operand_ids: frozenset[int]
    unary_ids: frozenset[int]
    binary_ids: frozenset[int]
    max_stack_depth: int | None = 8
    max_unary_chain: int | None = 2

    @classmethod
    def from_vocabulary(
        cls,
        vocabulary: SymbolicVocabulary,
        unary_tokens: Iterable[str] = DEFAULT_UNARY_TOKENS,
        binary_tokens: Iterable[str] = DEFAULT_BINARY_TOKENS,
        max_stack_depth: int | None = 8,
        max_unary_chain: int | None = 2,
    ) -> "RPNGrammar":
        if max_unary_chain is not None and max_unary_chain <= 0:
            raise ValueError("max_unary_chain must be positive or None")
        special_ids = {
            vocabulary.pad_id,
            vocabulary.bos_id,
            vocabulary.eos_id,
            vocabulary.unk_id,
        }
        unary_ids = frozenset(
            vocabulary.token_to_id_strict(token)
            for token in unary_tokens
            if SymbolicVocabulary.normalize_token(token) in vocabulary.tokens
        )
        binary_ids = frozenset(
            vocabulary.token_to_id_strict(token)
            for token in binary_tokens
            if SymbolicVocabulary.normalize_token(token) in vocabulary.tokens
        )
        operator_ids = unary_ids | binary_ids
        operand_ids = frozenset(
            idx
            for idx, _ in enumerate(vocabulary.tokens)
            if idx not in special_ids and idx not in operator_ids
        )
        if not operand_ids:
            raise ValueError("RPN grammar requires at least one operand token.")
        return cls(
            vocab_size=len(vocabulary),
            pad_id=vocabulary.pad_id,
            eos_id=vocabulary.eos_id,
            operand_ids=operand_ids,
            unary_ids=unary_ids,
            binary_ids=binary_ids,
            max_stack_depth=max_stack_depth,
            max_unary_chain=max_unary_chain,
        )

    def mask_logits(self, logits: torch.Tensor, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        mask = self.valid_next_mask(decoder_input_ids, output_length=logits.shape[1])
        mask = mask.to(device=logits.device)
        neg_inf = torch.finfo(logits.dtype).min
        return logits.masked_fill(~mask, neg_inf)

    def mask_next_logits(
        self,
        logits: torch.Tensor,
        generated_ids: torch.Tensor,
        remaining_steps: int | None = None,
    ) -> torch.Tensor:
        mask = self.valid_next_mask_after_prefix(generated_ids, remaining_steps=remaining_steps)
        mask = mask.to(device=logits.device)
        neg_inf = torch.finfo(logits.dtype).min
        return logits.masked_fill(~mask, neg_inf)

    def valid_next_mask_after_prefix(
        self,
        prefix_ids: torch.Tensor,
        remaining_steps: int | None = None,
    ) -> torch.Tensor:
        if prefix_ids.ndim != 2:
            raise ValueError("prefix_ids must have shape [B, T]")

        prefix_cpu = prefix_ids.detach().cpu()
        batch_size = prefix_cpu.shape[0]
        mask = torch.zeros(batch_size, self.vocab_size, dtype=torch.bool)

        for batch_idx in range(batch_size):
            depth = 0
            finished = False
            unary_chain = 0
            for pos in range(prefix_cpu.shape[1]):
                depth, finished, unary_chain = self._consume(
                    int(prefix_cpu[batch_idx, pos]),
                    depth,
                    finished,
                    unary_chain,
                )
            self._fill_valid_ids(
                mask[batch_idx],
                depth,
                finished,
                unary_chain,
                remaining_steps,
            )
        return mask

    def valid_next_mask(
        self,
        prefix_ids: torch.Tensor,
        output_length: int,
        remaining_steps: int | None = None,
    ) -> torch.Tensor:
        if prefix_ids.ndim != 2:
            raise ValueError("prefix_ids must have shape [B, T]")
        if output_length < 1:
            raise ValueError("output_length must be positive.")

        prefix_cpu = prefix_ids.detach().cpu()
        batch_size = prefix_cpu.shape[0]
        mask = torch.zeros(batch_size, output_length, self.vocab_size, dtype=torch.bool)

        for batch_idx in range(batch_size):
            depth = 0
            finished = False
            unary_chain = 0
            for pos in range(output_length):
                pos_remaining_steps = None
                if remaining_steps is not None:
                    pos_remaining_steps = remaining_steps + output_length - pos - 1
                self._fill_valid_ids(
                    mask[batch_idx, pos],
                    depth,
                    finished,
                    unary_chain,
                    pos_remaining_steps,
                )
                if pos < prefix_cpu.shape[1]:
                    depth, finished, unary_chain = self._consume(
                        int(prefix_cpu[batch_idx, pos]),
                        depth,
                        finished,
                        unary_chain,
                    )
        return mask

    def is_valid_sequence(self, ids: Iterable[int], require_eos: bool = True) -> bool:
        depth = 0
        finished = False
        unary_chain = 0
        saw_eos = False
        for token_id in ids:
            token_id = int(token_id)
            if token_id == self.pad_id:
                continue
            before_finished = finished
            depth, finished, unary_chain = self._consume(
                token_id,
                depth,
                finished,
                unary_chain,
            )
            if token_id == self.eos_id and not before_finished:
                saw_eos = True
            if depth < 0:
                return False
        if require_eos and not saw_eos:
            return False
        return finished and depth == 1

    def stack_depth_after(self, ids: Iterable[int]) -> int:
        depth = 0
        finished = False
        unary_chain = 0
        for token_id in ids:
            depth, finished, unary_chain = self._consume(
                int(token_id),
                depth,
                finished,
                unary_chain,
            )
        return depth

    def _fill_valid_ids(
        self,
        row_mask: torch.Tensor,
        depth: int,
        finished: bool,
        unary_chain: int,
        remaining_steps: int | None = None,
    ) -> None:
        if finished:
            row_mask[self.pad_id] = True
            return

        candidates = []
        if self.max_stack_depth is None or depth < self.max_stack_depth:
            candidates.extend(self.operand_ids)
        unary_allowed = (
            self.max_unary_chain is None
            or unary_chain < self.max_unary_chain
        )
        if depth >= 1 and unary_allowed:
            candidates.extend(self.unary_ids)
        if depth >= 1:
            if depth == 1:
                candidates.append(self.eos_id)
        if depth >= 2:
            candidates.extend(self.binary_ids)

        for token_id in candidates:
            next_depth, next_finished, _next_unary_chain = self._consume(
                token_id,
                depth,
                finished,
                unary_chain,
            )
            if next_depth < 0:
                continue
            if remaining_steps is not None:
                required_steps = self._min_steps_to_finish(next_depth, next_finished)
                if required_steps > remaining_steps:
                    continue
            row_mask[token_id] = True

        if not torch.any(row_mask):
            row_mask[self.pad_id] = True

    def _min_steps_to_finish(self, depth: int, finished: bool) -> int:
        if finished:
            return 0
        if depth <= 0:
            return 2
        if depth == 1:
            return 1
        return depth

    def _consume(
        self,
        token_id: int,
        depth: int,
        finished: bool,
        unary_chain: int,
    ) -> tuple[int, bool, int]:
        if token_id == self.pad_id:
            return depth, finished, unary_chain
        if finished:
            return -1, finished, unary_chain
        if token_id in self.operand_ids:
            return depth + 1, False, 0
        if token_id in self.unary_ids:
            if depth < 1:
                return -1, False, unary_chain
            if (
                self.max_unary_chain is not None
                and unary_chain >= self.max_unary_chain
            ):
                return -1, False, unary_chain
            return depth, False, unary_chain + 1
        if token_id in self.binary_ids:
            return (depth - 1 if depth >= 2 else -1), False, 0
        if token_id == self.eos_id:
            return depth, depth == 1, 0
        return -1, False, unary_chain
