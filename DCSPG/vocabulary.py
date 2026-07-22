from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")

DEFAULT_SYMBOLIC_TOKENS = (
    "Fisher",
    "GFLOPs",
    "GSynFlow",
    "GraSP",
    "Grad_Norm",
    "Jacob_fro",
    "Jacov",
    "L2-Norm",
    "MParams",
    "Snip",
    "ZiCo",
    "plain",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Square",
    "Sqrt",
    "Log",
    "Negative",
)

RAW_EOS_TOKEN = "<EOS>"


@dataclass(frozen=True)
class SymbolicVocabulary:
    tokens: tuple[str, ...]

    @classmethod
    def default(cls, extra_tokens: Iterable[str] = ()) -> "SymbolicVocabulary":
        merged = []
        seen = set()
        for token in (*SPECIAL_TOKENS, *DEFAULT_SYMBOLIC_TOKENS, *tuple(extra_tokens)):
            token = cls.normalize_token(token)
            if token not in seen:
                merged.append(token)
                seen.add(token)
        return cls(tokens=tuple(merged))

    @classmethod
    def from_ground_truth_dir(
        cls,
        ground_truth_dir: Path | str,
        extra_tokens: Iterable[str] = (),
        key: str = "rpn_tokens",
    ) -> "SymbolicVocabulary":
        import numpy as np

        tokens = []
        for path in sorted(Path(ground_truth_dir).glob("*.npz")):
            with np.load(path) as data:
                for formula in data[key].tolist():
                    tokens.extend(str(formula).split())
        return cls.default((*tokens, *tuple(extra_tokens)))

    @staticmethod
    def normalize_token(token: str) -> str:
        if token == RAW_EOS_TOKEN:
            return "<eos>"
        return token

    @property
    def pad_id(self) -> int:
        return self.token_to_id("<pad>")

    @property
    def bos_id(self) -> int:
        return self.token_to_id("<bos>")

    @property
    def eos_id(self) -> int:
        return self.token_to_id("<eos>")

    @property
    def unk_id(self) -> int:
        return self.token_to_id("<unk>")

    def __len__(self) -> int:
        return len(self.tokens)

    def token_to_id(self, token: str) -> int:
        token = self.normalize_token(token)
        try:
            return self.tokens.index(token)
        except ValueError:
            return self.tokens.index("<unk>")

    def token_to_id_strict(self, token: str) -> int:
        token = self.normalize_token(token)
        if token not in self.tokens:
            raise KeyError(f"Unknown token: {token}")
        return self.tokens.index(token)

    def encode(
        self,
        tokens: Iterable[str],
        add_bos: bool = False,
        add_eos: bool = False,
        strict: bool = False,
    ) -> list[int]:
        lookup = self.token_to_id_strict if strict else self.token_to_id
        ids = [lookup(token) for token in tokens]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def encode_rpn(self, rpn_tokens: str, strict: bool = True) -> list[int]:
        ids = self.encode(rpn_tokens.split(), strict=strict)
        if not ids or ids[-1] != self.eos_id:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> list[str]:
        decoded = []
        special = set(SPECIAL_TOKENS)
        for idx in ids:
            token = self.tokens[int(idx)] if 0 <= int(idx) < len(self.tokens) else "<unk>"
            if skip_special and token in special:
                continue
            decoded.append(token)
        return decoded
