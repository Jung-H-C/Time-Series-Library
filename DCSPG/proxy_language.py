from __future__ import annotations

from dataclasses import dataclass

from DCSPG.grammar import DEFAULT_BINARY_TOKENS, DEFAULT_UNARY_TOKENS


@dataclass(frozen=True)
class ParsedRPNFormula:
    tokens: tuple[str, ...]
    max_stack_depth: int


class SymbolicProxyParser:
    """Validates RPN symbolic proxy formulas with stack-depth grammar rules."""

    def __init__(
        self,
        unary_tokens: tuple[str, ...] = DEFAULT_UNARY_TOKENS,
        binary_tokens: tuple[str, ...] = DEFAULT_BINARY_TOKENS,
        eos_token: str = "<EOS>",
    ) -> None:
        self.unary_tokens = set(unary_tokens)
        self.binary_tokens = set(binary_tokens)
        self.eos_tokens = {eos_token, "<eos>"}

    def parse(self, tokens: list[str] | str) -> ParsedRPNFormula:
        if isinstance(tokens, str):
            tokens = tokens.split()

        depth = 0
        max_depth = 0
        finished = False
        parsed = []

        for token in tokens:
            if token in self.eos_tokens:
                if depth != 1:
                    raise ValueError(f"EOS is valid only when stack depth is 1, got {depth}.")
                finished = True
                parsed.append(token)
                break
            if finished:
                raise ValueError("No token is allowed after EOS.")
            if token in self.unary_tokens:
                if depth < 1:
                    raise ValueError(f"Unary operator {token} requires stack depth >= 1.")
            elif token in self.binary_tokens:
                if depth < 2:
                    raise ValueError(f"Binary operator {token} requires stack depth >= 2.")
                depth -= 1
            else:
                depth += 1

            max_depth = max(max_depth, depth)
            parsed.append(token)

        if not finished:
            raise ValueError("Formula must end with <EOS>.")
        return ParsedRPNFormula(tokens=tuple(parsed), max_stack_depth=max_depth)


class SymbolicProxyEvaluator:
    """Interface for connecting parsed formulas to a zero-cost proxy benchmark."""

    def evaluate(self, formula, batch):
        raise NotImplementedError("Connect this evaluator to the benchmark execution code.")
