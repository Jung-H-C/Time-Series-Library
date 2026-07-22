from __future__ import annotations

import math
import os
import random
from html import escape
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


EOS_TOKEN = "<EOS>"

PROXY_TOKENS = (
    "MParams",
    "L2-Norm",
    "GFLOPs",
    "Grad_Norm",
    "ZiCo",
    "Fisher",
    "GraSP",
    "Jacov",
    "Jacob_fro",
    "plain",
    "Snip",
    "GSynFlow",
)

UNARY_TOKENS = ("Log", "Sqrt", "Square", "Identity", "Negative")
SEARCH_UNARY_TOKENS = ("Log", "Sqrt", "Square", "Negative")
ALL_BINARY_TOKENS = ("Mul", "Add", "Sub", "Div")
BINARY_TOKENS = ALL_BINARY_TOKENS


def set_div_token_active(active: bool) -> None:
    global BINARY_TOKENS
    BINARY_TOKENS = ALL_BINARY_TOKENS if active else tuple(
        token for token in ALL_BINARY_TOKENS if token != "Div"
    )


def active_binary_tokens() -> tuple[str, ...]:
    return BINARY_TOKENS

TOKEN_ALIASES = {
    "params": "MParams",
    "mparams": "MParams",
    "MParams": "MParams",
    "l2_norm": "L2-Norm",
    "l2-norm": "L2-Norm",
    "L2-Norm": "L2-Norm",
    "flops": "GFLOPs",
    "gflops": "GFLOPs",
    "GFLOPs": "GFLOPs",
    "grad_norm": "Grad_Norm",
    "Grad_Norm": "Grad_Norm",
    "zico": "ZiCo",
    "ZiCo": "ZiCo",
    "fisher": "Fisher",
    "Fisher": "Fisher",
    "grasp": "GraSP",
    "GraSP": "GraSP",
    "jacov": "Jacov",
    "jacob_cov": "Jacov",
    "Jacov": "Jacov",
    "jacob_fro": "Jacob_fro",
    "Jacob_fro": "Jacob_fro",
    "plain": "plain",
    "snip": "Snip",
    "Snip": "Snip",
    "synflow": "GSynFlow",
    "gsynflow": "GSynFlow",
    "GSynFlow": "GSynFlow",
    "log": "Log",
    "Log": "Log",
    "sqrt": "Sqrt",
    "Sqrt": "Sqrt",
    "square": "Square",
    "Square": "Square",
    "identity": "Identity",
    "Identity": "Identity",
    "neg": "Negative",
    "negative": "Negative",
    "Negative": "Negative",
    "mul": "Mul",
    "Mul": "Mul",
    "add": "Add",
    "Add": "Add",
    "sub": "Sub",
    "Sub": "Sub",
    "div": "Div",
    "Div": "Div",
    EOS_TOKEN: EOS_TOKEN,
}

PROXY_TO_COLUMN = {
    "MParams": "MParams",
    "L2-Norm": "l2_norm",
    "GFLOPs": "GFLOPs",
    "Grad_Norm": "grad_norm",
    "ZiCo": "zico",
    "Fisher": "fisher",
    "GraSP": "grasp",
    "Jacov": "jacob_cov",
    "Jacob_fro": "jacob_fro",
    "plain": "plain",
    "Snip": "snip",
    "GSynFlow": "GSynFlow",
}

TOKEN_KIND: dict[str, str] = {}
TOKEN_KIND.update({token: "proxy" for token in PROXY_TOKENS})
TOKEN_KIND.update({token: "unary" for token in UNARY_TOKENS})
TOKEN_KIND.update({token: "binary" for token in ALL_BINARY_TOKENS})
TOKEN_KIND[EOS_TOKEN] = "eos"


@dataclass(frozen=True)
class TreeConstraints:
    max_binary_ops: int = 3
    max_unary_chain: int = 2
    max_tokens: int = 10
    reject_redundant_unary: bool = True


@dataclass(frozen=True)
class SymbolicNode:
    token: str
    children: tuple["SymbolicNode", ...] = ()

    @property
    def kind(self) -> str:
        return TOKEN_KIND[self.token]

    def validate(self, constraints: TreeConstraints = TreeConstraints()) -> None:
        if self.kind == "proxy" and self.children:
            raise ValueError(f"Proxy token '{self.token}' cannot have children.")
        if self.kind == "unary" and len(self.children) != 1:
            raise ValueError(f"Unary token '{self.token}' requires one child.")
        if self.kind == "binary" and len(self.children) != 2:
            raise ValueError(f"Binary token '{self.token}' requires two children.")
        if constraints.reject_redundant_unary:
            redundant_reason = self.redundant_unary_reason()
            if redundant_reason:
                raise ValueError(redundant_reason)

        binary_count = self.count_kind("binary")
        if binary_count > constraints.max_binary_ops:
            raise ValueError(
                f"Formula has {binary_count} binary ops; max is {constraints.max_binary_ops}."
            )
        unary_chain = self.max_unary_chain()
        if unary_chain > constraints.max_unary_chain:
            raise ValueError(
                f"Formula has unary chain length {unary_chain}; max is {constraints.max_unary_chain}."
            )
        token_count = len(self.to_rpn(include_eos=True))
        if token_count > constraints.max_tokens:
            raise ValueError(
                f"Formula has {token_count} RPN tokens including <EOS>; max is {constraints.max_tokens}."
            )
        for child in self.children:
            child.validate(constraints)

    def is_valid(self, constraints: TreeConstraints = TreeConstraints()) -> bool:
        try:
            self.validate(constraints)
        except ValueError:
            return False
        return True

    def count_kind(self, kind: str) -> int:
        return int(self.kind == kind) + sum(child.count_kind(kind) for child in self.children)

    def token_count(self) -> int:
        return len(self.to_rpn(include_eos=False))

    def depth(self) -> int:
        return 1 + max((child.depth() for child in self.children), default=0)

    def max_unary_chain(self, active_chain: int = 0) -> int:
        chain = active_chain + 1 if self.kind == "unary" else 0
        child_max = max((child.max_unary_chain(chain) for child in self.children), default=chain)
        return max(chain, child_max)

    def redundant_unary_reason(self) -> str:
        if self.kind != "unary" or not self.children:
            return ""

        child = self.children[0]
        if child.kind != "unary":
            return ""

        redundant_pairs = {
            ("Negative", "Negative"): "double_negative",
            ("Square", "Square"): "nested_square_rank_equivalent",
            ("Sqrt", "Sqrt"): "nested_sqrt_rank_equivalent",
            ("Identity", "Identity"): "nested_identity",
        }
        reason = redundant_pairs.get((self.token, child.token))
        if not reason:
            return ""
        return f"Redundant unary chain is not allowed: {self.token}({child.token}(.)) [{reason}]."

    def to_rpn(self, include_eos: bool = True) -> tuple[str, ...]:
        tokens: list[str] = []
        for child in self.children:
            tokens.extend(child.to_rpn(include_eos=False))
        tokens.append(self.token)
        if include_eos:
            tokens.append(EOS_TOKEN)
        return tuple(tokens)

    def formula_key(self) -> str:
        return " ".join(self.to_rpn(include_eos=True))

    def paths(self) -> list[tuple[int, ...]]:
        all_paths = [()]
        for index, child in enumerate(self.children):
            all_paths.extend((index, *path) for path in child.paths())
        return all_paths

    def at_path(self, path: Sequence[int]) -> "SymbolicNode":
        node = self
        for index in path:
            node = node.children[index]
        return node

    def replace_at_path(self, path: Sequence[int], replacement: "SymbolicNode") -> "SymbolicNode":
        if not path:
            return replacement
        index = path[0]
        children = list(self.children)
        children[index] = children[index].replace_at_path(path[1:], replacement)
        return SymbolicNode(self.token, tuple(children))

    def evaluate(self, values: Mapping[str, Sequence[float]], eps: float = 1e-12) -> list[float]:
        kind = self.kind
        if kind == "proxy":
            column = PROXY_TO_COLUMN[self.token]
            return [float(value) for value in values[column]]

        if kind == "unary":
            x = self.children[0].evaluate(values, eps=eps)
            if self.token == "Log":
                out = [math.log(abs(value) + eps) for value in x]
            elif self.token == "Sqrt":
                out = [math.sqrt(abs(value) + eps) for value in x]
            elif self.token == "Square":
                out = [clip(value, -1e154, 1e154) ** 2 for value in x]
            elif self.token == "Identity":
                out = x
            elif self.token == "Negative":
                out = [-value for value in x]
            else:
                raise ValueError(f"Unknown unary token: {self.token}")
            return sanitize_values(out)

        if kind == "binary":
            left = self.children[0].evaluate(values, eps=eps)
            right = self.children[1].evaluate(values, eps=eps)
            if self.token == "Add":
                out = [left_value + right_value for left_value, right_value in zip(left, right)]
            elif self.token == "Sub":
                out = [left_value - right_value for left_value, right_value in zip(left, right)]
            elif self.token == "Mul":
                out = [
                    clip(left_value, -1e154, 1e154) * clip(right_value, -1e154, 1e154)
                    for left_value, right_value in zip(left, right)
                ]
            elif self.token == "Div":
                out = []
                for left_value, right_value in zip(left, right):
                    if abs(right_value) < eps:
                        denom = eps if right_value >= 0.0 else -eps
                    else:
                        denom = right_value
                    out.append(left_value / denom)
            else:
                raise ValueError(f"Unknown binary token: {self.token}")
            return sanitize_values(out)

        raise ValueError(f"Cannot evaluate token kind: {kind}")

    def to_infix(self) -> str:
        if self.kind == "proxy":
            return self.token
        if self.kind == "unary":
            child = self.children[0].to_infix()
            if self.token == "Identity":
                return child
            if self.token == "Negative":
                return f"-({child})"
            return f"{self.token.lower()}({child})"

        left = self.children[0].to_infix()
        right = self.children[1].to_infix()
        op = {"Add": "+", "Sub": "-", "Mul": "*", "Div": "/"}[self.token]
        return f"({left} {op} {right})"

    def to_latex(self) -> str:
        if self.kind == "proxy":
            return proxy_latex(self.token)
        if self.kind == "unary":
            child = self.children[0].to_latex()
            if self.token == "Log":
                return rf"\log\left({child}\right)"
            if self.token == "Sqrt":
                return rf"\sqrt{{{child}}}"
            if self.token == "Square":
                return rf"\left({child}\right)^2"
            if self.token == "Identity":
                return child
            if self.token == "Negative":
                return rf"-{child}"
            raise ValueError(f"Unknown unary token: {self.token}")

        left = self.children[0].to_latex()
        right = self.children[1].to_latex()
        if self.token == "Add":
            return rf"{left} + {right}"
        if self.token == "Sub":
            return rf"{left} - {right}"
        if self.token == "Mul":
            return rf"\left({left}\right) \cdot \left({right}\right)"
        if self.token == "Div":
            return rf"\frac{{{left}}}{{{right}}}"
        raise ValueError(f"Unknown binary token: {self.token}")


def clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def sanitize_values(values: Iterable[float]) -> list[float]:
    return [float(value) if math.isfinite(float(value)) else float("nan") for value in values]


def canonical_token(token: str) -> str:
    normalized = token.strip()
    if normalized in TOKEN_ALIASES:
        return TOKEN_ALIASES[normalized]
    lowered = normalized.lower()
    if lowered in TOKEN_ALIASES:
        return TOKEN_ALIASES[lowered]
    raise ValueError(f"Unknown symbolic proxy token: {token!r}")


def parse_rpn(tokens: Iterable[str], constraints: TreeConstraints | None = None) -> SymbolicNode:
    stack: list[SymbolicNode] = []
    for raw_token in tokens:
        token = canonical_token(raw_token)
        if token == EOS_TOKEN:
            break
        kind = TOKEN_KIND[token]
        if kind == "proxy":
            stack.append(SymbolicNode(token))
        elif kind == "unary":
            if not stack:
                raise ValueError(f"Unary token '{token}' has no operand.")
            stack.append(SymbolicNode(token, (stack.pop(),)))
        elif kind == "binary":
            if len(stack) < 2:
                raise ValueError(f"Binary token '{token}' needs two operands.")
            right = stack.pop()
            left = stack.pop()
            stack.append(SymbolicNode(token, (left, right)))
        else:
            raise ValueError(f"Unexpected token kind: {kind}")
    if len(stack) != 1:
        raise ValueError(f"RPN sequence produced {len(stack)} trees instead of one.")
    root = stack[0]
    if constraints is not None:
        root.validate(constraints)
    return root


def tokens_from_string(text: str) -> tuple[str, ...]:
    pieces = text.replace(",", " ").split()
    return tuple(canonical_token(piece) for piece in pieces)


def proxy_latex(token: str) -> str:
    labels = {
        "MParams": r"\mathrm{MParams}",
        "L2-Norm": r"\mathrm{L2Norm}",
        "GFLOPs": r"\mathrm{GFLOPs}",
        "Grad_Norm": r"\mathrm{GradNorm}",
        "ZiCo": r"\mathrm{ZiCo}",
        "Fisher": r"\mathrm{Fisher}",
        "GraSP": r"\mathrm{GraSP}",
        "Jacov": r"\mathrm{Jacov}",
        "Jacob_fro": r"\mathrm{JacobFro}",
        "plain": r"\mathrm{plain}",
        "Snip": r"\mathrm{Snip}",
        "GSynFlow": r"\mathrm{GSynFlow}",
    }
    return labels[token]


def mutate_token_same_kind(node: SymbolicNode, rng: random.Random) -> SymbolicNode:
    choices: Sequence[str]
    if node.kind == "proxy":
        choices = PROXY_TOKENS
    elif node.kind == "unary":
        choices = SEARCH_UNARY_TOKENS
    elif node.kind == "binary":
        choices = BINARY_TOKENS
    else:
        raise ValueError(f"Cannot mutate token kind: {node.kind}")
    next_token = rng.choice([token for token in choices if token != node.token])
    return SymbolicNode(next_token, node.children)


def mutate_token(
    tree: SymbolicNode,
    rng: random.Random,
    constraints: TreeConstraints = TreeConstraints(),
) -> SymbolicNode:
    paths = tree.paths()
    for _ in range(100):
        path = rng.choice(paths)
        mutated = tree.replace_at_path(path, mutate_token_same_kind(tree.at_path(path), rng))
        if mutated.is_valid(constraints):
            return mutated
    raise RuntimeError("Failed to mutate a token under constraints.")


def _random_tree_with_budget(
    rng: random.Random,
    constraints: TreeConstraints,
) -> SymbolicNode:
    if constraints.max_tokens < 2:
        raise RuntimeError("No token budget for a replacement subtree.")
    max_binary_by_tokens = max(0, (constraints.max_tokens - 2) // 2)
    adjusted = TreeConstraints(
        max_binary_ops=min(constraints.max_binary_ops, max_binary_by_tokens),
        max_unary_chain=constraints.max_unary_chain,
        max_tokens=constraints.max_tokens,
        reject_redundant_unary=constraints.reject_redundant_unary,
    )
    return random_valid_tree(rng, constraints=adjusted)


def _replacement_constraints(
    tree: SymbolicNode,
    old_subtree: SymbolicNode,
    constraints: TreeConstraints,
) -> TreeConstraints:
    current_tokens = len(tree.to_rpn(include_eos=False))
    old_tokens = len(old_subtree.to_rpn(include_eos=False))
    max_replacement_tokens = constraints.max_tokens - current_tokens + old_tokens
    max_replacement_binary = (
        constraints.max_binary_ops
        - tree.count_kind("binary")
        + old_subtree.count_kind("binary")
    )
    return TreeConstraints(
        max_binary_ops=max(0, max_replacement_binary),
        max_unary_chain=constraints.max_unary_chain,
        max_tokens=max_replacement_tokens,
        reject_redundant_unary=constraints.reject_redundant_unary,
    )


def mutate_subtree_replace(
    tree: SymbolicNode,
    rng: random.Random,
    constraints: TreeConstraints = TreeConstraints(),
) -> SymbolicNode:
    paths = tree.paths()
    for _ in range(100):
        path = rng.choice(paths)
        old_subtree = tree.at_path(path)
        subtree_constraints = _replacement_constraints(tree, old_subtree, constraints)
        try:
            replacement = _random_tree_with_budget(rng, subtree_constraints)
        except RuntimeError:
            continue
        if replacement.formula_key() == old_subtree.formula_key():
            continue
        mutated = tree.replace_at_path(path, replacement)
        if mutated.is_valid(constraints):
            return mutated
    raise RuntimeError("Failed to replace a subtree under constraints.")


def mutate_unary_wrap(
    tree: SymbolicNode,
    rng: random.Random,
    constraints: TreeConstraints = TreeConstraints(),
) -> SymbolicNode:
    paths = tree.paths()
    for _ in range(100):
        path = rng.choice(paths)
        wrapped = SymbolicNode(rng.choice(SEARCH_UNARY_TOKENS), (tree.at_path(path),))
        mutated = tree.replace_at_path(path, wrapped)
        if mutated.is_valid(constraints):
            return mutated
    raise RuntimeError("Failed to wrap a subtree with a unary op under constraints.")


def mutate_unary_unwrap(
    tree: SymbolicNode,
    rng: random.Random,
    constraints: TreeConstraints = TreeConstraints(),
) -> SymbolicNode:
    paths = [path for path in tree.paths() if tree.at_path(path).kind == "unary"]
    if not paths:
        raise RuntimeError("No unary node to unwrap.")
    for _ in range(100):
        path = rng.choice(paths)
        node = tree.at_path(path)
        mutated = tree.replace_at_path(path, node.children[0])
        if mutated.is_valid(constraints):
            return mutated
    raise RuntimeError("Failed to unwrap a unary op under constraints.")


def mutate_binary_insert(
    tree: SymbolicNode,
    rng: random.Random,
    constraints: TreeConstraints = TreeConstraints(),
) -> SymbolicNode:
    remaining_binary = constraints.max_binary_ops - tree.count_kind("binary")
    if remaining_binary < 1:
        raise RuntimeError("No binary-op budget for insertion.")
    current_tokens = len(tree.to_rpn(include_eos=False))
    operand_constraints = TreeConstraints(
        max_binary_ops=remaining_binary - 1,
        max_unary_chain=constraints.max_unary_chain,
        max_tokens=constraints.max_tokens - current_tokens - 1,
        reject_redundant_unary=constraints.reject_redundant_unary,
    )
    paths = tree.paths()
    for _ in range(100):
        path = rng.choice(paths)
        try:
            operand = _random_tree_with_budget(rng, operand_constraints)
        except RuntimeError:
            continue
        selected = tree.at_path(path)
        if rng.random() < 0.5:
            children = (selected, operand)
        else:
            children = (operand, selected)
        inserted = SymbolicNode(rng.choice(BINARY_TOKENS), children)
        mutated = tree.replace_at_path(path, inserted)
        if mutated.is_valid(constraints):
            return mutated
    raise RuntimeError("Failed to insert a binary op under constraints.")


def mutate_prune(
    tree: SymbolicNode,
    rng: random.Random,
    constraints: TreeConstraints = TreeConstraints(),
) -> SymbolicNode:
    paths = [path for path in tree.paths() if tree.at_path(path).children]
    if not paths:
        raise RuntimeError("No non-leaf node to prune.")
    for _ in range(100):
        path = rng.choice(paths)
        node = tree.at_path(path)
        replacement = rng.choice(node.children)
        mutated = tree.replace_at_path(path, replacement)
        if mutated.is_valid(constraints):
            return mutated
    raise RuntimeError("Failed to prune a subtree under constraints.")


def mutate_tree_once(
    tree: SymbolicNode,
    rng: random.Random,
    constraints: TreeConstraints = TreeConstraints(),
    structural_only: bool = False,
) -> SymbolicNode:
    operators = [
        mutate_subtree_replace,
        mutate_unary_wrap,
        mutate_unary_unwrap,
        mutate_binary_insert,
        mutate_prune,
    ]
    if not structural_only:
        operators.append(mutate_token)
    rng.shuffle(operators)
    for operator in operators:
        try:
            mutated = operator(tree, rng, constraints)
        except RuntimeError:
            continue
        if mutated.is_valid(constraints):
            return mutated
    raise RuntimeError("Failed to apply any mutation operator under constraints.")


def mutate_tree(
    tree: SymbolicNode,
    rng: random.Random,
    constraints: TreeConstraints = TreeConstraints(),
    max_steps: int = 3,
    structural_only: bool = False,
) -> SymbolicNode:
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1.")
    original_key = tree.formula_key()
    for _ in range(300):
        mutated = tree
        steps = 1
        if max_steps > 1 and rng.random() < 0.1:
            steps = rng.randint(2, max_steps)
        try:
            for _step in range(steps):
                mutated = mutate_tree_once(
                    mutated,
                    rng,
                    constraints=constraints,
                    structural_only=structural_only,
                )
        except RuntimeError:
            continue
        if mutated.formula_key() != original_key and mutated.is_valid(constraints):
            return mutated
    raise RuntimeError("Failed to mutate tree under constraints.")


def crossover_trees(
    left: SymbolicNode,
    right: SymbolicNode,
    rng: random.Random,
    constraints: TreeConstraints = TreeConstraints(),
) -> SymbolicNode:
    left_paths = left.paths()
    right_paths = right.paths()
    for _ in range(200):
        left_path = rng.choice(left_paths)
        right_path = rng.choice(right_paths)
        child = left.replace_at_path(left_path, right.at_path(right_path))
        if child.is_valid(constraints):
            return child
    raise RuntimeError("Failed to crossover trees under constraints.")


def random_core_tree(binary_count: int, rng: random.Random) -> SymbolicNode:
    if binary_count == 0:
        return SymbolicNode(rng.choice(PROXY_TOKENS))

    left_binary_count = rng.randint(0, binary_count - 1)
    right_binary_count = binary_count - 1 - left_binary_count
    if rng.random() < 0.5:
        left_count, right_count = left_binary_count, right_binary_count
    else:
        left_count, right_count = right_binary_count, left_binary_count
    return SymbolicNode(
        rng.choice(BINARY_TOKENS),
        (
            random_core_tree(left_count, rng),
            random_core_tree(right_count, rng),
        ),
    )


def decorate_with_unary(
    node: SymbolicNode,
    rng: random.Random,
    unary_budget: int,
    max_chain: int,
) -> tuple[SymbolicNode, int]:
    children: list[SymbolicNode] = []
    budget = unary_budget
    for child in node.children:
        decorated_child, budget = decorate_with_unary(child, rng, budget, max_chain)
        children.append(decorated_child)
    decorated = SymbolicNode(node.token, tuple(children))

    if budget <= 0:
        return decorated, budget
    chain_len = rng.randint(0, min(max_chain, budget))
    for _ in range(chain_len):
        decorated = SymbolicNode(rng.choice(SEARCH_UNARY_TOKENS), (decorated,))
    return decorated, budget - chain_len


def random_valid_tree(
    rng: random.Random,
    constraints: TreeConstraints = TreeConstraints(),
    binary_count: int | None = None,
    allow_unary: bool = True,
) -> SymbolicNode:
    if binary_count is None:
        binary_count = rng.randint(0, constraints.max_binary_ops)
    if binary_count < 0 or binary_count > constraints.max_binary_ops:
        raise ValueError(f"binary_count must be between 0 and {constraints.max_binary_ops}.")

    for _ in range(500):
        core = random_core_tree(binary_count, rng)
        base_tokens = len(core.to_rpn(include_eos=True))
        unary_budget = max(0, constraints.max_tokens - base_tokens) if allow_unary else 0
        # Keep random formulas compact enough to leave room for mutations.
        sampled_budget = rng.randint(0, unary_budget) if unary_budget else 0
        tree, _ = decorate_with_unary(core, rng, sampled_budget, constraints.max_unary_chain)
        if tree.is_valid(constraints):
            return tree
    raise RuntimeError("Failed to sample a valid symbolic proxy tree.")


def single_proxy_trees() -> list[SymbolicNode]:
    return [SymbolicNode(token) for token in PROXY_TOKENS]


def draw_tree(
    tree: SymbolicNode,
    output_path: Path,
    title: str | None = None,
    figsize: tuple[float, float] = (8.0, 4.8),
) -> None:
    os.makedirs("/tmp/matplotlib", exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for PNG tree visualization.") from exc

    positions: dict[tuple[int, ...], tuple[float, float]] = {}
    labels: dict[tuple[int, ...], str] = {}
    leaf_index = 0

    def layout(node: SymbolicNode, path: tuple[int, ...], depth: int) -> float:
        nonlocal leaf_index
        labels[path] = node.token
        if not node.children:
            x = float(leaf_index)
            leaf_index += 1
        else:
            child_xs = [layout(child, (*path, idx), depth + 1) for idx, child in enumerate(node.children)]
            x = sum(child_xs) / len(child_xs)
        positions[path] = (x, -float(depth))
        return x

    layout(tree, (), 0)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    for path, (x, y) in positions.items():
        for idx, _child in enumerate(tree.at_path(path).children):
            child_path = (*path, idx)
            child_x, child_y = positions[child_path]
            ax.plot([x, child_x], [y, child_y], color="#6b7280", linewidth=1.3, zorder=1)

    for path, (x, y) in positions.items():
        token = labels[path]
        kind = TOKEN_KIND[token]
        color = {"proxy": "#e0f2fe", "unary": "#fef3c7", "binary": "#dcfce7"}[kind]
        ax.text(
            x,
            y,
            token,
            ha="center",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.28", "facecolor": color, "edgecolor": "#374151"},
            zorder=2,
        )

    if title:
        ax.set_title(title, fontsize=11)
    ax.axis("off")
    ax.margins(0.15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def draw_tree_svg(
    tree: SymbolicNode,
    output_path: Path,
    title: str | None = None,
) -> None:
    positions: dict[tuple[int, ...], tuple[float, float]] = {}
    labels: dict[tuple[int, ...], str] = {}
    leaf_index = 0

    def layout(node: SymbolicNode, path: tuple[int, ...], depth: int) -> float:
        nonlocal leaf_index
        labels[path] = node.token
        if not node.children:
            x = float(leaf_index)
            leaf_index += 1
        else:
            child_xs = [layout(child, (*path, idx), depth + 1) for idx, child in enumerate(node.children)]
            x = sum(child_xs) / len(child_xs)
        positions[path] = (x, float(depth))
        return x

    layout(tree, (), 0)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_gap = 150.0
    y_gap = 90.0
    margin_x = 70.0
    margin_y = 56.0 if title else 32.0
    max_x = max(x for x, _y in positions.values())
    max_y = max(y for _x, y in positions.values())
    width = max(240.0, max_x * x_gap + margin_x * 2)
    height = max(160.0, max_y * y_gap + margin_y + 48.0)

    def point(path: tuple[int, ...]) -> tuple[float, float]:
        x, y = positions[path]
        return margin_x + x * x_gap, margin_y + y * y_gap

    colors = {"proxy": "#e0f2fe", "unary": "#fef3c7", "binary": "#dcfce7"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    if title:
        lines.append(
            f'<text x="{width / 2:.1f}" y="24" text-anchor="middle" '
            'font-family="Arial, sans-serif" font-size="14" fill="#111827">'
            f"{escape(title)}</text>"
        )

    for path in positions:
        x1, y1 = point(path)
        for index, _child in enumerate(tree.at_path(path).children):
            child_path = (*path, index)
            x2, y2 = point(child_path)
            lines.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                'stroke="#6b7280" stroke-width="1.5"/>'
            )

    for path in positions:
        x, y = point(path)
        token = labels[path]
        fill = colors[TOKEN_KIND[token]]
        rect_width = max(72.0, len(token) * 8.0 + 18.0)
        rect_height = 30.0
        lines.append(
            f'<rect x="{x - rect_width / 2:.1f}" y="{y - rect_height / 2:.1f}" '
            f'width="{rect_width:.1f}" height="{rect_height:.1f}" rx="5" '
            f'fill="{fill}" stroke="#374151" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" '
            'font-family="Arial, sans-serif" font-size="12" fill="#111827">'
            f"{escape(token)}</text>"
        )

    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rounded_fitness(fitness: float, decimals: int = 4) -> str:
    if not math.isfinite(fitness):
        return "-inf"
    return f"{fitness:.{decimals}f}"
