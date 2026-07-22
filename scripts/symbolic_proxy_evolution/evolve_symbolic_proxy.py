from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

try:
    from symbolic_tree import (
        PROXY_TOKENS,
        PROXY_TO_COLUMN,
        SymbolicNode,
        TreeConstraints,
        active_binary_tokens,
        crossover_trees,
        draw_tree,
        draw_tree_svg,
        mutate_tree,
        parse_rpn,
        random_valid_tree,
        rounded_fitness,
        set_div_token_active,
        single_proxy_trees,
    )
except ImportError:  # pragma: no cover - supports package-style execution.
    from .symbolic_tree import (
        PROXY_TOKENS,
        PROXY_TO_COLUMN,
        SymbolicNode,
        TreeConstraints,
        active_binary_tokens,
        crossover_trees,
        draw_tree,
        draw_tree_svg,
        mutate_tree,
        parse_rpn,
        random_valid_tree,
        rounded_fitness,
        set_div_token_active,
        single_proxy_trees,
    )


DEFAULT_BACKBONE_ALIASES = {
    "autoformer": "autoformer",
    "mamba": "mamba",
}

DEFAULT_DATASET_ALIASES = {
    "ecl": "ECL",
    "etth1": "ETTh1",
    "exchange": "Exchange",
    "ili": "ILI",
    "traffic": "Traffic",
    "weather": "Weather",
}

_WORKER_VALUES: dict[str, list[float]] | None = None
_WORKER_TARGET: list[float] | None = None
_WORKER_VALIDATION_VALUES: dict[str, list[float]] | None = None
_WORKER_VALIDATION_TARGET: list[float] | None = None
_WORKER_PROXY_SCORE_DECIMALS: int | None = None
_WORKER_MAX_ABS_PROXY_SCORE: float | None = None
_WORKER_FITNESS_DECIMALS: int | None = None
_WORKER_FOLD_INDICES: tuple[tuple[int, ...], ...] | None = None
_WORKER_TRAIN_FITNESS_MODE: str | None = None
_WORKER_SOFT_DIV: bool | None = None
_WORKER_DENOMINATOR: float | None = None

FITNESS_KIND_LABEL = "original"
FITNESS_FOLD_COUNT = 2
FITNESS_FOLD_SIZE = 25
SPLIT_COLUMN_CANDIDATES = ("split", "model_label")
DEFAULT_VALIDATION_SPLIT = "proxy_valid"
DEFAULT_TRAIN_FITNESS_MODE = "direct"
TRAIN_FITNESS_MODES = ("folded", "direct")
VALIDATION_SPLIT_ALIASES = {
    "proxy_valid": ("proxy_valid", "proxy_val"),
    "proxy_val": ("proxy_val", "proxy_valid"),
}
INITIAL_PAIRWISE_BINARY_TOKENS = ("Add", "Mul", "Sub", "Div")
DEFAULT_EXPLORATION_BINARY_QUOTAS = ((0, 5), (1, 10), (2, 25), (3, 60))
EXPLOITATION_EXPLORATION_BINARY_QUOTAS = ((3, 10),)
EXPLOITATION_LOCAL_NEIGHBOR_COUNT = 90
LOCAL_NEIGHBOR_PARENT_COUNT = 3


@dataclass(frozen=True)
class EvolutionConfig:
    backbone: str
    dataset: str
    csv_path: Path
    output_dir: Path
    seed: int
    max_generations: int = 100
    population_size: int = 200
    elite_count: int = 5
    mutation_parent_pool: int = 30
    top30_mutations: int = 45
    archive_mutations: int = 25
    crossover_count: int = 25
    crossover_parent_pool: int = 30
    local_neighbor_count: int = 10
    exploration_binary_quotas: tuple[tuple[int, int], ...] = DEFAULT_EXPLORATION_BINARY_QUOTAS
    archive_size: int = 1000
    archive_threshold_margin: float = 0.05
    target_metric: str = "mse"
    target_direction: str = "minimize"
    train_fitness_mode: str = DEFAULT_TRAIN_FITNESS_MODE
    validation_split: str = DEFAULT_VALIDATION_SPLIT
    ablation_a: bool = False
    ablation_a_gamma: float = 0.5
    fitness_decimals: int = 4
    constraints: TreeConstraints = TreeConstraints()
    visualize_top_k: int = 20
    num_workers: int = 0
    proxy_score_decimals: int | None = 12
    max_abs_proxy_score: float = 1e12
    div_token: bool = True
    soft_div: bool = False
    denominator: float = 1e-8

    @property
    def non_exploration_population_count(self) -> int:
        return self.non_exploration_population_count_for_generation(1)

    def non_exploration_population_count_for_generation(self, generation: int) -> int:
        return (
            self.elite_count
            + self.top30_mutations
            + self.archive_mutations
            + self.crossover_count
            + self.local_neighbor_count_for_generation(generation)
        )

    @property
    def exploration_random_count(self) -> int:
        return self.exploration_random_count_for_generation(1)

    def exploration_random_count_for_generation(self, generation: int) -> int:
        return max(
            0,
            self.population_size - self.non_exploration_population_count_for_generation(generation),
        )

    @property
    def exploitation_start_generation(self) -> int:
        return max(1, math.ceil(self.max_generations / 2))

    def is_exploitation_generation(self, generation: int) -> bool:
        return generation >= self.exploitation_start_generation

    def local_neighbor_count_for_generation(self, generation: int) -> int:
        if self.is_exploitation_generation(generation):
            return EXPLOITATION_LOCAL_NEIGHBOR_COUNT
        return self.local_neighbor_count

    def exploration_binary_quotas_for_generation(
        self,
        generation: int,
    ) -> tuple[tuple[int, int], ...]:
        if self.is_exploitation_generation(generation):
            return EXPLOITATION_EXPLORATION_BINARY_QUOTAS
        return self.exploration_binary_quotas

    def effective_exploration_binary_quotas(self) -> tuple[tuple[int, int], ...]:
        return self.effective_exploration_binary_quotas_for_generation(1)

    def effective_exploration_binary_quotas_for_generation(
        self,
        generation: int,
    ) -> tuple[tuple[int, int], ...]:
        remaining = self.exploration_random_count_for_generation(generation)
        effective: list[tuple[int, int]] = []
        for binary_count, requested_count in self.exploration_binary_quotas_for_generation(
            generation
        ):
            if remaining <= 0:
                break
            count = min(requested_count, remaining)
            if count > 0:
                effective.append((binary_count, count))
            remaining -= count
        return tuple(effective)

    def validate(self) -> None:
        required = self.non_exploration_population_count
        if required > self.population_size:
            raise ValueError(
                "Non-exploration selection counts must fit within population-size; "
                f"got population_size={self.population_size}, non_exploration_total={required}."
            )
        exploitation_required = self.non_exploration_population_count_for_generation(
            self.exploitation_start_generation
        )
        if exploitation_required > self.population_size:
            raise ValueError(
                "Exploitation-phase non-exploration selection counts must fit within "
                "population-size; "
                f"got population_size={self.population_size}, "
                f"non_exploration_total={exploitation_required}."
            )
        if self.elite_count < 1:
            raise ValueError("--elite-count must be >= 1.")
        if self.mutation_parent_pool < 1:
            raise ValueError("--mutation-parent-pool must be >= 1.")
        if self.crossover_parent_pool < 2:
            raise ValueError("--crossover-parent-pool must be >= 2.")
        if self.target_direction not in {"minimize", "maximize"}:
            raise ValueError("--target-direction must be one of: minimize, maximize.")
        if self.train_fitness_mode not in TRAIN_FITNESS_MODES:
            modes = ", ".join(TRAIN_FITNESS_MODES)
            raise ValueError(f"--train-fitness-mode must be one of: {modes}.")
        if not self.validation_split.strip():
            raise ValueError("--validation-split must be non-empty.")
        if self.ablation_a_gamma < 0.0 or not math.isfinite(self.ablation_a_gamma):
            raise ValueError("--ablationA-gamma must be a non-negative finite value.")
        if self.num_workers < 0:
            raise ValueError("--num-workers must be >= 0. Use 0 for auto.")
        if self.proxy_score_decimals is not None and self.proxy_score_decimals < 0:
            raise ValueError("--proxy-score-decimals must be >= 0, or -1 to disable rounding.")
        if self.max_abs_proxy_score <= 0.0 or not math.isfinite(self.max_abs_proxy_score):
            raise ValueError("--max-abs-proxy-score must be a positive finite value.")
        for binary_count, count in self.exploration_binary_quotas:
            if binary_count < 0:
                raise ValueError("Exploration binary-count quotas must use non-negative binary counts.")
            if count < 0:
                raise ValueError("Exploration binary-count quotas must use non-negative counts.")
        if self.denominator <= 0.0 or not math.isfinite(self.denominator):
            raise ValueError("--denominator must be a positive finite value.")


@dataclass(frozen=True)
class FormulaEvaluation:
    tree: SymbolicNode
    raw_fitness: float
    resample_fitness_mean: float
    resample_fitness_quantile: float
    resample_fitness_variance: float
    resample_fitness_std: float
    resample_invalid_count: int
    objective_fitness: float
    token_count: int
    depth: int
    generation: int
    source: str
    validation_fitness: float
    validation_invalid_reason: str = ""
    invalid_reason: str = ""

    @property
    def fitness(self) -> float:
        return self.raw_fitness

    @property
    def key(self) -> str:
        return self.tree.formula_key()

    @property
    def is_valid(self) -> bool:
        return (
            not self.invalid_reason
            and math.isfinite(self.raw_fitness)
            and math.isfinite(self.resample_fitness_mean)
            and math.isfinite(self.resample_fitness_quantile)
            and math.isfinite(self.resample_fitness_std)
            and math.isfinite(self.objective_fitness)
        )

    @property
    def validation_is_valid(self) -> bool:
        return (
            self.is_valid
            and not self.validation_invalid_reason
            and math.isfinite(self.validation_fitness)
        )


def rounded_fitness_value(fitness: float, decimals: int) -> float:
    if not math.isfinite(fitness):
        return float("-inf")
    return float(f"{fitness:.{decimals}f}")


def make_formula_evaluation(
    tree: SymbolicNode,
    raw_fitness: float,
    resample_fitness_mean: float,
    resample_fitness_quantile: float,
    resample_fitness_variance: float,
    resample_invalid_count: int,
    generation: int,
    source: str,
    fitness_decimals: int,
    validation_fitness: float = float("-inf"),
    validation_invalid_reason: str = "",
    invalid_reason: str = "",
) -> FormulaEvaluation:
    token_count = tree.token_count()
    depth = tree.depth()
    resample_fitness_std = (
        math.sqrt(resample_fitness_variance)
        if math.isfinite(resample_fitness_variance) and resample_fitness_variance >= 0.0
        else float("inf")
    )
    objective_fitness = rounded_fitness_value(raw_fitness, fitness_decimals)
    if invalid_reason:
        objective_fitness = float("-inf")
    return FormulaEvaluation(
        tree=tree,
        raw_fitness=raw_fitness,
        resample_fitness_mean=resample_fitness_mean,
        resample_fitness_quantile=resample_fitness_quantile,
        resample_fitness_variance=resample_fitness_variance,
        resample_fitness_std=resample_fitness_std,
        resample_invalid_count=resample_invalid_count,
        objective_fitness=objective_fitness,
        token_count=token_count,
        depth=depth,
        generation=generation,
        source=source,
        validation_fitness=validation_fitness,
        validation_invalid_reason=validation_invalid_reason,
        invalid_reason=invalid_reason,
    )


def evaluation_sort_key(evaluation: FormulaEvaluation) -> tuple[float, float, int, int, float, float, float]:
    # Exclude the formula key so exact numeric ties keep the existing/archive order.
    if not evaluation.is_valid:
        return (
            float("-inf"),
            float("-inf"),
            -evaluation.token_count,
            -evaluation.depth,
            float("-inf"),
            float("-inf"),
            float("-inf"),
        )
    return (
        evaluation.objective_fitness,
        -evaluation.resample_fitness_std,
        -evaluation.token_count,
        -evaluation.depth,
        evaluation.resample_fitness_mean,
        evaluation.resample_fitness_quantile,
        evaluation.raw_fitness,
    )


def archive_admission_sort_key(
    evaluation: FormulaEvaluation,
) -> tuple[float, int, int, float, float, float, float]:
    # Archive admission is still based on proxy_train fitness. Ties on rounded
    # train fitness prefer lower-complexity formulas first.
    if not evaluation.is_valid:
        return (
            float("-inf"),
            -evaluation.token_count,
            -evaluation.depth,
            float("-inf"),
            float("-inf"),
            float("-inf"),
            float("-inf"),
        )
    return (
        evaluation.objective_fitness,
        -evaluation.token_count,
        -evaluation.depth,
        -evaluation.resample_fitness_std,
        evaluation.resample_fitness_mean,
        evaluation.resample_fitness_quantile,
        evaluation.raw_fitness,
    )


def archive_score(
    evaluation: FormulaEvaluation,
    ablation_a: bool = False,
    gamma: float = 0.5,
) -> float:
    if not evaluation.is_valid or not math.isfinite(evaluation.validation_fitness):
        return float("-inf")
    if not ablation_a:
        return evaluation.validation_fitness
    train_fitness = evaluation.objective_fitness
    valid_fitness = evaluation.validation_fitness
    if not math.isfinite(train_fitness) or not math.isfinite(valid_fitness):
        return float("-inf")
    return 0.5 * (train_fitness + valid_fitness) - gamma * abs(
        train_fitness - valid_fitness
    )


def archive_sort_key(
    evaluation: FormulaEvaluation,
    ablation_a: bool = False,
    gamma: float = 0.5,
) -> tuple[float, float, float, int, int, float, float]:
    # Final archive ranking is based on Score. By default Score is direct proxy
    # validation Spearman; ablationA balances proxy_train and proxy_valid.
    score = archive_score(evaluation, ablation_a=ablation_a, gamma=gamma)
    if not math.isfinite(score):
        return (
            float("-inf"),
            float("-inf"),
            float("-inf"),
            -evaluation.token_count,
            -evaluation.depth,
            float("-inf"),
            float("-inf"),
        )
    return (
        score,
        evaluation.validation_fitness,
        evaluation.objective_fitness,
        -evaluation.token_count,
        -evaluation.depth,
        -evaluation.resample_fitness_std,
        evaluation.raw_fitness,
    )


CachedEvaluation = tuple[
    SymbolicNode,
    float,
    float,
    float,
    float,
    float,
    int,
    float,
    int,
    int,
    float,
    str,
    str,
]


class FormulaArchive:
    def __init__(
        self,
        max_size: int,
        fitness_decimals: int = 4,
        fitness_kind: str = FITNESS_KIND_LABEL,
        ablation_a: bool = False,
        ablation_a_gamma: float = 0.5,
    ) -> None:
        self.max_size = max_size
        self.fitness_decimals = fitness_decimals
        self.fitness_kind = fitness_kind
        self.ablation_a = ablation_a
        self.ablation_a_gamma = ablation_a_gamma
        self._records: dict[str, FormulaEvaluation] = {}

    def __len__(self) -> int:
        return len(self._records)

    def add(self, evaluation: FormulaEvaluation) -> bool:
        if not evaluation.is_valid:
            return False

        evaluation_key = archive_admission_sort_key(evaluation)
        existing_formula = self._records.get(evaluation.key)
        if (
            existing_formula is not None
            and evaluation_key <= archive_admission_sort_key(existing_formula)
        ):
            return False

        existing_same_fitness = self._best_for_objective_fitness(evaluation.objective_fitness)
        if (
            existing_same_fitness is not None
            and evaluation_key <= archive_admission_sort_key(existing_same_fitness)
        ):
            return False

        if existing_formula is not None:
            self._records.pop(existing_formula.key, None)
        if existing_same_fitness is not None:
            self._drop_objective_fitness(evaluation.objective_fitness)
        self._records[evaluation.key] = evaluation
        self._prune()
        return True

    def add_many(self, evaluations: Iterable[FormulaEvaluation]) -> int:
        return sum(1 for evaluation in evaluations if self.add(evaluation))

    def ranked(self) -> list[FormulaEvaluation]:
        return sorted(
            self._records.values(),
            key=lambda evaluation: archive_sort_key(
                evaluation,
                ablation_a=self.ablation_a,
                gamma=self.ablation_a_gamma,
            ),
            reverse=True,
        )

    def score(self, evaluation: FormulaEvaluation) -> float:
        return archive_score(
            evaluation,
            ablation_a=self.ablation_a,
            gamma=self.ablation_a_gamma,
        )

    def train_ranked(self) -> list[FormulaEvaluation]:
        return sorted(
            self._records.values(),
            key=archive_admission_sort_key,
            reverse=True,
        )

    def sample_trees(self, count: int, rng: random.Random) -> list[SymbolicNode]:
        ranked = self.train_ranked()
        if not ranked or count <= 0:
            return []
        selected = ranked[: min(count, len(ranked))]
        rng.shuffle(selected)
        return [entry.tree for entry in selected]

    def save_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for rank, evaluation in enumerate(self.ranked(), start=1):
            rows.append(
                archive_row(
                    rank,
                    evaluation,
                    self.fitness_kind,
                    self.fitness_decimals,
                    self.score(evaluation),
                )
            )
        with path.open("w", newline="", encoding="utf-8") as handle:
            if self.fitness_kind == "original":
                fieldnames = [
                    "rank",
                    "generation",
                    "fitness_kind",
                    "Score",
                    "fitness",
                    "validation_fitness",
                    "validation_invalid_reason",
                    "token_count",
                    "depth",
                    "source",
                    "rpn_tokens",
                    "inflix",
                    "latex",
                ]
            else:
                fieldnames = [
                    "rank",
                    "generation",
                    "fitness_kind",
                    "Score",
                    "objective_fitness",
                    "raw_fitness",
                    "validation_fitness",
                    "validation_invalid_reason",
                    "token_count",
                    "depth",
                    "source",
                    "rpn_tokens",
                    "inflix",
                    "latex",
                ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def save_latex(self, path: Path, top_k: int | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ranked = self.ranked()
        if top_k is not None:
            ranked = ranked[:top_k]
        lines = [
            r"\begin{tabular}{r r r r r l}",
            r"Rank & Generation & Score & Validation Fitness & Train Fitness & Formula \\",
            r"\hline",
        ]
        for rank, evaluation in enumerate(ranked, start=1):
            score = rounded_fitness(self.score(evaluation), self.fitness_decimals)
            validation_fitness = rounded_fitness(
                evaluation.validation_fitness,
                self.fitness_decimals,
            )
            train_fitness = rounded_fitness(evaluation.objective_fitness, self.fitness_decimals)
            latex = evaluation.tree.to_latex()
            lines.append(
                rf"{rank} & {evaluation.generation} & {score} & {validation_fitness} & {train_fitness} & ${latex}$ \\"
            )
        lines.append(r"\end{tabular}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def visualize(self, output_dir: Path | str, top_k: int | None = None) -> None:
        output_dir = Path(output_dir)
        ranked = self.ranked()
        if top_k is not None:
            ranked = ranked[:top_k]
        svg_dir = output_dir / "tree_svg"
        png_dir = output_dir / "tree_png"
        svg_dir.mkdir(parents=True, exist_ok=True)
        for rank, evaluation in enumerate(ranked, start=1):
            score = rounded_fitness(self.score(evaluation), self.fitness_decimals)
            stem = (
                f"rank_{rank:03d}_score_"
                f"{score}"
            )
            title = (
                f"fitness_train={rounded_fitness(evaluation.objective_fitness, self.fitness_decimals)} "
                f"| fitness_valid={rounded_fitness(evaluation.validation_fitness, self.fitness_decimals)} "
                f"| score={score}"
            )
            draw_tree_svg(
                evaluation.tree,
                svg_dir / f"{stem}.svg",
                title=title,
            )
            try:
                png_dir.mkdir(parents=True, exist_ok=True)
                draw_tree(evaluation.tree, png_dir / f"{stem}.png", title=title)
            except RuntimeError as exc:
                # SVG visualization is dependency-free. PNG is optional because
                # headless Python environments may not include matplotlib.
                marker = output_dir / "png_visualization_skipped.txt"
                marker.write_text(f"{exc}\n", encoding="utf-8")
                if not any(png_dir.iterdir()):
                    png_dir.rmdir()
                break
        self.save_latex(output_dir / "archive_latex.tex", top_k=top_k)

    def _prune(self) -> None:
        records: Iterable[FormulaEvaluation] = self._records.values()
        best_by_fitness: dict[float, FormulaEvaluation] = {}
        for entry in records:
            existing = best_by_fitness.get(entry.objective_fitness)
            if (
                existing is None
                or archive_admission_sort_key(entry) > archive_admission_sort_key(existing)
            ):
                best_by_fitness[entry.objective_fitness] = entry
        ranked = sorted(
            best_by_fitness.values(),
            key=archive_admission_sort_key,
            reverse=True,
        )
        self._records = {entry.key: entry for entry in ranked[: self.max_size]}

    def _best_for_objective_fitness(self, objective_fitness: float) -> FormulaEvaluation | None:
        matches = (
            entry
            for entry in self._records.values()
            if entry.objective_fitness == objective_fitness
        )
        return max(matches, key=archive_admission_sort_key, default=None)

    def _drop_objective_fitness(self, objective_fitness: float) -> None:
        for key in [
            entry.key
            for entry in self._records.values()
            if entry.objective_fitness == objective_fitness
        ]:
            self._records.pop(key, None)


def archive_row(
    rank: int,
    evaluation: FormulaEvaluation,
    fitness_kind: str,
    fitness_decimals: int,
    score: float,
) -> dict[str, str | int | float]:
    row: dict[str, str | int | float] = {
        "rank": rank,
        "generation": evaluation.generation,
        "fitness_kind": fitness_kind,
        "Score": rounded_fitness(score, fitness_decimals),
        "token_count": evaluation.token_count,
        "depth": evaluation.depth,
        "source": evaluation.source,
        "rpn_tokens": " ".join(evaluation.tree.to_rpn(include_eos=True)),
        "inflix": evaluation.tree.to_infix(),
        "latex": evaluation.tree.to_latex(),
    }
    if fitness_kind == "original":
        row["fitness"] = rounded_fitness(evaluation.objective_fitness, fitness_decimals)
        row["validation_fitness"] = rounded_fitness(
            evaluation.validation_fitness,
            fitness_decimals,
        )
        row["validation_invalid_reason"] = evaluation.validation_invalid_reason
    else:
        row["objective_fitness"] = rounded_fitness(evaluation.objective_fitness, fitness_decimals)
        row["raw_fitness"] = rounded_fitness(evaluation.raw_fitness, fitness_decimals)
        row["validation_fitness"] = rounded_fitness(
            evaluation.validation_fitness,
            fitness_decimals,
        )
        row["validation_invalid_reason"] = evaluation.validation_invalid_reason
    return row


def normalize_backbone(name: str) -> str:
    compact = name.replace("_", "").replace("-", "").lower()
    if compact in DEFAULT_BACKBONE_ALIASES:
        return DEFAULT_BACKBONE_ALIASES[compact]
    return name.lower()


def normalize_dataset(name: str) -> str:
    compact = name.replace("_", "").replace("-", "").lower()
    if compact in DEFAULT_DATASET_ALIASES:
        return DEFAULT_DATASET_ALIASES[compact]
    return name


def find_groundtruth_csv(groundtruth_dir: Path, backbone: str, dataset: str) -> Path:
    backbone_dir = groundtruth_dir / normalize_backbone(backbone)
    # Support both the historical GroundTruth/<backbone>/ layout and a flat
    # directory such as proxy_scores/monash_time/. Multi-dataset launchers can
    # still use --csv-path to make the mapping completely explicit.
    search_dir = backbone_dir if backbone_dir.is_dir() else groundtruth_dir
    if not search_dir.is_dir():
        raise FileNotFoundError(f"GroundTruth directory not found: {groundtruth_dir}")
    dataset_name = normalize_dataset(dataset)
    candidates = sorted(search_dir.glob(f"*_{dataset_name}_proxy_scores_*.csv"))
    if not candidates:
        candidates = sorted(
            path for path in search_dir.glob("*.csv") if f"_{dataset_name}_" in path.name
        )
    if len(candidates) != 1:
        joined = ", ".join(str(path) for path in candidates) if candidates else "none"
        raise FileNotFoundError(
            f"Expected exactly one GroundTruth CSV for backbone={backbone}, dataset={dataset_name}; found {joined}."
        )
    return candidates[0]


def resolve_split_column(fieldnames: Sequence[str], csv_path: Path) -> str:
    for column in SPLIT_COLUMN_CANDIDATES:
        if column in fieldnames:
            return column
    expected = ", ".join(SPLIT_COLUMN_CANDIDATES)
    raise ValueError(f"{csv_path} is missing required split column; expected one of: {expected}.")


def split_name_candidates(split: str) -> tuple[str, ...]:
    normalized = split.strip()
    aliases = VALIDATION_SPLIT_ALIASES.get(normalized)
    if aliases is not None:
        return aliases
    return (normalized,)


def load_proxy_split_benchmark(
    csv_path: Path,
    split: str,
    target_metric: str,
    target_direction: str,
) -> tuple[dict[str, list[float]], list[float], str]:
    rows: list[dict[str, str]] = []
    requested_splits = split_name_candidates(split)
    matched_split = requested_splits[0]
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in PROXY_TO_COLUMN.values() if column not in fieldnames]
        if missing:
            raise ValueError(f"{csv_path} is missing proxy columns: {', '.join(missing)}")
        if target_metric not in fieldnames:
            raise ValueError(f"{csv_path} is missing target metric column: {target_metric}")
        split_column = resolve_split_column(fieldnames, csv_path)
        all_rows = list(reader)

    for candidate_split in requested_splits:
        rows = [row for row in all_rows if row.get(split_column) == candidate_split]
        if rows:
            matched_split = candidate_split
            break

    if len(rows) < 3:
        candidates = ", ".join(repr(candidate) for candidate in requested_splits)
        raise ValueError(
            f"Need at least 3 rows for split={candidates} to compute Spearman correlation; "
            f"got {len(rows)}."
        )

    values: dict[str, list[float]] = {}
    for column in sorted(set(PROXY_TO_COLUMN.values())):
        values[column] = [safe_float(row[column]) for row in rows]

    target = [safe_float(row[target_metric]) for row in rows]
    if target_direction == "minimize":
        target = [-value for value in target]
    return values, target, matched_split


def load_proxy_train_benchmark(csv_path: Path, target_metric: str, target_direction: str) -> tuple[dict[str, list[float]], list[float]]:
    values, target, _matched_split = load_proxy_split_benchmark(
        csv_path,
        split="proxy_train",
        target_metric=target_metric,
        target_direction=target_direction,
    )
    return values, target


def safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def rankdata_average(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    sorted_values = [values[index] for index in order]
    index = 0
    while index < len(values):
        next_index = index + 1
        while next_index < len(values) and sorted_values[next_index] == sorted_values[index]:
            next_index += 1
        # Rank positions are 1-based; ties get the average rank.
        avg_rank = (index + 1 + next_index) / 2.0
        for original_index in order[index:next_index]:
            ranks[original_index] = avg_rank
        index = next_index
    return ranks


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    pairs = [
        (float(left), float(right))
        for left, right in zip(x, y)
        if math.isfinite(float(left)) and math.isfinite(float(right))
    ]
    if len(pairs) < 3:
        return float("-inf")
    x_values = [left for left, _right in pairs]
    y_values = [right for _left, right in pairs]
    rx = rankdata_average(x_values)
    ry = rankdata_average(y_values)
    rx_mean = sum(rx) / len(rx)
    ry_mean = sum(ry) / len(ry)
    rx_centered = [value - rx_mean for value in rx]
    ry_centered = [value - ry_mean for value in ry]
    denom = math.sqrt(
        sum(value * value for value in rx_centered)
        * sum(value * value for value in ry_centered)
    )
    if denom == 0.0:
        return float("-inf")
    return sum(left * right for left, right in zip(rx_centered, ry_centered)) / denom


def normalize_proxy_scores(
    scores: Sequence[float],
    proxy_score_decimals: int | None,
    max_abs_proxy_score: float,
) -> tuple[list[float], str]:
    normalized: list[float] = []
    for index, score in enumerate(scores):
        value = float(score)
        if not math.isfinite(value):
            return [], f"nonfinite_proxy_score_at_index_{index}"
        if abs(value) > max_abs_proxy_score:
            return [], (
                f"proxy_score_abs_exceeds_limit_at_index_{index}:"
                f"{value:.6g}>{max_abs_proxy_score:.6g}"
            )
        if proxy_score_decimals is not None:
            value = round(value, proxy_score_decimals)
            if not math.isfinite(value):
                return [], f"nonfinite_proxy_score_after_rounding_at_index_{index}"
        normalized.append(value)
    return normalized, ""


def make_fixed_fold_indices(
    sample_count: int,
    fold_count: int = FITNESS_FOLD_COUNT,
    fold_size: int | None = None,
    seed: int = 0,
) -> tuple[tuple[int, ...], ...]:
    if fold_count < 1:
        raise ValueError("fold_count must be >= 1.")
    if fold_size is not None:
        expected_sample_count = fold_count * fold_size
        if sample_count != expected_sample_count:
            raise ValueError(
                f"Expected exactly {expected_sample_count} proxy_train rows for "
                f"{fold_count} folds of {fold_size}; got {sample_count}."
            )
        fold_sizes = [fold_size] * fold_count
    else:
        if sample_count < fold_count * 3:
            raise ValueError(
                f"Need at least {fold_count * 3} proxy_train rows for "
                f"{fold_count} folds with >=3 rows each; got {sample_count}."
            )
        base_size, remainder = divmod(sample_count, fold_count)
        fold_sizes = [
            base_size + (1 if index < remainder else 0)
            for index in range(fold_count)
        ]
    if min(fold_sizes) < 3:
        raise ValueError("Each fitness fold must contain at least 3 rows for Spearman correlation.")

    rng = random.Random(seed)
    indices = list(range(sample_count))
    rng.shuffle(indices)
    folds: list[tuple[int, ...]] = []
    start = 0
    for current_fold_size in fold_sizes:
        end = start + current_fold_size
        folds.append(tuple(sorted(indices[start:end])))
        start = end
    return tuple(folds)


def values_at_indices(values: Sequence[float], indices: Sequence[int]) -> list[float]:
    return [values[index] for index in indices]


def same_rank_signature(left: Sequence[float], right: Sequence[float]) -> bool:
    return rankdata_average(left) == rankdata_average(right)


def normalized_tree_scores(
    tree: SymbolicNode,
    values: dict[str, list[float]],
    proxy_score_decimals: int | None,
    max_abs_proxy_score: float,
) -> tuple[list[float], str]:
    return normalize_proxy_scores(
        tree.evaluate(values),
        proxy_score_decimals=proxy_score_decimals,
        max_abs_proxy_score=max_abs_proxy_score,
    )


def format_tree_path(path: Sequence[int]) -> str:
    return "root" if not path else ".".join(str(index) for index in path)


def soft_div_denominator_invalid_reason(
    tree: SymbolicNode,
    values: dict[str, list[float]],
    denominator_threshold: float,
    path: tuple[int, ...] = (),
) -> str:
    if tree.kind == "binary" and tree.token == "Div":
        denominator_node = tree.children[1]
        denominator_path = (*path, 1)
        denominator_values = denominator_node.evaluate(values)
        for index, value in enumerate(denominator_values):
            denominator_value = float(value)
            if not math.isfinite(denominator_value):
                return (
                    "soft_div_nonfinite_denominator"
                    f"_at_path_{format_tree_path(denominator_path)}"
                    f"_index_{index}"
                )
            abs_denominator = abs(denominator_value)
            if abs_denominator < denominator_threshold:
                return (
                    "soft_div_denominator_abs_below_threshold"
                    f"_at_path_{format_tree_path(denominator_path)}"
                    f"_index_{index}:"
                    f"{abs_denominator:.6g}<{denominator_threshold:.6g}"
                )

    for child_index, child in enumerate(tree.children):
        invalid_reason = soft_div_denominator_invalid_reason(
            child,
            values,
            denominator_threshold,
            (*path, child_index),
        )
        if invalid_reason:
            return invalid_reason
    return ""


def canonicalize_rank_preserving_root_unary(
    tree: SymbolicNode,
    scores: list[float],
    values: dict[str, list[float]],
    proxy_score_decimals: int | None,
    max_abs_proxy_score: float,
) -> tuple[SymbolicNode, list[float]]:
    canonical_tree = tree
    canonical_scores = scores
    while (
        canonical_tree.kind == "unary"
        and canonical_tree.token != "Negative"
        and canonical_tree.children
    ):
        child = canonical_tree.children[0]
        child_scores, invalid_reason = normalized_tree_scores(
            child,
            values,
            proxy_score_decimals=proxy_score_decimals,
            max_abs_proxy_score=max_abs_proxy_score,
        )
        if invalid_reason or not same_rank_signature(canonical_scores, child_scores):
            break
        canonical_tree = child
        canonical_scores = child_scores
    return canonical_tree, canonical_scores


def direct_spearman_fitness(
    tree: SymbolicNode,
    values: dict[str, list[float]],
    target: list[float],
    proxy_score_decimals: int | None,
    max_abs_proxy_score: float,
    soft_div: bool = False,
    denominator: float = 1e-8,
) -> tuple[float, str]:
    if soft_div:
        invalid_reason = soft_div_denominator_invalid_reason(
            tree,
            values,
            denominator_threshold=denominator,
        )
        if invalid_reason:
            return float("-inf"), invalid_reason

    scores, invalid_reason = normalized_tree_scores(
        tree,
        values,
        proxy_score_decimals=proxy_score_decimals,
        max_abs_proxy_score=max_abs_proxy_score,
    )
    if invalid_reason:
        return float("-inf"), invalid_reason

    fitness = spearman_correlation(scores, target)
    if not math.isfinite(fitness):
        return float("-inf"), "invalid_direct_spearman_correlation"
    return fitness, ""


def evaluate_tree(
    tree: SymbolicNode,
    values: dict[str, list[float]],
    target: list[float],
    fold_indices: tuple[tuple[int, ...], ...],
    generation: int,
    source: str,
    proxy_score_decimals: int | None,
    max_abs_proxy_score: float,
    fitness_decimals: int,
    soft_div: bool = False,
    denominator: float = 1e-8,
    train_fitness_mode: str = DEFAULT_TRAIN_FITNESS_MODE,
    validation_values: dict[str, list[float]] | None = None,
    validation_target: list[float] | None = None,
) -> FormulaEvaluation:
    if soft_div:
        invalid_reason = soft_div_denominator_invalid_reason(
            tree,
            values,
            denominator_threshold=denominator,
        )
        if invalid_reason:
            return make_formula_evaluation(
                tree=tree,
                raw_fitness=float("-inf"),
                resample_fitness_mean=float("-inf"),
                resample_fitness_quantile=float("-inf"),
                resample_fitness_variance=float("inf"),
                resample_invalid_count=len(fold_indices),
                generation=generation,
                source=source,
                fitness_decimals=fitness_decimals,
                validation_invalid_reason="train_invalid",
                invalid_reason=invalid_reason,
            )

    scores, invalid_reason = normalized_tree_scores(
        tree,
        values,
        proxy_score_decimals=proxy_score_decimals,
        max_abs_proxy_score=max_abs_proxy_score,
    )
    if invalid_reason:
        return make_formula_evaluation(
            tree=tree,
            raw_fitness=float("-inf"),
            resample_fitness_mean=float("-inf"),
            resample_fitness_quantile=float("-inf"),
            resample_fitness_variance=float("inf"),
            resample_invalid_count=len(fold_indices),
            generation=generation,
            source=source,
            fitness_decimals=fitness_decimals,
            validation_invalid_reason="train_invalid",
            invalid_reason=invalid_reason,
        )
    tree, scores = canonicalize_rank_preserving_root_unary(
        tree,
        scores,
        values,
        proxy_score_decimals=proxy_score_decimals,
        max_abs_proxy_score=max_abs_proxy_score,
    )

    if train_fitness_mode == "direct":
        fitness = spearman_correlation(scores, target)
        resample_invalid_count = 0
        if math.isfinite(fitness):
            resample_mean = fitness
            resample_quantile = fitness
            resample_variance = 0.0
            invalid_reason = ""
        else:
            fitness = float("-inf")
            resample_mean = float("-inf")
            resample_quantile = float("-inf")
            resample_variance = float("inf")
            resample_invalid_count = 1
            invalid_reason = "invalid_direct_train_spearman_correlation"
    elif train_fitness_mode == "folded":
        fold_fitnesses: list[float] = []
        resample_invalid_count = 0
        for indices in fold_indices:
            fold_fitness = spearman_correlation(
                values_at_indices(scores, indices),
                values_at_indices(target, indices),
            )
            if not math.isfinite(fold_fitness):
                resample_invalid_count += 1
            fold_fitnesses.append(fold_fitness)

        finite_fold_fitnesses = [
            fold_fitness for fold_fitness in fold_fitnesses if math.isfinite(fold_fitness)
        ]
        if len(finite_fold_fitnesses) != len(fold_indices):
            fitness = float("-inf")
            resample_mean = float("-inf")
            resample_quantile = float("-inf")
            resample_variance = float("inf")
            invalid_reason = "invalid_fold_spearman_correlation"
        else:
            min_fold_fitness = min(finite_fold_fitnesses)
            fold_spread = max(finite_fold_fitnesses) - min_fold_fitness
            fitness = min_fold_fitness - fold_spread
            resample_mean = sum(finite_fold_fitnesses) / len(finite_fold_fitnesses)
            resample_quantile = min_fold_fitness
            resample_variance = (
                sum((value - resample_mean) ** 2 for value in finite_fold_fitnesses)
                / len(finite_fold_fitnesses)
            )
            invalid_reason = ""
    else:
        raise ValueError(f"Unknown train_fitness_mode: {train_fitness_mode}")

    validation_fitness = float("-inf")
    validation_invalid_reason = "validation_not_evaluated"
    if validation_values is not None and validation_target is not None:
        validation_fitness, validation_invalid_reason = direct_spearman_fitness(
            tree,
            validation_values,
            validation_target,
            proxy_score_decimals=proxy_score_decimals,
            max_abs_proxy_score=max_abs_proxy_score,
            soft_div=soft_div,
            denominator=denominator,
        )

    return make_formula_evaluation(
        tree=tree,
        raw_fitness=fitness,
        resample_fitness_mean=resample_mean,
        resample_fitness_quantile=resample_quantile,
        resample_fitness_variance=resample_variance,
        resample_invalid_count=resample_invalid_count,
        generation=generation,
        source=source,
        fitness_decimals=fitness_decimals,
        validation_fitness=validation_fitness,
        validation_invalid_reason=validation_invalid_reason,
        invalid_reason=invalid_reason,
    )


def _init_evaluation_worker(
    values: dict[str, list[float]],
    target: list[float],
    fold_indices: tuple[tuple[int, ...], ...],
    validation_values: dict[str, list[float]],
    validation_target: list[float],
    train_fitness_mode: str,
    proxy_score_decimals: int | None,
    max_abs_proxy_score: float,
    fitness_decimals: int,
    soft_div: bool,
    denominator: float,
) -> None:
    global _WORKER_VALUES, _WORKER_TARGET, _WORKER_VALIDATION_VALUES, _WORKER_VALIDATION_TARGET, _WORKER_PROXY_SCORE_DECIMALS, _WORKER_MAX_ABS_PROXY_SCORE, _WORKER_FITNESS_DECIMALS, _WORKER_FOLD_INDICES, _WORKER_TRAIN_FITNESS_MODE, _WORKER_SOFT_DIV, _WORKER_DENOMINATOR
    _WORKER_VALUES = values
    _WORKER_TARGET = target
    _WORKER_VALIDATION_VALUES = validation_values
    _WORKER_VALIDATION_TARGET = validation_target
    _WORKER_FOLD_INDICES = fold_indices
    _WORKER_TRAIN_FITNESS_MODE = train_fitness_mode
    _WORKER_PROXY_SCORE_DECIMALS = proxy_score_decimals
    _WORKER_MAX_ABS_PROXY_SCORE = max_abs_proxy_score
    _WORKER_FITNESS_DECIMALS = fitness_decimals
    _WORKER_SOFT_DIV = soft_div
    _WORKER_DENOMINATOR = denominator


def _evaluate_population_worker(item: tuple[SymbolicNode, str, int]) -> FormulaEvaluation:
    if (
        _WORKER_VALUES is None
        or _WORKER_TARGET is None
        or _WORKER_VALIDATION_VALUES is None
        or _WORKER_VALIDATION_TARGET is None
        or _WORKER_MAX_ABS_PROXY_SCORE is None
        or _WORKER_FITNESS_DECIMALS is None
        or _WORKER_FOLD_INDICES is None
        or _WORKER_TRAIN_FITNESS_MODE is None
        or _WORKER_SOFT_DIV is None
        or _WORKER_DENOMINATOR is None
    ):
        raise RuntimeError("Evaluation worker was not initialized.")
    tree, source, generation = item
    return evaluate_tree(
        tree,
        _WORKER_VALUES,
        _WORKER_TARGET,
        _WORKER_FOLD_INDICES,
        generation=generation,
        source=source,
        proxy_score_decimals=_WORKER_PROXY_SCORE_DECIMALS,
        max_abs_proxy_score=_WORKER_MAX_ABS_PROXY_SCORE,
        fitness_decimals=_WORKER_FITNESS_DECIMALS,
        soft_div=_WORKER_SOFT_DIV,
        denominator=_WORKER_DENOMINATOR,
        train_fitness_mode=_WORKER_TRAIN_FITNESS_MODE,
        validation_values=_WORKER_VALIDATION_VALUES,
        validation_target=_WORKER_VALIDATION_TARGET,
    )


def evaluate_population(
    population: Sequence[tuple[SymbolicNode, str]],
    values: dict[str, list[float]],
    target: list[float],
    fold_indices: tuple[tuple[int, ...], ...],
    validation_values: dict[str, list[float]],
    validation_target: list[float],
    generation: int,
    proxy_score_decimals: int | None,
    max_abs_proxy_score: float,
    fitness_decimals: int,
    train_fitness_mode: str = DEFAULT_TRAIN_FITNESS_MODE,
    soft_div: bool = False,
    denominator: float = 1e-8,
    executor: ProcessPoolExecutor | None = None,
    evaluation_cache: dict[str, CachedEvaluation] | None = None,
) -> list[FormulaEvaluation]:
    cached_results: list[FormulaEvaluation | None] = [None] * len(population)
    uncached_tasks: list[tuple[SymbolicNode, str, int]] = []
    uncached_indices: list[int] = []
    for index, (tree, source) in enumerate(population):
        key = f"{train_fitness_mode}|{tree.formula_key()}"
        cached = evaluation_cache.get(key) if evaluation_cache is not None else None
        if cached is None:
            uncached_indices.append(index)
            uncached_tasks.append((tree, source, generation))
            continue
        (
            cached_tree,
            raw_fitness,
            resample_fitness_mean,
            resample_fitness_quantile,
            resample_fitness_variance,
            resample_fitness_std,
            resample_invalid_count,
            objective_fitness,
            token_count,
            depth,
            validation_fitness,
            validation_invalid_reason,
            invalid_reason,
        ) = cached
        cached_results[index] = FormulaEvaluation(
            tree=cached_tree,
            raw_fitness=raw_fitness,
            resample_fitness_mean=resample_fitness_mean,
            resample_fitness_quantile=resample_fitness_quantile,
            resample_fitness_variance=resample_fitness_variance,
            resample_fitness_std=resample_fitness_std,
            resample_invalid_count=resample_invalid_count,
            objective_fitness=objective_fitness,
            token_count=token_count,
            depth=depth,
            generation=generation,
            source=source,
            validation_fitness=validation_fitness,
            validation_invalid_reason=validation_invalid_reason,
            invalid_reason=invalid_reason,
        )

    if executor is not None:
        evaluated = list(executor.map(_evaluate_population_worker, uncached_tasks))
    else:
        evaluated = [
            evaluate_tree(
                tree,
                values,
                target,
                fold_indices,
                generation=generation,
                source=source,
                proxy_score_decimals=proxy_score_decimals,
                max_abs_proxy_score=max_abs_proxy_score,
                fitness_decimals=fitness_decimals,
                soft_div=soft_div,
                denominator=denominator,
                train_fitness_mode=train_fitness_mode,
                validation_values=validation_values,
                validation_target=validation_target,
            )
            for tree, source, generation in uncached_tasks
        ]

    for index, evaluation in zip(uncached_indices, evaluated):
        cached_results[index] = evaluation
        if evaluation_cache is not None:
            cache_value = (
                evaluation.tree,
                evaluation.raw_fitness,
                evaluation.resample_fitness_mean,
                evaluation.resample_fitness_quantile,
                evaluation.resample_fitness_variance,
                evaluation.resample_fitness_std,
                evaluation.resample_invalid_count,
                evaluation.objective_fitness,
                evaluation.token_count,
                evaluation.depth,
                evaluation.validation_fitness,
                evaluation.validation_invalid_reason,
                evaluation.invalid_reason,
            )
            original_tree, _source = population[index]
            evaluation_cache[f"{train_fitness_mode}|{original_tree.formula_key()}"] = cache_value
            evaluation_cache[f"{train_fitness_mode}|{evaluation.key}"] = cache_value

    return [
        evaluation
        for evaluation in cached_results
        if evaluation is not None
    ]


def ranked_evaluations(
    evaluations: Sequence[FormulaEvaluation],
    deduplicate: bool = True,
) -> list[FormulaEvaluation]:
    if not deduplicate:
        return sorted(evaluations, key=evaluation_sort_key, reverse=True)
    best_by_key: dict[str, FormulaEvaluation] = {}
    for evaluation in evaluations:
        existing = best_by_key.get(evaluation.key)
        if existing is None or evaluation_sort_key(evaluation) > evaluation_sort_key(existing):
            best_by_key[evaluation.key] = evaluation
    return sorted(best_by_key.values(), key=evaluation_sort_key, reverse=True)


def add_unique_population(
    population: list[tuple[SymbolicNode, str]],
    seen: set[str],
    tree: SymbolicNode,
    source: str,
    global_seen: set[str] | None = None,
    allow_global_duplicate: bool = False,
) -> bool:
    key = tree.formula_key()
    if key in seen:
        return False
    if global_seen is not None and not allow_global_duplicate and key in global_seen:
        return False
    population.append((tree, source))
    seen.add(key)
    if global_seen is not None:
        global_seen.add(key)
    return True


def sample_unique_random_tree(
    rng: random.Random,
    seen: set[str],
    constraints: TreeConstraints,
    source: str,
    binary_count: int | None = None,
    global_seen: set[str] | None = None,
    allow_global_duplicate: bool = False,
) -> tuple[SymbolicNode, str]:
    for _ in range(1000):
        tree = random_valid_tree(rng, constraints=constraints, binary_count=binary_count)
        key = tree.formula_key()
        if key in seen:
            continue
        if global_seen is not None and not allow_global_duplicate and key in global_seen:
            continue
        return tree, source
    raise RuntimeError("Failed to sample a unique random tree.")


def add_structural_mutation_fallback(
    population: list[tuple[SymbolicNode, str]],
    seen: set[str],
    global_seen: set[str],
    parents: Sequence[SymbolicNode],
    rng: random.Random,
    constraints: TreeConstraints,
    source: str,
) -> bool:
    if not parents:
        return False
    for _ in range(500):
        parent = rng.choice(parents)
        try:
            child = mutate_tree(
                parent,
                rng,
                constraints=constraints,
                max_steps=3,
                structural_only=True,
            )
        except RuntimeError:
            continue
        if add_unique_population(population, seen, child, source, global_seen=global_seen):
            return True
    return False


def add_random_fallback(
    population: list[tuple[SymbolicNode, str]],
    seen: set[str],
    global_seen: set[str],
    rng: random.Random,
    constraints: TreeConstraints,
    source: str,
) -> bool:
    try:
        tree, random_source = sample_unique_random_tree(
            rng,
            seen,
            constraints,
            source,
            global_seen=global_seen,
        )
    except RuntimeError:
        return False
    return add_unique_population(
        population,
        seen,
        tree,
        random_source,
        global_seen=global_seen,
    )


def fill_source_quota_with_fallbacks(
    population: list[tuple[SymbolicNode, str]],
    seen: set[str],
    global_seen: set[str],
    rng: random.Random,
    constraints: TreeConstraints,
    count: int,
    parents: Sequence[SymbolicNode],
    source: str,
) -> int:
    added = 0
    while added < count:
        if add_structural_mutation_fallback(
            population,
            seen,
            global_seen,
            parents,
            rng,
            constraints,
            f"{source}_fallback_subtree",
        ):
            added += 1
            continue
        if add_random_fallback(
            population,
            seen,
            global_seen,
            rng,
            constraints,
            f"{source}_fallback_random",
        ):
            added += 1
            continue
        raise RuntimeError(f"Failed to fill population quota for source={source}.")
    return added


def make_initial_population(
    constraints: TreeConstraints,
) -> list[tuple[SymbolicNode, str]]:
    population: list[tuple[SymbolicNode, str]] = []
    seen: set[str] = set()

    for tree in single_proxy_trees():
        add_unique_population(population, seen, tree, "initial_single_proxy")

    active_tokens = set(active_binary_tokens())
    binary_tokens = [
        token
        for token in INITIAL_PAIRWISE_BINARY_TOKENS
        if token in active_tokens
    ]
    for left_token, right_token in combinations(PROXY_TOKENS, 2):
        for binary_token in binary_tokens:
            tree = SymbolicNode(
                binary_token,
                (SymbolicNode(left_token), SymbolicNode(right_token)),
            )
            if tree.is_valid(constraints):
                add_unique_population(
                    population,
                    seen,
                    tree,
                    f"initial_pairwise_{binary_token.lower()}",
                )

    return population


def add_mutation_quota(
    population: list[tuple[SymbolicNode, str]],
    seen: set[str],
    global_seen: set[str],
    rng: random.Random,
    constraints: TreeConstraints,
    parents: Sequence[SymbolicNode],
    count: int,
    source: str,
    max_steps: int = 3,
) -> int:
    if count <= 0:
        return 0
    added = 0
    attempts = 0
    while added < count and attempts < count * 300:
        attempts += 1
        if not parents:
            break
        parent = rng.choice(parents)
        try:
            child = mutate_tree(parent, rng, constraints=constraints, max_steps=max_steps)
        except RuntimeError:
            continue
        if add_unique_population(population, seen, child, source, global_seen=global_seen):
            added += 1
    added += fill_source_quota_with_fallbacks(
        population,
        seen,
        global_seen,
        rng,
        constraints,
        count - added,
        parents,
        source,
    )
    return added


def add_archive_mutation_quota(
    population: list[tuple[SymbolicNode, str]],
    seen: set[str],
    global_seen: set[str],
    archive: FormulaArchive,
    rng: random.Random,
    constraints: TreeConstraints,
    fallback_parents: Sequence[SymbolicNode],
    count: int,
) -> int:
    added = 0
    archive_parents = archive.sample_trees(count, rng)
    while added < count:
        source = "archive_mutation"
        if archive_parents:
            parent = archive_parents.pop()
        else:
            source = "archive_mutation_fallback_random"
            try:
                parent = random_valid_tree(rng, constraints=constraints)
            except RuntimeError:
                parent = rng.choice(fallback_parents) if fallback_parents else None
        if parent is None:
            break

        attempts = 0
        inserted = False
        while attempts < 100:
            attempts += 1
            try:
                child = mutate_tree(parent, rng, constraints=constraints)
            except RuntimeError:
                break
            if add_unique_population(population, seen, child, source, global_seen=global_seen):
                added += 1
                inserted = True
                break
        if inserted:
            continue
        if source.endswith("fallback_random") and add_unique_population(
            population,
            seen,
            parent,
            source,
            global_seen=global_seen,
        ):
            added += 1

    added += fill_source_quota_with_fallbacks(
        population,
        seen,
        global_seen,
        rng,
        constraints,
        count - added,
        fallback_parents,
        "archive_mutation",
    )
    return added


def add_exploration_quota(
    population: list[tuple[SymbolicNode, str]],
    seen: set[str],
    global_seen: set[str],
    rng: random.Random,
    constraints: TreeConstraints,
    fallback_parents: Sequence[SymbolicNode],
    binary_count: int,
    count: int,
) -> int:
    added = 0
    attempts = 0
    source = f"explore_random_binary{binary_count}"
    while added < count and attempts < count * 300:
        attempts += 1
        try:
            tree, tree_source = sample_unique_random_tree(
                rng,
                seen,
                constraints,
                source,
                binary_count=binary_count,
                global_seen=global_seen,
            )
        except RuntimeError:
            break
        if add_unique_population(population, seen, tree, tree_source, global_seen=global_seen):
            added += 1
    added += fill_source_quota_with_fallbacks(
        population,
        seen,
        global_seen,
        rng,
        constraints,
        count - added,
        fallback_parents,
        source,
    )
    return added


def make_next_population(
    ranked: Sequence[FormulaEvaluation],
    archive: FormulaArchive,
    rng: random.Random,
    config: EvolutionConfig,
    global_seen: set[str],
    generation: int,
) -> list[tuple[SymbolicNode, str]]:
    population: list[tuple[SymbolicNode, str]] = []
    seen: set[str] = set()
    valid_ranked = [evaluation for evaluation in ranked if evaluation.is_valid]
    if not valid_ranked:
        raise RuntimeError("Cannot build the next population because no valid formulas were evaluated.")
    local_neighbor_count = config.local_neighbor_count_for_generation(generation)
    fallback_parent_count = max(
        config.mutation_parent_pool,
        config.crossover_parent_pool,
        config.elite_count,
    )
    fallback_parents = [evaluation.tree for evaluation in valid_ranked[:fallback_parent_count]]
    local_neighbor_parents = [
        evaluation.tree
        for evaluation in valid_ranked[:LOCAL_NEIGHBOR_PARENT_COUNT]
    ]

    for index, evaluation in enumerate(valid_ranked[: config.elite_count], start=1):
        add_unique_population(
            population,
            seen,
            evaluation.tree,
            f"elite_top{index}",
            global_seen=global_seen,
            allow_global_duplicate=True,
        )

    if len(population) < config.elite_count:
        fill_source_quota_with_fallbacks(
            population,
            seen,
            global_seen,
            rng,
            config.constraints,
            config.elite_count - len(population),
            fallback_parents,
            "elite",
        )

    top30_parents = [evaluation.tree for evaluation in valid_ranked[: config.mutation_parent_pool]]
    add_mutation_quota(
        population,
        seen,
        global_seen,
        rng,
        config.constraints,
        top30_parents,
        config.top30_mutations,
        "mutation_top30",
    )

    add_archive_mutation_quota(
        population,
        seen,
        global_seen,
        archive,
        rng,
        config.constraints,
        fallback_parents,
        config.archive_mutations,
    )

    parent_pool = list(valid_ranked[: config.crossover_parent_pool])
    crossover_added = 0
    attempts = 0
    while (
        len(parent_pool) >= 2
        and crossover_added < config.crossover_count
        and attempts < config.crossover_count * 300
    ):
        attempts += 1
        left, right = rng.sample(parent_pool, k=2)
        try:
            child = crossover_trees(left.tree, right.tree, rng, constraints=config.constraints)
        except RuntimeError:
            continue
        if add_unique_population(
            population,
            seen,
            child,
            "crossover_top30",
            global_seen=global_seen,
        ):
            crossover_added += 1
    crossover_added += fill_source_quota_with_fallbacks(
        population,
        seen,
        global_seen,
        rng,
        config.constraints,
        config.crossover_count - crossover_added,
        fallback_parents,
        "crossover",
    )

    add_mutation_quota(
        population,
        seen,
        global_seen,
        rng,
        config.constraints,
        local_neighbor_parents,
        local_neighbor_count,
        "local_neighbor_top3",
        max_steps=1,
    )

    for binary_count, count in config.effective_exploration_binary_quotas_for_generation(generation):
        add_exploration_quota(
            population,
            seen,
            global_seen,
            rng,
            config.constraints,
            fallback_parents,
            binary_count,
            count,
        )

    while len(population) < config.population_size:
        if add_random_fallback(
            population,
            seen,
            global_seen,
            rng,
            config.constraints,
            "fill_random",
        ):
            continue
        fill_source_quota_with_fallbacks(
            population,
            seen,
            global_seen,
            rng,
            config.constraints,
            1,
            fallback_parents,
            "fill_random",
        )

    if len(population) > config.population_size:
        return population[: config.population_size]
    return population


def write_history_row(
    path: Path,
    generation: int,
    best: FormulaEvaluation,
    fitness_decimals: int,
    fitness_kind: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "generation",
            "fitness_kind",
            "fitness",
            "fitness_rounded",
            "validation_fitness",
            "validation_fitness_rounded",
            "validation_invalid_reason",
            "token_count",
            "depth",
            "source",
            "rpn_tokens",
            "infix",
            "latex",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "generation": generation,
                "fitness_kind": fitness_kind,
                "fitness": best.objective_fitness,
                "fitness_rounded": rounded_fitness(best.objective_fitness, fitness_decimals),
                "validation_fitness": best.validation_fitness,
                "validation_fitness_rounded": rounded_fitness(
                    best.validation_fitness,
                    fitness_decimals,
                ),
                "validation_invalid_reason": best.validation_invalid_reason,
                "token_count": best.token_count,
                "depth": best.depth,
                "source": best.source,
                "rpn_tokens": " ".join(best.tree.to_rpn(include_eos=True)),
                "infix": best.tree.to_infix(),
                "latex": best.tree.to_latex(),
            }
        )


def write_generation_population(
    path: Path,
    generation: int,
    evaluations: Sequence[FormulaEvaluation],
    fitness_decimals: int,
    fitness_kind: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "generation",
            "rank_in_generation",
            "fitness_kind",
            "fitness",
            "fitness_rounded",
            "validation_fitness",
            "validation_fitness_rounded",
            "validation_invalid_reason",
            "token_count",
            "depth",
            "source",
            "status",
            "invalid_reason",
            "rpn_tokens",
            "infix",
            "latex",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for rank, evaluation in enumerate(ranked_evaluations(evaluations, deduplicate=False), start=1):
            writer.writerow(
                {
                    "generation": generation,
                    "rank_in_generation": rank,
                    "fitness_kind": fitness_kind,
                    "fitness": evaluation.objective_fitness,
                    "fitness_rounded": rounded_fitness(evaluation.objective_fitness, fitness_decimals),
                    "validation_fitness": evaluation.validation_fitness,
                    "validation_fitness_rounded": rounded_fitness(
                        evaluation.validation_fitness,
                        fitness_decimals,
                    ),
                    "validation_invalid_reason": evaluation.validation_invalid_reason,
                    "token_count": evaluation.token_count,
                    "depth": evaluation.depth,
                    "source": evaluation.source,
                    "status": "invalid" if evaluation.invalid_reason else "valid",
                    "invalid_reason": evaluation.invalid_reason,
                    "rpn_tokens": " ".join(evaluation.tree.to_rpn(include_eos=True)),
                    "infix": evaluation.tree.to_infix(),
                    "latex": evaluation.tree.to_latex(),
                }
            )


def resolve_num_workers(requested_workers: int, population_size: int) -> int:
    if requested_workers == 0:
        return max(1, min(population_size, os.cpu_count() or 1))
    return max(1, min(requested_workers, population_size))


def run_evolution(config: EvolutionConfig) -> FormulaArchive:
    config.validate()
    set_div_token_active(config.div_token)
    rng = random.Random(config.seed)
    values, target = load_proxy_train_benchmark(
        config.csv_path,
        target_metric=config.target_metric,
        target_direction=config.target_direction,
    )
    validation_values, validation_target, matched_validation_split = load_proxy_split_benchmark(
        config.csv_path,
        split=config.validation_split,
        target_metric=config.target_metric,
        target_direction=config.target_direction,
    )
    fold_indices = make_fixed_fold_indices(
        sample_count=len(target),
        seed=config.seed + 1_000_003,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = config.output_dir / "generation_best.csv"
    population_path = config.output_dir / "generation_population.csv"
    archive_path = config.output_dir / "archive.csv"
    for stale_path in (history_path, population_path, archive_path):
        if stale_path.exists():
            stale_path.unlink()

    archive = FormulaArchive(
        max_size=config.archive_size,
        fitness_decimals=config.fitness_decimals,
        fitness_kind=FITNESS_KIND_LABEL,
        ablation_a=config.ablation_a,
        ablation_a_gamma=config.ablation_a_gamma,
    )
    population = make_initial_population(config.constraints)
    global_seen = {tree.formula_key() for tree, _source in population}
    evaluation_cache: dict[str, CachedEvaluation] = {}
    num_workers = resolve_num_workers(config.num_workers, max(config.population_size, len(population)))

    executor: ProcessPoolExecutor | None = None
    try:
        if num_workers > 1:
            executor = ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=_init_evaluation_worker,
                initargs=(
                    values,
                    target,
                    fold_indices,
                    validation_values,
                    validation_target,
                    config.train_fitness_mode,
                    config.proxy_score_decimals,
                    config.max_abs_proxy_score,
                    config.fitness_decimals,
                    config.soft_div,
                    config.denominator,
                ),
            )

        for generation in range(1, config.max_generations + 1):
            evaluations = evaluate_population(
                population,
                values,
                target,
                fold_indices,
                validation_values,
                validation_target,
                generation=generation,
                proxy_score_decimals=config.proxy_score_decimals,
                max_abs_proxy_score=config.max_abs_proxy_score,
                fitness_decimals=config.fitness_decimals,
                train_fitness_mode=config.train_fitness_mode,
                soft_div=config.soft_div,
                denominator=config.denominator,
                executor=executor,
                evaluation_cache=evaluation_cache,
            )
            ranked = ranked_evaluations(evaluations)
            best = ranked[0]
            archive.add_many(ranked)

            write_history_row(
                history_path,
                generation,
                best,
                config.fitness_decimals,
                FITNESS_KIND_LABEL,
            )
            write_generation_population(
                population_path,
                generation,
                evaluations,
                config.fitness_decimals,
                FITNESS_KIND_LABEL,
            )
            archive.save_csv(archive_path)
            print(
                f"generation={generation:03d} "
                f"train_fitness_mode={config.train_fitness_mode} "
                f"ablationA={config.ablation_a} "
                f"score_gamma={config.ablation_a_gamma} "
                f"best_fitness={rounded_fitness(best.objective_fitness, config.fitness_decimals)} "
                f"best_validation_fitness={rounded_fitness(best.validation_fitness, config.fitness_decimals)} "
                f"validation_split={matched_validation_split} "
                f"best_formula={best.tree.to_infix()} "
                f"archive_size={len(archive)}",
                flush=True,
            )

            if generation < config.max_generations:
                population = make_next_population(
                    ranked,
                    archive,
                    rng,
                    config,
                    global_seen,
                    generation=generation + 1,
                )
    finally:
        if executor is not None:
            executor.shutdown()

    return archive


def parse_rpn_file(path: Path, constraints: TreeConstraints) -> list[SymbolicNode]:
    trees: list[SymbolicNode] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = re.split(r"[\s,]+", stripped)
            trees.append(parse_rpn(tokens, constraints=constraints))
    return trees


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected one of: true, false, 1, 0, yes, no, on, off.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evolutionary symbolic proxy formula search over GroundTruth proxy benchmark CSVs. "
            "Formulas are represented as trees during search and saved as RPN token sequences."
        )
    )
    parser.add_argument("--backbone", required=True, help="Backbone name, e.g. autoformer or mamba.")
    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "Dataset name, e.g. ECL, Monash__weather_dataset, or "
            "TIME__epf_electricity_price__H."
        ),
    )
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--groundtruth-dir",
        type=Path,
        default=None,
        help=(
            "GroundTruth root. Supports GroundTruth/<backbone>/ and flat CSV directories "
            "such as proxy_scores/monash_time/. Default: <repo-root>/GroundTruth."
        ),
    )
    parser.add_argument("--csv-path", type=Path, default=None, help="Explicit benchmark CSV path. Overrides --groundtruth-dir lookup.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <repo-root>/archive/symbolic_proxy_evolution/<backbone>/<dataset>/seed_<seed>/run_<YYYYmmdd_HHMMSS>.")

    parser.add_argument("--max-generations", type=int, default=200)
    parser.add_argument("--population-size", type=int, default=200)
    parser.add_argument("--archive-size", type=int, default=1000)
    parser.add_argument(
        "--archive-threshold-margin",
        type=float,
        default=0.05,
        help="Deprecated no-op; archive thresholding is disabled.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="CPU worker processes for formula evaluation within each generation. Use 0 for auto; 1 disables parallel evaluation.")

    parser.add_argument("--elite-count", type=int, default=5)
    parser.add_argument("--mutation-parent-pool", type=int, default=30)
    parser.add_argument("--top30-mutations", type=int, default=45)
    parser.add_argument("--archive-mutations", type=int, default=25)
    parser.add_argument("--crossover-count", type=int, default=25)
    parser.add_argument("--crossover-parent-pool", type=int, default=30)
    parser.add_argument("--local-neighbor-count", type=int, default=10)

    parser.add_argument("--max-binary-ops", type=int, default=3)
    parser.add_argument("--max-unary-chain", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=10)

    parser.add_argument("--target-metric", default="mse", help="GroundTruth metric column used as target.")
    parser.add_argument(
        "--target-direction",
        choices=("minimize", "maximize"),
        default="minimize",
        help="Use minimize for error metrics such as mse; maximize for accuracy-like metrics.",
    )
    parser.add_argument(
        "--train-fitness-mode",
        choices=TRAIN_FITNESS_MODES,
        default=DEFAULT_TRAIN_FITNESS_MODE,
        help=(
            "folded keeps the existing fold-robust proxy_train objective; "
            "direct uses Spearman over the full proxy_train split."
            "direct or folded"
        ),
    )
    parser.add_argument(
        "--validation-split",
        default=DEFAULT_VALIDATION_SPLIT,
        help=(
            "GroundTruth split used for archive ranking. "
            "proxy_valid and proxy_val are treated as aliases."
        ),
    )
    parser.add_argument(
        "--ablationA",
        dest="ablation_a",
        type=parse_bool,
        default=False,
        help=(
            "If true, rank the archive by Score = mean(train fitness, validation fitness) "
            "- gamma * abs(train fitness - validation fitness)."
        ),
    )
    parser.add_argument(
        "--ablationA-gamma",
        "--gamma",
        dest="ablation_a_gamma",
        type=float,
        default=0.5,
        help="Penalty weight gamma used by --ablationA. Default: 0.5.",
    )
    parser.add_argument("--fitness-decimals", type=int, default=4)
    parser.add_argument(
        "--proxy-score-decimals",
        type=int,
        default=12,
        help=(
            "Round every finite formula proxy score to this many decimal places before Spearman "
            "correlation. Use -1 to disable rounding."
        ),
    )
    parser.add_argument(
        "--max-abs-proxy-score",
        type=float,
        default=1e12,
        help=(
            "Mark a formula as invalid if any computed proxy score has absolute value above this limit."
        ),
    )
    parser.add_argument(
        "--div_token",
        type=parse_bool,
        default=True,
        help="If false, exclude Div from binary token choices during search.",
    )
    parser.add_argument(
        "--soft_div",
        type=parse_bool,
        default=False,
        help=(
            "If true, mark a formula invalid when any Div denominator subtree has "
            "absolute value below --denominator."
        ),
    )
    parser.add_argument(
        "--denominator",
        type=float,
        default=1e-8,
        help="Div denominator absolute-value threshold used when --soft_div is true.",
    )
    parser.add_argument("--visualize-archive", action="store_true", help="Generate PNG tree diagrams and a LaTeX table after search.")
    parser.add_argument("--visualize-top-k", type=int, default=10, help="Number of ranked archive formulas to visualize.")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> EvolutionConfig:
    repo_root = args.repo_root.resolve()
    groundtruth_dir = (args.groundtruth_dir or repo_root / "GroundTruth").resolve()
    backbone = normalize_backbone(args.backbone)
    dataset = normalize_dataset(args.dataset)
    csv_path = args.csv_path.resolve() if args.csv_path else find_groundtruth_csv(groundtruth_dir, backbone, dataset)
    if args.output_dir:
        output_dir = args.output_dir.resolve()
    else:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            repo_root
            / "archive"
            / "symbolic_proxy_evolution"
            / backbone
            / dataset
            / f"seed_{args.seed}"
            / f"run_{run_timestamp}"
        )
    constraints = TreeConstraints(
        max_binary_ops=args.max_binary_ops,
        max_unary_chain=args.max_unary_chain,
        max_tokens=args.max_tokens,
    )
    proxy_score_decimals = None if args.proxy_score_decimals < 0 else args.proxy_score_decimals
    return EvolutionConfig(
        backbone=backbone,
        dataset=dataset,
        csv_path=csv_path,
        output_dir=output_dir,
        seed=args.seed,
        max_generations=args.max_generations,
        population_size=args.population_size,
        elite_count=args.elite_count,
        mutation_parent_pool=args.mutation_parent_pool,
        top30_mutations=args.top30_mutations,
        archive_mutations=args.archive_mutations,
        crossover_count=args.crossover_count,
        crossover_parent_pool=args.crossover_parent_pool,
        archive_size=args.archive_size,
        archive_threshold_margin=args.archive_threshold_margin,
        target_metric=args.target_metric,
        target_direction=args.target_direction,
        train_fitness_mode=args.train_fitness_mode,
        validation_split=args.validation_split,
        ablation_a=args.ablation_a,
        ablation_a_gamma=args.ablation_a_gamma,
        fitness_decimals=args.fitness_decimals,
        constraints=constraints,
        visualize_top_k=args.visualize_top_k,
        num_workers=args.num_workers,
        proxy_score_decimals=proxy_score_decimals,
        max_abs_proxy_score=args.max_abs_proxy_score,
        local_neighbor_count=args.local_neighbor_count,
        div_token=args.div_token,
        soft_div=args.soft_div,
        denominator=args.denominator,
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    archive = run_evolution(config)
    if args.visualize_archive:
        archive.visualize(config.output_dir / "visualizations", top_k=config.visualize_top_k)


if __name__ == "__main__":
    main()
