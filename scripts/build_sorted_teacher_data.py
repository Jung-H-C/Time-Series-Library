from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DATASET_COUNT = 53
SEED_RE = re.compile(r"seed_(\d+)")


@dataclass(frozen=True)
class FormulaRecord:
    dataset: str
    rpn_tokens: str
    formula: str
    infix: str
    validation_fitness: float
    score: float
    train_fitness: float
    generation: int
    token_count: int
    depth: int
    source_seed: int
    source_run: str
    source_rank: int
    source_archive_latex: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect the top formulas from every Autoformer seed/run, deduplicate "
            "them by normalized RPN, sort by validation fitness, assign linear "
            "percentile weights, and write one CSV per dataset."
        )
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("archive/symbolic_proxy_evolution/autoformer"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sorted_teacher_data"),
    )
    parser.add_argument("--top-k-per-run", type=int, default=10)
    parser.add_argument("--max-formulas", type=int, default=1000)
    parser.add_argument("--maximum-weight", type=float, default=1.0)
    parser.add_argument("--minimum-weight", type=float, default=0.1)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def normalize_rpn_tokens(value: str) -> str:
    return " ".join(value.split())


def parse_float(value: str | None) -> float:
    if value is None or not value.strip():
        return float("nan")
    return float(value)


def parse_int(value: str | None) -> int:
    if value is None or not value.strip():
        return -1
    return int(value)


def seed_from_path(path: Path) -> int:
    for part in path.parts:
        match = SEED_RE.fullmatch(part)
        if match:
            return int(match.group(1))
    raise ValueError(f"Could not find seed directory in {path}")


def run_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("run_"):
            return part
    raise ValueError(f"Could not find run directory in {path}")


def read_run_top_k(
    dataset: str,
    archive_latex_path: Path,
    archive_root: Path,
    top_k: int,
) -> list[FormulaRecord]:
    # archive_latex.tex identifies a completed visualization run.  Its sibling
    # archive.csv contains the same formulas plus stable RPN tokens and exact
    # machine-readable fitness values, so it is used as the parse source.
    archive_csv_path = archive_latex_path.parent.parent / "archive.csv"
    if not archive_csv_path.is_file():
        raise FileNotFoundError(
            f"Missing archive.csv beside {archive_latex_path}: {archive_csv_path}"
        )

    with archive_csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "rank",
            "generation",
            "Score",
            "fitness",
            "validation_fitness",
            "rpn_tokens",
            "latex",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{archive_csv_path} is missing columns: {sorted(missing)}")
        rows = list(reader)

    rows.sort(key=lambda row: parse_int(row.get("rank")))
    records = []
    for row in rows[:top_k]:
        rpn_tokens = normalize_rpn_tokens(row["rpn_tokens"])
        if not rpn_tokens:
            raise ValueError(f"Empty rpn_tokens in {archive_csv_path}")
        records.append(
            FormulaRecord(
                dataset=dataset,
                rpn_tokens=rpn_tokens,
                formula=str(row.get("latex", "")).strip(),
                infix=str(row.get("inflix", row.get("infix", ""))).strip(),
                validation_fitness=parse_float(row.get("validation_fitness")),
                score=parse_float(row.get("Score")),
                train_fitness=parse_float(row.get("fitness")),
                generation=parse_int(row.get("generation")),
                token_count=parse_int(row.get("token_count")),
                depth=parse_int(row.get("depth")),
                source_seed=seed_from_path(archive_latex_path),
                source_run=run_from_path(archive_latex_path),
                source_rank=parse_int(row.get("rank")),
                source_archive_latex=archive_latex_path.relative_to(archive_root).as_posix(),
            )
        )
    return records


def descending_float_key(value: float) -> float:
    return -value if math.isfinite(value) else math.inf


def representative_key(record: FormulaRecord) -> tuple[object, ...]:
    return (
        descending_float_key(record.validation_fitness),
        descending_float_key(record.score),
        record.source_rank,
        record.source_seed,
        record.source_run,
        record.rpn_tokens,
    )


def select_unique_formulas(
    records: list[FormulaRecord],
    max_formulas: int,
) -> list[tuple[FormulaRecord, int]]:
    grouped: dict[str, list[FormulaRecord]] = {}
    for record in records:
        grouped.setdefault(record.rpn_tokens, []).append(record)

    representatives = [
        (min(occurrences, key=representative_key), len(occurrences))
        for occurrences in grouped.values()
    ]
    representatives.sort(key=lambda item: representative_key(item[0]))
    return representatives[:max_formulas]


def percentile_linear_weight(
    rank: int,
    count: int,
    maximum: float,
    minimum: float,
) -> float:
    if count <= 0:
        raise ValueError("count must be positive")
    if rank < 1 or rank > count:
        raise ValueError(f"rank must be in [1, {count}], got {rank}")
    if count == 1:
        return maximum
    percentile = (rank - 1) / (count - 1)
    return maximum + (minimum - maximum) * percentile


FIELDNAMES = [
    "rank",
    "weight",
    "validation_fitness",
    "formula",
    "rpn_tokens",
    "infix",
    "score",
    "train_fitness",
    "generation",
    "token_count",
    "depth",
    "source_seed",
    "source_run",
    "source_rank",
    "occurrence_count",
    "source_archive_latex",
]


def format_float(value: float) -> str:
    return f"{value:.9f}" if math.isfinite(value) else "nan"


def write_dataset_csv(
    output_path: Path,
    selected: list[tuple[FormulaRecord, int]],
    maximum_weight: float,
    minimum_weight: float,
) -> None:
    count = len(selected)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for rank, (record, occurrence_count) in enumerate(selected, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "weight": f"{percentile_linear_weight(rank, count, maximum_weight, minimum_weight):.9f}",
                    "validation_fitness": format_float(record.validation_fitness),
                    "formula": record.formula,
                    "rpn_tokens": record.rpn_tokens,
                    "infix": record.infix,
                    "score": format_float(record.score),
                    "train_fitness": format_float(record.train_fitness),
                    "generation": record.generation,
                    "token_count": record.token_count,
                    "depth": record.depth,
                    "source_seed": record.source_seed,
                    "source_run": record.source_run,
                    "source_rank": record.source_rank,
                    "occurrence_count": occurrence_count,
                    "source_archive_latex": record.source_archive_latex,
                }
            )


def validate_args(args: argparse.Namespace) -> None:
    if args.top_k_per_run <= 0:
        raise ValueError("--top-k-per-run must be positive")
    if args.max_formulas <= 0:
        raise ValueError("--max-formulas must be positive")
    if not math.isfinite(args.maximum_weight) or not math.isfinite(args.minimum_weight):
        raise ValueError("Weights must be finite")
    if args.maximum_weight < args.minimum_weight:
        raise ValueError("--maximum-weight must be >= --minimum-weight")


def main() -> int:
    args = parse_args()
    validate_args(args)
    archive_root = resolve_path(args.archive_root).resolve()
    output_dir = resolve_path(args.output_dir).resolve()

    dataset_dirs = sorted(path for path in archive_root.iterdir() if path.is_dir())
    if len(dataset_dirs) != EXPECTED_DATASET_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_DATASET_COUNT} dataset directories under {archive_root}, "
            f"got {len(dataset_dirs)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = [
        "dataset,archive_latex_files,top_k_records,unique_formulas,saved_formulas"
    ]
    expected_outputs = set()
    for dataset_number, dataset_dir in enumerate(dataset_dirs, start=1):
        archive_latex_paths = sorted(
            dataset_dir.glob("seed_*/run_*/visualizations/archive_latex.tex")
        )
        if not archive_latex_paths:
            raise FileNotFoundError(f"No archive_latex.tex files found under {dataset_dir}")

        records = []
        for archive_latex_path in archive_latex_paths:
            records.extend(
                read_run_top_k(
                    dataset_dir.name,
                    archive_latex_path,
                    archive_root,
                    args.top_k_per_run,
                )
            )
        unique_count = len({record.rpn_tokens for record in records})
        selected = select_unique_formulas(records, args.max_formulas)

        output_path = output_dir / f"{dataset_dir.name}.csv"
        expected_outputs.add(output_path.resolve())
        write_dataset_csv(
            output_path,
            selected,
            maximum_weight=args.maximum_weight,
            minimum_weight=args.minimum_weight,
        )
        print(
            f"[{dataset_number:02d}/{len(dataset_dirs)}] {dataset_dir.name}: "
            f"runs={len(archive_latex_paths)}, top-k={len(records)}, "
            f"unique={unique_count}, saved={len(selected)}",
            flush=True,
        )
        summary_lines.append(
            f"{dataset_dir.name},{len(archive_latex_paths)},{len(records)},"
            f"{unique_count},{len(selected)}"
        )

    actual_outputs = {path.resolve() for path in output_dir.glob("*.csv")}
    missing_outputs = sorted(path.name for path in expected_outputs.difference(actual_outputs))
    if missing_outputs:
        raise RuntimeError(f"Missing output CSV files: {missing_outputs}")

    (output_dir / "summary.txt").write_text(
        "\n".join(
            [
                "53-dataset validation-fitness-sorted teacher formulas",
                f"archive_root: {archive_root}",
                f"top_k_per_run: {args.top_k_per_run}",
                f"max_formulas_per_dataset: {args.max_formulas}",
                "unique_key: normalized rpn_tokens",
                "sort: validation_fitness descending",
                (
                    "weight: linear percentile from "
                    f"{args.maximum_weight:.1f} at rank 1 to "
                    f"{args.minimum_weight:.1f} at the last saved rank"
                ),
                "",
                *summary_lines,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(expected_outputs)} dataset CSV files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
