from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DATASET_COUNT = 53
EXPECTED_COLUMNS = {
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build 53 DCSPG GroundTruth NPZ files from validation-fitness-sorted "
            "teacher CSV files, including percentile-based formula weights."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("sorted_teacher_data"),
        help="Directory containing one sorted teacher CSV per dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("DCSPG/GroundTruth"),
        help="Directory where one NPZ per dataset will be written.",
    )
    parser.add_argument(
        "--requested-count",
        type=int,
        default=1000,
        help="Requested maximum formula count recorded in NPZ metadata.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def as_str_array(values: list[str]) -> np.ndarray:
    return np.asarray(values, dtype=str)


def parse_int(value: str, *, column: str, path: Path) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid integer in {path}, column={column}: {value!r}") from exc


def parse_float(value: str, *, column: str, path: Path) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid float in {path}, column={column}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite float in {path}, column={column}: {value!r}")
    return parsed


def read_teacher_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = EXPECTED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Teacher CSV is empty: {path}")
    return rows


def validate_teacher_rows(path: Path, rows: list[dict[str, str]], requested_count: int) -> None:
    if len(rows) > requested_count:
        raise ValueError(
            f"{path} contains {len(rows)} formulas, exceeding requested_count={requested_count}"
        )

    ranks = [parse_int(row["rank"], column="rank", path=path) for row in rows]
    if ranks != list(range(1, len(rows) + 1)):
        raise ValueError(f"Teacher ranks are not contiguous from 1 in {path}")

    rpn_tokens = [" ".join(row["rpn_tokens"].split()) for row in rows]
    if any(not formula for formula in rpn_tokens):
        raise ValueError(f"Empty rpn_tokens in {path}")
    if len(rpn_tokens) != len(set(rpn_tokens)):
        raise ValueError(f"Duplicate rpn_tokens in {path}")

    validation_fitness = [
        parse_float(row["validation_fitness"], column="validation_fitness", path=path)
        for row in rows
    ]
    if any(
        validation_fitness[index] < validation_fitness[index + 1]
        for index in range(len(validation_fitness) - 1)
    ):
        raise ValueError(f"validation_fitness is not sorted descending in {path}")

    weights = [parse_float(row["weight"], column="weight", path=path) for row in rows]
    if not math.isclose(weights[0], 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"Rank-1 weight must be 1.0 in {path}, got {weights[0]}")
    if len(weights) > 1 and not math.isclose(
        weights[-1], 0.1, rel_tol=0.0, abs_tol=1e-8
    ):
        raise ValueError(f"Last weight must be 0.1 in {path}, got {weights[-1]}")
    if any(weights[index] < weights[index + 1] for index in range(len(weights) - 1)):
        raise ValueError(f"Weights are not sorted descending in {path}")


def read_source_summary(input_dir: Path) -> dict[str, tuple[int, int]]:
    """Return dataset -> (total_top_k_rows, unique_count_before_limit)."""
    summary_path = input_dir / "summary.txt"
    if not summary_path.is_file():
        return {}

    lines = summary_path.read_text(encoding="utf-8").splitlines()
    try:
        header_index = lines.index(
            "dataset,archive_latex_files,top_k_records,unique_formulas,saved_formulas"
        )
    except ValueError:
        return {}

    result = {}
    reader = csv.DictReader(lines[header_index:])
    for row in reader:
        result[row["dataset"]] = (
            int(row["top_k_records"]),
            int(row["unique_formulas"]),
        )
    return result


def save_dataset_npz(
    output_path: Path,
    dataset: str,
    rows: list[dict[str, str]],
    requested_count: int,
    source_counts: tuple[int, int] | None,
) -> None:
    ranks = np.asarray([int(row["rank"]) for row in rows], dtype=np.int64)
    weights = np.asarray([float(row["weight"]) for row in rows], dtype=np.float32)
    validation_fitness = np.asarray(
        [float(row["validation_fitness"]) for row in rows], dtype=np.float32
    )
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float32)
    train_fitness = np.asarray(
        [float(row["train_fitness"]) for row in rows], dtype=np.float32
    )
    total_topk_rows, unique_count_before_limit = source_counts or (-1, len(rows))

    np.savez_compressed(
        output_path,
        dataset=np.asarray(dataset, dtype=str),
        rpn_tokens=as_str_array([" ".join(row["rpn_tokens"].split()) for row in rows]),
        infix=as_str_array([row["infix"] for row in rows]),
        latex=as_str_array([row["formula"] for row in rows]),
        teacher_rank=ranks,
        weight=weights,
        source_seed=np.asarray([int(row["source_seed"]) for row in rows], dtype=np.int64),
        source_run=as_str_array([row["source_run"] for row in rows]),
        source_rank=np.asarray([int(row["source_rank"]) for row in rows], dtype=np.int64),
        source_generation=np.asarray(
            [int(row["generation"]) for row in rows], dtype=np.int64
        ),
        best_score=scores,
        best_fitness=train_fitness,
        best_validation_fitness=validation_fitness,
        best_train_fitness=train_fitness,
        token_count=np.asarray([int(row["token_count"]) for row in rows], dtype=np.int64),
        depth=np.asarray([int(row["depth"]) for row in rows], dtype=np.int64),
        occurrence_count=np.asarray(
            [int(row["occurrence_count"]) for row in rows], dtype=np.int64
        ),
        # The aggregated occurrence strings are not present in sorted teacher
        # CSVs; retain the legacy key with empty values for schema compatibility.
        occurrences=as_str_array([""] * len(rows)),
        source_archive_path=as_str_array(
            [row["source_archive_latex"] for row in rows]
        ),
        selected_count=np.asarray(len(rows), dtype=np.int64),
        requested_count=np.asarray(requested_count, dtype=np.int64),
        total_topk_rows=np.asarray(total_topk_rows, dtype=np.int64),
        unique_count_before_limit=np.asarray(unique_count_before_limit, dtype=np.int64),
        selection_policy=np.asarray(
            "unique_rpn_tokens_sorted_by_validation_fitness_desc", dtype=str
        ),
        weight_policy=np.asarray(
            "linear_percentile_rank_1_to_last_saved_rank_1.0_to_0.1", dtype=str
        ),
        source_teacher_csv=np.asarray(f"{dataset}.csv", dtype=str),
    )


def main() -> int:
    args = parse_args()
    if args.requested_count <= 0:
        raise ValueError("--requested-count must be positive")

    input_dir = resolve_path(args.input_dir).resolve()
    output_dir = resolve_path(args.output_dir).resolve()
    input_paths = sorted(input_dir.glob("*.csv"))
    if len(input_paths) != EXPECTED_DATASET_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_DATASET_COUNT} teacher CSV files in {input_dir}, "
            f"got {len(input_paths)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    source_summary = read_source_summary(input_dir)
    expected_outputs = set()
    total_formulas = 0
    for dataset_number, input_path in enumerate(input_paths, start=1):
        dataset = input_path.stem
        rows = read_teacher_rows(input_path)
        validate_teacher_rows(input_path, rows, args.requested_count)
        output_path = output_dir / f"{dataset}.npz"
        save_dataset_npz(
            output_path,
            dataset,
            rows,
            args.requested_count,
            source_summary.get(dataset),
        )
        expected_outputs.add(output_path.resolve())
        total_formulas += len(rows)
        print(
            f"[{dataset_number:02d}/{len(input_paths)}] {dataset}: "
            f"saved {len(rows)} weighted formulas",
            flush=True,
        )

    actual_outputs = {path.resolve() for path in output_dir.glob("*.npz")}
    missing_outputs = sorted(path.name for path in expected_outputs.difference(actual_outputs))
    if missing_outputs:
        raise RuntimeError(f"Missing GroundTruth NPZ files: {missing_outputs}")

    print(
        f"Wrote {len(expected_outputs)} GroundTruth NPZ files with "
        f"{total_formulas} weighted formulas to {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
