from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

try:
    from evolve_symbolic_proxy import (
        find_groundtruth_csv,
        normalize_backbone,
        normalize_dataset,
        normalize_proxy_scores,
        resolve_split_column,
        safe_float,
        spearman_correlation,
    )
    from symbolic_tree import PROXY_TO_COLUMN, parse_rpn
except ImportError:  # pragma: no cover - supports package-style execution.
    from .evolve_symbolic_proxy import (
        find_groundtruth_csv,
        normalize_backbone,
        normalize_dataset,
        normalize_proxy_scores,
        resolve_split_column,
        safe_float,
        spearman_correlation,
    )
    from .symbolic_tree import PROXY_TO_COLUMN, parse_rpn


def infer_backbone_dataset(archive_csv: Path) -> tuple[str, str]:
    parts = archive_csv.resolve().parts
    try:
        index = parts.index("symbolic_proxy_evolution")
    except ValueError as exc:
        raise ValueError(
            "Could not infer backbone/dataset from archive path. "
            "Pass --backbone and --dataset, or pass --groundtruth-csv."
        ) from exc
    if len(parts) <= index + 2:
        raise ValueError(
            "Could not infer backbone/dataset from archive path. "
            "Pass --backbone and --dataset, or pass --groundtruth-csv."
        )
    return normalize_backbone(parts[index + 1]), normalize_dataset(parts[index + 2])


def resolve_groundtruth_csv(args: argparse.Namespace) -> Path:
    if args.groundtruth_csv is not None:
        return args.groundtruth_csv.resolve()

    backbone = normalize_backbone(args.backbone) if args.backbone else None
    dataset = normalize_dataset(args.dataset) if args.dataset else None
    if backbone is None or dataset is None:
        inferred_backbone, inferred_dataset = infer_backbone_dataset(args.archive_csv)
        backbone = backbone or inferred_backbone
        dataset = dataset or inferred_dataset

    groundtruth_dir = (args.groundtruth_dir or args.repo_root / "GroundTruth").resolve()
    return find_groundtruth_csv(groundtruth_dir, backbone, dataset)


def load_split_values(
    groundtruth_csv: Path,
    split: str,
    target_metric: str,
    target_direction: str,
) -> tuple[dict[str, list[float]], list[float], list[float], int]:
    with groundtruth_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in PROXY_TO_COLUMN.values() if column not in fieldnames]
        if missing:
            raise ValueError(f"{groundtruth_csv} is missing proxy columns: {', '.join(missing)}")
        if target_metric not in fieldnames:
            raise ValueError(f"{groundtruth_csv} is missing target metric column: {target_metric}")
        split_column = resolve_split_column(fieldnames, groundtruth_csv)
        rows = [row for row in reader if row.get(split_column) == split]

    if not rows:
        raise ValueError(f"No rows found for split={split!r} in {groundtruth_csv}.")

    values = {
        column: [safe_float(row[column]) for row in rows]
        for column in sorted(set(PROXY_TO_COLUMN.values()))
    }
    raw_target = [safe_float(row[target_metric]) for row in rows]
    directed_target = [-value for value in raw_target] if target_direction == "minimize" else raw_target
    return values, directed_target, raw_target, len(rows)


def default_output_path(archive_csv: Path, split: str) -> Path:
    return archive_csv.with_name(f"{archive_csv.stem}_{split}_spearman.csv")


def format_score(value: float) -> str:
    return "" if not math.isfinite(value) else f"{value:.6f}"


def output_fieldnames(
    archive_fieldnames: list[str],
    split_rank_column: str,
    directed_column: str,
    raw_column: str,
    invalid_column: str,
) -> list[str]:
    extra_columns = {split_rank_column, directed_column, raw_column, invalid_column}
    fieldnames = [name for name in archive_fieldnames if name not in extra_columns]
    result: list[str] = []
    inserted_rank = False
    inserted_scores = False
    for name in fieldnames:
        result.append(name)
        if name == "rank":
            result.append(split_rank_column)
            inserted_rank = True
        if name in {"fitness", "objective_fitness"} and not inserted_scores:
            result.extend([directed_column, raw_column, invalid_column])
            inserted_scores = True
    if not inserted_rank:
        result.insert(0, split_rank_column)
    if not inserted_scores:
        result.extend([directed_column, raw_column, invalid_column])
    return result


def score_archive(args: argparse.Namespace) -> tuple[Path, list[dict[str, str]], int, int]:
    archive_csv = args.archive_csv.resolve()
    output_csv = (args.output_csv or default_output_path(archive_csv, args.split)).resolve()
    groundtruth_csv = resolve_groundtruth_csv(args)

    values, directed_target, raw_target, split_count = load_split_values(
        groundtruth_csv=groundtruth_csv,
        split=args.split,
        target_metric=args.target_metric,
        target_direction=args.target_direction,
    )

    with archive_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        archive_fieldnames = reader.fieldnames or []
        archive_rows = list(reader)

    direction_prefix = "neg_" if args.target_direction == "minimize" else ""
    split_rank_column = f"{args.split}_rank"
    directed_column = f"{args.split}_spearman_{direction_prefix}{args.target_metric}"
    raw_column = f"{args.split}_spearman_raw_{args.target_metric}"
    invalid_column = f"{args.split}_invalid_reason"

    scored_rows: list[dict[str, str]] = []
    for row in archive_rows:
        scored = dict(row)
        invalid_reason = ""
        directed_corr = float("nan")
        raw_corr = float("nan")
        try:
            tree = parse_rpn(row["rpn_tokens"].split())
            scores, invalid_reason = normalize_proxy_scores(
                tree.evaluate(values),
                proxy_score_decimals=args.proxy_score_decimals,
                max_abs_proxy_score=args.max_abs_proxy_score,
            )
            if not invalid_reason:
                directed_corr = spearman_correlation(scores, directed_target)
                raw_corr = spearman_correlation(scores, raw_target)
        except Exception as exc:  # Keep batch scoring robust for malformed formulas.
            invalid_reason = f"{type(exc).__name__}: {exc}"

        scored[directed_column] = format_score(directed_corr)
        scored[raw_column] = format_score(raw_corr)
        scored[invalid_column] = invalid_reason
        scored["_directed_corr"] = directed_corr
        scored_rows.append(scored)

    valid_rows = [
        row
        for row in scored_rows
        if math.isfinite(float(row["_directed_corr"]))
    ]
    valid_rows_by_score = sorted(
        valid_rows,
        key=lambda row: float(row["_directed_corr"]),
        reverse=True,
    )
    split_rank_by_archive_rank = {
        row.get("rank", ""): str(rank)
        for rank, row in enumerate(valid_rows_by_score, start=1)
    }
    for row in scored_rows:
        row[split_rank_column] = split_rank_by_archive_rank.get(row.get("rank", ""), "")

    if args.sort_by == "split_spearman":
        output_rows = sorted(
            scored_rows,
            key=lambda row: (
                math.isfinite(float(row["_directed_corr"])),
                float(row["_directed_corr"]),
            ),
            reverse=True,
        )
    else:
        output_rows = scored_rows

    fieldnames = output_fieldnames(
        archive_fieldnames,
        split_rank_column,
        directed_column,
        raw_column,
        invalid_column,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in output_rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    return output_csv, valid_rows_by_score, split_count, len(scored_rows) - len(valid_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score symbolic proxy archive formulas on a GroundTruth split with Spearman correlation."
    )
    parser.add_argument("archive_csv", type=Path)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--groundtruth-dir", type=Path, default=None)
    parser.add_argument("--groundtruth-csv", type=Path, default=None)
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default="proxy_test")
    parser.add_argument("--target-metric", default="mse")
    parser.add_argument(
        "--target-direction",
        choices=("minimize", "maximize"),
        default="minimize",
    )
    parser.add_argument(
        "--proxy-score-decimals",
        type=int,
        default=12,
        help="Round formula proxy scores before Spearman. Use -1 to disable rounding.",
    )
    parser.add_argument("--max-abs-proxy-score", type=float, default=1e12)
    parser.add_argument(
        "--sort-by",
        choices=("archive_rank", "split_spearman"),
        default="archive_rank",
    )
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    args.proxy_score_decimals = None if args.proxy_score_decimals < 0 else args.proxy_score_decimals
    return args


def main() -> None:
    args = parse_args()
    output_csv, ranked_rows, split_count, invalid_count = score_archive(args)
    print(f"output_csv={output_csv}")
    print(
        f"scored={len(ranked_rows) + invalid_count} "
        f"valid={len(ranked_rows)} invalid={invalid_count} "
        f"split={args.split} split_count={split_count}"
    )
    direction_prefix = "neg_" if args.target_direction == "minimize" else ""
    score_column = f"{args.split}_spearman_{direction_prefix}{args.target_metric}"
    for index, row in enumerate(ranked_rows[: args.top_k], start=1):
        print(
            f"{index},archive_rank={row.get('rank', '')},"
            f"{score_column}={row.get(score_column, '')},"
            f"fitness={row.get('fitness', row.get('objective_fitness', ''))},"
            f"infix={row.get('inflix', '')}"
        )


if __name__ == "__main__":
    main()
