#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean


ACCURACY_PATTERN = re.compile(r"accuracy:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_candidate_ids(candidates_path: Path) -> tuple[list[str], str]:
    with candidates_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"Candidates file must contain a non-empty 'candidates' list: {candidates_path}")

    metadata = payload.get("metadata", {})
    candidate_prefix = str(metadata.get("candidate_prefix", "timesnet_classification_uea"))
    candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
    return candidate_ids, candidate_prefix


def _extract_task_and_candidate_id(directory_name: str, candidate_prefix: str) -> tuple[str, str] | None:
    marker = f"_{candidate_prefix}_"
    if not directory_name.startswith("classification_") or marker not in directory_name:
        return None

    task_name, suffix = directory_name[len("classification_") :].split(marker, 1)
    candidate_id = f"{candidate_prefix}_{suffix}"
    return task_name, candidate_id


def _read_accuracy(result_file: Path) -> float:
    text = result_file.read_text(encoding="utf-8", errors="ignore")
    match = ACCURACY_PATTERN.search(text)
    if match is None:
        raise ValueError(f"Could not find accuracy in {result_file}")
    return float(match.group(1))


def _collect_results(results_dir: Path, candidate_prefix: str) -> tuple[dict[tuple[str, str], float], list[str]]:
    scores: dict[tuple[str, str], float] = {}
    tasks: set[str] = set()

    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue

        parsed = _extract_task_and_candidate_id(child.name, candidate_prefix)
        if parsed is None:
            continue

        task_name, candidate_id = parsed
        result_file = child / "result_classification.txt"
        if not result_file.exists():
            raise FileNotFoundError(f"Missing result file: {result_file}")

        scores[(candidate_id, task_name)] = _read_accuracy(result_file)
        tasks.add(task_name)

    if not scores:
        raise ValueError(f"No classification results matched prefix '{candidate_prefix}' in {results_dir}")

    return scores, sorted(tasks)


def _build_rows(
    candidate_ids: list[str],
    tasks: list[str],
    scores: dict[tuple[str, str], float],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[tuple[str, str]]]:
    rows_with_id: list[dict[str, object]] = []
    rows_metrics_only: list[dict[str, object]] = []
    missing_entries: list[tuple[str, str]] = []

    for candidate_id in candidate_ids:
        metric_row: dict[str, object] = {}
        metric_values: list[float] = []
        has_missing = False

        for task_name in tasks:
            value = scores.get((candidate_id, task_name))
            if value is None:
                metric_row[task_name] = ""
                missing_entries.append((candidate_id, task_name))
                has_missing = True
            else:
                metric_row[task_name] = value
                metric_values.append(value)

        metric_row["mean_accuracy"] = "" if has_missing else mean(metric_values)
        row_with_id = {"candidate_id": candidate_id, **metric_row}
        rows_with_id.append(row_with_id)
        rows_metrics_only.append(metric_row)

    return rows_with_id, rows_metrics_only, missing_entries


def _write_csv(output_path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_arg_parser() -> argparse.ArgumentParser:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(
        description="Aggregate Time-Series-Library UEA classification accuracies into CSV tables."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repo_root / "results",
        help="Directory containing classification result folders.",
    )
    parser.add_argument(
        "--candidates-file",
        type=Path,
        default=repo_root / "candidates" / "timesnet_classification_uea_revised_candidates.json",
        help="Candidates JSON used to define the 100-model row order.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=repo_root / "results" / "timesnet_classification_uea_accuracy_summary.csv",
        help="Output CSV including candidate_id plus per-task accuracies and mean_accuracy.",
    )
    parser.add_argument(
        "--metrics-only-csv",
        type=Path,
        default=repo_root / "results" / "timesnet_classification_uea_accuracy_metrics_only.csv",
        help="Output CSV containing only the 10 task columns plus mean_accuracy.",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    candidate_ids, candidate_prefix = _load_candidate_ids(args.candidates_file)
    scores, tasks = _collect_results(args.results_dir, candidate_prefix)
    rows_with_id, rows_metrics_only, missing_entries = _build_rows(candidate_ids, tasks, scores)

    with_id_fieldnames = ["candidate_id", *tasks, "mean_accuracy"]
    metrics_only_fieldnames = [*tasks, "mean_accuracy"]
    _write_csv(args.output_csv, rows_with_id, with_id_fieldnames)
    _write_csv(args.metrics_only_csv, rows_metrics_only, metrics_only_fieldnames)

    print(f"Wrote {len(rows_with_id)} rows to {args.output_csv}")
    print(f"Wrote {len(rows_metrics_only)} rows to {args.metrics_only_csv}")
    print(f"Task columns ({len(tasks)}): {', '.join(tasks)}")
    if missing_entries:
        print(f"Missing entries: {len(missing_entries)}")
        for candidate_id, task_name in missing_entries[:20]:
            print(f"  - {candidate_id} / {task_name}")
    else:
        print("Missing entries: 0")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
