from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


METRIC_NAMES = ("mae", "mse", "rmse", "mape", "mspe")
CANDIDATE_PATTERN = re.compile(
    r"_(?P<candidate_id>[A-Za-z][A-Za-z0-9]*_\d+)_sl(?P<seq_len>\d+)_pl(?P<pred_len>\d+)_"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect multi-backbone proxy result folders into one CSV. "
            "Rows are candidate models; columns include parsed identifiers, "
            "candidate hyperparameters, and metrics.npy values."
        )
    )
    parser.add_argument(
        "folder_prefix",
        help=(
            "Prefix of result folder names to collect, e.g. "
            "long_term_forecast_mbproxy_ECL_Autoformer"
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Time-Series-Library repo root. Default: inferred from this script path.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Results directory. Default: <repo-root>/results.",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        nargs="*",
        default=None,
        help=(
            "Candidate JSON file(s) used to recover hyperparameters. "
            "Default: auto-load <repo-root>/candidates/*.json."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output CSV path. Default: "
            "<repo-root>/results/<folder_prefix>_summary.csv."
        ),
    )
    parser.add_argument(
        "--strict-candidates",
        action="store_true",
        help="Fail if a result candidate_id is missing from candidate JSON metadata.",
    )
    return parser.parse_args()


def load_candidate_lookup(paths: list[Path]) -> tuple[dict[str, dict[str, Any]], set[str]]:
    lookup: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Candidate JSON not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            candidate_id = candidate.get("candidate_id")
            if not isinstance(candidate_id, str) or not candidate_id:
                continue
            record = {
                "candidate_json": str(path),
                "candidate_id": candidate_id,
                "backbone": candidate.get("backbone"),
                "candidate_split": candidate.get("split"),
                "is_default": candidate.get("is_default"),
                **dict(candidate.get("run_args") or {}),
            }
            if candidate_id in lookup and lookup[candidate_id] != record:
                duplicates.add(candidate_id)
                # Keep the first record deterministic; duplicate ids can be
                # disambiguated by passing the intended --candidates file.
                continue
            lookup[candidate_id] = record

    return lookup, duplicates


def parse_result_folder(folder_name: str) -> dict[str, Any]:
    match = CANDIDATE_PATTERN.search(folder_name)
    if not match:
        return {}

    candidate_id = match.group("candidate_id")
    backbone = candidate_id.rsplit("_", 1)[0]
    split = ""
    if "_proxy_train_" in folder_name:
        split = "proxy_train"
    elif "_proxy_eval_" in folder_name:
        split = "proxy_eval"

    return {
        "candidate_id": candidate_id,
        "backbone": backbone,
        "seq_len": int(match.group("seq_len")),
        "pred_len": int(match.group("pred_len")),
        "split": split,
    }


def load_metrics(metrics_path: Path) -> dict[str, float]:
    values = np.load(metrics_path)
    flat = np.asarray(values).reshape(-1)
    if flat.shape[0] < len(METRIC_NAMES):
        raise ValueError(f"Expected at least {len(METRIC_NAMES)} metrics in {metrics_path}, got {flat.shape[0]}")
    return {name: float(flat[index]) for index, name in enumerate(METRIC_NAMES)}


def natural_candidate_key(row: dict[str, Any]) -> tuple[str, int, str]:
    candidate_id = str(row.get("candidate_id", ""))
    match = re.search(r"_(\d+)$", candidate_id)
    candidate_num = int(match.group(1)) if match else -1
    return (str(row.get("backbone", "")), candidate_num, candidate_id)


def collect_rows(
    results_dir: Path,
    folder_prefix: str,
    candidate_lookup: dict[str, dict[str, Any]],
    strict_candidates: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    folders = sorted(
        path for path in results_dir.iterdir()
        if path.is_dir() and path.name.startswith(folder_prefix)
    )
    if not folders:
        raise FileNotFoundError(f"No result folders start with '{folder_prefix}' under {results_dir}")

    for folder in folders:
        metrics_path = folder / "metrics.npy"
        if not metrics_path.exists():
            continue

        parsed = parse_result_folder(folder.name)
        candidate_id = parsed.get("candidate_id")
        candidate_record = candidate_lookup.get(candidate_id, {}) if candidate_id else {}
        if strict_candidates and candidate_id and not candidate_record:
            raise KeyError(f"Candidate '{candidate_id}' from {folder.name} was not found in candidate JSON files.")

        metrics = load_metrics(metrics_path)
        row = {
            "result_folder": folder.name,
            "metrics_path": str(metrics_path),
            **parsed,
            **candidate_record,
            **metrics,
        }
        # Keep folder-parsed split if candidate JSON is absent; otherwise expose
        # both names without losing the JSON split.
        if "candidate_split" in row and row.get("split") and row["candidate_split"] != row["split"]:
            row["folder_split"] = row["split"]
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No metrics.npy files found for folders starting with '{folder_prefix}'.")

    return sorted(rows, key=natural_candidate_key)


def ordered_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "candidate_id",
        "backbone",
        "candidate_split",
        "folder_split",
        "is_default",
        "seq_len",
        "label_len",
        "pred_len",
        "mae",
        "mse",
        "rmse",
        "mape",
        "mspe",
        "candidate_json",
        "result_folder",
        "metrics_path",
    ]
    all_keys = {key for row in rows for key in row}
    remaining = sorted(key for key in all_keys if key not in preferred)
    return [key for key in preferred if key in all_keys] + remaining


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    results_dir = args.results_dir.resolve() if args.results_dir else repo_root / "results"
    candidate_paths = (
        [path.resolve() for path in args.candidates]
        if args.candidates is not None and len(args.candidates) > 0
        else sorted((repo_root / "candidates").glob("*.json"))
    )
    output_path = (
        args.output.resolve()
        if args.output
        else results_dir / f"{args.folder_prefix}_summary.csv"
    )

    candidate_lookup, duplicates = load_candidate_lookup(candidate_paths)
    rows = collect_rows(results_dir, args.folder_prefix, candidate_lookup, args.strict_candidates)
    fieldnames = ordered_fieldnames(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")
    if duplicates:
        duplicate_text = ", ".join(sorted(duplicates)[:10])
        suffix = " ..." if len(duplicates) > 10 else ""
        print(
            "Warning: duplicate candidate_id metadata found while auto-loading candidate JSON files. "
            f"Kept the first record for: {duplicate_text}{suffix}. "
            "Pass --candidates with the exact source JSON to disambiguate."
        )


if __name__ == "__main__":
    main()
