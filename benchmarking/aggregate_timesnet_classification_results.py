from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


EXPERIMENT_PATTERNS = (
    re.compile(
        r"^classification_(?P<dataset>.+?)_(?P<signature>timesnet_classification_uea_"
        r"(?P<config>.+?)_(?P<candidate_id>\d{4}))$"
    ),
    re.compile(
        r"^classification_(?P<dataset>.+?)_TimesNet_UEA_.*_Exp__"
        r"(?P<signature>timesnet_classification_uea_(?P<config>.+?)_(?P<candidate_id>\d{4})_0)$"
    ),
)
CONFIG_PATTERN = re.compile(
    r"^el(?P<e_layers>\d+)_dm(?P<d_model>\d+)_df(?P<d_ff>\d+)_tk(?P<top_k>\d+)_nk(?P<num_kernels>\d+)$"
)
METRIC_PATTERN = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*):(?P<value>[^,\s]+)")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_metric_value(raw_value: str) -> int | float | str:
    if re.fullmatch(r"[-+]?\d+", raw_value):
        return int(raw_value)

    try:
        return float(raw_value)
    except ValueError:
        return raw_value


def _match_experiment_name(experiment_name: str) -> re.Match[str] | None:
    for pattern in EXPERIMENT_PATTERNS:
        match = pattern.match(experiment_name)
        if match is not None:
            return match
    return None


def _parse_metrics(result_path: Path) -> dict[str, int | float | str]:
    metrics: dict[str, int | float | str] = {}
    for line in result_path.read_text(encoding="utf-8").splitlines():
        for match in METRIC_PATTERN.finditer(line):
            metrics[match.group("key")] = _parse_metric_value(match.group("value"))
    return metrics


def _metric_sort_key(metric_name: str) -> tuple[int, str]:
    preferred_order = {"accuracy": 0, "final_epoch": 1}
    return (preferred_order.get(metric_name, 2), metric_name)


def aggregate_results(results_root: Path) -> tuple[list[dict[str, object]], list[str], list[str]]:
    rows_by_candidate: dict[str, dict[str, object]] = {}
    datasets: set[str] = set()
    metrics_seen: set[str] = set()

    for result_path in sorted(results_root.glob("classification_*/result_classification.txt")):
        experiment_name = result_path.parent.name
        experiment_match = _match_experiment_name(experiment_name)
        if experiment_match is None:
            continue

        dataset = experiment_match.group("dataset")
        candidate_id = experiment_match.group("candidate_id")
        candidate_config = experiment_match.group("config")
        candidate_signature = experiment_match.group("signature")

        config_match = CONFIG_PATTERN.match(candidate_config)
        if config_match is None:
            raise ValueError(f"Unable to parse candidate configuration from '{candidate_config}'.")

        parsed_metrics = _parse_metrics(result_path)
        if not parsed_metrics:
            raise ValueError(f"No metrics found in '{result_path}'.")

        datasets.add(dataset)
        metrics_seen.update(parsed_metrics.keys())

        row = rows_by_candidate.setdefault(
            candidate_id,
            {
                "candidate_id": candidate_id,
                "candidate_signature": candidate_signature,
                "candidate_config": candidate_config,
                "e_layers": int(config_match.group("e_layers")),
                "d_model": int(config_match.group("d_model")),
                "d_ff": int(config_match.group("d_ff")),
                "top_k": int(config_match.group("top_k")),
                "num_kernels": int(config_match.group("num_kernels")),
                "task_metrics": {},
            },
        )

        if row["candidate_config"] != candidate_config or row["candidate_signature"] != candidate_signature:
            raise ValueError(f"Inconsistent candidate mapping found for candidate_id='{candidate_id}'.")

        task_metrics = row["task_metrics"]
        if dataset in task_metrics:
            raise ValueError(f"Duplicate task result found for candidate_id='{candidate_id}', dataset='{dataset}'.")
        task_metrics[dataset] = parsed_metrics

    sorted_datasets = sorted(datasets)
    sorted_metrics = sorted(metrics_seen, key=_metric_sort_key)

    flattened_rows: list[dict[str, object]] = []
    for candidate_id in sorted(rows_by_candidate):
        row = rows_by_candidate[candidate_id]
        task_metrics = row.pop("task_metrics")
        flattened_row = dict(row)
        flattened_row["num_tasks_completed"] = len(task_metrics)

        accuracy_values: list[float] = []
        for dataset in sorted_datasets:
            dataset_metrics = task_metrics.get(dataset, {})
            for metric_name in sorted_metrics:
                value = dataset_metrics.get(metric_name, "")
                flattened_row[f"{dataset}__{metric_name}"] = value
                if metric_name == "accuracy" and isinstance(value, (int, float)):
                    accuracy_values.append(float(value))

        flattened_row["mean_accuracy"] = (
            round(sum(accuracy_values) / len(accuracy_values), 6) if accuracy_values else ""
        )
        flattened_rows.append(flattened_row)

    return flattened_rows, sorted_datasets, sorted_metrics


def write_csv(
    output_path: Path, rows: list[dict[str, object]], datasets: list[str], metrics: list[str]
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "candidate_id",
        "candidate_signature",
        "candidate_config",
        "e_layers",
        "d_model",
        "d_ff",
        "top_k",
        "num_kernels",
        "num_tasks_completed",
        "mean_accuracy",
    ]
    for dataset in datasets:
        for metric_name in metrics:
            fieldnames.append(f"{dataset}__{metric_name}")

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(
        description="Aggregate TimesNet UEA classification results into a single CSV file."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=repo_root / "results",
        help="Directory that contains classification result folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "results" / "timesnet_classification_candidate_metrics.csv",
        help="CSV file path to write.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    rows, datasets, metrics = aggregate_results(args.results_root)
    if not rows:
        raise SystemExit(f"No TimesNet UEA classification results were found in '{args.results_root}'.")

    write_csv(args.output, rows, datasets, metrics)

    print(f"Wrote {len(rows)} candidate rows to {args.output}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Metrics: {', '.join(metrics)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
