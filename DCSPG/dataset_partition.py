from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_VALIDATION_DATASETS = (
    "Coastal_T_S__H",
    "sunspot_dataset_without_missing_values",
    "Australia_Solar__H",
    "Water_Quality_Darwin__15T",
    "current_velocity__20T",
    "wind_4_seconds_dataset",
    "SG_Carpark__15T",
    "Port_Activity__D",
)

DEFAULT_TEST_DATASETS = (
    "electricity",
    "exchange_rate",
    "illness",
    "traffic",
    "weather",
    "ETT-small",
)

# requested alias -> (TS feature NPZ dataset name, proxy-score/GroundTruth name)
BENCHMARK_ALIASES = {
    "ECL": ("electricity", "ECL"),
    "electricity": ("electricity", "ECL"),
    "ETTh1": ("ETT-small", "ETTh1"),
    "ETT-small": ("ETT-small", "ETTh1"),
    "Exchange": ("exchange_rate", "Exchange"),
    "exchange_rate": ("exchange_rate", "Exchange"),
    "Illness": ("illness", "ILI"),
    "ILI": ("illness", "ILI"),
    "illness": ("illness", "ILI"),
    "Traffic": ("traffic", "Traffic"),
    "traffic": ("traffic", "Traffic"),
    "Weather": ("weather", "Weather"),
    "weather": ("weather", "Weather"),
}


@dataclass(frozen=True)
class DatasetPartition:
    train_datasets: tuple[str, ...]
    validation_datasets: tuple[str, ...]
    test_datasets: tuple[str, ...]
    cluster_datasets: dict[int, tuple[str, ...]]
    proxy_dataset_names: dict[str, str]


def parse_dataset_csv_list(raw: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(raw, str):
        values = raw.split(",")
    else:
        values = raw
    return tuple(value.strip() for value in values if value.strip())


def read_cluster_assignments(
    cluster_csv: Path | str,
) -> tuple[dict[str, int], dict[str, str]]:
    cluster_csv = Path(cluster_csv)
    with cluster_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        rows = list(reader)

    cluster_by_ts_name = {}
    proxy_name_by_ts_name = {}
    dataset_level_columns = {"cluster_id", "dataset_id", "dataset_name"}
    summary_columns = {"cluster_id", "cluster_size", "dataset_names"}
    if dataset_level_columns.issubset(fieldnames):
        for row in rows:
            ts_name = row["dataset_name"].strip()
            if ts_name in cluster_by_ts_name:
                raise ValueError(f"Duplicate dataset_name in {cluster_csv}: {ts_name}")
            cluster_by_ts_name[ts_name] = int(row["cluster_id"])
            proxy_name_by_ts_name[ts_name] = row["dataset_id"].strip()
    elif summary_columns.issubset(fieldnames):
        for row in rows:
            cluster_id = int(row["cluster_id"])
            dataset_names = tuple(
                name.strip()
                for name in row["dataset_names"].split(";")
                if name.strip()
            )
            if len(dataset_names) != int(row["cluster_size"]):
                raise ValueError(
                    f"Cluster {cluster_id} in {cluster_csv} declares size "
                    f"{row['cluster_size']}, but contains {len(dataset_names)} names"
                )
            for ts_name in dataset_names:
                if ts_name in cluster_by_ts_name:
                    raise ValueError(f"Duplicate dataset_name in {cluster_csv}: {ts_name}")
                cluster_by_ts_name[ts_name] = cluster_id

        companion_name = cluster_csv.name.replace(
            "cluster_summary_", "dataset_clusters_", 1
        )
        companion_csv = cluster_csv.with_name(companion_name)
        if companion_csv == cluster_csv or not companion_csv.is_file():
            raise FileNotFoundError(
                "A cluster summary requires its dataset-level companion for "
                f"dataset_id mapping: {companion_csv}"
            )
        with companion_csv.open(newline="", encoding="utf-8") as handle:
            companion_reader = csv.DictReader(handle)
            missing = dataset_level_columns.difference(companion_reader.fieldnames or [])
            if missing:
                raise ValueError(f"{companion_csv} is missing columns: {sorted(missing)}")
            companion_rows = list(companion_reader)
        companion_clusters = {
            row["dataset_name"].strip(): int(row["cluster_id"])
            for row in companion_rows
        }
        if companion_clusters != cluster_by_ts_name:
            raise ValueError(
                f"Cluster assignments disagree between {cluster_csv} and {companion_csv}"
            )
        proxy_name_by_ts_name = {
            row["dataset_name"].strip(): row["dataset_id"].strip()
            for row in companion_rows
        }
    else:
        expected = sorted(dataset_level_columns) + sorted(summary_columns)
        raise ValueError(
            f"{cluster_csv} must be a dataset-level or summary cluster CSV; "
            f"expected columns from {expected}, got {sorted(fieldnames)}"
        )
    return cluster_by_ts_name, proxy_name_by_ts_name


def canonicalize_dataset_name(
    requested_name: str,
    available_ts_names: tuple[str, ...],
    proxy_name_by_ts_name: dict[str, str],
) -> str:
    if requested_name in available_ts_names:
        return requested_name
    if requested_name in BENCHMARK_ALIASES:
        ts_name = BENCHMARK_ALIASES[requested_name][0]
        if ts_name in available_ts_names:
            return ts_name

    reverse_proxy_names = {
        proxy_name: ts_name for ts_name, proxy_name in proxy_name_by_ts_name.items()
    }
    if requested_name in reverse_proxy_names:
        return reverse_proxy_names[requested_name]

    # Accept a uniquely abbreviated suffix such as the requested
    # "Port_Activity__" for the actual "Port_Activity__D" dataset.
    prefix_matches = [name for name in available_ts_names if name.startswith(requested_name)]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    casefold_matches = [
        name for name in available_ts_names if name.casefold() == requested_name.casefold()
    ]
    if len(casefold_matches) == 1:
        return casefold_matches[0]
    raise ValueError(
        f"Unknown or ambiguous dataset {requested_name!r}. "
        f"Available TS datasets: {available_ts_names}"
    )


def build_dataset_partition(
    available_ts_names: Iterable[str],
    validation_datasets: str | Iterable[str],
    test_datasets: str | Iterable[str],
    cluster_csv: Path | str,
) -> DatasetPartition:
    available = tuple(available_ts_names)
    if len(available) != 53 or len(set(available)) != 53:
        raise ValueError(f"Expected exactly 53 unique TS datasets, got {len(set(available))}")

    cluster_by_ts_name, proxy_name_by_ts_name = read_cluster_assignments(cluster_csv)
    validation = tuple(
        canonicalize_dataset_name(name, available, proxy_name_by_ts_name)
        for name in parse_dataset_csv_list(validation_datasets)
    )
    test = tuple(
        canonicalize_dataset_name(name, available, proxy_name_by_ts_name)
        for name in parse_dataset_csv_list(test_datasets)
    )
    if len(validation) != 8 or len(set(validation)) != 8:
        raise ValueError(f"Validation split must contain 8 unique datasets, got {validation}")
    if len(test) != 6 or len(set(test)) != 6:
        raise ValueError(f"Test split must contain 6 unique datasets, got {test}")
    overlap = sorted(set(validation).intersection(test))
    if overlap:
        raise ValueError(f"Validation and test splits overlap: {overlap}")

    cluster_ids = tuple(sorted(set(cluster_by_ts_name.values())))
    expected_cluster_ids = tuple(range(1, len(cluster_ids) + 1))
    if cluster_ids != expected_cluster_ids:
        raise ValueError(
            "Cluster ids must be contiguous and start at 1; "
            f"got {cluster_ids} in {cluster_csv}"
        )
    validation_without_cluster = sorted(set(validation).difference(cluster_by_ts_name))
    if validation_without_cluster:
        raise ValueError(
            "Every validation dataset needs a cluster assignment. Missing from "
            f"{cluster_csv}: {validation_without_cluster}"
        )
    validation_cluster_ids = tuple(cluster_by_ts_name[name] for name in validation)
    missing_validation_cluster_ids = tuple(
        cluster_id
        for cluster_id in expected_cluster_ids
        if cluster_id not in validation_cluster_ids
    )
    if missing_validation_cluster_ids:
        raise ValueError(
            "Validation split must represent every cluster; missing cluster ids "
            f"{missing_validation_cluster_ids}, got {validation_cluster_ids}"
        )

    excluded = set(validation).union(test)
    train = tuple(name for name in available if name not in excluded)
    if len(train) != 39:
        raise RuntimeError(f"Expected 39 training datasets, got {len(train)}")
    unassigned = sorted(set(train).difference(cluster_by_ts_name))
    if unassigned:
        raise ValueError(
            "Every training dataset needs a cluster assignment. Missing from "
            f"{cluster_csv}: {unassigned}"
        )

    all_cluster_datasets = {
        cluster_id: tuple(name for name in train if cluster_by_ts_name[name] == cluster_id)
        for cluster_id in expected_cluster_ids
    }
    cluster_datasets = {
        cluster_id: names
        for cluster_id, names in all_cluster_datasets.items()
        if names
    }
    if not cluster_datasets:
        raise ValueError("At least one cluster must contain training datasets")

    proxy_names = dict(proxy_name_by_ts_name)
    for _alias, (ts_name, proxy_name) in BENCHMARK_ALIASES.items():
        proxy_names[ts_name] = proxy_name
    missing_proxy_names = sorted(set(available).difference(proxy_names))
    if missing_proxy_names:
        raise ValueError(f"Missing proxy-score dataset names: {missing_proxy_names}")

    return DatasetPartition(
        train_datasets=train,
        validation_datasets=validation,
        test_datasets=test,
        cluster_datasets=cluster_datasets,
        proxy_dataset_names=proxy_names,
    )


def resolve_proxy_score_csv(
    proxy_score_dir: Path | str,
    proxy_dataset_name: str,
) -> Path:
    proxy_score_dir = Path(proxy_score_dir)
    matches = sorted(
        path
        for path in proxy_score_dir.rglob(
            f"*_{proxy_dataset_name}_proxy_scores_*.csv"
        )
        if path.is_file()
    )
    if len(matches) != 1:
        joined = ", ".join(str(path) for path in matches) or "none"
        raise FileNotFoundError(
            f"Expected one proxy-score CSV for {proxy_dataset_name!r} under "
            f"{proxy_score_dir}, found: {joined}"
        )
    return matches[0]
