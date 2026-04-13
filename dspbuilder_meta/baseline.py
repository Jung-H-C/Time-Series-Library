from __future__ import annotations

import csv
from pathlib import Path
from typing import TypedDict

from .data_loader import normalize_name


class SpearmanBaselineEntry(TypedDict):
    dataset: str
    best_proxy: str
    coefficient: float


def load_spearman_baselines(csv_path: Path) -> dict[str, SpearmanBaselineEntry]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Spearman baseline CSV not found: {csv_path}")

    baselines: dict[str, SpearmanBaselineEntry] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"dataset", "best_proxy", "coefficient"}
        if reader.fieldnames is None or not required_columns.issubset(set(reader.fieldnames)):
            raise ValueError(
                "Spearman baseline CSV must contain the columns: dataset, best_proxy, coefficient"
            )

        for row in reader:
            dataset_name = str(row["dataset"]).strip()
            if not dataset_name:
                continue
            baselines[normalize_name(dataset_name)] = {
                "dataset": dataset_name,
                "best_proxy": str(row["best_proxy"]).strip(),
                "coefficient": float(row["coefficient"]),
            }
    return baselines
