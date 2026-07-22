from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


CANONICAL_BACKBONES = (
    "Autoformer",
    "Crossformer",
    "FiLM",
    "MICN",
    "Mamba",
    "PatchTST",
    "TimesNet",
    "Transformer",
)


BACKBONE_ALIASES = {
    "autoformer": "Autoformer",
    "crossformer": "Crossformer",
    "film": "FiLM",
    "micn": "MICN",
    "mamba": "Mamba",
    "patchtst": "PatchTST",
    "timesnet": "TimesNet",
    "transformer": "Transformer",
    "dlinear": "DLinear",
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    data: str
    root_path: str
    data_path: str
    enc_in: int
    pred_lens: tuple[int, ...]
    default_seq_len: int = 96
    default_label_len: int = 48
    default_batch_size: int = 32
    features: str = "M"
    target: str = "OT"
    freq: str = "h"


DATASETS = {
    "ECL": DatasetSpec(
        name="ECL",
        data="custom",
        root_path="./dataset/electricity/",
        data_path="electricity.csv",
        enc_in=321,
        pred_lens=(96, 192, 336, 720),
        default_batch_size=16,
    ),
    "ETTh1": DatasetSpec(
        name="ETTh1",
        data="ETTh1",
        root_path="./dataset/ETT-small/",
        data_path="ETTh1.csv",
        enc_in=7,
        pred_lens=(96, 192, 336, 720),
    ),
    "Exchange": DatasetSpec(
        name="Exchange",
        data="custom",
        root_path="./dataset/exchange_rate/",
        data_path="exchange_rate.csv",
        enc_in=8,
        pred_lens=(96, 192, 336, 720),
    ),
    "ILI": DatasetSpec(
        name="ILI",
        data="custom",
        root_path="./dataset/illness/",
        data_path="national_illness.csv",
        enc_in=7,
        pred_lens=(24, 36, 48, 60),
        default_seq_len=36,
        default_label_len=18,
    ),
    "Traffic": DatasetSpec(
        name="Traffic",
        data="custom",
        root_path="./dataset/traffic/",
        data_path="traffic.csv",
        enc_in=862,
        pred_lens=(96, 192, 336, 720),
        default_batch_size=4,
    ),
    "Weather": DatasetSpec(
        name="Weather",
        data="custom",
        root_path="./dataset/weather/",
        data_path="weather.csv",
        enc_in=21,
        pred_lens=(96, 192, 336, 720),
    ),
}


def _summary_datasets() -> dict[str, DatasetSpec]:
    """Build runnable multi-series specs from rows explicitly marked horizon=96."""
    repo_root = Path(__file__).resolve().parents[2]
    families = (
        ("Monash", repo_root / "dataset/monash_dataset_summary.csv", "./dataset/Monash_Dataset/"),
        ("TIME", repo_root / "dataset/time_dataset_summary.csv", "./dataset/Time_Dataset/"),
    )
    result: dict[str, DatasetSpec] = {}
    for family, summary_path, root_path in families:
        if not summary_path.is_file():
            continue
        with summary_path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                try:
                    horizon = float(str(row.get("horizon", "")).strip())
                except ValueError:
                    continue
                if not math.isfinite(horizon) or horizon != 96:
                    continue
                dataset_name = str(row.get("dataset_name", "")).strip()
                source_files = str(row.get("source_files", "")).strip()
                source_format = str(row.get("source_format", "")).strip().lower()
                if not dataset_name or not source_files or source_format not in {"tsf", "rds", "arrow"}:
                    continue

                # Every TSF row and the supported RDS files are independent
                # univariate series. TIME Arrow target rows preserve their
                # actual multivariate channel dimension.
                if source_format == "arrow":
                    try:
                        enc_in = int(row["channel_count"])
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Invalid channel_count for {family} dataset {dataset_name!r}"
                        ) from exc
                else:
                    enc_in = 1

                registry_name = f"{family}__{dataset_name}"
                result[registry_name] = DatasetSpec(
                    name=registry_name,
                    data="multi_series",
                    root_path=root_path,
                    data_path=source_files,
                    enc_in=enc_in,
                    pred_lens=(96,),
                    default_seq_len=96,
                    default_label_len=48,
                    default_batch_size=16 if enc_in > 16 else 32,
                    features="M",
                    target="OT",
                    # The loader emits zero-valued 4-column time marks; hourly
                    # keeps that contract valid for timeF embeddings.
                    freq="h",
                )
    return result


DATASETS.update(_summary_datasets())


def normalize_backbone(name: str) -> str:
    normalized = str(name).replace("_", "").replace("-", "").lower()
    if normalized not in BACKBONE_ALIASES:
        allowed = ", ".join(CANONICAL_BACKBONES)
        raise ValueError(f"Unknown backbone '{name}'. Allowed: {allowed}")
    return BACKBONE_ALIASES[normalized]


def normalize_dataset(name: str) -> str:
    lookup = {dataset_name.lower(): dataset_name for dataset_name in DATASETS}
    normalized = str(name).replace("_", "").replace("-", "").lower()
    compact_lookup = {
        dataset_name.replace("_", "").replace("-", "").lower(): dataset_name
        for dataset_name in DATASETS
    }
    if normalized in compact_lookup:
        return compact_lookup[normalized]
    if str(name).lower() in lookup:
        return lookup[str(name).lower()]
    allowed = ", ".join(DATASETS)
    raise ValueError(f"Unknown dataset '{name}'. Allowed: {allowed}")
