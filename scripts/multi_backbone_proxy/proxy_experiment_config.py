from __future__ import annotations

from dataclasses import dataclass


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
