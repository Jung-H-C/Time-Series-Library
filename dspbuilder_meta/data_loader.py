from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import torch

from data_provider.data_factory import data_dict
from data_provider.m4 import M4Meta


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def split_dataset_input(raw_value: str) -> list[str]:
    return [token for token in re.split(r"[\s,]+", raw_value.strip()) if token]


@dataclass(frozen=True)
class BenchmarkTask:
    key: str
    display_name: str
    csv_path: Path
    metric_name: str
    proxy_names: tuple[str, ...]
    metrics: torch.Tensor
    proxies: torch.Tensor
    proxy_signature: torch.Tensor | None = None

    @property
    def num_candidates(self) -> int:
        return int(self.metrics.shape[0])


@dataclass
class TaskContext:
    benchmark: BenchmarkTask
    candidate_json_path: Path
    data_args: SimpleNamespace
    train_dataset: object
    sample_shape: tuple[int, ...]
    dataset_class_id: int | None = None


def extract_display_name(csv_path: Path) -> str:
    stem = csv_path.stem
    stem = stem.removeprefix("DSPBuilder_")
    suffixes = ("_Benchmark", "_benchmark", "_zscore", "_ZScore")
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                changed = True
    return stem


def benchmark_file_priority(csv_path: Path) -> tuple[int, int, str]:
    stem = csv_path.stem
    has_zscore_suffix = stem.lower().endswith("_zscore")
    is_legacy_named = stem.startswith("DSPBuilder_") or stem.endswith("_Benchmark") or stem.endswith("_benchmark")
    return (
        0 if has_zscore_suffix else 1,
        1 if is_legacy_named else 0,
        str(csv_path),
    )


def load_benchmark_task(
    csv_path: Path,
    proxy_signature_lookup: dict[str, dict[str, float]] | None = None,
) -> BenchmarkTask:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = [str(column).replace("\ufeff", "").strip() for column in next(reader)]
        rows = [[float(value) for value in row] for row in reader if row]

    if not rows:
        raise ValueError(f"Benchmark CSV is empty: {csv_path}")

    row_array = np.asarray(rows, dtype=np.float32)
    metric_name = str(header[0]).strip()
    proxy_names = tuple(str(column).strip() for column in header[1:])
    metrics = torch.tensor(row_array[:, 0], dtype=torch.float32)
    proxies = torch.tensor(row_array[:, 1:], dtype=torch.float32)
    display_name = extract_display_name(csv_path)
    key = normalize_name(display_name)
    proxy_signature: torch.Tensor | None = None
    if proxy_signature_lookup is not None:
        signature_row = proxy_signature_lookup.get(key)
        if signature_row is None:
            raise KeyError(f"Proxy signature for dataset '{display_name}' not found in lookup CSV.")
        missing_proxy_names = [proxy_name for proxy_name in proxy_names if proxy_name not in signature_row]
        if missing_proxy_names:
            raise KeyError(
                f"Proxy signature lookup is missing columns for dataset '{display_name}': {missing_proxy_names}"
            )
        proxy_signature = torch.tensor(
            [float(signature_row[proxy_name]) for proxy_name in proxy_names],
            dtype=torch.float32,
        )
    return BenchmarkTask(
        key=key,
        display_name=display_name,
        csv_path=csv_path,
        metric_name=metric_name,
        proxy_names=proxy_names,
        metrics=metrics,
        proxies=proxies,
        proxy_signature=proxy_signature,
    )


def discover_benchmark_tasks(
    benchmark_dir: Path,
    proxy_signature_lookup: dict[str, dict[str, float]] | None = None,
) -> dict[str, BenchmarkTask]:
    tasks: dict[str, BenchmarkTask] = {}
    for csv_path in sorted(benchmark_dir.glob("*.csv")):
        task = load_benchmark_task(csv_path, proxy_signature_lookup=proxy_signature_lookup)
        existing_task = tasks.get(task.key)
        if existing_task is None:
            tasks[task.key] = task
            continue

        current_priority = benchmark_file_priority(existing_task.csv_path)
        incoming_priority = benchmark_file_priority(csv_path)
        if incoming_priority < current_priority:
            tasks[task.key] = task
            continue
        if incoming_priority == current_priority:
            raise ValueError(
                f"Duplicate benchmark CSVs detected for dataset key '{task.key}': "
                f"{existing_task.csv_path.name}, {csv_path.name}"
            )
    if not tasks:
        raise FileNotFoundError(f"No benchmark CSV files found under {benchmark_dir}")
    return tasks


def discover_candidate_configs(candidate_dir: Path) -> dict[str, Path]:
    candidate_paths: dict[str, Path] = {}
    pattern = re.compile(r"^DSPBuilder_[^_]+_(.+)_candidates$")
    tsf_aliases: list[tuple[str, Path]] = []
    for json_path in sorted(candidate_dir.glob("DSPBuilder_*_candidates.json")):
        match = pattern.match(json_path.stem)
        if match is None:
            continue
        key = normalize_name(match.group(1))
        if key in candidate_paths:
            raise ValueError(f"Duplicate candidate config detected for dataset key: {key}")
        candidate_paths[key] = json_path
        if key.endswith("tsf"):
            tsf_aliases.append((key[:-3], json_path))
    for alias_key, json_path in tsf_aliases:
        candidate_paths.setdefault(alias_key, json_path)
    if not candidate_paths:
        raise FileNotFoundError(f"No candidate JSON files found under {candidate_dir}")
    return candidate_paths


def load_candidate_fixed_config(candidate_json_path: Path) -> dict[str, object]:
    payload = json.loads(candidate_json_path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    fixed_config = metadata.get("fixed_config")
    if fixed_config:
        return dict(fixed_config)
    candidates = payload.get("candidates", [])
    if candidates:
        return dict(candidates[0].get("run_args", {}))
    raise ValueError(f"Candidate JSON does not contain fixed_config or candidates: {candidate_json_path}")


def resolve_dataset_names(
    raw_names: Iterable[str],
    available_tasks: dict[str, BenchmarkTask],
    split_name: str,
) -> list[str]:
    resolved: list[str] = []
    missing: list[str] = []
    for raw_name in raw_names:
        key = normalize_name(raw_name)
        if key not in available_tasks:
            missing.append(raw_name)
            continue
        resolved.append(key)
    if missing:
        choices = ", ".join(task.display_name for task in available_tasks.values())
        raise ValueError(
            f"Unknown {split_name} dataset(s): {', '.join(missing)}. "
            f"Available datasets: {choices}"
        )
    return resolved


def prompt_dataset_names(split_name: str, available_tasks: dict[str, BenchmarkTask]) -> list[str]:
    choices = ", ".join(task.display_name for task in available_tasks.values())
    raw_value = input(f"{split_name} datasets (comma-separated) [{choices}]: ").strip()
    while not raw_value:
        raw_value = input(f"{split_name} datasets cannot be empty. Please enter again: ").strip()
    return resolve_dataset_names(split_dataset_input(raw_value), available_tasks, split_name)


def ensure_disjoint_splits(train_names: list[str], val_names: list[str], test_names: list[str]) -> None:
    seen: dict[str, str] = {}
    for split_name, names in (("train", train_names), ("val", val_names), ("test", test_names)):
        for name in names:
            if name in seen:
                raise ValueError(
                    f"Dataset '{name}' is assigned to both '{seen[name]}' and '{split_name}'. "
                    "Please keep train/val/test splits disjoint."
                )
            seen[name] = split_name


def build_dataset_namespace(data_config: dict[str, object], repo_root: Path) -> SimpleNamespace:
    defaults: dict[str, object] = {
        "data_path": "",
        "target": "OT",
        "embed": "timeF",
        "freq": "h",
        "batch_size": 32,
        "num_workers": 0,
        "seasonal_patterns": "Monthly",
        "augmentation_ratio": 0,
        "inverse": False,
        "features": "M",
    }
    merged = {**defaults, **data_config}

    root_path = Path(str(merged["root_path"]))
    if not root_path.is_absolute():
        root_path = (repo_root / root_path).resolve()
    merged["root_path"] = str(root_path)

    if merged["task_name"] == "short_term_forecast" and merged["data"] == "m4":
        seasonal_patterns = str(merged["seasonal_patterns"])
        pred_len = M4Meta.horizons_map[seasonal_patterns]
        merged["pred_len"] = int(pred_len)
        merged["seq_len"] = int(2 * pred_len)
        merged["label_len"] = int(pred_len)

    required_keys = ("seq_len", "label_len", "pred_len", "data", "task_name", "root_path")
    missing = [key for key in required_keys if key not in merged]
    if missing:
        raise ValueError(f"Missing required dataset config keys: {missing}")

    return SimpleNamespace(**merged)


class MetaSingleWindowM4Dataset:
    def __init__(self, base_dataset: object) -> None:
        self.base_dataset = base_dataset
        self.seq_len = int(base_dataset.seq_len)
        self.label_len = int(base_dataset.label_len)
        self.pred_len = int(base_dataset.pred_len)
        self.timeseries = base_dataset.timeseries
        self.ids = getattr(base_dataset, "ids", None)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getattr__(self, name: str):
        return getattr(self.base_dataset, name)

    def _deterministic_cut_point(self, series_length: int) -> int:
        # Use the latest cut point that still leaves up to pred_len future targets in the train series.
        return max(1, series_length - self.pred_len)

    def __getitem__(self, index: int):
        sampled_timeseries = np.asarray(self.timeseries[int(index)], dtype=np.float32).reshape(-1)
        insample = np.zeros((self.seq_len, 1), dtype=np.float32)
        insample_mask = np.zeros((self.seq_len, 1), dtype=np.float32)
        outsample = np.zeros((self.pred_len + self.label_len, 1), dtype=np.float32)
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1), dtype=np.float32)

        cut_point = self._deterministic_cut_point(int(sampled_timeseries.shape[0]))

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        if insample_window.size:
            insample[-insample_window.shape[0]:, 0] = insample_window
            insample_mask[-insample_window.shape[0]:, 0] = 1.0

        outsample_window = sampled_timeseries[
            max(0, cut_point - self.label_len):min(sampled_timeseries.shape[0], cut_point + self.pred_len)
        ]
        if outsample_window.size:
            outsample[:outsample_window.shape[0], 0] = outsample_window
            outsample_mask[:outsample_window.shape[0], 0] = 1.0

        return insample, outsample, insample_mask, outsample_mask


def instantiate_train_dataset(args: SimpleNamespace):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != "timeF" else 1
    dataset = Data(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag="train",
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq,
        seasonal_patterns=args.seasonal_patterns,
    )
    if args.task_name == "short_term_forecast" and args.data == "m4":
        return MetaSingleWindowM4Dataset(dataset)
    return dataset


def extract_input_sequence(sample) -> torch.Tensor:
    if isinstance(sample, (list, tuple)):
        if not sample:
            raise ValueError("Dataset sample is empty.")
        sample = sample[0]

    tensor = torch.as_tensor(sample, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim != 2:
        raise ValueError(f"Expected sample input with 2 dimensions [time, feature], got shape {tuple(tensor.shape)}")
    return tensor


def build_task_contexts(
    task_keys: list[str],
    available_tasks: dict[str, BenchmarkTask],
    candidate_configs: dict[str, Path],
    repo_root: Path,
    dataset_class_ids: dict[str, int] | None = None,
) -> list[TaskContext]:
    contexts: list[TaskContext] = []
    for key in task_keys:
        benchmark = available_tasks[key]
        candidate_json_path = candidate_configs.get(key)
        if candidate_json_path is None:
            raise FileNotFoundError(
                f"No candidate JSON found for dataset '{benchmark.display_name}' under {repo_root / 'candidates'}"
            )
        data_args = build_dataset_namespace(load_candidate_fixed_config(candidate_json_path), repo_root)
        train_dataset = instantiate_train_dataset(data_args)
        sample_shape = tuple(extract_input_sequence(train_dataset[0]).shape)
        contexts.append(
            TaskContext(
                benchmark=benchmark,
                candidate_json_path=candidate_json_path,
                data_args=data_args,
                train_dataset=train_dataset,
                sample_shape=sample_shape,
                dataset_class_id=dataset_class_ids.get(key) if dataset_class_ids is not None else None,
            )
        )
    return contexts
