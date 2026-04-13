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
    stem = stem.removesuffix("_Benchmark")
    return stem


def load_benchmark_task(csv_path: Path) -> BenchmarkTask:
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
    return BenchmarkTask(
        key=normalize_name(display_name),
        display_name=display_name,
        csv_path=csv_path,
        metric_name=metric_name,
        proxy_names=proxy_names,
        metrics=metrics,
        proxies=proxies,
    )


def discover_benchmark_tasks(benchmark_dir: Path) -> dict[str, BenchmarkTask]:
    tasks: dict[str, BenchmarkTask] = {}
    for csv_path in sorted(benchmark_dir.glob("*.csv")):
        task = load_benchmark_task(csv_path)
        if task.key in tasks:
            raise ValueError(f"Duplicate normalized dataset key detected: {task.key}")
        tasks[task.key] = task
    if not tasks:
        raise FileNotFoundError(f"No benchmark CSV files found under {benchmark_dir}")
    return tasks


def discover_candidate_configs(candidate_dir: Path) -> dict[str, Path]:
    candidate_paths: dict[str, Path] = {}
    pattern = re.compile(r"^DSPBuilder_[^_]+_(.+)_candidates$")
    for json_path in sorted(candidate_dir.glob("DSPBuilder_*_candidates.json")):
        match = pattern.match(json_path.stem)
        if match is None:
            continue
        key = normalize_name(match.group(1))
        if key in candidate_paths:
            raise ValueError(f"Duplicate candidate config detected for dataset key: {key}")
        candidate_paths[key] = json_path
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


def instantiate_train_dataset(args: SimpleNamespace):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != "timeF" else 1
    return Data(
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
