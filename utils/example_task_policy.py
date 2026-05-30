from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _normalize_name(value: object) -> str:
    return str(value or "").strip()


def _iter_recipe_pairs(payload: dict) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()

    runs = payload.get("runs")
    if isinstance(runs, list):
        for run in runs:
            if not isinstance(run, dict):
                continue
            run_args = run.get("run_args")
            if not isinstance(run_args, dict):
                continue
            model_name = _normalize_name(run_args.get("model"))
            task_name = _normalize_name(run_args.get("task_name"))
            if model_name and task_name:
                pairs.add((model_name, task_name))

    if pairs:
        return pairs

    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return pairs

    models = summary.get("models")
    tasks = summary.get("tasks")
    if not isinstance(models, list) or not isinstance(tasks, list):
        return pairs

    normalized_models = [_normalize_name(model_name) for model_name in models]
    normalized_tasks = [_normalize_name(task_name) for task_name in tasks]
    for model_name in normalized_models:
        for task_name in normalized_tasks:
            if model_name and task_name:
                pairs.add((model_name, task_name))
    return pairs


@lru_cache(maxsize=None)
def discover_example_task_backbone_pairs(repo_root: str | None = None) -> dict[str, tuple[str, ...]]:
    base_dir = Path(repo_root) if repo_root else _repo_root()
    examples_dir = base_dir / "examples"
    allowed_pairs: dict[str, set[str]] = {}

    if not examples_dir.exists():
        return {}

    for recipe_path in sorted(examples_dir.rglob("*.json")):
        payload = json.loads(recipe_path.read_text(encoding="utf-8"))
        for model_name, task_name in _iter_recipe_pairs(payload):
            allowed_pairs.setdefault(model_name, set()).add(task_name)

    return {
        model_name: tuple(sorted(task_names))
        for model_name, task_names in sorted(allowed_pairs.items())
    }


def ensure_example_task_backbone_supported(
    model_name: object,
    task_name: object,
    repo_root: str | None = None,
) -> None:
    normalized_model = _normalize_name(model_name)
    normalized_task = _normalize_name(task_name)
    if not normalized_model or not normalized_task:
        return

    allowed_pairs = discover_example_task_backbone_pairs(repo_root)
    allowed_tasks = allowed_pairs.get(normalized_model)
    if allowed_tasks is None:
        raise NotImplementedError(
            f"Backbone '{normalized_model}' is blocked by the examples-based task policy. "
            "No recipe under examples/ defines this backbone."
        )
    if normalized_task not in allowed_tasks:
        allowed_display = ", ".join(allowed_tasks)
        raise NotImplementedError(
            f"Task '{normalized_task}' is blocked for backbone '{normalized_model}' by the "
            f"examples-based task policy. Allowed tasks from examples/: {allowed_display}."
        )
