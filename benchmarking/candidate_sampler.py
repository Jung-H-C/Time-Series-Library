from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import queue
import random
import re
import shlex
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


RUN_ARG_PATTERN = re.compile(r"add_argument\('--(?P<name>[^']+)'")
MULTIPLIER_PATTERN = re.compile(r"^x(?P<factor>\d+(?:\.\d+)?)$")
CONFIG_ATTR_PATTERN = re.compile(r"configs\.(?P<name>[A-Za-z_][A-Za-z0-9_]*)")
CANDIDATE_SUFFIX_PATTERN = re.compile(r"_(\d+)$")
TRAINING_SETTING_PATTERN = re.compile(r"^>>>>>>>start training : (?P<setting>.+?)>+$")
TESTING_SETTING_PATTERN = re.compile(r"^>>>>>>>testing : (?P<setting>.+?)<+$")
ACCURACY_LINE_PATTERN = re.compile(r"^accuracy:(?P<accuracy>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$")

PARAMETER_ALIASES = {
    "e_layer": "e_layers",
    "encoder_layers": "e_layers",
    "top_j": "top_k",
    "topk": "top_k",
    "num_kernel": "num_kernels",
    "num_kernals": "num_kernels",
}


def _positive_int(param_name: str, value: Any, _: dict[str, Any]) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"'{param_name}' must resolve to a positive integer, got {value!r}.")


def _positive_even_int(param_name: str, value: Any, _: dict[str, Any]) -> None:
    _positive_int(param_name, value, {})
    if value % 2 != 0:
        raise ValueError(
            f"'{param_name}' must be even for the current positional embedding implementation, got {value}."
        )


def _timesnet_top_k(param_name: str, value: Any, merged_config: dict[str, Any]) -> None:
    _positive_int(param_name, value, merged_config)

    seq_len = merged_config.get("seq_len")
    if not isinstance(seq_len, int) or seq_len < 1:
        return

    task_name = merged_config.get("task_name", "long_term_forecast")
    pred_len = merged_config.get("pred_len", 0)
    if not isinstance(pred_len, int):
        pred_len = 0

    if task_name in {"long_term_forecast", "short_term_forecast", "zero_shot_forecast"}:
        total_length = seq_len + pred_len
    else:
        total_length = seq_len

    max_bins = math.floor(total_length / 2) + 1
    if value > max_bins:
        raise ValueError(
            f"'{param_name}'={value} is too large for task='{task_name}' with seq_len={seq_len} "
            f"and pred_len={pred_len}. Maximum valid value is {max_bins}."
        )


@dataclass(frozen=True)
class ParamRule:
    validator: Callable[[str, Any, dict[str, Any]], None]
    relative_to: str | None = None
    allow_multiplier: bool = False


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    param_rules: dict[str, ParamRule]


@dataclass(frozen=True)
class CandidateRunPlan:
    index: int
    total: int
    candidate_name: str
    candidate: dict[str, Any]


BACKBONE_SPECS = {
    "TimesNet": BackboneSpec(
        name="TimesNet",
        param_rules={
            "e_layers": ParamRule(validator=_positive_int),
            "d_model": ParamRule(validator=_positive_even_int),
            "d_ff": ParamRule(validator=_positive_int, relative_to="d_model", allow_multiplier=True),
            "top_k": ParamRule(validator=_timesnet_top_k),
            "num_kernels": ParamRule(validator=_positive_int),
        },
    ),
}

NAME_TOKEN_ALIASES = {
    "alpha": "alpha",
    "anomaly_ratio": "ar",
    "batch_size": "bs",
    "c_out": "co",
    "channel_independence": "ci",
    "d_conv": "dc",
    "d_ff": "df",
    "d_layers": "dl",
    "d_model": "dm",
    "dt_rank": "dtr",
    "dropout": "do",
    "e_layers": "el",
    "embed": "eb",
    "enc_in": "ei",
    "factor": "fc",
    "features": "ft",
    "freq": "fq",
    "gcn_depth": "gd",
    "individual": "ind",
    "label_len": "ll",
    "learning_rate": "lr",
    "moving_avg": "ma",
    "n_heads": "nh",
    "num_kernels": "nk",
    "patch_len": "patch",
    "pred_len": "pl",
    "seq_len": "sl",
    "seg_len": "seg",
    "top_k": "tk",
    "top_p": "tp",
}

DEFAULT_SPEC_FIXED_KEYS = [
    "task_name",
    "is_training",
    "data",
    "root_path",
    "data_path",
    "features",
    "target",
    "freq",
    "checkpoints",
    "seq_len",
    "label_len",
    "pred_len",
    "enc_in",
    "dec_in",
    "c_out",
]

STORE_TRUE_RUN_ARGS = {
    "inverse",
    "use_amp",
    "use_multi_gpu",
    "use_dtw",
    "jitter",
    "scaling",
    "permutation",
    "randompermutation",
    "magwarp",
    "timewarp",
    "windowslice",
    "windowwarp",
    "rotation",
    "spawner",
    "dtwwarp",
    "shapedtwwarp",
    "wdba",
    "discdtw",
    "discsdtw",
    "individual",
}

STORE_FALSE_RUN_ARGS = {
    "distil": "--distil",
    "use_gpu": "--no_use_gpu",
}

UEA_RECIPE_OVERRIDE_KEYS = (
    "root_path",
    "model_id",
    "batch_size",
    "learning_rate",
    "train_epochs",
    "patience",
    "itr",
)

UEA_AVERAGE_SUMMARY_FIELDS = [
    "candidate_id",
    "candidate_name",
    "model",
    "task_name",
    "data",
    "model_num",
    "e_layers",
    "d_model",
    "d_ff",
    "top_k",
    "num_kernels",
    "num_subsets_total",
    "num_subsets_completed",
    "average_accuracy",
    "status",
    "error",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def discover_available_backbones(repo_root: Path | None = None) -> list[str]:
    repo_root = repo_root or _repo_root()
    models_dir = repo_root / "models"
    return sorted(
        path.stem for path in models_dir.glob("*.py") if path.name != "__init__.py"
    )


def _resolve_backbone_name(
    raw_backbone: Any,
    available_backbones: list[str],
    *,
    strict_known: bool,
) -> str:
    if not isinstance(raw_backbone, str) or not raw_backbone.strip():
        if strict_known:
            available = ", ".join(sorted(available_backbones))
            raise ValueError(f"'backbone' must be one of: {available}")
        raise ValueError("'backbone' must be a non-empty string.")

    candidate = raw_backbone.strip()
    available_set = set(available_backbones)
    if candidate in available_set:
        return candidate

    casefold_matches = [name for name in available_backbones if name.casefold() == candidate.casefold()]
    if len(casefold_matches) == 1:
        return casefold_matches[0]
    if len(casefold_matches) > 1:
        options = ", ".join(casefold_matches)
        raise ValueError(f"Backbone '{raw_backbone}' is ambiguous (case-insensitive matches: {options}).")

    if strict_known:
        available = ", ".join(sorted(available_backbones))
        raise ValueError(f"'backbone' must be one of: {available}")
    return candidate


def discover_run_arguments(repo_root: Path | None = None) -> set[str]:
    repo_root = repo_root or _repo_root()
    run_py = repo_root / "run.py"
    contents = run_py.read_text(encoding="utf-8")
    return {match.group("name") for match in RUN_ARG_PATTERN.finditer(contents)}


def discover_run_argument_defaults(repo_root: Path | None = None) -> dict[str, Any]:
    repo_root = repo_root or _repo_root()
    run_py = repo_root / "run.py"
    tree = ast.parse(run_py.read_text(encoding="utf-8"))

    defaults: dict[str, Any] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args:
            continue
        first_arg = node.args[0]
        if not isinstance(first_arg, ast.Constant) or not isinstance(first_arg.value, str):
            continue
        if not first_arg.value.startswith("--"):
            continue

        arg_name = first_arg.value[2:]
        arg_default: Any = None
        has_default = False
        action_value: str | None = None

        for keyword in node.keywords:
            if keyword.arg == "default":
                has_default = True
                try:
                    arg_default = ast.literal_eval(keyword.value)
                except Exception:
                    arg_default = None
            elif keyword.arg == "action" and isinstance(keyword.value, ast.Constant):
                action_value = keyword.value.value

        if not has_default:
            if action_value == "store_true":
                arg_default = False
                has_default = True
            elif action_value == "store_false":
                arg_default = True
                has_default = True

        if has_default:
            defaults[arg_name] = arg_default

    return defaults


def discover_backbone_hyperparameters(
    backbone: str,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root or _repo_root()
    model_file = repo_root / "models" / f"{backbone}.py"
    if not model_file.exists():
        raise ValueError(f"Backbone '{backbone}' was not found under {repo_root / 'models'}.")

    valid_run_args = discover_run_arguments(repo_root)
    contents = model_file.read_text(encoding="utf-8")
    discovered = sorted(
        {
            match.group("name")
            for match in CONFIG_ATTR_PATTERN.finditer(contents)
            if match.group("name") in valid_run_args
        }
    )
    custom_rules = sorted(BACKBONE_SPECS.get(backbone, BackboneSpec(backbone, {})).param_rules.keys())

    return {
        "backbone": backbone,
        "model_file": str(model_file),
        "discovered_model_hyperparameters": discovered,
        "custom_validated_parameters": custom_rules,
        "parameter_aliases": {
            alias: canonical
            for alias, canonical in PARAMETER_ALIASES.items()
            if canonical in discovered or canonical in custom_rules
        },
    }


def _normalize_param_name(raw_name: str) -> str:
    return PARAMETER_ALIASES.get(raw_name, raw_name)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if not isinstance(value, str):
        raise ValueError(f"Unsupported value type: {type(value).__name__}")

    stripped = value.strip()
    if stripped == "":
        raise ValueError("Empty strings are not valid search-space values.")

    if stripped.lower() in {"true", "false"}:
        return stripped.lower() == "true"
    if re.fullmatch(r"[-+]?\d+", stripped):
        return int(stripped)
    if re.fullmatch(r"[-+]?\d+\.\d+", stripped):
        return float(stripped)
    return stripped


def _coerce_config_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_coerce_config_value(item) for item in value]
    if isinstance(value, dict):
        raise ValueError("Nested dictionaries are not supported in config values.")
    return _coerce_scalar(value)


def _parse_cli_assignment(raw_assignment: str) -> tuple[str, str]:
    if "=" not in raw_assignment:
        raise ValueError(
            f"Invalid assignment '{raw_assignment}'. Use the form key=value."
        )
    raw_name, raw_value = raw_assignment.split("=", 1)
    name = raw_name.strip()
    value = raw_value.strip()
    if not name:
        raise ValueError(f"Invalid assignment '{raw_assignment}'. Parameter name is empty.")
    if value == "":
        raise ValueError(f"Invalid assignment '{raw_assignment}'. Parameter value is empty.")
    return name, value


def _parse_cli_fixed_value(raw_value: str) -> Any:
    stripped = raw_value.strip()
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            loaded = json.loads(stripped)
        except json.JSONDecodeError:
            return _coerce_scalar(stripped)
        return _coerce_config_value(loaded)
    return _coerce_scalar(stripped)


def _parse_cli_search_values(raw_value: str) -> list[Any]:
    stripped = raw_value.strip()
    if stripped.startswith("["):
        try:
            loaded = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON list for search values: {raw_value!r}") from exc
        if not isinstance(loaded, list) or not loaded:
            raise ValueError("Search values must be a non-empty list.")
        return loaded

    values = [item.strip() for item in stripped.split(",") if item.strip()]
    if not values:
        raise ValueError("Search values must contain at least one choice.")
    return [_coerce_scalar(item) for item in values]


def _legacy_search_config_path(backbone: str, repo_root: Path | None = None) -> Path:
    repo_root = repo_root or _repo_root()
    return repo_root / "search_config" / f"{_slugify(backbone)}_search_spec.json"


def _search_config_key(backbone: str, fixed_config: dict[str, Any]) -> str:
    task_name = fixed_config.get("task_name", "task")
    data = fixed_config.get("data", "data")
    return _slugify(f"{backbone}_{task_name}_{data}")


def _search_config_path(
    backbone: str,
    fixed_config: dict[str, Any],
    repo_root: Path | None = None,
) -> Path:
    repo_root = repo_root or _repo_root()
    return repo_root / "search_config" / f"{_search_config_key(backbone, fixed_config)}_search_spec.json"


def _resolve_search_config_path_from_name(
    config_name: str,
    repo_root: Path | None = None,
) -> Path:
    repo_root = repo_root or _repo_root()
    config_dir = repo_root / "search_config"

    # Prefer concrete paths first so users can pass any existing file without naming constraints.
    raw_input = Path(config_name).expanduser()
    direct_candidates = []
    if raw_input.is_absolute():
        direct_candidates.append(raw_input)
    else:
        direct_candidates.extend(
            [
                Path(config_name),
                repo_root / config_name,
                config_dir / config_name,
            ]
        )

    if raw_input.suffix:
        suffixed_candidates = direct_candidates
    else:
        if raw_input.is_absolute():
            suffixed_candidates = direct_candidates + [
                raw_input.with_suffix(".json"),
                raw_input.with_name(f"{raw_input.name}_search_spec.json"),
            ]
        else:
            suffixed_candidates = direct_candidates + [
                Path(f"{config_name}.json"),
                Path(f"{config_name}_search_spec.json"),
                repo_root / f"{config_name}.json",
                repo_root / f"{config_name}_search_spec.json",
                config_dir / f"{config_name}.json",
                config_dir / f"{config_name}_search_spec.json",
            ]

    for candidate in suffixed_candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    normalized = _slugify(Path(config_name).stem.replace("_search_spec", ""))
    exact_path = config_dir / f"{normalized}_search_spec.json"

    combo_matches = sorted(config_dir.glob(f"{normalized}_*_search_spec.json"))
    if exact_path.exists() and combo_matches:
        options = ", ".join([exact_path.stem] + [path.stem for path in combo_matches])
        raise ValueError(
            f"Multiple search config specs match '{config_name}': {options}. "
            "Use the full backbone_task_dataset key."
        )
    if exact_path.exists():
        return exact_path

    matches = combo_matches
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        options = ", ".join(path.stem for path in matches)
        raise ValueError(
            f"Multiple search config specs match '{config_name}': {options}. "
            "Use the full backbone_task_dataset key."
        )
    raise ValueError(f"Search config spec not found for '{config_name}'.")


def _default_candidates_output_path(
    spec_path: Path,
    repo_root: Path | None = None,
) -> Path:
    repo_root = repo_root or _repo_root()
    stem = spec_path.stem
    if stem.endswith("_search_spec"):
        stem = stem[: -len("_search_spec")]
    return repo_root / "candidates" / f"{stem}_candidates.json"


def _filename_in_dir(
    directory: Path,
    raw_name: str,
) -> Path:
    normalized_name = Path(raw_name).name.strip()
    if not normalized_name:
        raise ValueError("File name must not be empty.")
    if normalized_name in {".", ".."}:
        raise ValueError(f"Invalid file name: {raw_name!r}")
    if not normalized_name.endswith(".json"):
        normalized_name = f"{normalized_name}.json"
    return directory / normalized_name


def _resolve_candidates_path_from_name(
    candidate_name: str,
    repo_root: Path | None = None,
) -> Path:
    repo_root = repo_root or _repo_root()
    candidates_dir = repo_root / "candidates"
    normalized = _slugify(Path(candidate_name).stem.replace("_candidates", ""))
    candidate_path = candidates_dir / f"{normalized}_candidates.json"
    if not candidate_path.exists():
        raise ValueError(f"Candidate JSON not found for '{candidate_name}'.")
    return candidate_path


def _default_fixed_config(repo_root: Path | None = None) -> dict[str, Any]:
    defaults = discover_run_argument_defaults(repo_root)
    return {
        key: defaults[key]
        for key in DEFAULT_SPEC_FIXED_KEYS
        if key in defaults
    }


def _iter_example_recipe_runs(
    repo_root: Path | None = None,
):
    repo_root = repo_root or _repo_root()
    examples_root = repo_root / "examples"
    for recipe_path in sorted(examples_root.rglob("*.json")):
        payload = _load_json(recipe_path)
        if payload.get("script_type") != "run_py_recipe":
            continue
        for run in payload.get("runs", []):
            run_args = run.get("run_args")
            if isinstance(run_args, dict):
                yield recipe_path, payload, run


def _dataset_lookup_tokens(value: Any) -> set[str]:
    raw = str(value or "").strip()
    if not raw:
        return set()

    normalized = raw.replace("\\", "/").strip("/")
    candidates = {raw, normalized}
    try:
        path_value = Path(normalized)
        candidates.add(path_value.name)
        candidates.add(path_value.stem)
    except Exception:
        pass

    return {_slugify(candidate) for candidate in candidates if str(candidate).strip()}


def _recipe_dataset_aliases(
    recipe_path: Path,
    payload: dict[str, Any],
    run: dict[str, Any],
) -> set[str]:
    run_args = run["run_args"]
    aliases: set[str] = set()

    for key in ("data", "data_path", "root_path", "model_id"):
        aliases.update(_dataset_lookup_tokens(run_args.get(key)))

    summary = payload.get("summary")
    if isinstance(summary, dict):
        for dataset_name in summary.get("datasets", []):
            aliases.update(_dataset_lookup_tokens(dataset_name))

    aliases.update(_dataset_lookup_tokens(recipe_path.stem))
    aliases.update(_dataset_lookup_tokens(recipe_path.parent.name))

    parent_name = recipe_path.parent.name
    if parent_name.lower().endswith("_script"):
        aliases.update(_dataset_lookup_tokens(parent_name[: -len("_script")]))

    return aliases


def _recipe_matches_requested_data(
    recipe_path: Path,
    payload: dict[str, Any],
    run: dict[str, Any],
    requested_data: str,
) -> bool:
    run_args = run["run_args"]
    actual_data = run_args.get("data")
    if actual_data == requested_data:
        return True

    requested_slug = _slugify(requested_data)
    if not requested_slug:
        return False

    return requested_slug in _recipe_dataset_aliases(recipe_path, payload, run)


def _default_recipe_sort_key(
    recipe_path: Path,
    run: dict[str, Any],
    *,
    task_name: str,
    repo_root: Path | None = None,
) -> tuple[Any, ...]:
    repo_root = repo_root or _repo_root()
    rel_path = recipe_path.relative_to(repo_root / "examples")
    task_folder_rank = 0 if rel_path.parts and rel_path.parts[0] == task_name else 1
    model_name = str(run.get("run_args", {}).get("model", ""))
    dataset_name = str(run.get("run_args", {}).get("data", ""))
    stem = _slugify(recipe_path.stem)
    stem_model_rank = 0 if _slugify(model_name) in stem else 1
    stem_dataset_rank = 0 if _slugify(dataset_name) in stem else 1
    run_index = run.get("index", 0)
    return (task_folder_rank, stem_model_rank, stem_dataset_rank, str(rel_path), run_index)


def _find_default_recipe_run(
    backbone: str,
    task_name: str,
    data: str,
    repo_root: Path | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    repo_root = repo_root or _repo_root()
    matches: list[tuple[tuple[Any, ...], Path, dict[str, Any], dict[str, Any]]] = []
    for recipe_path, payload, run in _iter_example_recipe_runs(repo_root):
        run_args = run["run_args"]
        if (
            run_args.get("model") == backbone
            and run_args.get("task_name") == task_name
            and _recipe_matches_requested_data(recipe_path, payload, run, data)
        ):
            matches.append(
                (
                    _default_recipe_sort_key(recipe_path, run, task_name=task_name, repo_root=repo_root),
                    recipe_path,
                    payload,
                    run,
                )
            )

    if not matches:
        raise ValueError(
            f"No default recipe was found under examples/ for backbone='{backbone}', "
            f"task_name='{task_name}', data='{data}'."
        )

    _, recipe_path, payload, run = min(matches, key=lambda item: item[0])
    return recipe_path, payload, run


def _find_default_recipe_runs(
    backbone: str,
    task_name: str,
    data: str,
    repo_root: Path | None = None,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    repo_root = repo_root or _repo_root()
    recipe_path, payload, _ = _find_default_recipe_run(backbone, task_name, data, repo_root)
    runs = [
        run
        for run in payload.get("runs", [])
        if isinstance(run.get("run_args"), dict)
        and run["run_args"].get("model") == backbone
        and run["run_args"].get("task_name") == task_name
        and _recipe_matches_requested_data(recipe_path, payload, run, data)
    ]
    runs.sort(key=lambda run: run.get("index", 0))
    return recipe_path, payload, runs


def _example_based_fixed_config(
    backbone: str,
    task_name: str,
    data: str,
    repo_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    repo_root = repo_root or _repo_root()
    recipe_path, payload, run = _find_default_recipe_run(backbone, task_name, data, repo_root)
    run_args = dict(run["run_args"])
    fixed_config = {
        key: value
        for key, value in run_args.items()
        if key not in {"model", "model_id"}
    }
    recipe_reference = {
        "source_recipe": str(recipe_path.relative_to(repo_root)),
        "relative_script_path": payload.get("relative_script_path"),
        "source_run_index": run.get("index"),
        "source_command": run.get("command"),
        "requested_data": data,
        "resolved_data": fixed_config.get("data"),
    }
    return fixed_config, recipe_reference


def _default_spec_for_backbone(
    backbone: str,
    repo_root: Path | None = None,
    fixed_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repo_root = repo_root or _repo_root()
    fixed_config = dict(fixed_config or _default_fixed_config(repo_root))
    return {
        "backbone": backbone,
        "num_samples": 10,
        "seed": 2026,
        "candidate_prefix": _default_candidate_prefix(backbone, fixed_config),
        "fixed_config": fixed_config,
        "search_space": {},
    }


def _load_or_initialize_search_config(
    backbone: str,
    target_fixed_config: dict[str, Any],
    repo_root: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    repo_root = repo_root or _repo_root()
    spec_path = _search_config_path(backbone, target_fixed_config, repo_root)
    if spec_path.exists():
        spec = _load_json(spec_path)
    else:
        task_name = target_fixed_config.get("task_name")
        data = target_fixed_config.get("data")
        if not isinstance(task_name, str) or not task_name:
            raise ValueError("'task_name' is required to initialize a new search config.")
        if not isinstance(data, str) or not data:
            raise ValueError("'data' is required to initialize a new search config.")
        example_fixed_config, recipe_reference = _example_based_fixed_config(
            backbone,
            task_name,
            data,
            repo_root,
        )
        spec = _default_spec_for_backbone(backbone, repo_root, fixed_config=example_fixed_config)
        spec["candidate_prefix"] = _default_candidate_prefix(backbone, target_fixed_config)
        spec["default_recipe"] = recipe_reference

    spec["backbone"] = backbone
    spec.setdefault("num_samples", 10)
    spec.setdefault("seed", 2026)
    spec.setdefault("fixed_config", dict(target_fixed_config))
    spec.setdefault("search_space", {})
    spec.setdefault("candidate_prefix", _default_candidate_prefix(backbone, spec["fixed_config"]))
    return spec, spec_path


def _build_or_update_spec_from_cli_args(
    args: argparse.Namespace,
    repo_root: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    repo_root = repo_root or _repo_root()
    if not args.backbone:
        raise ValueError("--backbone is required when --spec is not used.")

    valid_run_args = discover_run_arguments(repo_root)
    cli_fixed_overrides: dict[str, Any] = {}
    for raw_assignment in args.fixed or []:
        key, raw_value = _parse_cli_assignment(raw_assignment)
        cli_fixed_overrides[_normalize_param_name(key)] = _parse_cli_fixed_value(raw_value)
    cli_search_overrides: dict[str, list[Any]] = {}
    for raw_assignment in args.search or []:
        key, raw_value = _parse_cli_assignment(raw_assignment)
        cli_search_overrides[_normalize_param_name(key)] = _parse_cli_search_values(raw_value)

    conflicting_cli_params = sorted(set(cli_fixed_overrides) & set(cli_search_overrides))
    if conflicting_cli_params:
        joined = ", ".join(conflicting_cli_params)
        raise ValueError(
            "Do not specify the same parameter in both --fixed and --search in a single command: "
            f"{joined}"
        )

    task_name = cli_fixed_overrides.get("task_name")
    data = cli_fixed_overrides.get("data")
    if not isinstance(task_name, str) or not task_name:
        raise ValueError(
            "Direct CLI mode requires '--fixed task_name=...'. "
            "search_config files are keyed by backbone/task_name/data."
        )
    if not isinstance(data, str) or not data:
        raise ValueError(
            "Direct CLI mode requires '--fixed data=...'. "
            "search_config files are keyed by backbone/task_name/data."
        )

    target_fixed_config = {
        "task_name": task_name,
        "data": data,
    }

    spec, _ = _load_or_initialize_search_config(args.backbone, target_fixed_config, repo_root)

    fixed_config = _normalize_fixed_config(spec.get("fixed_config", {}), valid_run_args)
    effective_cli_fixed_overrides = dict(cli_fixed_overrides)

    default_recipe = spec.get("default_recipe")
    if isinstance(default_recipe, dict):
        requested_data = default_recipe.get("requested_data")
        resolved_data = default_recipe.get("resolved_data")
        if (
            effective_cli_fixed_overrides.get("data") == requested_data
            and isinstance(resolved_data, str)
            and resolved_data
            and requested_data != resolved_data
            and fixed_config.get("data") == resolved_data
        ):
            effective_cli_fixed_overrides.pop("data", None)

    fixed_config.update(effective_cli_fixed_overrides)

    raw_existing_search_space = spec.get("search_space") or {}
    if raw_existing_search_space:
        search_space = _normalize_search_space(raw_existing_search_space, valid_run_args)
    else:
        search_space = {}

    for param_name in effective_cli_fixed_overrides:
        search_space.pop(param_name, None)
    search_space.update(cli_search_overrides)

    for param_name in search_space:
        fixed_config.pop(param_name, None)

    spec["fixed_config"] = fixed_config
    spec["search_space"] = search_space

    if args.num_samples is not None:
        spec["num_samples"] = args.num_samples
    if args.seed is not None:
        spec["seed"] = args.seed
    if args.candidate_prefix is not None:
        spec["candidate_prefix"] = args.candidate_prefix
    elif not spec.get("candidate_prefix"):
        spec["candidate_prefix"] = _default_candidate_prefix(args.backbone, fixed_config)
    if args.allow_replacement:
        spec["allow_replacement"] = True
    if "default_recipe" not in spec:
        _, recipe_reference = _example_based_fixed_config(args.backbone, task_name, data, repo_root)
        spec["default_recipe"] = recipe_reference

    spec_path_fixed_config = dict(fixed_config)
    if isinstance(default_recipe, dict):
        requested_data = default_recipe.get("requested_data")
        resolved_data = default_recipe.get("resolved_data")
        if (
            isinstance(requested_data, str)
            and requested_data
            and isinstance(resolved_data, str)
            and resolved_data
            and requested_data != resolved_data
        ):
            spec_path_fixed_config["data"] = requested_data

    spec_path = _search_config_path(args.backbone, spec_path_fixed_config, repo_root)
    return spec, spec_path


def _parse_multiplier(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    match = MULTIPLIER_PATTERN.fullmatch(value.strip())
    if not match:
        return None
    return float(match.group("factor"))


def _normalize_fixed_config(raw_config: Any, valid_run_args: set[str]) -> dict[str, Any]:
    if raw_config is None:
        return {}
    if not isinstance(raw_config, dict):
        raise ValueError("'fixed_config' must be a JSON object.")

    normalized: dict[str, Any] = {}
    for raw_name, raw_value in raw_config.items():
        name = _normalize_param_name(raw_name)
        if name not in valid_run_args and name != "model":
            raise ValueError(f"Unknown run.py argument in fixed_config: '{raw_name}'.")
        normalized[name] = _coerce_config_value(raw_value)
    return normalized


def _normalize_search_space(
    raw_search_space: Any,
    valid_run_args: set[str],
) -> dict[str, list[Any]]:
    if not isinstance(raw_search_space, dict) or not raw_search_space:
        raise ValueError("'search_space' must be a non-empty JSON object.")

    normalized: dict[str, list[Any]] = {}
    for raw_name, raw_choices in raw_search_space.items():
        name = _normalize_param_name(raw_name)
        if name in normalized:
            raise ValueError(
                f"Duplicate parameter after alias normalization: '{raw_name}' -> '{name}'."
            )
        if name not in valid_run_args:
            raise ValueError(f"Unknown run.py argument in search_space: '{raw_name}'.")
        if not isinstance(raw_choices, list) or not raw_choices:
            raise ValueError(f"'{raw_name}' must map to a non-empty list of choices.")
        normalized[name] = raw_choices
    return normalized


def _dependency_order(
    search_space: dict[str, list[Any]],
    spec: BackboneSpec | None,
    fixed_config: dict[str, Any],
) -> list[str]:
    remaining = list(search_space.keys())
    ordered: list[str] = []

    while remaining:
        progressed = False
        for param_name in list(remaining):
            rule = spec.param_rules.get(param_name) if spec else None
            needs_dependency = False
            if rule and rule.relative_to and rule.allow_multiplier:
                needs_dependency = any(_parse_multiplier(choice) is not None for choice in search_space[param_name])
            if not needs_dependency:
                ordered.append(param_name)
                remaining.remove(param_name)
                progressed = True
                continue

            dependency = rule.relative_to
            if dependency in ordered or dependency in fixed_config:
                ordered.append(param_name)
                remaining.remove(param_name)
                progressed = True

        if not progressed:
            unresolved = ", ".join(remaining)
            raise ValueError(f"Could not resolve parameter dependencies for: {unresolved}.")

    return ordered


def _resolve_choice(
    param_name: str,
    raw_choice: Any,
    partial_config: dict[str, Any],
    fixed_config: dict[str, Any],
    rule: ParamRule | None,
) -> Any:
    multiplier = _parse_multiplier(raw_choice)
    if multiplier is not None:
        if not rule or not rule.allow_multiplier or not rule.relative_to:
            raise ValueError(
                f"'{param_name}' received multiplier syntax {raw_choice!r}, but that parameter does not "
                "support relative values."
            )
        if rule.relative_to in partial_config:
            base_value = partial_config[rule.relative_to]
        elif rule.relative_to in fixed_config:
            base_value = fixed_config[rule.relative_to]
        else:
            raise ValueError(
                f"'{param_name}' uses relative value {raw_choice!r}, but its base parameter "
                f"'{rule.relative_to}' is missing."
            )
        if not isinstance(base_value, (int, float)) or isinstance(base_value, bool):
            raise ValueError(
                f"'{param_name}' depends on numeric base parameter '{rule.relative_to}', got {base_value!r}."
            )
        resolved = base_value * multiplier
        if abs(resolved - round(resolved)) < 1e-9:
            return int(round(resolved))
        return resolved

    if isinstance(raw_choice, list):
        return [_coerce_config_value(item) for item in raw_choice]

    return _coerce_scalar(raw_choice)


def _validate_search_choices(
    search_space: dict[str, list[Any]],
    spec: BackboneSpec | None,
    fixed_config: dict[str, Any],
) -> None:
    for param_name, raw_choices in search_space.items():
        rule = spec.param_rules.get(param_name) if spec else None
        for raw_choice in raw_choices:
            multiplier = _parse_multiplier(raw_choice)
            if (
                multiplier is not None
                and rule
                and rule.allow_multiplier
                and rule.relative_to
                and rule.relative_to in search_space
            ):
                continue
            resolved = _resolve_choice(param_name, raw_choice, {}, fixed_config, rule)
            if rule and isinstance(resolved, list):
                raise ValueError(f"'{param_name}' expects scalar values, but received a list choice.")
            merged_config = dict(fixed_config)
            merged_config[param_name] = resolved
            if rule:
                rule.validator(param_name, resolved, merged_config)


def _build_all_candidates(
    search_space: dict[str, list[Any]],
    spec: BackboneSpec | None,
    fixed_config: dict[str, Any],
    max_enumeration: int = 200_000,
) -> list[dict[str, Any]]:
    rough_size = 1
    for raw_choices in search_space.values():
        rough_size *= len(raw_choices)
    if rough_size > max_enumeration:
        raise ValueError(
            f"Search space is too large to enumerate safely ({rough_size} combinations). "
            f"Please reduce it below {max_enumeration} combinations or extend the sampler."
        )

    ordered_params = _dependency_order(search_space, spec, fixed_config)
    all_candidates: list[dict[str, Any]] = []

    def backtrack(index: int, partial_config: dict[str, Any]) -> None:
        if index == len(ordered_params):
            all_candidates.append(dict(partial_config))
            return

        param_name = ordered_params[index]
        rule = spec.param_rules.get(param_name) if spec else None
        for raw_choice in search_space[param_name]:
            resolved = _resolve_choice(param_name, raw_choice, partial_config, fixed_config, rule)
            if rule and isinstance(resolved, list):
                raise ValueError(f"'{param_name}' expects scalar values, but received a list choice.")
            merged = dict(fixed_config)
            merged.update(partial_config)
            merged[param_name] = resolved
            if rule:
                rule.validator(param_name, resolved, merged)
            partial_config[param_name] = resolved
            backtrack(index + 1, partial_config)
            partial_config.pop(param_name, None)

    backtrack(0, {})
    return all_candidates


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return slug or "candidate"


def _stringify_token_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        normalized = f"{value:.6g}".replace(".", "p")
        return normalized.replace("-", "m")
    if isinstance(value, list):
        return "x".join(_stringify_token_value(item) for item in value)
    return _slugify(str(value))


def _parameter_token(param_name: str, value: Any) -> str:
    token_name = NAME_TOKEN_ALIASES.get(param_name, _slugify(param_name))
    token_value = _stringify_token_value(value)
    return f"{token_name}{token_value}"


def _candidate_name(
    candidate_prefix: str,
    hyperparameters: dict[str, Any],
    search_param_order: list[str],
    index: int,
) -> str:
    tokens = [
        _parameter_token(param_name, hyperparameters[param_name])
        for param_name in search_param_order
        if param_name in hyperparameters
    ]
    base_name = candidate_prefix if not tokens else f"{candidate_prefix}_{'_'.join(tokens)}"
    return f"{base_name}_{index:04d}"


def _default_candidate_prefix(backbone: str, fixed_config: dict[str, Any]) -> str:
    parts = [backbone]
    for key in ("task_name", "data"):
        raw_value = fixed_config.get(key)
        if raw_value:
            parts.append(str(raw_value))
    return _slugify("_".join(parts))


def _build_candidate_record(
    candidate_name: str,
    backbone: str,
    hyperparameters: dict[str, Any],
    fixed_config: dict[str, Any],
) -> dict[str, Any]:
    run_args = dict(fixed_config)
    run_args["model"] = backbone
    if not run_args.get("model_id"):
        run_args["model_id"] = candidate_name
    run_args.update(hyperparameters)
    return {
        "candidate_id": candidate_name,
        "candidate_name": candidate_name,
        "model": backbone,
        "hyperparameters": hyperparameters,
        "run_args": run_args,
    }


def sample_candidates_from_spec(
    spec: dict[str, Any],
    *,
    repo_root: Path | None = None,
    output_path: str | Path | None = None,
    num_samples_override: int | None = None,
    seed_override: int | None = None,
    candidate_prefix_override: str | None = None,
    allow_replacement_override: bool | None = None,
    strict_backbone_validation: bool = True,
) -> dict[str, Any]:
    repo_root = repo_root or _repo_root()
    available_backbones = discover_available_backbones(repo_root)
    valid_run_args = discover_run_arguments(repo_root)

    backbone = _resolve_backbone_name(
        spec.get("backbone"),
        available_backbones,
        strict_known=strict_backbone_validation,
    )

    fixed_config = _normalize_fixed_config(spec.get("fixed_config"), valid_run_args)
    spec_model_name = fixed_config.get("model")
    fixed_config.pop("model", None)
    search_space = _normalize_search_space(spec.get("search_space"), valid_run_args)
    try:
        backbone_hparam_info = discover_backbone_hyperparameters(backbone, repo_root)
    except ValueError:
        backbone_hparam_info = {
            "backbone": backbone,
            "model_file": "",
            "discovered_model_hyperparameters": [],
            "custom_validated_parameters": [],
            "parameter_aliases": {},
            "warning": (
                f"Backbone '{backbone}' was not found under models/. "
                "Candidates were generated without model-file hyperparameter discovery."
            ),
        }

    if spec_model_name and spec_model_name != backbone:
        raise ValueError(
            f"'fixed_config.model' conflicts with backbone '{backbone}'. Remove it or make them match."
        )

    spec_for_backbone = BACKBONE_SPECS.get(backbone)
    _validate_search_choices(search_space, spec_for_backbone, fixed_config)
    all_candidates = _build_all_candidates(search_space, spec_for_backbone, fixed_config)

    if not all_candidates:
        raise ValueError("No valid candidates were generated from the provided search space.")

    num_samples = num_samples_override if num_samples_override is not None else spec.get("num_samples")
    if not isinstance(num_samples, int) or num_samples < 1:
        raise ValueError("'num_samples' must be a positive integer.")

    allow_replacement = (
        allow_replacement_override
        if allow_replacement_override is not None
        else bool(spec.get("allow_replacement", False))
    )

    if not allow_replacement and num_samples > len(all_candidates):
        raise ValueError(
            f"Requested {num_samples} unique samples, but only {len(all_candidates)} valid combinations exist. "
            "Reduce 'num_samples' or set 'allow_replacement' to true."
        )

    seed = seed_override if seed_override is not None else spec.get("seed", 2026)
    if not isinstance(seed, int):
        raise ValueError("'seed' must be an integer.")
    rng = random.Random(seed)

    if allow_replacement:
        sampled_candidates = [dict(rng.choice(all_candidates)) for _ in range(num_samples)]
    else:
        sampled_candidates = [dict(candidate) for candidate in rng.sample(all_candidates, num_samples)]

    candidate_prefix = candidate_prefix_override or spec.get("candidate_prefix")
    if candidate_prefix is None:
        candidate_prefix = _default_candidate_prefix(backbone, fixed_config)
    candidate_prefix = _slugify(str(candidate_prefix))
    search_param_order = list(search_space.keys())

    requested_uea_subset_names = spec.get("uea_subset_names")
    if requested_uea_subset_names is not None:
        if not isinstance(requested_uea_subset_names, list):
            raise ValueError("'uea_subset_names' must be a list of subset names if provided.")
        requested_uea_subset_names = [
            str(subset_name).strip()
            for subset_name in requested_uea_subset_names
            if str(subset_name).strip()
        ]
        if not requested_uea_subset_names:
            raise ValueError("'uea_subset_names' must contain at least one non-empty subset name.")

    candidate_records = []
    for index, hyperparameters in enumerate(sampled_candidates, start=1):
        candidate_name = _candidate_name(candidate_prefix, hyperparameters, search_param_order, index)
        candidate_records.append(
            _build_candidate_record(candidate_name, backbone, hyperparameters, fixed_config)
        )

    payload = {
        "metadata": {
            "backbone": backbone,
            "seed": seed,
            "num_requested": num_samples,
            "num_generated": len(candidate_records),
            "allow_replacement": allow_replacement,
            "candidate_prefix": candidate_prefix,
            "total_valid_combinations": len(all_candidates),
            "fixed_config": fixed_config,
            "search_space": search_space,
            "backbone_hyperparameters": backbone_hparam_info,
        },
        "candidates": candidate_records,
    }
    if requested_uea_subset_names is not None:
        payload["metadata"]["uea_subset_names"] = requested_uea_subset_names

    if output_path is not None:
        _write_json(Path(output_path), payload)

    return payload


def _load_candidate_payload(candidate_path: Path) -> dict[str, Any]:
    payload = _load_json(candidate_path)
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"Candidate JSON must contain a non-empty 'candidates' list: {candidate_path}")
    return payload


def _requested_uea_subset_names(payload: dict[str, Any] | None) -> list[str] | None:
    if not isinstance(payload, dict):
        return None

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None

    raw_subset_names = metadata.get("uea_subset_names")
    if raw_subset_names is None:
        return None
    if not isinstance(raw_subset_names, list):
        raise ValueError("'metadata.uea_subset_names' must be a list of subset names if provided.")

    normalized_subset_names: list[str] = []
    seen_subset_slugs: set[str] = set()
    for raw_name in raw_subset_names:
        subset_name = str(raw_name).strip()
        if not subset_name:
            continue
        subset_slug = _slugify(subset_name)
        if subset_slug in seen_subset_slugs:
            continue
        seen_subset_slugs.add(subset_slug)
        normalized_subset_names.append(subset_name)

    if not normalized_subset_names:
        raise ValueError("'metadata.uea_subset_names' must contain at least one non-empty subset name.")
    return normalized_subset_names


def _parse_gpu_ids(raw_gpu_ids: list[str] | None) -> list[int]:
    if raw_gpu_ids is None:
        return []

    parsed_gpu_ids: list[int] = []
    seen_gpu_ids: set[int] = set()
    for raw_value in raw_gpu_ids:
        for token in str(raw_value).split(","):
            token = token.strip()
            if not token:
                continue
            try:
                gpu_id = int(token)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid GPU id '{token}'. Use non-negative integers like '0', '0 1 2 3', or '0,1,2,3'."
                ) from exc
            if gpu_id < 0:
                raise ValueError(f"GPU ids must be non-negative integers, got {gpu_id}.")
            if gpu_id in seen_gpu_ids:
                continue
            seen_gpu_ids.add(gpu_id)
            parsed_gpu_ids.append(gpu_id)

    if not parsed_gpu_ids:
        raise ValueError("Provide at least one GPU id after --gpu-id.")

    return parsed_gpu_ids


def _parse_candidate_indices(
    raw_candidate_indices: list[str] | None,
    *,
    total_candidates: int,
) -> list[int]:
    if raw_candidate_indices is None:
        return []

    parsed_indices: list[int] = []
    seen_indices: set[int] = set()
    for raw_value in raw_candidate_indices:
        for token in str(raw_value).split(","):
            token = token.strip()
            if not token:
                continue
            try:
                candidate_index = int(token)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid candidate index '{token}'. "
                    "Use 1-based integers like '97', '1 5 97', or '1,5,97'."
                ) from exc
            if candidate_index < 1:
                raise ValueError(
                    f"Candidate indices are 1-based and must be >= 1, got {candidate_index}."
                )
            if candidate_index > total_candidates:
                raise ValueError(
                    f"Candidate index {candidate_index} is out of range for this file "
                    f"(total candidates: {total_candidates})."
                )
            if candidate_index in seen_indices:
                continue
            seen_indices.add(candidate_index)
            parsed_indices.append(candidate_index)

    if not parsed_indices:
        raise ValueError("Provide at least one candidate index after --run-candidates-specify-id.")

    return parsed_indices


def _build_gpu_worker_specs(gpu_ids: list[int], workers_per_gpu: int) -> list[tuple[int, int]]:
    return [
        (gpu_id, worker_index)
        for gpu_id in gpu_ids
        for worker_index in range(1, workers_per_gpu + 1)
    ]


def _format_gpu_worker_pool(gpu_ids: list[int], workers_per_gpu: int) -> str:
    if workers_per_gpu == 1:
        return ", ".join(f"cuda:{gpu_id}" for gpu_id in gpu_ids)
    return ", ".join(f"cuda:{gpu_id}[workers:{workers_per_gpu}]" for gpu_id in gpu_ids)


def _prepare_candidate_run_args(
    candidate: dict[str, Any],
    *,
    gpu_id: int | None = None,
) -> dict[str, Any]:
    run_args = dict(candidate.get("run_args", {}))
    if not isinstance(run_args, dict) or not run_args:
        raise ValueError(
            f"Candidate '{candidate.get('candidate_name', candidate.get('candidate_id', 'unknown'))}' "
            "is missing a valid 'run_args' object."
        )

    if "model" not in run_args:
        model_name = candidate.get("model")
        if not model_name:
            raise ValueError(
                f"Candidate '{candidate.get('candidate_name', candidate.get('candidate_id', 'unknown'))}' "
                "is missing both 'run_args.model' and top-level 'model'."
            )
        run_args["model"] = model_name

    candidate_name = (
        candidate.get("candidate_name")
        or candidate.get("candidate_id")
        or run_args.get("model_id")
    )
    if not candidate_name:
        raise ValueError("Each candidate must define candidate_name/candidate_id or run_args.model_id.")

    if "model_id" not in run_args:
        run_args["model_id"] = candidate_name

    original_results_id = run_args.get("results_id")
    normalized_results_id = str(original_results_id).strip() if original_results_id is not None else ""
    if normalized_results_id:
        run_args["results_id"] = normalized_results_id
    else:
        run_args["results_id"] = candidate_name

    # Keep model_id stable for tasks like classification where it doubles as
    # the dataset identifier, and isolate checkpoint/result namespaces via des.
    original_des = run_args.get("des")
    normalized_des = str(original_des).strip() if original_des is not None else ""
    if normalized_des:
        if normalized_des == candidate_name or normalized_des.endswith(f"__{candidate_name}"):
            run_args["des"] = normalized_des
        else:
            run_args["des"] = f"{normalized_des}__{candidate_name}"
    else:
        run_args["des"] = candidate_name

    if gpu_id is not None:
        run_args["use_gpu"] = True
        run_args["gpu"] = 0
        run_args["use_multi_gpu"] = False

    return run_args


def _candidate_display_name(candidate: dict[str, Any], index: int) -> str:
    return candidate.get("candidate_name") or candidate.get("candidate_id") or f"candidate_{index:04d}"


def _build_candidate_run_plans(payload: dict[str, Any]) -> list[CandidateRunPlan]:
    candidates = payload["candidates"]
    total = len(candidates)
    plans: list[CandidateRunPlan] = []

    for index, candidate in enumerate(candidates, start=1):
        # Validate the candidate payload early before any long-running execution starts.
        _prepare_candidate_run_args(candidate, gpu_id=None)
        plans.append(
            CandidateRunPlan(
                index=index,
                total=total,
                candidate_name=_candidate_display_name(candidate, index),
                candidate=candidate,
            )
        )

    return plans


def _build_run_command(
    run_args: dict[str, Any],
    *,
    repo_root: Path | None = None,
    python_executable: str | None = None,
) -> list[str]:
    repo_root = repo_root or _repo_root()
    python_executable = python_executable or sys.executable
    run_py = repo_root / "run.py"
    command = [python_executable, "-u", str(run_py)]

    for key, value in run_args.items():
        if value is None:
            continue
        if key == "no_use_gpu":
            if value:
                command.append("--no_use_gpu")
            continue
        if key in STORE_FALSE_RUN_ARGS:
            if value is False:
                command.append(STORE_FALSE_RUN_ARGS[key])
            continue
        if key in STORE_TRUE_RUN_ARGS:
            if value:
                command.append(f"--{key}")
            continue
        if isinstance(value, bool):
            command.extend([f"--{key}", str(value)])
            continue
        if isinstance(value, list):
            command.append(f"--{key}")
            command.extend(str(item) for item in value)
            continue
        command.extend([f"--{key}", str(value)])

    return command


def _format_command(command: list[str]) -> str:
    return shlex.join(command)


def _candidate_sort_key(candidate_id: str) -> tuple[int, int, str]:
    match = CANDIDATE_SUFFIX_PATTERN.search(candidate_id)
    if match is not None:
        return (0, int(match.group(1)), candidate_id)
    return (1, 0, candidate_id)


def _uea_accuracy_column(dataset_name: str) -> str:
    return f"{dataset_name}_accuracy"


def _summary_csv_path(candidate_path: Path, repo_root: Path | None = None) -> Path:
    repo_root = repo_root or _repo_root()
    return repo_root / "results" / f"{candidate_path.stem}_uea_average_accuracy.csv"


def _load_existing_summary_rows(summary_path: Path) -> list[dict[str, Any]]:
    if not summary_path.exists():
        return []
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _write_summary_rows(summary_path: Path, rows: list[dict[str, Any]], dataset_names: list[str]) -> None:
    fieldnames = UEA_AVERAGE_SUMMARY_FIELDS + [_uea_accuracy_column(dataset_name) for dataset_name in dataset_names]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            normalized = {key: row.get(key, "") for key in fieldnames}
            writer.writerow(normalized)


def _parse_candidate_model_num(candidate_id: str) -> str:
    match = CANDIDATE_SUFFIX_PATTERN.search(candidate_id)
    if match is None:
        return ""
    return str(int(match.group(1)))


def _build_uea_average_row(
    candidate: dict[str, Any],
    *,
    dataset_names: list[str],
    subset_accuracies: dict[str, float],
    total_subsets: int,
    status: str,
    error: str = "",
) -> dict[str, Any]:
    run_args = dict(candidate.get("run_args", {}))
    hyperparameters = dict(candidate.get("hyperparameters", {}))
    candidate_id = str(candidate.get("candidate_id", candidate.get("candidate_name", run_args.get("model_id", ""))))
    candidate_name = str(candidate.get("candidate_name", candidate_id))
    accuracy_values = list(subset_accuracies.values())

    row: dict[str, Any] = {
        "candidate_id": candidate_id,
        "candidate_name": candidate_name,
        "model": candidate.get("model", run_args.get("model", "")),
        "task_name": run_args.get("task_name", ""),
        "data": run_args.get("data", ""),
        "model_num": _parse_candidate_model_num(candidate_id),
        "e_layers": hyperparameters.get("e_layers", run_args.get("e_layers", "")),
        "d_model": hyperparameters.get("d_model", run_args.get("d_model", "")),
        "d_ff": hyperparameters.get("d_ff", run_args.get("d_ff", "")),
        "top_k": hyperparameters.get("top_k", run_args.get("top_k", "")),
        "num_kernels": hyperparameters.get("num_kernels", run_args.get("num_kernels", "")),
        "num_subsets_total": total_subsets,
        "num_subsets_completed": len(subset_accuracies),
        "average_accuracy": sum(accuracy_values) / len(accuracy_values) if accuracy_values else "",
        "status": status,
        "error": error,
    }
    for dataset_name in dataset_names:
        row[_uea_accuracy_column(dataset_name)] = subset_accuracies.get(dataset_name, "")
    return row


def _candidate_recipe_runs(
    candidate: dict[str, Any],
    *,
    requested_subset_names: list[str] | None = None,
    repo_root: Path | None = None,
) -> tuple[Path, list[dict[str, Any]]] | None:
    repo_root = repo_root or _repo_root()
    run_args = dict(candidate.get("run_args", {}))
    backbone = run_args.get("model") or candidate.get("model")
    task_name = run_args.get("task_name")
    data = run_args.get("data")

    if not isinstance(backbone, str) or task_name != "classification" or data != "UEA":
        return None

    recipe_path, _, runs = _find_default_recipe_runs(backbone, task_name, data, repo_root)
    if requested_subset_names:
        run_by_subset_slug: dict[str, dict[str, Any]] = {}
        for run in runs:
            subset_name = str(run.get("run_args", {}).get("model_id", "")).strip()
            if not subset_name:
                continue
            run_by_subset_slug.setdefault(_slugify(subset_name), run)

        missing_subset_names = [
            subset_name
            for subset_name in requested_subset_names
            if _slugify(subset_name) not in run_by_subset_slug
        ]
        if missing_subset_names:
            raise ValueError(
                "Requested UEA subset(s) were not found in the default recipe "
                f"{recipe_path.relative_to(repo_root)}: {', '.join(missing_subset_names)}"
            )

        runs = [run_by_subset_slug[_slugify(subset_name)] for subset_name in requested_subset_names]

    if len(runs) <= 1:
        return None
    return recipe_path, runs


def _recipe_adjusted_run_args(
    candidate: dict[str, Any],
    recipe_run: dict[str, Any],
    *,
    gpu_id: int | None = None,
) -> dict[str, Any]:
    run_args = _prepare_candidate_run_args(candidate, gpu_id=gpu_id)
    recipe_run_args = dict(recipe_run.get("run_args", {}))
    candidate_hparam_keys = {
        str(key)
        for key in dict(candidate.get("hyperparameters", {})).keys()
    }
    protected_keys = candidate_hparam_keys | {"model", "results_id", "des", "is_training"}

    for key, value in recipe_run_args.items():
        if key in protected_keys:
            continue
        run_args[key] = value

    for key in UEA_RECIPE_OVERRIDE_KEYS:
        if key in recipe_run_args:
            run_args[key] = recipe_run_args[key]
    return run_args


def _print_runner_message(message: str, *, print_lock: threading.Lock | None = None) -> None:
    if print_lock is None:
        print(message, flush=True)
        return

    with print_lock:
        print(message, flush=True)


def _stream_subprocess(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    line_prefix: str | None = None,
    print_lock: threading.Lock | None = None,
    line_callback: Callable[[str], None] | None = None,
) -> int:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    try:
        if process.stdout is not None:
            for line in process.stdout:
                text = line.rstrip("\n")
                if line_callback is not None:
                    line_callback(text)
                if line_prefix:
                    rendered = f"{line_prefix} {text}" if text else line_prefix
                else:
                    rendered = text
                _print_runner_message(rendered, print_lock=print_lock)
        return process.wait()
    finally:
        if process.stdout is not None:
            process.stdout.close()


def _execute_run_args(
    run_args: dict[str, Any],
    *,
    repo_root: Path,
    gpu_id: int | None,
    python_executable: str | None,
    dry_run: bool,
    display_name: str,
    header_prefix: str,
    print_lock: threading.Lock | None = None,
    stream_prefix: str | None = None,
) -> tuple[int, dict[str, Any]]:
    command = _build_run_command(run_args, repo_root=repo_root, python_executable=python_executable)

    _print_runner_message(f"{header_prefix} Running {display_name}", print_lock=print_lock)
    if gpu_id is not None:
        _print_runner_message(
            f"{header_prefix} GPU: physical cuda:{gpu_id} (process-local cuda:0)",
            print_lock=print_lock,
        )
    _print_runner_message(f"{header_prefix} Command: {_format_command(command)}", print_lock=print_lock)

    observed: dict[str, Any] = {
        "training_setting": None,
        "testing_setting": None,
        "accuracy": None,
    }

    def capture_line(text: str) -> None:
        training_match = TRAINING_SETTING_PATTERN.match(text)
        if training_match is not None:
            observed["training_setting"] = training_match.group("setting")

        testing_match = TESTING_SETTING_PATTERN.match(text)
        if testing_match is not None:
            observed["testing_setting"] = testing_match.group("setting")

        accuracy_match = ACCURACY_LINE_PATTERN.match(text)
        if accuracy_match is not None:
            observed["accuracy"] = float(accuracy_match.group("accuracy"))

    if dry_run:
        return 0, observed

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        return_code = _stream_subprocess(
            command,
            cwd=repo_root,
            env=env,
            line_prefix=stream_prefix,
            print_lock=print_lock,
            line_callback=capture_line,
        )
    except OSError as exc:
        _print_runner_message(
            f"{header_prefix} Status: failed to launch ({exc})",
            print_lock=print_lock,
        )
        observed["error"] = str(exc)
        return 1, observed

    if return_code == 0:
        _print_runner_message(f"{header_prefix} Status: success", print_lock=print_lock)
        return 0, observed

    _print_runner_message(
        f"{header_prefix} Status: failed (exit code {return_code})",
        print_lock=print_lock,
    )
    return return_code, observed


def _execute_candidate_plan(
    plan: CandidateRunPlan,
    *,
    repo_root: Path,
    gpu_id: int | None,
    python_executable: str | None,
    dry_run: bool,
    requested_uea_subset_names: list[str] | None = None,
    print_lock: threading.Lock | None = None,
    stream_prefix: str | None = None,
    summary_callback: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    header_prefix = f"[{plan.index}/{plan.total}]"
    recipe_config = _candidate_recipe_runs(
        plan.candidate,
        requested_subset_names=requested_uea_subset_names,
        repo_root=repo_root,
    )
    if recipe_config is None:
        run_args = _prepare_candidate_run_args(plan.candidate, gpu_id=gpu_id)
        return_code, _ = _execute_run_args(
            run_args,
            repo_root=repo_root,
            gpu_id=gpu_id,
            python_executable=python_executable,
            dry_run=dry_run,
            display_name=plan.candidate_name,
            header_prefix=header_prefix,
            print_lock=print_lock,
            stream_prefix=stream_prefix,
        )
        return return_code

    recipe_path, recipe_runs = recipe_config
    dataset_names = [str(run.get("run_args", {}).get("model_id", f"subset_{index + 1}")) for index, run in enumerate(recipe_runs)]
    _print_runner_message(
        f"{header_prefix} UEA classification sweep: {len(recipe_runs)} subsets from {recipe_path.relative_to(repo_root)}",
        print_lock=print_lock,
    )

    subset_accuracies: dict[str, float] = {}
    total_subsets = len(recipe_runs)
    for subset_index, recipe_run in enumerate(recipe_runs, start=1):
        dataset_name = str(recipe_run.get("run_args", {}).get("model_id", f"subset_{subset_index}"))
        subset_header_prefix = f"{header_prefix}[subset:{subset_index}/{total_subsets}]"
        subset_stream_prefix = stream_prefix
        if stream_prefix:
            subset_stream_prefix = f"{stream_prefix}[subset:{dataset_name}]"
        run_args = _recipe_adjusted_run_args(plan.candidate, recipe_run, gpu_id=gpu_id)
        return_code, observed = _execute_run_args(
            run_args,
            repo_root=repo_root,
            gpu_id=gpu_id,
            python_executable=python_executable,
            dry_run=dry_run,
            display_name=f"{plan.candidate_name} [{dataset_name}]",
            header_prefix=subset_header_prefix,
            print_lock=print_lock,
            stream_prefix=subset_stream_prefix,
        )
        if return_code != 0:
            if not dry_run and summary_callback is not None:
                summary_callback(
                    _build_uea_average_row(
                        plan.candidate,
                        dataset_names=dataset_names,
                        subset_accuracies=subset_accuracies,
                        total_subsets=total_subsets,
                        status="failed",
                        error=f"subset '{dataset_name}' failed with exit code {return_code}",
                    )
                )
            return return_code

        accuracy = observed.get("accuracy")
        if dry_run:
            continue
        if accuracy is None:
            if summary_callback is not None:
                summary_callback(
                    _build_uea_average_row(
                        plan.candidate,
                        dataset_names=dataset_names,
                        subset_accuracies=subset_accuracies,
                        total_subsets=total_subsets,
                        status="failed",
                        error=f"subset '{dataset_name}' completed without an 'accuracy:' line in stdout",
                    )
                )
            return 1
        subset_accuracies[dataset_name] = accuracy

    if not dry_run and summary_callback is not None:
        summary_callback(
            _build_uea_average_row(
                plan.candidate,
                dataset_names=dataset_names,
                subset_accuracies=subset_accuracies,
                total_subsets=total_subsets,
                status="success",
            )
        )

    if not dry_run:
        average_accuracy = sum(subset_accuracies.values()) / len(subset_accuracies) if subset_accuracies else float("nan")
        _print_runner_message(
            f"{header_prefix} Average accuracy across {len(subset_accuracies)}/{total_subsets} UEA subsets: {average_accuracy:.6f}",
            print_lock=print_lock,
        )
    return 0


def run_candidates_from_payload(
    payload: dict[str, Any],
    *,
    candidate_path: Path,
    repo_root: Path | None = None,
    gpu_ids: list[int] | None = None,
    workers_per_gpu: int = 1,
    python_executable: str | None = None,
    dry_run: bool = False,
    continue_on_error: bool = False,
) -> int:
    repo_root = repo_root or _repo_root()
    plans = _build_candidate_run_plans(payload)
    total = len(plans)
    failures: list[tuple[int, str, int]] = []
    normalized_gpu_ids = list(gpu_ids or [])
    worker_specs = _build_gpu_worker_specs(normalized_gpu_ids, workers_per_gpu) if normalized_gpu_ids else []
    worker_count = len(worker_specs)
    summary_callback: Callable[[dict[str, Any]], None] | None = None
    requested_uea_subset_names = _requested_uea_subset_names(payload)

    if plans:
        recipe_config = _candidate_recipe_runs(
            plans[0].candidate,
            requested_subset_names=requested_uea_subset_names,
            repo_root=repo_root,
        )
        if recipe_config is not None:
            recipe_path, recipe_runs = recipe_config
            dataset_names = [
                str(run.get("run_args", {}).get("model_id", f"subset_{index + 1}"))
                for index, run in enumerate(recipe_runs)
            ]
            summary_path = _summary_csv_path(candidate_path, repo_root)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_rows = {
                str(row.get("candidate_id", "")): dict(row)
                for row in _load_existing_summary_rows(summary_path)
                if row.get("candidate_id")
            }
            summary_lock = threading.Lock()

            def write_summary_row(row: dict[str, Any]) -> None:
                with summary_lock:
                    summary_rows[str(row["candidate_id"])] = dict(row)
                    ordered_rows = sorted(
                        summary_rows.values(),
                        key=lambda row_item: _candidate_sort_key(str(row_item.get("candidate_id", ""))),
                    )
                    _write_summary_rows(summary_path, ordered_rows, dataset_names)

            summary_callback = write_summary_row
            print(
                f"UEA classification average-accuracy summary will be written to {summary_path} "
                f"using {len(dataset_names)} subsets from {recipe_path.relative_to(repo_root)}"
            )

    if dry_run and worker_count > 1:
        gpu_pool = _format_gpu_worker_pool(normalized_gpu_ids, workers_per_gpu)
        print(f"Dry run preview across {worker_count} GPU workers: {gpu_pool}")
        print("Preview assignment uses round-robin order; real execution dispatches to the first available worker.")

    if worker_count <= 1 or dry_run:
        for plan in plans:
            assigned_gpu_id = None
            if worker_specs:
                if dry_run and worker_count > 1:
                    assigned_gpu_id = worker_specs[(plan.index - 1) % worker_count][0]
                else:
                    assigned_gpu_id = worker_specs[0][0]

            return_code = _execute_candidate_plan(
                plan,
                repo_root=repo_root,
                gpu_id=assigned_gpu_id,
                python_executable=python_executable,
                dry_run=dry_run,
                requested_uea_subset_names=requested_uea_subset_names,
                summary_callback=summary_callback,
            )
            if return_code == 0:
                continue

            failures.append((plan.index, plan.candidate_name, return_code))
            if not continue_on_error:
                print(
                    f"Stopped after the first failure while running '{plan.candidate_name}'. "
                    f"Source candidate file: {candidate_path}"
                )
                return return_code
    else:
        gpu_pool = _format_gpu_worker_pool(normalized_gpu_ids, workers_per_gpu)
        print(
            f"Launching {worker_count} parallel GPU workers for {total} candidates from {candidate_path}: {gpu_pool}"
        )
        if not continue_on_error:
            print("A failure will stop new dispatches, but jobs already running on other GPUs are allowed to finish.")

        task_queue: queue.Queue[CandidateRunPlan] = queue.Queue()
        for plan in plans:
            task_queue.put(plan)

        print_lock = threading.Lock()
        failures_lock = threading.Lock()
        stop_event = threading.Event()

        def worker(gpu_id: int, worker_index: int) -> None:
            while True:
                if stop_event.is_set() and not continue_on_error:
                    return

                try:
                    plan = task_queue.get_nowait()
                except queue.Empty:
                    return

                if stop_event.is_set() and not continue_on_error:
                    return

                return_code = _execute_candidate_plan(
                    plan,
                    repo_root=repo_root,
                    gpu_id=gpu_id,
                    python_executable=python_executable,
                    dry_run=False,
                    requested_uea_subset_names=requested_uea_subset_names,
                    print_lock=print_lock,
                    stream_prefix=f"[{plan.index}/{plan.total}][gpu:{gpu_id}][worker:{worker_index}]",
                    summary_callback=summary_callback,
                )
                if return_code == 0:
                    continue

                with failures_lock:
                    failures.append((plan.index, plan.candidate_name, return_code))
                if not continue_on_error:
                    stop_event.set()

        threads = [
            threading.Thread(
                target=worker,
                name=f"candidate-runner-gpu-{gpu_id}-worker-{worker_index}",
                args=(gpu_id, worker_index),
            )
            for gpu_id, worker_index in worker_specs
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    if dry_run:
        if worker_specs:
            gpu_pool = _format_gpu_worker_pool(normalized_gpu_ids, workers_per_gpu)
            print(
                f"Dry run completed. {total} candidate execution plans were generated from {candidate_path} "
                f"using {worker_count} GPU worker(s): {gpu_pool}."
            )
        else:
            print(f"Dry run completed. {total} candidate execution plans were generated from {candidate_path}.")
        return 0

    if failures:
        failures.sort(key=lambda item: item[0])
        failed_names = ", ".join(f"{name}(exit={code})" for _, name, code in failures)
        print(
            f"Finished with {len(failures)} failure(s) out of {total} candidates from {candidate_path}: {failed_names}"
        )
        if worker_count > 1 and not continue_on_error:
            print("New dispatches were stopped after the first failure; some in-flight GPU jobs may have completed before shutdown.")
        return failures[0][2]

    if worker_count > 1:
        print(
            f"Finished successfully. {total} candidate runs completed from {candidate_path} "
            f"across {worker_count} GPU workers on {len(normalized_gpu_ids)} GPU(s)."
        )
    else:
        print(f"Finished successfully. {total} candidate runs completed from {candidate_path}.")
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage search configs, sample candidate models, and run sampled candidates with run.py."
    )
    parser.add_argument("--spec", type=str, help="Path to a JSON spec file.")
    parser.add_argument(
        "--sample-search-config",
        type=str,
        default=None,
        help=(
            "Load search_config/<backbone_task_dataset>_search_spec.json, sample candidates from it, "
            "and write to candidates/<backbone_task_dataset>_candidates.json unless --output is provided."
        ),
    )
    parser.add_argument(
        "--sample-search-config-file",
        type=str,
        default=None,
        help=(
            "Load a specific search_config JSON file, sample candidates from it, "
            "and write to candidates/<spec>_candidates.json unless --output is provided."
        ),
    )
    parser.add_argument(
        "--run-candidates",
        type=str,
        default=None,
        help=(
            "Load candidates/<name>_candidates.json and execute every candidate with run.py. "
            "Execution is sequential by default, or parallel when multiple GPU workers are configured."
        ),
    )
    parser.add_argument(
        "--run-candidates-file",
        type=str,
        default=None,
        help=(
            "Load a specific candidate JSON file and execute every candidate with run.py. "
            "Execution is sequential by default, or parallel when multiple GPU workers are configured."
        ),
    )
    parser.add_argument(
        "--run-candidates-specify-id",
        "--run-candidate-specify_id",
        nargs="+",
        type=str,
        default=None,
        metavar="IDX",
        help=(
            "When running candidates, execute only selected 1-based candidate indices from the candidate JSON. "
            "Examples: --run-candidates-specify-id 97, --run-candidates-specify-id 1 5 97, "
            "--run-candidates-specify-id 1,5,97."
        ),
    )
    parser.add_argument("--backbone", type=str, help="Backbone name when specifying the search space directly in CLI.")
    parser.add_argument("--output", type=str, help="Path to write the sampled candidate JSON.")
    parser.add_argument(
        "--search-spec-name",
        type=str,
        default=None,
        help=(
            "File name to use under search_config/ when creating or updating a search spec in direct CLI mode. "
            "Example: --search-spec-name my_timesnet_search_spec.json"
        ),
    )
    parser.add_argument(
        "--candidates-name",
        type=str,
        default=None,
        help=(
            "File name to use under candidates/ for sampled candidate output, or when loading/running by name. "
            "Example: --candidates-name my_timesnet_candidates.json"
        ),
    )
    parser.add_argument("--num-samples", type=int, default=None, help="Override num_samples from the spec.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed from the spec.")
    parser.add_argument(
        "--candidate-prefix",
        type=str,
        default=None,
        help="Override candidate_prefix from the spec.",
    )
    parser.add_argument(
        "--allow-replacement",
        action="store_true",
        help="Sample with replacement even if num_samples exceeds the number of unique combinations.",
    )
    parser.add_argument(
        "--fixed",
        action="append",
        default=[],
        help="Fixed run.py argument in key=value form. Repeatable. Example: --fixed task_name=long_term_forecast",
    )
    parser.add_argument(
        "--search",
        action="append",
        default=[],
        help=(
            "Search-space assignment in key=v1,v2,... form. Repeatable. "
            "Example: --search d_model=64,128,256 or --search d_ff=x2,x4,x6"
        ),
    )
    parser.add_argument(
        "--list-backbones",
        action="store_true",
        help="Print available backbone names from models/ and exit.",
    )
    parser.add_argument(
        "--describe-backbone",
        type=str,
        default=None,
        help="Print the discovered configurable hyperparameters for a backbone and exit.",
    )
    parser.add_argument(
        "--describe-all-backbones",
        action="store_true",
        help="Print discovered configurable hyperparameters for every backbone and exit.",
    )
    parser.add_argument(
        "--refresh-default-recipes",
        action="store_true",
        help="Regenerate examples/ JSON recipe files from scripts/ and exit.",
    )
    parser.add_argument(
        "--gpu-id",
        nargs="+",
        type=str,
        default=None,
        metavar="GPU",
        help=(
            "One or more physical GPU ids to use when running candidate JSON files. "
            "Examples: --gpu-id 0, --gpu-id 0 1 2 3, --gpu-id 0,1,2,3. "
            "When multiple ids are given, the runner launches one worker per GPU by default, sets "
            "CUDA_VISIBLE_DEVICES=<gpu-id> for that worker, and passes --gpu 0 to run.py."
        ),
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help=(
            "How many candidate workers to launch per physical GPU id when running candidates. "
            "Example: --gpu-id 3 --workers-per-gpu 3 launches 3 parallel candidate processes on physical cuda:3."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the run.py commands that would be executed without launching training.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="When running candidates, continue with later candidates even if one candidate fails.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    strict_backbone_validation = True

    if args.list_backbones:
        for backbone in discover_available_backbones():
            print(backbone)
        return 0

    if args.describe_backbone:
        print(json.dumps(discover_backbone_hyperparameters(args.describe_backbone), indent=2, ensure_ascii=False))
        return 0

    if args.describe_all_backbones:
        payload = {
            backbone: discover_backbone_hyperparameters(backbone)
            for backbone in discover_available_backbones()
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if args.refresh_default_recipes:
        from benchmarking.default_recipe_generator import generate_default_recipes

        generated_paths = generate_default_recipes()
        print(f"Generated {len(generated_paths)} default recipe JSON files under examples/")
        return 0

    run_candidate_mode = bool(args.run_candidates or args.run_candidates_file)

    if args.sample_search_config and args.sample_search_config_file:
        parser.error("Use either --sample-search-config or --sample-search-config-file, not both.")
    if args.run_candidates and args.run_candidates_file:
        parser.error("Use either --run-candidates or --run-candidates-file, not both.")
    if args.spec and (
        args.sample_search_config
        or args.sample_search_config_file
        or args.run_candidates
        or args.run_candidates_file
        or args.backbone
        or args.fixed
        or args.search
    ):
        parser.error(
            "Use only one input mode: --spec, --sample-search-config/--sample-search-config-file, "
            "--run-candidates/--run-candidates-file, "
            "or the direct CLI options (--backbone/--fixed/--search)."
        )
    if (args.sample_search_config or args.sample_search_config_file) and (
        args.backbone or args.fixed or args.search or run_candidate_mode
    ):
        parser.error(
            "Do not mix --sample-search-config/--sample-search-config-file with the direct CLI options "
            "(--backbone/--fixed/--search)."
        )
    if run_candidate_mode and (args.backbone or args.fixed or args.search):
        parser.error(
            "Do not mix --run-candidates/--run-candidates-file with the direct CLI options "
            "(--backbone/--fixed/--search)."
        )
    if args.run_candidates_specify_id and not run_candidate_mode:
        parser.error("--run-candidates-specify-id can only be used with --run-candidates or --run-candidates-file.")
    if not args.spec and not args.sample_search_config and not args.sample_search_config_file and not run_candidate_mode and not args.backbone:
        parser.error(
            "Provide --spec, --sample-search-config, --sample-search-config-file, "
            "--run-candidates, --run-candidates-file, or use the direct CLI mode "
            "with --backbone, --fixed task_name=..., --fixed data=..., and --search."
        )

    if run_candidate_mode and args.output:
        parser.error("--output is not used with --run-candidates/--run-candidates-file.")
    if args.search_spec_name and (args.spec or args.sample_search_config_file or run_candidate_mode):
        parser.error("--search-spec-name is only used with direct CLI mode or --sample-search-config.")

    if run_candidate_mode:
        try:
            if args.workers_per_gpu < 1:
                parser.error("--workers-per-gpu must be a positive integer.")
            if args.workers_per_gpu != 1 and not args.gpu_id:
                parser.error("--workers-per-gpu requires --gpu-id.")
            gpu_ids = _parse_gpu_ids(args.gpu_id) if args.gpu_id else None
            repo_root = _repo_root()
            if args.run_candidates:
                if args.candidates_name:
                    candidate_path = _filename_in_dir(repo_root / "candidates", args.candidates_name)
                    if not candidate_path.exists():
                        parser.error(f"Candidate JSON not found: {candidate_path}")
                else:
                    candidate_path = _resolve_candidates_path_from_name(args.run_candidates)
            else:
                candidate_path = Path(args.run_candidates_file)
                if not candidate_path.exists():
                    parser.error(f"Candidate JSON not found: {candidate_path}")
            payload = _load_candidate_payload(candidate_path)
            selected_candidate_indices = _parse_candidate_indices(
                args.run_candidates_specify_id,
                total_candidates=len(payload["candidates"]),
            )
            if selected_candidate_indices:
                source_candidates = payload["candidates"]
                payload = dict(payload)
                payload["candidates"] = [
                    source_candidates[candidate_index - 1]
                    for candidate_index in selected_candidate_indices
                ]
                selected_text = ", ".join(str(candidate_index) for candidate_index in selected_candidate_indices)
                print(
                    f"Running {len(selected_candidate_indices)} selected candidate(s) by index: {selected_text}"
                )
        except ValueError as exc:
            parser.error(str(exc))
        return run_candidates_from_payload(
            payload,
            candidate_path=candidate_path,
            gpu_ids=gpu_ids,
            workers_per_gpu=args.workers_per_gpu,
            dry_run=args.dry_run,
            continue_on_error=args.continue_on_error,
        )

    if args.spec:
        if not args.output:
            parser.error(
                "--output is required in --spec mode."
            )
        spec_path = Path(args.spec)
        spec = _load_json(spec_path)
        updated_search_config_path = None
        strict_backbone_validation = False
    elif args.sample_search_config or args.sample_search_config_file:
        updated_search_config_path = None
        if args.sample_search_config:
            try:
                if args.search_spec_name:
                    spec_path = _filename_in_dir(_repo_root() / "search_config", args.search_spec_name)
                    if not spec_path.exists():
                        parser.error(f"Search config spec not found: {spec_path}")
                else:
                    spec_path = _resolve_search_config_path_from_name(args.sample_search_config)
            except ValueError as exc:
                parser.error(str(exc))
        else:
            spec_path = Path(args.sample_search_config_file)
            if not spec_path.exists():
                parser.error(f"Search config spec not found: {spec_path}")

        spec = _load_json(spec_path)
        strict_backbone_validation = False
        if not args.output:
            if args.candidates_name:
                args.output = str(_filename_in_dir(_repo_root() / "candidates", args.candidates_name))
            else:
                args.output = str(_default_candidates_output_path(spec_path))
    else:
        try:
            spec, updated_search_config_path = _build_or_update_spec_from_cli_args(args)
        except ValueError as exc:
            parser.error(str(exc))
        if args.search_spec_name:
            updated_search_config_path = _filename_in_dir(_repo_root() / "search_config", args.search_spec_name)
        _write_json(updated_search_config_path, spec)
        print(f"Updated search config: {updated_search_config_path}")
        if not args.output and args.candidates_name:
            args.output = str(_filename_in_dir(_repo_root() / "candidates", args.candidates_name))
        if not args.output:
            return 0

    payload = sample_candidates_from_spec(
        spec,
        output_path=args.output,
        num_samples_override=args.num_samples,
        seed_override=args.seed,
        candidate_prefix_override=args.candidate_prefix,
        allow_replacement_override=args.allow_replacement if args.allow_replacement else None,
        strict_backbone_validation=strict_backbone_validation,
    )

    print(
        f"Generated {payload['metadata']['num_generated']} candidates for "
        f"{payload['metadata']['backbone']} and wrote them to {args.output}"
    )
    print(f"Total valid combinations: {payload['metadata']['total_valid_combinations']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
