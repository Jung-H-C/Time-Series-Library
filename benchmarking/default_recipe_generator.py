from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any


RUN_MARKER = "__RUNPY_CALL__"
EXPORT_PATTERN = re.compile(r"^export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
SCRIPT_REFERENCE_PATTERN = re.compile(r"^(\./scripts/.*?\.sh)\b")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _coerce_scalar(value: str) -> Any:
    stripped = value.strip()
    if stripped.lower() in {"true", "false"}:
        return stripped.lower() == "true"
    if re.fullmatch(r"[-+]?\d+", stripped):
        return int(stripped)
    if re.fullmatch(r"[-+]?\d+\.\d+", stripped):
        return float(stripped)
    return stripped


def _extract_exported_env(script_text: str) -> dict[str, str]:
    exported: dict[str, str] = {}
    for raw_line in script_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = EXPORT_PATTERN.match(stripped)
        if not match:
            continue
        key, raw_value = match.groups()
        exported[key] = raw_value.strip().strip("'").strip('"')
    return exported


def _extract_referenced_scripts(script_text: str) -> list[str]:
    references: list[str] = []
    for raw_line in script_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = SCRIPT_REFERENCE_PATTERN.match(stripped)
        if match:
            references.append(match.group(1))
    return references


def _parse_run_args(tokens: list[str]) -> dict[str, Any]:
    if "run.py" not in tokens:
        raise ValueError(f"Unable to find run.py in token list: {tokens}")

    run_py_index = tokens.index("run.py")
    run_tokens = tokens[run_py_index + 1 :]
    run_args: dict[str, Any] = {}

    index = 0
    while index < len(run_tokens):
        token = run_tokens[index]
        if not token.startswith("--"):
            index += 1
            continue

        key = token[2:]
        values: list[Any] = []
        index += 1
        while index < len(run_tokens) and not run_tokens[index].startswith("--"):
            values.append(_coerce_scalar(run_tokens[index]))
            index += 1

        if not values:
            run_args[key] = True
        elif len(values) == 1:
            run_args[key] = values[0]
        else:
            run_args[key] = values

    return run_args


def _capture_run_commands(script_path: Path, repo_root: Path) -> tuple[list[dict[str, Any]], str]:
    capture_script = rf"""
python() {{
  printf '{RUN_MARKER} '
  printf '%q ' "$@"
  printf '\n'
}}
tee() {{
  cat
}}
export -f python tee
source "$1"
"""

    completed = subprocess.run(
        ["bash", "-lc", capture_script, "bash", str(script_path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    runs: list[dict[str, Any]] = []
    for raw_line in completed.stdout.splitlines():
        if not raw_line.startswith(f"{RUN_MARKER} "):
            continue
        shell_escaped = raw_line[len(RUN_MARKER) + 1 :]
        tokens = shlex.split(shell_escaped)
        run_args = _parse_run_args(tokens)
        interpreter_tokens = tokens[: tokens.index("run.py")]
        command_text = "python " + " ".join(shlex.quote(token) for token in tokens)
        runs.append(
            {
                "command": command_text,
                "interpreter_args": interpreter_tokens,
                "run_args": run_args,
            }
        )

    stderr = completed.stderr.strip()
    if completed.returncode != 0 and not runs:
        raise RuntimeError(
            f"Failed to parse {script_path} with bash capture.\n"
            f"returncode={completed.returncode}\n{stderr}"
        )

    return runs, stderr


def generate_recipe_for_script(script_path: Path, repo_root: Path | None = None) -> dict[str, Any]:
    repo_root = repo_root or _repo_root()
    scripts_root = repo_root / "scripts"
    relative_script = script_path.relative_to(repo_root)
    relative_to_scripts = script_path.relative_to(scripts_root)
    script_text = script_path.read_text(encoding="utf-8", errors="ignore")

    exported_env = _extract_exported_env(script_text)
    references = _extract_referenced_scripts(script_text)

    if "run.py" not in script_text:
        return {
            "source_script": str(relative_script),
            "relative_script_path": str(relative_to_scripts),
            "script_type": "wrapper_recipe",
            "exported_env": exported_env,
            "referenced_scripts": references,
            "runs": [],
        }

    runs, stderr = _capture_run_commands(script_path, repo_root)
    models = sorted({run["run_args"].get("model") for run in runs if run["run_args"].get("model")})
    tasks = sorted({run["run_args"].get("task_name") for run in runs if run["run_args"].get("task_name")})
    datasets = sorted({run["run_args"].get("data") for run in runs if run["run_args"].get("data")})

    return {
        "source_script": str(relative_script),
        "relative_script_path": str(relative_to_scripts),
        "script_type": "run_py_recipe",
        "exported_env": exported_env,
        "referenced_scripts": references,
        "summary": {
            "models": models,
            "tasks": tasks,
            "datasets": datasets,
            "num_runs": len(runs),
        },
        "runs": [
            {
                "index": idx,
                **run,
            }
            for idx, run in enumerate(runs, start=1)
        ],
        "capture_stderr": stderr,
    }


def generate_default_recipes(
    repo_root: Path | None = None,
) -> list[Path]:
    repo_root = repo_root or _repo_root()
    scripts_root = repo_root / "scripts"
    examples_root = repo_root / "examples"

    generated_paths: list[Path] = []
    for script_path in sorted(scripts_root.rglob("*.sh")):
        relative_to_scripts = script_path.relative_to(scripts_root)
        output_path = examples_root / relative_to_scripts.with_suffix(".json")
        recipe = generate_recipe_for_script(script_path, repo_root)
        _write_json(output_path, recipe)
        generated_paths.append(output_path)

    return generated_paths


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate default recipe JSON files under examples/ from scripts/."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _build_arg_parser().parse_args(argv)
    generated_paths = generate_default_recipes()
    print(f"Generated {len(generated_paths)} default recipe JSON files under examples/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
