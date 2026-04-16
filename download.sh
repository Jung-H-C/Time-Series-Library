#!/usr/bin/env bash

set -euo pipefail

ROOT_PATH="${ROOT_PATH:-./dataset}"

usage() {
  cat <<'EOF'
Usage:
  ./download.sh all
  ./download.sh bespoke
  ./download.sh generic
  ./download.sh tourism_monthly nn5 car_parts web_traffic dominick_tsf
  ./download.sh cif_2016 m1_monthly electricity_hourly

Notes:
  - `bespoke` downloads the branch-added bespoke datasets into ./dataset with the
    filenames expected by their dedicated loaders.
  - `generic` downloads every branch-added generic Monash TSF dataset into
    ./dataset/<dataset_key>/.
  - `all` downloads both groups.
  - You can override the root folder via `ROOT_PATH=./dataset ./download.sh ...`.
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

python - "$ROOT_PATH" "$@" <<'PY'
import glob
import os
import shutil
import sys

from data_provider.data_loader import MONASH_GENERIC_DATASETS, _ensure_monash_artifact


root_path = os.path.abspath(sys.argv[1])
requested = sys.argv[2:]

BESPOKE_CONFIGS = {
    "dominick_tsf": {
        "display_name": "Dominick TSF",
        "zenodo_records": ["https://zenodo.org/record/4654802"],
        "target_filename": "dominick_dataset.tsf",
    },
    "tourism_monthly": {
        "display_name": "Tourism Monthly",
        "zenodo_records": ["https://zenodo.org/record/4656096"],
        "target_filename": "tourism_monthly_dataset.tsf",
    },
    "nn5": {
        "display_name": "NN5 Daily W/O Missing",
        "zenodo_records": ["https://zenodo.org/record/4656117"],
        "target_filename": "nn5_daily_dataset_without_missing_values.tsf",
    },
    "car_parts": {
        "display_name": "Car Parts W/O Missing",
        "zenodo_records": ["https://zenodo.org/record/4656021"],
        "target_filename": "car_parts_dataset_without_missing_values.tsf",
    },
    "web_traffic": {
        "display_name": "Web Traffic Daily W/O Missing",
        "zenodo_records": ["https://zenodo.org/record/4656075"],
        "target_filename": "kaggle_web_traffic_dataset_without_missing_values.tsf",
    },
}

BESPOKE_ALIASES = {
    "dominick": "dominick_tsf",
    "dominik": "dominick_tsf",
    "tourism": "tourism_monthly",
    "nn5_daily": "nn5",
    "carparts": "car_parts",
    "webtraffic": "web_traffic",
    "kaggle_web_traffic": "web_traffic",
}


def normalize_name(name: str) -> str:
    lowered = name.strip().lower()
    return BESPOKE_ALIASES.get(lowered, lowered)


def find_downloaded_data_file(folder: str) -> str:
    candidates = []
    for pattern in ("**/*.tsf", "**/*.csv", "**/*.tsv", "**/*.parquet", "**/*.feather", "**/*.pkl", "**/*.pickle"):
        candidates.extend(glob.glob(os.path.join(folder, pattern), recursive=True))
    candidates = [path for path in candidates if os.path.isfile(path)]
    if not candidates:
        raise FileNotFoundError(f"No extracted dataset file found under {folder}")
    candidates.sort()
    return candidates[0]


def download_bespoke(dataset_key: str):
    cfg = BESPOKE_CONFIGS[dataset_key]
    stage_dir = os.path.join(root_path, "_downloads", dataset_key)
    os.makedirs(stage_dir, exist_ok=True)
    _ensure_monash_artifact(stage_dir, dataset_key, cfg)
    source_file = find_downloaded_data_file(os.path.join(stage_dir, dataset_key))
    target_file = os.path.join(root_path, cfg["target_filename"])
    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    shutil.copy2(source_file, target_file)
    print(f"[done] {dataset_key} -> {target_file}")


def download_generic(dataset_key: str):
    cfg = MONASH_GENERIC_DATASETS[dataset_key]
    dataset_dir = _ensure_monash_artifact(root_path, dataset_key, cfg)
    print(f"[done] {dataset_key} -> {dataset_dir}")


all_bespoke = sorted(BESPOKE_CONFIGS.keys())
all_generic = sorted(MONASH_GENERIC_DATASETS.keys())

expanded = []
for item in requested:
    lowered = item.lower()
    if lowered == "all":
        expanded.extend(all_bespoke)
        expanded.extend(all_generic)
    elif lowered == "bespoke":
        expanded.extend(all_bespoke)
    elif lowered == "generic":
        expanded.extend(all_generic)
    else:
        expanded.append(normalize_name(lowered))

seen = set()
final_targets = []
for item in expanded:
    if item in seen:
        continue
    seen.add(item)
    final_targets.append(item)

unknown = [
    item for item in final_targets
    if item not in BESPOKE_CONFIGS and item not in MONASH_GENERIC_DATASETS
]
if unknown:
    valid = ", ".join(sorted(list(BESPOKE_CONFIGS) + list(MONASH_GENERIC_DATASETS) + ["all", "bespoke", "generic"]))
    raise SystemExit(f"Unknown dataset key(s): {unknown}\nValid keys: {valid}")

print(f"ROOT_PATH={root_path}")
print(f"targets={final_targets}")

for dataset_key in final_targets:
    if dataset_key in BESPOKE_CONFIGS:
        download_bespoke(dataset_key)
    else:
        download_generic(dataset_key)
PY
