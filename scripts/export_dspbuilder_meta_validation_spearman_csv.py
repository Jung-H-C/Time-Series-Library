from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export DSPBuilder meta validation spearman summaries to a single CSV file."
    )
    parser.add_argument("--base-dir", type=Path, required=True, help="Root directory containing validation run folders.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to spearman.csv in the base directory.",
    )
    return parser.parse_args()


def parse_key_value_line(line: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for token in line.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        values[key] = value
    return values


def load_best_epoch(summary_path: Path) -> int | None:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    best_epoch = payload.get("best_epoch")
    if isinstance(best_epoch, int):
        return best_epoch
    return None


def collect_rows(base_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for summary_path in sorted(base_dir.rglob("summary.json")):
        run_dir = summary_path.parent
        run_name = run_dir.parent.name
        best_epoch = load_best_epoch(summary_path)
        valid_logs_dir = run_dir / "valid_logs"
        valid_logs = sorted(valid_logs_dir.glob("*.txt"))

        if len(valid_logs) != 1:
            raise ValueError(f"Expected exactly one validation log under {valid_logs_dir}, found {len(valid_logs)}.")

        valid_log_path = valid_logs[0]
        dataset_name = valid_log_path.stem

        for raw_line in valid_log_path.read_text(encoding="utf-8").splitlines():
            if not raw_line.startswith("[VALID-SPEARMAN-SUMMARY]"):
                continue

            parsed = parse_key_value_line(raw_line)
            epoch = int(parsed["epoch"])
            spearman_mean = float(parsed["spearman_mean"])
            baseline_best_proxy = parsed.get("baseline_best_proxy", "")
            baseline_coefficient = parsed.get("baseline_coefficient", "")

            rows.append(
                {
                    "dataset": parsed.get("dataset", dataset_name),
                    "run_name": run_name,
                    "epoch": epoch,
                    "spearman_mean": spearman_mean,
                    "best_epoch": best_epoch,
                    "is_best_epoch": best_epoch == epoch,
                    "baseline_best_proxy": baseline_best_proxy,
                    "baseline_coefficient": baseline_coefficient,
                    "valid_log_path": str(valid_log_path),
                    "summary_path": str(summary_path),
                }
            )

    rows.sort(key=lambda row: (str(row["run_name"]), int(row["epoch"])))
    return rows


def write_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "run_name",
        "epoch",
        "spearman_mean",
        "best_epoch",
        "is_best_epoch",
        "baseline_best_proxy",
        "baseline_coefficient",
        "valid_log_path",
        "summary_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    output_path = args.output.resolve() if args.output is not None else base_dir / "spearman.csv"

    rows = collect_rows(base_dir)
    write_csv(output_path, rows)
    print(f"Wrote {len(rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
