from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export epoch-wise DSPBuilder meta validation losses from valid_logs/*.txt to a single CSV file."
    )
    parser.add_argument("--base-dir", type=Path, required=True, help="Root directory containing validation run folders.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    return parser.parse_args()


def parse_key_value_line(line: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for token in line.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        values[key] = value
    return values


def collect_rows(base_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for valid_logs_dir in sorted(base_dir.rglob("valid_logs")):
        run_dir = valid_logs_dir.parent
        validation_dir = run_dir.parent

        for valid_log_path in sorted(valid_logs_dir.glob("*.txt")):
            dataset_name = valid_log_path.stem

            for raw_line in valid_log_path.read_text(encoding="utf-8").splitlines():
                if not raw_line.startswith("[EPOCH-SUMMARY]"):
                    continue

                parsed = parse_key_value_line(raw_line)
                rows.append(
                    {
                        "dataset": dataset_name,
                        "validation_dir": validation_dir.name,
                        "run_dir": run_dir.name,
                        "epoch": int(parsed["epoch"]),
                        "val_loss": float(parsed["val_loss"]),
                        "early_stopping_counter": int(parsed.get("early_stopping_counter", 0)),
                        "valid_log_path": str(valid_log_path),
                    }
                )

    rows.sort(
        key=lambda row: (
            str(row["validation_dir"]),
            str(row["run_dir"]),
            str(row["dataset"]),
            int(row["epoch"]),
        )
    )
    return rows


def write_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "validation_dir",
        "run_dir",
        "epoch",
        "val_loss",
        "early_stopping_counter",
        "valid_log_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows = collect_rows(args.base_dir.resolve())
    write_csv(args.output.resolve(), rows)
    print(f"Wrote {len(rows)} rows to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
