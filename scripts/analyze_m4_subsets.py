#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load_lengths(npz_path: Path) -> np.ndarray:
    series = np.load(npz_path, allow_pickle=True)
    if not isinstance(series, np.ndarray):
        raise TypeError(f"Expected numpy array in {npz_path}, got {type(series)!r}")
    return np.asarray([len(sample) for sample in series], dtype=np.int64)


def _summarize_subset(lengths: np.ndarray, threshold: int) -> dict[str, float | int]:
    if lengths.size == 0:
        return {
            "sample_count": 0,
            "avg_length": float("nan"),
            "min_length": 0,
            "max_length": 0,
            f"count_len_gt_{threshold}": 0,
        }
    return {
        "sample_count": int(lengths.size),
        "avg_length": float(lengths.mean()),
        "min_length": int(lengths.min()),
        "max_length": int(lengths.max()),
        f"count_len_gt_{threshold}": int(np.sum(lengths > threshold)),
    }


def build_report(dataset_root: Path, threshold: int) -> pd.DataFrame:
    info = pd.read_csv(dataset_root / "M4-info.csv")
    train_lengths = _load_lengths(dataset_root / "training.npz")
    test_lengths = _load_lengths(dataset_root / "test.npz")

    if len(info) != len(train_lengths) or len(info) != len(test_lengths):
        raise ValueError(
            "M4-info.csv row count must match training/test series counts: "
            f"info={len(info)}, train={len(train_lengths)}, test={len(test_lengths)}"
        )

    rows: list[dict[str, object]] = []
    for subset_name in ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]:
        subset_mask = info["SP"] == subset_name
        subset_train = train_lengths[subset_mask.to_numpy()]
        subset_test = test_lengths[subset_mask.to_numpy()]
        train_summary = _summarize_subset(subset_train, threshold)
        test_summary = _summarize_subset(subset_test, threshold)
        rows.append(
            {
                "subset": subset_name,
                "train_sample_count": train_summary["sample_count"],
                "train_avg_length": train_summary["avg_length"],
                "train_min_length": train_summary["min_length"],
                "train_max_length": train_summary["max_length"],
                f"train_count_len_gt_{threshold}": train_summary[f"count_len_gt_{threshold}"],
                "test_sample_count": test_summary["sample_count"],
                "test_avg_length": test_summary["avg_length"],
                "test_min_length": test_summary["min_length"],
                "test_max_length": test_summary["max_length"],
                f"test_count_len_gt_{threshold}": test_summary[f"count_len_gt_{threshold}"],
            }
        )

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze M4 subset sizes and sequence lengths from local training/test caches."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset/m4"),
        help="Path containing M4-info.csv, training.npz, and test.npz",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=200,
        help="Count sequences longer than this length",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Optional path to save the analysis as CSV",
    )
    args = parser.parse_args()

    report = build_report(args.dataset_root, args.threshold)

    display = report.copy()
    for column in display.columns:
        if column.endswith("avg_length"):
            display[column] = display[column].map(lambda value: f"{value:.2f}")

    print(display.to_string(index=False))

    if args.csv_path is not None:
        args.csv_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.csv_path, index=False)
        print(f"\nSaved CSV report to {args.csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
