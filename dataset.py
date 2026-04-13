import argparse
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

from data_provider.data_loader import (
    Dataset_DominickPanel,
    Dataset_DominickTSF,
    _parse_tsf_series,
)


def _build_dataset_args():
    return SimpleNamespace(augmentation_ratio=0)


def _format_stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def _print_section(title):
    print(f"\n[{title}]")


def _print_key_values(mapping):
    for key, value in mapping.items():
        print(f"- {key}: {value}")


def _print_bullets(lines):
    for line in lines:
        print(f"- {line}")


def _shape_to_str(shape):
    return "(" + ", ".join(str(int(dim)) for dim in shape) + ")"


def _resolve_existing_path(root_path, data_path):
    candidates = []
    if root_path and os.path.isfile(root_path):
        candidates.append(root_path)
    if root_path and data_path:
        candidates.append(os.path.join(root_path, data_path))
    if root_path and os.path.isdir(root_path):
        candidates.extend(
            os.path.join(root_path, name)
            for name in [
                data_path,
                "dominick_panel.csv",
                "dominick.csv",
                "dominik_panel.csv",
                "dominik.csv",
                "dominick_dataset.tsf",
            ]
            if name
        )

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find a dataset file under root_path={root_path}, data_path={data_path}")


def _summarize_window_dataset(dataset_cls, root_path, data_path, seq_len, label_len, pred_len, features, target):
    init_args = dict(
        args=_build_dataset_args(),
        root_path=root_path,
        data_path=data_path,
        size=[seq_len, label_len, pred_len],
        features=features,
        target=target,
        timeenc=1,
        freq="w",
    )
    train_dataset = dataset_cls(flag="train", **init_args)
    val_dataset = dataset_cls(flag="val", **init_args)
    test_dataset = dataset_cls(flag="test", **init_args)
    return train_dataset, val_dataset, test_dataset


def summarize_dominick_panel(root_path, data_path, seq_len, label_len, pred_len, show_rows):
    train_dataset, val_dataset, test_dataset = _summarize_window_dataset(
        Dataset_DominickPanel,
        root_path=root_path,
        data_path=data_path,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        features="M",
        target="sales",
    )

    panel_df = train_dataset._load_panel_frame()
    series_lengths = []
    usable_pairs = set(train_dataset.series_keys)
    for _, group_df in panel_df.groupby(["store_id", "sku_id"], sort=False):
        weekly = (
            group_df.groupby("week", as_index=False)
            .agg({"sales": "sum", "price": "mean", "margin": "mean", "promo": "max"})
        )
        series_lengths.append(len(weekly))

    sample_x, sample_y, sample_x_mark, sample_y_mark = train_dataset[0]
    first_store, first_sku = train_dataset.series_keys[0]
    first_group = (
        panel_df[
            (panel_df["store_id"].astype(str) == str(first_store))
            & (panel_df["sku_id"].astype(str) == str(first_sku))
        ]
        .sort_values("week")
        .head(show_rows)
    )

    _print_section("Format")
    _print_key_values(
        {
            "detected_format": "dominick_panel",
            "resolved_target": train_dataset.target,
            "features": ", ".join(train_dataset.feature_names),
            "window_spec": f"L={seq_len}, label_len={label_len}, P={pred_len}",
        }
    )

    _print_section("Dimension Meanings")
    _print_bullets(
        [
            f"L={seq_len}: number of past weeks fed into the encoder/input window",
            f"P={pred_len}: number of future weeks to predict",
            f"label_len={label_len}: decoder warm-up length kept for pipeline compatibility",
            f"D={sample_x.shape[1]}: input feature count -> {', '.join(train_dataset.feature_names)}",
            f"M={sample_x_mark.shape[1]}: mark feature count; for Dominick this first version uses zero-valued compatibility marks",
        ]
    )

    _print_section("Raw Counts")
    _print_key_values(
        {
            "rows": len(panel_df),
            "stores": panel_df["store_id"].nunique(),
            "skus": panel_df["sku_id"].nunique(),
            "store_sku_pairs": panel_df[["store_id", "sku_id"]].drop_duplicates().shape[0],
            "usable_pairs_after_filtering": len(usable_pairs),
        }
    )

    _print_section("Count Meanings")
    _print_bullets(
        [
            "`rows`: raw weekly retail observations before grouping into series",
            "`stores`: unique store ids in the panel file",
            "`skus`: unique sku/upc ids in the panel file",
            "`store_sku_pairs`: unique `(store_id, sku_id)` combinations before filtering",
            "`usable_pairs_after_filtering`: combinations that survived regular-weekly and minimum-length checks",
            "`train_windows` / `val_windows` / `test_windows`: number of sliding windows, not number of unique series",
        ]
    )

    _print_section("Series Lengths")
    _print_key_values(_format_stats(series_lengths))

    _print_section("Series Length Meaning")
    _print_bullets(
        [
            "These statistics are measured in weekly observations per `(store_id, sku_id)` series after grouping by week.",
        ]
    )

    _print_section("Window Counts")
    _print_key_values(
        {
            "train_windows": len(train_dataset),
            "val_windows": len(val_dataset),
            "test_windows": len(test_dataset),
        }
    )

    _print_section("Tensor Meanings")
    _print_bullets(
        [
            f"`seq_x`: past multivariate input window, shape {_shape_to_str(sample_x.shape)} = (L, D)",
            "`seq_x` rows correspond to weeks, and its columns are `[sales, price, margin, promo]`",
            f"`seq_y`: future target window, shape {_shape_to_str(sample_y.shape)} = (P, 1)",
            "`seq_y` contains only future `sales` values for the next horizon",
            f"`seq_x_mark`: mark tensor aligned with `seq_x`, shape {_shape_to_str(sample_x_mark.shape)} = (L, M)",
            f"`seq_y_mark`: mark tensor aligned with `seq_y`, shape {_shape_to_str(sample_y_mark.shape)} = (P, M)",
            "In the first Dominick implementation the mark tensors are all zeros and exist only for interface compatibility",
        ]
    )

    _print_section("Example Raw Rows")
    print(first_group.to_string(index=False))

    _print_section("Example Window")
    print("seq_x head:")
    print(pd.DataFrame(sample_x[: min(show_rows, len(sample_x))], columns=train_dataset.feature_names).to_string(index=False))
    print("seq_y head:")
    print(pd.DataFrame(sample_y[: min(show_rows, len(sample_y))], columns=[train_dataset.target]).to_string(index=False))


def summarize_dominick_tsf(root_path, data_path, seq_len, label_len, pred_len, show_rows):
    resolved_path = _resolve_existing_path(root_path, data_path)
    series_ids, series_values = _parse_tsf_series(resolved_path)
    train_dataset, val_dataset, test_dataset = _summarize_window_dataset(
        Dataset_DominickTSF,
        root_path=root_path,
        data_path=data_path,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        features="S",
        target="value",
    )

    lengths = [len(series) for series in series_values]
    sample_x, sample_y, sample_x_mark, sample_y_mark = train_dataset[0]
    first_series_id = series_ids[0]
    first_values = series_values[0][:show_rows]

    _print_section("Format")
    _print_key_values(
        {
            "detected_format": "dominick_tsf",
            "note": "TSF file is useful for series-scale inspection, but not for the new panel short-term spec.",
            "window_spec": f"L={seq_len}, label_len={label_len}, P={pred_len}",
        }
    )

    _print_section("Dimension Meanings")
    _print_bullets(
        [
            f"L={seq_len}: number of past weekly values used as input",
            f"label_len={label_len}: decoder warm-up length used by the legacy forecast interface",
            f"P={pred_len}: forecast horizon in weeks",
            f"D={sample_x.shape[1]}: value channels per time step; TSF file is univariate so D=1",
            f"M={sample_x_mark.shape[1]}: mark feature count returned by the generic forecast interface",
        ]
    )

    _print_section("Series Counts")
    _print_key_values(
        {
            "series": len(series_ids),
            "train_windows": len(train_dataset),
            "val_windows": len(val_dataset),
            "test_windows": len(test_dataset),
        }
    )

    _print_section("Count Meanings")
    _print_bullets(
        [
            "`series`: independent weekly sequences in the TSF file",
            "`train_windows` / `val_windows` / `test_windows`: number of sliding windows produced from all series",
        ]
    )

    _print_section("Series Lengths")
    _print_key_values(_format_stats(lengths))

    _print_section("Series Length Meaning")
    _print_bullets(
        [
            "These statistics are measured in weekly observations per TSF series.",
        ]
    )

    _print_section("Tensor Meanings")
    _print_bullets(
        [
            f"`seq_x`: past value window, shape {_shape_to_str(sample_x.shape)} = (L, 1)",
            f"`seq_y`: decoder context plus target horizon, shape {_shape_to_str(sample_y.shape)} = (label_len + P, 1)",
            "This legacy TSF loader returns the last `label_len` context values together with the future horizon",
            f"`seq_x_mark`: placeholder time features aligned with `seq_x`, shape {_shape_to_str(sample_x_mark.shape)} = (L, M)",
            f"`seq_y_mark`: placeholder time features aligned with `seq_y`, shape {_shape_to_str(sample_y_mark.shape)} = (label_len + P, M)",
        ]
    )

    _print_section("Example Series")
    print(f"- series_id: {first_series_id}")
    print(f"- first_values: {np.asarray(first_values).tolist()}")

    _print_section("Example Window")
    print("seq_x head:")
    print(pd.DataFrame(sample_x[: min(show_rows, len(sample_x))], columns=["value"]).to_string(index=False))
    print("seq_y head:")
    print(pd.DataFrame(sample_y[: min(show_rows, len(sample_y))], columns=["value"]).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Inspect Dominick dataset statistics and example samples.")
    parser.add_argument("--root_path", type=str, default="./dataset", help="Dataset root directory or full file path.")
    parser.add_argument("--data_path", type=str, default="dominick_dataset.tsf", help="Dataset filename under root_path.")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "panel", "tsf"], help="Dataset format to inspect.")
    parser.add_argument("--seq_len", type=int, default=16, help="Input window length.")
    parser.add_argument("--label_len", type=int, default=8, help="Decoder warmup length or compatibility field.")
    parser.add_argument("--pred_len", type=int, default=8, help="Prediction horizon.")
    parser.add_argument("--show_rows", type=int, default=5, help="Number of preview rows to print.")
    args = parser.parse_args()

    resolved_path = _resolve_existing_path(args.root_path, args.data_path)
    mode = args.mode
    if mode == "auto":
        mode = "tsf" if resolved_path.endswith(".tsf") else "panel"

    print(f"Resolved dataset file: {resolved_path}")
    if mode == "panel":
        summarize_dominick_panel(
            root_path=args.root_path,
            data_path=args.data_path,
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
            show_rows=args.show_rows,
        )
    else:
        summarize_dominick_tsf(
            root_path=args.root_path,
            data_path=args.data_path,
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
            show_rows=args.show_rows,
        )


if __name__ == "__main__":
    main()
