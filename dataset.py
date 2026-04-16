import argparse
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

from data_provider.data_loader import (
    Dataset_DominickPanel,
    Dataset_DominickTSF,
    Dataset_MonashTSFGeneric,
    Dataset_TourismMonthlyTSF,
    Dataset_NN5DailyTSF,
    Dataset_CarPartsTSF,
    Dataset_WebTrafficTSF,
    MONASH_GENERIC_DATASETS,
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
                "tourism_monthly_dataset.tsf",
                "tourism_monthly.tsf",
                "nn5_daily_dataset_without_missing_values.tsf",
                "nn5_daily.tsf",
                "car_parts_dataset_without_missing_values.tsf",
                "car_parts_dataset.tsf",
                "car_parts.tsf",
                "kaggle_web_traffic_dataset_without_missing_values.tsf",
                "web_traffic_dataset.tsf",
                "web_traffic.tsf",
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


def summarize_web_traffic_tsf(root_path, data_path, seq_len, label_len, pred_len, show_rows):
    init_args = dict(
        args=_build_dataset_args(),
        root_path=root_path,
        data_path=data_path,
        size=[seq_len, label_len, pred_len],
        features="S",
        target="traffic",
        timeenc=1,
        freq="web",
    )
    train_dataset = Dataset_WebTrafficTSF(flag="train", **init_args)
    val_dataset = Dataset_WebTrafficTSF(flag="val", **init_args)
    test_dataset = Dataset_WebTrafficTSF(flag="test", **init_args)

    sample_x, sample_y, sample_x_mark, sample_y_mark = train_dataset[0]
    first_series_id = train_dataset.series_ids[0]
    first_series = train_dataset.series_values[0]
    first_marks = sample_x_mark[: min(show_rows, len(sample_x_mark))]

    _print_section("Format")
    _print_key_values(
        {
            "detected_format": "web_traffic_tsf",
            "resolved_target": train_dataset.target,
            "window_spec": f"L={seq_len}, label_len={label_len}, P={pred_len}",
            "internal_freq_token": train_dataset.freq,
        }
    )

    _print_section("Dimension Meanings")
    _print_bullets(
        [
            f"L={seq_len}: number of past daily traffic values used as input",
            f"P={pred_len}: number of future daily traffic values to predict",
            f"label_len={label_len}: decoder compatibility length; the short-term target itself is the future P days",
            f"D={sample_x.shape[1]}: input feature count; Web Traffic is univariate so D=1",
            f"M={sample_x_mark.shape[1]}: calendar mark dimension -> {', '.join(train_dataset.mark_feature_names)}",
        ]
    )

    _print_section("Series Counts")
    _print_key_values(
        {
            "pages": train_dataset.num_series,
            "series_length_days": train_dataset.series_length,
            "train_windows": len(train_dataset),
            "val_windows": len(val_dataset),
            "test_windows": len(test_dataset),
        }
    )

    _print_section("Count Meanings")
    _print_bullets(
        [
            "`pages`: independent Wikipedia page series",
            "`series_length_days`: number of daily observations per page after loading the TSF file",
            "`train_windows` / `val_windows` / `test_windows`: number of fixed sliding windows across all pages",
        ]
    )

    _print_section("Series Lengths")
    _print_key_values(
        {
            "min": train_dataset.series_length,
            "mean": float(train_dataset.series_length),
            "median": float(train_dataset.series_length),
            "max": train_dataset.series_length,
        }
    )

    _print_section("Series Length Meaning")
    _print_bullets(
        [
            "All page series in this TSF file are equal-length daily sequences.",
        ]
    )

    _print_section("Tensor Meanings")
    _print_bullets(
        [
            f"`seq_x`: past daily traffic window, shape {_shape_to_str(sample_x.shape)} = (L, 1)",
            f"`seq_y`: future daily traffic target window, shape {_shape_to_str(sample_y.shape)} = (P, 1)",
            f"`seq_x_mark`: calendar marks aligned with `seq_x`, shape {_shape_to_str(sample_x_mark.shape)} = (L, M)",
            f"`seq_y_mark`: calendar marks aligned with `seq_y`, shape {_shape_to_str(sample_y_mark.shape)} = (P, M)",
            "Mark columns are `[dow_sin, dow_cos, month_sin, month_cos]` in the first implementation",
        ]
    )

    _print_section("Example Series")
    print(f"- series_id: {first_series_id}")
    print(f"- first_values: {np.asarray(first_series[:show_rows]).tolist()}")

    _print_section("Example Marks")
    print(pd.DataFrame(first_marks, columns=train_dataset.mark_feature_names).to_string(index=False))

    _print_section("Example Window")
    print("seq_x head:")
    print(pd.DataFrame(sample_x[: min(show_rows, len(sample_x))], columns=["traffic"]).to_string(index=False))
    print("seq_y head:")
    print(pd.DataFrame(sample_y[: min(show_rows, len(sample_y))], columns=["traffic"]).to_string(index=False))


def summarize_tourism_monthly_tsf(root_path, data_path, seq_len, label_len, pred_len, show_rows):
    resolved_path = _resolve_existing_path(root_path, data_path)
    series_ids, series_values = _parse_tsf_series(resolved_path)
    init_args = dict(
        args=_build_dataset_args(),
        root_path=root_path,
        data_path=data_path,
        size=[seq_len, label_len, pred_len],
        features="S",
        target="tourism",
        timeenc=1,
        freq="tourism_monthly",
    )
    train_dataset = Dataset_TourismMonthlyTSF(flag="train", **init_args)
    val_dataset = Dataset_TourismMonthlyTSF(flag="val", **init_args)
    test_dataset = Dataset_TourismMonthlyTSF(flag="test", **init_args)

    lengths = [len(series) for series in series_values]
    sample_x, sample_y, sample_x_mark, sample_y_mark = train_dataset[0]
    first_series_id = train_dataset.series_ids[0]
    first_raw_idx = series_ids.index(first_series_id) if first_series_id in series_ids else 0
    first_values = series_values[first_raw_idx][:show_rows]
    first_marks = sample_x_mark[: min(show_rows, len(sample_x_mark))]

    _print_section("Format")
    _print_key_values(
        {
            "detected_format": "tourism_monthly_tsf",
            "resolved_target": train_dataset.target,
            "frequency": "monthly",
            "window_spec": f"L={seq_len}, label_len={label_len}, P={pred_len}",
            "internal_freq_token": train_dataset.freq,
        }
    )

    _print_section("Dimension Meanings")
    _print_bullets(
        [
            f"L={seq_len}: number of past monthly tourism values used as input",
            f"P={pred_len}: number of future monthly tourism values to predict",
            f"label_len={label_len}: decoder compatibility length; decoder input reuses the last label_len values from seq_x",
            f"D={sample_x.shape[1]}: input feature count; Tourism monthly is univariate so D=1",
            f"M={sample_x_mark.shape[1]}: calendar mark dimension -> {', '.join(train_dataset.mark_feature_names)}",
        ]
    )

    _print_section("Series Counts")
    _print_key_values(
        {
            "raw_series": len(series_ids),
            "usable_series": train_dataset.num_series,
            "train_windows": len(train_dataset),
            "val_windows": len(val_dataset),
            "test_windows": len(test_dataset),
        }
    )

    _print_section("Count Meanings")
    _print_bullets(
        [
            "`raw_series`: independent tourism time series in the TSF file",
            "`usable_series`: series that are long enough for the requested fixed window",
            "`train_windows` / `val_windows` / `test_windows`: number of temporal sliding windows across all usable series",
        ]
    )

    _print_section("Series Lengths")
    _print_key_values(_format_stats(lengths))

    _print_section("Series Length Meaning")
    _print_bullets(
        [
            "These statistics are measured in monthly observations per tourism series.",
        ]
    )

    _print_section("Tensor Meanings")
    _print_bullets(
        [
            f"`seq_x`: past monthly tourism window, shape {_shape_to_str(sample_x.shape)} = (L, 1)",
            f"`seq_y`: future monthly tourism target window, shape {_shape_to_str(sample_y.shape)} = (P, 1)",
            f"`seq_x_mark`: month-of-year marks aligned with `seq_x`, shape {_shape_to_str(sample_x_mark.shape)} = (L, M)",
            f"`seq_y_mark`: month-of-year marks aligned with `seq_y`, shape {_shape_to_str(sample_y_mark.shape)} = (P, M)",
            "Mark columns are `[month_sin, month_cos]`, so the model can identify annual monthly seasonality",
        ]
    )

    _print_section("Example Series")
    print(f"- series_id: {first_series_id}")
    print(f"- first_values: {np.asarray(first_values).tolist()}")

    _print_section("Example Marks")
    print(pd.DataFrame(first_marks, columns=train_dataset.mark_feature_names).to_string(index=False))

    _print_section("Example Window")
    print("seq_x head:")
    print(pd.DataFrame(sample_x[: min(show_rows, len(sample_x))], columns=["tourism"]).to_string(index=False))
    print("seq_y head:")
    print(pd.DataFrame(sample_y[: min(show_rows, len(sample_y))], columns=["tourism"]).to_string(index=False))


def summarize_nn5_daily_tsf(root_path, data_path, seq_len, label_len, pred_len, show_rows):
    resolved_path = _resolve_existing_path(root_path, data_path)
    series_ids, series_values = _parse_tsf_series(resolved_path)
    init_args = dict(
        args=_build_dataset_args(),
        root_path=root_path,
        data_path=data_path,
        size=[seq_len, label_len, pred_len],
        features="S",
        target="cash",
        timeenc=1,
        freq="nn5_daily",
    )
    train_dataset = Dataset_NN5DailyTSF(flag="train", **init_args)
    val_dataset = Dataset_NN5DailyTSF(flag="val", **init_args)
    test_dataset = Dataset_NN5DailyTSF(flag="test", **init_args)

    lengths = [len(series) for series in series_values]
    sample_x, sample_y, sample_x_mark, sample_y_mark = train_dataset[0]
    first_series_id = train_dataset.series_ids[0]
    first_raw_idx = series_ids.index(first_series_id) if first_series_id in series_ids else 0
    first_values = series_values[first_raw_idx][:show_rows]
    first_marks = sample_x_mark[: min(show_rows, len(sample_x_mark))]

    _print_section("Format")
    _print_key_values(
        {
            "detected_format": "nn5_daily_tsf",
            "resolved_target": train_dataset.target,
            "frequency": "daily",
            "window_spec": f"L={seq_len}, label_len={label_len}, P={pred_len}",
            "internal_freq_token": train_dataset.freq,
        }
    )

    _print_section("Dimension Meanings")
    _print_bullets(
        [
            f"L={seq_len}: number of past daily ATM cash-withdrawal values used as input",
            f"P={pred_len}: number of future daily ATM cash-withdrawal values to predict",
            f"label_len={label_len}: decoder compatibility length; decoder input reuses the last label_len values from seq_x",
            f"D={sample_x.shape[1]}: input feature count; NN5 daily is univariate so D=1",
            f"M={sample_x_mark.shape[1]}: calendar mark dimension -> {', '.join(train_dataset.mark_feature_names)}",
        ]
    )

    _print_section("Series Counts")
    _print_key_values(
        {
            "raw_series": len(series_ids),
            "usable_series": train_dataset.num_series,
            "series_length_days": train_dataset.series_length,
            "train_windows": len(train_dataset),
            "val_windows": len(val_dataset),
            "test_windows": len(test_dataset),
        }
    )

    _print_section("Count Meanings")
    _print_bullets(
        [
            "`raw_series`: independent ATM cash-withdrawal series in the TSF file",
            "`usable_series`: series that are long enough for the requested fixed window",
            "`series_length_days`: number of daily observations per ATM series",
            "`train_windows` / `val_windows` / `test_windows`: number of temporal sliding windows across all usable series",
        ]
    )

    _print_section("Series Lengths")
    _print_key_values(_format_stats(lengths))

    _print_section("Series Length Meaning")
    _print_bullets(
        [
            "These statistics are measured in daily observations per NN5 ATM series.",
        ]
    )

    _print_section("Tensor Meanings")
    _print_bullets(
        [
            f"`seq_x`: past daily ATM cash-withdrawal window, shape {_shape_to_str(sample_x.shape)} = (L, 1)",
            f"`seq_y`: future daily ATM cash-withdrawal target window, shape {_shape_to_str(sample_y.shape)} = (P, 1)",
            f"`seq_x_mark`: daily calendar marks aligned with `seq_x`, shape {_shape_to_str(sample_x_mark.shape)} = (L, M)",
            f"`seq_y_mark`: daily calendar marks aligned with `seq_y`, shape {_shape_to_str(sample_y_mark.shape)} = (P, M)",
            "Mark columns are `[dow_sin, dow_cos, month_sin, month_cos]`, so the model can use weekly and monthly seasonality",
        ]
    )

    _print_section("Example Series")
    print(f"- series_id: {first_series_id}")
    print(f"- first_values: {np.asarray(first_values).tolist()}")

    _print_section("Example Marks")
    print(pd.DataFrame(first_marks, columns=train_dataset.mark_feature_names).to_string(index=False))

    _print_section("Example Window")
    print("seq_x head:")
    print(pd.DataFrame(sample_x[: min(show_rows, len(sample_x))], columns=["cash"]).to_string(index=False))
    print("seq_y head:")
    print(pd.DataFrame(sample_y[: min(show_rows, len(sample_y))], columns=["cash"]).to_string(index=False))


def summarize_car_parts_tsf(root_path, data_path, seq_len, label_len, pred_len, show_rows):
    resolved_path = _resolve_existing_path(root_path, data_path)
    series_ids, series_values = _parse_tsf_series(resolved_path)
    init_args = dict(
        args=_build_dataset_args(),
        root_path=root_path,
        data_path=data_path,
        size=[seq_len, label_len, pred_len],
        features="M",
        target="sales",
        timeenc=1,
        freq="car_parts_monthly",
    )
    train_dataset = Dataset_CarPartsTSF(flag="train", **init_args)
    val_dataset = Dataset_CarPartsTSF(flag="val", **init_args)
    test_dataset = Dataset_CarPartsTSF(flag="test", **init_args)

    lengths = [len(series) for series in series_values]
    sample_x, sample_y, sample_x_mark, sample_y_mark = train_dataset[0]
    preview_dim = min(show_rows, train_dataset.feature_dim)
    preview_columns = train_dataset.feature_names[:preview_dim]
    first_marks = sample_x_mark[: min(show_rows, len(sample_x_mark))]

    _print_section("Format")
    _print_key_values(
        {
            "detected_format": "car_parts_tsf",
            "resolved_target": train_dataset.target,
            "frequency": "monthly",
            "window_spec": f"L={seq_len}, label_len={label_len}, P={pred_len}",
            "internal_freq_token": train_dataset.freq,
        }
    )

    _print_section("Dimension Meanings")
    _print_bullets(
        [
            f"L={seq_len}: number of past monthly sales vectors used as input",
            f"P={pred_len}: number of future monthly sales vectors to predict",
            f"label_len={label_len}: decoder compatibility length; decoder input reuses the last label_len vectors from seq_x",
            f"D={sample_x.shape[1]}: input feature count; each feature/channel is one car-part series",
            f"M={sample_x_mark.shape[1]}: calendar mark dimension -> {', '.join(train_dataset.mark_feature_names)}",
            "Unlike most TSF loaders here, Car Parts is loaded as one aligned multivariate matrix, not as many D=1 samples",
        ]
    )

    _print_section("Series Counts")
    _print_key_values(
        {
            "raw_part_series": len(series_ids),
            "feature_dim": train_dataset.feature_dim,
            "series_length_months": train_dataset.series_length,
            "train_windows": len(train_dataset),
            "val_windows": len(val_dataset),
            "test_windows": len(test_dataset),
        }
    )

    _print_section("Count Meanings")
    _print_bullets(
        [
            "`raw_part_series`: independent car-part demand series in the TSF file",
            "`feature_dim`: number of part series stacked into the multivariate input channel dimension D",
            "`series_length_months`: common monthly timeline length shared by every part series",
            "`train_windows` / `val_windows` / `test_windows`: number of temporal windows from the single aligned multivariate matrix",
        ]
    )

    _print_section("Series Lengths")
    _print_key_values(_format_stats(lengths))

    _print_section("Series Length Meaning")
    _print_bullets(
        [
            "These statistics are measured in monthly observations per car-part series.",
        ]
    )

    _print_section("Tensor Meanings")
    _print_bullets(
        [
            f"`seq_x`: past monthly multivariate sales window, shape {_shape_to_str(sample_x.shape)} = (L, D)",
            f"`seq_y`: future monthly multivariate sales target, shape {_shape_to_str(sample_y.shape)} = (P, D)",
            f"`seq_x_mark`: month-of-year marks aligned with `seq_x`, shape {_shape_to_str(sample_x_mark.shape)} = (L, M)",
            f"`seq_y_mark`: month-of-year marks aligned with `seq_y`, shape {_shape_to_str(sample_y_mark.shape)} = (P, M)",
            "Mark columns are `[month_sin, month_cos]`, so the model can identify annual monthly seasonality",
        ]
    )

    _print_section("Example Channels")
    print(f"- first_channels: {preview_columns}")
    print(f"- raw_first_values_for_first_channel: {np.asarray(series_values[0][:show_rows]).tolist()}")

    _print_section("Example Marks")
    print(pd.DataFrame(first_marks, columns=train_dataset.mark_feature_names).to_string(index=False))

    _print_section("Example Window")
    print(f"Showing first {preview_dim} of {train_dataset.feature_dim} channels.")
    print("seq_x head:")
    print(
        pd.DataFrame(
            sample_x[: min(show_rows, len(sample_x)), :preview_dim],
            columns=preview_columns,
        ).to_string(index=False)
    )
    print("seq_y head:")
    print(
        pd.DataFrame(
            sample_y[: min(show_rows, len(sample_y)), :preview_dim],
            columns=preview_columns,
        ).to_string(index=False)
    )


def summarize_generic_monash_tsf(root_path, data_path, dataset_key, seq_len, label_len, pred_len, show_rows):
    resolved_path = _resolve_existing_path(root_path, data_path)
    series_ids, series_values = _parse_tsf_series(resolved_path)
    init_args = dict(
        args=SimpleNamespace(augmentation_ratio=0, data=dataset_key),
        root_path=root_path,
        data_path=data_path,
        size=[seq_len, label_len, pred_len],
        features="S",
        target=MONASH_GENERIC_DATASETS[dataset_key].get("target", "value"),
        timeenc=0,
        freq="generic",
    )
    train_dataset = Dataset_MonashTSFGeneric(flag="train", **init_args)
    val_dataset = Dataset_MonashTSFGeneric(flag="val", **init_args)
    test_dataset = Dataset_MonashTSFGeneric(flag="test", **init_args)

    lengths = [len(series) for series in series_values]
    sample_x, sample_y, sample_x_mark, sample_y_mark = train_dataset[0]

    _print_section("Format")
    _print_key_values(
        {
            "detected_format": "generic_monash_tsf",
            "dataset_key": dataset_key,
            "display_name": MONASH_GENERIC_DATASETS[dataset_key]["display_name"],
            "window_spec": f"L={seq_len}, label_len={label_len}, P={pred_len}",
            "task": MONASH_GENERIC_DATASETS[dataset_key]["task"],
        }
    )

    _print_section("Dimension Meanings")
    _print_bullets(
        [
            f"L={seq_len}: number of past values used as input",
            f"P={pred_len}: number of future values to predict",
            f"label_len={label_len}: compatibility field kept aligned with the short-term pipeline",
            f"D={sample_x.shape[1]}: feature dimension; first-pass generic integration is univariate so D=1",
            f"M={sample_x_mark.shape[1]}: compatibility mark dimension; generic integration currently uses zero marks",
        ]
    )

    _print_section("Series Counts")
    _print_key_values(
        {
            "raw_series": len(series_ids),
            "train_windows": len(train_dataset),
            "val_windows": len(val_dataset),
            "test_windows": len(test_dataset),
        }
    )

    _print_section("Series Lengths")
    _print_key_values(_format_stats(lengths))

    _print_section("Tensor Meanings")
    _print_bullets(
        [
            f"`seq_x`: past window, shape {_shape_to_str(sample_x.shape)} = (L, 1)",
            f"`seq_y`: future target window, shape {_shape_to_str(sample_y.shape)} = (P, 1)",
            f"`seq_x_mark`: zero compatibility mark, shape {_shape_to_str(sample_x_mark.shape)} = (L, M)",
            f"`seq_y_mark`: zero compatibility mark, shape {_shape_to_str(sample_y_mark.shape)} = (P, M)",
        ]
    )

    _print_section("Example Series")
    print(f"- series_id: {series_ids[0]}")
    print(f"- first_values: {np.asarray(series_values[0][:show_rows]).tolist()}")

    _print_section("Example Window")
    print("seq_x head:")
    print(pd.DataFrame(sample_x[: min(show_rows, len(sample_x))], columns=["value"]).to_string(index=False))
    print("seq_y head:")
    print(pd.DataFrame(sample_y[: min(show_rows, len(sample_y))], columns=["value"]).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Inspect local dataset statistics and example samples.")
    parser.add_argument("--root_path", type=str, default="./dataset", help="Dataset root directory or full file path.")
    parser.add_argument("--data_path", type=str, default="dominick_dataset.tsf", help="Dataset filename under root_path.")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "panel", "tsf", "tourism", "nn5", "car_parts", "web", "monash"], help="Dataset format to inspect.")
    parser.add_argument("--dataset_key", type=str, default="", help="Generic Monash dataset key for `--mode monash`.")
    parser.add_argument("--seq_len", type=int, default=16, help="Input window length.")
    parser.add_argument("--label_len", type=int, default=8, help="Decoder warmup length or compatibility field.")
    parser.add_argument("--pred_len", type=int, default=8, help="Prediction horizon.")
    parser.add_argument("--show_rows", type=int, default=5, help="Number of preview rows to print.")
    args = parser.parse_args()

    resolved_path = _resolve_existing_path(args.root_path, args.data_path)
    mode = args.mode
    if mode == "auto":
        filename = os.path.basename(resolved_path).lower()
        if "web_traffic" in filename:
            mode = "web"
        elif "tourism" in filename:
            mode = "tourism"
        elif "nn5" in filename:
            mode = "nn5"
        elif "dominick" in filename or "dominik" in filename:
            mode = "tsf"
        elif "car_parts" in filename or ("car" in filename and "parts" in filename):
            mode = "car_parts"
        elif resolved_path.endswith(".tsf"):
            mode = "monash"
        else:
            mode = "panel"

    seq_len = args.seq_len
    label_len = args.label_len
    pred_len = args.pred_len
    if mode == "web" and (seq_len, label_len, pred_len) == (16, 8, 8):
        seq_len, label_len, pred_len = 90, 30, 30
    elif mode == "tourism" and (seq_len, label_len, pred_len) == (16, 8, 8):
        seq_len, label_len, pred_len = 48, 24, 24
    elif mode == "nn5" and (seq_len, label_len, pred_len) == (16, 8, 8):
        seq_len, label_len, pred_len = 112, 56, 56
    elif mode == "car_parts" and (seq_len, label_len, pred_len) == (16, 8, 8):
        seq_len, label_len, pred_len = 24, 12, 12

    print(f"Resolved dataset file: {resolved_path}")
    if mode == "panel":
        summarize_dominick_panel(
            root_path=args.root_path,
            data_path=args.data_path,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            show_rows=args.show_rows,
        )
    elif mode == "tsf":
        summarize_dominick_tsf(
            root_path=args.root_path,
            data_path=args.data_path,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            show_rows=args.show_rows,
        )
    elif mode == "tourism":
        summarize_tourism_monthly_tsf(
            root_path=args.root_path,
            data_path=args.data_path,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            show_rows=args.show_rows,
        )
    elif mode == "nn5":
        summarize_nn5_daily_tsf(
            root_path=args.root_path,
            data_path=args.data_path,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            show_rows=args.show_rows,
        )
    elif mode == "car_parts":
        summarize_car_parts_tsf(
            root_path=args.root_path,
            data_path=args.data_path,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            show_rows=args.show_rows,
        )
    elif mode == "monash":
        if not args.dataset_key:
            raise ValueError("`--mode monash` requires `--dataset_key`.")
        summarize_generic_monash_tsf(
            root_path=args.root_path,
            data_path=args.data_path,
            dataset_key=args.dataset_key,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            show_rows=args.show_rows,
        )
    else:
        summarize_web_traffic_tsf(
            root_path=args.root_path,
            data_path=args.data_path,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            show_rows=args.show_rows,
        )


if __name__ == "__main__":
    main()
