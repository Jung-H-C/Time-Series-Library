import os
from bisect import bisect_right
import numpy as np
import pandas as pd
import glob
import re
import shutil
import torch
import zipfile
import tarfile
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.uea import subsample, interpolate_missing, Normalizer
import warnings
from utils.augmentation import run_augmentation_single
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen

try:
    from data_provider.m4 import M4Dataset, M4Meta
except ImportError:
    M4Dataset = None
    M4Meta = None

try:
    from sktime.datasets import load_from_tsfile_to_dataframe
except ImportError:
    load_from_tsfile_to_dataframe = None

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

warnings.filterwarnings('ignore')

HUGGINGFACE_REPO = "thuml/Time-Series-Library"

GENERIC_TS_ZERO_MARK_DIM = 1
MONASH_GENERIC_DATASETS = {
    "m1_yearly": {
        "display_name": "M1 Yearly",
        "target": "value",
        "task": "short_term_forecast",
        "default_size": (8, 2, 2),
        "zenodo_records": ["https://zenodo.org/record/4656193"],
        "search_tokens": ["m1", "yearly"],
    },
    "m1_quarterly": {
        "display_name": "M1 Quarterly",
        "target": "value",
        "task": "short_term_forecast",
        "default_size": (16, 4, 4),
        "zenodo_records": ["https://zenodo.org/record/4656154"],
        "search_tokens": ["m1", "quarterly"],
    },
    "m1_monthly": {
        "display_name": "M1 Monthly",
        "target": "value",
        "task": "short_term_forecast",
        "default_size": (48, 12, 12),
        "zenodo_records": ["https://zenodo.org/record/4656159"],
        "search_tokens": ["m1", "monthly"],
    },
    "m3_yearly": {
        "display_name": "M3 Yearly",
        "target": "value",
        "task": "short_term_forecast",
        "default_size": (8, 2, 2),
        "zenodo_records": ["https://zenodo.org/record/4656222"],
        "search_tokens": ["m3", "yearly"],
    },
    "m3_quarterly": {
        "display_name": "M3 Quarterly",
        "target": "value",
        "task": "short_term_forecast",
        "default_size": (16, 4, 4),
        "zenodo_records": ["https://zenodo.org/record/4656262"],
        "search_tokens": ["m3", "quarterly"],
    },
    "m3_monthly": {
        "display_name": "M3 Monthly",
        "target": "value",
        "task": "short_term_forecast",
        "default_size": (48, 12, 12),
        "zenodo_records": ["https://zenodo.org/record/4656298"],
        "search_tokens": ["m3", "monthly"],
    },
    "m3_other": {
        "display_name": "M3 Other",
        "target": "value",
        "task": "short_term_forecast",
        "default_size": (48, 12, 12),
        "zenodo_records": ["https://zenodo.org/record/4656335"],
        "search_tokens": ["m3", "other"],
    },
    "cif_2016": {
        "display_name": "CIF 2016",
        "target": "value",
        "task": "short_term_forecast",
        "default_size": (48, 12, 12),
        "zenodo_records": ["https://zenodo.org/record/4656042"],
        "search_tokens": ["cif", "2016"],
    },
    "london_smart_meters": {
        "display_name": "London Smart Meters",
        "target": "value",
        "task": "short_term_forecast",
        "default_size": (168, 24, 24),
        "zenodo_records": ["https://zenodo.org/record/4656091"],
        "search_tokens": ["london", "smart", "meter"],
    },
    "aus_electricity_demand": {
        "display_name": "Aus. Electricity Demand",
        "target": "value",
        "task": "short_term_forecast",
        "default_size": (336, 48, 48),
        "zenodo_records": ["https://zenodo.org/record/4659727"],
        "search_tokens": ["aus", "electricity", "demand"],
    },
    "wind_farms": {
        "display_name": "Wind Farms",
        "target": "value",
        "task": "short_term_forecast",
        "default_size": (168, 24, 24),
        "zenodo_records": ["https://zenodo.org/record/4654858"],
        "search_tokens": ["wind", "farm"],
    },
    "bitcoin": {
        "display_name": "Bitcoin",
        "target": "value",
        "task": "short_term_forecast",
        "default_size": (90, 30, 30),
        "zenodo_records": ["https://zenodo.org/record/5122101"],
        "search_tokens": ["bitcoin"],
    },
    "pedestrian_counts": {
        "display_name": "Pedestrian Counts",
        "target": "value",
        "task": "short_term_forecast",
        "default_size": (168, 24, 24),
        "zenodo_records": ["https://zenodo.org/record/4656626"],
        "search_tokens": ["pedestrian", "count"],
    },
    "vehicle_trips": {
        "display_name": "Vehicle Trips",
        "target": "value",
        "task": "short_term_forecast",
        "default_size": (52, 13, 13),
        "zenodo_records": ["https://zenodo.org/record/5122537"],
        "search_tokens": ["vehicle", "trip"],
    },
    "kdd_cup_2018": {
        "display_name": "KDD Cup 2018",
        "target": "value",
        "task": "short_term_forecast",
        "default_size": (90, 30, 30),
        "zenodo_records": ["https://zenodo.org/record/4656756"],
        "search_tokens": ["kdd", "2018"],
    },
    "weather_tsf": {
        "display_name": "Weather",
        "target": "value",
        "task": "short_term_forecast",
        "default_size": (90, 30, 30),
        "zenodo_records": ["https://zenodo.org/record/4654822"],
        "search_tokens": ["weather"],
    },
    "solar_10min": {
        "display_name": "Solar 10 Minutes",
        "target": "value",
        "task": "short_term_forecast",
        "default_size": (1008, 144, 144),
        "zenodo_records": ["https://zenodo.org/record/4656144"],
        "search_tokens": ["solar", "10", "minute"],
    },
    "solar_weekly": {
        "display_name": "Solar Weekly",
        "target": "value",
        "task": "short_term_forecast",
        "default_size": (52, 13, 13),
        "zenodo_records": ["https://zenodo.org/record/4656151"],
        "search_tokens": ["solar", "weekly"],
    },
    "electricity_hourly": {
        "display_name": "Electricity Hourly",
        "target": "value",
        "task": "short_term_forecast",
        "default_size": (168, 24, 24),
        "zenodo_records": ["https://zenodo.org/record/4656140"],
        "search_tokens": ["electricity", "hourly"],
    },
    "electricity_weekly": {
        "display_name": "Electricity Weekly",
        "target": "value",
        "task": "short_term_forecast",
        "default_size": (52, 13, 13),
        "zenodo_records": ["https://zenodo.org/record/4656141"],
        "search_tokens": ["electricity", "weekly"],
    },
    "fred_md": {
        "display_name": "FRED-MD",
        "target": "value",
        "task": "short_term_forecast",
        "default_size": (48, 12, 12),
        "zenodo_records": ["https://zenodo.org/record/4654833"],
        "search_tokens": ["fred", "md"],
    },
    "san_francisco_traffic_hourly": {
        "display_name": "San Francisco Traffic Hourly",
        "target": "value",
        "task": "short_term_forecast",
        "default_size": (168, 24, 24),
        "zenodo_records": ["https://zenodo.org/record/4656132"],
        "search_tokens": ["san", "francisco", "traffic", "hourly"],
    },
    "san_francisco_traffic_weekly": {
        "display_name": "San Francisco Traffic Weekly",
        "target": "value",
        "task": "short_term_forecast",
        "default_size": (52, 13, 13),
        "zenodo_records": ["https://zenodo.org/record/4656135"],
        "search_tokens": ["san", "francisco", "traffic", "weekly"],
    },
    "rideshare": {
        "display_name": "Rideshare",
        "target": "value",
        "task": "short_term_forecast",
        "default_size": (52, 13, 13),
        "zenodo_records": ["https://zenodo.org/record/5122232"],
        "search_tokens": ["rideshare"],
    },
    "hospital": {
        "display_name": "Hospital",
        "target": "value",
        "task": "short_term_forecast",
        "default_size": (48, 12, 12),
        "zenodo_records": ["https://zenodo.org/record/4656014"],
        "search_tokens": ["hospital"],
    },
    "covid_deaths": {
        "display_name": "COVID Deaths",
        "target": "value",
        "task": "short_term_forecast",
        "default_size": (90, 30, 30),
        "zenodo_records": ["https://zenodo.org/record/4656009"],
        "search_tokens": ["covid", "death"],
    },
    "temperature_rain": {
        "display_name": "Temperature Rain",
        "target": "value",
        "task": "short_term_forecast",
        "default_size": (90, 30, 30),
        "zenodo_records": ["https://zenodo.org/record/5129091"],
        "search_tokens": ["temperature", "rain"],
    },
    "sunspot": {
        "display_name": "Sunspot",
        "target": "value",
        "task": "short_term_forecast",
        "default_size": (365, 90, 90),
        "zenodo_records": ["https://zenodo.org/record/4654722"],
        "search_tokens": ["sunspot"],
    },
    "saugeen_river_flow": {
        "display_name": "Saugeen River Flow",
        "target": "value",
        "task": "short_term_forecast",
        "default_size": (90, 30, 30),
        "zenodo_records": ["https://zenodo.org/record/4656058"],
        "search_tokens": ["saugeen", "river", "flow"],
    },
    "us_births": {
        "display_name": "US Births",
        "target": "value",
        "task": "short_term_forecast",
        "default_size": (90, 30, 30),
        "zenodo_records": ["https://zenodo.org/record/4656049"],
        "search_tokens": ["us", "birth"],
    },
    "solar_power": {
        "display_name": "Solar Power",
        "target": "value",
        "task": "short_term_forecast",
        "default_size": (900, 225, 225),
        "zenodo_records": ["https://zenodo.org/record/4656027"],
        "search_tokens": ["solar", "power"],
    },
    "wind_power": {
        "display_name": "Wind Power",
        "target": "value",
        "task": "short_term_forecast",
        "default_size": (900, 225, 225),
        "zenodo_records": ["https://zenodo.org/record/4656032"],
        "search_tokens": ["wind", "power"],
    },
}


def _require_optional_dependency(dependency, import_name, context):
    if dependency is None:
        raise ImportError(
            f"{import_name} is required for {context}. "
            f"Install `{import_name}` or place the expected local dataset files so this path is not needed."
        )


def _download_url_to_file(url, destination):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with urlopen(url) as response, open(destination, "wb") as handle:
        shutil.copyfileobj(response, handle)


def _extract_archive_if_needed(filepath, destination_dir):
    lower = filepath.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(filepath, "r") as archive:
            archive.extractall(destination_dir)
        return
    if lower.endswith((".tar.gz", ".tgz", ".tar")):
        mode = "r:gz" if lower.endswith((".tar.gz", ".tgz")) else "r:"
        with tarfile.open(filepath, mode) as archive:
            archive.extractall(destination_dir)


def _resolve_zenodo_download_url(record_url):
    with urlopen(record_url) as response:
        html = response.read().decode("utf-8", errors="replace")

    matches = re.findall(r'href="([^"]+/files/[^"]+\?download=1)"', html)
    filtered = [link for link in matches if "/preview/" not in link]
    if not filtered:
        raise RuntimeError(f"Could not locate a downloadable file on the Zenodo page: {record_url}")
    return urljoin(record_url, filtered[0])


def _ensure_monash_artifact(root_path, dataset_key, config):
    dataset_root = os.path.join(root_path, dataset_key)
    os.makedirs(dataset_root, exist_ok=True)

    preferred_extensions = {".tsf", ".csv", ".tsv", ".parquet", ".feather", ".pkl", ".pickle"}
    for existing_path in glob.glob(os.path.join(dataset_root, "**", "*"), recursive=True):
        if os.path.isfile(existing_path) and os.path.splitext(existing_path)[1].lower() in preferred_extensions:
            return dataset_root

    for record_url in config.get("zenodo_records", []):
        download_url = _resolve_zenodo_download_url(record_url)
        filename = os.path.basename(urlparse(download_url).path)
        local_archive = os.path.join(dataset_root, filename)
        if not os.path.exists(local_archive):
            _download_url_to_file(download_url, local_archive)
        _extract_archive_if_needed(local_archive, dataset_root)

        for extracted_path in glob.glob(os.path.join(dataset_root, "**", "*"), recursive=True):
            if os.path.isfile(extracted_path) and os.path.splitext(extracted_path)[1].lower() in preferred_extensions:
                return dataset_root

    raise FileNotFoundError(
        f"Could not find an extracted dataset file for `{dataset_key}` under {dataset_root} "
        "after attempting the configured downloads."
    )


def _read_tsf_header(filepath):
    metadata = {}
    attribute_names = []

    with open(filepath, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            lowered = line.lower()
            if lowered == "@data":
                if not attribute_names:
                    raise ValueError(f"TSF file is missing `@attribute` definitions: {filepath}")
                return metadata, attribute_names

            if lowered.startswith("@attribute"):
                parts = re.split(r"\s+", line, maxsplit=2)
                if len(parts) < 3:
                    raise ValueError(f"Malformed TSF attribute line: `{line}`")
                attribute_names.append(parts[1])
                continue

            for key in ["@relation", "@frequency", "@horizon", "@missing", "@equallength"]:
                if lowered.startswith(key):
                    metadata[key[1:]] = line.split(None, 1)[1].strip() if " " in line else ""
                    break

    raise ValueError(f"TSF file is missing an `@data` section: {filepath}")


def _iter_tsf_records(filepath, attribute_names=None):
    if attribute_names is None:
        _, attribute_names = _read_tsf_header(filepath)

    in_data_section = False
    with open(filepath, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            lowered = line.lower()
            if lowered == "@data":
                in_data_section = True
                continue
            if not in_data_section:
                continue

            parts = line.split(":", len(attribute_names))
            if len(parts) != len(attribute_names) + 1:
                raise ValueError(
                    f"Malformed TSF record in {filepath}. Expected {len(attribute_names)} attributes "
                    f"before the value field, got line: `{line[:200]}`"
                )

            record = {name: value for name, value in zip(attribute_names, parts[:-1])}
            record["values"] = np.asarray(
                [np.nan if token == "?" else float(token) for token in parts[-1].split(",")],
                dtype=np.float32,
            )
            yield record


def _parse_tsf_series(filepath):
    """Parse the id/value pairs needed by the lightweight TSF inspection paths."""
    _, attribute_names = _read_tsf_header(filepath)
    series_ids = []
    series_values = []

    for record in _iter_tsf_records(filepath, attribute_names):
        series_id = record.get("series_name")
        if series_id is None:
            series_id = record.get(attribute_names[0], f"series_{len(series_ids)}")

        values = np.asarray(record["values"], dtype=np.float32)
        values = values[~np.isnan(values)]
        if values.size == 0:
            continue

        series_ids.append(series_id)
        series_values.append(values)

    if not series_values:
        raise ValueError(f"No time series found in TSF file: {filepath}")

    return series_ids, series_values


def _default_time_mark(length, timeenc, freq):
    """Build placeholder time features for timestamp-less series.

    Dominick TSF provides weekly ordering but no absolute calendar timestamps.
    Because there is no reliable calendar origin, we use zero-valued time marks
    with the expected dimensionality so the standard forecast interface still
    works without injecting fake seasonality.
    """
    if timeenc == 0:
        return np.zeros((length, 4), dtype=np.float32)

    dummy_dates = pd.date_range("2000-01-02", periods=2, freq="W")
    mark_dim = time_features(pd.to_datetime(dummy_dates.values), freq=freq).shape[0]
    return np.zeros((length, mark_dim), dtype=np.float32)


def _build_daily_cyclical_marks(start_timestamp, length):
    start = pd.Timestamp(start_timestamp).normalize()
    dates = pd.date_range(start=start, periods=length, freq="D")
    day_of_week = dates.dayofweek.to_numpy(dtype=np.float32)
    month_of_year = dates.month.to_numpy(dtype=np.float32)
    two_pi = np.float32(2.0 * np.pi)

    return np.column_stack(
        [
            np.sin(two_pi * day_of_week / 7.0),
            np.cos(two_pi * day_of_week / 7.0),
            np.sin(two_pi * month_of_year / 12.0),
            np.cos(two_pi * month_of_year / 12.0),
        ]
    ).astype(np.float32)


def _build_monthly_cyclical_marks(start_timestamp, length):
    start = pd.Timestamp(start_timestamp).tz_localize(None).normalize()
    dates = pd.date_range(start=start, periods=length, freq="MS")
    month_of_year = dates.month.to_numpy(dtype=np.float32)
    two_pi = np.float32(2.0 * np.pi)

    return np.column_stack(
        [
            np.sin(two_pi * month_of_year / 12.0),
            np.cos(two_pi * month_of_year / 12.0),
        ]
    ).astype(np.float32)


def _read_table_file(filepath):
    suffix = os.path.splitext(filepath)[1].lower()
    if suffix == ".csv":
        return pd.read_csv(filepath)
    if suffix == ".tsv":
        return pd.read_csv(filepath, sep="\t")
    if suffix in {".parquet", ".parq", ".pqt"}:
        return pd.read_parquet(filepath)
    if suffix in {".feather", ".ftr"}:
        return pd.read_feather(filepath)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(filepath)
    raise ValueError(
        f"Unsupported Dominick panel file format: {filepath}. "
        "Expected one of .csv, .tsv, .parquet, .feather, .pkl, or .pickle."
    )


def _resolve_column_name(columns, aliases, description, required=True):
    lowered = {str(col).lower(): col for col in columns}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    if required:
        raise ValueError(
            f"Missing required Dominick column for {description}. "
            f"Tried aliases: {aliases}"
        )
    return None


def _binarize_promo_flag(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(np.float32)
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
        return (numeric > 0).astype(np.float32)

    normalized = series.fillna("").astype(str).str.strip().str.lower()
    negative_tokens = {"", "0", "0.0", "false", "f", "no", "n", "none", "null", "nan"}
    return (~normalized.isin(negative_tokens)).astype(np.float32)


def _normalize_week_values(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.to_numpy(dtype=np.float64), "numeric"

    datetimes = pd.to_datetime(series, errors="coerce")
    if datetimes.notna().all():
        return datetimes.to_numpy(), "datetime"

    return None, None


def _is_regular_weekly(values, value_type):
    if len(values) < 2:
        return True

    if value_type == "numeric":
        diffs = np.diff(values)
        return np.all(diffs > 0) and np.allclose(diffs, diffs[0])

    diffs = np.diff(values).astype("timedelta64[D]").astype(np.int64)
    return np.all(diffs == 7)


class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        local_fp = os.path.join(self.root_path, self.data_path)
        cfg_name = os.path.splitext(os.path.basename(self.data_path))[0]

        if os.path.exists(local_fp):
            df_raw = pd.read_csv(local_fp)
        else:
            _require_optional_dependency(load_dataset, "datasets", "ETT hourly Hugging Face fallback")
            ds = load_dataset(HUGGINGFACE_REPO, name=cfg_name)
            df_raw = ds["train"].to_pandas()
            
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        
        local_fp = os.path.join(self.root_path, self.data_path)
        cfg_name = os.path.splitext(os.path.basename(self.data_path))[0]

        if os.path.exists(local_fp):
            df_raw = pd.read_csv(local_fp)
        else:
            _require_optional_dependency(load_dataset, "datasets", "ETT minute Hugging Face fallback")
            ds = load_dataset(HUGGINGFACE_REPO, name=cfg_name)
            df_raw = ds["train"].to_pandas()

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        local_fp = os.path.join(self.root_path, self.data_path)
        cfg_name = os.path.splitext(os.path.basename(self.data_path))[0]

        if os.path.exists(local_fp):
            df_raw = pd.read_csv(local_fp)
        else:
            _require_optional_dependency(load_dataset, "datasets", "custom forecast Hugging Face fallback")
            ds = load_dataset(HUGGINGFACE_REPO, name=cfg_name)
            split_name = "train" if "train" in ds else list(ds.keys())[0]
            df_raw = ds[split_name].to_pandas()

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, args, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        # size [seq_len, label_len, pred_len]
        # init
        _require_optional_dependency(M4Meta, "patoolib/data_provider.m4 dependencies", "M4 metadata access")
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        _require_optional_dependency(M4Dataset, "patoolib/data_provider.m4 dependencies", "M4 loading")
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = [
            np.asarray(v[~np.isnan(v)], dtype=np.float32)
            for v in dataset.values[dataset.groups == self.seasonal_patterns]
        ]  # split different frequencies while preserving variable-length histories
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = training_values

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           max(0, cut_point - self.label_len):min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class Dataset_DominickPanel(Dataset):
    """Fixed-window Dominick short-term forecasting dataset.

    Each `(store_id, sku_id)` pair is treated as one weekly multivariate time
    series with features `[sales, price, margin, promo]`. The dataset returns
    `(seq_x, seq_y, seq_x_mark, seq_y_mark)` to match the repo's forecasting
    interface, while the mark tensors are zero placeholders used only for
    interface compatibility.
    """

    SUPPORTED_EXTENSIONS = (".csv", ".tsv", ".parquet", ".parq", ".pqt", ".feather", ".ftr", ".pkl", ".pickle")
    FEATURE_NAMES = ["sales", "price", "margin", "promo"]
    CONTINUOUS_FEATURES = ["sales", "price", "margin"]
    TARGET_FEATURE = "sales"
    TARGET_FEATURE_INDEX = 0

    STORE_ALIASES = ["store_id", "store", "store_nbr", "store_num", "store_code"]
    SKU_ALIASES = ["sku_id", "sku", "upc", "upc_id", "item_id", "product_id"]
    WEEK_ALIASES = ["week", "week_id", "week_num", "week_start", "week_end", "date"]
    SALES_ALIASES = ["sales", "unit_sales", "sales_units", "movement", "quantity", "qty"]
    PRICE_ALIASES = ["price", "unit_price", "retail_price", "avg_price"]
    MARGIN_ALIASES = ["margin", "gross_margin", "unit_margin"]
    PROMO_ALIASES = ["promo", "promotion", "promoted", "promo_flag", "promotion_flag", "deal_flag"]
    DEAL_CODE_ALIASES = ["deal_code", "deal", "promo_code"]

    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=None,
        features="MS",
        data_path="dominick.csv",
        target="sales",
        scale=True,
        timeenc=0,
        freq="w",
        seasonal_patterns=None,
    ):
        self.args = args
        if size is None:
            self.seq_len = 16
            self.label_len = 8
            self.pred_len = 8
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ["train", "val", "test"]
        self.flag = flag
        self.features = features
        self.target = target or self.TARGET_FEATURE
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq or "w"
        self.root_path = root_path
        self.data_path = data_path

        self.target_feature_index = self.TARGET_FEATURE_INDEX
        self.feature_names = list(self.FEATURE_NAMES)
        self.use_time_marks = False
        self.series_keys = []
        self.series_values = []
        self.series_start_offsets = []
        self.series_window_counts = []
        self.cumulative_windows = np.array([], dtype=np.int64)
        self.seq_x_mark_template = np.zeros((self.seq_len, 1), dtype=np.float32)
        self.seq_y_mark_template = np.zeros((self.pred_len, 1), dtype=np.float32)
        self.feature_scalers = {name: StandardScaler() for name in self.CONTINUOUS_FEATURES}
        self.sales_scaler = self.feature_scalers[self.TARGET_FEATURE]

        self.__read_data__()

    def _resolve_panel_path(self):
        candidates = []
        if os.path.isfile(self.root_path):
            candidates.append(self.root_path)

        if self.data_path:
            candidates.append(os.path.join(self.root_path, self.data_path))

        if os.path.isdir(self.root_path):
            candidates.extend(
                os.path.join(self.root_path, name)
                for name in [
                    "dominick.csv",
                    "dominick_panel.csv",
                    "dominik.csv",
                    "dominik_panel.csv",
                    "dominick.parquet",
                    "dominick_panel.parquet",
                ]
            )
            for extension in self.SUPPORTED_EXTENSIONS:
                candidates.extend(sorted(glob.glob(os.path.join(self.root_path, f"*{extension}"))))

        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate

        tsf_hint = os.path.join(self.root_path, "dominick_dataset.tsf") if os.path.isdir(self.root_path) else self.root_path
        raise FileNotFoundError(
            "Dominick panel file not found. Expected a raw weekly table with columns like "
            "`store_id`, `sku_id`, `week`, `sales`, `price`, `margin`, and `promo`. "
            f"The checked-in TSF file at `{tsf_hint}` is not enough for the short-term panel spec."
        )

    def _load_panel_frame(self):
        panel_path = self._resolve_panel_path()
        df_raw = _read_table_file(panel_path)

        store_col = _resolve_column_name(df_raw.columns, self.STORE_ALIASES, "store id")
        sku_col = _resolve_column_name(df_raw.columns, self.SKU_ALIASES, "sku id")
        week_col = _resolve_column_name(df_raw.columns, self.WEEK_ALIASES, "week")
        sales_col = _resolve_column_name(df_raw.columns, self.SALES_ALIASES, "sales")
        price_col = _resolve_column_name(df_raw.columns, self.PRICE_ALIASES, "price")
        margin_col = _resolve_column_name(df_raw.columns, self.MARGIN_ALIASES, "margin")
        promo_col = _resolve_column_name(df_raw.columns, self.PROMO_ALIASES, "promotion flag", required=False)
        deal_code_col = _resolve_column_name(df_raw.columns, self.DEAL_CODE_ALIASES, "deal code", required=False)

        promo_series = None
        if promo_col is not None:
            promo_series = _binarize_promo_flag(df_raw[promo_col])
        elif deal_code_col is not None:
            promo_series = _binarize_promo_flag(df_raw[deal_code_col])
        else:
            promo_series = pd.Series(np.zeros(len(df_raw), dtype=np.float32), index=df_raw.index)

        panel_df = pd.DataFrame(
            {
                "store_id": df_raw[store_col].astype(str),
                "sku_id": df_raw[sku_col].astype(str),
                "week": df_raw[week_col],
                "sales": pd.to_numeric(df_raw[sales_col], errors="coerce"),
                "price": pd.to_numeric(df_raw[price_col], errors="coerce"),
                "margin": pd.to_numeric(df_raw[margin_col], errors="coerce"),
                "promo": promo_series.astype(np.float32),
            }
        )
        panel_df = panel_df.dropna(subset=["store_id", "sku_id", "week", "sales", "price", "margin"])
        return panel_df

    def _split_bounds(self, series_length):
        max_start = series_length - self.seq_len - self.pred_len
        if max_start < 0:
            return None

        train_end = max(self.seq_len + self.pred_len, int(np.floor(series_length * 0.7)))
        val_end = max(train_end, int(np.floor(series_length * 0.8)))
        train_end = min(train_end, series_length)
        val_end = min(max(val_end, train_end), series_length)

        split_start = {"train": 0, "val": max(0, train_end - self.seq_len), "test": max(0, val_end - self.seq_len)}
        split_end = {
            "train": train_end - self.seq_len - self.pred_len,
            "val": val_end - self.seq_len - self.pred_len,
            "test": series_length - self.seq_len - self.pred_len,
        }

        start = split_start[self.flag]
        end = split_end[self.flag]
        if end < start:
            return None
        return int(start), int(end - start + 1), int(train_end)

    def _apply_feature_scaling(self, values):
        if not self.scale:
            return values.astype(np.float32)

        scaled = values.astype(np.float32).copy()
        for feature_name in self.CONTINUOUS_FEATURES:
            feature_idx = self.feature_names.index(feature_name)
            scaler = self.feature_scalers[feature_name]
            scaled[:, feature_idx] = scaler.transform(values[:, feature_idx:feature_idx + 1]).reshape(-1)
        return scaled

    def __read_data__(self):
        panel_df = self._load_panel_frame()

        usable_groups = []
        training_rows = []
        min_series_points = max(self.seq_len + self.pred_len, 2 * (self.seq_len + self.pred_len))

        for (store_id, sku_id), group_df in panel_df.groupby(["store_id", "sku_id"], sort=False):
            aggregated = (
                group_df.groupby("week", as_index=False)
                .agg({"sales": "sum", "price": "mean", "margin": "mean", "promo": "max"})
            )

            normalized_week, week_type = _normalize_week_values(aggregated["week"])
            if normalized_week is None:
                continue

            order = np.argsort(normalized_week)
            aggregated = aggregated.iloc[order].reset_index(drop=True)
            normalized_week = normalized_week[order]

            if not _is_regular_weekly(normalized_week, week_type):
                continue

            series_length = len(aggregated)
            if series_length < min_series_points:
                continue

            split_bounds = self._split_bounds(series_length)
            if split_bounds is None:
                continue

            split_start, split_count, train_end = split_bounds
            if split_count <= 0:
                continue

            values = aggregated[self.feature_names].to_numpy(dtype=np.float32)
            training_rows.append(values[:train_end, : len(self.CONTINUOUS_FEATURES)])
            usable_groups.append(((store_id, sku_id), values, split_start, split_count))

        if not usable_groups:
            raise ValueError(
                "No Dominick series matched the short-term panel requirements. "
                "Check that the file has regular weekly rows and enough history for "
                f"`seq_len={self.seq_len}` and `pred_len={self.pred_len}`."
            )

        if self.scale and training_rows:
            stacked_train = np.concatenate(training_rows, axis=0)
            for feature_idx, feature_name in enumerate(self.CONTINUOUS_FEATURES):
                self.feature_scalers[feature_name].fit(stacked_train[:, feature_idx:feature_idx + 1])

        cumulative = []
        running_total = 0
        for (store_id, sku_id), values, split_start, split_count in usable_groups:
            scaled_values = self._apply_feature_scaling(values)
            self.series_keys.append((store_id, sku_id))
            self.series_values.append(scaled_values)
            self.series_start_offsets.append(int(split_start))
            self.series_window_counts.append(int(split_count))
            running_total += int(split_count)
            cumulative.append(running_total)

        self.cumulative_windows = np.asarray(cumulative, dtype=np.int64)

    def __getitem__(self, index):
        series_idx = bisect_right(self.cumulative_windows, index)
        prev_total = 0 if series_idx == 0 else int(self.cumulative_windows[series_idx - 1])
        local_window_offset = index - prev_total

        start = self.series_start_offsets[series_idx] + local_window_offset
        end = start + self.seq_len
        target_start = end
        target_end = target_start + self.pred_len

        series = self.series_values[series_idx]
        seq_x = series[start:end]
        seq_y = series[target_start:target_end, self.target_feature_index:self.target_feature_index + 1]
        seq_x_mark = self.seq_x_mark_template
        seq_y_mark = self.seq_y_mark_template
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.cumulative_windows.size == 0:
            return 0
        return int(self.cumulative_windows[-1])

    def inverse_transform(self, data):
        if not self.scale:
            return data
        return self.sales_scaler.inverse_transform(data)


class Dataset_DominickTSF(Dataset):
    """Weekly retail panel forecast dataset backed by the local Dominick TSF file.

    The checked-in TSF file contains one weekly value sequence per `series_name`
    and does not expose the richer exogenous covariates from the raw retail
    tables yet. This class therefore implements a first-pass univariate panel
    forecast dataset that still matches the repo's standard forecast contract:
    `(seq_x, seq_y, seq_x_mark, seq_y_mark)`.
    """

    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="dominick_dataset.tsf",
        target="value",
        scale=True,
        timeenc=0,
        freq="w",
        seasonal_patterns=None,
    ):
        self.args = args
        if size is None:
            self.seq_len = 52
            self.label_len = 26
            self.pred_len = 13
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ["train", "val", "test"]
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq or "w"
        self.root_path = root_path
        self.data_path = data_path
        self.use_time_marks = False

        self.series_ids = []
        self.series_values = []
        self.series_start_offsets = []
        self.series_window_counts = []
        self.cumulative_windows = np.array([], dtype=np.int64)
        self.seq_x_mark_template = _default_time_mark(self.seq_len, self.timeenc, self.freq)
        self.seq_y_mark_template = _default_time_mark(self.label_len + self.pred_len, self.timeenc, self.freq)

        self.__read_data__()

    def _resolve_tsf_path(self):
        candidates = []
        if os.path.isfile(self.root_path):
            candidates.append(self.root_path)
        if self.data_path:
            candidates.append(os.path.join(self.root_path, self.data_path))
        if os.path.isdir(self.root_path):
            candidates.append(os.path.join(self.root_path, "dominick_dataset.tsf"))
            candidates.extend(sorted(glob.glob(os.path.join(self.root_path, "*.tsf"))))

        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate

        raise FileNotFoundError(
            "Dominick TSF file not found. Pass either "
            "`--root_path` pointing to the `.tsf` file or a directory containing it, "
            "and optionally `--data_path dominick_dataset.tsf`."
        )

    def _window_split_bounds(self, window_count):
        if window_count < 3:
            return None

        train_count = max(1, int(window_count * 0.7))
        val_count = max(1, int(window_count * 0.1))
        if train_count + val_count >= window_count:
            train_count = max(1, window_count - 2)
            val_count = 1
        test_count = window_count - train_count - val_count
        if test_count < 1:
            return None

        split_start = {
            "train": 0,
            "val": train_count,
            "test": train_count + val_count,
        }
        split_count = {
            "train": train_count,
            "val": val_count,
            "test": test_count,
        }
        return split_start[self.flag], split_count[self.flag]

    def __read_data__(self):
        tsf_path = self._resolve_tsf_path()
        raw_ids, raw_series = _parse_tsf_series(tsf_path)

        self.scaler = StandardScaler()
        train_values = []
        usable_rows = []
        min_points = self.seq_len + self.pred_len

        for series_id, series in zip(raw_ids, raw_series):
            if series.size < min_points:
                continue

            window_count = int(series.size - self.seq_len - self.pred_len + 1)
            bounds = self._window_split_bounds(window_count)
            if bounds is None:
                continue

            train_cutoff = max(self.seq_len, int(series.size * 0.7))
            train_values.append(series[:train_cutoff].reshape(-1, 1))
            usable_rows.append((series_id, series.astype(np.float32), bounds))

        if not usable_rows:
            raise ValueError(
                "No Dominick series were long enough for the configured "
                f"`seq_len={self.seq_len}` and `pred_len={self.pred_len}`."
            )

        if self.scale and train_values:
            self.scaler.fit(np.concatenate(train_values, axis=0))

        cumulative = []
        running_total = 0
        for series_id, series, (split_start, split_count) in usable_rows:
            if self.scale:
                scaled_series = self.scaler.transform(series.reshape(-1, 1)).reshape(-1)
            else:
                scaled_series = series

            self.series_ids.append(series_id)
            self.series_values.append(scaled_series.astype(np.float32))
            self.series_start_offsets.append(int(split_start))
            self.series_window_counts.append(int(split_count))

            running_total += int(split_count)
            cumulative.append(running_total)

        self.cumulative_windows = np.asarray(cumulative, dtype=np.int64)

    def __getitem__(self, index):
        series_idx = bisect_right(self.cumulative_windows, index)
        prev_total = 0 if series_idx == 0 else int(self.cumulative_windows[series_idx - 1])
        local_window_offset = index - prev_total

        start = self.series_start_offsets[series_idx] + local_window_offset
        end = start + self.seq_len
        target_start = end - self.label_len
        target_end = target_start + self.label_len + self.pred_len

        series = self.series_values[series_idx]

        seq_x = series[start:end].reshape(-1, 1)
        seq_y = series[target_start:target_end].reshape(-1, 1)
        seq_x_mark = self.seq_x_mark_template
        seq_y_mark = self.seq_y_mark_template
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.cumulative_windows.size == 0:
            return 0
        return int(self.cumulative_windows[-1])

    def inverse_transform(self, data):
        if not self.scale:
            return data
        return self.scaler.inverse_transform(data)


class Dataset_MonashTSFGeneric(Dataset):
    """Generic fixed-window loader for Monash/forecastingdata TSF datasets.

    This first-pass integration treats each TSF row as one univariate series and
    maps it to the repo's `short_term_forecast` contract:
    `(seq_x, seq_y, seq_x_mark, seq_y_mark)`.
    """

    _CACHE = {}

    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="",
        target="value",
        scale=True,
        timeenc=0,
        freq="generic",
        seasonal_patterns=None,
    ):
        self.args = args
        dataset_key = getattr(args, "data", "")
        if dataset_key not in MONASH_GENERIC_DATASETS:
            raise ValueError(
                f"`Dataset_MonashTSFGeneric` does not know how to configure dataset key `{dataset_key}`."
            )
        self.config = MONASH_GENERIC_DATASETS[dataset_key]

        default_seq_len, default_label_len, default_pred_len = self.config["default_size"]
        if size is None:
            self.seq_len = default_seq_len
            self.label_len = default_label_len
            self.pred_len = default_pred_len
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ["train", "val", "test"]
        self.flag = flag
        self.features = features
        self.target = target if target != "OT" else self.config.get("target", "value")
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq or "generic"
        self.root_path = root_path
        self.data_path = data_path
        self.use_time_marks = False

        self.feature_names = [self.target]
        self.mark_feature_names = ["compat_mark"]
        self.target_feature_index = 0
        self.series_ids = []
        self.series_values = []
        self.series_start_offsets = []
        self.series_window_counts = []
        self.cumulative_windows = np.array([], dtype=np.int64)
        self.metadata = {}
        self.scaler = StandardScaler()
        self.seq_x_mark_template = np.zeros((self.seq_len, GENERIC_TS_ZERO_MARK_DIM), dtype=np.float32)
        self.seq_y_mark_template = np.zeros((self.pred_len, GENERIC_TS_ZERO_MARK_DIM), dtype=np.float32)

        self.__read_data__()

    def _search_local_candidates(self, base_dir):
        tokens = [token.lower() for token in self.config.get("search_tokens", [])]
        explicit_candidates = []
        if self.data_path:
            explicit_candidates.append(os.path.join(base_dir, self.data_path))

        for candidate in explicit_candidates:
            if candidate and os.path.exists(candidate):
                return os.path.abspath(candidate)

        recursive_candidates = []
        for extension in ("*.tsf", "*.csv", "*.tsv", "*.parquet", "*.feather", "*.pkl", "*.pickle"):
            recursive_candidates.extend(glob.glob(os.path.join(base_dir, "**", extension), recursive=True))

        for candidate in sorted(recursive_candidates):
            lowered = os.path.basename(candidate).lower()
            if all(token in lowered for token in tokens):
                return os.path.abspath(candidate)

        if len(recursive_candidates) == 1:
            return os.path.abspath(recursive_candidates[0])
        return None

    def _resolve_dataset_path(self):
        if os.path.isfile(self.root_path):
            return os.path.abspath(self.root_path)

        if self.data_path:
            explicit_path = os.path.join(self.root_path, self.data_path)
            if os.path.exists(explicit_path):
                return os.path.abspath(explicit_path)

        search_root = self.root_path if os.path.isdir(self.root_path) else os.path.dirname(self.root_path) or "."
        local_match = self._search_local_candidates(search_root)
        if local_match is not None:
            return local_match

        artifact_root = _ensure_monash_artifact(search_root, getattr(self.args, "data", "monash_dataset"), self.config)
        downloaded_match = self._search_local_candidates(artifact_root)
        if downloaded_match is not None:
            return downloaded_match

        raise FileNotFoundError(
            f"Could not resolve a local file for dataset key `{getattr(self.args, 'data', '')}` "
            f"under root_path={self.root_path} data_path={self.data_path}."
        )

    @classmethod
    def _load_cached_source(cls, filepath):
        filepath = os.path.abspath(filepath)
        if filepath in cls._CACHE:
            return cls._CACHE[filepath]

        if filepath.lower().endswith(".tsf"):
            metadata, attribute_names = _read_tsf_header(filepath)
            series_ids = []
            series_values = []
            for record in _iter_tsf_records(filepath, attribute_names):
                series_name = record.get("series_name", f"series_{len(series_ids)}")
                values = np.asarray(record["values"], dtype=np.float32)
                values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
                if values.size == 0:
                    continue
                series_ids.append(series_name)
                series_values.append(values)
        else:
            frame = _read_table_file(filepath)
            metadata = {"frequency": "unknown", "relation": os.path.basename(filepath)}
            series_ids = list(frame.columns[1:])
            series_values = [
                pd.to_numeric(frame[column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
                for column in frame.columns[1:]
            ]

        if not series_values:
            raise ValueError(f"No usable time series found in `{filepath}`.")

        cls._CACHE[filepath] = {
            "metadata": metadata,
            "series_ids": series_ids,
            "series_values": series_values,
        }
        return cls._CACHE[filepath]

    def _split_bounds(self, series_length):
        max_start = series_length - self.seq_len - self.pred_len
        if max_start < 0:
            return None

        train_end = max(self.seq_len + self.pred_len, int(np.floor(series_length * 0.7)))
        val_end = max(train_end, int(np.floor(series_length * 0.8)))
        train_end = min(train_end, series_length)
        val_end = min(max(val_end, train_end), series_length)

        split_start = {
            "train": 0,
            "val": max(0, train_end - self.seq_len),
            "test": max(0, val_end - self.seq_len),
        }
        split_end = {
            "train": train_end - self.seq_len - self.pred_len,
            "val": val_end - self.seq_len - self.pred_len,
            "test": series_length - self.seq_len - self.pred_len,
        }

        start = split_start[self.flag]
        end = split_end[self.flag]
        if end < start:
            return None
        return int(start), int(end - start + 1), int(train_end)

    def __read_data__(self):
        resolved_path = self._resolve_dataset_path()
        cached = self._load_cached_source(resolved_path)
        self.metadata = cached["metadata"]

        usable_rows = []
        train_values = []
        min_points = self.seq_len + self.pred_len

        for series_id, values in zip(cached["series_ids"], cached["series_values"]):
            values = np.asarray(values, dtype=np.float32)
            if values.size < min_points:
                continue

            split_bounds = self._split_bounds(len(values))
            if split_bounds is None:
                continue

            split_start, split_count, train_end = split_bounds
            if split_count <= 0:
                continue

            train_values.append(values[:train_end].reshape(-1, 1))
            usable_rows.append((series_id, values, split_start, split_count))

        if not usable_rows:
            raise ValueError(
                f"No series in `{resolved_path}` were long enough for seq_len={self.seq_len}, pred_len={self.pred_len}."
            )

        if self.scale and train_values:
            self.scaler.fit(np.concatenate(train_values, axis=0))

        cumulative = []
        running_total = 0
        for series_id, values, split_start, split_count in usable_rows:
            if self.scale:
                values = self.scaler.transform(values.reshape(-1, 1)).reshape(-1)

            self.series_ids.append(series_id)
            self.series_values.append(values.astype(np.float32))
            self.series_start_offsets.append(int(split_start))
            self.series_window_counts.append(int(split_count))

            running_total += int(split_count)
            cumulative.append(running_total)

        self.cumulative_windows = np.asarray(cumulative, dtype=np.int64)

    def __getitem__(self, index):
        series_idx = bisect_right(self.cumulative_windows, index)
        prev_total = 0 if series_idx == 0 else int(self.cumulative_windows[series_idx - 1])
        local_window_offset = index - prev_total

        start = self.series_start_offsets[series_idx] + local_window_offset
        end = start + self.seq_len
        target_start = end
        target_end = target_start + self.pred_len

        series = self.series_values[series_idx]
        seq_x = series[start:end].reshape(-1, 1)
        seq_y = series[target_start:target_end].reshape(-1, 1)
        seq_x_mark = self.seq_x_mark_template.copy()
        seq_y_mark = self.seq_y_mark_template.copy()
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.cumulative_windows.size == 0:
            return 0
        return int(self.cumulative_windows[-1])

    def inverse_transform(self, data):
        if not self.scale:
            return data
        return self.scaler.inverse_transform(data)


class Dataset_TourismMonthlyTSF(Dataset):
    """Monthly Tourism competition dataset for fixed-window short-term forecasting.

    Each TSF row is one monthly univariate tourism series. The loader builds
    temporal sliding windows and returns calendar marks based on month-of-year
    sine/cosine features, keeping the same `(seq_x, seq_y, seq_x_mark,
    seq_y_mark)` contract used by the fixed-window short-term pipeline.
    """

    _CACHE = {}
    MARK_FEATURE_NAMES = ["month_sin", "month_cos"]

    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="tourism_monthly_dataset.tsf",
        target="tourism",
        scale=True,
        timeenc=1,
        freq="tourism_monthly",
        seasonal_patterns=None,
    ):
        self.args = args
        if size is None:
            self.seq_len = 48
            self.label_len = 24
            self.pred_len = 24
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ["train", "val", "test"]
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq or "tourism_monthly"
        self.root_path = root_path
        self.data_path = data_path
        self.use_time_marks = True

        self.feature_names = [self.target]
        self.mark_feature_names = list(self.MARK_FEATURE_NAMES)
        self.target_feature_index = 0
        self.series_ids = []
        self.series_values = []
        self.series_marks = []
        self.series_start_offsets = []
        self.series_window_counts = []
        self.cumulative_windows = np.array([], dtype=np.int64)
        self.num_series = 0
        self.scaler = StandardScaler()

        self.__read_data__()

    def _resolve_tsf_path(self):
        candidates = []
        if os.path.isfile(self.root_path):
            candidates.append(self.root_path)
        if self.data_path:
            candidates.append(os.path.join(self.root_path, self.data_path))
        if os.path.isdir(self.root_path):
            candidates.extend(
                os.path.join(self.root_path, name)
                for name in [
                    "tourism_monthly_dataset.tsf",
                    "tourism_monthly.tsf",
                    "tourism_dataset.tsf",
                    "tourism.tsf",
                ]
            )
            candidates.extend(sorted(glob.glob(os.path.join(self.root_path, "*tourism*monthly*.tsf"))))
            candidates.extend(sorted(glob.glob(os.path.join(self.root_path, "*tourism*.tsf"))))

        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return os.path.abspath(candidate)

        raise FileNotFoundError(
            "Tourism monthly TSF file not found. Expected a file like "
            "`tourism_monthly_dataset.tsf` under `--root_path`."
        )

    @classmethod
    def _load_cached_source(cls, filepath):
        filepath = os.path.abspath(filepath)
        if filepath in cls._CACHE:
            return cls._CACHE[filepath]

        metadata, attribute_names = _read_tsf_header(filepath)
        if "start_timestamp" not in attribute_names:
            raise ValueError(
                "Tourism monthly TSF file is missing the `start_timestamp` attribute "
                "needed to construct month-of-year marks."
            )

        series_ids = []
        series_values = []
        series_marks = []

        for record in _iter_tsf_records(filepath, attribute_names):
            values = np.nan_to_num(record["values"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            if values.size == 0:
                continue

            series_name = record.get("series_name", f"series_{len(series_ids)}")
            start_timestamp = pd.to_datetime(record["start_timestamp"], errors="coerce")
            if pd.isna(start_timestamp):
                raise ValueError(
                    f"Invalid `start_timestamp` for tourism series `{series_name}`: "
                    f"{record['start_timestamp']}"
                )

            series_ids.append(series_name)
            series_values.append(values)
            series_marks.append(_build_monthly_cyclical_marks(start_timestamp, len(values)))

        if not series_ids:
            raise ValueError(f"No usable tourism monthly series found in TSF file: {filepath}")

        cls._CACHE[filepath] = {
            "metadata": metadata,
            "series_ids": series_ids,
            "series_values": series_values,
            "series_marks": series_marks,
        }
        return cls._CACHE[filepath]

    def _split_bounds(self, series_length):
        max_start = series_length - self.seq_len - self.pred_len
        if max_start < 0:
            return None

        train_end = max(self.seq_len + self.pred_len, int(np.floor(series_length * 0.7)))
        val_end = max(train_end, int(np.floor(series_length * 0.8)))
        train_end = min(train_end, series_length)
        val_end = min(max(val_end, train_end), series_length)

        split_start = {
            "train": 0,
            "val": max(0, train_end - self.seq_len),
            "test": max(0, val_end - self.seq_len),
        }
        split_end = {
            "train": train_end - self.seq_len - self.pred_len,
            "val": val_end - self.seq_len - self.pred_len,
            "test": series_length - self.seq_len - self.pred_len,
        }

        start = split_start[self.flag]
        end = split_end[self.flag]
        if end < start:
            return None
        return int(start), int(end - start + 1), int(train_end)

    def __read_data__(self):
        tsf_path = self._resolve_tsf_path()
        cached = self._load_cached_source(tsf_path)

        usable_rows = []
        train_values = []
        min_points = self.seq_len + self.pred_len

        for series_id, values, marks in zip(
            cached["series_ids"],
            cached["series_values"],
            cached["series_marks"],
        ):
            if len(values) < min_points:
                continue

            split_bounds = self._split_bounds(len(values))
            if split_bounds is None:
                continue

            split_start, split_count, train_end = split_bounds
            if split_count <= 0:
                continue

            train_values.append(values[:train_end].reshape(-1, 1))
            usable_rows.append((series_id, values, marks, split_start, split_count))

        if not usable_rows:
            raise ValueError(
                "No tourism monthly series were long enough for the configured "
                f"`seq_len={self.seq_len}` and `pred_len={self.pred_len}`."
            )

        if self.scale and train_values:
            self.scaler.fit(np.concatenate(train_values, axis=0))

        cumulative = []
        running_total = 0
        for series_id, values, marks, split_start, split_count in usable_rows:
            if self.scale:
                values = self.scaler.transform(values.reshape(-1, 1)).reshape(-1)

            self.series_ids.append(series_id)
            self.series_values.append(values.astype(np.float32))
            self.series_marks.append(marks.astype(np.float32))
            self.series_start_offsets.append(int(split_start))
            self.series_window_counts.append(int(split_count))

            running_total += int(split_count)
            cumulative.append(running_total)

        self.cumulative_windows = np.asarray(cumulative, dtype=np.int64)
        self.num_series = len(self.series_ids)

    def __getitem__(self, index):
        series_idx = bisect_right(self.cumulative_windows, index)
        prev_total = 0 if series_idx == 0 else int(self.cumulative_windows[series_idx - 1])
        local_window_offset = index - prev_total

        start = self.series_start_offsets[series_idx] + local_window_offset
        end = start + self.seq_len
        target_start = end
        target_end = target_start + self.pred_len

        series = self.series_values[series_idx]
        marks = self.series_marks[series_idx]

        seq_x = series[start:end].reshape(-1, 1)
        seq_y = series[target_start:target_end].reshape(-1, 1)
        seq_x_mark = marks[start:end]
        seq_y_mark = marks[target_start:target_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.cumulative_windows.size == 0:
            return 0
        return int(self.cumulative_windows[-1])

    def inverse_transform(self, data):
        if not self.scale:
            return data
        return self.scaler.inverse_transform(data)


class Dataset_NN5DailyTSF(Dataset):
    """Daily NN5 ATM cash-withdrawal dataset for short-term forecasting.

    Each TSF row is one ATM cash-withdrawal series. We construct fixed daily
    sliding windows and use real daily calendar marks because ATM usage usually
    has strong day-of-week effects.
    """

    _CACHE = {}
    MARK_FEATURE_NAMES = ["dow_sin", "dow_cos", "month_sin", "month_cos"]

    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="nn5_daily_dataset_without_missing_values.tsf",
        target="cash",
        scale=True,
        timeenc=1,
        freq="nn5_daily",
        seasonal_patterns=None,
    ):
        self.args = args
        if size is None:
            self.seq_len = 112
            self.label_len = 56
            self.pred_len = 56
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ["train", "val", "test"]
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq or "nn5_daily"
        self.root_path = root_path
        self.data_path = data_path
        self.use_time_marks = True

        self.feature_names = [self.target]
        self.mark_feature_names = list(self.MARK_FEATURE_NAMES)
        self.target_feature_index = 0
        self.series_ids = []
        self.series_start_keys = np.array([], dtype=object)
        self.series_values = np.empty((0, 0), dtype=np.float32)
        self.series_length = 0
        self.window_start = 0
        self.window_count = 0
        self.num_series = 0
        self.scaler = StandardScaler()
        self._marks_by_start = {}

        self.__read_data__()

    def _resolve_tsf_path(self):
        candidates = []
        if os.path.isfile(self.root_path):
            candidates.append(self.root_path)
        if self.data_path:
            candidates.append(os.path.join(self.root_path, self.data_path))
        if os.path.isdir(self.root_path):
            candidates.extend(
                os.path.join(self.root_path, name)
                for name in [
                    "nn5_daily_dataset_without_missing_values.tsf",
                    "nn5_daily_dataset.tsf",
                    "nn5_daily.tsf",
                    "nn5.tsf",
                ]
            )
            candidates.extend(sorted(glob.glob(os.path.join(self.root_path, "*nn5*daily*.tsf"))))
            candidates.extend(sorted(glob.glob(os.path.join(self.root_path, "*nn5*.tsf"))))

        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return os.path.abspath(candidate)

        raise FileNotFoundError(
            "NN5 daily TSF file not found. Expected a file like "
            "`nn5_daily_dataset_without_missing_values.tsf` under `--root_path`."
        )

    @classmethod
    def _load_cached_source(cls, filepath):
        filepath = os.path.abspath(filepath)
        if filepath in cls._CACHE:
            return cls._CACHE[filepath]

        metadata, attribute_names = _read_tsf_header(filepath)
        if "start_timestamp" not in attribute_names:
            raise ValueError(
                "NN5 daily TSF file is missing the `start_timestamp` attribute "
                "needed to construct daily calendar marks."
            )

        series_ids = []
        series_start_keys = []
        series_length = None

        for record in _iter_tsf_records(filepath, attribute_names):
            values = np.nan_to_num(record["values"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            current_length = int(values.size)
            if current_length == 0:
                continue
            if series_length is None:
                series_length = current_length
            elif current_length != series_length:
                raise ValueError(
                    "NN5 daily integration expects equal-length ATM series, "
                    f"but found lengths {series_length} and {current_length}."
                )

            series_name = record.get("series_name", f"series_{len(series_ids)}")
            start_timestamp = pd.to_datetime(record["start_timestamp"], errors="coerce")
            if pd.isna(start_timestamp):
                raise ValueError(
                    f"Invalid `start_timestamp` for NN5 series `{series_name}`: "
                    f"{record['start_timestamp']}"
                )

            series_ids.append(series_name)
            series_start_keys.append(start_timestamp.normalize().strftime("%Y-%m-%d"))

        if not series_ids or series_length is None:
            raise ValueError(f"No usable NN5 daily series found in TSF file: {filepath}")

        series_values = np.empty((len(series_ids), series_length), dtype=np.float32)
        for row_idx, record in enumerate(_iter_tsf_records(filepath, attribute_names)):
            values = np.nan_to_num(record["values"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            series_values[row_idx] = values

        marks_by_start = {
            start_key: _build_daily_cyclical_marks(start_key, series_length)
            for start_key in sorted(set(series_start_keys))
        }

        cls._CACHE[filepath] = {
            "metadata": metadata,
            "series_ids": series_ids,
            "series_start_keys": np.asarray(series_start_keys, dtype=object),
            "series_values": series_values,
            "series_length": series_length,
            "marks_by_start": marks_by_start,
        }
        return cls._CACHE[filepath]

    def _split_bounds(self, series_length):
        max_start = series_length - self.seq_len - self.pred_len
        if max_start < 0:
            return None

        train_end = max(self.seq_len + self.pred_len, int(np.floor(series_length * 0.7)))
        val_end = max(train_end, int(np.floor(series_length * 0.8)))
        train_end = min(train_end, series_length)
        val_end = min(max(val_end, train_end), series_length)

        split_start = {
            "train": 0,
            "val": max(0, train_end - self.seq_len),
            "test": max(0, val_end - self.seq_len),
        }
        split_end = {
            "train": train_end - self.seq_len - self.pred_len,
            "val": val_end - self.seq_len - self.pred_len,
            "test": series_length - self.seq_len - self.pred_len,
        }

        start = split_start[self.flag]
        end = split_end[self.flag]
        if end < start:
            return None
        return int(start), int(end - start + 1), int(train_end)

    def __read_data__(self):
        tsf_path = self._resolve_tsf_path()
        cached = self._load_cached_source(tsf_path)

        self.series_ids = cached["series_ids"]
        self.series_start_keys = cached["series_start_keys"]
        self.series_values = cached["series_values"].astype(np.float32).copy()
        self.series_length = int(cached["series_length"])
        self._marks_by_start = cached["marks_by_start"]
        self.num_series = int(self.series_values.shape[0])

        minimum_length = self.seq_len + self.pred_len
        if self.series_length < minimum_length:
            raise ValueError(
                "NN5 daily series are too short for the configured short-term setup. "
                f"Need at least {minimum_length} points, found {self.series_length}."
            )

        split_bounds = self._split_bounds(self.series_length)
        if split_bounds is None:
            raise ValueError(
                "Could not construct temporal short-term windows for the NN5 daily dataset "
                f"with seq_len={self.seq_len} and pred_len={self.pred_len}."
            )

        self.window_start, self.window_count, train_end = split_bounds
        if self.scale:
            self.scaler.fit(self.series_values[:, :train_end].reshape(-1, 1))
            self.series_values = self.scaler.transform(self.series_values.reshape(-1, 1)).reshape(self.series_values.shape)

    def __getitem__(self, index):
        series_idx = index // self.window_count
        local_window_offset = index % self.window_count

        start = self.window_start + local_window_offset
        end = start + self.seq_len
        target_start = end
        target_end = target_start + self.pred_len

        series = self.series_values[series_idx]
        marks = self._marks_by_start[self.series_start_keys[series_idx]]

        seq_x = series[start:end].reshape(-1, 1)
        seq_y = series[target_start:target_end].reshape(-1, 1)
        seq_x_mark = marks[start:end]
        seq_y_mark = marks[target_start:target_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return int(self.num_series * self.window_count)

    def inverse_transform(self, data):
        if not self.scale:
            return data
        return self.scaler.inverse_transform(data)


class Dataset_CarPartsTSF(Dataset):
    """Multivariate monthly car-parts sales forecasting dataset.

    The TSF file contains 2674 equal-length monthly intermittent-demand series
    on the same calendar grid. Unlike the page/ATM loaders that emit one
    univariate sample per row, this loader stacks all car-part series as feature
    channels and predicts all channels together.
    """

    _CACHE = {}
    MARK_FEATURE_NAMES = ["month_sin", "month_cos"]

    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=None,
        features="M",
        data_path="car_parts_dataset_without_missing_values.tsf",
        target="sales",
        scale=True,
        timeenc=1,
        freq="car_parts_monthly",
        seasonal_patterns=None,
    ):
        self.args = args
        if size is None:
            self.seq_len = 24
            self.label_len = 12
            self.pred_len = 12
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ["train", "val", "test"]
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq or "car_parts_monthly"
        self.root_path = root_path
        self.data_path = data_path
        self.use_time_marks = True
        self.predict_all_channels = True

        self.mark_feature_names = list(self.MARK_FEATURE_NAMES)
        self.target_feature_index = 0
        self.series_ids = []
        self.feature_names = []
        self.series_values = np.empty((0, 0), dtype=np.float32)
        self.series_length = 0
        self.feature_dim = 0
        self.window_start = 0
        self.window_count = 0
        self.scaler = StandardScaler()
        self.marks = np.empty((0, len(self.MARK_FEATURE_NAMES)), dtype=np.float32)

        self.__read_data__()

    def _resolve_tsf_path(self):
        candidates = []
        if os.path.isfile(self.root_path):
            candidates.append(self.root_path)
        if self.data_path:
            candidates.append(os.path.join(self.root_path, self.data_path))
        if os.path.isdir(self.root_path):
            candidates.extend(
                os.path.join(self.root_path, name)
                for name in [
                    "car_parts_dataset_without_missing_values.tsf",
                    "car_parts_dataset.tsf",
                    "car_parts.tsf",
                ]
            )
            candidates.extend(sorted(glob.glob(os.path.join(self.root_path, "*car*parts*.tsf"))))

        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return os.path.abspath(candidate)

        raise FileNotFoundError(
            "Car Parts TSF file not found. Expected a file like "
            "`car_parts_dataset_without_missing_values.tsf` under `--root_path`."
        )

    @classmethod
    def _load_cached_source(cls, filepath):
        filepath = os.path.abspath(filepath)
        if filepath in cls._CACHE:
            return cls._CACHE[filepath]

        metadata, attribute_names = _read_tsf_header(filepath)
        if "start_timestamp" not in attribute_names:
            raise ValueError(
                "Car Parts TSF file is missing the `start_timestamp` attribute "
                "needed to construct month-of-year marks."
            )

        series_ids = []
        series_values = []
        start_keys = []
        series_length = None

        for record in _iter_tsf_records(filepath, attribute_names):
            values = np.nan_to_num(record["values"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            current_length = int(values.size)
            if current_length == 0:
                continue
            if series_length is None:
                series_length = current_length
            elif current_length != series_length:
                raise ValueError(
                    "Car Parts integration expects equal-length monthly series, "
                    f"but found lengths {series_length} and {current_length}."
                )

            series_name = record.get("series_name", f"series_{len(series_ids)}")
            start_timestamp = pd.to_datetime(record["start_timestamp"], errors="coerce")
            if pd.isna(start_timestamp):
                raise ValueError(
                    f"Invalid `start_timestamp` for car-parts series `{series_name}`: "
                    f"{record['start_timestamp']}"
                )

            series_ids.append(series_name)
            series_values.append(values)
            start_keys.append(start_timestamp.normalize().strftime("%Y-%m-%d"))

        if not series_ids or series_length is None:
            raise ValueError(f"No usable car-parts series found in TSF file: {filepath}")
        if len(set(start_keys)) != 1:
            raise ValueError(
                "Car Parts multivariate loader expects all part series to share one calendar start."
            )

        values_matrix = np.stack(series_values, axis=1).astype(np.float32)
        marks = _build_monthly_cyclical_marks(start_keys[0], series_length)

        cls._CACHE[filepath] = {
            "metadata": metadata,
            "series_ids": series_ids,
            "series_values": values_matrix,
            "series_length": series_length,
            "marks": marks,
        }
        return cls._CACHE[filepath]

    def _window_split_bounds(self, window_count):
        if window_count < 3:
            return None

        train_count = max(1, int(window_count * 0.7))
        val_count = max(1, int(window_count * 0.1))
        if train_count + val_count >= window_count:
            train_count = max(1, window_count - 2)
            val_count = 1
        test_count = window_count - train_count - val_count
        if test_count < 1:
            return None

        split_start = {
            "train": 0,
            "val": train_count,
            "test": train_count + val_count,
        }
        split_count = {
            "train": train_count,
            "val": val_count,
            "test": test_count,
        }
        return split_start[self.flag], split_count[self.flag], train_count

    def __read_data__(self):
        tsf_path = self._resolve_tsf_path()
        cached = self._load_cached_source(tsf_path)

        self.series_ids = cached["series_ids"]
        self.feature_names = list(self.series_ids)
        self.series_values = cached["series_values"].astype(np.float32).copy()
        self.series_length = int(cached["series_length"])
        self.feature_dim = int(self.series_values.shape[1])
        self.marks = cached["marks"]

        window_count = self.series_length - self.seq_len - self.pred_len + 1
        bounds = self._window_split_bounds(window_count)
        if bounds is None:
            raise ValueError(
                "Could not construct Car Parts windows. "
                f"Need at least 3 windows, got {window_count} with "
                f"seq_len={self.seq_len} and pred_len={self.pred_len}."
            )

        self.window_start, self.window_count, train_window_count = bounds
        train_end = min(self.series_length, train_window_count + self.seq_len + self.pred_len - 1)
        if self.scale:
            self.scaler.fit(self.series_values[:train_end])
            self.series_values = self.scaler.transform(self.series_values)

    def __getitem__(self, index):
        start = self.window_start + index
        end = start + self.seq_len
        target_start = end
        target_end = target_start + self.pred_len

        seq_x = self.series_values[start:end]
        seq_y = self.series_values[target_start:target_end]
        seq_x_mark = self.marks[start:end]
        seq_y_mark = self.marks[target_start:target_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return int(self.window_count)

    def inverse_transform(self, data):
        if not self.scale:
            return data

        if data.shape[-1] == self.feature_dim:
            original_shape = data.shape
            return self.scaler.inverse_transform(data.reshape(-1, self.feature_dim)).reshape(original_shape)

        target_idx = self.target_feature_index
        return data * self.scaler.scale_[target_idx] + self.scaler.mean_[target_idx]


class Dataset_WebTrafficTSF(Dataset):
    """Short-term daily web traffic forecasting dataset backed by the Kaggle TSF file.

    Each series corresponds to one Wikipedia page. We construct fixed-length
    sliding windows over the daily traffic values and return real calendar-based
    marks using day-of-week and month sine/cosine features.
    """

    _CACHE = {}
    MARK_FEATURE_NAMES = ["dow_sin", "dow_cos", "month_sin", "month_cos"]

    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="kaggle_web_traffic_dataset_without_missing_values.tsf",
        target="traffic",
        scale=False,
        timeenc=1,
        freq="web",
        seasonal_patterns=None,
    ):
        self.args = args
        if size is None:
            self.seq_len = 90
            self.label_len = 30
            self.pred_len = 30
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ["train", "val", "test"]
        self.flag = flag
        self.features = features
        self.target = target
        self.timeenc = timeenc
        self.freq = freq or "web"
        self.root_path = root_path
        self.data_path = data_path

        self.use_log1p = bool(getattr(args, "web_traffic_log1p", False))
        self.scale = bool(scale or self.use_log1p)
        self.use_time_marks = True
        self.feature_names = [self.target]
        self.mark_feature_names = list(self.MARK_FEATURE_NAMES)
        self.target_feature_index = 0

        self.series_ids = []
        self.series_start_keys = np.array([], dtype=object)
        self.series_values = np.empty((0, 0), dtype=np.float32)
        self.series_length = 0
        self.window_start = 0
        self.window_count = 0
        self.num_series = 0
        self._marks_by_start = {}

        self.__read_data__()

    def _resolve_tsf_path(self):
        candidates = []
        if os.path.isfile(self.root_path):
            candidates.append(self.root_path)
        if self.data_path:
            candidates.append(os.path.join(self.root_path, self.data_path))
        if os.path.isdir(self.root_path):
            candidates.extend(
                os.path.join(self.root_path, name)
                for name in [
                    "kaggle_web_traffic_dataset_without_missing_values.tsf",
                    "web_traffic_dataset.tsf",
                    "web_traffic.tsf",
                ]
            )
            candidates.extend(sorted(glob.glob(os.path.join(self.root_path, "*web*traffic*.tsf"))))

        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return os.path.abspath(candidate)

        raise FileNotFoundError(
            "Web Traffic TSF file not found. Expected a file like "
            "`kaggle_web_traffic_dataset_without_missing_values.tsf`."
        )

    @classmethod
    def _load_cached_source(cls, filepath, use_log1p):
        cache_key = (os.path.abspath(filepath), bool(use_log1p))
        if cache_key in cls._CACHE:
            return cls._CACHE[cache_key]

        metadata, attribute_names = _read_tsf_header(filepath)
        if "start_timestamp" not in attribute_names:
            raise ValueError(
                "Web Traffic TSF file is missing the `start_timestamp` attribute needed "
                "to construct calendar marks."
            )

        series_ids = []
        series_start_keys = []
        series_length = None

        for record in _iter_tsf_records(filepath, attribute_names):
            values = np.nan_to_num(record["values"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            values = np.clip(values, a_min=0.0, a_max=None)

            current_length = int(values.size)
            if series_length is None:
                series_length = current_length
            elif current_length != series_length:
                raise ValueError(
                    "Web Traffic TSF integration expects equal-length page series, "
                    f"but found lengths {series_length} and {current_length}."
                )

            series_name = record.get("series_name", f"series_{len(series_ids)}")
            start_timestamp = pd.to_datetime(record["start_timestamp"], errors="coerce")
            if pd.isna(start_timestamp):
                raise ValueError(
                    f"Invalid `start_timestamp` for web traffic series `{series_name}`: "
                    f"{record['start_timestamp']}"
                )

            series_ids.append(series_name)
            series_start_keys.append(start_timestamp.normalize().strftime("%Y-%m-%d"))

        if not series_ids or series_length is None:
            raise ValueError(f"No usable web traffic series found in TSF file: {filepath}")

        series_values = np.empty((len(series_ids), series_length), dtype=np.float32)
        for row_idx, record in enumerate(_iter_tsf_records(filepath, attribute_names)):
            values = np.nan_to_num(record["values"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            values = np.clip(values, a_min=0.0, a_max=None)
            if use_log1p:
                values = np.log1p(values)
            series_values[row_idx] = values

        marks_by_start = {
            start_key: _build_daily_cyclical_marks(start_key, series_length)
            for start_key in sorted(set(series_start_keys))
        }

        cls._CACHE[cache_key] = {
            "metadata": metadata,
            "series_ids": series_ids,
            "series_start_keys": np.asarray(series_start_keys, dtype=object),
            "series_values": series_values,
            "series_length": series_length,
            "marks_by_start": marks_by_start,
        }
        return cls._CACHE[cache_key]

    def _split_bounds(self, series_length):
        max_start = series_length - self.seq_len - self.pred_len
        if max_start < 0:
            return None

        train_end = max(self.seq_len + self.pred_len, int(np.floor(series_length * 0.7)))
        val_end = max(train_end, int(np.floor(series_length * 0.8)))
        train_end = min(train_end, series_length)
        val_end = min(max(val_end, train_end), series_length)

        split_start = {
            "train": 0,
            "val": max(0, train_end - self.seq_len),
            "test": max(0, val_end - self.seq_len),
        }
        split_end = {
            "train": train_end - self.seq_len - self.pred_len,
            "val": val_end - self.seq_len - self.pred_len,
            "test": series_length - self.seq_len - self.pred_len,
        }

        start = split_start[self.flag]
        end = split_end[self.flag]
        if end < start:
            return None
        return int(start), int(end - start + 1)

    def __read_data__(self):
        tsf_path = self._resolve_tsf_path()
        cached = self._load_cached_source(tsf_path, self.use_log1p)

        self.series_ids = cached["series_ids"]
        self.series_start_keys = cached["series_start_keys"]
        self.series_values = cached["series_values"]
        self.series_length = int(cached["series_length"])
        self._marks_by_start = cached["marks_by_start"]
        self.num_series = int(self.series_values.shape[0])

        minimum_length = max(self.seq_len + self.pred_len, 2 * (self.seq_len + self.pred_len))
        if self.series_length < minimum_length:
            raise ValueError(
                "Web Traffic series are too short for the configured short-term setup. "
                f"Need at least {minimum_length} points, found {self.series_length}."
            )

        split_bounds = self._split_bounds(self.series_length)
        if split_bounds is None:
            raise ValueError(
                "Could not construct temporal short-term windows for the Web Traffic dataset "
                f"with seq_len={self.seq_len} and pred_len={self.pred_len}."
            )

        self.window_start, self.window_count = split_bounds

    def __getitem__(self, index):
        series_idx = index // self.window_count
        local_window_offset = index % self.window_count

        start = self.window_start + local_window_offset
        end = start + self.seq_len
        target_start = end
        target_end = target_start + self.pred_len

        series = self.series_values[series_idx]
        marks = self._marks_by_start[self.series_start_keys[series_idx]]

        seq_x = series[start:end].reshape(-1, 1)
        seq_y = series[target_start:target_end].reshape(-1, 1)
        seq_x_mark = marks[start:end]
        seq_y_mark = marks[target_start:target_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return int(self.num_series * self.window_count)

    def inverse_transform(self, data):
        if not self.use_log1p:
            return data
        if isinstance(data, torch.Tensor):
            return torch.expm1(data)
        return np.expm1(data)


class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        train_path = os.path.join(root_path, "train.csv")
        test_path = os.path.join(root_path, "test.csv")
        label_path = os.path.join(root_path, "test_label.csv")

        if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_df      = pd.read_csv(train_path)
            test_df       = pd.read_csv(test_path)
            test_label_df = pd.read_csv(label_path)
        else:
            _require_optional_dependency(load_dataset, "datasets", "PSM Hugging Face fallback")
            ds_data  = load_dataset(HUGGINGFACE_REPO, name="PSM-data")
            ds_label = load_dataset(HUGGINGFACE_REPO, name="PSM-label")
            train_df      = ds_data["train"].to_pandas()
            test_df       = ds_data["test"].to_pandas()
            test_label_df = ds_label[next(iter(ds_label))].to_pandas()

        data = train_df.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        
        test_data = test_df.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = test_label_df.values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        
        train_path = os.path.join(root_path, "MSL_train.npy")
        test_path  = os.path.join(root_path, "MSL_test.npy")
        label_path = os.path.join(root_path, "MSL_test_label.npy")

        if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_data = np.load(train_path)
            test_data  = np.load(test_path)
            test_label = np.load(label_path)
        else:
            _require_optional_dependency(hf_hub_download, "huggingface_hub", "MSL file download")
            train_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="MSL/MSL_train.npy",repo_type="dataset")
            test_path  = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="MSL/MSL_test.npy",repo_type="dataset")
            label_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="MSL/MSL_test_label.npy",repo_type="dataset")

            train_data  = np.load(train_path)
            test_data   = np.load(test_path)
            test_label  = np.load(label_path)

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data  = self.scaler.transform(test_data)

        self.train = train_data
        self.test  = test_data
        self.test_labels = test_label

        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8):]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        
        train_path = os.path.join(root_path, "SMAP_train.npy")
        test_path  = os.path.join(root_path, "SMAP_test.npy")
        label_path = os.path.join(root_path, "SMAP_test_label.npy")

        if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_data = np.load(train_path)
            test_data  = np.load(test_path)
            test_label = np.load(label_path)
        else:
            _require_optional_dependency(hf_hub_download, "huggingface_hub", "SMAP file download")
            train_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMAP/SMAP_train.npy",repo_type="dataset")
            test_path  = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMAP/SMAP_test.npy",repo_type="dataset")
            label_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMAP/SMAP_test_label.npy",repo_type="dataset")

            train_data  = np.load(train_path)
            test_data   = np.load(test_path)
            test_label = np.load(label_path)

        # 标准化
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data  = self.scaler.transform(test_data)

        self.train = train_data
        self.test  = test_data
        self.test_labels = test_label

        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8):]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        
        train_path = os.path.join(root_path, "SMD_train.npy")
        test_path  = os.path.join(root_path, "SMD_test.npy")
        label_path = os.path.join(root_path, "SMD_test_label.npy")

        if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_data = np.load(train_path)
            test_data  = np.load(test_path)
            test_label = np.load(label_path)
        else:
            _require_optional_dependency(hf_hub_download, "huggingface_hub", "SMD file download")
            train_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMD/SMD_train.npy",repo_type="dataset")
            test_path  = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMD/SMD_test.npy",repo_type="dataset")
            label_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMD/SMD_test_label.npy",repo_type="dataset")

            train_data  = np.load(train_path)
            test_data   = np.load(test_path)
            test_label = np.load(label_path)
            
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = test_label
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train2_path = os.path.join(root_path, "swat_train2.csv")
        test_path   = os.path.join(root_path, "swat2.csv")
        if all(os.path.exists(p) for p in [train2_path, test_path]):
            train_data = pd.read_csv(train2_path)
            test_data   = pd.read_csv(test_path)
        else:
            _require_optional_dependency(load_dataset, "datasets", "SWAT Hugging Face fallback")
            ds = load_dataset(HUGGINGFACE_REPO, name="SWaT")
            train_data = ds["train"].to_pandas()
            test_data  = ds["test"].to_pandas()
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def _resolve_ts_path(self, root_path, dataset_name, flag):
        split = "TRAIN" if "train" in str(flag).lower() else "TEST"
        fname = f"{dataset_name}_{split}.ts"
        local = os.path.join(root_path, fname)
        if os.path.exists(local):
            return local
        _require_optional_dependency(hf_hub_download, "huggingface_hub", "UEA .ts dataset download")
        return hf_hub_download(HUGGINGFACE_REPO, filename=f"{dataset_name}/{fname}", repo_type="dataset")

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from ts files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .ts files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        dataset_name = self.args.model_id
        ts_path = self._resolve_ts_path(root_path, dataset_name, flag or "train")

        all_df, labels_df = self.load_single(ts_path)
        return all_df, labels_df

    def load_single(self, filepath):
        _require_optional_dependency(load_from_tsfile_to_dataframe, "sktime", "UEA .ts parsing")
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                             replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values
        labels = self.labels_df.loc[self.all_IDs[ind]].values
        if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
            num_samples = len(self.all_IDs)
            num_columns = self.feature_df.shape[1]
            seq_len = int(self.feature_df.shape[0] / num_samples)
            batch_x = batch_x.reshape((1, seq_len, num_columns))
            batch_x, labels, augmentation_tags = run_augmentation_single(batch_x, labels, self.args)

            batch_x = batch_x.reshape((1 * seq_len, num_columns))

        return self.instance_norm(torch.from_numpy(batch_x)), \
               torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)
