from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, Dataset_DominickPanel, Dataset_DominickTSF, Dataset_MonashTSFGeneric, Dataset_MultiSeriesForecast, Dataset_TourismMonthlyTSF, Dataset_NN5DailyTSF, Dataset_CarPartsTSF, Dataset_WebTrafficTSF, \
    PSMSegLoader, MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
from data_provider.uea import collate_fn
import hashlib

import numpy as np
from torch.utils.data import DataLoader, Dataset

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'multi_series': Dataset_MultiSeriesForecast,
    'm4': Dataset_M4,
    'dominick': Dataset_DominickPanel,
    'dominik': Dataset_DominickPanel,
    'dominick_tsf': Dataset_DominickTSF,
    'm1_yearly': Dataset_MonashTSFGeneric,
    'm1_quarterly': Dataset_MonashTSFGeneric,
    'm1_monthly': Dataset_MonashTSFGeneric,
    'm3_yearly': Dataset_MonashTSFGeneric,
    'm3_quarterly': Dataset_MonashTSFGeneric,
    'm3_monthly': Dataset_MonashTSFGeneric,
    'm3_other': Dataset_MonashTSFGeneric,
    'cif_2016': Dataset_MonashTSFGeneric,
    'london_smart_meters': Dataset_MonashTSFGeneric,
    'aus_electricity_demand': Dataset_MonashTSFGeneric,
    'wind_farms': Dataset_MonashTSFGeneric,
    'bitcoin': Dataset_MonashTSFGeneric,
    'pedestrian_counts': Dataset_MonashTSFGeneric,
    'vehicle_trips': Dataset_MonashTSFGeneric,
    'kdd_cup_2018': Dataset_MonashTSFGeneric,
    'weather_tsf': Dataset_MonashTSFGeneric,
    'solar_10min': Dataset_MonashTSFGeneric,
    'solar_weekly': Dataset_MonashTSFGeneric,
    'electricity_hourly': Dataset_MonashTSFGeneric,
    'electricity_weekly': Dataset_MonashTSFGeneric,
    'fred_md': Dataset_MonashTSFGeneric,
    'san_francisco_traffic_hourly': Dataset_MonashTSFGeneric,
    'san_francisco_traffic_weekly': Dataset_MonashTSFGeneric,
    'rideshare': Dataset_MonashTSFGeneric,
    'hospital': Dataset_MonashTSFGeneric,
    'covid_deaths': Dataset_MonashTSFGeneric,
    'temperature_rain': Dataset_MonashTSFGeneric,
    'sunspot': Dataset_MonashTSFGeneric,
    'saugeen_river_flow': Dataset_MonashTSFGeneric,
    'us_births': Dataset_MonashTSFGeneric,
    'solar_power': Dataset_MonashTSFGeneric,
    'wind_power': Dataset_MonashTSFGeneric,
    'tourism_monthly': Dataset_TourismMonthlyTSF,
    'tourism': Dataset_TourismMonthlyTSF,
    'nn5': Dataset_NN5DailyTSF,
    'nn5_daily': Dataset_NN5DailyTSF,
    'car_parts': Dataset_CarPartsTSF,
    'carparts': Dataset_CarPartsTSF,
    'web_traffic': Dataset_WebTrafficTSF,
    'webtraffic': Dataset_WebTrafficTSF,
    'kaggle_web_traffic': Dataset_WebTrafficTSF,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader
}


class FixedIndexSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, index):
        return self.dataset[int(self.indices[index])]

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def _candidate_subset_key(args, flag):
    parts = [
        getattr(args, "task_name", ""),
        getattr(args, "data", ""),
        getattr(args, "root_path", ""),
        getattr(args, "data_path", ""),
        getattr(args, "features", ""),
        getattr(args, "target", ""),
        getattr(args, "freq", ""),
        str(getattr(args, "seq_len", "")),
        str(getattr(args, "label_len", "")),
        str(getattr(args, "pred_len", "")),
        str(flag).lower(),
    ]
    return "|".join(str(part) for part in parts)


def _candidate_subset_indices(args, flag, dataset_len, limit):
    seed = int(getattr(args, "candidate_sample_seed", 2026))
    digest = hashlib.blake2b(_candidate_subset_key(args, flag).encode("utf-8"), digest_size=8).digest()
    split_seed = (seed + int.from_bytes(digest, byteorder="little", signed=False)) % (2 ** 32)
    rng = np.random.default_rng(split_seed)
    return np.sort(rng.choice(dataset_len, size=limit, replace=False))


def _maybe_limit_candidate_dataset(args, data_set, flag, batch_size):
    task_name = getattr(args, "task_name", "")
    data_name = getattr(args, "data", "")
    if task_name == "long_term_forecast":
        # The lazy multi-series loader performs selection before loading values.
        if data_name == "multi_series":
            return data_set
        limit_names = {
            "train": "long_term_train_sample_limit",
            "val": "long_term_val_sample_limit",
            "test": "long_term_test_sample_limit",
        }
        flag_name = str(flag).lower()
        if flag_name not in limit_names:
            return data_set
        limit = int(getattr(args, limit_names[flag_name], 0) or 0)
        if limit <= 0 or len(data_set) <= limit:
            return data_set
        indices = _candidate_subset_indices(args, flag_name, len(data_set), limit)
        print(
            f"[long-term-subset] {flag_name}: original={len(data_set)}, selected={limit}, "
            f"seed={getattr(args, 'candidate_sample_seed', 2026)}"
        )
        return FixedIndexSubset(data_set, indices)

    if task_name != "short_term_forecast" or data_name == "m4":
        return data_set

    dataset_len = len(data_set)
    flag_name = str(flag).lower()
    sample_limit = int(getattr(args, "candidate_sample_limit", 0) or 0)
    train_iteration_limit = int(getattr(args, "candidate_train_iteration_limit", 0) or 0)

    if flag_name == "train" and train_iteration_limit > 0:
        limit = train_iteration_limit * int(batch_size)
        limit_label = f"{train_iteration_limit} mini-batches"
    elif flag_name in {"val", "test"} and sample_limit > 0:
        limit = sample_limit
        limit_label = f"{sample_limit} samples"
    else:
        return data_set

    if dataset_len <= limit:
        return data_set

    indices = _candidate_subset_indices(args, flag_name, dataset_len, limit)
    print(
        f"[candidate-subset] {flag_name}: original={dataset_len}, selected={len(indices)} "
        f"({limit_label}), seed={getattr(args, 'candidate_sample_seed', 2026)}"
    )
    return FixedIndexSubset(data_set, indices)


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    if args.data == 'multi_series':
        # Deterministically sampled indices are sorted by global window index,
        # preserving source-series locality so bounded lazy caches do not
        # repeatedly reopen large RDS/Arrow sources.
        shuffle_flag = False
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        data_set = _maybe_limit_candidate_dataset(args, data_set, flag, batch_size)
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        data_set = _maybe_limit_candidate_dataset(args, data_set, flag, batch_size)
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
