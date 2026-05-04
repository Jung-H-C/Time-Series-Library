from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, Dataset_DominickPanel, Dataset_DominickTSF, Dataset_MonashTSFGeneric, Dataset_TourismMonthlyTSF, Dataset_NN5DailyTSF, Dataset_CarPartsTSF, Dataset_WebTrafficTSF, \
    PSMSegLoader, MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
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


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
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
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
