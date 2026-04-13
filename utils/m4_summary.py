# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright 2020 Element AI Inc. All rights reserved.

"""
M4 Summary
"""
from collections import OrderedDict

import numpy as np
import pandas as pd

from data_provider.m4 import M4Dataset
from data_provider.m4 import M4Meta
import os


def _trim_nan_series(values):
    return [np.asarray(v[~np.isnan(v)], dtype=np.float32) for v in values]


def _stack_if_uniform(series_list):
    if not series_list:
        return np.empty((0, 0), dtype=np.float32)

    lengths = {len(series) for series in series_list}
    if len(lengths) == 1:
        return np.stack(series_list).astype(np.float32, copy=False)
    return series_list


def group_values(values, groups, group_name):
    return _stack_if_uniform(_trim_nan_series(values[groups == group_name]))


def evaluate_m4_subset_metrics(
    forecast: np.ndarray,
    *,
    root_path: str,
    group_name: str,
) -> dict[str, float]:
    training_set = M4Dataset.load(training=True, dataset_file=root_path)
    test_set = M4Dataset.load(training=False, dataset_file=root_path)

    model_forecast = np.asarray(forecast, dtype=np.float32)
    if model_forecast.ndim != 2:
        raise ValueError(
            f"Expected forecast to have shape [num_series, pred_len], got {model_forecast.shape}."
        )

    target = np.asarray(group_values(test_set.values, test_set.groups, group_name), dtype=np.float32)
    insample = group_values(training_set.values, test_set.groups, group_name)

    if model_forecast.shape != target.shape:
        raise ValueError(
            f"Forecast shape {model_forecast.shape} does not match target shape {target.shape} "
            f"for M4 subset '{group_name}'."
        )

    frequency = training_set.frequencies[test_set.groups == group_name][0]

    model_mase = np.mean([
        mase(
            forecast=model_forecast[i],
            insample=insample[i],
            outsample=target[i],
            frequency=frequency,
        )
        for i in range(len(model_forecast))
    ])

    model_smape = np.mean([
        np.mean(smape_2(model_forecast[i], target[i]))
        for i in range(len(model_forecast))
    ])
    model_mape = np.mean([
        np.mean(mape(model_forecast[i], target[i]))
        for i in range(len(model_forecast))
    ])

    naive_path = os.path.join(root_path, 'submission-Naive2.csv')
    if os.path.exists(naive_path):
        naive2_forecasts = pd.read_csv(naive_path).values[:, 1:].astype(np.float32)
        naive2_forecasts = _stack_if_uniform(_trim_nan_series(naive2_forecasts))
        naive2_forecast = np.asarray(group_values(naive2_forecasts, test_set.groups, group_name), dtype=np.float32)

        naive2_mase = np.mean([
            mase(
                forecast=naive2_forecast[i],
                insample=insample[i],
                outsample=target[i],
                frequency=frequency,
            )
            for i in range(len(model_forecast))
        ])
        naive2_smape = np.mean([
            np.mean(smape_2(naive2_forecast[i], target[i]))
            for i in range(len(model_forecast))
        ])

        if naive2_mase == 0.0 or naive2_smape == 0.0:
            owa = float("nan")
        else:
            owa = (model_mase / naive2_mase + model_smape / naive2_smape) / 2.0
    else:
        owa = float("nan")

    return {
        "smape": float(model_smape),
        "owa": float(owa),
        "mape": float(model_mape),
        "mase": float(model_mase),
    }


def mase(forecast, insample, outsample, frequency):
    return np.mean(np.abs(forecast - outsample)) / np.mean(np.abs(insample[:-frequency] - insample[frequency:]))


def smape_2(forecast, target):
    denom = np.abs(target) + np.abs(forecast)
    # divide by 1.0 instead of 0.0, in case when denom is zero the enumerator will be 0.0 anyway.
    denom[denom == 0.0] = 1.0
    return 200 * np.abs(forecast - target) / denom


def mape(forecast, target):
    denom = np.abs(target)
    # divide by 1.0 instead of 0.0, in case when denom is zero the enumerator will be 0.0 anyway.
    denom[denom == 0.0] = 1.0
    return 100 * np.abs(forecast - target) / denom


class M4Summary:
    def __init__(self, file_path, root_path):
        self.file_path = file_path
        self.training_set = M4Dataset.load(training=True, dataset_file=root_path)
        self.test_set = M4Dataset.load(training=False, dataset_file=root_path)
        self.naive_path = os.path.join(root_path, 'submission-Naive2.csv')

    def evaluate(self):
        """
        Evaluate forecasts using M4 test dataset.

        :param forecast: Forecasts. Shape: timeseries, time.
        :return: sMAPE and OWA grouped by seasonal patterns.
        """
        grouped_owa = OrderedDict()
        has_naive2 = os.path.exists(self.naive_path)
        if has_naive2:
            naive2_forecasts = pd.read_csv(self.naive_path).values[:, 1:].astype(np.float32)
            naive2_forecasts = _stack_if_uniform(_trim_nan_series(naive2_forecasts))

        model_mases = {}
        naive2_smapes = {}
        naive2_mases = {}
        grouped_smapes = {}
        grouped_mapes = {}
        for group_name in M4Meta.seasonal_patterns:
            file_name = self.file_path + group_name + "_forecast.csv"
            if os.path.exists(file_name):
                model_forecast = pd.read_csv(file_name).values

            target = group_values(self.test_set.values, self.test_set.groups, group_name)
            # all timeseries within group have same frequency
            frequency = self.training_set.frequencies[self.test_set.groups == group_name][0]
            insample = group_values(self.training_set.values, self.test_set.groups, group_name)

            model_mases[group_name] = np.mean([mase(forecast=model_forecast[i],
                                                    insample=insample[i],
                                                    outsample=target[i],
                                                    frequency=frequency) for i in range(len(model_forecast))])
            grouped_smapes[group_name] = np.mean([
                np.mean(smape_2(model_forecast[i], target[i])) for i in range(len(model_forecast))
            ])
            grouped_mapes[group_name] = np.mean([
                np.mean(mape(model_forecast[i], target[i])) for i in range(len(model_forecast))
            ])
            if has_naive2:
                naive2_forecast = group_values(naive2_forecasts, self.test_set.groups, group_name)
                naive2_mases[group_name] = np.mean([mase(forecast=naive2_forecast[i],
                                                         insample=insample[i],
                                                         outsample=target[i],
                                                         frequency=frequency) for i in range(len(model_forecast))])

                naive2_smapes[group_name] = np.mean([
                    np.mean(smape_2(naive2_forecast[i], target[i])) for i in range(len(model_forecast))
                ])

        grouped_smapes = self.summarize_groups(grouped_smapes)
        grouped_mapes = self.summarize_groups(grouped_mapes)
        grouped_model_mases = self.summarize_groups(model_mases)
        if has_naive2:
            grouped_naive2_smapes = self.summarize_groups(naive2_smapes)
            grouped_naive2_mases = self.summarize_groups(naive2_mases)
            for k in grouped_model_mases.keys():
                grouped_owa[k] = (grouped_model_mases[k] / grouped_naive2_mases[k] +
                                  grouped_smapes[k] / grouped_naive2_smapes[k]) / 2
        else:
            for k in grouped_model_mases.keys():
                grouped_owa[k] = float("nan")

        def round_all(d):
            return dict(map(lambda kv: (kv[0], np.round(kv[1], 3)), d.items()))

        return round_all(grouped_smapes), round_all(grouped_owa), round_all(grouped_mapes), round_all(
            grouped_model_mases)

    def summarize_groups(self, scores):
        """
        Re-group scores respecting M4 rules.
        :param scores: Scores per group.
        :return: Grouped scores.
        """
        scores_summary = OrderedDict()

        def group_count(group_name):
            return len(np.where(self.test_set.groups == group_name)[0])

        weighted_score = {}
        for g in ['Yearly', 'Quarterly', 'Monthly']:
            weighted_score[g] = scores[g] * group_count(g)
            scores_summary[g] = scores[g]

        others_score = 0
        others_count = 0
        for g in ['Weekly', 'Daily', 'Hourly']:
            others_score += scores[g] * group_count(g)
            others_count += group_count(g)
        weighted_score['Others'] = others_score
        scores_summary['Others'] = others_score / others_count

        average = np.sum(list(weighted_score.values())) / len(self.test_set.groups)
        scores_summary['Average'] = average

        return scores_summary
