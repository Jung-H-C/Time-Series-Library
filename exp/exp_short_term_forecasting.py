from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.losses import mape_loss, mase_loss, smape_loss
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas

warnings.filterwarnings('ignore')

try:
    from data_provider.m4 import M4Meta
except ImportError:
    M4Meta = None

try:
    from utils.m4_summary import M4Summary, evaluate_m4_subset_metrics
except ImportError:
    M4Summary = None
    evaluate_m4_subset_metrics = None


def _pad_m4_targets(timeseries, pred_len, device):
    target = torch.zeros((len(timeseries), pred_len), dtype=torch.float32, device=device)
    mask = torch.zeros_like(target)

    for index, series in enumerate(timeseries):
        series_array = np.asarray(series, dtype=np.float32).reshape(-1)
        valid_len = min(int(series_array.size), pred_len)
        if valid_len == 0:
            continue
        target[index, :valid_len] = torch.as_tensor(series_array[:valid_len], dtype=torch.float32, device=device)
        mask[index, :valid_len] = 1.0

    return target, mask


class Exp_Short_Term_Forecast(Exp_Basic):
    DOMINICK_DATASETS = {"dominick", "dominik"}

    def __init__(self, args):
        super(Exp_Short_Term_Forecast, self).__init__(args)
        self.final_train_epoch = None

    def _is_m4_dataset(self):
        return self.args.data == 'm4'

    def _is_dominick_dataset(self):
        return self.args.data in self.DOMINICK_DATASETS

    def _configure_short_term_defaults(self):
        if self._is_m4_dataset():
            if M4Meta is None:
                raise ImportError(
                    "M4 short-term forecasting requires the optional m4 dependencies "
                    "from `data_provider.m4`."
                )
            if M4Summary is None or evaluate_m4_subset_metrics is None:
                raise ImportError(
                    "M4 short-term forecasting evaluation requires the optional m4 summary dependencies."
                )
            self.args.pred_len = M4Meta.horizons_map[self.args.seasonal_patterns]
            self.args.seq_len = 2 * self.args.pred_len
            self.args.label_len = self.args.pred_len
            self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]
            return

        if self._is_dominick_dataset():
            if (self.args.seq_len, self.args.label_len, self.args.pred_len) == (96, 48, 96):
                self.args.seq_len = 16
                self.args.label_len = 8
                self.args.pred_len = 8
            if self.args.enc_in == 7:
                self.args.enc_in = 4
            if self.args.dec_in == 7:
                self.args.dec_in = 1
            if self.args.c_out == 7:
                self.args.c_out = 1
            if self.args.freq == 'h':
                self.args.freq = 'w'
            if self.args.target == 'OT':
                self.args.target = 'sales'

            if self.args.label_len > self.args.seq_len:
                raise ValueError(
                    f"`label_len` must be <= `seq_len` for Dominick short-term forecasting, "
                    f"got label_len={self.args.label_len}, seq_len={self.args.seq_len}."
                )

    def _build_model(self):
        self._configure_short_term_defaults()
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, loss_name='MSE'):
        if loss_name == 'MSE':
            return nn.MSELoss()
        if loss_name == 'MAE':
            return nn.L1Loss()
        if loss_name == 'MAPE':
            return mape_loss()
        if loss_name == 'MASE':
            return mase_loss()
        if loss_name == 'SMAPE':
            return smape_loss()
        raise ValueError(f"Unsupported loss for short_term_forecast: {loss_name}")

    def _select_target_channel(self, outputs, dataset):
        if outputs.shape[-1] == 1:
            return outputs

        target_idx = getattr(dataset, 'target_feature_index', 0)
        if 0 <= target_idx < outputs.shape[-1]:
            return outputs[:, :, target_idx:target_idx + 1]
        return outputs[:, :, -1:]

    def _build_fixed_window_decoder_input(self, batch_x, dataset):
        target_idx = getattr(dataset, 'target_feature_index', 0)
        history = batch_x[:, -self.args.label_len:, target_idx:target_idx + 1]
        future_pad = torch.zeros(
            (batch_x.size(0), self.args.pred_len, 1),
            dtype=batch_x.dtype,
            device=batch_x.device,
        )
        return torch.cat([history, future_pad], dim=1)

    def _validate_fixed_window_loss(self):
        if self.args.loss not in {'MSE', 'MAE'}:
            raise ValueError(
                "Dominick short-term forecasting currently supports only `MSE` or `MAE` loss. "
                f"Received `{self.args.loss}`."
            )

    def train(self, setting):
        if self._is_m4_dataset():
            return self._train_m4(setting)
        return self._train_fixed_window(setting)

    def _train_fixed_window(self, setting):
        self._validate_fixed_window_loss()

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        executed_epochs = 0
        for epoch in range(self.args.train_epochs):
            executed_epochs = epoch + 1
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                dec_inp = self._build_fixed_window_decoder_input(batch_x, train_data)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, None, dec_inp, None)
                        outputs = self._select_target_channel(outputs[:, -self.args.pred_len:, :], train_data)
                        loss = criterion(outputs, batch_y)
                else:
                    outputs = self.model(batch_x, None, dec_inp, None)
                    outputs = self._select_target_channel(outputs[:, -self.args.pred_len:, :], train_data)
                    loss = criterion(outputs, batch_y)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss) if train_loss else 0.0
            vali_loss = self._vali_fixed_window(vali_data, vali_loader, criterion)
            test_loss = self._vali_fixed_window(test_data, test_loader, criterion)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        self.final_train_epoch = executed_epochs
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def _vali_fixed_window(self, data_set, data_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                dec_inp = self._build_fixed_window_decoder_input(batch_x, data_set)
                outputs = self.model(batch_x, None, dec_inp, None)
                outputs = self._select_target_channel(outputs[:, -self.args.pred_len:, :], data_set)
                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())

        self.model.train()
        return np.average(total_loss) if total_loss else 0.0

    def _test_fixed_window(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        input_history = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        target_idx = getattr(test_data, 'target_feature_index', 0)
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                dec_inp = self._build_fixed_window_decoder_input(batch_x, test_data)
                outputs = self.model(batch_x, None, dec_inp, None)
                outputs = self._select_target_channel(outputs[:, -self.args.pred_len:, :], test_data)

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                history = batch_x[:, :, target_idx:target_idx + 1].detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    pred = test_data.inverse_transform(pred.reshape(-1, 1)).reshape(pred.shape)
                    true = test_data.inverse_transform(true.reshape(-1, 1)).reshape(true.shape)
                    history = test_data.inverse_transform(history.reshape(-1, 1)).reshape(history.shape)

                preds.append(pred)
                trues.append(true)
                input_history.append(history)

                if i % 20 == 0:
                    gt = np.concatenate((history[0, :, 0], true[0, :, 0]), axis=0)
                    pd = np.concatenate((history[0, :, 0], pred[0, :, 0]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        input_history = np.concatenate(input_history, axis=0)

        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        results_folder = self._results_folder_path(setting)
        os.makedirs(results_folder, exist_ok=True)

        mae = np.mean(np.abs(preds - trues))
        mse = np.mean((preds - trues) ** 2)
        rmse = np.sqrt(mse)
        mape = np.nan
        mspe = np.nan
        print('mse:{}, mae:{}'.format(mse, mae))

        final_epoch = self.final_train_epoch if self.final_train_epoch is not None else 'N/A'
        with open("result_short_term_forecast.txt", 'a') as handle:
            handle.write(setting + "  \n")
            handle.write('final_epoch:{}, mse:{}, mae:{}'.format(final_epoch, mse, mae))
            handle.write('\n\n')

        np.save(os.path.join(results_folder, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(results_folder, 'pred.npy'), preds)
        np.save(os.path.join(results_folder, 'true.npy'), trues)
        np.save(os.path.join(results_folder, 'history.npy'), input_history)
        return

    def _train_m4(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)
        mse = nn.MSELoss()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = self.model(batch_x, None, dec_inp, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss_value = criterion(batch_x, self.args.frequency_map, outputs, batch_y, batch_y_mark)
                loss_sharpness = mse((outputs[:, 1:, :] - outputs[:, :-1, :]), (batch_y[:, 1:, :] - batch_y[:, :-1, :]))
                loss = loss_value  # + loss_sharpness * 1e-5
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self._vali_m4(train_loader, vali_loader, criterion)
            test_loss = vali_loss
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def _vali_m4(self, train_loader, vali_loader, criterion):
        x, _ = train_loader.dataset.last_insample_window()
        y = vali_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)

        self.model.eval()
        with torch.no_grad():
            B, _, C = x.shape
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
            outputs = torch.zeros((B, self.args.pred_len, C)).float()
            id_list = np.arange(0, B, 500)
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                outputs[id_list[i]:id_list[i + 1], :, :] = self.model(
                    x[id_list[i]:id_list[i + 1]],
                    None,
                    dec_inp[id_list[i]:id_list[i + 1]],
                    None,
                ).detach().cpu()
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            pred = outputs
            true, batch_y_mark = _pad_m4_targets(y, self.args.pred_len, pred.device)

            loss = criterion(x.detach().cpu()[:, :, 0], self.args.frequency_map, pred[:, :, 0], true, batch_y_mark)

        self.model.train()
        return loss

    def test(self, setting, test=0):
        if self._is_m4_dataset():
            return self._test_m4(setting, test=test)
        return self._test_fixed_window(setting, test=test)

    def _test_m4(self, setting, test=0):
        _, train_loader = self._get_data(flag='train')
        _, test_loader = self._get_data(flag='test')
        x, _ = train_loader.dataset.last_insample_window()
        y = test_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            B, _, C = x.shape
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
            outputs = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            id_list = np.arange(0, B, 1)
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                outputs[id_list[i]:id_list[i + 1], :, :] = self.model(
                    x[id_list[i]:id_list[i + 1]],
                    None,
                    dec_inp[id_list[i]:id_list[i + 1]],
                    None,
                )

                if id_list[i] % 1000 == 0:
                    print(id_list[i])

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            outputs = outputs.detach().cpu().numpy()

            preds = outputs
            trues = y
            x = x.detach().cpu().numpy()

            visual_stride = max(1, preds.shape[0] // 10)
            for i in range(0, preds.shape[0], visual_stride):
                gt = np.concatenate((x[i, :, 0], trues[i]), axis=0)
                pd = np.concatenate((x[i, :, 0], preds[i, :, 0]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        print('test shape:', preds.shape)

        results_folder = self._results_folder_path(setting)
        os.makedirs(results_folder, exist_ok=True)

        subset_metrics = evaluate_m4_subset_metrics(
            preds[:, :, 0],
            root_path=self.args.root_path,
            group_name=self.args.seasonal_patterns,
        )
        np.save(
            os.path.join(results_folder, 'metrics.npy'),
            np.array(
                [
                    subset_metrics['smape'],
                    subset_metrics['owa'],
                    subset_metrics['mape'],
                    subset_metrics['mase'],
                ],
                dtype=np.float32,
            ),
        )

        folder_path = './m4_results/' + self.args.model + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        forecasts_df = pandas.DataFrame(preds[:, :, 0], columns=[f'V{i + 1}' for i in range(self.args.pred_len)])
        forecasts_df.index = test_loader.dataset.ids[:preds.shape[0]]
        forecasts_df.index.name = 'id'
        forecasts_df.set_index(forecasts_df.columns[0], inplace=True)
        forecasts_df.to_csv(folder_path + self.args.seasonal_patterns + '_forecast.csv')

        print(self.args.model)
        file_path = './m4_results/' + self.args.model + '/'
        if 'Weekly_forecast.csv' in os.listdir(file_path) \
                and 'Monthly_forecast.csv' in os.listdir(file_path) \
                and 'Yearly_forecast.csv' in os.listdir(file_path) \
                and 'Daily_forecast.csv' in os.listdir(file_path) \
                and 'Hourly_forecast.csv' in os.listdir(file_path) \
                and 'Quarterly_forecast.csv' in os.listdir(file_path):
            m4_summary = M4Summary(file_path, self.args.root_path)
            smape_results, owa_results, mape, mase = m4_summary.evaluate()
            print('smape:', smape_results)
            print('mape:', mape)
            print('mase:', mase)
            print('owa:', owa_results)
        else:
            print('After all 6 tasks are finished, you can calculate the averaged index')
        return
