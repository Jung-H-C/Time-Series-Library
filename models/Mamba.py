import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba

from layers.Embed import DataEmbedding

class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len

        self.d_inner = configs.d_model * configs.expand
        raw_dt_rank = getattr(configs, 'dt_rank', 'auto')
        if isinstance(raw_dt_rank, str):
            normalized_dt_rank = raw_dt_rank.strip().lower()
            if normalized_dt_rank == 'auto':
                self.dt_rank = 'auto'
            else:
                self.dt_rank = int(normalized_dt_rank)
        else:
            self.dt_rank = int(raw_dt_rank)
        if self.dt_rank != 'auto' and self.dt_rank < 1:
            raise ValueError(f"dt_rank must be a positive integer or 'auto', got {raw_dt_rank!r}")
        
        self.embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.mamba = Mamba(
            d_model = configs.d_model,
            d_state = configs.d_ff,
            d_conv = configs.d_conv,
            expand = configs.expand,
            dt_rank = self.dt_rank,
        )

        self.out_layer = nn.Linear(configs.d_model, configs.c_out, bias=False)

    def forecast(self, x_enc, x_mark_enc):
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        x = self.embedding(x_enc, x_mark_enc)
        x = self.mamba(x)
        x_out = self.out_layer(x)

        x_out = x_out * std_enc + mean_enc
        return x_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['short_term_forecast', 'long_term_forecast']:
            x_out = self.forecast(x_enc, x_mark_enc)
            return x_out[:, -self.pred_len:, :]

        # other tasks not implemented
