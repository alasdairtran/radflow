import logging
import math
from typing import Any, Dict

import numpy as np
import numpy.linalg as la
import pandas as pd
import torch
import torch.nn as nn
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn.initializers import InitializerApplicator
from sklearn.metrics import mean_absolute_error, mean_squared_error

from nos.modules.metrics import get_smape

from .base import BaseModel
from .radflow import LSTMDecoder


@Model.register('radflow_taxi')
class RADflowTaxi(BaseModel):
    def __init__(self,
                 vocab: Vocabulary,
                 hidden_size: int,
                 forecast_len: int,
                 n_layers: int,
                 dropout: float,
                 variant: str,
                 adj_path: str = 'data/taxi/sz_adj.csv',
                 initializer: InitializerApplicator = InitializerApplicator()):
        super().__init__(vocab)

        self.hidden_size = hidden_size
        adj_df = pd.read_csv(adj_path, header=None)
        adjacency_matrix = np.array(adj_df, dtype=np.float32)

        self.decoder = LSTMDecoder(
            hidden_size, n_layers, dropout, variant, 1)

        self.forecast_len = forecast_len
        self.mse = nn.MSELoss()

        # Shortcut to create new tensors in the same device as the module
        self.register_buffer('_long', torch.LongTensor(1))

        if initializer:
            initializer(self)

    def forward(self, x, y, scale, keys, splits) -> Dict[str, Any]:
        # Enable anomaly detection to find the operation that failed to compute
        # its gradient.
        # torch.autograd.set_detect_anomaly(True)

        B, S, N = x.shape
        T = y.shape[1]
        H = self.hidden_size
        # x.shape == [batch_size, seq_len, n_nodes]
        # y.shape == [batch_size, pre_len, n_nodes]

        x = x.transpose(1, 2).reshape(B * N, S)
        y = y.transpose(1, 2).reshape(B * N, T)
        z = torch.cat([x, y], dim=1)
        # z.shape == [batch_size * n_nodes, total_len]

        hiddens, forecasts, _ = self.decoder(z)
        # forecasts.shape == [batch_size * n_nodes, total_len]

        if splits[0] == 'train':
            forecasts = forecasts[:, :-1]
            targets = z[:, 1:]
        else:
            targets = y
            forecast_list = []
            for i in range(targets.shape[1]):
                hiddens, forecast_i, _ = self.decoder(x)
                preds = forecast_i[:, -1:]
                forecast_list.append(preds)
                x = torch.cat([x, preds], dim=1)
            forecasts = torch.cat(forecast_list, dim=1)

        numerator = torch.abs(targets - forecasts)
        denominator = torch.abs(targets) + torch.abs(forecasts)
        loss = numerator / denominator
        loss = loss.mean()

        targets = targets.detach().cpu().numpy()
        forecasts = forecasts.detach().cpu().numpy()
        rmse = math.sqrt(mean_squared_error(targets, forecasts)) * scale[0]
        mae = mean_absolute_error(targets, forecasts) * scale[0]
        F_norm = la.norm(targets - forecasts, 'fro') / la.norm(targets, 'fro')
        r2 = 1 - ((targets - forecasts)**2).sum() / \
            ((targets - targets.mean())**2).sum()
        var = 1 - (np.var(targets - forecasts)) / np.var(targets)

        self.history['_n_batches'] += 1
        self.history['_n_samples'] += B
        self.batch_history['rmse'] += rmse
        self.batch_history['mae'] += mae
        self.batch_history['acc'] += 1 - F_norm
        self.batch_history['r2'] += r2
        self.batch_history['var'] += var

        smapes, _ = get_smape(targets, forecasts)
        self.history['_n_steps'] += smapes.shape[0] * smapes.shape[1]
        self.step_history['smape'] += np.sum(smapes)

        out_dict = {
            'loss': loss,
            'sample_size': self._long.new_tensor(B),
        }

        return out_dict
