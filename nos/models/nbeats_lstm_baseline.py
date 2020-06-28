import json
import logging
import os
import pickle
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn.initializers import InitializerApplicator
from overrides import overrides
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from tqdm import tqdm

from nos.modules import Decoder
from nos.modules.linear import GehringLinear
from nos.utils import keystoint

from .base import BaseModel
from .metrics import get_smape

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register('nbeats_lstm_baseline')
class NBEATSLSTMBaseline(BaseModel):
    def __init__(self,
                 vocab: Vocabulary,
                 data_dir: str,
                 seed_word: str = 'programming',
                 forecast_length: int = 28,
                 backcast_length: int = 224,
                 max_neighbours: int = 0,
                 hidden_size: int = 128,
                 dropout: float = 0.2,
                 n_stacks: int = 16,
                 missing_p: float = 0.0,
                 agg: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator()):
        super().__init__(vocab)
        self.mse = nn.MSELoss()
        self.max_neighbours = max_neighbours
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.total_length = backcast_length + forecast_length
        self.rs = np.random.RandomState(1234)
        self.hidden_size = hidden_size
        self.device = torch.device('cuda:0')
        self.max_start = None
        self.missing_p = missing_p
        self.agg = agg

        self.layer = LSTMLayer(hidden_size, n_stacks, dropout, agg)

        with open(f'{data_dir}/{seed_word}.pkl', 'rb') as f:
            self.in_degrees, _, self.series = pickle.load(f)

        # Shortcut to create new tensors in the same device as the module
        self.register_buffer('_long', torch.LongTensor(1))
        self.register_buffer('_float', torch.FloatTensor(1))
        initializer(self)

    def _initialize_series(self):
        if isinstance(next(iter(self.series.values())), torch.Tensor):
            return

        # Check how long the time series is
        series_len = len(next(iter(self.series.values())))
        n_weeks = series_len // 7 + 1

        p = next(self.parameters())
        for k, v in self.series.items():
            self.series[k] = np.asarray(v).astype(float)

        # Sort by view counts
        for node, neighs in self.in_degrees.items():
            counts = []
            for neigh in neighs:
                count = self.series[neigh][:self.backcast_length].sum()
                counts.append(count)
            keep_idx = np.argsort(counts)[::-1][:self.max_neighbours]
            self.in_degrees[node] = np.array(neighs)[keep_idx]

        for k, v in self.series.items():
            if self.missing_p > 0:
                size = len(v) - self.forecast_length
                indices = self.rs.choice(np.arange(1, size), replace=False,
                                         size=int(size * self.missing_p))
                self.series[k][indices] = np.nan

                mask = np.isnan(self.series[k])
                idx = np.where(~mask, np.arange(len(mask)), 0)
                np.maximum.accumulate(idx, out=idx)
                self.series[k][mask] = self.series[k][idx[mask]]

            self.series[k] = p.new_tensor(self.series[k])
            # self.diff[k] = p.new_tensor(self.diff[k])

        self.max_start = len(
            self.series[k]) - self.forecast_length * 2 - self.total_length

    def _get_neighbours(self, keys, split, start):
        sources = self._float.new_zeros(
            (len(keys), self.max_neighbours, self.total_length))
        masks = self._long.new_ones((len(keys), self.max_neighbours)).bool()

        for i, key in enumerate(keys):
            if key in self.in_degrees:
                neighs = self.in_degrees[key][:self.max_neighbours]
            else:
                neighs = []
            for j, n in enumerate(neighs):
                s = self.series[n]
                s = s[start:start+self.total_length]
                sources[i, j] = s
                masks[i, j] = 0

        return sources, masks

    def forward(self, keys, splits) -> Dict[str, Any]:
        # Enable anomaly detection to find the operation that failed to compute
        # its gradient.
        # torch.autograd.set_detect_anomaly(True)

        self._initialize_series()

        split = splits[0]
        B = len(keys)
        p = next(self.parameters())
        # keys.shape == [batch_size]

        self.history['_n_batches'] += 1
        self.history['_n_samples'] += B

        out_dict = {
            'loss': None,
            'sample_size': p.new_tensor(B),
        }

        series_list = []
        for key in keys:
            s = self.series[key]
            if split == 'train':
                start = self.rs.randint(0, self.max_start)
            elif split == 'valid':
                start = self.max_start + self.forecast_length
            elif split == 'test':
                start = self.max_start + self.forecast_length * 2
            s = s[start:start+self.total_length]
            series_list.append(s)

        series = torch.stack(series_list, dim=0)
        # series.shape == [batch_size, total_length]

        sources = series[:, :-1]
        targets = series[:, 1:]

        X = torch.log1p(sources)

        n_sources, neighbour_masks = self._get_neighbours(
            keys, split, start)
        X_neighs = torch.log1p(n_sources[:, :, :-1])
        y_neighs = torch.log1p(n_sources[:, :, 1:])
        forecast = self._forward(X, y_neighs, neighbour_masks)
        # forecast.shape == [batch_size, seq_len]

        preds = torch.exp(forecast) - 1
        numerator = torch.abs(targets - preds)
        denominator = torch.abs(targets) + torch.abs(preds)
        loss = numerator / denominator
        loss[torch.isnan(loss)] = 0
        loss = loss.mean()

        out_dict['loss'] = loss

        # During evaluation, we compute one time step at a time
        if splits[0] in ['test']:
            X = X[:, :-self.forecast_length]
            yn = y_neighs[:, :, :-self.forecast_length]

            targets = series[:, -self.forecast_length:]
            preds = torch.zeros(B, self.forecast_length)

            for i in range(self.forecast_length):
                forecast = self._forward(X, yn, neighbour_masks)[:, -1]
                # forecast.shape == [batch_size, seq_len]
                pred = torch.exp(forecast) - 1
                preds[:, i] = pred

                X = torch.cat([X, forecast.unsqueeze(1)], dim=1)
                offset = (-(self.forecast_length - i - 1)) or None
                yn = y_neighs[:, :, :offset]

            smapes, daily_errors = get_smape(targets, preds)

            out_dict['smapes'] = smapes
            out_dict['daily_errors'] = daily_errors
            out_dict['keys'] = keys
            out_dict['preds'] = preds.cpu().numpy().tolist()

        return out_dict

    def _forward(self, X, yn, masks):
        # X.shape == [batch_size, seq_len]
        # Xn.shape == [batch_size, n_neighs, seq_len]

        forecast = self.layer(X, yn, masks)

        return forecast

    @classmethod
    def get_params(cls, vocab: Vocabulary, params: Params) -> Dict[str, Any]:

        params_dict: Dict[str, Any] = {}

        params_dict['initializer'] = InitializerApplicator.from_params(
            params.pop('initializer', None))

        return params_dict


class LSTMLayer(nn.Module):
    def __init__(self, hidden_size, n_layers, dropout, agg):
        super().__init__()
        self.layer = nn.LSTM(1, hidden_size, n_layers,
                             batch_first=True, dropout=dropout)
        self.proj_f = GehringLinear(hidden_size, 1)
        self.agg = agg
        if agg:
            self.agg_proj = GehringLinear(hidden_size * 2, hidden_size)

    def forward(self, X, yn, masks):
        # X.shape == [batch_size, seq_len]
        # yn.shape == [batch_size, n_neighs, seq_len]

        X = X.unsqueeze(-1)
        # X.shape == [batch_size, seq_len, 1]

        X, _ = self.layer(X)
        # X.shape == [batch_size, seq_len, hidden_size]

        if self.agg:
            X = self.aggregate(X, yn, masks)
            # X.shape == [batch_size, seq_len, hidden_size]

        f = self.proj_f(X).squeeze(-1)
        # b.shape == f.shape == [batch_size, seq_len]

        return f

    def aggregate(self, X, yn, masks):
        B, N, S = yn.shape
        yn = yn.reshape(B * N, S, 1)
        yn, _ = self.layer(yn)
        # clone is needed for in-place masking
        yn = yn.reshape(B, N, S, -1).clone()
        yn[masks] = 0
        # yn.shape == [batch_size, n_neighs, seq_len, hidden_size]

        yn = yn.mean(dim=1)

        X = torch.cat([X, yn], dim=-1)
        # X.shape == [batch_size, seq_len, 2 * hidden_size]

        X = F.gelu(self.agg_proj(X))
        # X.shape == [batch_size, seq_len, hidden_size]

        return X
