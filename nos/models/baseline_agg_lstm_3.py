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


@Model.register('baseline_agg_lstm_3')
class BaselineAggLSTM3(BaseModel):
    def __init__(self,
                 vocab: Vocabulary,
                 data_dir: str,
                 agg_type: str,
                 forecast_length: int = 7,
                 backcast_length: int = 42,
                 peek: bool = False,
                 seed_word: str = 'vevo',
                 num_layers: int = 8,
                 hidden_size: int = 128,
                 dropout: float = 0.1,
                 log: bool = False,
                 max_neighbours: int = 8,
                 initializer: InitializerApplicator = InitializerApplicator()):
        super().__init__(vocab)
        self.decoder = LSTMDecoder(hidden_size, num_layers, dropout)
        self.mse = nn.MSELoss()
        self.hidden_size = hidden_size
        self.peek = peek
        self.max_neighbours = max_neighbours
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.total_length = forecast_length + backcast_length
        self.log = log

        self.max_start = None
        self.rs = np.random.RandomState(1234)
        initializer(self)

        assert agg_type in ['mean', 'none']
        self.agg_type = agg_type
        if agg_type in ['none']:
            self.fc = GehringLinear(self.hidden_size, 1)
        elif agg_type in ['mean']:
            self.fc = GehringLinear(self.hidden_size * 2, 1)

        with open(f'{data_dir}/{seed_word}.pkl', 'rb') as f:
            self.sources, _, self.series = pickle.load(f)

        # Shortcut to create new tensors in the same device as the module
        self.register_buffer('_long', torch.LongTensor(1))

    def _initialize_series(self):
        if isinstance(next(iter(self.series.values())), torch.Tensor):
            return

        p = next(self.parameters())
        for k, v in self.series.items():
            self.series[k] = np.asarray(v).astype(float)

        # Sort by view counts
        for node, neighs in self.sources.items():
            counts = []
            for neigh in neighs:
                count = self.series[neigh][:self.backcast_length].sum()
                counts.append(count)
            keep_idx = np.argsort(counts)[::-1][:self.max_neighbours]
            self.sources[node] = np.array(neighs)[keep_idx]

        for k, v in self.series.items():
            v_array = np.asarray(v)
            self.series[k] = p.new_tensor(v_array)

        self.max_start = len(
            self.series[k]) - self.forecast_length * 2 - self.total_length

    def _forward(self, series):
        # series.shape == [batch_size, seq_len]

        training_series = series.clone().detach()
        training_series[training_series == 0] = 1

        # Take the difference
        diff = training_series[:, 1:] / training_series[:, :-1]
        targets = diff[:, 1:]
        inputs = diff[:, :-1]

        X = inputs
        # X.shape == [batch_size, seq_len - 1]

        X, forecast = self.decoder(X)

        return X, forecast, targets

    def _forward_full(self, series):
        # series.shape == [batch_size, seq_len]

        training_series = series.clone().detach()
        training_series[training_series == 0] = 1

        # Take the difference
        inputs = training_series[:, 1:] / training_series[:, :-1]
        X = inputs
        # X.shape == [batch_size, seq_len]

        X, forecast = self.decoder(X)

        return X, forecast

    def _get_neighbour_embeds(self, X, keys, start, total_len):
        if self.agg_type == 'none':
            return X

        neighbor_lens = []
        source_list = []

        for key in keys:
            if key in self.sources:
                sources = self.sources[key][:self.max_neighbours]
            else:
                sources = []
            neighbor_lens.append(len(sources))
            for s in sources:
                s_series = self.series[s]
                s_series = s_series[start:start+total_len]
                source_list.append(s_series)

        sources = torch.stack(source_list, dim=0)
        # sources.shape == [batch_size * n_neighbors, seq_len]

        if self.log:
            sources = torch.log1p(sources)

        X_neighbors, _ = self._forward_full(sources)
        X_neighbors = X_neighbors[:, 1:]

        # X_neighbors.shape == [batch_size * n_neighbors, seq_len, hidden_size]

        # Go through each element in the batch
        cursor = 0
        X_full_list = []
        for n_neighbors, X_i in zip(neighbor_lens, X):
            X_neighbors_i = X_neighbors[cursor:cursor + n_neighbors]
            # X_neighbors_i == [n_neighbors, seq_len, hidden_size]

            X_full = self._aggregate(X_neighbors_i, X_i)
            X_full_list.append(X_full)

            cursor += n_neighbors

        X_full = torch.cat(X_full_list, dim=0)
        # X_full.shape [batch_size, seq_len, hidden_size]

        return X_full

    def _aggregate(self, X_neighbors_i, X_i):
        # X_neighbors_i.shape = [n_neighbors, seq_len, hidden_size]
        # X_i.shape == [seq_len, hidden_size]

        X_i = X_i.unsqueeze(0)
        # X_i.shape == [1, seq_len, hidden_size]

        if X_neighbors_i.shape[0] == 0:
            X_out = X_i.new_zeros(*X_i.shape)
        else:
            X_out = X_neighbors_i.mean(dim=0).unsqueeze(0)
            # X_out.shape == [1, seq_len, hidden_size]

        # Combine own embedding with neighbor embedding
        X_full = torch.cat([X_i, X_out], dim=-1)
        # X_full.shape == [1, seq_len, 2 * hidden_size]

        return X_full

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
                if self.max_start == 0:
                    start = 0
                else:
                    start = self.rs.randint(0, self.max_start)
            elif split == 'valid':
                start = self.max_start + self.forecast_length
            elif split == 'test':
                start = self.max_start + self.forecast_length * 2
            s = s[start:start+self.total_length]
            series_list.append(s)

        series = torch.stack(series_list, dim=0)
        # series.shape == [batch_size, seq_len]

        if self.log:
            series = torch.log1p(series)

        X, preds, targets = self._forward(series)
        # X.shape == [batch_size, seq_len, hidden_size]
        # targets.shape == [batch_size, seq_len]

        if self.agg_type != 'none':
            X_full = self._get_neighbour_embeds(
                X, keys, start, self.total_length)
            # X_full.shape == [batch_size, seq_len, out_hidden_size]

            X_full = self.fc(X_full)
            # X_full.shape == [batch_size, seq_len, 1]

            preds = preds + X_full.squeeze(-1)
            # preds.shape == [batch_size, seq_len]

        loss = self.mse(preds, targets)
        out_dict['loss'] = loss

        # During evaluation, we compute one time step at a time
        if splits[0] in ['test']:
            target_list = []
            for key in keys:
                target_list.append(self.series[key][-self.forecast_length:])
            targets = torch.stack(target_list, dim=0)
            # targets.shape == [batch_size, forecast_len]

            preds = targets.new_zeros(*targets.shape)

            series = series[:, :-self.forecast_length]
            current_views = series[:, -1]
            for i in range(self.forecast_length):
                X, pred = self._forward_full(series)
                pred = pred[:, -1]
                if self.agg_type != 'none':
                    seq_len = self.total_length - self.forecast_length + i + 1
                    X_full = self._get_neighbour_embeds(
                        X, keys, start, seq_len)
                    X_full = self.fc(X_full)
                    pred = pred + X_full.squeeze(-1)[:, -1]
                    # delta.shape == [batch_size]

                current_views = current_views * pred
                preds[:, i] = current_views
                series = torch.cat(
                    [series, current_views.unsqueeze(-1)], dim=-1)

            if self.log:
                preds = torch.exp(preds) - 1
            smape, daily_errors = get_smape(targets, preds)

            out_dict['smapes'] = smape
            out_dict['daily_errors'] = daily_errors
            out_dict['keys'] = keys

        return out_dict

    @classmethod
    def get_params(cls, vocab: Vocabulary, params: Params) -> Dict[str, Any]:

        params_dict: Dict[str, Any] = {}

        params_dict['initializer'] = InitializerApplicator.from_params(
            params.pop('initializer', None))

        return params_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, Any]:
        return super().get_metrics(reset)


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size, n_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            self.layers.append(LSTMLayer(hidden_size, dropout))

        self.out_proj = GehringLinear(hidden_size * n_layers, hidden_size)

    def forward(self, X):
        # X.shape == [batch_size, seq_len]
        # yn.shape == [batch_size, n_neighs, seq_len]

        forecast = X.new_zeros(*X.shape)
        h_list = []
        for layer in self.layers:
            h, b, f = layer(X)
            X = X - b
            forecast = forecast + f
            h_list.append(h)

        h = torch.cat(h_list,  dim=-1)
        # h.shape == [batch_size, seq_len, n_layers * hidden_size]

        h = self.out_proj(h)
        # h.shape == [batch_size, seq_len, hidden_size]

        return h, f


class LSTMLayer(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.layer = nn.LSTM(1, hidden_size, 1,
                             batch_first=True, dropout=dropout)
        self.proj_f = GehringLinear(hidden_size, hidden_size)
        self.proj_b = GehringLinear(hidden_size, hidden_size)

        self.out_f = GehringLinear(hidden_size, 1)
        self.out_b = GehringLinear(hidden_size, 1)

    def forward(self, X):
        # X.shape == [batch_size, seq_len]
        # yn.shape == [batch_size, n_neighs, seq_len]

        X = X.unsqueeze(-1)
        # X.shape == [batch_size, seq_len, 1]

        X, _ = self.layer(X)
        # X.shape == [batch_size, seq_len, hidden_size]

        b = self.out_b(F.gelu(self.proj_b(X))).squeeze(-1)
        f = self.out_f(F.gelu(self.proj_f(X))).squeeze(-1)
        # b.shape == f.shape == [batch_size, seq_len]

        return X, b, f
