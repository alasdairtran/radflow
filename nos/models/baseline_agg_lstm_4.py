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


@Model.register('new_naive')
class NewNaive(BaseModel):
    def __init__(self,
                 vocab: Vocabulary,
                 data_dir: str,
                 seed_word: str = 'vevo',
                 method: str = 'previous_day',
                 forecast_length: int = 7,
                 backcast_length: int = 42,
                 test_lengths: List[int] = [7],
                 initializer: InitializerApplicator = InitializerApplicator()):
        super().__init__(vocab)
        self.mse = nn.MSELoss()
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.total_length = backcast_length + forecast_length
        self.test_lengths = test_lengths
        self.rs = np.random.RandomState(1234)
        self.device = torch.device('cuda:0')
        self.method = method

        assert method in ['previous_day', 'previous_week']

        with open(f'{data_dir}/{seed_word}.pkl', 'rb') as f:
            self.in_degrees, _, self.series = pickle.load(f)

        # Shortcut to create new tensors in the same device as the module
        self.register_buffer('_long', torch.LongTensor(1))
        self.register_buffer('_float', torch.tensor(0.1))

        initializer(self)

    def _initialize_series(self):
        if isinstance(next(iter(self.series.values())), torch.Tensor):
            return

        for k, v in self.series.items():
            v_array = np.asarray(v)
            self.series[k] = self._float.new_tensor(v_array)

        self.max_start = len(
            self.series[k]) - self.forecast_length * 2 - self.total_length

    def forward(self, keys, splits) -> Dict[str, Any]:
        # Enable anomaly detection to find the operation that failed to compute
        # its gradient.
        # torch.autograd.set_detect_anomaly(True)

        self._initialize_series()

        split = splits[0]
        B = len(keys)
        # keys.shape == [batch_size]

        self.history['_n_batches'] += 1
        self.history['_n_samples'] += B

        out_dict = {
            'loss': None,
            'sample_size': self._float.new_tensor(B),
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
        # series.shape == [batch_size, total_length]

        sources = series[:, :self.backcast_length]
        targets = series[:, -self.forecast_length:]
        B = sources.shape[0]
        if self.method == 'previous_day':
            preds = sources[:, -1:]
            preds = preds.expand(B, self.forecast_length)
        elif self.method == 'previous_week':
            preds = sources[:, -7:]
            preds = preds.repeat(1, self.forecast_length // 7 + 1)
            preds = preds[:, :self.forecast_length]

        loss = self.mse(torch.log1p(preds), torch.log1p(targets))
        out_dict['loss'] = loss

        # During evaluation, we compute one time step at a time
        if splits[0] in ['test']:
            smapes, daily_errors = get_smape(targets, preds)

            out_dict['smapes'] = smapes.tolist()
            out_dict['daily_errors'] = daily_errors.tolist()
            out_dict['keys'] = keys
            out_dict['preds'] = preds.cpu().numpy().tolist()
            self.history['_n_samples'] += len(keys)
            self.history['_n_steps'] += smapes.shape[0] * smapes.shape[1]

            for k in self.test_lengths:
                self.step_history[f'smape_{k}'] += np.sum(smapes[:, :k])

        return out_dict


@Model.register('baseline_agg_lstm_4')
class BaselineAggLSTM4(BaseModel):
    def __init__(self,
                 vocab: Vocabulary,
                 data_dir: str,
                 agg_type: str,
                 forecast_length: int = 7,
                 backcast_length: int = 42,
                 test_lengths: List[int] = [7],
                 peek: bool = True,
                 seed_word: str = 'vevo',
                 num_layers: int = 8,
                 hidden_size: int = 128,
                 dropout: float = 0.1,
                 log: bool = False,
                 opt_smape: bool = False,
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
        self.test_lengths = test_lengths
        self.log = log
        self.opt_smape = opt_smape

        self.max_start = None
        self.rs = np.random.RandomState(1234)

        assert agg_type in ['mean', 'none']
        self.agg_type = agg_type
        if agg_type in ['none']:
            self.fc = GehringLinear(self.hidden_size, 1)
        elif agg_type in ['mean']:
            self.fc = GehringLinear(self.hidden_size * 2, 1)

        with open(f'{data_dir}/{seed_word}.pkl', 'rb') as f:
            self.in_degrees, _, self.series = pickle.load(f)

        # Shortcut to create new tensors in the same device as the module
        self.register_buffer('_long', torch.LongTensor(1))

        initializer(self)

    def _initialize_series(self):
        if isinstance(next(iter(self.series.values())), torch.Tensor):
            return

        p = next(self.parameters())
        for k, v in self.series.items():
            self.series[k] = np.asarray(v).astype(float)

        # Sort by view counts
        logger.info('Processing edges')
        for node, neighs in tqdm(self.in_degrees.items()):
            counts = []
            for n in neighs:
                n['mask'] = p.new_tensor(np.asarray(n['mask']))
                count = self.series[n['id']][:self.backcast_length].sum()
                counts.append(count)
            keep_idx = np.argsort(counts)[::-1][:self.max_neighbours]
            self.in_degrees[node] = np.array(neighs)[keep_idx]

        for k, v in self.series.items():
            v_array = np.asarray(v)
            self.series[k] = p.new_tensor(v_array)

        self.max_start = len(
            self.series[k]) - self.forecast_length * 2 - self.total_length

    def _forward(self, series):
        # series.shape == [batch_size, seq_len]

        # Take the difference
        if not self.log:
            training_series = series.clone().detach()
            training_series[training_series == 0] = 1
            diff = training_series[:, 1:] / training_series[:, :-1]
            targets = diff[:, 1:]
            inputs = diff[:, :-1]
        else:
            inputs = series[:, :-1]
            targets = series[:, 1:]

        X = inputs
        # X.shape == [batch_size, seq_len - 1]

        X, forecast = self.decoder(X)

        return X, forecast, targets

    def _forward_full(self, series):
        # series.shape == [batch_size, seq_len]

        if not self.log:
            training_series = series.clone().detach()
            training_series[training_series == 0] = 1
            X = training_series[:, 1:] / training_series[:, :-1]
        else:
            X = series
        # X.shape == [batch_size, seq_len]

        X, forecast = self.decoder(X)

        return X, forecast

    def _get_neighbour_embeds(self, X, keys, start, total_len):
        if self.agg_type == 'none':
            return X

        B, T, _ = X.shape
        N = self.max_neighbours

        # We plus one to give us option to either peek or not
        masks = X.new_ones(B, N, total_len).bool()
        # We set neighs to be full precision since some view counts are
        # bigger than 65504
        neighs = X.new_zeros(B, N, total_len, dtype=torch.float32)

        for b, key in enumerate(keys):
            # in_degrees maps node_id to a sorted list of dicts
            # a dict key looks like: {'id': 123, 'mask'; [0, 0, 1]}
            if key in self.in_degrees:
                if i >= len(self.in_degrees[key]):
                    break
                for i in range(N):
                    n = self.in_degrees[key][i]
                    neighs[b, i] = self.series[n['id']][start:start+total_len]
                    masks[b, i] = n['mask'][start:start+total_len]

        # neighs.shape == [batch_size, n_neighbors, seq_len]
        # masks.shape == [batch_size, n_neighbors, seq_len]

        if self.log:
            neighs = torch.log1p(neighs)

        neighs = neighs.reshape(B * N, total_len)
        # Detach as we won't back-propagate to the neighbours
        # This will also prevent gradient overflow in mixed precision training
        Xn, _ = self._forward_full(neighs).detach()

        if not self.log:
            Xn = Xn.reshape(B, N, total_len - 1, -1)
            masks = masks[:, :, 1:]
        else:
            Xn = Xn.reshape(B, N, total_len, -1)

        if self.peek:
            Xn = Xn[:, :, 1:]
            masks = masks[:, :, 1:]
        else:
            Xn = Xn[:, :, :-1]
            masks = masks[:, :, -1:]

        Xn = Xn.detach()

        X_out = self._aggregate(X, Xn, masks)
        return X_out

    def _aggregate(self, X, Xn, masks):
        # X.shape == [batch_size, seq_len, hidden_size]
        # Xn.shape == [batch_size, n_neighs, seq_len, hidden_size]
        # masks.shape == [batch_size, n_neighs, seq_len]

        # Mask out irrelevant values.
        Xn = Xn.clone()
        Xn[masks] = 0

        # Let's just take the average
        Xn = Xn.sum(dim=1)
        # Xn.shape == [batch_size, seq_len, hidden_size]

        n_neighs = (~masks).sum(dim=1).unsqueeze(-1)
        # Avoid division by zero
        n_neighs = n_neighs.clamp(min=1)
        # n_neighs.shape == [batch_size, seq_len, 1]

        Xn = Xn / n_neighs
        # Xn.shape == [batch_size, seq_len, hidden_size]

        X_out = torch.cat([X, Xn], dim=-1)
        # Xn.shape == [batch_size, seq_len, 2 * hidden_size]

        return X_out

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

        raw_series = torch.stack(series_list, dim=0)
        # series.shape == [batch_size, seq_len]

        series = torch.log1p(raw_series) if self.log else raw_series

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

        if split in ['valid', 'test']:
            preds = preds[:, -self.forecast_length:]
            targets = targets[:, -self.forecast_length:]

        if self.log and self.opt_smape:
            preds = torch.exp(preds)
            if split in ['valid', 'test']:
                targets = raw_series[:, -self.forecast_length:]
            else:
                targets = raw_series[:, 1:]
            numerator = torch.abs(targets - preds)
            denominator = torch.abs(targets) + torch.abs(preds)
            loss = numerator / denominator
            loss[torch.isnan(loss)] = 0
            loss = loss.mean()
        else:
            loss = self.mse(preds, targets)
        out_dict['loss'] = loss

        # During evaluation, we compute one time step at a time
        if split in ['valid', 'test']:
            target_list = []
            for key in keys:
                s = start + self.backcast_length
                e = s + self.forecast_length
                target_list.append(self.series[key][s:e])
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

                if not self.log:
                    current_views = current_views * pred
                else:
                    current_views = pred
                preds[:, i] = current_views
                series = torch.cat(
                    [series, current_views.unsqueeze(-1)], dim=-1)

            if self.log:
                preds = torch.exp(preds)
            smapes, daily_errors = get_smape(targets, preds)

            out_dict['smapes'] = smapes.tolist()
            out_dict['daily_errors'] = daily_errors.tolist()
            out_dict['keys'] = keys
            self.history['_n_samples'] += len(keys)
            self.history['_n_steps'] += smapes.shape[0] * smapes.shape[1]

            for k in self.test_lengths:
                self.step_history[f'smape_{k}'] += np.sum(smapes[:, :k])

        return out_dict


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size, n_layers, dropout):
        super().__init__()
        self.in_proj = GehringLinear(1, hidden_size)
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            self.layers.append(LSTMLayer(hidden_size, dropout))

        self.out_proj = GehringLinear(hidden_size * n_layers, hidden_size)

    def forward(self, X):
        # X.shape == [batch_size, seq_len]
        # yn.shape == [batch_size, n_neighs, seq_len]

        X = X.unsqueeze(-1)
        # X.shape == [batch_size, seq_len, 1]

        X = self.in_proj(X)
        # X.shape == [batch_size, seq_len, hidden_size]

        forecast = X.new_zeros(X.shape[0], X.shape[1])
        h_list = []
        for layer in self.layers:
            h, b, f = layer(X)
            X = X - b
            h_list.append(h)
            forecast = forecast + f

        h = torch.cat(h_list,  dim=-1)
        # h.shape == [batch_size, seq_len, n_layers * hidden_size]

        h = self.out_proj(h)
        # h.shape == [batch_size, seq_len, hidden_size]

        return h, f


class LSTMLayer(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.layer = nn.LSTM(hidden_size, hidden_size, 1,
                             batch_first=True, dropout=dropout)
        self.proj_f = GehringLinear(hidden_size, hidden_size)
        self.proj_b = GehringLinear(hidden_size, hidden_size)

        self.out_f = GehringLinear(hidden_size, 1)
        self.out_b = GehringLinear(hidden_size, hidden_size)

    def forward(self, X):
        # X.shape == [batch_size, seq_len]
        # yn.shape == [batch_size, n_neighs, seq_len]

        X, _ = self.layer(X)
        # X.shape == [batch_size, seq_len, hidden_size]

        b = self.out_b(F.gelu(self.proj_b(X)))
        f = self.out_f(F.gelu(self.proj_f(X))).squeeze(-1)
        # b.shape == f.shape == [batch_size, seq_len]

        return X, b, f
