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
from .nbeats_base import NBeatsNet
from .nbeats_plus import NBeatsPlusNet

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register('nbeats_transformer')
class NBEATSTransformer(BaseModel):
    def __init__(self,
                 vocab: Vocabulary,
                 data_dir: str,
                 seed_word: str = 'programming',
                 forecast_length: int = 28,
                 backcast_length: int = 224,
                 max_neighbours: int = 0,
                 hidden_size: int = 128,
                 dropout: float = 0.1,
                 n_stacks: int = 8,
                 missing_p: float = 0.0,
                 n_heads: int = 4,
                 peek: bool = False,
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
        initializer(self)

        with open(f'{data_dir}/{seed_word}.pkl', 'rb') as f:
            self.in_degrees, _, self.series = pickle.load(f)

        self.layers = nn.ModuleList([])
        for _ in range(n_stacks):
            layer = TBEATLayer(hidden_size, dropout,
                               n_heads, forecast_length, agg)
            self.layers.append(layer)
        self.upsample = GehringLinear(1, hidden_size)

        # Shortcut to create new tensors in the same device as the module
        self.register_buffer('_long', torch.LongTensor(1))
        self.register_buffer('_float', torch.FloatTensor(1))

    def _initialize_series(self):
        if isinstance(next(iter(self.series.values())), torch.Tensor):
            return

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

        self.max_start = len(
            self.series[k]) - self.forecast_length * 2 - self.total_length

    def _get_neighbours(self, keys, split, start):
        # neighbor_lens = []
        # mask_list = []

        sources = self._float.new_zeros(
            (len(keys), self.max_neighbours, self.total_length))
        masks = self._long.new_ones((len(keys), self.max_neighbours)).bool()

        for i, key in enumerate(keys):
            if key in self.in_degrees:
                neighs = self.in_degrees[key][:self.max_neighbours]
            else:
                neighs = []
            # neighbor_lens.append([len(neighs)])
            # mask_list.append([False] * len(neighs) + [True]
            #                  * (self.max_neighbours - len(neighs)))
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
                # start = len(s) - self.total_length
                start = self.max_start + self.forecast_length * 2
            s = s[start:start+self.total_length]
            series_list.append(s)

        series = torch.stack(series_list, dim=0)
        # series.shape == [batch_size, total_length]

        # targets_diff = diffs[:, -self.forecast_length:]
        S = torch.log1p(series)
        # S = sources

        n_series, masks = self._get_neighbours(
            keys, split, start)
        Sn = torch.log1p(n_series)

        X = S[:, :-self.forecast_length]
        Xn = Sn[:, :, :-self.forecast_length]
        # S.shape == [batch_size, total_len]
        # Sn.shape == [batch_size, n_neighs, total_len]
        # masks.shape == [batch_size, n_neighs]

        forecast = self._forward(X, Xn, masks)

        preds = torch.exp(forecast) - 1
        # targets = series[:, -self.forecast_length:]
        targets = series.unfold(1, self.forecast_length, 1)[:, 1:]
        # targets.shape == [batch_size, seq_len, forecast_len]

        targets = targets.transpose(0, 1)
        # targets.shape == [seq_len, batch_size, forecast_len]

        S, B, T = targets.shape
        targets = targets.reshape(S * B, T)
        preds = preds.reshape(S * B, T)

        loss = self._get_smape_loss(targets, preds)

        # loss = self.mse(X, targets)
        out_dict['loss'] = loss

        # During evaluation, we compute one time step at a time
        if splits[0] in ['test']:
            smapes, daily_errors = get_smape(targets, preds)
            out_dict['smapes'] = smapes
            out_dict['daily_errors'] = daily_errors
            out_dict['keys'] = keys
            out_dict['preds'] = preds.cpu().numpy().tolist()

        return out_dict

    def _forward(self, X, Xn, masks):
        X = X.transpose(0, 1)
        # X.shape == [seq_len, batch_size]

        X = X.unsqueeze(-1)
        # X.shape == [seq_len, batch_size, 1]

        # Step 1: Upsample the time series
        X = self.upsample(X)

        Xn = Xn.transpose(2, 1).transpose(1, 0)
        # Xn.shape == [seq_len, batch_size, n_neighs]

        masks = masks.unsqueeze(0).expand_as(Xn)
        # masks.shape == [seq_len, batch_size, n_neighs]

        Xn = Xn.unsqueeze(-1)

        Xn = self.upsample(Xn)

        T, B, E = X.shape
        forecast = X.new_zeros(T, B, self.forecast_length)
        # X.shape == [seq_len, batch_size, hidden_size]
        for layer in self.layers:
            b, bn, f = layer(X, Xn, masks)
            forecast = forecast + f
            X = X - b
            Xn = Xn - bn

        # forecast.shape == [seq_len, batch_size, forecast_len]

        return forecast

    def _get_smape_loss(self, targets, preds):
        numerator = torch.abs(targets - preds)
        denominator = torch.abs(targets) + torch.abs(preds)
        loss = numerator / denominator
        loss[torch.isnan(loss)] = 0
        loss = loss.mean()
        return loss

    @classmethod
    def get_params(cls, vocab: Vocabulary, params: Params) -> Dict[str, Any]:

        params_dict: Dict[str, Any] = {}

        params_dict['initializer'] = InitializerApplicator.from_params(
            params.pop('initializer', None))

        return params_dict


class TBEATLayer(nn.Module):
    def __init__(self, hidden_size, dropout, n_heads, forecast_len, agg):
        super().__init__()
        self.agg = agg
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, dropout)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.norm_1 = nn.LayerNorm(hidden_size)
        self.norm_2 = nn.LayerNorm(hidden_size)

        self.linear_1 = GehringLinear(hidden_size, hidden_size * 4)
        self.linear_2 = GehringLinear(hidden_size * 4, hidden_size)

        self.final_out = GehringLinear(hidden_size, forecast_len)

        self.activation = F.gelu

        if agg:
            self.neigh_attn = nn.MultiheadAttention(
                hidden_size, n_heads, dropout,
                add_bias_kv=True, add_zero_attn=True)
            self.norm_3 = nn.LayerNorm(hidden_size)
            self.dropout_4 = nn.Dropout(dropout)

    def forward(self, X, Xn, masks):
        # X.shape == [seq_len, batch_size, hidden_size]
        # Xn.shape == [seq_len, batch_size, n_neighs, hidden_size]
        # masks.shape == [seq_len, batch_size, n_neighs]

        b, f = self._forward(X, Xn, masks)

        S, B, N, E = Xn.shape
        Xn = Xn.reshape(S, B * N, E)
        # Xn.shape == [seq_len, batch_size * n_neighs]

        bn, _ = self._forward(Xn)
        bn = bn.reshape(S, B, N, E)
        bn[masks] = 0

        return b, bn, f

    def _forward(self, X, Xn=None, masks=None):
        # We can't attend positions which are True
        T, B, E = X.shape
        attn_mask = X.new_full((T, T), 1)
        # Zero out the diagonal and everything below
        # We can attend to ourselves and the past
        attn_mask = torch.triu(attn_mask, diagonal=1)
        # attn_mask.shape == [T, T]

        X_1, _ = self.attn(X, X, X, need_weights=False, attn_mask=attn_mask)
        # X.shape == [seq_len, batch_size, hidden_size]

        X = X + self.dropout_1(X_1)
        X = self.norm_1(X)

        # Attention over neighbours
        if self.agg and Xn is not None:
            T, B, N, E = Xn.shape
            X = X.reshape(1, T * B, E)

            Xn = Xn.transpose(2, 1).transpose(1, 0)
            # Xn.shape == [n_neighs, seq_len, batch_size, hidden_size]

            Xn = Xn.reshape(N, T * B, E)
            # Xn.shape == [n_neighs, seq_len * batch_size, hidden_size]

            masks = masks.reshape(T * B, N)

            X_a, _ = self.neigh_attn(
                X, Xn, Xn, key_padding_mask=masks, need_weights=False)

            X = X + self.dropout_4(X_a)
            X = self.norm_3(X)

            X = X.reshape(T, B, E)

        X_2 = self.linear_2(self.dropout_2(self.activation(self.linear_1(X))))
        X = X + self.dropout_3(X_2)
        X = self.norm_2(X)
        # X.shape == [seq_len, batch_size, hidden_size]

        f = self.final_out(X)
        # out.shape == [seq_len, batch_size, forecast_len]

        return X, f
