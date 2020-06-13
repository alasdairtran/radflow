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

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register('nbeats_naive')
class NaiveWiki(BaseModel):
    def __init__(self,
                 vocab: Vocabulary,
                 data_dir: str,
                 seed_word: str = 'programming',
                 method: str = 'previous_day',
                 forecast_length: int = 28,
                 backcast_length: int = 140,
                 initializer: InitializerApplicator = InitializerApplicator()):
        super().__init__(vocab)
        self.mse = nn.MSELoss()
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.total_length = backcast_length + forecast_length
        self.rs = np.random.RandomState(1234)
        self.device = torch.device('cuda:0')
        self.method = method
        initializer(self)

        assert method in ['previous_day', 'previous_week']

        with open(f'{data_dir}/{seed_word}.pkl', 'rb') as f:
            self.in_degrees, _, self.series = pickle.load(f)

        # Shortcut to create new tensors in the same device as the module
        self.register_buffer('_long', torch.LongTensor(1))
        self.register_buffer('_float', torch.tensor(0.1))

    def _initialize_series(self):
        if isinstance(next(iter(self.series.values())), torch.Tensor):
            return

        # Remove trends
        # for k, v in self.series.items():
        #     r = 0 if self.total_length % 7 == 0 else (
        #         7 - (self.total_length % 7))
        #     v_full = np.zeros(self.total_length + r)
        #     v_full[1:self.total_length+1] = v[:self.total_length]
        #     v_full = v_full.reshape((self.total_length + r) // 7, 7)
        #     avg_all = v_full.mean()
        #     avg_week = v_full.mean(axis=0)
        #     diff = avg_week - avg_all
        #     diff = np.tile(diff, 53)
        #     diff = diff[1:366]
        #     self.series[k] = v - diff

        for k, v in self.series.items():
            v_array = np.asarray(v)
            self.series[k] = self._float.new_tensor(v_array)

        self.max_start = len(
            self.series[k]) - self.forecast_length - self.total_length

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
            if split in ['train', 'valid']:
                start = self.rs.randint(0, self.max_start)
                s = s[start:start+self.total_length]
            elif split == 'test':
                s = s[-self.total_length:]
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

            out_dict['smapes'] = smapes
            out_dict['daily_errors'] = daily_errors
            out_dict['keys'] = keys
            out_dict['preds'] = preds.cpu().numpy().tolist()

        return out_dict

    @classmethod
    def get_params(cls, vocab: Vocabulary, params: Params) -> Dict[str, Any]:

        params_dict: Dict[str, Any] = {}

        params_dict['initializer'] = InitializerApplicator.from_params(
            params.pop('initializer', None))

        return params_dict


@Model.register('nbeats_wiki')
class NBEATSWiki(BaseModel):
    def __init__(self,
                 vocab: Vocabulary,
                 data_dir: str,
                 seed_word: str = 'programming',
                 forecast_length: int = 28,
                 backcast_length: int = 140,
                 max_neighbours: int = 0,
                 hidden_size: int = 128,
                 dropout: float = 0.2,
                 attn: bool = False,
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
        self.diff = {}
        initializer(self)

        with open(f'{data_dir}/{seed_word}.pkl', 'rb') as f:
            self.in_degrees, _, self.series = pickle.load(f)

        self.net = NBeatsNet(device=torch.device('cuda:0'),
                             stack_types=[NBeatsNet.GENERIC_BLOCK] * 16,
                             nb_blocks_per_stack=1,
                             forecast_length=forecast_length,
                             backcast_length=backcast_length,
                             thetas_dims=[128] * 16,
                             hidden_layer_units=hidden_size,
                             share_weights_in_stack=False,
                             dropout=dropout,
                             max_neighbours=max_neighbours,
                             attn=attn,
                             )

        # Shortcut to create new tensors in the same device as the module
        self.register_buffer('_long', torch.LongTensor(1))

    def _initialize_series(self):
        if isinstance(next(iter(self.series.values())), torch.Tensor):
            return

        # Remove trends using the first year of data
        train_len = 140
        for k, v in self.series.items():
            v_full = np.array(v[:train_len]).reshape(train_len // 7, 7)
            avg_all = v_full.mean()
            avg_week = v_full.mean(axis=0)
            diff = avg_week - avg_all
            diff = np.tile(diff, 53 * 9)
            diff = diff[:len(v)]
            # self.series[k] = v - diff
            self.diff[k] = diff
            # self.diff[k] = max(v[:train_len])

        p = next(self.parameters())
        for k, v in self.series.items():
            v_array = np.asarray(v)
            self.series[k] = p.new_tensor(v_array)
            self.diff[k] = p.new_tensor(self.diff[k])

        self.max_start = len(
            self.series[k]) - self.forecast_length - self.total_length

    def _get_neighbours(self, keys, split):
        neighbor_lens = []
        mask_list = []

        sources = torch.zeros(
            (len(keys), self.max_neighbours, self.backcast_length))

        for i, key in enumerate(keys):
            if key in self.in_degrees:
                neighs = self.in_degrees[key][:self.max_neighbours]
            else:
                neighs = []
            neighbor_lens.append([len(neighs)])
            mask_list.append([False] * len(neighs) + [True]
                             * (self.max_neighbours - len(neighs)))
            for j, n in enumerate(neighs):
                s = self.series[n]
                if split in ['train', 'valid']:
                    s = s[:self.backcast_length]
                elif split == 'test':
                    s = s[-self.total_length:-self.forecast_length]
                sources[i, j] = s

        # sources.shape == [batch_size, max_neighbours, backcast_length]
        sources = sources.reshape(
            len(keys) * self.max_neighbours, self.backcast_length)

        neighbor_lens = torch.tensor(neighbor_lens)
        # sources.shape == [batch_size, 1]

        mask_list = torch.tensor(mask_list)

        return sources, neighbor_lens.to(self.device), mask_list.to(self.device)

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
        diff_list = []
        for key in keys:
            s = self.series[key]
            if split == 'train':
                start = self.rs.randint(0, self.max_start)
                s = s[start:start+self.total_length]
                d = self.diff[key][start:start+self.total_length]
            elif split == 'valid':
                start = self.max_start
                s = s[start:start+self.total_length]
                d = self.diff[key][start:start+self.total_length]
            elif split == 'test':
                s = s[-self.total_length:]
                d = self.diff[key][-self.total_length:]
            series_list.append(s)
            diff_list.append(d)

        series = torch.stack(series_list, dim=0)
        # series.shape == [batch_size, total_length]

        diffs = torch.stack(diff_list, dim=0)
        # diffs = series.new_tensor(diff_list)

        sources = series[:, :self.backcast_length]
        # sources_diff = diffs[:, :self.backcast_length]
        targets = series[:, -self.forecast_length:]

        # targets_diff = diffs[:, -self.forecast_length:]
        X = torch.log1p(sources.clamp(min=0))
        # X = sources

        if self.max_neighbours == 0:
            _, X = self.net(X)
        else:
            neighbours, neighbor_lens, mask_list = self._get_neighbours(
                keys, split)
            X_neighs = torch.log1p(neighbours)
            _, X = self.net(X, X_neighs, neighbor_lens, mask_list)

        # X.shape == [batch_size, forecast_len]

        # loss = self.mse(X, torch.log1p(targets.clamp(min=0)))
        t = torch.log1p(targets)
        numerator = torch.abs(t - X)
        denominator = torch.abs(t) + torch.abs(X)
        loss = numerator / denominator
        loss[torch.isnan(loss)] = 0
        loss = loss.mean()

        # loss = self.mse(X, targets)
        out_dict['loss'] = loss

        # During evaluation, we compute one time step at a time
        if splits[0] in ['test']:
            preds = torch.exp(X) - 1  # + diffs[:, -self.forecast_length:]
            # preds = X
            smapes, daily_errors = get_smape(targets, preds)

            out_dict['smapes'] = smapes
            out_dict['daily_errors'] = daily_errors
            out_dict['keys'] = keys
            out_dict['preds'] = preds.cpu().numpy().tolist()

        return out_dict

    @classmethod
    def get_params(cls, vocab: Vocabulary, params: Params) -> Dict[str, Any]:

        params_dict: Dict[str, Any] = {}

        params_dict['initializer'] = InitializerApplicator.from_params(
            params.pop('initializer', None))

        return params_dict
