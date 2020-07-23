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
from pymongo import MongoClient
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


@Model.register('nbeats_wiki')
class NBEATSWiki(BaseModel):
    def __init__(self,
                 vocab: Vocabulary,
                 database: str = 'vevo',
                 collection: str = 'graph',
                 series_len: int = 63,
                 forecast_length: int = 28,
                 backcast_length: int = 224,
                 test_lengths: List[int] = [7],
                 max_neighbours: int = 0,
                 hidden_size: int = 128,
                 dropout: float = 0.2,
                 n_stacks: int = 16,
                 nb_blocks_per_stack: int = 1,
                 missing_p: float = 0.0,
                 thetas_dims: int = 128,
                 share_weights_in_stack: bool = False,
                 peek: bool = False,
                 net: str = 'nbeats',
                 end_offset: int = 0,
                 initializer: InitializerApplicator = InitializerApplicator()):
        super().__init__(vocab)
        self.mse = nn.MSELoss()
        self.max_neighbours = max_neighbours
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.total_length = backcast_length + forecast_length
        self.test_lengths = test_lengths
        self.rs = np.random.RandomState(1234)
        self.hidden_size = hidden_size
        self.device = torch.device('cuda:0')
        self.missing_p = missing_p
        self.end_offset = end_offset
        initializer(self)

        client = MongoClient(host='localhost', port=27017)
        db = client[database]
        self.col = db[collection]

        self.series_len = series_len
        self.max_start = series_len - self.forecast_length * \
            2 - self.total_length - self.end_offset

        if net == 'nbeats':
            self.net = NBeatsNet(device=torch.device('cuda:0'),
                                 stack_types=[
                                     NBeatsNet.GENERIC_BLOCK] * n_stacks,
                                 nb_blocks_per_stack=nb_blocks_per_stack,
                                 forecast_length=forecast_length,
                                 backcast_length=backcast_length,
                                 thetas_dims=[thetas_dims] * n_stacks,
                                 hidden_layer_units=hidden_size,
                                 share_weights_in_stack=share_weights_in_stack,
                                 dropout=dropout,
                                 max_neighbours=max_neighbours,
                                 peek=peek,
                                 )
        elif net == 'plus':
            self.net = NBeatsPlusNet(device=torch.device('cuda:0'),
                                     stack_types=[
                                     NBeatsNet.GENERIC_BLOCK] * n_stacks,
                                     nb_blocks_per_stack=nb_blocks_per_stack,
                                     forecast_length=forecast_length,
                                     backcast_length=backcast_length,
                                     thetas_dims=[thetas_dims] * n_stacks,
                                     hidden_layer_units=hidden_size,
                                     share_weights_in_stack=share_weights_in_stack,
                                     dropout=dropout,
                                     max_neighbours=max_neighbours,
                                     peek=peek,
                                     )

        # Shortcut to create new tensors in the same device as the module
        self.register_buffer('_long', torch.LongTensor(1))
        self.register_buffer('_float', torch.FloatTensor(1))

    def _initialize_series(self):
        if isinstance(next(iter(self.series.values())), torch.Tensor):
            return

        # Check how long the time series is
        # series_len = len(next(iter(self.series.values())))
        # n_weeks = series_len // 7 + 1

        # Remove trends using the first year of data
        # train_len = self.backcast_length
        # for k, v in self.series.items():
        #     v_full = np.array(v[:train_len]).reshape(train_len // 7, 7)
        #     avg_all = v_full.mean()
        #     avg_week = v_full.mean(axis=0)
        #     diff = avg_week - avg_all
        #     diff = np.tile(diff, n_weeks)
        #     diff = diff[:len(v)]
        #     # self.series[k] = v - diff
        #     self.diff[k] = diff
        #     # self.diff[k] = max(v[:train_len])

        # Compute correlation
        # for node, neighs in self.in_degrees.items():
        #     x = self.series[node][:self.backcast_length]
        #     corrs = []
        #     for neigh in neighs:
        #         y = self.series[neigh][:self.backcast_length]
        #         r = np.corrcoef(x, y)[0, 1]
        #         corrs.append(r)
        #     keep_idx = np.argsort(corrs)[::-1][:self.max_neighbours]
        #     self.in_degrees[node] = np.array(neighs)[keep_idx]

        p = next(self.parameters())
        for k, v in self.series.items():
            self.series[k] = np.asarray(v).astype(float)

        if self.max_neighbours > 0:
            for k, v in self.neighs.items():
                for t in v.keys():
                    self.neighs[k][t] = self.neighs[k][t]

            # Sort by view counts
            logger.info('Processing edges')
            self.mask_dict = {}
            for node, neighs in tqdm(self.in_degrees.items()):
                neigh_dict = {}
                node_masks = np.ones((len(neighs), len(self.series[node])),
                                     dtype=bool)
                for i, n in enumerate(neighs):
                    node_masks[i] = n['mask']
                    neigh_dict[n['id']] = i
                self.in_degrees[node] = neigh_dict
                self.mask_dict[node] = p.new_tensor(node_masks)

        for k, v in self.series.items():
            v_array = np.asarray(v)
            self.series[k] = p.new_tensor(v_array)

        self.max_start = len(
            self.series[k]) - self.forecast_length * 2 - self.total_length - self.end_offset

        # for k, v in self.series.items():
        #     if self.missing_p > 0:
        #         size = len(v) - self.forecast_length
        #         indices = self.rs.choice(np.arange(1, size), replace=False,
        #                                  size=int(size * self.missing_p))
        #         self.series[k][indices] = np.nan

        #         mask = np.isnan(self.series[k])
        #         idx = np.where(~mask, np.arange(len(mask)), 0)
        #         np.maximum.accumulate(idx, out=idx)
        #         self.series[k][mask] = self.series[k][idx[mask]]

        #     self.series[k] = p.new_tensor(self.series[k])
        #     # self.diff[k] = p.new_tensor(self.diff[k])

    def _get_neighbours(self, keys, split, start):
        # neighbor_lens = []
        # mask_list = []

        sources = self._float.new_zeros(
            (len(keys), self.max_neighbours, self.backcast_length))
        targets = self._float.new_zeros(
            (len(keys), self.max_neighbours, self.forecast_length))
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
                sources[i, j] = s[:self.backcast_length]
                targets[i, j] = s[self.backcast_length:]
                masks[i, j] = 0

        # sources.shape == [batch_size, max_neighbours, backcast_length]
        # sources = sources.reshape(
        #     len(keys) * self.max_neighbours, self.backcast_length)

        # neighbor_lens = torch.tensor(neighbor_lens)
        # sources.shape == [batch_size, 1]

        # mask_list = torch.tensor(mask_list)

        return sources, targets, masks

    def forward(self, keys, splits) -> Dict[str, Any]:
        # Enable anomaly detection to find the operation that failed to compute
        # its gradient.
        # torch.autograd.set_detect_anomaly(True)

        # self._initialize_series()

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

        if split == 'train':
            if self.max_start == 0:
                start = 0
            else:
                start = self.rs.randint(0, self.max_start)
        elif split == 'valid':
            start = self.max_start + self.forecast_length
        elif split == 'test':
            start = self.max_start + self.forecast_length * 2

        # Find all series of given keys
        query = {'_id': {'$in': keys}}
        projection = {'s': {'$slice': [start, self.total_length]}}
        cursor = self.col.find(query, projection, batch_size=len(keys))
        series_dict = {}
        for page in cursor:
            key = int(page['_id'])
            series = np.array(page['s'])
            series_dict[key] = series

        series_list = np.array([series_dict[k]
                                for k in keys], dtype=np.float32)
        series = torch.from_numpy(series_list).to(p.device)

        # diffs = torch.stack(diff_list, dim=0)
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
            n_sources, n_targets, neighbour_masks = self._get_neighbours(
                keys, split, start)
            X_neighs = torch.log1p(n_sources)
            y_neighs = torch.log1p(n_targets)
            _, X = self.net(X, X_neighs, y_neighs, neighbour_masks)

        # X.shape == [batch_size, forecast_len]

        # loss = self.mse(X, torch.log1p(targets.clamp(min=0)))
        preds = torch.exp(X)
        numerator = torch.abs(targets - preds)
        denominator = torch.abs(targets) + torch.abs(preds)
        loss = numerator / denominator
        loss[torch.isnan(loss)] = 0
        loss = loss.mean()

        # loss = self.mse(X, targets)
        out_dict['loss'] = loss

        # During evaluation, we compute one time step at a time
        if splits[0] in ['valid', 'test']:
            smapes, daily_errors = get_smape(targets, preds)

            out_dict['smapes'] = smapes.tolist()
            out_dict['daily_errors'] = daily_errors.tolist()
            out_dict['keys'] = keys
            out_dict['preds'] = preds.cpu().numpy().tolist()
            self.history['_n_steps'] += smapes.shape[0] * smapes.shape[1]

            for k in self.test_lengths:
                self.step_history[f'smape_{k}'] += np.sum(smapes[:, :k])

        return out_dict

    @classmethod
    def get_params(cls, vocab: Vocabulary, params: Params) -> Dict[str, Any]:

        params_dict: Dict[str, Any] = {}

        params_dict['initializer'] = InitializerApplicator.from_params(
            params.pop('initializer', None))

        return params_dict
