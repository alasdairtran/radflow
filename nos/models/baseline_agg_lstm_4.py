import logging
import os
import pickle
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Dict, List

import h5py
import numpy as np
import pandas as pd
import redis
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
from torch_geometric.nn import GATConv, SAGEConv
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
                 data_path: str = './data/vevo/vevo.hdf5',
                 series_len: int = 63,
                 method: str = 'previous_day',
                 forecast_length: int = 7,
                 backcast_length: int = 42,
                 test_lengths: List[int] = [7],
                 end_offset: int = 0,
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
        self.end_offset = end_offset

        self.data = h5py.File(data_path, 'r')
        self.series = self.data['views'][...]

        assert method in ['previous_day', 'previous_week']

        # Shortcut to create new tensors in the same device as the module
        self.register_buffer('_long', torch.LongTensor(1))
        self.register_buffer('_float', torch.tensor(0.1))

        self.series_len = series_len
        self.max_start = series_len - self.forecast_length * \
            2 - self.total_length - self.end_offset

        initializer(self)

    def _initialize_series(self):
        if isinstance(next(iter(self.series.values())), torch.Tensor):
            return

        for k, v in self.series.items():
            v_array = np.asarray(v)
            self.series[k] = self._float.new_tensor(v_array)

        self.max_start = len(
            self.series[k]) - self.forecast_length * 2 - self.total_length - self.end_offset

    def forward(self, keys, splits) -> Dict[str, Any]:
        # Enable anomaly detection to find the operation that failed to compute
        # its gradient.
        # torch.autograd.set_detect_anomaly(True)

        # self._initialize_series()

        # Occasionally we get duplicate keys due random sampling
        keys = sorted(set(keys))
        split = splits[0]
        B = len(keys)
        # keys.shape == [batch_size]

        self.history['_n_batches'] += 1
        self.history['_n_samples'] += B

        out_dict = {
            'loss': None,
            'sample_size': self._float.new_tensor(B),
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
        series_dict = {}
        sorted_keys = sorted(keys)
        end = start + self.total_length
        sorted_series = self.series[sorted_keys, start:end]
        for i, k in enumerate(sorted_keys):
            series_dict[sorted_keys[i]] = sorted_series[i]

        series_list = np.array([series_dict[k] for k in keys])
        series = torch.from_numpy(series_list)
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

        loss = self.mse(torch.log1p(preds.float()),
                        torch.log1p(targets.float()))
        out_dict['loss'] = loss

        # During evaluation, we compute one time step at a time
        if splits[0] in ['test']:
            smapes, daily_errors = get_smape(targets, preds)

            out_dict['smapes'] = smapes.tolist()
            out_dict['daily_errors'] = daily_errors.tolist()
            out_dict['keys'] = keys
            out_dict['preds'] = preds.cpu().numpy().tolist()
            self.history['_n_steps'] += smapes.shape[0] * smapes.shape[1]

            for k in self.test_lengths:
                self.step_history[f'smape_{k}'] += np.sum(smapes[:, :k])

        return out_dict


@Model.register('baseline_agg_lstm_4')
class BaselineAggLSTM4(BaseModel):
    def __init__(self,
                 vocab: Vocabulary,
                 agg_type: str,
                 forecast_length: int = 7,
                 backcast_length: int = 42,
                 test_lengths: List[int] = [7],
                 peek: bool = True,
                 data_path: str = './data/vevo/vevo.hdf5',
                 key2pos_path: str = './data/vevo/vevo.key2pos.pkl',
                 series_len: int = 63,
                 num_layers: int = 8,
                 hidden_size: int = 128,
                 dropout: float = 0.1,
                 max_neighbours: int = 4,
                 max_agg_neighbours: int = 4,
                 cut_off_edge_prob: float = 0.8,
                 hop_scale: int = 4,
                 neigh_sample: bool = False,
                 t_total: int = 163840,
                 variant: str = 'full',
                 static_graph: bool = False,
                 end_offset: int = 0,
                 view_missing_p: float = 0,
                 edge_missing_p: float = 0,
                 view_randomize_p: bool = True,
                 forward_fill: bool = True,
                 n_hops: int = 1,
                 initializer: InitializerApplicator = InitializerApplicator()):
        super().__init__(vocab)
        self.decoder = LSTMDecoder(hidden_size, num_layers, dropout, variant)
        self.mse = nn.MSELoss()
        self.hidden_size = hidden_size
        self.peek = peek
        self.max_neighbours = max_neighbours
        self.max_agg_neighbours = max_agg_neighbours
        self.cut_off_edge_prob = cut_off_edge_prob
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.total_length = forecast_length + backcast_length
        self.test_lengths = test_lengths
        self.t_total = t_total
        self.current_t = 0
        self.static_graph = static_graph
        self.end_offset = end_offset
        self.neigh_sample = neigh_sample
        self.evaluate_mode = False
        self.view_missing_p = view_missing_p
        self.edge_missing_p = edge_missing_p
        self.n_hops = n_hops
        self.hop_scale = hop_scale
        self.n_layers = num_layers

        # Initialising RandomState is slow!
        self.rs = np.random.RandomState(1234)
        self.edge_rs = np.random.RandomState(63242)
        self.sample_rs = np.random.RandomState(3456)
        self.view_randomize_p = view_randomize_p
        self.forward_fill = forward_fill

        self.data = h5py.File(data_path, 'r')
        self.series = self.data['views'][...]
        self.edges = self.data['edges']
        self.masks = self.data['masks']
        self.probs = self.data['probs']
        with open(key2pos_path, 'rb') as f:
            self.key2pos = pickle.load(f)

        assert agg_type in ['mean', 'none', 'attention', 'sage', 'gat']
        self.agg_type = agg_type
        if agg_type in ['mean', 'attention', 'sage', 'gat']:
            self.fc = GehringLinear(2 * self.hidden_size, 1)

        if agg_type == 'attention':
            self.attn = nn.MultiheadAttention(
                hidden_size * 2, 4, dropout=0.1, bias=True,
                add_bias_kv=True, add_zero_attn=True, kdim=None, vdim=None)
        elif agg_type == 'sage':
            self.conv = SAGEConv(hidden_size * 2, hidden_size * 2)
        elif agg_type == 'gat':
            self.conv = GATConv(hidden_size * 2, hidden_size * 2 // 4,
                                heads=4, dropout=0.1)

        if n_hops == 2:
            self.hop_rs = np.random.RandomState(4321)
            if agg_type == 'attention':
                self.attn2 = nn.MultiheadAttention(
                    hidden_size * 2, 4, dropout=0.1, bias=True,
                    add_bias_kv=True, add_zero_attn=True, kdim=None, vdim=None)

        self.series_len = series_len
        self.max_start = series_len - self.forecast_length * \
            2 - self.total_length - self.end_offset

        # Shortcut to create new tensors in the same device as the module
        self.register_buffer('_long', torch.LongTensor(1))

        initializer(self)

    def _forward_full(self, series):
        # series.shape == [batch_size, seq_len]

        X = series
        # X.shape == [batch_size, seq_len]

        X, forecast, f_parts = self.decoder(X)

        return X, forecast, f_parts

    def _get_neighbour_embeds(self, X, keys, start, total_len):
        if self.agg_type == 'none':
            return X

        parents = [-99] * len(keys)
        Xm, masks = self._construct_neighs(
            X, keys, start, total_len, 1, parents)

        if self.agg_type == 'mean':
            Xm = self._aggregate_mean(Xm, masks)
        elif self.agg_type == 'attention':
            Xm = self._aggregate_attn(X, Xm, masks, 1)
        elif self.agg_type in ['gat', 'sage']:
            Xm = self._aggregate_gat(X, Xm, masks)

        X_out = self._pool(X, Xm)
        return X_out

    def _get_training_edges(self, keys, sorted_keys, key_map, start, total_len, parents):
        sorted_edges = self.edges[sorted_keys, start:start+total_len]
        sorted_probs = self.probs[sorted_keys, start:start+total_len]
        # sorted_edges.shape == [batch_size, total_len, max_neighs]

        edge_counters = []
        for k, parent in zip(keys, parents):
            counter = Counter()
            key_edges = sorted_edges[key_map[k]]
            key_probs = sorted_probs[key_map[k]]
            for d in range(total_len):
                day_edges = key_edges[d]
                day_cdf = key_probs[d]
                if len(day_cdf) == 0:
                    continue
                n_neighs = min(self.max_neighbours, len(day_edges))
                rands = self.edge_rs.rand(n_neighs)
                rand_mask = day_cdf[None, 0] < rands[:, None]
                chosen_idx = rand_mask.sum(axis=1)
                counter.update(day_edges[chosen_idx])

            if parent in counter:
                del counter[parent]

            edge_counters.append(counter)

        return edge_counters

    def _get_test_edges(self, keys, sorted_keys, key_map, start, total_len, parents):
        sorted_edges = self.edges[sorted_keys, start:start+total_len]
        sorted_probs = self.probs[sorted_keys, start:start+total_len]
        # sorted_edges.shape == [batch_size, total_len, max_neighs]

        edge_counters = []
        for k, parent in zip(keys, parents):
            counter = Counter()
            key_edges = sorted_edges[key_map[k]]
            key_probs = sorted_probs[key_map[k]]
            for d in range(total_len):
                day_edges = key_edges[d]
                day_cdf = key_probs[d]
                cut_off = (day_cdf <= self.cut_off_edge_prob).sum()
                counter.update(day_edges[:cut_off])

            if parent in counter:
                del counter[parent]

            edge_counters.append(counter)

        return edge_counters

    def _construct_neighs(self, X, keys, start, total_len, level, parents=None):
        B, T, E = X.shape

        sorted_keys = sorted(set(keys))
        key_map = {k: i for i, k in enumerate(sorted_keys)}

        if not self.evaluate_mode:
            edge_counters = self._get_training_edges(
                keys, sorted_keys, key_map, start, total_len, parents)
        else:
            edge_counters = self._get_test_edges(
                keys, sorted_keys, key_map, start, total_len, parents)

        # First iteration: grab the top neighbours from each sample
        key_neighs = {}
        max_n_neighs = 1
        neigh_set = set()
        for i, (key, counter) in enumerate(zip(keys, edge_counters)):
            kn = set(counter)
            if not kn:
                continue

            if self.neigh_sample and not self.evaluate_mode:
                pairs = counter.items()
                candidates = np.array([p[0] for p in pairs])
                probs = np.array([p[1] for p in pairs])
                probs = probs / probs.sum()
                kn = self.sample_rs.choice(
                    candidates,
                    size=min(len(candidates), self.max_agg_neighbours),
                    replace=False,
                    p=probs,
                ).tolist()

            key_neighs[key] = list(kn)
            neigh_set |= set(kn)
            max_n_neighs = max(max_n_neighs, len(kn))

        neighs = np.zeros((B, max_n_neighs, total_len), dtype=np.float32)
        n_masks = X.new_zeros(B, max_n_neighs).bool()
        parents = np.full((B, max_n_neighs), -99, dtype=np.uint32)
        neigh_keys = X.new_full((B, max_n_neighs), -1).long()

        neigh_list = sorted(neigh_set)
        end = start + self.total_length
        neigh_map = {k: i for i, k in enumerate(neigh_list)}
        neigh_series = self.series[neigh_list, start:end].astype(np.float32)

        if self.view_missing_p > 0:
            # Don't delete test data during evaluation
            if self.evaluate_mode:
                o_series = neigh_series[:, :self.backcast_length]
            else:
                o_series = neigh_series

            if self.view_randomize_p:
                seeds = [self.epoch, int(self.history['_n_samples']),
                         level, 124241]
                view_p_rs = np.random.RandomState(seeds)
                prob = view_p_rs.uniform(0, self.view_missing_p)
            else:
                prob = self.view_missing_p
            seeds = [self.epoch, int(self.history['_n_samples']),
                     level, 52212]
            view_rs = np.random.RandomState(seeds)
            indices = view_rs.choice(np.arange(o_series.size),
                                     replace=False,
                                     size=int(round(o_series.size * prob)))
            o_series[np.unravel_index(indices, o_series.shape)] = -1

            if self.evaluate_mode:
                neigh_series[:, :self.backcast_length] = o_series
            else:
                neigh_series = o_series

        if self.forward_fill:
            mask = neigh_series == -1
            idx = np.where(~mask, np.arange(mask.shape[1]), 0)
            np.maximum.accumulate(idx, axis=1, out=idx)
            neigh_series = neigh_series[np.arange(idx.shape[0])[:, None], idx]

        neigh_series[neigh_series == -1] = 0

        for i, key in enumerate(keys):
            if key in key_neighs:
                for j, n in enumerate(key_neighs[key]):
                    neighs[i, j] = neigh_series[neigh_map[n]][:total_len]
                    parents[i, j] = key
                    n_masks[i, j] = True
                    neigh_keys[i, j] = n

        neighs = torch.from_numpy(neighs).to(X.device)
        neighs = neighs.reshape(B * max_n_neighs, total_len)
        n_masks = n_masks.reshape(B * max_n_neighs)
        parents = parents.reshape(B * max_n_neighs)
        neigh_keys = neigh_keys.reshape(B * max_n_neighs)
        # neighs.shape == [batch_size * max_n_neighs, seq_len]

        neighs = neighs[n_masks]
        parents = parents[n_masks.cpu().numpy()]
        neigh_keys = neigh_keys[n_masks]
        # neighs.shape == [neigh_batch_size, seq_len]

        if neighs.shape[0] == 0:
            masks = X.new_ones(B, 1, T).bool()
            Xm = X.new_zeros(B, 1, T, E)
            return Xm, masks

        neighs = torch.log1p(neighs)
        Xn, _, _ = self._forward_full(neighs)
        # Xn.shape == [neigh_batch_size, seq_len, hidden_size]

        if self.peek:
            Xn = Xn[:, 1:]
        else:
            Xn = Xn[:, :-1]

        if self.n_hops - level > 0:
            if not self.evaluate_mode and self.hop_scale > 1:
                size = int(round(len(Xn) / self.hop_scale))
                idx = self.hop_rs.choice(len(Xn), size=size, replace=False)
            else:
                idx = list(range(len(Xn)))

            sampled_keys = neigh_keys[idx].cpu().tolist()

            Xm_2, masks_2 = self._construct_neighs(
                Xn[idx], sampled_keys, start, total_len,
                level + 1, parents[idx])
            if self.agg_type == 'mean':
                Xm_2 = self._aggregate_mean(Xm_2, masks_2)
            elif self.agg_type == 'attention':
                Xm_2 = self._aggregate_attn(Xn[idx], Xm_2, masks_2, level + 1)
            elif self.agg_type in ['gat', 'sage']:
                Xm_2 = self._aggregate_gat(Xn[idx], Xm_2, masks_2)
            Xn[idx] = self._pool(Xn[idx], Xm_2)

        _, S, E = Xn.shape

        # We plus one to give us option to either peek or not
        Xm = X.new_zeros(B * max_n_neighs, S, E)
        Xm[n_masks] = Xn
        Xm = Xm.reshape(B, max_n_neighs, S, E)

        masks = np.ones((B, max_n_neighs, S), dtype=bool)
        sorted_masks = self.masks[sorted_keys]

        if self.edge_missing_p > 0:
            for sorted_mask, key in zip(sorted_masks, sorted_keys):
                seeds = [key, self.epoch, int(self.history['_n_samples']),
                         level, 124241]
                edge_rs = np.random.RandomState(seeds)
                edge_idx = (~sorted_mask).nonzero()[0]
                size = int(round(len(edge_idx) * self.edge_missing_p))
                if size > 0:
                    delete_idx = edge_rs.choice(edge_idx,
                                                replace=False,
                                                size=size)
                    sorted_mask[delete_idx] = True

        for b, key in enumerate(keys):
            if key not in key_neighs:
                continue
            n_mask = sorted_masks[key_map[key]].reshape(-1, self.series_len)
            for i, k in enumerate(key_neighs[key]):
                mask = n_mask[self.key2pos[key][k]]
                mask = mask[start:start+total_len]
                if self.peek:
                    mask = mask[1:]
                else:
                    mask = mask[:-1]
                masks[b, i] = mask

        masks = torch.from_numpy(masks).to(X.device)

        return Xm, masks

    def _pool(self, X, Xn):
        X_out = X + Xn
        return X_out

    def _aggregate_mean(self, Xn, masks):
        # X.shape == [batch_size, seq_len, hidden_size]
        # Xn.shape == [batch_size, n_neighs, seq_len, hidden_size]
        # masks.shape == [batch_size, n_neighs, seq_len]

        # Mask out irrelevant values.
        # Xn = Xn.clone()
        # Xn[masks] = 0

        # Let's just take the average
        Xn = Xn.sum(dim=1)
        # Xn.shape == [batch_size, seq_len, hidden_size]

        n_neighs = (~masks).sum(dim=1).unsqueeze(-1)
        # Avoid division by zero
        n_neighs = n_neighs.clamp(min=1)
        # n_neighs.shape == [batch_size, seq_len, 1]

        Xn = Xn / n_neighs
        # Xn.shape == [batch_size, seq_len, hidden_size]

        return Xn

    def _aggregate_attn(self, X, Xn, masks, level):
        # X.shape == [batch_size, seq_len, hidden_size]
        # Xn.shape == [batch_size, n_neighs, seq_len, hidden_size]
        # masks.shape == [batch_size, n_neighs, seq_len]

        B, N, T, E = Xn.shape

        X = X.reshape(1, B * T, E)
        # X.shape == [1, batch_size * seq_len, hidden_size]

        Xn = Xn.transpose(0, 1).reshape(N, B * T, E)
        # Xn.shape == [n_neighs, batch_size * seq_len, hidden_size]

        key_padding_mask = masks.transpose(1, 2).reshape(B * T, N)
        # key_padding_mask.shape == [n_neighs, batch_size  * seq_len]

        if level == 1:
            X_attn, _ = self.attn(X, Xn, Xn, key_padding_mask, False)
        elif level == 2:
            X_attn, _ = self.attn2(X, Xn, Xn, key_padding_mask, False)

        # X_attn.shape == [1, batch_size * seq_len, hidden_size]

        X_out = X_attn.reshape(B, T, E)

        X_out = F.gelu(X_out).type_as(X)

        return X_out

    def _aggregate_gat(self, X, Xn, masks):
        # X.shape == [batch_size, seq_len, hidden_size]
        # Xn.shape == [batch_size, n_neighs, seq_len, hidden_size]
        # masks.shape == [batch_size, n_neighs, seq_len]

        B, N, T, E = Xn.shape

        Xn = Xn.reshape(B * N * T, E)
        # Xn.shape == [batch_size * n_neighs * seq_len, hidden_size]

        X_in = X.reshape(B * T, E)
        # X_in.shape == [batch_size * seq_len, hidden_size]

        # The indices 0...(BT - 1) will enumerate the central nodes
        # The indices BT...(BT + BNT - 1) will enumerate the neighbours

        # Add self-loops to central nodes
        sources = [i for i in range(B * T)]
        targets = [i for i in range(B * T)]

        for b in range(B):
            for t in range(T):
                for n in range(N):
                    if not masks[b, n, t]:
                        sources.append(B * T + N * T * b + T * n + t)
                        targets.append(T * b + t)

        edges = torch.tensor([sources, targets]).to(X.device)
        nodes = torch.cat([X_in, Xn], dim=0)
        # nodes.shape == [BT + BNT, hidden_size]

        nodes = self.conv(nodes, edges)
        # nodes.shape == [BT + BNT, hidden_size]

        nodes = nodes[:B * T]
        # nodes.shape == [BT, hidden_size]

        # nodes = F.elu(nodes)
        # nodes.shape == [BT, hidden_size]

        X_agg = nodes.reshape(B, T, E)
        # X_agg.shape == [batch_size, seq_len, hidden_size]

        X_agg = F.gelu(X_agg).type_as(X_agg)

        return X_agg

    def forward(self, keys, splits) -> Dict[str, Any]:
        # Enable anomaly detection to find the operation that failed to compute
        # its gradient.
        # torch.autograd.set_detect_anomaly(True)

        # self._initialize_series()

        # Occasionally we get duplicate keys due random sampling
        keys = sorted(set(keys))
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
        end = start + self.total_length
        series = self.series[keys, start:end].astype(np.float32)

        if self.view_missing_p > 0:
            # Don't delete test data during evaluation
            if self.evaluate_mode:
                o_series = series[:, :self.backcast_length]
            else:
                o_series = series

            if self.view_randomize_p:
                seeds = [self.epoch, int(self.history['_n_samples']), 6235]
                view_p_rs = np.random.RandomState(seeds)
                prob = view_p_rs.uniform(0, self.view_missing_p)
            else:
                prob = self.view_missing_p
            seeds = [self.epoch, int(self.history['_n_samples']), 12421]
            view_rs = np.random.RandomState(seeds)
            indices = view_rs.choice(np.arange(o_series.size),
                                     replace=False,
                                     size=int(round(o_series.size * prob)))
            o_series[np.unravel_index(indices, o_series.shape)] = -1

            if self.evaluate_mode:
                series[:, :self.backcast_length] = o_series
            else:
                series = o_series

        non_missing_idx = torch.from_numpy(series[:, 1:] != -1).to(p.device)

        if self.forward_fill:
            mask = series == -1
            idx = np.where(~mask, np.arange(mask.shape[1]), 0)
            np.maximum.accumulate(idx, axis=1, out=idx)
            series = series[np.arange(idx.shape[0])[:, None], idx]

        series[series == -1] = 0
        raw_series = torch.from_numpy(series).to(p.device)
        # raw_series.shape == [batch_size, seq_len]

        # non_missing_idx = torch.stack(non_missing_list, dim=0)[:, 1:]

        log_raw_series = torch.log1p(raw_series)

        series = torch.log1p(raw_series)

        X_full, preds_full, _ = self._forward_full(series)
        X = X_full[:, :-1]
        preds = preds_full[:, :-1]
        # X.shape == [batch_size, seq_len, hidden_size]

        if self.agg_type != 'none':
            X_agg = self._get_neighbour_embeds(
                X, keys, start, self.total_length)
            # X_agg.shape == [batch_size, seq_len, out_hidden_size]

            X_agg = self.fc(X_agg)
            # X_agg.shape == [batch_size, seq_len, 1]

            preds = preds + X_agg.squeeze(-1)
            # preds.shape == [batch_size, seq_len]

        preds = torch.exp(preds)
        targets = raw_series[:, 1:]

        preds = torch.masked_select(preds, non_missing_idx)
        targets = torch.masked_select(targets, non_missing_idx)

        numerator = torch.abs(targets - preds)
        denominator = torch.abs(targets) + torch.abs(preds)
        loss = numerator / denominator
        loss[torch.isnan(loss)] = 0
        loss = loss.mean()
        out_dict['loss'] = loss

        # During evaluation, we compute one time step at a time
        if split in ['valid', 'test']:
            s = self.backcast_length
            e = s + self.forecast_length
            targets = raw_series[:, s:e]
            # targets.shape == [batch_size, forecast_len]

            preds = targets.new_zeros(*targets.shape)

            series = log_raw_series[:, :-self.forecast_length]
            current_views = series[:, -1]
            all_f_parts = [[[] for _ in range(self.n_layers + 1)]
                           for _ in keys]
            for i in range(self.forecast_length):
                X, pred, f_parts = self._forward_full(series)
                pred = pred[:, -1]
                for b in range(len(keys)):
                    for l, f_part in enumerate(f_parts):
                        all_f_parts[b][l].append(f_part[b])
                if self.agg_type != 'none':
                    seq_len = self.total_length - self.forecast_length + i + 1
                    X_agg = self._get_neighbour_embeds(
                        X, keys, start, seq_len)
                    X_agg = self.fc(X_agg)
                    X_agg = X_agg.squeeze(-1)[:, -1]
                    pred = pred + X_agg
                    # delta.shape == [batch_size]

                    for b, f in enumerate(X_agg.cpu().tolist()):
                        all_f_parts[b][-1].append(f)

                current_views = pred
                preds[:, i] = current_views
                series = torch.cat(
                    [series, current_views.unsqueeze(-1)], dim=-1)

            preds = torch.exp(preds)
            smapes, daily_errors = get_smape(targets, preds)

            out_dict['smapes'] = smapes.tolist()
            out_dict['daily_errors'] = daily_errors.tolist()
            out_dict['keys'] = keys
            out_dict['preds'] = preds.cpu().numpy().tolist()
            out_dict['f_parts'] = all_f_parts
            self.history['_n_steps'] += smapes.shape[0] * smapes.shape[1]

            for k in self.test_lengths:
                self.step_history[f'smape_{k}'] += np.sum(smapes[:, :k])
        else:
            self.current_t += 1

        return out_dict


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size, n_layers, dropout, variant):
        super().__init__()
        self.variant = variant
        self.in_proj = GehringLinear(1, hidden_size)
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            self.layers.append(LSTMLayer(hidden_size, dropout, variant))

        self.out_f = GehringLinear(hidden_size, 1)

    def forward(self, X):
        # X.shape == [batch_size, seq_len]
        # yn.shape == [batch_size, n_neighs, seq_len]

        X = X.unsqueeze(-1)
        # X.shape == [batch_size, seq_len, 1]

        X = self.in_proj(X)
        # X.shape == [batch_size, seq_len, hidden_size]

        forecast = X.new_zeros(*X.shape)
        hidden = X.new_zeros(X.shape[0], X.shape[1], 2 * X.shape[2])
        f_parts = []

        for layer in self.layers:
            h, b, f = layer(X)
            X = X - b
            hidden = hidden + torch.cat([h, b], dim=-1)
            forecast = forecast + f

            if not self.training:
                f_part = self.out_f(f[:, -1]).squeeze(-1)
                f_parts.append(f_part.cpu().tolist())

        hidden = hidden / len(self.layers)

        # h = torch.cat(h_list, dim=-1)
        # h.shape == [batch_size, seq_len, n_layers * hidden_size]

        # h = self.out_proj(h)
        # h.shape == [batch_size, seq_len, hidden_size]

        f = self.out_f(forecast).squeeze(-1)

        return hidden, f, f_parts


class LSTMLayer(nn.Module):
    def __init__(self, hidden_size, dropout, variant):
        super().__init__()
        self.variant = variant

        self.layer = nn.LSTM(hidden_size, hidden_size, 1,
                             batch_first=True, dropout=dropout)

        self.proj_f = GehringLinear(hidden_size, hidden_size)
        self.proj_b = GehringLinear(hidden_size, hidden_size)
        self.out_f = GehringLinear(hidden_size, hidden_size)
        self.out_b = GehringLinear(hidden_size, hidden_size)

    def forward(self, X):
        # X.shape == [batch_size, seq_len]
        # yn.shape == [batch_size, n_neighs, seq_len]

        X, _ = self.layer(X)
        # X.shape == [batch_size, seq_len, hidden_size]

        b = self.out_b(F.gelu(self.proj_b(X)))
        f = self.out_f(F.gelu(self.proj_f(X)))
        # b.shape == f.shape == [batch_size, seq_len, hidden_size]

        return X, b, f
