import logging
import math
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


@Model.register('baseline_agg_lstm_2')
class BaselineAggLSTM2(BaseModel):
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
                 max_eval_neighbours: int = 32,
                 hop_scale: int = 4,
                 neigh_sample: bool = False,
                 t_total: int = 163840,
                 static_graph: bool = False,
                 end_offset: int = 0,
                 view_missing_p: float = 0,
                 edge_missing_p: float = 0,
                 view_randomize_p: bool = True,
                 forward_fill: bool = True,
                 n_hops: int = 1,
                 initializer: InitializerApplicator = InitializerApplicator()):
        super().__init__(vocab)
        self.mse = nn.MSELoss()
        self.hidden_size = hidden_size
        self.peek = peek
        self.max_neighbours = max_neighbours
        self.max_agg_neighbours = max_agg_neighbours
        self.max_eval_neighbours = max_eval_neighbours
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.total_length = forecast_length + backcast_length
        self.test_lengths = test_lengths
        self.t_total = t_total
        self.current_t = 0
        self.static_graph = static_graph
        self.end_offset = end_offset
        self.neigh_sample = neigh_sample
        self.rs = np.random.RandomState(1234)
        self.sample_rs = np.random.RandomState(3456)
        self.view_randomize_p = view_randomize_p
        self.forward_fill = forward_fill

        self.evaluate_mode = False
        self.view_missing_p = view_missing_p
        self.edge_missing_p = edge_missing_p
        self.n_hops = n_hops
        self.hop_scale = hop_scale

        self.data = h5py.File(data_path, 'r')
        self.series = self.data['views']
        self.edges = self.data['edges']
        self.masks = self.data['masks']
        self.probs = self.data['probs']

        with open(key2pos_path, 'rb') as f:
            self.key2pos = pickle.load(f)

        self.decoder = nn.LSTM(1, hidden_size, num_layers,
                               bias=True, batch_first=True, dropout=dropout)

        assert agg_type in ['mean', 'none', 'attention', 'sage', 'gat']
        self.agg_type = agg_type
        self.fc = GehringLinear(self.hidden_size, 1)

        if agg_type in ['mean', 'attention', 'sage', 'gat']:
            self.out_proj = GehringLinear(
                self.hidden_size * 2, self.hidden_size)

        if agg_type == 'attention':
            self.attn = nn.MultiheadAttention(
                self.hidden_size, 4, dropout=0.1, bias=True,
                add_bias_kv=True, add_zero_attn=True, kdim=None, vdim=None)
        elif agg_type == 'sage':
            self.conv = SAGEConv(hidden_size * 3, hidden_size * 3)
        elif agg_type == 'gat':
            self.conv = GATConv(hidden_size * 3, hidden_size * 3 // 4,
                                heads=4, dropout=0.1)

        if n_hops == 2:
            self.proj_hop_2 = GehringLinear(
                6 * self.hidden_size, 3 * self.hidden_size)
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

    def _initialize_series(self):
        if isinstance(self.series, torch.Tensor):
            return

        p = next(self.parameters())
        for k, v in self.series.items():
            v = np.asarray(v).astype(float)
            # Fill out missing values
            # mask = v == -1
            # idx = np.where(~mask, np.arange(len(mask)), 0)
            # np.maximum.accumulate(idx, out=idx)
            # v[mask] = v[idx[mask]]

            # Replace remaining initial views with 0
            v[v == -1] = 0

            self.series[k] = v

        if self.agg_type != 'none':
            # Sort by view counts
            logger.info('Processing edges')
            self.mask_dict = {}
            for node, neighs in tqdm(self.in_degrees.items()):
                node_rs = np.random.RandomState(node + 1234567)
                neigh_dict = {}
                seq_len = len(self.series[node])
                max_size = seq_len - self.end_offset - self.forecast_length
                node_masks = np.ones((len(neighs), seq_len), dtype=bool)
                for i, n in enumerate(neighs):
                    mask = np.asarray(n['mask'][:seq_len])
                    if self.edge_missing_p > 0:
                        edge_rs = np.random.RandomState(
                            node_rs.randint(0, 1000000))
                        edge_idx = (~mask[:max_size]).nonzero()[0]
                        size = int(round(len(edge_idx) * self.edge_missing_p))
                        if size > 0:
                            delete_idx = edge_rs.choice(edge_idx,
                                                        replace=False,
                                                        size=size)
                            mask[delete_idx] = True

                    node_masks[i] = mask
                    neigh_dict[n['id']] = i
                self.in_degrees[node] = neigh_dict
                self.mask_dict[node] = torch.from_numpy(
                    node_masks).to(p.device)

        logger.info('Processing series')
        series_matrix = np.zeros((len(self.series),
                                  len(self.series[k])))
        non_missing_matrix = np.ones((len(self.series),
                                      len(self.series[k])), dtype=bool)
        self.series_map = {}
        for i, (k, v) in enumerate(tqdm(self.series.items())):
            if self.view_missing_p > 0:
                view_rs = np.random.RandomState(k)
                max_size = len(v) - self.end_offset - self.forecast_length
                size = int(round(max_size * self.view_missing_p))
                indices = view_rs.choice(np.arange(max_size),
                                         replace=False,
                                         size=size)
                v[indices] = 0  # np.nan
                non_missing_idx = np.ones(len(v), dtype=bool)
                non_missing_idx[indices] = False
                # mask = np.isnan(v)
                # idx = np.where(~mask, np.arange(len(mask)), 0)
                # np.maximum.accumulate(idx, out=idx)
                # v[mask] = v[idx[mask]]
                non_missing_matrix[i] = non_missing_idx

            series_matrix[i] = v
            self.series_map[k] = i
        self.series = p.new_tensor(series_matrix)
        self.non_missing = torch.from_numpy(non_missing_matrix).to(p.device)

        # If a neighbour has a missing view on day t, all outgoing edges
        # will also be deleted.
        if self.agg_type != 'none':
            if self.view_missing_p > 0:
                logger.info('Updating neighbour masks.')
                for key in tqdm(self.mask_dict):
                    for n, i in self.in_degrees[key].items():
                        mask_1 = self.mask_dict[key][i]
                        mask_2 = ~self.non_missing[self.series_map[n]]
                        self.mask_dict[key][i] = mask_1 | mask_2

            if self.view_missing_p > 0 or self.edge_missing_p > 0:
                logger.info(
                    'Removing edges from neighbours with missing views.')
                for key in tqdm(self.neighs):
                    for day in self.neighs[key]:
                        neighs = self.neighs[key][day]
                        k = self.series_map[key]
                        neighs = [n for n in neighs
                                  if not self.mask_dict[key][self.in_degrees[key][n]][day]]
                        self.neighs[key][day] = neighs

    def _forward_full(self, series):
        # series.shape == [batch_size, seq_len]

        inputs = series

        X = inputs.unsqueeze(-1)
        # X.shape == [batch_size, seq_len, 1]

        X, _ = self.decoder(X)

        return X

    def _get_neighbour_embeds(self, X, keys, start, total_len):
        if self.agg_type == 'none':
            return X

        Xm, masks = self._construct_neighs(
            X, keys, start, total_len, 1)

        if self.agg_type == 'mean':
            Xm = self._aggregate_mean(Xm, masks)
        elif self.agg_type == 'attention':
            Xm = self._aggregate_attn(X, Xm, masks, 1)
        elif self.agg_type in ['gat', 'sage']:
            Xm = self._aggregate_gat(X, Xm, masks)

        X_out = self._pool(X, Xm)
        return X_out

    def _get_training_edges(self, keys, sorted_keys, key_map, start, total_len, level):
        edges = np.full((len(keys), total_len, self.max_neighbours),
                        -1, np.int32)

        sorted_edges = self.edges[sorted_keys, start:start+total_len]
        sorted_probs = self.probs[sorted_keys, start:start+total_len]
        # sorted_edges.shape == [batch_size, total_len, max_neighs]

        # Initialising RandomState is slow!
        seeds = [self.epoch, int(self.history['_n_samples']), level, 24124]
        edge_rs = np.random.RandomState(seeds)

        for i, k in enumerate(keys):
            key_edges = sorted_edges[key_map[k]]
            key_probs = sorted_probs[key_map[k]]
            for d in range(total_len):
                day_edges = key_edges[d]
                day_probs = key_probs[d]
                n_neighs = min(self.max_neighbours, (day_probs > 0).sum())
                if n_neighs == 0:
                    continue

                day_cdf = day_probs.cumsum()
                rands = edge_rs.rand(n_neighs)
                rand_mask = day_cdf[None, 0] < rands[:, None]
                chosen_idx = rand_mask.sum(axis=1)
                edges[i, d, :n_neighs] = day_edges[chosen_idx]

        return edges

    def _get_test_edges(self, keys, sorted_keys, key_map, start, total_len):
        edges = np.full((len(keys), total_len, self.max_eval_neighbours),
                        -1, np.int32)

        sorted_edges = self.edges[sorted_keys, start:start+total_len]
        # sorted_edges.shape == [batch_size, total_len, max_neighs]

        for i, k in enumerate(keys):
            key_edges = sorted_edges[key_map[k]]
            for d in range(total_len):
                day_edges = key_edges[d][:self.max_eval_neighbours]
                n_neighs = len(day_edges)
                edges[i, d, :n_neighs] = day_edges

        return edges

    def _construct_neighs(self, X, keys, start, total_len, level, parents=None):
        B, T, E = X.shape

        sorted_keys = sorted(set(keys))
        key_map = {k: i for i, k in enumerate(sorted_keys)}

        if not self.evaluate_mode:
            edges = self._get_training_edges(
                keys, sorted_keys, key_map, start, total_len, level)
        else:
            edges = self._get_test_edges(
                keys, sorted_keys, key_map, start, total_len)

        # Mask out parents
        if parents is None:
            parents = np.full_like(edges, -99, dtype=np.int32)
        edges[edges == parents] = -1

        # First iteration: grab the top neighbours from each sample
        key_neighs = {}
        max_n_neighs = 1
        neigh_set = set()
        for i, (key, neighs) in enumerate(zip(keys, edges)):
            # neighs.shape == [total_len, max_neighs]
            non_empty_mask = neighs != -1
            if not non_empty_mask.any():
                continue

            counter = Counter(neighs[non_empty_mask])
            kn = set(counter)

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
                view_p_rs = np.random.RandomState(
                    (524288 * self.epoch) + int(self.history['_n_samples']) + 124241 * level)
                prob = view_p_rs.uniform(0, self.view_missing_p)
            else:
                prob = self.view_missing_p
            view_rs = np.random.RandomState(
                (524288 * self.epoch) + int(self.history['_n_samples']) + 52212 * level)
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
        parents = parents[n_masks.cpu().numpy(), np.newaxis, np.newaxis]
        neigh_keys = neigh_keys[n_masks]
        # neighs.shape == [neigh_batch_size, seq_len]

        if neighs.shape[0] == 0:
            masks = X.new_ones(B, 1, T).bool()
            Xm = X.new_zeros(B, 1, T, E)
            return Xm, masks

        neighs = torch.log1p(neighs)
        Xn = self._forward_full(neighs)
        # Xn.shape == [batch_size * max_n_neighs, seq_len, hidden_size]

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
            X_out = self._pool(Xn[idx], Xm_2)
            Xn[idx] = F.gelu(self.proj_hop_2(X_out)).type_as(Xn)

        _, S, E = Xn.shape

        # We plus one to give us option to either peek or not
        Xm = X.new_zeros(B * max_n_neighs, S, E)
        Xm[n_masks] = Xn
        Xm = Xm.reshape(B, max_n_neighs, S, E)

        masks = np.ones((B, max_n_neighs, S), dtype=bool)
        sorted_masks = self.masks[sorted_keys]

        if self.edge_missing_p > 0:
            for sorted_mask, key in zip(sorted_masks, sorted_keys):
                edge_rs = np.random.RandomState(
                    key + (524288 * self.epoch) + int(self.history['_n_samples']) + 124241 * level)
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
        X_out = torch.cat([X, Xn], dim=-1)
        # Xn.shape == [batch_size, seq_len, 2 * hidden_size]

        X_out = F.relu(self.out_proj(X_out))
        # Xn.shape == [batch_size, seq_len, hidden_size]

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

    def _aggregate_attn(self, Xn, X, masks, level):
        # X.shape == [batch_size, seq_len, hidden_size]
        # Xn.shape == [batch_size, n_neighs, seq_len, hidden_size]
        # masks.shape == [batch_size, n_neighs, seq_len]

        B, N, T, E = Xn.shape

        X = X.reshape(1, B * T, E)
        # X.shape == [1, batch_size * seq_len, hidden_size]

        Xn = Xn.transpose(0, 1).reshape(N, B * T, E)
        # Xn.shape == [n_neighs, batch_size * seq_len, hidden_size]

        key_padding_mask = masks.transpose(0, 1).reshape(N, B * T)
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
                view_p_rs = np.random.RandomState(
                    (524288 * self.epoch) + int(self.history['_n_samples']))
                prob = view_p_rs.uniform(0, self.view_missing_p)
            else:
                prob = self.view_missing_p
            view_rs = np.random.RandomState(
                (524288 * self.epoch) + int(self.history['_n_samples']) + 12421)
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

        X_full = self._forward_full(series)
        X = X_full[:, :-1]
        # X.shape == [batch_size, seq_len, hidden_size]

        X_agg = self._get_neighbour_embeds(X, keys, start, self.total_length)
        # X_agg.shape == [batch_size, seq_len, out_hidden_size]

        X_agg = self.fc(X_agg)
        # X_agg.shape == [batch_size, seq_len, 1]

        preds = X_agg.squeeze(-1)
        preds = torch.exp(preds)
        # preds.shape == [batch_size, seq_len]

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
            for i in range(self.forecast_length):
                X = self._forward_full(series)
                seq_len = self.total_length - self.forecast_length + i + 1
                X_agg = self._get_neighbour_embeds(
                    X, keys, start, seq_len)
                X_agg = self.fc(X_agg)
                pred = X_agg.squeeze(-1)[:, -1]
                # delta.shape == [batch_size]

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
            self.history['_n_steps'] += smapes.shape[0] * smapes.shape[1]

            for k in self.test_lengths:
                self.step_history[f'smape_{k}'] += np.sum(smapes[:, :k])
        else:
            self.current_t += 1

        return out_dict


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_size, n_layers, dropout, total_len):
        super().__init__()
        self.in_proj = GehringLinear(1, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(TBEATLayer(hidden_size, dropout, 4))

        pos_weights = self._get_embedding(256, hidden_size)
        self.register_buffer('pos_weights', pos_weights)

    @staticmethod
    def _get_embedding(n_embeds, embed_dim, padding_idx=0):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        max_ts = 256
        min_ts = 1
        n_timescales = embed_dim // 2
        increment = math.log(max_ts / min_ts) / (n_timescales - 1)
        # Example increment: 9 / 384 = 0.024

        timescales = torch.arange(n_timescales, dtype=torch.float)

        # inv_timescales ranges from 1 to 1/10000 with log spacing
        inv_timescales = min_ts * torch.exp(timescales * -increment)
        # inv_timescales.shape == [embed_size // 2]

        positions = torch.arange(n_embeds, dtype=torch.float).unsqueeze(1)
        # positions.shape ==  [n_embeds, 1]

        inv_timescales = inv_timescales.unsqueeze(0)
        # inv_timescales.shape == [1, embed_size // 2]

        scaled_time = positions * inv_timescales
        # scaled_time.shape == [n_embeds, embed_size // 2]

        sin_signal = torch.sin(scaled_time)
        cos_signal = torch.cos(scaled_time)
        signal = torch.cat([sin_signal, cos_signal], dim=1)
        # signal.shape == [n_embeds, embed_dim]

        # Ensure that embed_dim is even
        if embed_dim % 2 == 1:
            signal = torch.cat([signal, torch.zeros(n_embeds, 1)], dim=1)

        if padding_idx is not None:
            signal[padding_idx, :] = 0

        return signal

    def forward(self, X):
        B, T, _ = X.shape
        # X.shape == [batch_size, seq_len, 1]

        X = self.in_proj(X)
        # X.shape == [batch_size, seq_len, hidden_size]

        X = X.transpose(0, 1)
        # X.shape == [seq_len, batch_size, hidden_size]

        pos = torch.arange(1, T + 1, device=X.device).unsqueeze(-1)
        pos = pos.expand(-1, B)
        # pos.shape = [seq_len, batch_size]

        pos_embeds = self.pos_weights.index_select(0, pos.reshape(-1))
        pos_embeds = pos_embeds.reshape(T, B, -1)

        for layer in self.layers:
            X = layer(X, pos_embeds)

        X = X.transpose(0, 1)
        # X.shape == [batch_size, seq_len, hidden_size]

        return X, None


class TBEATLayer(nn.Module):
    def __init__(self, hidden_size, dropout, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, dropout)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.norm_1 = nn.LayerNorm(hidden_size)
        self.norm_2 = nn.LayerNorm(hidden_size)

        self.linear_1 = GehringLinear(hidden_size, hidden_size * 2)
        self.linear_2 = GehringLinear(hidden_size * 2, hidden_size)

        self.activation = F.gelu

    def forward(self, X, pos_embeds):
        # We can't attend positions which are True
        T, B, E = X.shape
        attn_mask = X.new_ones(T, T)
        # Zero out the diagonal and everything below
        # We can attend to ourselves and the past
        attn_mask = torch.triu(attn_mask, diagonal=1)
        # attn_mask.shape == [T, T]

        X = X + pos_embeds

        X_1, _ = self.attn(X, X, X, need_weights=False, attn_mask=attn_mask)
        # X.shape == [seq_len, batch_size, hidden_size]

        X = X + self.dropout_1(X_1)
        X = self.norm_1(X)

        X_2 = self.linear_2(self.dropout_2(self.activation(self.linear_1(X))))
        X = X + self.dropout_3(X_2)
        X = self.norm_2(X)
        # X.shape == [seq_len, batch_size, hidden_size]

        return X
