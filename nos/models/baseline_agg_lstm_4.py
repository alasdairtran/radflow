import logging
import os
import pickle
from collections import Counter, defaultdict
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
                 data_dir: str,
                 seed_word: str = 'vevo',
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

        assert method in ['previous_day', 'previous_week']

        series_path = f'{data_dir}/{seed_word}/series.pkl'
        logger.info(f'Loading {series_path} into model')
        with open(series_path, 'rb') as f:
            self.series = pickle.load(f)

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
            self.series[k]) - self.forecast_length * 2 - self.total_length - self.end_offset

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
                 max_neighbours: int = 4,
                 max_agg_neighbours: int = 4,
                 batch_as_subgraph: bool = False,
                 neigh_sample: bool = False,
                 t_total: int = 163840,
                 variant: str = 'full',
                 static_graph: bool = False,
                 end_offset: int = 0,
                 initializer: InitializerApplicator = InitializerApplicator()):
        super().__init__(vocab)
        self.decoder = LSTMDecoder(hidden_size, num_layers, dropout, variant)
        self.mse = nn.MSELoss()
        self.hidden_size = hidden_size
        self.peek = peek
        self.max_neighbours = max_neighbours
        self.max_agg_neighbours = max_agg_neighbours
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.total_length = forecast_length + backcast_length
        self.test_lengths = test_lengths
        self.batch_as_subgraph = batch_as_subgraph
        self.t_total = t_total
        self.current_t = 0
        self.static_graph = static_graph
        self.end_offset = end_offset
        self.neigh_sample = neigh_sample
        self.evaluate_mode = False

        self.max_start = None
        self.rs = np.random.RandomState(1234)
        self.sample_rs = np.random.RandomState(3456)

        assert agg_type in ['mean', 'none', 'attention', 'sage', 'gat']
        self.agg_type = agg_type
        if agg_type in ['mean', 'attention', 'sage', 'gat']:
            self.out_proj = GehringLinear(
                6 * self.hidden_size, self.hidden_size)
            self.fc = GehringLinear(self.hidden_size, 1)

        if agg_type == 'attention':
            self.attn = nn.MultiheadAttention(
                hidden_size * 3, 4, dropout=0.1, bias=True,
                add_bias_kv=True, add_zero_attn=True, kdim=None, vdim=None)
        elif agg_type == 'sage':
            self.conv = SAGEConv(hidden_size * 3, hidden_size * 3)
        elif agg_type == 'gat':
            self.conv = GATConv(hidden_size * 3, hidden_size * 3 // 4,
                                heads=4, dropout=0.1)

        series_path = f'{data_dir}/{seed_word}/series.pkl'
        logger.info(f'Loading {series_path} into model')
        with open(series_path, 'rb') as f:
            self.series = pickle.load(f)

        if self.agg_type != 'none':
            in_degrees_path = f'{data_dir}/{seed_word}/in_degrees.pkl'
            logger.info(f'Loading {in_degrees_path} into model')
            with open(in_degrees_path, 'rb') as f:
                self.in_degrees = pickle.load(f)

            neighs_path = f'{data_dir}/{seed_word}/neighs.pkl'
            logger.info(f'Loading {neighs_path} into model')
            with open(neighs_path, 'rb') as f:
                self.neighs = pickle.load(f)
        self.mask_dict = None

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
            mask = v == -1
            idx = np.where(~mask, np.arange(len(mask)), 0)
            np.maximum.accumulate(idx, out=idx)
            v[mask] = v[idx[mask]]

            # Replace remaining initial views with 0
            v[v == -1] = 0

            self.series[k] = v

        if self.agg_type != 'none':
            # Sort by view counts
            logger.info('Processing edges')
            self.mask_dict = {}
            for node, neighs in tqdm(self.in_degrees.items()):
                neigh_dict = {}
                seq_len = len(self.series[node])
                node_masks = np.ones((len(neighs), seq_len), dtype=bool)
                for i, n in enumerate(neighs):
                    node_masks[i] = n['mask'][:seq_len]
                    neigh_dict[n['id']] = i
                self.in_degrees[node] = neigh_dict
                self.mask_dict[node] = p.new_tensor(node_masks)

        logger.info('Processing series')
        series_matrix = np.zeros((len(self.series),
                                  len(self.series[k])))
        self.series_map = {}
        for i, (k, v) in enumerate(tqdm(self.series.items())):
            series_matrix[i] = np.asarray(v)
            self.series_map[k] = i
        self.series = p.new_tensor(series_matrix)

        self.max_start = len(
            self.series[i]) - self.forecast_length * 2 - self.total_length - self.end_offset

    def _forward_full(self, series):
        # series.shape == [batch_size, seq_len]

        X = series
        # X.shape == [batch_size, seq_len]

        X, forecast = self.decoder(X)

        return X, forecast

    def _get_neighbour_embeds(self, X, keys, start, total_len, X_cache=None):
        if self.agg_type == 'none':
            return X

        Xm, masks = self._construct_neighs(
            X, keys, start, total_len, X_cache)

        if self.agg_type == 'mean':
            Xm = self._aggregate_mean(Xm, masks)
        elif self.agg_type == 'attention':
            Xm = self._aggregate_attn(X, Xm, masks)
        elif self.agg_type in ['gat', 'sage']:
            Xm = self._aggregate_gat(X, Xm, masks)

        X_out = self._pool(X, Xm)
        return X_out

    def _construct_neighs(self, X, keys, start, total_len, X_cache):
        B, T, E = X.shape
        batch_set = set(keys)

        # First iteration: grab the top neighbours from each sample
        key_neighs = {}
        max_n_neighs = 1
        all_neigh_keys = set()
        for key in keys:
            if key in self.neighs:
                kn = set(self.neighs[key][start]) \
                    if self.static_graph else set()
                counter = Counter()
                for day in range(start, start+total_len):
                    if self.training and self.batch_as_subgraph:
                        kn |= set(self.neighs[key][day]) & batch_set
                    elif self.static_graph:
                        kn &= set(self.neighs[key][day])
                    elif self.neigh_sample:
                        candidates = set(
                            self.neighs[key][day][:self.max_neighbours])
                        kn |= candidates
                        counter.update(candidates)
                    else:
                        kn |= set(self.neighs[key][day][:self.max_neighbours])

                if self.static_graph:
                    views = {k: self.series[self.series_map[k],
                                            start:start+self.backcast_length].sum().cpu().item()
                             for k in kn}
                    sorted_kn = sorted(views.items(), key=lambda x: x[1])
                    kn = set(p[0] for p in sorted_kn[:self.max_neighbours])
                elif self.neigh_sample and not self.evaluate_mode:
                    pairs = counter.items()
                    candidates = np.array([p[0] for p in pairs])
                    probs = np.array([p[1] for p in pairs])
                    probs = probs / probs.sum()
                    kn = set(self.sample_rs.choice(
                        candidates,
                        size=min(len(candidates), self.max_agg_neighbours),
                        replace=False,
                        p=probs,
                    ))
                # elif self.neigh_sample:
                #     pairs = counter.most_common(self.max_agg_neighbours)
                #     candidates = [p[0] for p in pairs]
                #     kn = set(candidates)

                key_neighs[key] = kn
                all_neigh_keys |= kn
                max_n_neighs = max(max_n_neighs, len(kn))

        all_neigh_keys = list(all_neigh_keys)
        all_neigh_dict = {k: i for i, k in enumerate(all_neigh_keys)}

        if not all_neigh_keys:
            masks = X.new_ones(B, 1, T).bool()
            Xm = X.new_zeros(B, 1, T, E)
            return Xm, masks

        if self.training and self.batch_as_subgraph and X_cache is not None:
            cache_pos = {k: i for i, k in enumerate(keys)}
            Xn_list = []
            for key in all_neigh_keys:
                Xn_list.append(X_cache[cache_pos[key]])
            Xn = torch.stack(Xn_list, dim=0)
        else:
            neigh_series_list = []
            for key in all_neigh_keys:
                neigh_series_list.append(
                    self.series[self.series_map[key], start:start+total_len])
            neighs = torch.stack(neigh_series_list, dim=0)
            # neighs.shape == [batch_size * max_n_neighs, seq_len]

            neighs = torch.log1p(neighs)
            Xn, _ = self._forward_full(neighs)
            # Xn.shape == [batch_size * max_n_neighs, seq_len, hidden_size]

        if self.peek:
            Xn = Xn[:, 1:]
        else:
            Xn = Xn[:, :-1]

        _, S, E = Xn.shape

        # We plus one to give us option to either peek or not
        masks = X.new_ones(B, max_n_neighs, S).bool()
        Xm = X.new_zeros(B, max_n_neighs, S, E)

        for b, key in enumerate(keys):
            # in_degrees maps node_id to a sorted list of dicts
            # a dict key looks like: {'id': 123, 'mask'; [0, 0, 1]}
            if key in key_neighs:
                for i, k in enumerate(key_neighs[key]):
                    Xm[b, i] = Xn[all_neigh_dict[k]]
                    mask = self.mask_dict[key][self.in_degrees[key][k]]
                    mask = mask[start:start+total_len]
                    if self.peek:
                        mask = mask[1:]
                    else:
                        mask = mask[:-1]
                    masks[b, i] = mask

        return Xm, masks

    def _pool(self, X, Xn):
        X_out = torch.cat([X, Xn], dim=-1)
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

    def _aggregate_attn(self, X, Xn, masks):
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

        X_attn, _ = self.attn(X, Xn, Xn, key_padding_mask, False)
        # X_attn.shape == [1, batch_size * seq_len, hidden_size]

        X_out = X_attn.reshape(B, T, E)

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

        nodes = F.elu(nodes)
        # nodes.shape == [BT, hidden_size]

        X_agg = nodes.reshape(B, T, E)
        # X_agg.shape == [batch_size, seq_len, hidden_size]

        return X_agg

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
            s = self.series[self.series_map[key]]
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

        log_raw_series = torch.log1p(raw_series)

        series = torch.log1p(raw_series)

        X_full, preds_full = self._forward_full(series)
        X = X_full[:, :-1]
        preds = preds_full[:, :-1]
        # X.shape == [batch_size, seq_len, hidden_size]

        if self.agg_type != 'none':
            X_agg = self._get_neighbour_embeds(
                X, keys, start, self.total_length, X_full)
            # X_agg.shape == [batch_size, seq_len, out_hidden_size]

            X_agg = self.fc(F.gelu(self.out_proj(X_agg)))
            # X_agg.shape == [batch_size, seq_len, 1]

            preds = preds + X_agg.squeeze(-1)
            # preds.shape == [batch_size, seq_len]

        preds = torch.exp(preds)

        targets = raw_series[:, 1:]
        numerator = torch.abs(targets - preds)
        denominator = torch.abs(targets) + torch.abs(preds)
        loss = numerator / denominator
        loss[torch.isnan(loss)] = 0
        loss = loss.mean()
        out_dict['loss'] = loss

        # During evaluation, we compute one time step at a time
        if split in ['valid', 'test']:
            target_list = []
            for key in keys:
                s = start + self.backcast_length
                e = s + self.forecast_length
                target_list.append(self.series[self.series_map[key], s:e])
            targets = torch.stack(target_list, dim=0)
            # targets.shape == [batch_size, forecast_len]

            preds = targets.new_zeros(*targets.shape)

            series = log_raw_series[:, :-self.forecast_length]
            current_views = series[:, -1]
            for i in range(self.forecast_length):
                X, pred = self._forward_full(series)
                pred = pred[:, -1]
                if self.agg_type != 'none':
                    seq_len = self.total_length - self.forecast_length + i + 1
                    X_agg = self._get_neighbour_embeds(
                        X, keys, start, seq_len)
                    X_agg = self.fc(F.gelu(self.out_proj(X_agg)))
                    pred = pred + X_agg.squeeze(-1)[:, -1]
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


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size, n_layers, dropout, variant):
        super().__init__()
        self.variant = variant
        self.in_proj = GehringLinear(1, hidden_size)
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            self.layers.append(LSTMLayer(hidden_size, dropout, variant))

        self.proj_f = GehringLinear(hidden_size, hidden_size)
        self.out_f = GehringLinear(hidden_size, 1)

    def forward(self, X):
        # X.shape == [batch_size, seq_len]
        # yn.shape == [batch_size, n_neighs, seq_len]

        X = X.unsqueeze(-1)
        # X.shape == [batch_size, seq_len, 1]

        X = self.in_proj(X)
        # X.shape == [batch_size, seq_len, hidden_size]

        forecast = X.new_zeros(*X.shape)
        if self.variant == 'direct':
            hidden = X.new_zeros(X.shape[0], X.shape[1], 2 * X.shape[2])
        else:
            hidden = X.new_zeros(X.shape[0], X.shape[1], 3 * X.shape[2])

        for layer in self.layers:
            h, b, f = layer(X)
            X = X - b
            hidden = hidden + torch.cat([h, b, f], dim=-1)
            forecast = forecast + f
        hidden = hidden / len(self.layers)

        # h = torch.cat(h_list, dim=-1)
        # h.shape == [batch_size, seq_len, n_layers * hidden_size]

        # h = self.out_proj(h)
        # h.shape == [batch_size, seq_len, hidden_size]

        f = self.out_f(F.gelu(self.proj_f(X))).squeeze(-1)

        return hidden, f


class LSTMLayer(nn.Module):
    def __init__(self, hidden_size, dropout, variant):
        super().__init__()
        self.variant = variant

        if variant in ['full', 'half']:
            self.layer = nn.LSTM(hidden_size, hidden_size, 1,
                                 batch_first=True, dropout=dropout)
        elif variant == 'direct':
            self.layer = nn.LSTM(hidden_size, hidden_size * 2, 1,
                                 batch_first=True, dropout=dropout)

        if variant == 'full':
            self.proj_f = GehringLinear(hidden_size, hidden_size)
            self.proj_b = GehringLinear(hidden_size, hidden_size)
            self.out_f = GehringLinear(hidden_size, hidden_size)
            self.out_b = GehringLinear(hidden_size, hidden_size)
        elif variant == 'half':
            self.out_f = GehringLinear(hidden_size, hidden_size)
            self.out_b = GehringLinear(hidden_size, hidden_size)

    def forward(self, X):
        # X.shape == [batch_size, seq_len]
        # yn.shape == [batch_size, n_neighs, seq_len]

        X, _ = self.layer(X)
        # X.shape == [batch_size, seq_len, hidden_size]

        if self.variant == 'full':
            b = self.out_b(F.gelu(self.proj_b(X)))
            f = self.out_f(F.gelu(self.proj_f(X)))
            # b.shape == f.shape == [batch_size, seq_len, hidden_size]
        elif self.variant == 'half':
            b = self.out_b(X)
            f = self.out_f(X)
        elif self.variant == 'direct':
            b, f = torch.chunk(X, 2, dim=-1)

        return X, b, f
