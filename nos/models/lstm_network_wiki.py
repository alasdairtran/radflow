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


@Model.register('time_series_lstm_network_wiki')
class TimeSeriesLSTMNetworkWiki(BaseModel):
    def __init__(self,
                 vocab: Vocabulary,
                 decoder: Decoder,
                 data_dir: str,
                 agg_type: str,
                 peak: bool = False,
                 diff_type: str = 'yesterday',
                 seed_word: str = 'Programming languages',
                 n_days: int = 30,
                 missing_p: float = 0.0,
                 optimize_non_missing: bool = False,
                 max_neighbours: int = 8,
                 remove_trends: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator()):
        super().__init__(vocab)
        self.decoder = decoder
        self.mse = nn.MSELoss()
        self.hidden_size = decoder.get_output_dim()
        self.peak = peak
        self.diff_type = diff_type
        self.max_neighbours = max_neighbours
        self.remove_trends = remove_trends

        self.n_days = n_days
        self.missing_p = missing_p
        self.optimize_non_missing = optimize_non_missing
        self.rs = np.random.RandomState(1234)
        initializer(self)

        assert agg_type in ['attention', 'mean', 'sage', 'none']
        assert diff_type in ['yesterday', 'last_week']
        self.agg_type = agg_type
        if agg_type == 'attention':
            self.attn = nn.MultiheadAttention(
                self.hidden_size, 4, dropout=0.1, bias=True,
                add_bias_kv=True, add_zero_attn=True, kdim=None, vdim=None)
        elif agg_type == 'sage':
            self.conv1 = SAGEConv(self.hidden_size, self.hidden_size)

        if agg_type in ['sage', 'none']:
            self.fc = GehringLinear(self.hidden_size, 1)
        else:
            self.fc = GehringLinear(self.hidden_size * 2, 1)

        with open(f'data/wiki/subgraphs/{seed_word}.cleaned.pkl', 'rb') as f:
            self.sources, _ = pickle.load(f)

        with open(f'data/wiki/subgraphs/{seed_word}.series.pkl', 'rb') as f:
            self.series = pickle.load(f)

        # Shortcut to create new tensors in the same device as the module
        self.register_buffer('_long', torch.LongTensor(1))

        self.cached_series = {}
        self.non_missing = {}

    def _initialize_series(self):
        if isinstance(next(iter(self.series.values())), torch.Tensor):
            return

        p = next(self.parameters())

        if self.remove_trends:
            for k, v in self.series.items():
                n_train = 1 + 365 - self.n_days
                r = 0 if n_train % 7 == 0 else (7 - (n_train % 7))
                v_full = np.zeros(n_train + r)
                v_full[1:366-self.n_days] = v[:-self.n_days]
                v_full = v_full.reshape((n_train + r) // 7, 7)
                avg_all = v_full.mean()
                avg_week = v_full.mean(axis=0)
                diff = avg_week - avg_all
                diff = np.tile(diff, 53)
                diff = diff[1:366]
                self.series[k] = v - diff

        for k, v in self.series.items():
            v_array = np.asarray(v)
            if self.missing_p > 0:
                size = v_array.size - self.n_days
                start = 2 if self.optimize_non_missing else 1
                indices = self.rs.choice(np.arange(start, size), replace=False,
                                         size=int(size * self.missing_p))
                prev_indices = indices - 1
                v_array[indices] = v_array[prev_indices]
                non_missing_idx = np.ones(size, dtype=bool)
                non_missing_idx[indices] = False
                non_missing_idx = non_missing_idx[2:]
                self.non_missing[k] = non_missing_idx
            self.series[k] = p.new_tensor(v_array)

    def _forward(self, series):
        # series.shape == [batch_size, seq_len]

        training_series = series.clone().detach()
        training_series[training_series == 0] = 1

        # Take the difference
        if self.diff_type == 'yesterday':
            diff = training_series[:, 1:] / training_series[:, :-1]
        elif self.diff_type == 'last_week':
            diff = training_series[:, 7:] / training_series[:, :-7]
        targets = diff[:, 1:]
        inputs = diff[:, :-1]

        X = inputs.unsqueeze(-1)
        # X.shape == [batch_size, seq_len - 1, 1]

        incremental_state: Dict[str, Any] = {}
        X, _ = self.decoder(X, incremental_state=incremental_state)

        return X, targets

    def _forward_full(self, series):
        # series.shape == [batch_size, seq_len]

        training_series = series.clone().detach()
        training_series[training_series == 0] = 1

        # Take the difference
        if self.diff_type == 'yesterday':
            inputs = training_series[:, 1:] / training_series[:, :-1]
        elif self.diff_type == 'last_week':
            inputs = training_series[:, 7:] / training_series[:, :-7]
        X = inputs.unsqueeze(-1)
        # X.shape == [batch_size, seq_len, 1]

        incremental_state: Dict[str, Any] = {}
        X, _ = self.decoder(X, incremental_state=incremental_state)

        return X

    def _get_neighbour_embeds(self, X, keys, n_skips=None, forward_full=False,
                              use_cache=False, use_gt_day=None):
        if self.agg_type == 'none':
            return X

        neighbor_lens = []
        source_list = []

        if use_gt_day is not None and self.peak:
            use_gt_day += 1
        elif self.peak and n_skips is not None and n_skips > 0:
            n_skips -= 1

        for key in keys:
            if key in self.sources:
                sources = self.sources[key][:self.max_neighbours]
            else:
                sources = []
            neighbor_lens.append(len(sources))
            for s in sources:
                if use_gt_day is not None:
                    s_series = self.series[s]
                    cutoff = - \
                        (self.n_days-use_gt_day) if use_gt_day < 7 else None
                    s_series = s_series[:cutoff]
                elif use_cache:
                    s_series = self.cached_series[s]
                else:
                    s_series = self.series[s]
                    s_series = s_series[:-n_skips if n_skips > 0 else None]
                source_list.append(s_series)

        sources = torch.stack(source_list, dim=0)
        # sources.shape == [batch_size * n_neighbors, seq_len]

        if not forward_full and not self.peak:
            X_neighbors, _ = self._forward(sources)
        elif not forward_full and self.peak and n_skips == 0:
            X_neighbors = self._forward_full(sources)
            X_neighbors = X_neighbors[:, 1:]
        elif not forward_full and self.peak:
            X_neighbors, _ = self._forward(sources)
            X_neighbors = X_neighbors[:, 1:]
        elif forward_full and self.peak:
            X_neighbors = self._forward_full(sources)
            X_neighbors = X_neighbors[:, 1:]
        else:
            X_neighbors = self._forward_full(sources)
        # X_neighbors.shape == [batch_size * n_neighbors, seq_len, hidden_size]

        if X.shape[1] == 1:
            X_neighbors = X_neighbors[:, -1:]

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

        if self.agg_type == 'attention':
            if X_neighbors_i.shape == 0:
                X_out = X_i.new_zeros(*X_i.shape)
            else:
                X_out, _ = self.attn(X_i, X_neighbors_i, X_neighbors_i)
                # X_out.shape == [1, seq_len, hidden_size]

            # Combine own embedding with neighbor embedding
            X_full = torch.cat([X_i, X_out], dim=-1)
            # X_full.shape == [1, seq_len, 2 * hidden_size]

        elif self.agg_type == 'mean':
            if X_neighbors_i.shape == 0:
                X_out = X_i.new_zeros(*X_i.shape)
            else:
                X_out = X_neighbors_i.mean(dim=0).unsqueeze(0)
                # X_out.shape == [1, seq_len, hidden_size]

            # Combine own embedding with neighbor embedding
            X_full = torch.cat([X_i, X_out], dim=-1)
            # X_full.shape == [1, seq_len, 2 * hidden_size]

        elif self.agg_type == 'sage':
            # The central node is the first node. The rest are neighbors
            feats = torch.cat([X_i, X_neighbors_i], dim=0)
            # feats.shape == [n_nodes, seq_len, hidden_size]

            N, S, H = feats.shape
            feats = feats.transpose(0, 1).reshape(S * N, H)
            # feats.shape == [seq_len * n_nodes, hidden_size]

            # We add self-loops as well to make life easier
            source_idx = torch.arange(0, len(feats))
            source_idx = source_idx.to(self._long.device)
            # source_idx.shape == [seq_len * n_neighbors]

            target_idx = torch.arange(0, N * S, N)
            target_idx = target_idx.to(self._long.device)
            target_idx = target_idx.repeat(N, 1).reshape(-1)

            edge_list = [source_idx, target_idx]
            edge_index = torch.stack(edge_list, dim=0)

            X_full = self.conv1(feats, edge_index)
            # X_full.shape == [seq_len * n_nodes, hidden_size]

            X_full = X_full.reshape(S, N, H).transpose(0, 1)
            # X_full.shape == [n_nodes, seq_len, hidden_size]

            X_full = X_full[:1]
            # X_full.shape == [1, seq_len, hidden_size]

        return X_full

    def forward(self, keys, splits) -> Dict[str, Any]:
        # Enable anomaly detection to find the operation that failed to compute
        # its gradient.
        # torch.autograd.set_detect_anomaly(True)

        self._initialize_series()

        B = len(keys)
        p = next(self.parameters())
        # keys.shape == [batch_size]

        self.history['_n_batches'] += 1
        self.history['_n_samples'] += B

        out_dict = {
            'loss': None,
            'sample_size': p.new_tensor(B),
        }

        if splits[0] in ['train', 'valid']:
            n_skips = self.n_days
        elif splits[0] == 'test':
            n_skips = 0

        series_list = []
        for key in keys:
            s = self.series[key]
            s = s[:-n_skips if n_skips > 0 else None]
            series_list.append(s)

        series = torch.stack(series_list, dim=0)
        # series.shape == [batch_size, seq_len]

        X, targets = self._forward(series)
        # X.shape == [batch_size, seq_len, hidden_size]
        # targets.shape == [batch_size, seq_len]

        X_full = self._get_neighbour_embeds(X, keys, n_skips)
        # X_full.shape == [batch_size, seq_len, out_hidden_size]

        X_full = self.fc(X_full)
        # X_full.shape == [batch_size, seq_len, 1]

        preds = X_full.squeeze(-1)
        # preds.shape == [batch_size, seq_len]

        if splits[0] in ['test']:
            preds = preds[-self.n_days:]
            targets = targets[-self.n_days:]

        if self.training and self.optimize_non_missing:
            new_pred_list = []
            new_target_list = []
            for i, key in enumerate(keys):
                new_pred_list.append(preds[i][self.non_missing[key]])
                new_target_list.append(targets[i][self.non_missing[key]])
            preds = torch.stack(new_pred_list, dim=0)
            targets = torch.stack(new_target_list, dim=0)

        loss = self.mse(preds, targets)
        out_dict['loss'] = loss

        # During evaluation, we compute one time step at a time
        if splits[0] in ['test']:
            self._populate_cache(splits)

            pred_list, target_list = [], []
            for key in keys:
                pred_list.append(self.cached_series[key][-self.n_days:])
                target_list.append(self.series[key][-self.n_days:])

            preds = torch.stack(pred_list, dim=0)
            targets = torch.stack(target_list, dim=0)
            smape, daily_errors = get_smape(targets, preds)

            out_dict['smapes'] = smape
            out_dict['daily_errors'] = daily_errors
            out_dict['keys'] = keys

        return out_dict

    def _populate_cache(self, splits):
        if self.cached_series:
            return

        hidden_dict = {}
        n_skips = self.n_days

        # Remove the final n_skips days from the input series since they are
        # what we want to predict. As we evaluate, series_dict will
        # store the entire series including the latest predictions.
        for key in self.series.keys():
            s = self.series[key]
            s = s[:-n_skips]
            self.cached_series[key] = s

        batch_size = 512
        keys = sorted(self.series.keys())

        for day in range(self.n_days):
            logger.info(f'Populating cache day {day}')

            # First pass: Get the latest hidden state of all nodes with an edge
            # (incoming or outgoing) before taking the network effect into
            # account. We do this in batch of 512
            for i in range(0, len(keys), batch_size):
                batch_keys = keys[i:i+batch_size]

                series_list = []
                for key in batch_keys:
                    s = self.cached_series[key]
                    series_list.append(s)
                series = torch.stack(series_list, dim=0)

                hidden = self._forward_full(series)
                # hidden.shape == [batch_size, seq_len, hidden_size]

                # We store the hidden state in the latest timestep (the
                # step that we are predicting in hidden_dict
                for key, h in zip(batch_keys, hidden):
                    hidden_dict[key] = h[-1:]
                    # hidden_dict[key].shape == [1, hidden_size]

            # Second pass: aggregate effect from neighboring nodes
            new_cached_series = {}
            for i in range(0, len(keys), batch_size):
                batch_keys = keys[i:i+batch_size]

                hidden_list = []
                for key in batch_keys:
                    h = hidden_dict[key]
                    hidden_list.append(h)
                X = torch.stack(hidden_list, dim=0)
                # X.shape == [batch_size, 1, hidden_size]

                X_full = self._get_neighbour_embeds(
                    X, batch_keys, forward_full=True, use_cache=True, use_gt_day=day)
                # X_full.shape == [batch_size, 1, hidden_size]

                # This is our prediction, the percentage change from
                # the previous time step.
                pct = self.fc(X_full).squeeze(2).squeeze(1)
                # pct.shape == [batch_size]

                # Calculate the predicted view count
                for i, key in enumerate(batch_keys):
                    if self.diff_type == 'yesterday':
                        pred = self.cached_series[key][-1:] * pct[i]
                    elif self.diff_type == 'last_week':
                        pred = self.cached_series[key][-7:] * pct[i]

                    new_cached_series[key] = torch.cat(
                        [self.cached_series[key], pred], dim=0)

            self.cached_series = new_cached_series

    @classmethod
    def get_params(cls, vocab: Vocabulary, params: Params) -> Dict[str, Any]:

        params_dict: Dict[str, Any] = {}

        params_dict['decoder'] = Decoder.from_params(
            vocab=vocab, params=params.pop('decoder'))
        params_dict['initializer'] = InitializerApplicator.from_params(
            params.pop('initializer', None))

        return params_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, Any]:
        if reset:
            self.cached_series = {}

        return super().get_metrics(reset)