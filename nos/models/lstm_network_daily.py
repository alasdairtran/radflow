import json
import logging
import os
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


@Model.register('time_series_lstm_network_daily')
class TimeSeriesLSTMNetworkDaily(BaseModel):
    def __init__(self,
                 vocab: Vocabulary,
                 decoder: Decoder,
                 data_dir: str,
                 agg_type: str,
                 peak: bool = False,
                 diff_type: str = 'yesterday',
                 max_neighbors: int = 20,
                 initializer: InitializerApplicator = InitializerApplicator()):
        super().__init__(vocab)
        self.decoder = decoder
        self.mse = nn.MSELoss()
        self.hidden_size = decoder.get_output_dim()
        self.peak = peak
        self.diff_type = diff_type
        self.max_neighbors = max_neighbors

        self.n_days = 7
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

        # Load persistent network
        network_path = os.path.join(data_dir, 'persistent_network_2.csv')
        network_df = pd.read_csv(network_path)
        target_ids = set(network_df['target'])
        source_ids = set(network_df['source'])
        node_ids = sorted(target_ids | source_ids)
        n_nodes = len(node_ids)

        # Node features. Let's just make everything 1
        node_features = torch.ones(n_nodes, 1).cuda()

        # Get the edges
        edge_indices = network_df[['source', 'target']].to_numpy()
        edge_indices = torch.tensor(edge_indices.transpose()).cuda()

        source_path = os.path.join(data_dir, 'snapshots.json')
        with open(source_path) as f:
            self.snapshots = json.load(f, object_pairs_hook=keystoint)

        self.network = Data(x=node_features,
                            edge_index=edge_indices)

        series_path = os.path.join(data_dir, 'vevo_full_series.json')
        with open(series_path) as f:
            self.series = json.load(f, object_pairs_hook=keystoint)

        # Shortcut to create new tensors in the same device as the module
        self.register_buffer('_long', torch.LongTensor(1))

        self.cached_series = {}

    def _initialize_series(self):
        if isinstance(next(iter(self.series.values())), torch.Tensor):
            return

        p = next(self.parameters())

        for k, v in self.series.items():
            self.series[k] = p.new_tensor(v)

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

    def _get_neighbour_embeds(self, X, keys, day=None):
        if self.agg_type == 'none':
            return X

        neighbor_lens = []
        source_list = []

        for key in keys:
            if key in self.snapshots[day]:
                # Keep only the top 20 neighbours
                sources = self.snapshots[day][key][:self.max_neighbors]
            else:
                sources = []
            neighbor_lens.append(len(sources))
            for s in sources:
                s_series = self.series[s]
                s_series = s_series[:day+3]
                assert len(s_series) == day + 3
                source_list.append(s_series)

        sources = torch.stack(source_list, dim=0)
        # sources.shape == [batch_size * n_neighbors, seq_len]

        X_neighbors = self._forward_full(sources)
        X_neighbors = X_neighbors[:, -1:]
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
            if X_neighbors_i.shape[0] == 0:
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

        X_full_list = []
        # Ignore the first week to save memory
        targets = targets[:, 7:]
        for day in range(7, X.shape[1]):
            X_i = X[:, day: day+1]
            X_full_list.append(self._get_neighbour_embeds(X_i, keys, day))
            # X_full.shape == [batch_size, seq_len, out_hidden_size]

        X_full = torch.cat(X_full_list, dim=1)

        X_full = self.fc(X_full)
        # X_full.shape == [batch_size, seq_len, 1]

        preds = X_full.squeeze(-1)
        # preds.shape == [batch_size, seq_len]

        if splits[0] in ['test']:
            preds = preds[-self.n_days:]
            targets = targets[-self.n_days:]

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
            smape, _ = get_smape(targets, preds)

            out_dict['smape'] = smape

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
                    X, batch_keys, day=day+56)
                # X_full.shape == [batch_size, 1, hidden_size]

                # This is our prediction, the percentage change from
                # the previous time step.
                pct = self.fc(X_full).squeeze(2).squeeze(1)
                # pct.shape == [batch_size]

                # Calculate the predicted view count
                for i, key in enumerate(batch_keys):
                    pred = self.cached_series[key][-1:] * pct[i]
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
