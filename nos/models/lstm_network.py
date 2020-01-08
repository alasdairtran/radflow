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
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from tqdm import tqdm

from nos.modules import Decoder
from nos.modules.linear import GehringLinear
from nos.utils import keystoint

from .base import BaseModel
from .metrics import get_smape

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register('time_series_lstm_network')
class TimeSeriesLSTMNetwork(BaseModel):
    def __init__(self,
                 vocab: Vocabulary,
                 decoder: Decoder,
                 data_dir: str,
                 evaluate_mode: bool,
                 agg_type: str,
                 initializer: InitializerApplicator = InitializerApplicator()):
        super().__init__(vocab)
        self.decoder = decoder
        self.mse = nn.MSELoss()
        self.hidden_size = decoder.get_output_dim()

        self.n_days = 7
        self.evaluate_mode = evaluate_mode
        initializer(self)

        assert agg_type in ['attention', 'mean', 'gcn', 'none']
        self.agg_type = agg_type
        if agg_type == 'attention':
            self.attn = nn.MultiheadAttention(
                self.hidden_size, 4, dropout=0.1, bias=True,
                add_bias_kv=True, add_zero_attn=True, kdim=None, vdim=None)
        elif agg_type == 'gcn':
            self.conv1 = GCNConv(self.hidden_size, self.hidden_size)

        if agg_type in ['gcn', 'none']:
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

        source_path = os.path.join(data_dir, 'adjacency_list.json')
        with open(source_path) as f:
            self.sources = json.load(f, object_pairs_hook=keystoint)

        self.network = Data(x=node_features,
                            edge_index=edge_indices)

        series_path = os.path.join(data_dir, 'vevo_full_series.json')
        with open(series_path) as f:
            self.series = json.load(f, object_pairs_hook=keystoint)

        target_series_path = os.path.join(data_dir, 'vevo_series.json')
        with open(target_series_path) as f:
            self.target_series = json.load(f, object_pairs_hook=keystoint)

        # Shortcut to create new tensors in the same device as the module
        self.register_buffer('_long', torch.LongTensor(1))

    def _initialize_series(self):
        if isinstance(next(iter(self.series.values())), torch.Tensor):
            return

        p = next(self.parameters())

        for k, v in self.series.items():
            self.series[k] = p.new_tensor(v)

    def _forward(self, key, series):
        training_series = series.clone().detach()
        training_series[training_series == 0] = 1

        # Take the difference
        diff = training_series[1:] / training_series[:-1]
        targets = diff[1:]
        inputs = diff[:-1]

        X = inputs.unsqueeze(-1).unsqueeze(0)
        # X.shape == [1, seq_len, 1]

        incremental_state: Dict[str, Any] = {}
        X, _ = self.decoder(X, incremental_state=incremental_state)

        return X, targets

    def _get_source_embeds(self, key, X_key):
        if self.agg_type == 'none':
            return X_key

        sources = self.sources[key]
        X_source_list = []
        for s in sources:
            s_series = self.series[s]
            s_series = s_series[:-self.n_days]
            X_source_list.append(self._forward(s, s_series)[0])
        X_source = torch.cat(X_source_list, dim=0)
        # X_source.shape == [n_neighbors, seq_len, hidden_size]

        if self.agg_type == 'attention':
            X_out, _ = self.attn(X_key, X_source, X_source)
            # X_out.shape == [1, seq_len, hidden_size]

            # Combine own embedding with neighbor embedding
            X_full = torch.cat([X_key, X_out], dim=-1)
            # X_full.shape == [1, seq_len, 2 * hidden_size]

        elif self.agg_type == 'mean':
            X_out = X_source.mean(dim=0).unsqueeze(0)
            # X_out.shape == [1, seq_len, hidden_size]

            # Combine own embedding with neighbor embedding
            X_full = torch.cat([X_key, X_out], dim=-1)
            # X_full.shape == [1, seq_len, 2 * hidden_size]

        elif self.agg_type == 'gcn':
            # The central node is the first node. The rest are neighbors
            feats = torch.cat([X_key] + X_source_list, dim=0)
            # feats.shape == [n_nodes, seq_len, hidden_size]

            N, S, H = feats.shape
            feats = feats.transpose(0, 1).reshape(S * N, H)
            # feats.shape == [seq_len * n_nodes, hidden_size]

            # We add self-loops as well to make easier
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

    def _get_source_embeds_eval(self, key, hidden_dict):
        if self.agg_type == 'none':
            return hidden_dict[key]

        # if a node has neighbors, we concatenate all of the
        # neighboring hidden states at the same time step and
        # do a multi-head attention over them.
        if key in self.sources:
            neighbors = self.sources[key]
            n_hidden_list = [hidden_dict[n] for n in neighbors]
            n_hidden = torch.cat(n_hidden_list, dim=0)
            # n_hidden.shape == [n_neighbors, 1, hidden_size]

            if self.agg_type == 'attention':
                X_out, _ = self.attn(hidden_dict[key],
                                     n_hidden, n_hidden)
                # X_out.shape == [1, 1, hidden_size]

                X_full = torch.cat([hidden_dict[key], X_out], dim=-1)
                # X_full.shape == [1, 1, 2 * hidden_size]

            elif self.agg_type == 'mean':
                X_out = n_hidden.mean(dim=0).unsqueeze(0)
                # X_out.shape == [1, 1, hidden_size]

                X_full = torch.cat([hidden_dict[key], X_out], dim=-1)
                # X_full.shape == [1, 1, 2 * hidden_size]

            elif self.agg_type == 'gcn':
                # The central node is the first node. The rest are neighbors
                feats = torch.cat(
                    [hidden_dict[key]] + n_hidden_list, dim=0)
                # feats.shape == [n_nodes, 1, hidden_size]

                N, S, H = feats.shape
                feats = feats.transpose(0, 1).reshape(S * N, H)
                # feats.shape == [1 * n_nodes, hidden_size]

                # We add self-loops as well to make easier
                source_idx = torch.arange(0, len(feats))
                source_idx = source_idx.to(self._long.device)
                # source_idx.shape == [1 * n_neighbors]

                target_idx = torch.arange(0, N * S, N)
                target_idx = target_idx.to(self._long.device)
                target_idx = target_idx.repeat(N, 1).reshape(-1)

                edge_list = [source_idx, target_idx]
                edge_index = torch.stack(edge_list, dim=0)

                X_full = self.conv1(feats, edge_index)
                # X_full.shape == [1 * n_nodes, hidden_size]

                X_full = X_full.reshape(S, N, H).transpose(0, 1)
                # X_full.shape == [n_nodes, 1, hidden_size]

                X_full = X_full[:1]
                # X_full.shape == [1, 1, hidden_size]

        # If there are no neighbors, we simply append a zero vector
        elif self.agg_type != 'gcn':
            X_out = hidden_dict[key].new_zeros(1, 1, self.hidden_size)
            X_full = torch.cat([hidden_dict[key], X_out], dim=-1)
            # X_full.shape == [1, 1, 2 * hidden_size]
        else:
            X_full = hidden_dict[key]

        return X_full

    def forward(self, keys) -> Dict[str, Any]:
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

        X_full_list = []
        target_list = []

        for key in keys:
            series = self.series[key]
            series = series[:-self.n_days]
            X_key, X_target = self._forward(key, series)
            # X_key.shape == [1, seq_len, hidden_size]
            # X_target.shape == [seq_len]

            target_list.append(X_target)
            X_full_list.append(self._get_source_embeds(key, X_key))

        X_full = torch.cat(X_full_list, dim=0)
        # X_full.shape == [batch_size, seq_len, 2 * hidden_size]

        X_full = self.fc(X_full)
        # X_full.shape == [batch_size, seq_len, 1]

        preds = X_full.squeeze(-1)
        # preds.shape == [batch_size, seq_len]

        targets = torch.stack(target_list, dim=0)
        # targets.shape == [batch_size, seq_len]

        if not self.training:
            preds = preds[-self.n_days:]
            targets = targets[-self.n_days:]

        loss = self.mse(preds, targets)
        out_dict['loss'] = loss

        # During evaluation, we compute all the time series at once
        if not self.training and self.evaluate_mode:
            series_dict = {}
            hidden_dict = {}

            # Remove the final 7 days from the input series since they are
            # what we want to predict. As we evaluate, series_dict will
            # store the entire series including the latest predictions.
            for key in self.series.keys():
                series = self.series[key]
                series = series[:-self.n_days]
                series_dict[key] = series

            for day in range(self.n_days):
                logger.info(f'Validating day {day}')

                # First pass: Get the latest hidden state of all nodes before
                # taking the network effect into account.
                for key in tqdm(self.series.keys()):
                    hidden, _ = self._forward(key, series_dict[key])
                    # hidden.shape == [1, seq_len, hidden_size]

                    # We store the hidden state in the latest timestep (the
                    # step that we are predicting( in hidden_dict
                    hidden_dict[key] = hidden[:, -1:]
                    # hidden_dict[key].shape == [1, 1, hidden_size]

                # Second pass: aggregate effect from neighboring nodes
                for key in tqdm(self.series.keys()):
                    X_full = self._get_source_embeds_eval(key, hidden_dict)

                    # This is our prediction, the percentage change from
                    # the previous time step.
                    pct = self.fc(X_full)[0, 0]
                    # pct.shape == [1]

                    # Calculate the predicted view count
                    pred = series_dict[key][-1:] * pct

                    series_dict[key] = torch.cat(
                        [series_dict[key], pred], dim=0)

            smape_list = []
            for key in self.target_series.keys():
                preds = series_dict[key][-self.n_days:].unsqueeze(0)
                targets = self.series[key][-self.n_days:].unsqueeze(0)
                smape, _ = get_smape(targets, preds)
                smape_list.append(smape[0])

            out_dict['smape'] = smape_list
            self.history['_n_samples'] = len(smape_list)

        return out_dict

    @classmethod
    def get_params(cls, vocab: Vocabulary, params: Params) -> Dict[str, Any]:

        params_dict: Dict[str, Any] = {}

        params_dict['decoder'] = Decoder.from_params(
            vocab=vocab, params=params.pop('decoder'))
        params_dict['initializer'] = InitializerApplicator.from_params(
            params.pop('initializer', None))

        return params_dict
