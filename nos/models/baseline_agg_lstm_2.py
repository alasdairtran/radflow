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


@Model.register('baseline_agg_lstm_2')
class BaselineAggLSTM2(BaseModel):
    def __init__(self,
                 vocab: Vocabulary,
                 decoder: Decoder,
                 data_dir: str,
                 agg_type: str,
                 forecast_length: int = 7,
                 backcast_length: int = 35,
                 peek: bool = False,
                 seed_word: str = 'vevo',
                 n_days: int = 7,
                 max_neighbours: int = 8,
                 initializer: InitializerApplicator = InitializerApplicator()):
        super().__init__(vocab)
        self.decoder = decoder
        self.mse = nn.MSELoss()
        self.hidden_size = decoder.get_output_dim()
        self.peek = peek
        self.max_neighbours = max_neighbours
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length

        self.n_days = n_days
        self.rs = np.random.RandomState(1234)
        initializer(self)

        assert agg_type in ['mean', 'none']
        self.agg_type = agg_type
        if agg_type in ['none']:
            self.fc = GehringLinear(self.hidden_size, 1)
        elif agg_type in ['mean']:
            self.fc = GehringLinear(self.hidden_size * 2, 1)

        with open(f'{data_dir}/{seed_word}.pkl', 'rb') as f:
            self.sources, _, self.series = pickle.load(f)

        # Shortcut to create new tensors in the same device as the module
        self.register_buffer('_long', torch.LongTensor(1))

        self.cached_series = {}
        self.non_missing = {}

    def _initialize_series(self):
        if isinstance(next(iter(self.series.values())), torch.Tensor):
            return

        p = next(self.parameters())
        for k, v in self.series.items():
            self.series[k] = np.asarray(v).astype(float)

        # Sort by view counts
        for node, neighs in self.sources.items():
            counts = []
            for neigh in neighs:
                count = self.series[neigh][:self.backcast_length].sum()
                counts.append(count)
            keep_idx = np.argsort(counts)[::-1][:self.max_neighbours]
            self.sources[node] = np.array(neighs)[keep_idx]

        for k, v in self.series.items():
            v_array = np.asarray(v)
            self.series[k] = p.new_tensor(v_array)

    def _forward(self, series):
        # series.shape == [batch_size, seq_len]

        training_series = series.clone().detach()
        training_series[training_series == 0] = 1

        # Take the difference
        diff = training_series[:, 1:] / training_series[:, :-1]
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
        inputs = training_series[:, 1:] / training_series[:, :-1]
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

        if use_gt_day is not None and self.peek:
            use_gt_day += 1
        elif self.peek and n_skips is not None and n_skips > 0:
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

        if not forward_full and not self.peek:
            X_neighbors, _ = self._forward(sources)
        elif not forward_full and self.peek and n_skips == 0:
            X_neighbors = self._forward_full(sources)
            X_neighbors = X_neighbors[:, 1:]
        elif not forward_full and self.peek:
            X_neighbors, _ = self._forward(sources)
            X_neighbors = X_neighbors[:, 1:]
        elif forward_full and self.peek:
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

        if X_neighbors_i.shape == 0:
            X_out = X_i.new_zeros(*X_i.shape)
        else:
            X_out = X_neighbors_i.mean(dim=0).unsqueeze(0)
            # X_out.shape == [1, seq_len, hidden_size]

        # Combine own embedding with neighbor embedding
        X_full = torch.cat([X_i, X_out], dim=-1)
        # X_full.shape == [1, seq_len, 2 * hidden_size]

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

        if splits[0] in ['train']:
            n_skips = self.n_days * 2
        elif splits[0] in ['valid']:
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
