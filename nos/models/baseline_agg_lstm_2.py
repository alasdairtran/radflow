import logging
import math
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
                 opt_smape: bool = False,
                 max_neighbours: int = 4,
                 detach: bool = False,
                 batch_as_subgraph: bool = False,
                 attn: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator()):
        super().__init__(vocab)
        self.attn = attn
        self.mse = nn.MSELoss()
        self.hidden_size = hidden_size
        self.peek = peek
        self.max_neighbours = max_neighbours
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.total_length = forecast_length + backcast_length
        self.test_lengths = test_lengths
        self.opt_smape = opt_smape
        self.detach = detach
        self.batch_as_subgraph = batch_as_subgraph
        self.max_start = None
        self.rs = np.random.RandomState(1234)

        if not attn:
            self.decoder = nn.LSTM(1, hidden_size, num_layers,
                                   bias=True, batch_first=True, dropout=dropout)
        else:
            self.decoder = TransformerDecoder(
                hidden_size, num_layers, dropout, self.total_length)

        assert agg_type in ['mean', 'none']
        self.agg_type = agg_type
        self.fc = GehringLinear(self.hidden_size, 1)
        if agg_type in ['mean']:
            self.out_proj = GehringLinear(
                self.hidden_size * 2, self.hidden_size)

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
            self.series[k] = np.asarray(v).astype(float)

        if self.agg_type != 'none':
            for k, v in self.neighs.items():
                for t in v.keys():
                    self.neighs[k][t] = self.neighs[k][t][:self.max_neighbours]

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

        logger.info('Processing series')
        series_matrix = np.zeros((len(self.series),
                                  len(self.series[k])))
        self.series_map = {}
        for i, (k, v) in enumerate(tqdm(self.series.items())):
            series_matrix[i] = np.asarray(v)
            self.series_map[k] = i
        self.series = p.new_tensor(series_matrix)

        self.max_start = len(
            self.series[i]) - self.forecast_length * 2 - self.total_length

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

        if self.detach:
            with torch.no_grad():
                Xm, masks = self._construct_neighs(X, keys, start, total_len)
                Xm = self._aggregate(Xm, masks)
        else:
            Xm, masks = self._construct_neighs(X, keys, start, total_len)
            Xm = self._aggregate(Xm, masks)

        X_out = self._pool(X, Xm)
        return X_out

    def _construct_neighs(self, X, keys, start, total_len):
        B, T, E = X.shape

        # First iteration: grab the top neighbours from each sample
        key_neighs = {}
        max_n_neighs = 1
        all_neigh_keys = set()
        for key in keys:
            if key in self.neighs:
                kn = set()
                for day in range(start, start+total_len):
                    kn |= set(self.neighs[key][day])
                key_neighs[key] = kn
                all_neigh_keys |= kn
                max_n_neighs = max(max_n_neighs, len(kn))

        all_neigh_keys = list(all_neigh_keys)
        all_neigh_dict = {k: i for i, k in enumerate(all_neigh_keys)}
        neigh_series_list = []
        for key in all_neigh_keys:
            neigh_series_list.append(
                self.series[self.series_map[key], start:start+total_len])
        neighs = torch.stack(neigh_series_list, dim=0)
        # neighs.shape == [batch_size * max_n_neighs, seq_len]

        neighs = torch.log1p(neighs)
        Xn = self._forward_full(neighs)
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
        # Xn.shape == [batch_size, seq_len, 2 * hidden_size]

        X_out = F.relu(self.out_proj(X_out))
        # Xn.shape == [batch_size, seq_len, hidden_size]

        return X_out

    def _aggregate(self, Xn, masks):
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

        series = torch.log1p(raw_series)

        X = self._forward_full(series)
        X = X[:, :-1]
        targets = series[:, 1:]
        # X.shape == [batch_size, seq_len, hidden_size]
        # targets.shape == [batch_size, seq_len]

        X_agg = self._get_neighbour_embeds(X, keys, start, self.total_length)
        # X_agg.shape == [batch_size, seq_len, out_hidden_size]

        X_agg = self.fc(X_agg)
        # X_agg.shape == [batch_size, seq_len, 1]

        preds = X_agg.squeeze(-1)
        # preds.shape == [batch_size, seq_len]

        if split in ['valid', 'test']:
            preds = preds[:, -self.forecast_length:]
            targets = targets[:, -self.forecast_length:]

        if self.opt_smape:
            preds = torch.exp(preds)
            if split in ['valid', 'test']:
                targets = raw_series[:, -self.forecast_length:]
            else:
                targets = raw_series[:, 1:]
            numerator = torch.abs(targets - preds)
            denominator = torch.abs(targets) + torch.abs(preds)
            loss = numerator / denominator
            loss[torch.isnan(loss)] = 0
            loss = loss.mean()
        else:
            loss = self.mse(preds, targets)
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

            series = series[:, :-self.forecast_length]
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
            self.history['_n_samples'] += len(keys)
            self.history['_n_steps'] += smapes.shape[0] * smapes.shape[1]

            for k in self.test_lengths:
                self.step_history[f'smape_{k}'] += np.sum(smapes[:, :k])

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
