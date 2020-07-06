import copy
import logging
import math
import os
import pickle
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional

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
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from tqdm import tqdm

from nos.modules import Decoder
from nos.modules.linear import GehringLinear
from nos.utils import keystoint

from .base import BaseModel
from .metrics import get_smape

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register('baseline_agg_lstm_3')
class BaselineAggLSTM3(BaseModel):
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
                 log: bool = False,
                 opt_smape: bool = False,
                 max_neighbours: int = 8,
                 attn: bool = False,
                 detach: bool = True,
                 n_heads: int = 4,
                 arch: str = 'lstm',
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
        self.log = log
        self.opt_smape = opt_smape
        self.max_start = None
        self.detach = detach
        self.rs = np.random.RandomState(1234)

        if not attn:
            self.decoder = nn.LSTM(1, hidden_size, num_layers,
                                   bias=True, batch_first=True, dropout=dropout)
        elif arch == 'transformer':
            self.decoder = SeriesTransformer(
                hidden_size, num_layers, num_layers, n_heads, dropout,
                forecast_length, backcast_length)
        elif arch == 'lstm':
            self.decoder = SeriesBiLSTM(
                hidden_size, num_layers, num_layers, n_heads, dropout,
                forecast_length, backcast_length)

        assert agg_type in ['mean', 'none']
        self.agg_type = agg_type
        if agg_type in ['none']:
            self.fc = GehringLinear(self.hidden_size, 1)
        elif agg_type in ['mean']:
            self.fc = GehringLinear(self.hidden_size * 2, 1)

        with open(f'{data_dir}/{seed_word}.pkl', 'rb') as f:
            self.in_degrees, _, self.series = pickle.load(f)

        # Shortcut to create new tensors in the same device as the module
        self.register_buffer('_long', torch.LongTensor(1))

        initializer(self)

    def _initialize_series(self):
        if isinstance(next(iter(self.series.values())), torch.Tensor):
            return

        p = next(self.parameters())
        for k, v in self.series.items():
            self.series[k] = np.asarray(v).astype(float)

        # Sort by view counts
        logger.info('Processing edges')
        for node, neighs in tqdm(self.in_degrees.items()):
            counts = []
            for n in neighs:
                n['mask'] = p.new_tensor(np.asarray(n['mask']))
                count = self.series[n['id']][:self.backcast_length].sum()
                counts.append(count)
            keep_idx = np.argsort(counts)[::-1][:self.max_neighbours]
            self.in_degrees[node] = np.array(neighs)[keep_idx]

        for k, v in self.series.items():
            v_array = np.asarray(v)
            self.series[k] = p.new_tensor(v_array)

        self.max_start = len(
            self.series[k]) - self.forecast_length * 2 - self.total_length

    def _forward(self, series):
        # series.shape == [batch_size, seq_len]

        # Take the difference
        if not self.log:
            training_series = series.clone().detach()
            training_series[training_series == 0] = 1
            diff = training_series[:, 1:] / training_series[:, :-1]
            targets = diff[:, 1:]
            inputs = diff[:, :-1]
            if self.attn:
                targets = series[:, -self.forecast_length+1:]
        else:
            inputs = series[:, :-1]
            targets = series[:, 1:]
            if self.attn:
                targets = series[:, -self.forecast_length:]

        X = inputs.unsqueeze(-1)
        # X.shape == [batch_size, seq_len - 1, 1]

        X, _ = self.decoder(X)

        return X, targets

    def _forward_full(self, series):
        # series.shape == [batch_size, seq_len]

        if not self.log:
            training_series = series.clone().detach()
            training_series[training_series == 0] = 1
            inputs = training_series[:, 1:] / training_series[:, :-1]
        else:
            inputs = series

        X = inputs.unsqueeze(-1)
        # X.shape == [batch_size, seq_len, 1]

        X, _ = self.decoder(X)

        return X

    def _get_neighbour_embeds(self, X, keys, start, total_len):
        if self.agg_type == 'none':
            return X

        B, T, _ = X.shape
        N = self.max_neighbours

        # We plus one to give us option to either peek or not
        masks = X.new_ones(B, N, total_len).bool()
        # We set neighs to be full precision since some view counts are
        # bigger than 65504
        neighs = X.new_zeros(B, N, total_len, dtype=torch.float32)

        for b, key in enumerate(keys):
            # in_degrees maps node_id to a sorted list of dicts
            # a dict key looks like: {'id': 123, 'mask'; [0, 0, 1]}
            if key in self.in_degrees:
                for i in range(N):
                    if i >= len(self.in_degrees[key]):
                        break
                    n = self.in_degrees[key][i]
                    neighs[b, i] = self.series[n['id']][start:start+total_len]
                    masks[b, i] = n['mask'][start:start+total_len]

        # neighs.shape == [batch_size, n_neighbors, seq_len]
        # masks.shape == [batch_size, n_neighbors, seq_len]

        if self.log:
            neighs = torch.log1p(neighs)

        neighs = neighs.reshape(B * N, total_len)
        # Detach as we won't back-propagate to the neighbours
        # This will also prevent gradient overflow in mixed precision training
        Xn = self._forward_full(neighs)
        if self.detach:
            Xn = Xn.detach()

        if not self.log:
            Xn = Xn.reshape(B, N, total_len - 1, -1)
            masks = masks[:, :, 1:]
        else:
            Xn = Xn.reshape(B, N, total_len, -1)

        if self.peek:
            Xn = Xn[:, :, 1:]
            masks = masks[:, :, 1:]
        else:
            Xn = Xn[:, :, :-1]
            masks = masks[:, :, -1:]

        X_out = self._aggregate(X, Xn, masks)
        return X_out

    def _aggregate(self, X, Xn, masks):
        # X.shape == [batch_size, seq_len, hidden_size]
        # Xn.shape == [batch_size, n_neighs, seq_len, hidden_size]
        # masks.shape == [batch_size, n_neighs, seq_len]

        # Mask out irrelevant values.
        Xn = Xn.clone()
        Xn[masks] = 0

        # Let's just take the average
        Xn = Xn.sum(dim=1)
        # Xn.shape == [batch_size, seq_len, hidden_size]

        n_neighs = (~masks).sum(dim=1).unsqueeze(-1)
        # Avoid division by zero
        n_neighs = n_neighs.clamp(min=1)
        # n_neighs.shape == [batch_size, seq_len, 1]

        Xn = Xn / n_neighs
        # Xn.shape == [batch_size, seq_len, hidden_size]

        if self.detach:
            Xn = Xn.detach()

        X_out = torch.cat([X, Xn], dim=-1)
        # Xn.shape == [batch_size, seq_len, 2 * hidden_size]

        return X_out

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

        raw_series = torch.stack(series_list, dim=0)
        # series.shape == [batch_size, seq_len]

        series = torch.log1p(raw_series) if self.log else raw_series

        X, targets = self._forward(series)
        # X.shape == [batch_size, seq_len, hidden_size]
        # targets.shape == [batch_size, seq_len]

        X_full = self._get_neighbour_embeds(X, keys, start, self.total_length)
        # X_full.shape == [batch_size, seq_len, out_hidden_size]

        X_full = self.fc(X_full)
        # X_full.shape == [batch_size, seq_len, 1]

        preds = X_full.squeeze(-1)
        # preds.shape == [batch_size, seq_len]

        if split in ['valid', 'test']:
            preds = preds[:, -self.forecast_length:]
            targets = targets[:, -self.forecast_length:]

        if self.log and self.opt_smape:
            preds = torch.exp(preds)
            if split in ['valid', 'test'] or self.attn:
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
                target_list.append(self.series[key][s:e])
            targets = torch.stack(target_list, dim=0)
            # targets.shape == [batch_size, forecast_len]

            preds = targets.new_zeros(*targets.shape)

            series = series[:, :-self.forecast_length]
            current_views = series[:, -1]
            for i in range(self.forecast_length):
                X = self._forward_full(series)
                seq_len = self.total_length - self.forecast_length + i + 1
                X_full = self._get_neighbour_embeds(
                    X, keys, start, seq_len)
                X_full = self.fc(X_full)
                pred = X_full.squeeze(-1)[:, -1]
                # delta.shape == [batch_size]

                if not self.log:
                    current_views = current_views * pred
                else:
                    current_views = pred
                preds[:, i] = current_views
                series = torch.cat(
                    [series, current_views.unsqueeze(-1)], dim=-1)

            if self.log:
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


class SeriesBiLSTM(nn.Module):
    def __init__(self, hidden_size, n_encoder_layers, n_decoder_layers, n_heads, dropout, forecast_length, backcast_length):
        super().__init__()
        self.in_proj = GehringLinear(1, hidden_size)
        self.dropout = nn.Dropout(dropout)
        # self.layers = nn.ModuleList([])
        # for _ in range(n_layers):
        #     self.layers.append(TBEATLayer(hidden_size, dropout, 4))

        self.encoder = nn.LSTM(hidden_size, hidden_size, n_encoder_layers, bidirectional=True,
                               bias=True, batch_first=True, dropout=dropout)

        self.decoder = LSTMDecoder(n_decoder_layers, hidden_size, dropout)

        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.total_length = forecast_length + backcast_length

    def forward(self, X):
        B, T, _ = X.shape
        # X.shape == [batch_size, seq_len, 1]

        X = self.in_proj(X)
        # X.shape == [batch_size, seq_len, hidden_size]

        backcast_seq = X[:, :self.backcast_length]
        # Use the last step in encoder as seed to decoder
        forecast_seq = X[:, self.backcast_length-1:]

        X_backcast, _ = self.encoder(backcast_seq)
        # X_backcast.shape == [backcast_len, batch_size, hidden_size]

        X_forecast = self.decoder(forecast_seq, X_backcast)
        # X_forecast.shape == [backcast_len, batch_size, hidden_size]

        return X_forecast, None


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, bias=False):
        super().__init__()

        self.input_proj = GehringLinear(
            input_embed_dim, source_embed_dim, bias=bias)
        self.output_proj = GehringLinear(
            input_embed_dim + source_embed_dim, output_embed_dim, bias=bias)

    def forward(self, input, source_hids, encoder_padding_mask):
        # input: bsz x input_embed_dim
        # source_hids: srclen x bsz x output_embed_dim

        # x: bsz x output_embed_dim
        x = self.input_proj(input)

        # compute attention
        attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)
        # attn_scores.shape == [src_len, bsz]

        # don't attend over padding
        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.transpose(0, 1)
            # encoder_padding_mask.shape == [src_len, bsz]
            attn_scores = attn_scores.float().masked_fill_(
                encoder_padding_mask,
                float('-inf')
            ).type_as(attn_scores)  # FP16 support: cast to float and back

        attn_scores = F.softmax(attn_scores, dim=0)  # srclen x bsz

        # sum weighted sources
        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)

        x = torch.tanh(self.output_proj(torch.cat((x, input), dim=1)))
        return x, attn_scores


class LSTMDecoder(nn.Module):
    def __init__(self, num_layers, hidden_size, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.layers = nn.ModuleList([])
        self.h = nn.ParameterList([])
        self.c = nn.ParameterList([])
        for layer in range(num_layers):
            input_size = hidden_size * 2 if layer == 0 else hidden_size
            rnn = LSTMCell(input_size=input_size, hidden_size=hidden_size)
            self.layers.append(rnn)
            self.h.append(nn.Parameter(torch.zeros(1, hidden_size)))
            self.c.append(nn.Parameter(torch.zeros(1, hidden_size)))

        self.context_attention = AttentionLayer(
            hidden_size, hidden_size * 2, hidden_size, bias=True)

    def forward(self, X, context, context_mask=None):
        # B x T x C -> T x B x C
        X = X.transpose(0, 1)
        context = context.transpose(0, 1)

        T, B, _ = X.shape
        S = context.shape[0]
        n_layers = len(self.layers)

        prev_hiddens = [self.h[i].expand(B, -1) for i in range(n_layers)]
        prev_cells = [self.c[i].expand(B, -1) for i in range(n_layers)]
        input_feed = X.new_zeros(B, self.hidden_size)
        attn_scores = X.new_zeros(S, T, B)
        outs = []

        for step in range(T):
            # input feeding: concatenate context vector from previous time step
            rnn_input = torch.cat((X[step, :, :], input_feed), dim=1)
            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(rnn_input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                rnn_input = F.dropout(hidden, p=self.dropout,
                                      training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            out, attn_scores[:, step, :] = self.context_attention(
                hidden, context, context_mask)

            out = F.dropout(out, p=self.dropout, training=self.training)
            # out.shape == [B, hidden_size * 2]

            input_feed = out
            outs.append(out)

        # collect outputs across time steps
        X = torch.cat(outs, dim=0).view(T, B, self.hidden_size)

        # T x B x C -> B x T x C
        X = X.transpose(1, 0)

        return X


class SeriesTransformer(nn.Module):
    def __init__(self, hidden_size, n_encoder_layers, n_decoder_layers, n_heads, dropout, forecast_length, backcast_length):
        super().__init__()
        self.in_proj = GehringLinear(1, hidden_size)
        self.dropout = nn.Dropout(dropout)
        # self.layers = nn.ModuleList([])
        # for _ in range(n_layers):
        #     self.layers.append(TBEATLayer(hidden_size, dropout, 4))

        self.transformer = nn.Transformer(
            hidden_size, n_heads, n_encoder_layers, n_decoder_layers,
            hidden_size * 4, dropout, 'gelu')

        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.total_length = forecast_length + backcast_length

        self.embed_pos = nn.Embedding(self.total_length, hidden_size)
        # pos_weights = self._get_embedding(256, hidden_size)
        # self.register_buffer('pos_weights', pos_weights)

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

        # pos_embeds = self.pos_weights.index_select(0, pos.reshape(-1))
        # pos_embeds = pos_embeds.reshape(T, B, -1)

        pos_embeds = self.embed_pos(pos)
        X = X + pos_embeds

        backcast_seq = X[:self.backcast_length]
        # Use the last step in encoder as seed to decoder
        forecast_seq = X[self.backcast_length-1:]

        # We can't attend positions which are True
        T, B, E = forecast_seq.shape
        tgt_mask = X.new_ones(T, T)
        # Zero out the diagonal and everything below
        # We can attend to ourselves and the past
        tgt_mask = torch.triu(tgt_mask, diagonal=1)
        # tgt_mask.shape == [T, T]

        # Call transformer
        forecast_embeds = self.transformer(src=backcast_seq,
                                           src_mask=None,  # encoder can attend to all positions
                                           src_key_padding_mask=None,  # all backcast seqs have the same length
                                           tgt=forecast_seq,
                                           tgt_mask=tgt_mask,  # Prevent decoder from attending to future
                                           tgt_key_padding_mask=None,  # All forecast seqs have same length
                                           memory_mask=None,  # Decoder can attend everywhere in history
                                           memory_key_padding_mask=None,  # all backcast seqs have the same length
                                           )
        # forecast_embeds.shape == [forecast_len, batch_size, hidden_size]

        forecast_embeds = forecast_embeds.transpose(0, 1)
        # forecast_embeds.shape == [batch_size, forecast_len, hidden_size]

        return forecast_embeds, None
