# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
from typing import Type

import torch
import torch.nn.functional as F
from allennlp.common.params import Params
from allennlp.modules import Seq2SeqEncoder

from nos.modules.linear import GehringLinear

from .base import Decoder

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Decoder.register('lstm_decoder')
class LSTMDecoder(Decoder):
    def __init__(self, vocab, rnn: Seq2SeqEncoder, decoder_input_dim, hidden_size=32,
                 decoder_output_dim=None, dropout=0.1, dilation=1):
        super().__init__()
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.rnn = rnn

        self.project_in_dim = GehringLinear(
            decoder_input_dim, hidden_size, bias=False)

        if decoder_output_dim is None:
            decoder_output_dim = hidden_size
        elif hidden_size != decoder_output_dim:
            self.project_out_dim = GehringLinear(
                hidden_size, decoder_output_dim, bias=False)

        self.dilation = dilation
        self.decoder_output_dim = decoder_output_dim

    def get_output_dim(self) -> int:
        return self.decoder_output_dim

    def forward(self, X, encoder_out=None, incremental_state=None, **kwargs):
        """
        Args:
            X (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the last decoder layer's output of shape `(batch, tgt_len,
                  vocab)`
                - the last decoder layer's attention weights of shape `(batch,
                  tgt_len, src_len)`
        """

        B, T, C = X.shape
        D = self.dilation
        if self.dilation > 1:
            S = T // D

            # For seeding, remove oldest time steps that don't fit
            if not incremental_state:
                T_new = S * D
                X = X[:, -T_new:]

            # For non-seeds, pad the right with zeros
            elif T % D != 0:
                P = D - (T % D)
                pad = X.new_zeros(B, P, C)
                X = torch.cat([X, pad], dim=1)
                S = S + 1

            dims = (B, D, S, C)
            strides = (D * S * C, C, D * C, 1)
            # Forgetting to make X contiguous before calling as_strided is
            # the most difficult bug to spot in the last year :-/
            X = X.contiguous().as_strided(dims, strides).contiguous()
            X = X.view(B * D, S, C)

        if hasattr(self, 'project_in_dim'):
            X = self.project_in_dim(X)

        X = F.dropout(X, p=self.dropout, training=self.training)

        # Assume that there's no padding
        mask = (~torch.isnan(X)).any(dim=-1)
        # mask.shape == [batch_size, seq_len]

        if not incremental_state:
            self.rnn.reset_states()

        X = self.rnn(X, mask)
        # X.shape == [batch_size, seq_len, hidden_size]

        if self.dilation > 1:
            X = X.view(B, D, S, -1).transpose(1, 2).reshape(B, S * D, -1)
            if incremental_state:
                X = X[:, :T]

        if not incremental_state:
            incremental_state['_state'] = self.rnn._states

        if hasattr(self, 'project_out_dim'):
            X = self.project_out_dim(X)
            X = F.dropout(X, p=self.dropout, training=self.training)

        return X, {'attn': None, 'inner_states': None}

    @classmethod
    def from_params(cls: Type['LSTMDecoder'],
                    params: Params,
                    **extras) -> 'LSTMDecoder':

        for k, v in extras.items():
            params[k] = v

        rnn_params = params['rnn']
        rnn_type = rnn_params.get('type')
        rnn_params['input_size'] = params['hidden_size']
        rnn_params['hidden_size'] = params['hidden_size']
        if rnn_type in ['lstm', 'gru']:
            rnn_params['dropout'] = params['dropout']
        elif rnn_type in ['augmented_lstm']:
            rnn_params['recurrent_dropout_probability'] = params['dropout']
        else:
            raise ValueError(f'Unknown RNN type: {rnn_type}')

        return super().from_params(params)
