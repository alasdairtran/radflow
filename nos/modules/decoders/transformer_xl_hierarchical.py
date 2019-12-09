import logging

import torch
import torch.nn as nn

from nos.modules import GehringLinear

from .base import Decoder
from .transformer_xl import (PositionalEmbedding, PositionwiseFF,
                             RelPartialLearnableMultiHeadAttn)

logger = logging.getLogger(__name__)


@Decoder.register('transformer_xl_hierarchical_decoder')
class TransformerXLHierarchicalDecoder(nn.Module):
    def __init__(self, dilation, decoder_input_dim, n_layers=12,
                 d_model=512, n_head=8, dropout=0.1, dropatt=0, d_head=64,
                 d_inner=2048, pre_lnorm=False, mem_len=1000, ext_len=0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        self.mem_len = mem_len
        self.d_model = d_model
        self.dilation = dilation

        for _ in range(n_layers):
            layer = TransformerXLLayer(dilation=dilation,
                                       d_model=d_model,
                                       n_head=n_head,
                                       dropout=dropout,
                                       dropatt=dropatt,
                                       d_head=d_head,
                                       d_inner=d_inner,
                                       pre_lnorm=pre_lnorm,
                                       mem_len=mem_len,
                                       ext_len=ext_len)
            self.layers.append(layer)

        self.in_proj = GehringLinear(decoder_input_dim, d_model, bias=False)

        self.mems = [None for _ in self.layers]

    def forward(self, X, seeding=False):
        X = self.in_projs(X)

        all_hiddens = []
        for i, layer in enumerate(self.layers):
            X, hiddens = layer(X, self.mems[i], seeding)
            all_hiddens.append(hiddens)

        self.update_mems(all_hiddens)

        return X

    def reset_mems(self):
        self.mems = [None for _ in self.layers]

    def init_mems(self, B):
        self.mems = []
        param = next(self.parameters())
        for _ in range(self.n_layers):
            D = self.dilation
            bsz = D * B
            mem = param.new_zeros(self.mem_len, bsz, self.d_model)
            self.mems.append(mem)

    def update_mems(self, all_hiddens):
        with torch.no_grad():
            new_mems = []
            for layer in range(self.n_layers):
                hiddens = all_hiddens[layer]
                mems = self.mems[layer]
                new_mem = {}

                if mems is None:
                    new_mem = hiddens[-self.mem_len:].detach()
                else:
                    cat = torch.cat([mems, hiddens], dim=0)
                    new_mem = cat[-self.mem_len:].detach()

                new_mems.append(new_mem)

        self.mems = new_mems

    def get_output_dim(self):
        return self.d_model


class TransformerXLLayer(nn.Module):
    def __init__(self, dilation, d_model=512, n_head=8,
                 dropout=0.1, dropatt=0, d_head=64, d_inner=2048,
                 pre_lnorm=False, mem_len=1000, ext_len=0, clamp_len=10000):
        super().__init__()
        self.layers = RelPartialLearnableDecoderLayer(
            n_head, d_model, d_head, d_inner, dropout,
            ext_len=ext_len, mem_len=mem_len,
            dropatt=dropatt, pre_lnorm=pre_lnorm,
            r_w_bias=None,
            r_r_bias=None,
            output_attentions=True)

        self.dilation = dilation
        self.pos_embed = PositionalEmbedding(d_model)
        self.drop = nn.Dropout(dropout)
        self.clamp_len = clamp_len

    def forward(self, X, mems=None, seeding=False):
        hiddens = {}

        layer = self.layer
        mem = None if mems is None else mems

        X, T = self.dilate(X, seeding)
        X = self.drop(X)

        X = X.transpose(0, 1)
        Q, B, C = X.shape
        M = mem.shape[0] if mem is not None else 0
        K = M + Q

        # Masking current and future time steps (set these masks to 1)
        dec_attn_mask = torch.triu(X.new_ones(
            Q, K), diagonal=1+M).byte()[:, :, None]

        pos_seq = torch.arange(K-1, -1, -1.0, device=X.device,
                               dtype=X.dtype)
        # pos_seq == [999, 998, ..., 0]

        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)

        pos_emb = self.pos_embed(pos_seq)

        hiddens = X
        layer_outputs = layer(X, pos_emb, dec_attn_mask=dec_attn_mask,
                              mems=mem)
        X = layer_outputs[0]

        X = X.transpose(0, 1)
        X = self.undilate(X, T)

        return X, hiddens

    def dilate(self, X, seeding=False):
        D = self.dilation
        B, T, C = X.shape

        if D > 1:
            S = T // D

            # For seeding, remove oldest time steps that don't fit
            if seeding:
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

        return X, T

    def undilate(self, X, T):
        D = self.dilation
        BD, S, C = X.shape
        B = BD // D

        if D > 1:
            X = X.view(B, D, S, C).transpose(1, 2).reshape(B, S * D, C)
            X = X[:, :T]

        return X


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super().__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                                                         d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r, dec_attn_mask=None, mems=None, head_mask=None):

        attn_outputs = self.dec_attn(dec_inp, r,
                                     attn_mask=dec_attn_mask,
                                     mems=mems, head_mask=head_mask)
        ff_output = self.pos_ff(attn_outputs[0])

        outputs = [ff_output] + attn_outputs[1:]

        return outputs
