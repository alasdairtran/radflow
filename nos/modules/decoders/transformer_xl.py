import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from nos.modules import GehringLinear

from .base import Decoder

logger = logging.getLogger(__name__)

# try:
#     from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
# except ImportError:
#     logger.info('Better speed can be achieved with FusedLayerNorm from apex.')

#     class LayerNorm(nn.Module):
#         """Construct LayerNorm in TF style (epsilon inside the square root)."""

#         def __init__(self, hidden_size, eps=1e-12):
#             super().__init__()
#             self.weight = nn.Parameter(torch.ones(hidden_size))
#             self.bias = nn.Parameter(torch.zeros(hidden_size))
#             self.variance_epsilon = eps

#         def forward(self, x):
#             u = x.mean(-1, keepdim=True)
#             s = (x - u).pow(2).mean(-1, keepdim=True)
#             x = (x - u) / torch.sqrt(s + self.variance_epsilon)
#             return self.weight * x + self.bias


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super().__init__()

        self.demb = demb

        inv_freq = 1 / (1000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        self.inv_freq = self.inv_freq.type_as(pos_seq)

        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


@Decoder.register('transformer_xl_decoder')
class TransformerXLDecoder(Decoder):
    def __init__(self, decoder_input_dim, d_model=512, n_head=8, d_head=64, d_inner=2048,
                 pre_lnorm=False, n_layers=12,
                 mem_len=1000, tgt_len=128,
                 ext_len=0, clamp_len=1000, same_length=True, attn_type=0,
                 dropout=0.1, dropatt=0):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_head
        self.d_inner = d_inner
        self.n_layers = n_layers
        self.tgt_len = tgt_len
        self.ext_len = ext_len
        self.mem_len = mem_len
        self.max_klen = tgt_len + ext_len + mem_len
        self.clamp_len = clamp_len
        self.d_model = d_model
        self.same_length = same_length
        self.attn_type = attn_type

        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList()
        if attn_type == 0:  # the default attention
            for i in range(n_layers):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm,
                        r_w_bias=None,
                        r_r_bias=None,
                        output_attentions=True)
                )
        elif attn_type == 1:  # learnable embeddings
            for i in range(n_layers):
                self.layers.append(
                    RelLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm,
                        r_w_bias=None,
                        r_r_bias=None,
                        output_attentions=True)
                )

        if self.attn_type == 0:  # default attention
            self.pos_emb = PositionalEmbedding(self.d_model)
        elif self.attn_type == 1:  # learnable
            self.r_emb = nn.Parameter(torch.Tensor(
                self.n_layer, self.max_klen, self.n_head, self.d_head))
            self.r_bias = nn.Parameter(torch.Tensor(
                self.n_layer, self.max_klen, self.n_head))

        self.project_in_dim = GehringLinear(
            decoder_input_dim, d_model, bias=False)

    def get_output_dim(self):
        return self.d_model

    def forward(self, X, encoder_out=None, incremental_state=None):
        if hasattr(self, 'project_in_dim'):
            X = self.project_in_dim(X)
            # X.shape == [B, T, C]

        # The original code for Transformer-XL uses [T, B, C]
        X = X.transpose(0, 1).contiguous()
        # X.shape == [T, B, C]

        if incremental_state is None:
            incremental_state = {}
        if 'memories' not in incremental_state:
            mems = self.init_mems(X)
        else:
            mems = incremental_state['memories']

        head_mask = [None] * self.n_layers
        X, mems, inner_states, attn = self._forward(
            X, mems=mems, head_mask=head_mask)
        incremental_state['memories'] = mems

        return X, {'attn': attn, 'inner_states': inner_states}

    def init_mems(self, data):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for _ in range(self.n_layers):
                empty = torch.zeros(self.mem_len, data.size(1), self.d_model,
                                    dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None:
            return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):

                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def _forward(self, X, mems=None, head_mask=None):
        Q, B, C = X.shape
        M = mems[0].shape[0] if mems is not None else 0
        K = M + Q

        if self.same_length:
            all_ones = X.new_ones(Q, K)
            mask_len = K - self.mem_len
            if mask_len > 0:
                mask_shift_len = Q - mask_len
            else:
                mask_shift_len = Q
            dec_attn_mask = (torch.triu(all_ones, 1+M)
                             + torch.tril(all_ones, -mask_shift_len)).byte()[:, :, None]  # -1
        else:
            dec_attn_mask = torch.triu(X.new_ones(
                Q, K), diagonal=1+M).byte()[:, :, None]

        hids = []
        attentions = []
        if self.attn_type == 0:  # default
            pos_seq = torch.arange(K-1, -1, -1.0, device=X.device,
                                   dtype=X.dtype)
            # pos_seq == [999, 998, ..., 0]
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(X)
            pos_emb = self.drop(pos_emb)

            for i, layer in enumerate(self.layers):
                hids.append(core_out)
                mems_i = None if mems is None else mems[i]
                layer_outputs = layer(core_out, pos_emb, dec_attn_mask=dec_attn_mask,
                                      mems=mems_i, head_mask=head_mask[i])
                core_out = layer_outputs[0]
                attentions.append(layer_outputs[1])
        elif self.attn_type == 1:  # learnable
            core_out = self.drop(X)
            for i, layer in enumerate(self.layers):
                hids.append(core_out)
                if self.clamp_len > 0:
                    r_emb = self.r_emb[i][-self.clamp_len:]
                    r_bias = self.r_bias[i][-self.clamp_len:]
                else:
                    r_emb, r_bias = self.r_emb[i], self.r_bias[i]

                mems_i = None if mems is None else mems[i]
                layer_outputs = layer(core_out, r_emb, self.r_w_bias[i],
                                      r_bias, dec_attn_mask=dec_attn_mask,
                                      mems=mems_i, head_mask=head_mask[i])
                core_out = layer_outputs[0]
                attentions.append(layer_outputs[1])

        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, M, Q)

        # We transpose back here to shape [bsz, len, hidden_dim]
        core_out = core_out.transpose(0, 1).contiguous()

        # Add last layer and transpose to library standard shape [bsz, len, hidden_dim]
        hids.append(core_out)
        hids = list(t.transpose(0, 1).contiguous() for t in hids)

        # Transpose to library standard shape [bsz, n_heads, query_seq_len, key_seq_len]
        attentions = list(t.permute(2, 3, 0, 1).contiguous()
                          for t in attentions)

        return core_out, new_mems, hids, attentions


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super().__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            GehringLinear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            GehringLinear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = self.CoreNet(inp)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out).type_as(inp)

        return output


class Projection(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.ff = GehringLinear(d_in, d_out)
        self.layer_norm = nn.LayerNorm(d_out)

    def forward(self, inp):
        core_out = self.ff(inp)

        # residual connection + layer normalization
        output = self.layer_norm(core_out).type_as(inp)

        return output


class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False,
                 r_r_bias=None, r_w_bias=None, output_attentions=False):
        super(RelMultiHeadAttn, self).__init__()

        self.output_attentions = output_attentions
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = GehringLinear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = GehringLinear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

        if r_r_bias is None or r_w_bias is None:  # Biases are not shared
            self.r_r_bias = nn.Parameter(
                torch.Tensor(self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(
                torch.Tensor(self.n_head, self.d_head))
            nn.init.uniform_(self.r_r_bias, -0.1, 0.1)
            nn.init.uniform_(self.r_w_bias, -0.1, 0.1)
        else:
            self.r_r_bias = r_r_bias
            self.r_w_bias = r_w_bias

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m, :m] = torch.triu(mask[:m, :m])
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                   device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:, :, None, None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        # x.shape == [Q, K, B, H]
        zero_pad_shape = (x.size(0), 1) + x.size()[2:]
        zero_pad = torch.zeros(zero_pad_shape, device=x.device, dtype=x.dtype)
        # zero_pad.shape == [Q, 1, B, H]

        x_padded = torch.cat([zero_pad, x], dim=1)
        # x_padded.shape == [Q, 1 + K, B, H]

        x_padded_shape = (x.size(1) + 1, x.size(0)) + x.size()[2:]
        x_padded = x_padded.view(*x_padded_shape)
        # x_padded.shape == [1 + K, Q, B, H]

        x_padded = x_padded[1:]
        # x_padded.shape == [K, Q, B, H]

        x = x_padded.view_as(x)
        # x_padded.shape == [Q, K, B, H]

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.r_net = GehringLinear(
            self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, attn_mask=None, mems=None, head_mask=None):
        r_r_bias = self.r_r_bias.type_as(w)
        r_w_bias = self.r_w_bias.type_as(w)

        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)
            # r_head_k.shape == [K, H * R]

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        # qlen x bsz x n_head x d_head
        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        # qlen x n_head x d_head
        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)
        # r_head_k.shape == [K, H, R]

        # compute attention score
        # qlen x bsz x n_head x d_head
        rw_head_q = w_head_q + r_w_bias
        # qlen x klen x bsz x n_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))

        rr_head_q = w_head_q + r_r_bias
        # rr_head_q.shape == [Q, B, H, R]

        # qlen x klen x bsz x n_head
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))
        # BD.shape == [Q, K, B, H]
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # compute attention probability
        attn_mask = attn_mask.bool()
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.type_as(w).masked_fill(
                    attn_mask[None, :, :, None], float('-inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.type_as(w).masked_fill(
                    attn_mask[:, :, :, None], float('-inf')).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # Mask heads if we want to
        if head_mask is not None:
            attn_prob = attn_prob * head_mask

        # compute attention vector
        attn_vec = torch.einsum(
            'ijbn,jbnd->ibnd', (attn_prob.type_as(w_head_v), w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            outputs = [w + attn_out]
        else:
            # residual connection + layer normalization
            outputs = [self.layer_norm(w + attn_out).type_as(w)]

        if self.output_attentions:
            outputs.append(attn_prob)

        return outputs


class RelLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None, head_mask=None):
        # r_emb: [klen, n_head, d_head], used for term B
        # r_w_bias: [n_head, d_head], used for term C
        # r_bias: [klen, n_head], used for term D

        qlen, bsz = w.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        if klen > r_emb.size(0):
            r_emb_pad = r_emb[0:1].expand(klen-r_emb.size(0), -1, -1)
            r_emb = torch.cat([r_emb_pad, r_emb], 0)
            r_bias_pad = r_bias[0:1].expand(klen-r_bias.size(0), -1)
            r_bias = torch.cat([r_bias_pad, r_bias], 0)
        else:
            r_emb = r_emb[-klen:]
            r_bias = r_bias[-klen:]

        # compute attention score
        # qlen x bsz x n_head x d_head
        rw_head_q = w_head_q + r_w_bias[None]

        # qlen x klen x bsz x n_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))
        # qlen x klen x bsz x n_head
        B_ = torch.einsum('ibnd,jnd->ijbn', (w_head_q, r_emb))
        # 1    x klen x 1   x n_head
        D_ = r_bias[None, :, None]
        BD = self._rel_shift(B_ + D_)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(
                    attn_mask[None, :, :, None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(
                    attn_mask[:, :, :, None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        if head_mask is not None:
            attn_prob = attn_prob * head_mask

        # compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            outputs = [w + attn_out]
        else:
            # residual connection + layer normalization
            outputs = [self.layer_norm(w + attn_out)]

        if self.output_attentions:
            outputs.append(attn_prob)

        return outputs


class RelLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super().__init__()

        self.dec_attn = RelLearnableMultiHeadAttn(n_head, d_model, d_head, dropout,
                                                  **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r_emb, r_w_bias, r_bias, dec_attn_mask=None, mems=None, head_mask=None):

        attn_outputs = self.dec_attn(dec_inp, r_emb, r_w_bias, r_bias,
                                     attn_mask=dec_attn_mask,
                                     mems=mems, head_mask=head_mask)
        ff_output = self.pos_ff(attn_outputs[0])

        outputs = [ff_output] + attn_outputs[1:]

        return outputs


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
