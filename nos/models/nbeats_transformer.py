# Source: https://github.com/philipperemy/n-beats/blob/master/nbeats_pytorch/model.py
import numpy as np
import torch
import torch.nn as nn
from torch import nn as nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv

from nos.modules.linear import GehringLinear


class NBeatsTransformer(nn.Module):
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    GENERIC_BLOCK = 'generic'

    def __init__(self,
                 device,
                 stack_types=(TREND_BLOCK, SEASONALITY_BLOCK),
                 nb_blocks_per_stack=3,
                 forecast_length=5,
                 backcast_length=10,
                 share_weights_in_stack=False,
                 hidden_layer_units=256,
                 nb_harmonics=None,
                 dropout=0.1,
                 max_neighbours=0,
                 peek=False):
        super().__init__()
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.hidden_layer_units = hidden_layer_units
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        self.nb_harmonics = nb_harmonics
        self.stack_types = stack_types
        self.stacks = []
        self.parameters = []
        self.device = device
        self.dropout = dropout
        self.max_neighbours = max_neighbours
        self.peek = peek
        print(f'| N-Beats')
        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))
        self.parameters = nn.ParameterList(self.parameters)
        self.to(self.device)

        self.project_in_dim = GehringLinear(
            1, hidden_layer_units, bias=False)

        if max_neighbours > 0:
            self.attn = nn.MultiheadAttention(
                hidden_layer_units, 4, dropout=dropout, bias=True,
                add_bias_kv=True, add_zero_attn=True, kdim=None, vdim=None)
            self.theta_f_fc = self.theta_b_fc = GehringLinear(
                hidden_layer_units, hidden_layer_units, bias=False)
            self.forecast_fc = GehringLinear(
                hidden_layer_units, forecast_length)

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        print(
            f'| --  Stack {stack_type.title()} (#{stack_id}) (share_weights_in_stack={self.share_weights_in_stack})')
        blocks = []
        for block_id in range(self.nb_blocks_per_stack):
            block_init = NBeatsTransformer.select_block(stack_type)
            if self.share_weights_in_stack and block_id != 0:
                # pick up the last one when we share weights.
                block = blocks[-1]
            else:
                block = block_init(self.hidden_layer_units,
                                   self.device, self.backcast_length, self.forecast_length,
                                   self.nb_harmonics, self.dropout, self.max_neighbours)
                self.parameters.extend(block.parameters())
            print(f'     | -- {block}')
            blocks.append(block)
        return blocks

    @staticmethod
    def select_block(block_type):
        return GenericBlock

    def forward(self, backcast, backcast_n=None, target_n=None, X_neigh_masks=None):
        # backcast.shape == [batch_size, backcast_len]
        # backcast_n.shape == [batch_size, n_neighs, backcast_len]
        backcast = backcast.to(self.device)

        B = backcast.shape[0]
        T = self.forecast_length
        forecast = torch.zeros(size=(B, T)).to(self.device)
        # forecast.shape == [batch_size, forecast_len]

        if self.max_neighbours > 0:
            backcast_n = backcast_n.to(self.device)
            # backcast_n.shape == [batch_size, n_neighs, backcast_len]

            B, N, S = backcast_n.shape
            forecast_n = torch.zeros(size=(B, N, T)).to(self.device)
            # forecast_n.shape == [batch_size, n_neighs, forecast_len]

        x_list = []
        xn_list = []
        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                backcast_embed = backcast.unsqueeze(-1)
                # backcast.shape == [batch_size, backcast_len, 1]

                backcast_embed = self.project_in_dim(backcast_embed)
                # backcast.shape == [batch_size, backcast_len, hidden_size]

                backcast_n_embed = None
                if self.max_neighbours > 0:
                    backcast_n_embed = backcast_n.unsqueeze(-1)
                    # backcast_n.shape == [batch_size, n_neighs, backcast_len, 1]

                    backcast_n_embed = self.project_in_dim(backcast_n_embed)
                    # backcast_n.shape == [batch_size, n_neighs, backcast_len, hidden_size]

                b, f, bn, fn, x_i, xn_i = self.stacks[stack_id][block_id](
                    backcast_embed, backcast_n_embed, X_neigh_masks)
                # x_i.shape == [batch_size, embed_dim]
                # xn_i.shape == [batch_size, n_neighs, embed_dim]
                x_list.append(x_i)
                xn_list.append(xn_i)

                backcast = backcast - b
                # backcast.shape == [B, S]

                forecast = forecast + f
                # forecast.shape == [B, T]

                if self.max_neighbours > 0:
                    backcast_n = backcast_n - bn
                    # backcast_n.shape == [B, N, S]

                    forecast_n = forecast_n + fn
                    # forecast_n.shape == [B, N, T]

        if self.max_neighbours > 0:
            # Aggregate the neighbours
            x_embed = torch.stack(x_list, dim=0)
            # x_embed.shape == [n_layers, batch_size, backcast_len, embed_dim]

            query = x_embed.mean(dim=0).unsqueeze(0)[:, :, -1]
            # query.shape == [1, batch_size, embed_dim]

            xn_embeds = torch.stack(xn_list, dim=0)
            # xn_embeds.shape == [n_layers, batch_size, n_neighs, backcast_len, embed_dim]

            xn_embeds = xn_embeds.mean(dim=0)[:, :, -1]
            # xn_embeds.shape == [batch_size, n_neighs, embed_dim]

            key = value = xn_embeds.transpose(0, 1)
            # xn_embeds.shape == [n_neighs, batch_size, embed_dim]

            attn_output, attn_weights = self.attn(
                query, key, value, key_padding_mask=X_neigh_masks)
            # attn_output.shape == [1, batch_size, embed_dim]
            # attn_weights.shape == [batch_size, 1, n_neighs + 2]

            attn_output = attn_output.squeeze(0)
            # attn_output.shape == [batch_size, embed_dim]

            attn_weights = attn_weights.squeeze(1)[:, :-2]
            # attn_weights.shape == [batch_size, n_neighs]

            theta_f = F.relu(self.theta_f_fc(attn_output))
            fa = self.forecast_fc(theta_f)

            forecast = forecast + fa

        return backcast, forecast


class Block(nn.Module):

    def __init__(self, units, device, backcast_length=10, forecast_length=5,
                 nb_harmonics=None, dropout=0.0, max_neighbours=0):
        super(Block, self).__init__()
        self.units = units
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length

        self.encoder = nn.TransformerEncoderLayer(
            units, 4, dim_feedforward=units * 2, dropout=dropout, activation='gelu')

        self.dropout = dropout
        self.device = device
        self.max_neighbours = max_neighbours
        self.theta_b_fc = GehringLinear(units, units, bias=False)
        self.theta_f_fc = GehringLinear(units, units, bias=False)

        # Shortcut to create new tensors in the same device as the module
        self.register_buffer('_long', torch.LongTensor(1))
        self.register_buffer('_float', torch.FloatTensor(1))

    def forward(self, x, X_neighs=None, X_neigh_masks=None):
        x = self.encoder(x)
        # x.shape == [batch_size, backcast_len, hidden_size]

        # attn_weights = None
        if X_neighs is not None:
            B, N, S, E = X_neighs.shape
            X_neighs = X_neighs.reshape(B * N, S, E)
            X_neigh_masks = X_neigh_masks.reshape(B * N)
            X_neighs[X_neigh_masks] = 0

            X_neighs = X_neighs.reshape(B, N, S, E)
            X_neigh_masks = X_neigh_masks.reshape(B, N)

            # attn_weights = self._get_attn_weights(X_neighs, X_neigh_masks, x)
            # attn_weights.shape == [batch_size, 1, n_neighs]

        return x, X_neighs

    def __str__(self):
        block_type = type(self).__name__
        return f'{block_type}(units={self.units}, ' \
            f'backcast_length={self.backcast_length}, forecast_length={self.forecast_length}, ' \
            f'at @{id(self)}'


class GenericBlock(Block):

    def __init__(self, units, device, backcast_length=10, forecast_length=5, nb_harmonics=None, dropout=0.1, max_neighbours=0):
        super(GenericBlock, self).__init__(units,
                                           device, backcast_length, forecast_length,
                                           dropout=dropout, max_neighbours=max_neighbours)

        self.backcast_fc = GehringLinear(units, 1)
        self.forecast_fc = GehringLinear(units, forecast_length)

    def forward(self, x, X_neighs=None, X_neigh_masks=None):
        # no constraint for generic arch.
        x, X_neighs = super(GenericBlock, self).forward(
            x, X_neighs, X_neigh_masks)

        theta_b = F.relu(self.theta_b_fc(x))
        backcast = self.backcast_fc(theta_b).squeeze(-1)

        x_last = x[:, -1]
        theta_f = F.relu(self.theta_f_fc(x_last))
        forecast = self.forecast_fc(theta_f)

        if X_neighs is not None:
            theta_b_n = F.relu(self.theta_b_fc(X_neighs))
            backcast_n = self.backcast_fc(theta_b_n).squeeze(-1)

            x_n_last = X_neighs[:, :, -1]
            theta_f_n = F.relu(self.theta_f_fc(x_n_last))
            forcast_n = self.forecast_fc(theta_f_n)
        else:
            backcast_n = forcast_n = None

        return backcast, forecast, backcast_n, forcast_n, x, X_neighs
