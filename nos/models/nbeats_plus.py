# Source: https://github.com/philipperemy/n-beats/blob/master/nbeats_pytorch/model.py
import numpy as np
import torch
import torch.nn as nn
from torch import nn as nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv

from nos.modules.linear import GehringLinear


class NBeatsPlusNet(nn.Module):
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    GENERIC_BLOCK = 'generic'

    def __init__(self,
                 device,
                 stack_types=(TREND_BLOCK, SEASONALITY_BLOCK),
                 nb_blocks_per_stack=3,
                 forecast_length=5,
                 backcast_length=10,
                 thetas_dims=(4, 8),
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
        self.thetas_dim = thetas_dims
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

        if max_neighbours > 0:
            embed_dim = hidden_layer_units * len(self.stack_types)
            kdim = vdim = hidden_layer_units * 2
            self.downsample = GehringLinear(embed_dim, hidden_layer_units)
            self.attn_forecast = nn.MultiheadAttention(
                hidden_layer_units, 4, dropout=dropout, bias=True,
                add_bias_kv=True, add_zero_attn=True, kdim=kdim, vdim=vdim)
            self.theta_f_fc = self.theta_b_fc = GehringLinear(
                hidden_layer_units, thetas_dims[-1], bias=False)
            self.forecast_fc = GehringLinear(thetas_dims[-1], 1)
            self.peek_fc = GehringLinear(1, hidden_layer_units)

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        print(
            f'| --  Stack {stack_type.title()} (#{stack_id}) (share_weights_in_stack={self.share_weights_in_stack})')
        blocks = []
        for block_id in range(self.nb_blocks_per_stack):
            block_init = NBeatsPlusNet.select_block(stack_type)
            if self.share_weights_in_stack and block_id != 0:
                # pick up the last one when we share weights.
                block = blocks[-1]
            else:
                block = block_init(self.hidden_layer_units, self.thetas_dim[stack_id],
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
        # maybe batch size here.
        B = backcast.shape[0]
        backcast = backcast.to(self.device)
        forecast = torch.zeros(size=(B, self.forecast_length)).to(self.device)

        if self.max_neighbours > 0:
            B, N, S = backcast_n.shape
            forecast_n = torch.zeros(
                size=(B, N, self.forecast_length)).to(self.device)

        x_list = []
        xn_list = []
        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f, bn, fn, x_i, xn_i = self.stacks[stack_id][block_id](
                    backcast, backcast_n, X_neigh_masks)
                # x_i.shape == [batch_size, embed_dim]
                # xn_i.shape == [batch_size, n_neighs, embed_dim]
                x_list.append(x_i)
                xn_list.append(xn_i)

                backcast = backcast - b
                # backcast.shape == [B, S]

                forecast = forecast + f
                # forecast.shape == [B, T]

                if self.max_neighbours > 0:
                    # attn_weights = attn_weights.unsqueeze(2)
                    # attn_weights.shape == [B, N, 1]
                    # if self.peek and stack_id == 0 and block_id == 0:
                    #     backcast = backcast - (attn_weights * bn).sum(1)
                    #     forecast = forecast + (attn_weights * target_n).sum(1)
                    # else:
                        # backcast = backcast - (attn_weights * bn).sum(1)
                        # forecast = forecast + (attn_weights * fn).sum(1)

                    backcast_n = backcast_n - bn
                    # backcast_n.shape == [B, N, S]

                    forecast_n = forecast_n + fn
                    # forecast_n.shape == [B, N, T]

        if self.max_neighbours > 0:
            # Aggregate the neighbours
            x_embed = torch.cat(x_list, dim=-1)
            # x_embed.shape == [batch_size, n_layers * embed_dim]

            x_embed = x_embed.unsqueeze(0).unsqueeze(2)
            # x_embed.shape == [1, batch_size, 1, n_layers * embed_dim]

            x_embed = x_embed.expand(-1, -1, self.forecast_length, -1)
            # x_embed.shape == [1, batch_size, forecast_len, n_layers * embed_dim]

            x_embed = self.downsample(x_embed)
            # x_embed.shape == [1, batch_size, forecast_len, embed_dim]

            T, B, L, E = x_embed.shape
            query = x_embed.reshape(T, B * L, E)
            # x_embed.shape == [1, batch_size * forecast_len, embed_dim]

            target_n = target_n.unsqueeze(-1)
            # target_n.shape == [batch_size, n_neighs, forecast_len, 1]

            target_n = self.peek_fc(target_n)
            # target_n.shape == [batch_size, n_neighs, forecast_len, hidden_size]

            xn_embeds = torch.cat(xn_list, dim=-1)
            # xn_embeds.shape == [batch_size, n_neighs, n_layers * embed_dim]

            xn_embeds = self.downsample(xn_embeds)
            # xn_embeds.shape == [batch_size, n_neighs, embed_dim]

            xn_embeds = xn_embeds.unsqueeze(2)
            # xn_embeds.shape == [batch_size, n_neighs, 1, embed_dim]

            xn_embeds = xn_embeds.expand(-1, -1, self.forecast_length, -1)
            # xn_embeds.shape == [batch_size, n_neighs, forecast_len, embed_dim]

            xn_embeds = torch.cat([xn_embeds, target_n], dim=-1)
            # xn_embeds.shape == [batch_size, n_neighs, forecast_len, 2 * embed_dim]

            B, N, L, E = xn_embeds.shape
            key = value = xn_embeds.transpose(0, 1).reshape(N, B * L, E)
            # xn_embeds.shape == [n_neighs, batch_size * forecast_len, 2 * embed_dim]

            X_neigh_masks = X_neigh_masks.unsqueeze(1)
            X_neigh_masks = X_neigh_masks.expand(-1, self.forecast_length, -1)
            X_neigh_masks = X_neigh_masks.reshape(B * L, N)
            # X_neigh_masks.shape == [batch_size * forecast_len, n_neighs]

            attn_output, attn_weights = self.attn_forecast(
                query, key, value, key_padding_mask=X_neigh_masks)
            # attn_output.shape == [1, batch_size * forecast_len, embed_dim]
            # attn_weights.shape == [batch_size, 1, n_neighs + 2]

            attn_output = attn_output.squeeze(0)
            # attn_output.shape == [batch_size * forecast_len, n_layers * embed_dim]

            # attn_weights = attn_weights.squeeze(1)[:, :-1].unsqueeze(2)
            # attn_weights.shape == [batch_size, n_neighs, 1]

            theta_f = F.relu(self.theta_f_fc(attn_output))
            fa = self.forecast_fc(theta_f)
            # fa.shape == [batch_size * forecast_len, 1]

            fa = fa.reshape(B, L)

            # fb = (attn_weights * target_n).sum(1)

            forecast = forecast + fa

        return backcast, forecast


class Block(nn.Module):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, share_thetas=False,
                 nb_harmonics=None, dropout=0.0, max_neighbours=0):
        super(Block, self).__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas
        self.proj_1 = GehringLinear(1, units)
        self.proj_2 = GehringLinear(units, 1)
        self.attn = nn.MultiheadAttention(
            units, units, dropout=dropout, add_bias_kv=True, add_zero_attn=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(units)
        self.device = device
        self.max_neighbours = max_neighbours
        self.theta_b_fc = GehringLinear(units, units)
        self.theta_f_fc = GehringLinear(
            backcast_length * units, units, bias=False)

        # Shortcut to create new tensors in the same device as the module
        self.register_buffer('_long', torch.LongTensor(1))
        self.register_buffer('_float', torch.FloatTensor(1))

    def forward(self, x, X_neighs=None, X_neigh_masks=None):
        # x.shape == [batch_size, backcast_len]

        x = x.unsqueeze(-1)
        # x.shape == [batch_size, backcast_len, 1]

        x = self.proj_1(x)
        # x.shape == [batch_size, backcast_len, hidden_size]

        x1, _ = self.attn(x, x, x, need_weights=False)
        x = x + self.dropout(x1)
        x = self.norm(x)
        # x.shape == [batch_size, backcast_len, hidden_size]

        # x = self.proj_2(x).squeeze(-1)
        # x.shape == [batch_size, backcast_len]

        return x, X_neighs

    def __str__(self):
        block_type = type(self).__name__
        return f'{block_type}(units={self.units}, thetas_dim={self.thetas_dim}, ' \
               f'backcast_length={self.backcast_length}, forecast_length={self.forecast_length}, ' \
               f'share_thetas={self.share_thetas}) at @{id(self)}'


class GenericBlock(Block):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, nb_harmonics=None, dropout=0.1, max_neighbours=0):
        super(GenericBlock, self).__init__(units, thetas_dim,
                                           device, backcast_length, forecast_length,
                                           dropout=dropout, max_neighbours=max_neighbours)

        self.backcast_fc = GehringLinear(units, 1)
        self.forecast_fc = GehringLinear(units, forecast_length)

        # if max_neighbours > 0:
        #     self.backcast_fc_n = GehringLinear(thetas_dim, backcast_length)
        #     self.forecast_fc_n = GehringLinear(thetas_dim, forecast_length)

    def forward(self, x, X_neighs=None, X_neigh_masks=None):
        # no constraint for generic arch.
        x, X_neighs = super(GenericBlock, self).forward(
            x, X_neighs, X_neigh_masks)
        # x.shape == [batch_size, backcast_len, hidden_size]

        theta_b = F.relu(self.theta_b_fc(x))
        # x.shape == [batch_size, backcast_len, hidden_size]

        backcast = self.backcast_fc(theta_b).squeeze(-1)  # generic. 3.3.
        # backcast.shape == [batch_size, backcast_len]

        x = x.view(x.shape[0], -1)
        # x.shape == [batch_size, backcast_len * hidden_size]

        theta_f = F.relu(self.theta_f_fc(x))
        # theta_f.shape == [batch_size, hidden_size]

        forecast = self.forecast_fc(theta_f)  # generic. 3.3.
        # backcast.shape == [batch_size, forecast_len]

        backcast_n = forcast_n = None

        return backcast, forecast, backcast_n, forcast_n, x, X_neighs
