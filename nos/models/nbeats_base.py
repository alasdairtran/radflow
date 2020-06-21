# Source: https://github.com/philipperemy/n-beats/blob/master/nbeats_pytorch/model.py
import numpy as np
import torch
import torch.nn as nn
from torch import nn as nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv

from nos.modules.linear import GehringLinear


class NBeatsNet(nn.Module):
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
        super(NBeatsNet, self).__init__()
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

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        print(
            f'| --  Stack {stack_type.title()} (#{stack_id}) (share_weights_in_stack={self.share_weights_in_stack})')
        blocks = []
        for block_id in range(self.nb_blocks_per_stack):
            block_init = NBeatsNet.select_block(stack_type)
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
        if block_type == NBeatsNet.SEASONALITY_BLOCK:
            return SeasonalityBlock
        elif block_type == NBeatsNet.TREND_BLOCK:
            return TrendBlock
        else:
            return GenericBlock

    def forward(self, backcast, backcast_n=None, target_n=None, X_neigh_masks=None):
        # maybe batch size here.
        B = backcast.shape[0]
        backcast = backcast.to(self.device)
        forecast = torch.zeros(size=(B, self.forecast_length)).to(self.device)

        if backcast_n is not None:
            B, N, S = backcast_n.shape
            forecast_n = torch.zeros(
                size=(B, N, self.forecast_length)).to(self.device)

        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f, bn, fn, attn_weights = self.stacks[stack_id][block_id](
                    backcast, backcast_n, X_neigh_masks)

                backcast = backcast - b
                # backcast.shape == [B, S]

                forecast = forecast + f
                # forecast.shape == [B, T]

                if bn is not None and fn is not None:
                    attn_weights = attn_weights.unsqueeze(2)
                    # attn_weights.shape == [B, N, 1]
                    if self.peek:
                        backcast = backcast - (attn_weights * bn).sum(1)
                        forecast = forecast + \
                            (attn_weights * (target_n - forecast_n)).sum(1)
                    else:
                        backcast = backcast - (attn_weights * bn).sum(1)
                        forecast = forecast + (attn_weights * fn).sum(1)

                    backcast_n = backcast_n - bn
                    # backcast_n.shape == [B, N, S]

                    forecast_n = forecast_n + fn
                    # forecast_n.shape == [B, N, T]

        return backcast, forecast


def seasonality_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p <= thetas.shape[1], 'thetas_dim is too big.'
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    s1 = torch.tensor([np.cos(2 * np.pi * i * t)
                       for i in range(p1)]).float()  # H/2-1
    s2 = torch.tensor([np.sin(2 * np.pi * i * t) for i in range(p2)]).float()
    S = torch.cat([s1, s2])
    return thetas.mm(S.to(device))


def trend_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p <= 4, 'thetas_dim is too big.'
    T = torch.tensor([t ** i for i in range(p)]).float()
    return thetas.mm(T.to(device))


def linspace(backcast_length, forecast_length):
    lin_space = np.linspace(-backcast_length, forecast_length,
                            backcast_length + forecast_length)
    b_ls = lin_space[:backcast_length]
    f_ls = lin_space[backcast_length:]
    return b_ls, f_ls


class Block(nn.Module):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, share_thetas=False,
                 nb_harmonics=None, dropout=0.0, max_neighbours=0):
        super(Block, self).__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas
        self.fc1 = GehringLinear(backcast_length, units, dropout=dropout)
        self.fc2 = GehringLinear(units, units, dropout=dropout)
        self.fc3 = GehringLinear(units, units, dropout=dropout)
        self.fc4 = GehringLinear(units, units, dropout=dropout)
        self.dropout = dropout
        self.device = device
        self.max_neighbours = max_neighbours
        self.backcast_linspace, self.forecast_linspace = linspace(
            backcast_length, forecast_length)
        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = GehringLinear(
                units, thetas_dim, bias=False)
        else:
            self.theta_b_fc = GehringLinear(units, thetas_dim, bias=False)
            self.theta_f_fc = GehringLinear(units, thetas_dim, bias=False)

        if max_neighbours > 0:
            pass
            # self.fc1n = GehringLinear(backcast_length, units, dropout=dropout)
            # self.fc2n = GehringLinear(units, units, dropout=dropout)
            # self.fc3n = GehringLinear(units, units, dropout=dropout)
            # self.fc4n = GehringLinear(units, units, dropout=dropout)
            # self.theta_b_fc_n = GehringLinear(units, thetas_dim, bias=False)
            self.attn = nn.MultiheadAttention(
                units, 4, dropout=dropout, bias=True,
                add_bias_kv=False, add_zero_attn=True, kdim=None, vdim=None)
        #     self.conv = SAGEConv(units, units)

        # Shortcut to create new tensors in the same device as the module
        self.register_buffer('_long', torch.LongTensor(1))
        self.register_buffer('_float', torch.FloatTensor(1))

    def forward(self, x, X_neighs=None, X_neigh_masks=None):
        x = F.relu(self.fc1(x.to(self.device)))
        x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self.fc4(x))
        x = F.dropout(x, self.dropout, self.training)

        attn_weights = None
        if X_neighs is not None:
            B, N, E = X_neighs.shape
            X_neighs = X_neighs.reshape(B * N, E)
            X_neigh_masks = X_neigh_masks.reshape(B * N)

            X_neighs = F.relu(self.fc1(X_neighs.to(self.device)))
            X_neighs[X_neigh_masks] = 0
            X_neighs = F.dropout(X_neighs, self.dropout, self.training)

            X_neighs = F.relu(self.fc2(X_neighs))
            X_neighs[X_neigh_masks] = 0
            X_neighs = F.dropout(X_neighs, self.dropout, self.training)

            X_neighs = F.relu(self.fc3(X_neighs))
            X_neighs[X_neigh_masks] = 0
            X_neighs = F.dropout(X_neighs, self.dropout, self.training)

            X_neighs = F.relu(self.fc4(X_neighs))
            X_neighs[X_neigh_masks] = 0
            X_neighs = F.dropout(X_neighs, self.dropout, self.training)

            X_neighs = X_neighs.reshape(B, N, -1)
            X_neigh_masks = X_neigh_masks.reshape(B, N)

            attn_weights = self._get_attn_weights(X_neighs, X_neigh_masks, x)
            # attn_weights.shape == [batch_size, 1, n_neighs]

        return x, X_neighs, attn_weights

    def _get_attn_weights(self, X_neighs, X_neigh_masks, X):
        # X.shape == [batch_size, embed_dim]
        # X_neighs.shape == [batch_size, n_neighs, embed_dim]

        query = X.unsqueeze(1).transpose(0, 1)
        # query.shape == [1, batch_size, embed_dim]

        key = value = X_neighs.transpose(0, 1)
        # key.shape == value.shape == [n_neighs, batch_size, embed_dim]

        # key_padding_mask: The padding positions are labelled with 1
        attn_output, attn_weights = self.attn(
            query, key, value, key_padding_mask=X_neigh_masks)
        # attn_output.shape == [1, batch_size, embed_dim]
        # attn_weights.shape == [batch_size, 1, n_neighs + 1]

        attn_weights = attn_weights.squeeze(1)[:, :-1]
        # attn_weights.shape == [batch_size, n_neighs]

        # X_attended = attn_output.squeeze(0)
        # X_attended.shape == [batch_size, embed_dim]

        # X_out = torch.cat([X_attended, X], dim=-1)
        # X_out = X + X_attended

        # return self.avg_fc(X_out)
        return attn_weights

    def _get_neighbour_embeds_avg(self, X_neighs, X_neigh_masks, X):
        B = X.shape[0]
        X_neighbors = X_neighbors.reshape(B, self.max_neighbours, -1)
        X_neighbors[mask_list] = 0

        X_sum = X_neighbors.sum(dim=1)
        # X_sum.shape == [B, hidden_size]

        X_mean = X_sum / neighbor_lens
        X_mean[torch.isnan(X_mean)] = 0

        X_out = torch.cat([X_mean, X], dim=-1)

        return self.avg_fc(X_out)

    def _get_neighbour_embeds_old(self, X_neighbors, neighbor_lens, mask_list, X):
        # Go through each element in the batch
        cursor = 0
        X_full_list = []
        for n_neighbors, X_i in zip(neighbor_lens, X):
            X_neighbors_i = X_neighbors[cursor:cursor + n_neighbors]
            # X_neighbors_i == [n_neighbors, seq_len]

            if hasattr(self, 'attn'):
                X_full = self._aggregate_attn(X_neighbors_i, X_i)
            else:
                X_full = self._aggregate_avg(X_neighbors_i, X_i)
            X_full_list.append(X_full)

            cursor += n_neighbors

        X_full = torch.cat(X_full_list, dim=0)
        # X_full.shape [batch_size, seq_len]

        return X_full

    def _aggregate_attn(self, X_neighbors_i, X_i):
        # X_neighbors_i.shape = [n_neighbors, seq_len]
        # X_i.shape == [seq_len]

        X_i = X_i.unsqueeze(0)
        # X_i.shape == [1, seq_len]

        if X_neighbors_i.shape == 0:
            X_out = X_i.new_zeros(*X_i.shape)
        else:
            X_neighbors_i = X_neighbors_i.unsqueeze(1)
            X_i = X_i.unsqueeze(1)
            X_out, _ = self.attn(X_i, X_neighbors_i, X_neighbors_i)
            X_out = X_out.squeeze(1)
            # X_out.shape == [1, 1, seq_len]

        return X_out

    def _aggregate_avg(self, X_neighbors_i, X_i):
        # X_neighbors_i.shape = [n_neighbors, seq_len]
        # X_i.shape == [seq_len]

        X_i = X_i.unsqueeze(0)
        # X_i.shape == [1, seq_len]

        if X_neighbors_i.shape[0] == 0:
            X_out = X_i.new_zeros(*X_i.shape)
        else:
            X_out = X_neighbors_i.mean(dim=0).unsqueeze(0)
            # X_out.shape == [1, seq_len]

            X_full = torch.cat([X_i, X_out], dim=-1)
            X_out = self.avg_fc(X_full)

        return X_out

    def _aggregate_sage(self, X_neighbors_i, X_i):
        # X_neighbors_i.shape = [n_neighbors, seq_len]
        # X_i.shape == [seq_len]

        X_i = X_i.unsqueeze(0)
        # X_i.shape == [1, seq_len]

        # The central node is the first node. The rest are neighbors
        feats = torch.cat([X_i, X_neighbors_i], dim=0)
        # feats.shape == [n_nodes, seq_len]

        N, S = feats.shape
        # feats.shape == [n_nodes, seq_len]

        # We add self-loops as well to make life easier
        source_idx = torch.arange(0, len(feats))
        source_idx = source_idx.to(self.device)
        # source_idx.shape == [n_nodes]

        target_idx = source_idx.new_zeros(*source_idx.shape)

        edge_list = [source_idx, target_idx]
        edge_index = torch.stack(edge_list, dim=0)

        X_full = self.conv(feats, edge_index)
        # X_full.shape == [n_nodes, seq_len]

        X_full = X_full[:1]
        # X_full.shape == [1, seq_len]

        return X_full

    def __str__(self):
        block_type = type(self).__name__
        return f'{block_type}(units={self.units}, thetas_dim={self.thetas_dim}, ' \
               f'backcast_length={self.backcast_length}, forecast_length={self.forecast_length}, ' \
               f'share_thetas={self.share_thetas}) at @{id(self)}'


class SeasonalityBlock(Block):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, nb_harmonics=None, dropout=0.1, max_neighbours=0):
        if nb_harmonics:
            super(SeasonalityBlock, self).__init__(units, nb_harmonics, device, backcast_length,
                                                   forecast_length, share_thetas=True,
                                                   dropout=dropout, max_neighbours=max_neighbours)
        else:
            super(SeasonalityBlock, self).__init__(units, forecast_length, device, backcast_length,
                                                   forecast_length, share_thetas=True,
                                                   dropout=dropout, max_neighbours=max_neighbours)

    def forward(self, x, X_neighs=None, X_neigh_masks=None):
        x, X_neighs = super(SeasonalityBlock, self).forward(
            x, X_neighs, X_neigh_masks)
        backcast = seasonality_model(self.theta_b_fc(
            x), self.backcast_linspace, self.device)
        forecast = seasonality_model(self.theta_f_fc(
            x), self.forecast_linspace, self.device)

        if X_neighs is not None:
            X_neighs = seasonality_model(self.theta_b_fc(
                X_neighs), self.backcast_linspace, self.device)
        return backcast, forecast, X_neighs


class TrendBlock(Block):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, nb_harmonics=None, dropout=0.1, max_neighbours=0):
        super(TrendBlock, self).__init__(units, thetas_dim, device, backcast_length,
                                         forecast_length, share_thetas=True,
                                         dropout=dropout, max_neighbours=max_neighbours)

    def forward(self, x, X_neighs=None, X_neigh_masks=None):
        x, X_neighs = super(TrendBlock, self).forward(
            x, X_neighs, X_neigh_masks)
        backcast = trend_model(self.theta_b_fc(
            x), self.backcast_linspace, self.device)
        forecast = trend_model(self.theta_f_fc(
            x), self.forecast_linspace, self.device)

        if X_neighs is not None:
            X_neighs = trend_model(self.theta_b_fc(
                X_neighs), self.backcast_linspace, self.device)

        return backcast, forecast, X_neighs


class GenericBlock(Block):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, nb_harmonics=None, dropout=0.1, max_neighbours=0):
        super(GenericBlock, self).__init__(units, thetas_dim,
                                           device, backcast_length, forecast_length,
                                           dropout=dropout, max_neighbours=max_neighbours)

        self.backcast_fc = GehringLinear(thetas_dim, backcast_length)
        self.forecast_fc = GehringLinear(thetas_dim, forecast_length)

        # if max_neighbours > 0:
        #     self.backcast_fc_n = GehringLinear(thetas_dim, backcast_length)
        #     self.forecast_fc_n = GehringLinear(thetas_dim, forecast_length)

    def forward(self, x, X_neighs=None, X_neigh_masks=None):
        # no constraint for generic arch.
        x, X_neighs, attn_weights = super(GenericBlock, self).forward(
            x, X_neighs, X_neigh_masks)

        theta_b = F.relu(self.theta_b_fc(x))
        theta_f = F.relu(self.theta_f_fc(x))

        backcast = self.backcast_fc(theta_b)  # generic. 3.3.
        forecast = self.forecast_fc(theta_f)  # generic. 3.3.

        if X_neighs is not None:
            theta_b_n = F.relu(self.theta_b_fc(X_neighs))
            theta_f_n = F.relu(self.theta_f_fc(X_neighs))

            backcast_n = self.backcast_fc(theta_b_n)
            forcast_n = self.forecast_fc(theta_f_n)
        else:
            backcast_n = forcast_n = None

        return backcast, forecast, backcast_n, forcast_n, attn_weights
