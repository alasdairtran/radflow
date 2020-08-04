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
                 dropout=0.1):
        super().__init__()
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.hidden_layer_units = hidden_layer_units
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        self.nb_harmonics = nb_harmonics
        self.stack_types = stack_types
        self.stacks = nn.ModuleList([])
        self.thetas_dim = thetas_dims
        self.device = device
        self.dropout = dropout
        print(f'| N-Beats')
        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))
        self.to(self.device)

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        print(
            f'| --  Stack {stack_type.title()} (#{stack_id}) (share_weights_in_stack={self.share_weights_in_stack})')
        blocks = nn.ModuleList([])
        for block_id in range(self.nb_blocks_per_stack):
            block_init = NBeatsNet.select_block(stack_type)
            if self.share_weights_in_stack and block_id != 0:
                # pick up the last one when we share weights.
                block = blocks[-1]
            else:
                block = block_init(self.hidden_layer_units, self.thetas_dim[stack_id],
                                   self.device, self.backcast_length, self.forecast_length,
                                   self.nb_harmonics, self.dropout)
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

    def forward(self, backcast):
        # maybe batch size here.
        B = backcast.shape[0]
        backcast = backcast.to(self.device)

        forecast = torch.zeros(size=(B, self.forecast_length)).to(self.device)

        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast)

                backcast = backcast - b
                # backcast.shape == [B, S]

                forecast = forecast + f
                # forecast.shape == [B, T]

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
                 nb_harmonics=None, dropout=0.0):
        super().__init__()
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
        self.backcast_linspace, self.forecast_linspace = linspace(
            backcast_length, forecast_length)
        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = GehringLinear(
                units, thetas_dim, bias=False)
        else:
            self.theta_b_fc = GehringLinear(units, thetas_dim, bias=False)
            self.theta_f_fc = GehringLinear(units, thetas_dim, bias=False)

        # Shortcut to create new tensors in the same device as the module
        self.register_buffer('_long', torch.LongTensor(1))
        self.register_buffer('_float', torch.FloatTensor(1))

    def forward(self, x):
        x = F.relu(self.fc1(x.to(self.device)))
        x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self.fc4(x))
        x = F.dropout(x, self.dropout, self.training)

        return x

    def __str__(self):
        block_type = type(self).__name__
        return f'{block_type}(units={self.units}, thetas_dim={self.thetas_dim}, ' \
               f'backcast_length={self.backcast_length}, forecast_length={self.forecast_length}, ' \
               f'share_thetas={self.share_thetas}) at @{id(self)}'


class SeasonalityBlock(Block):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, nb_harmonics=None, dropout=0.1):
        if nb_harmonics:
            super().__init__(units, nb_harmonics, device, backcast_length,
                             forecast_length, share_thetas=True,
                             dropout=dropout)
        else:
            super().__init__(units, forecast_length, device, backcast_length,
                             forecast_length, share_thetas=True,
                             dropout=dropout)

    def forward(self, x):
        x = super(SeasonalityBlock, self).forward(x)
        backcast = seasonality_model(self.theta_b_fc(
            x), self.backcast_linspace, self.device)
        forecast = seasonality_model(self.theta_f_fc(
            x), self.forecast_linspace, self.device)
        return backcast, forecast


class TrendBlock(Block):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, nb_harmonics=None, dropout=0.1):
        super(TrendBlock, self).__init__(units, thetas_dim, device, backcast_length,
                                         forecast_length, share_thetas=True,
                                         dropout=dropout)

    def forward(self, x):
        x = super().forward(x)
        backcast = trend_model(self.theta_b_fc(
            x), self.backcast_linspace, self.device)
        forecast = trend_model(self.theta_f_fc(
            x), self.forecast_linspace, self.device)
        return backcast, forecast


class GenericBlock(Block):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, nb_harmonics=None, dropout=0.1):
        super().__init__(units, thetas_dim,
                         device, backcast_length, forecast_length,
                         dropout=dropout)

        self.backcast_fc = GehringLinear(thetas_dim, backcast_length)
        self.forecast_fc = GehringLinear(thetas_dim, forecast_length)

        # if max_neighbours > 0:
        #     self.backcast_fc_n = GehringLinear(thetas_dim, backcast_length)
        #     self.forecast_fc_n = GehringLinear(thetas_dim, forecast_length)

    def forward(self, x):
        # no constraint for generic arch.
        x = super().forward(x)

        theta_b = F.relu(self.theta_b_fc(x))
        theta_f = F.relu(self.theta_f_fc(x))

        backcast = self.backcast_fc(theta_b)  # generic. 3.3.
        forecast = self.forecast_fc(theta_f)  # generic. 3.3.

        return backcast, forecast
