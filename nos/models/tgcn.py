import logging
import math
from typing import Any, Dict

import numpy as np
import numpy.linalg as la
import pandas as pd
import torch
import torch.nn as nn
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn.initializers import InitializerApplicator
from scipy.sparse.csgraph import laplacian
from sklearn.metrics import mean_absolute_error, mean_squared_error

from nos.modules.metrics import get_smape

from .base import BaseModel


@Model.register('tgcn')
class TGCN(BaseModel):
    def __init__(self,
                 vocab: Vocabulary,
                 hidden_size: int,
                 forecast_len: int,
                 adj_path: str = 'data/taxi/sz_adj.csv',
                 initializer: InitializerApplicator = InitializerApplicator()):
        super().__init__(vocab)

        self.hidden_size = hidden_size
        adj_df = pd.read_csv(adj_path, header=None)
        adjacency_matrix = np.array(adj_df, dtype=np.float32)
        self.tgcn = TGCNCell(hidden_size, adjacency_matrix)

        self.W_out = nn.Linear(hidden_size, forecast_len, bias=False)
        self.b_out = nn.Parameter(torch.zeros(forecast_len))
        self.forecast_len = forecast_len
        self.mse = nn.MSELoss()

        # Shortcut to create new tensors in the same device as the module
        self.register_buffer('_long', torch.LongTensor(1))

        if initializer:
            initializer(self)

    def forward(self, x, y, scale, keys, splits) -> Dict[str, Any]:
        # Enable anomaly detection to find the operation that failed to compute
        # its gradient.
        # torch.autograd.set_detect_anomaly(True)

        B, S, N = x.shape
        H = self.hidden_size
        # x.shape == [batch_size, seq_len, n_nodes]
        # y.shape == [batch_size, pre_len, n_nodes]

        outputs = self.tgcn(x)
        # len(outputs) == seq_len
        # outputs[i].shape == [batch_size, n_nodes * hidden_size]

        output = outputs[-1].reshape(B, N, H)
        # output.shape == [batch_size, n_nodes, hidden_size]

        forecasts = self.W_out(output) + self.b_out
        # forecasts.shape == [batch_size, n_nodes, forecast_len]

        forecasts = forecasts.transpose(1, 2)
        # forecasts.shape == [batch_size, forecast_len, n_nodes]

        forecasts = forecasts.reshape(B * self.forecast_len, N)
        # forecasts.shape == [batch_size * forecast_len, n_nodes]

        targets = y.reshape(B * self.forecast_len, N)

        loss = self.mse(forecasts, targets)

        targets = targets.detach().cpu().numpy()
        forecasts = forecasts.detach().cpu().numpy()
        rmse = math.sqrt(mean_squared_error(targets, forecasts)) * scale[0]
        mae = mean_absolute_error(targets, forecasts) * scale[0]
        F_norm = la.norm(targets - forecasts, 'fro') / la.norm(targets, 'fro')
        r2 = 1 - ((targets - forecasts)**2).sum() / \
            ((targets - targets.mean())**2).sum()
        var = 1 - (np.var(targets - forecasts)) / np.var(targets)

        self.history['_n_batches'] += 1
        self.history['_n_samples'] += B
        self.batch_history['rmse'] += rmse
        self.batch_history['mae'] += mae
        self.batch_history['acc'] += 1 - F_norm
        self.batch_history['r2'] += r2
        self.batch_history['var'] += var

        smapes, _ = get_smape(targets, forecasts)
        self.history['_n_steps'] += smapes.shape[0] * smapes.shape[1]
        self.step_history['smape'] += np.sum(smapes)

        out_dict = {
            'loss': loss,
            'sample_size': self._long.new_tensor(B),
        }

        return out_dict


class TGCNCell(nn.Module):
    def __init__(self, hidden_size, adjacency_matrix):
        super().__init__()
        self.hidden_size = hidden_size

        # I'm not convinced that the original authors had the correct code
        # to calculate the Laplacian. Here we'll use the built-in
        # function in scipy.
        self.register_buffer('L_sym', torch.from_numpy(
            laplacian(adjacency_matrix, normed=True)))
        # L_sym.shape == [n_nodes, n_nodes]

        self.W_gates = nn.Linear(hidden_size + 1, hidden_size * 2, bias=False)
        self.b_gates = nn.Parameter(torch.ones(hidden_size * 2))

        self.W_cand = nn.Linear(hidden_size + 1, hidden_size, bias=False)
        self.b_cand = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        B, S, N = x.shape
        state = x.new_zeros(B, N * self.hidden_size)

        outputs = []
        for i in range(S):
            state = self.forward_step(x[:, i], state)
            outputs.append(state)

        return outputs

    def forward_step(self, x, state):
        # x.shape == [batch_size, n_nodes]
        # state.shape == [batch_size, n_nodes * hidden_size]

        gates = self.forward_cell(x, state, self.W_gates, self.b_gates)
        # gates.shape == [batch_size, n_nodes * hidden_size * 2]

        gates = torch.sigmoid(gates)
        # gates.shape == [batch_size, n_nodes * hidden_size * 2]

        NH2 = gates.shape[1]
        r_gates, u_gates = gates.split(NH2 // 2, dim=1)
        # r_gates.shape == u_gates.shape == [batch_size, n_nodes * hidden_size]

        r_state = r_gates * state
        # r_state.shape == [batch_size, n_nodes * hidden_size]

        cand = self.forward_cell(x, r_state, self.W_cand, self.b_cand)
        cand = torch.tanh(cand)
        # cand.shape == [batch_size, n_nodes * hidden_size]

        new_h = u_gates * state + (1 - u_gates) * cand
        # new_h.shape == [batch_size, n_nodes * hidden_size]

        return new_h

    def forward_cell(self, x, state, W, b):
        # x.shape == [batch_size, n_nodes]
        # state.shape == [batch_size, n_nodes * hidden_size]

        B, N = x.shape
        x = x.unsqueeze(2)
        # x.shape == [batch_size, n_nodes, 1]

        state = state.reshape(B, N, self.hidden_size)
        # state.shape == [batch_size, n_nodes, hidden_size]

        x = torch.cat([x, state], dim=2)
        # x.shape == [batch_size, n_nodes, 1 + hidden_size]

        H = x.shape[2]
        # H == 1 + hidden_size

        x = x.transpose(0, 1).transpose(1, 2)
        # x.shape == [n_nodes, 1 + hidden_size, batch_size]

        x = x.reshape(N, H * B)
        # x.shape == [n_nodes, (1 + hidden_size) * batch_size]

        # Graph convolution step
        x = torch.matmul(self.L_sym, x)
        # x.shape == [n_nodes, (1 + hidden_size) * batch_size]

        x = x.reshape(N, H, B)
        # x.shape == [n_nodes, 1 + hidden_size, batch_size]

        x = x.transpose(2, 1).transpose(1, 0)
        # x.shape == [batch_size, n_nodes, 1 + hidden_size]

        # Recurrent step
        x = W(x) + b
        # x.shape == [batch_size, n_nodes, output_size]

        x = x.reshape(B, -1)
        # x.shape == [batch_size, n_nodes * output_size]

        return x
