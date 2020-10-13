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

from nos.modules.linear import GehringLinear
from nos.modules.metrics import get_smape

from .base import BaseModel
from .radflow import LSTMDecoder

logger = logging.getLogger(__name__)


@Model.register('naive_taxi')
class NaiveTaxi(BaseModel):
    def __init__(self,
                 vocab: Vocabulary,
                 lambda_loss: float,
                 initializer: InitializerApplicator = InitializerApplicator()):
        super().__init__(vocab)

        self.lambda_loss = lambda_loss

        # Shortcut to create new tensors in the same device as the module
        self.register_buffer('_long', torch.LongTensor(1))

        if initializer:
            initializer(self)

    def forward(self, x, y, scale, keys, splits) -> Dict[str, Any]:
        # Enable anomaly detection to find the operation that failed to compute
        # its gradient.
        # torch.autograd.set_detect_anomaly(True)

        B, S, N = x.shape
        T = y.shape[1]
        # x.shape == [batch_size, seq_len, n_nodes]
        # y.shape == [batch_size, pre_len, n_nodes]

        forecasts = x[:, -1:, :]
        # forecasts.shape == [batch_size, 1, n_nodes]

        forecasts = forecasts.expand(-1, T, -1)
        # forecasts.shape == [batch_size, pre_len, n_nodes]

        forecasts = forecasts.reshape(B * T, N)

        targets = y.reshape(B * T, N)
        # targets.shape == [batch_size * pre_len, n_nodes]

        loss = ((forecasts - targets)**2).sum() / 2

        l2_reg = 0
        for p in self.parameters():
            l2_reg += (p**2).sum() / 2
        l2_reg = self.lambda_loss * l2_reg
        loss += l2_reg

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
        self.batch_history['_mae'] += mae
        self.batch_history['_acc'] += 1 - F_norm
        self.batch_history['_r2'] += r2
        self.batch_history['_var'] += var

        smapes, _ = get_smape(targets, forecasts)
        self.history['_n_steps'] += smapes.shape[0] * smapes.shape[1]
        self.step_history['smape'] += np.sum(smapes)

        out_dict = {
            'loss': loss,
            'sample_size': self._long.new_tensor(B),
            'keys': keys,
        }

        return out_dict


@Model.register('tgcn')
class TGCN(BaseModel):
    def __init__(self,
                 vocab: Vocabulary,
                 hidden_size: int,
                 forecast_len: int,
                 lambda_loss: float,
                 adj_path: str = 'data/taxi/sz_adj.csv',
                 initializer: InitializerApplicator = InitializerApplicator()):
        super().__init__(vocab)

        self.hidden_size = hidden_size
        adj_df = pd.read_csv(adj_path, header=None)
        adjacency_matrix = np.array(adj_df, dtype=np.float32)
        self.tgcn = TGCNCell(hidden_size, adjacency_matrix)
        self.lambda_loss = lambda_loss

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

        loss = ((forecasts - targets)**2).sum() / 2

        l2_reg = 0
        for p in self.parameters():
            l2_reg += (p**2).sum() / 2
        l2_reg = self.lambda_loss * l2_reg
        loss += l2_reg

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
        self.batch_history['_mae'] += mae
        self.batch_history['_acc'] += 1 - F_norm
        self.batch_history['_r2'] += r2
        self.batch_history['_var'] += var

        smapes, _ = get_smape(targets, forecasts)
        self.history['_n_steps'] += smapes.shape[0] * smapes.shape[1]
        self.step_history['smape'] += np.sum(smapes)

        out_dict = {
            'loss': loss,
            'sample_size': self._long.new_tensor(B),
            'keys': keys,
        }

        return out_dict


class TGCNCell(nn.Module):
    def __init__(self, hidden_size, adjacency_matrix):
        super().__init__()
        self.hidden_size = hidden_size

        # I'm not convinced that the original authors had the correct code
        # to calculate the Laplacian. Here we'll use the built-in
        # function in scipy.

        A = adjacency_matrix + np.eye(adjacency_matrix.shape[0])
        row_sum = A.sum(1)
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        L_sym = A.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        L_sym = torch.from_numpy(L_sym).float()

        # L_sym = torch.from_numpy(laplacian(adjacency_matrix, normed=True))
        self.register_buffer('L_sym', L_sym)
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


@Model.register('radflow_taxi')
class RADflowTaxi(BaseModel):
    def __init__(self,
                 vocab: Vocabulary,
                 hidden_size: int,
                 forecast_len: int,
                 n_layers: int,
                 dropout: float,
                 variant: str,
                 remove_self_loops: bool = False,
                 adj_path: str = 'data/taxi/sz_adj.csv',
                 agg_type: str = 'none',
                 n_heads: int = 4,
                 initializer: InitializerApplicator = InitializerApplicator()):
        super().__init__(vocab)

        self.hidden_size = hidden_size
        assert agg_type in ['none', 'attention']
        self.agg_type = agg_type
        adj_df = pd.read_csv(adj_path, header=None)
        self.A = np.array(adj_df, dtype=np.float32) != 0

        # Remove self-loops
        if remove_self_loops:
            self.A = self.A - np.eye(self.A.shape[0])

        self.max_neighs = int(self.A.sum(1).max())
        logger.info(f'Max neighbors: {self.max_neighs}')

        self.decoder = LSTMDecoder(
            hidden_size, n_layers, dropout, variant, 1)

        self.forecast_len = forecast_len

        if agg_type in ['attention']:
            self.attn = nn.MultiheadAttention(
                hidden_size, n_heads, dropout=0.1, bias=True,
                add_bias_kv=True, add_zero_attn=True, kdim=None, vdim=None)
            self.proj_alter = GehringLinear(hidden_size, hidden_size)
            self.proj_ego = GehringLinear(hidden_size, hidden_size)
            self.fc = GehringLinear(hidden_size, 1)

        # Shortcut to create new tensors in the same device as the module
        self.register_buffer('_long', torch.LongTensor(1))
        self.evaluate_mode = False

        if initializer:
            initializer(self)

    def _aggregate_attn(self, X, Xn, masks):
        # X.shape == [batch_size, seq_len, hidden_size]
        # Xn.shape == [batch_size, n_neighs, seq_len, hidden_size]
        # masks.shape == [batch_size, n_neighs, seq_len]

        B, N, T, E = Xn.shape

        X = X.reshape(1, B * T, E)
        # X.shape == [1, batch_size * seq_len, hidden_size]

        Xn = Xn.transpose(0, 1).reshape(N, B * T, E)
        # Xn.shape == [n_neighs, batch_size * seq_len, hidden_size]

        key_padding_mask = masks.transpose(1, 2).reshape(B * T, N)
        # key_padding_mask.shape == [n_neighs, batch_size  * seq_len]

        return_weights = self.evaluate_mode

        X_attn, scores = self.attn(
            X, Xn, Xn, key_padding_mask, return_weights)

        # X_attn.shape == [1, batch_size * seq_len, hidden_size]

        X_out = X_attn.reshape(B, T, E)

        if scores is not None:
            scores = scores.reshape(B, T, -1)
            scores = scores.cpu().numpy()

        return X_out, scores

    def _get_neighbour_embeds(self, X_ego):
        X_ego = X_ego.transpose(0, 1)
        N, B, T, H = X_ego.shape
        # X_ego.shape == [n_nodes, batch_size, total_len, hidden_size]

        # The dataset is fairly small, so a for-loop is not too expensive
        X_neighs = X_ego.new_zeros(N, self.max_neighs, B, T, H)
        masks = np.ones((N, self.max_neighs, B, T), dtype=bool)
        for i in range(N):
            neighs = self.A[i].nonzero()[0]
            n_neighs = len(neighs)
            X_neighs[i, :n_neighs] = X_ego[neighs]
            masks[i, :n_neighs] = 0
            # X_neighs[i].shape == [n_neighs, batch_size, total_len, hidden_size]

        X_neighs = X_neighs.transpose(2, 1).transpose(1, 0)
        # X_neighs.shape == [batch_size, n_nodes, n_neighs, total_len, hidden_size]

        X_ego = X_ego.transpose(0, 1).reshape(B * N, T, H)
        # X_ego.shape == [batch_size * n_nodes, total_len, hidden_size]

        X_neighs = X_neighs.reshape(B * N, self.max_neighs, T, H)
        # X_neighs.shape == [batch_size * n_nodes, n_neighs, total_len, hidden_size]

        masks = torch.from_numpy(masks).to(X_ego.device)
        masks = masks.transpose(2, 1).transpose(1, 0)
        masks = masks.reshape(B * N, self.max_neighs, T)
        # masks.shape == [batch_size * n_nodes, n_neighs, total_len]

        X_out, _ = self._aggregate_attn(X_ego, X_neighs, masks)
        # X_out.shape == [batch_size * n_nodes, total_len, hidden_size]

        return X_out

    def forward(self, x, y, scale, keys, splits) -> Dict[str, Any]:
        # Enable anomaly detection to find the operation that failed to compute
        # its gradient.
        # torch.autograd.set_detect_anomaly(True)

        B, S, N = x.shape
        T = y.shape[1]
        H = self.hidden_size
        # x.shape == [batch_size, seq_len, n_nodes]
        # y.shape == [batch_size, pre_len, n_nodes]

        x = x.transpose(1, 2).reshape(B * N, S)
        y = y.transpose(1, 2).reshape(B * N, T)
        z = torch.cat([x, y], dim=1)
        # z.shape == [batch_size * n_nodes, total_len]

        hiddens, forecasts, _ = self.decoder(z)
        # forecasts.shape == [batch_size * n_nodes, total_len]

        if self.agg_type != 'none':
            hiddens = hiddens.reshape(B, N, S + T, H)
            X_agg = self._get_neighbour_embeds(hiddens)
            # X_agg.shape == [batch_size * n_nodes, total_len, hidden_size]

            X_agg = self.fc(X_agg).squeeze(-1)
            # X_agg.shape == [batch_size * n_nodes, total_len]

            forecasts = forecasts + X_agg
            # forecasts.shape == [batch_size * n_nodes, total_len]

        if splits[0] == 'train':
            forecasts = forecasts[:, :-1]
            targets = z[:, 1:]
        else:
            targets = y
            forecast_list = []
            for i in range(targets.shape[1]):
                hiddens, forecast_i, _ = self.decoder(x)
                preds = forecast_i[:, -1:]
                forecast_list.append(preds)
                x = torch.cat([x, preds], dim=1)
            forecasts = torch.cat(forecast_list, dim=1)

        numerator = torch.abs(targets - forecasts)
        denominator = torch.abs(targets) + torch.abs(forecasts)
        loss = numerator / denominator
        loss = loss.mean()

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
        self.batch_history['_mae'] += mae
        self.batch_history['_acc'] += 1 - F_norm
        self.batch_history['_r2'] += r2
        self.batch_history['_var'] += var

        smapes, _ = get_smape(targets, forecasts)
        self.history['_n_steps'] += smapes.shape[0] * smapes.shape[1]
        self.step_history['smape'] += np.sum(smapes)

        out_dict = {
            'loss': loss,
            'sample_size': self._long.new_tensor(B),
            'keys': keys,
        }

        return out_dict
