import json
import logging
from typing import Any, Dict, List

import h5py
import numpy as np
import torch
from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn.initializers import InitializerApplicator

from radflow.modules.metrics import get_smape
from radflow.modules.nbeats import NBeatsNet

from .base import BaseModel

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register('nbeats')
class NBEATS(BaseModel):
    def __init__(self,
                 vocab: Vocabulary,
                 data_path: str = './data/vevo/vevo.hdf5',
                 series_len: int = 63,
                 forecast_length: int = 28,
                 backcast_length: int = 224,
                 test_lengths: List[int] = [7],
                 hidden_size: int = 128,
                 dropout: float = 0.2,
                 n_stacks: int = 16,
                 nb_blocks_per_stack: int = 1,
                 missing_p: float = 0.0,
                 thetas_dims: int = 128,
                 share_weights_in_stack: bool = False,
                 slice_idx: int = None,
                 end_offset: int = 0,
                 initializer: InitializerApplicator = InitializerApplicator()):
        super().__init__(vocab)
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.total_length = backcast_length + forecast_length
        self.test_lengths = test_lengths
        self.rs = np.random.RandomState(1234)
        self.hidden_size = hidden_size
        self.device = torch.device('cuda:0')
        self.missing_p = missing_p
        self.end_offset = end_offset
        if initializer:
            initializer(self)

        self.data = h5py.File(data_path, 'r')
        self.series = self.data['views'][...]
        if slice_idx is not None:
            self.series = self.series[:, :, slice_idx]

        self.series_len = series_len
        self.max_start = series_len - self.forecast_length * \
            2 - self.total_length - self.end_offset

        self.net = NBeatsNet(device=torch.device('cuda:0'),
                             stack_types=[
            NBeatsNet.GENERIC_BLOCK] * n_stacks,
            nb_blocks_per_stack=nb_blocks_per_stack,
            forecast_length=forecast_length,
            backcast_length=backcast_length,
            thetas_dims=[thetas_dims] * n_stacks,
            hidden_layer_units=hidden_size,
            share_weights_in_stack=share_weights_in_stack,
            dropout=dropout,
        )

        # Shortcut to create new tensors in the same device as the module
        self.register_buffer('_long', torch.LongTensor(1))
        self.register_buffer('_float', torch.FloatTensor(1))

    def forward(self, keys, splits) -> Dict[str, Any]:
        # Enable anomaly detection to find the operation that failed to compute
        # its gradient.
        # torch.autograd.set_detect_anomaly(True)

        # Occasionally we get duplicate keys due random sampling
        keys = sorted(set(keys))
        split = splits[0]

        p = next(self.parameters())
        # keys.shape == [batch_size]

        if split == 'train':
            if self.max_start == 0:
                start = 0
            else:
                start = self.rs.randint(0, self.max_start)
        elif split == 'valid':
            start = self.max_start + self.forecast_length
        elif split == 'test':
            start = self.max_start + self.forecast_length * 2

        # Find all series of given keys
        end = start + self.total_length
        series = self.series[keys, start:end].astype(np.float32)

        # Forward-fill missing values
        mask = series == -1
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        series = series[np.arange(idx.shape[0])[:, None], idx]

        # Mask out series with missing values in forecast
        keep_mask = series[:, -self.forecast_length] > -1
        series = series[keep_mask]
        keys = np.array(keys)[keep_mask].tolist()

        B = len(keys)
        self.history['_n_batches'] += 1
        self.history['_n_samples'] += B

        out_dict = {
            'loss': None,
            'sample_size': p.new_tensor(B),
        }

        # Fill remaining missing values with 0
        series[series == -1] = 0

        series = torch.from_numpy(series).to(p.device)

        sources = series[:, :self.backcast_length]
        targets = series[:, -self.forecast_length:]

        X = torch.log1p(sources.clamp(min=0))
        _, X = self.net(X)

        # X.shape == [batch_size, forecast_len]

        preds = torch.exp(X)
        numerator = torch.abs(targets - preds)
        denominator = torch.abs(targets) + torch.abs(preds)
        loss = numerator / denominator
        loss[torch.isnan(loss)] = 0
        loss = loss.mean()

        out_dict['loss'] = loss

        # During evaluation, we compute one time step at a time
        if splits[0] in ['valid', 'test']:
            smapes, daily_errors = get_smape(targets, preds)

            out_dict['smapes'] = smapes.tolist()
            out_dict['daily_errors'] = daily_errors.tolist()
            out_dict['keys'] = keys
            out_dict['preds'] = preds.cpu().numpy().tolist()
            self.history['_n_steps'] += smapes.shape[0] * smapes.shape[1]

            for k in self.test_lengths:
                self.step_history[f'smape_{k}'] += np.sum(smapes[:, :k])

        return out_dict

    @classmethod
    def get_params(cls, vocab: Vocabulary, params: Params) -> Dict[str, Any]:

        params_dict: Dict[str, Any] = {}

        params_dict['initializer'] = InitializerApplicator.from_params(
            params.pop('initializer', None))

        return params_dict
