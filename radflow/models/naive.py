import logging
import os
from typing import Any, Dict, List

import h5py
import numpy as np
import torch
import torch.nn as nn
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn.initializers import InitializerApplicator

from radflow.modules.metrics import get_smape

from .base import BaseModel

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register('naive')
class NaiveForecaster(BaseModel):
    def __init__(self,
                 vocab: Vocabulary,
                 data_path: str = './data/vevo/vevo.hdf5',
                 multi_views_path: str = None,
                 series_len: int = 63,
                 method: str = 'previous_day',
                 forecast_length: int = 7,
                 backcast_length: int = 42,
                 test_lengths: List[int] = [7],
                 end_offset: int = 0,
                 ignore_test_zeros: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator()):
        super().__init__(vocab)
        self.mse = nn.MSELoss()
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.total_length = backcast_length + forecast_length
        self.test_lengths = test_lengths
        self.rs = np.random.RandomState(1234)
        self.device = torch.device('cuda:0')
        self.method = method
        self.end_offset = end_offset
        self.ignore_test_zeros = ignore_test_zeros

        if os.path.exists(data_path):
            self.data = h5py.File(data_path, 'r')
            self.series = self.data['views'][...]

        self.views_all = None
        if multi_views_path and os.path.exists(multi_views_path):
            self.views_all = h5py.File(multi_views_path, 'r')['views']

        assert method in ['previous_day', 'previous_week']

        # Shortcut to create new tensors in the same device as the module
        self.register_buffer('_long', torch.LongTensor(1))
        self.register_buffer('_float', torch.tensor(0.1))

        self.series_len = series_len
        self.max_start = series_len - self.forecast_length * \
            2 - self.total_length - self.end_offset

    def forward(self, keys, splits) -> Dict[str, Any]:
        # Enable anomaly detection to find the operation that failed to compute
        # its gradient.
        # torch.autograd.set_detect_anomaly(True)

        # Occasionally we get duplicate keys due random sampling
        keys = sorted(set(keys))
        split = splits[0]
        B = len(keys)
        # keys.shape == [batch_size]

        self.history['_n_batches'] += 1
        self.history['_n_samples'] += B

        out_dict = {
            'loss': None,
            'sample_size': self._float.new_tensor(B),
        }

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

        if self.views_all:
            series = self.views_all[keys, start:end].astype(np.float32)
        else:
            series = self.series[keys, start:end].astype(np.float32)

        if self.views_all:
            B, S, E = series.shape
            series = series.transpose((0, 2, 1)).reshape(B, E * S)
            mask = series == -1
            idx = np.where(~mask, np.arange(mask.shape[1]), 0)
            np.maximum.accumulate(idx, axis=1, out=idx)
            series = series[np.arange(idx.shape[0])[:, None], idx]
            series = series.reshape(B, E, S).transpose((0, 2, 1))
        else:
            mask = series == -1
            idx = np.where(~mask, np.arange(mask.shape[1]), 0)
            np.maximum.accumulate(idx, axis=1, out=idx)
            series = series[np.arange(idx.shape[0])[:, None], idx]

        series = torch.from_numpy(series)

        sources = series[:, :self.backcast_length]
        targets = series[:, -self.forecast_length:]
        B = sources.shape[0]
        if self.method == 'previous_day':
            preds = sources[:, -1:]
            if self.views_all:
                preds = preds.expand(-1, self.forecast_length, -1)
            else:
                preds = preds.expand(-1, self.forecast_length)
        elif self.method == 'previous_week':
            preds = sources[:, -7:]
            if self.views_all:
                preds = preds.repeat(1, self.forecast_length // 7 + 1, 1)
            else:
                preds = preds.repeat(1, self.forecast_length // 7 + 1)
            preds = preds[:, :self.forecast_length]

        loss = self.mse(torch.log1p(preds.float()),
                        torch.log1p(targets.float()))
        out_dict['loss'] = loss

        # During evaluation, we compute one time step at a time
        if splits[0] in ['test']:
            targets = targets.cpu().numpy()
            preds = preds.cpu().numpy()

            if self.ignore_test_zeros:
                nz = targets != 0
                targets = targets[nz]
                preds = preds[nz]

            smapes, daily_errors = get_smape(targets, preds)
            # if self.views_all:
            #     n_cats = smapes.shape[-1]
            #     for i in range(n_cats):
            #         for k in self.test_lengths:
            #             self.step_history[f'smape_{i}_{k}'] += np.sum(
            #                 smapes[:, :k, i])

            rmse = (targets - preds)**2
            mae = np.abs(targets - preds)

            out_dict['smapes'] = smapes.tolist()
            out_dict['daily_errors'] = daily_errors.tolist()
            out_dict['keys'] = keys
            out_dict['preds'] = preds.tolist()
            if self.views_all:
                self.history['_n_steps'] += smapes.shape[0] * \
                    smapes.shape[1] * smapes.shape[2]
            elif len(smapes.shape) == 2:
                self.history['_n_steps'] += smapes.shape[0] * smapes.shape[1]
            else:
                self.history['_n_steps'] += smapes.shape[0]

            k = self.test_lengths[-1]
            self.step_history[f'smape_{k}'] += np.sum(smapes)
            self.squared_step_history[f'_rmse_{k}'] += np.sum(rmse)
            self.step_history[f'_mae_{k}'] += np.sum(mae)

        return out_dict
