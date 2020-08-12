import logging
from typing import Any, Dict, List

import h5py
import numpy as np
import pmdarima as pm
import torch
import torch.nn as nn
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn.initializers import InitializerApplicator
from joblib import Parallel, delayed

from .base import BaseModel
from .metrics import get_smape

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def train_arima(series, forecast_len, k):
    # Ignore missing values
    series = series[series > -1]
    if len(series) == 0:
        return [0] * forecast_len, ''
    automodel = pm.auto_arima(np.log1p(series), m=7, seasonal=True,
                              suppress_warnings=True)
    preds = np.exp(automodel.predict(forecast_len)) - 1
    summary = automodel.summary().as_text()

    return k, preds, summary


@Model.register('arima')
class ARIMA(BaseModel):
    def __init__(self,
                 vocab: Vocabulary,
                 data_path: str = './data/vevo/vevo.hdf5',
                 forecast_length: int = 7,
                 backcast_length: int = 42,
                 test_lengths: List[int] = [7],
                 initializer: InitializerApplicator = InitializerApplicator()):
        super().__init__(vocab)
        self.mse = nn.MSELoss()
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.total_length = forecast_length + backcast_length
        self.test_lengths = test_lengths
        self.rs = np.random.RandomState(1234)

        self.data = h5py.File(data_path, 'r')
        self.series = self.data['views'][...]

        self.register_buffer('_float', torch.tensor(0.1))

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
        assert split == 'test'

        # Find all series of given keys
        series = self.series[keys]
        # series.shape == [batch_size, total_length]

        # Forward-fill out missing values
        mask = series == -1
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        series = series[np.arange(idx.shape[0])[:, None], idx]

        # Train data
        train_series = series[:, :-self.forecast_length]
        with Parallel(n_jobs=B, backend='loky') as parallel:
            results = parallel(delayed(train_arima)(s, self.forecast_length, k)
                               for s, k in zip(train_series, keys))

        preds_list = []
        summary_list = []

        for i, (k, pred, summary) in enumerate(list(results)):
            assert k == keys[i]
            preds_list.append(pred)
            summary_list.append(summary)

        preds = np.vstack(preds_list)
        targets = series[:, -self.forecast_length:]

        # During evaluation, we compute one time step at a time
        smapes, daily_errors = get_smape(targets, preds)

        out_dict['smapes'] = smapes.tolist()
        out_dict['daily_errors'] = daily_errors.tolist()
        out_dict['keys'] = keys
        out_dict['preds'] = preds.tolist()
        out_dict['arima'] = summary_list
        self.history['_n_steps'] += smapes.shape[0] * smapes.shape[1]

        for k in self.test_lengths:
            self.step_history[f'smape_{k}'] += np.sum(smapes[:, :k])
        return out_dict
