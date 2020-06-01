import logging
from typing import Any, Dict

import numpy as np
import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model

from .base import BaseModel
from .metrics import get_smape

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register('naive_previous_day')
class NaivePreviousDayModel(BaseModel):
    def __init__(self, vocab: Vocabulary, n_days=7):
        self.n_days = n_days
        super().__init__(vocab)

    def forward(self, series, keys) -> Dict[str, Any]:
        B = series.shape[0]
        # series.shape == [batch_size, seq_len]

        self.history['_n_batches'] += 1
        self.history['_n_samples'] += B

        out_dict = {
            'loss': None,
            'sample_size': torch.tensor(B).to(series.device),
        }

        if not self.training:
            # Predict the last n_days using the day before
            previous = series[:, -self.n_days-1:-self.n_days]
            # previous.shape == [batch_size, 1]

            preds = previous.expand(B, self.n_days)
            # previous.shape == [batch_size, n_days]

            targets = series[:, -self.n_days:]
            # targets.shape == [batch_size, n_days]

            smape, daily_smapes = get_smape(targets, preds)
            # smape.shape == [batch_size]

            out_dict['smapes'] = smape
            out_dict['daily_smape'] = daily_smapes
            out_dict['keys'] = keys
            out_dict['preds'] = preds.cpu().numpy().tolist()

        return out_dict


@Model.register('naive_seasonal')
class NaiveSeasonalModel(BaseModel):
    def __init__(self, vocab: Vocabulary, n_days=7):
        self.n_days = n_days
        super().__init__(vocab)

    def forward(self, series, keys) -> Dict[str, Any]:
        B = series.shape[0]
        # series.shape == [batch_size, seq_len]

        self.history['_n_batches'] += 1
        self.history['_n_samples'] += B

        out_dict = {
            'loss': None,
            'sample_size': torch.tensor(B).to(series.device),
        }

        if not self.training:
            # Predict the last n_days using the week before
            preds = series[:, -(self.n_days+7):-self.n_days]
            # preds.shape == [batch_size, 7]

            preds = preds.repeat(1, self.n_days // 7 + 1)[:, :self.n_days]
            # preds.shape == [batch_size, n_days]

            targets = series[:, -self.n_days:]
            # targets.shape == [batch_size, n_days]

            smape, daily_smapes = get_smape(targets, preds)
            # smape.shape == [batch_size]

            out_dict['smapes'] = smape
            out_dict['daily_smape'] = daily_smapes
            out_dict['keys'] = keys
            out_dict['preds'] = preds.cpu().numpy().tolist()

        return out_dict


@Model.register('naive_seasonal_diff')
class NaiveSeasonalDiffModel(BaseModel):
    def __init__(self,
                 vocab: Vocabulary):
        self.n_days = 7
        super().__init__(vocab)

    def forward(self, series, keys) -> Dict[str, Any]:
        B = series.shape[0]
        # series.shape == [batch_size, seq_len]

        self.history['_n_batches'] += 1
        self.history['_n_samples'] += B

        out_dict = {
            'loss': None,
            'sample_size': torch.tensor(B).to(series.device),
        }

        if not self.training:
            # Ignore last 7 days
            training_series = series[:, :-self.n_days]

            # Compute the percentage change from last week
            diffs = training_series[:, -7:] / training_series[:, -8:-1]

            targets = series[:, -self.n_days:]
            # targets.shape == [batch_size, n_days]

            preds = series.new_zeros(*targets.shape)

            # Predict the last n_days using pct change from the week before
            curr = training_series[:, -1]
            for i in range(preds.shape[1]):
                preds[:, i] = curr * diffs[:, i]
                curr = preds[:, i]

            smape, _ = get_smape(targets, preds)
            # smape.shape == [batch_size]

            out_dict['smapes'] = smape

        return out_dict


@Model.register('naive_rolling_average')
class NaiveRollingAverageModel(BaseModel):
    def __init__(self, vocab: Vocabulary, n_days=7):
        self.n_days = n_days
        super().__init__(vocab)

    def forward(self, series, keys) -> Dict[str, Any]:
        B = series.shape[0]
        # series.shape == [batch_size, seq_len]

        self.history['_n_batches'] += 1
        self.history['_n_samples'] += B

        out_dict = {
            'loss': None,
            'sample_size': torch.tensor(B).to(series.device),
        }

        if not self.training:
            # Predict using the rolling average of the week before
            last_week = series[:, -self.n_days*2:-self.n_days]
            # last_week.shape == [batch_size, n_days]

            preds = last_week.new_zeros(*last_week.shape)
            for i in range(preds.shape[1]):
                seq = torch.cat([last_week[:, i:], preds[:, :i]], dim=1)
                preds[:, i] = seq.mean(dim=1)

            targets = series[:, -self.n_days:]
            # targets.shape == [batch_size, n_days]

            smape, _ = get_smape(targets, preds)
            # smape.shape == [batch_size]

            out_dict['smapes'] = smape

        return out_dict
