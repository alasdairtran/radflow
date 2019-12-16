import logging
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn.initializers import InitializerApplicator

from nos.modules import Decoder, GehringLinear

from .base import BaseModel
from .metrics import get_smape

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register('time_series_transformer')
class TimeSeriesTransformer(BaseModel):
    def __init__(self,
                 vocab: Vocabulary,
                 decoder: Decoder,
                 initializer: InitializerApplicator = InitializerApplicator()):
        super().__init__(vocab)
        self.decoder = decoder
        self.mse = nn.MSELoss()
        self.fc = GehringLinear(32, 1)
        self.n_days = 7
        initializer(self)

    def forward(self, series) -> Dict[str, Any]:
        # Enable anomaly detection to find the operation that failed to compute
        # its gradient.
        # torch.autograd.set_detect_anomaly(True)

        B = series.shape[0]
        # series.shape == [batch_size, seq_len]

        self.history['_n_batches'] += 1
        self.history['_n_samples'] += B

        out_dict = {
            'loss': None,
            'sample_size': torch.tensor(B).to(series.device),
        }

        training_series = series.clone().detach()
        training_series[training_series == 0] = 1

        # Take the difference
        diff = training_series[:, 1:] / training_series[:, :-1]
        targets = diff[:, 1:]
        inputs = diff[:, :-1]

        X = inputs.unsqueeze(-1)
        # X.shape == [batch_size, seq_len, 1]

        self.decoder.reset_mems()
        X = self.decoder(X, seeding=False)
        X = self.fc(X)
        preds = X.squeeze(-1)

        if not self.training:
            preds = preds[-self.n_days:]
            targets = targets[-self.n_days:]

        loss = self.mse(preds, targets)
        out_dict['loss'] = loss

        if not self.training:
            seed = series[:, :-self.n_days]
            seed[seed == 0] = 1
            # seed.shape == [batch_size, seq_len]

            seed_diff = seed[:, 1:] / seed[:, :-1]
            # seed_diff.shape == [batch_size, seq_len]

            X = seed_diff.unsqueeze(-1)
            # X.shape == [batch_size, seq_len, 1]

            self.decoder.reset_mems()
            X_out = self.decoder(X, seeding=False)
            X_out = self.fc(X_out).squeeze(-1)
            # X_out.shape == [batch_size, seq_len]

            pred_diff = X_out[:, -1]
            # pred_diff.shape == [batch_size]

            pred_i = seed[:, -1] * pred_diff

            pred_list = [pred_i]

            for i in range(self.n_days - 1):
                X = pred_diff.unsqueeze(1).unsqueeze(2)
                # X.shape == [batch_size, 1, 1]

                X_out = self.decoder(X, seeding=False)
                pred_diff = self.fc(X_out).squeeze(-1).squeeze(-1)
                # pred_diff.shape == [batch_size]

                pred_i = pred_list[-1] * pred_diff
                pred_list.append(pred_i)

            preds = torch.stack(pred_list, dim=1)
            targets = series[:, -self.n_days:]
            smape, _ = get_smape(targets, preds)
            # smape.shape == [batch_size]

            out_dict['smape'] = smape

        return out_dict

    @classmethod
    def get_params(cls, vocab: Vocabulary, params: Params) -> Dict[str, Any]:

        params_dict: Dict[str, Any] = {}

        params_dict['decoder'] = Decoder.from_params(
            vocab=vocab, params=params.pop('decoder'))
        params_dict['initializer'] = InitializerApplicator.from_params(
            params.pop('initializer', None))

        return params_dict
