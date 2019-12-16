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
        initializer(self)

    def forward(self, series) -> Dict[str, Any]:
        # Enable anomaly detection to find the operation that failed to compute
        # its gradient.
        # torch.autograd.set_detect_anomaly(True)

        # series.shape == [batch_size, seq_len]

        series[series == 0] = 1

        # Take the difference
        diff = series[:, 1:] / series[:, :-1]
        targets = diff[:, 1:]
        inputs = diff[:, :-1]

        X = inputs.unsqueeze(-1)
        # X.shape == [batch_size, seq_len, 1]

        self.decoder.reset_mems()
        X = self.decoder(X, seeding=False)
        X = self.fc(X)
        preds = X.squeeze(-1)

        loss = self.mse(preds, targets)
        batch_size = series.shape[0]

        out_dict = {
            'loss': loss,
            'sample_size': torch.tensor(batch_size).to(X.device),
        }

        return out_dict

    @classmethod
    def get_params(cls, vocab: Vocabulary, params: Params) -> Dict[str, Any]:

        params_dict: Dict[str, Any] = {}

        params_dict['decoder'] = Decoder.from_params(
            vocab=vocab, params=params.pop('decoder'))
        params_dict['initializer'] = InitializerApplicator.from_params(
            params.pop('initializer', None))

        return params_dict
