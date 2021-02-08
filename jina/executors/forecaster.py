import os
import pickle
from typing import Any

import numpy as np
import torch
from allennlp.common.util import prepare_environment
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.models.archival import load_archive
from jina.executors.encoders import BaseNumericEncoder

from radflow.commands.train import yaml_to_params


class RadflowForecaster(BaseNumericEncoder):

    def __init__(self, greetings: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._greetings = greetings
        self.logger.info('%s Initializing forecaster' % (self._greetings))
        config = yaml_to_params(kwargs['config_path'], kwargs['overrides'])
        prepare_environment(config)
        config_dir = os.path.dirname(kwargs['config_path'])
        serialization_dir = os.path.join(config_dir, 'serialization')

        os.makedirs(serialization_dir, exist_ok=True)
        vocab = Vocabulary.from_params(config.pop('vocabulary'))

        model = Model.from_params(vocab=vocab, params=config.pop('model'))
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{kwargs['device']}")
        else:
            device = torch.device('cpu')

        self.logger.info(f"Loading best model weights from "
                         f"{kwargs['model_path']}")
        best_model_state = torch.load(kwargs['model_path'], device)
        model.load_state_dict(best_model_state)

        # We want to keep the dropout to generate confidence intervals
        # model = model.eval()

        self.model = model
        self.forecast_len = 28

    def encode(self, data: Any, *args, **kwargs):

        p = next(self.model.parameters())

        series = pickle.loads(data)
        self.logger.info(f"Received data: {series[:4].tolist()}...")
        # series.shape == [seq_len]

        series = torch.from_numpy(series).to(p.device)
        series = series.unsqueeze(0)
        series = torch.log1p(series)
        # series.shape == [1, seq_len]

        # Predict 28 days
        preds = p.new_zeros(1, self.forecast_len)

        for i in range(self.forecast_len):
            _, pred, _ = self.model._forward_full(series)
            pred = pred[:, -1]
            preds[:, i] = pred[:, -1]
            series = torch.cat([series, pred.unsqueeze(1)], dim=1)

        preds = torch.exp(preds)
        preds = preds.cpu().numpy()

        self.logger.info(f"Returning forecasts: {preds[:4]}...")

        return preds
