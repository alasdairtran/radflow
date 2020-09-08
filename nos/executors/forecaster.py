import pickle
from typing import Any

import numpy as np
from jina.executors.encoders import BaseEncoder


class RadflowForecaster(BaseEncoder):

    def __init__(self, greetings: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._greetings = greetings
        # self.model = kwargs['model']
        print('init', greetings)

    def encode(self, data: Any, *args, **kwargs):
        print('encode', data)

        self.logger.info('%s %s' % (self._greetings, data))
        return np.random.random([data.shape[0], 3])
