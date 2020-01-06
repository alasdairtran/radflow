import json
import logging
import os
import random
from typing import Dict, List

import numpy as np
import pandas as pd
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import MetadataField
from allennlp.data.instance import Instance
from overrides import overrides

from nos.utils import keystoint

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register('vevo_network')
class VevovNetworkReader(DatasetReader):
    def __init__(self,
                 data_dir: str,
                 fp16: bool = True,
                 evaluate_mode: bool = False,
                 lazy: bool = True) -> None:
        super().__init__(lazy)
        self.data_dir = data_dir
        self.dtype = np.float16 if fp16 else np.float32
        self.evaluate_mode = evaluate_mode

        series_path = os.path.join(data_dir, 'vevo_series.json')
        with open(series_path) as f:
            self.series = json.load(f, object_pairs_hook=keystoint)

        random.seed(1234)
        self.rs = np.random.RandomState(1234)

    @overrides
    def _read(self, split: str):
        if split not in ['train', 'valid', 'test']:
            raise ValueError(f'Unknown split: {split}')

        while True:
            keys = list(self.series.keys())
            self.rs.shuffle(keys)

            if split != 'train' and self.evaluate_mode:
                for key in keys:
                    yield self.series_to_instance(key)
                    return

            for key in keys:
                yield self.series_to_instance(key)

            if split != 'train':
                break

    def series_to_instance(self, key) -> Instance:
        fields = {
            'keys': MetadataField(key),
        }

        return Instance(fields)
