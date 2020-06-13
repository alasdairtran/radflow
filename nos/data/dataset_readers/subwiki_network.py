import json
import logging
import os
import pickle
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


@DatasetReader.register('subwiki_network')
class SubWikivNetworkReader(DatasetReader):
    def __init__(self,
                 data_dir: str,
                 seed_word: str = 'programming',
                 fp16: bool = True,
                 lazy: bool = True) -> None:
        super().__init__(lazy)
        self.data_dir = data_dir
        self.dtype = np.float16 if fp16 else np.float32

        with open(f'data/wiki/subgraphs/{seed_word}.pkl', 'rb') as f:
            _, _, self.series = pickle.load(f)

        random.seed(1234)
        self.rs = np.random.RandomState(1234)

        keys = sorted(self.series.keys())
        self.rs.shuffle(keys)
        keys = list(keys)
        self.valid_keys = keys[:200]
        self.train_keys = keys[200:]

        # Check for reproducibility
        # assert sum(self.valid_keys) == 2939816

    @overrides
    def _read(self, split: str):
        if split not in ['train', 'valid', 'test']:
            raise ValueError(f'Unknown split: {split}')

        if split == 'train':
            keys = sorted(self.train_keys)
            while True:
                self.rs.shuffle(keys)
                for key in keys:
                    yield self.series_to_instance(key, split)

        elif split == 'valid':
            keys = sorted(self.valid_keys)
            for key in keys:
                yield self.series_to_instance(key, split)

        elif split == 'test':
            keys = sorted(self.series.keys())
            for key in keys:
                yield self.series_to_instance(key, split)

    def series_to_instance(self, key, split) -> Instance:
        fields = {
            'keys': MetadataField(key),
            'splits': MetadataField(split),
        }

        return Instance(fields)
