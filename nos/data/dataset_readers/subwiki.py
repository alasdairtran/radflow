import json
import logging
import os
import pickle
import random
from typing import Dict, List

import numpy as np
import pandas as pd
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, MetadataField
from allennlp.data.instance import Instance
from overrides import overrides

from nos.utils import keystoint

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register('subwiki')
class SubWikiReader(DatasetReader):
    def __init__(self,
                 data_dir: str,
                 seed_word: str = 'Programming languages',
                 fp16: bool = True,
                 lazy: bool = True) -> None:
        super().__init__(lazy)
        self.data_dir = data_dir
        self.dtype = np.float16 if fp16 else np.float32

        with open(f'data/wiki/subgraphs/{seed_word}.series.pkl', 'rb') as f:
            self.series = pickle.load(f)

        random.seed(1234)
        self.rs = np.random.RandomState(1234)

    @overrides
    def _read(self, split: str):
        if split not in ['train', 'valid', 'test']:
            raise ValueError(f'Unknown split: {split}')

        if split == 'train':
            while True:
                keys = list(self.series.keys())
                self.rs.shuffle(keys)
                for key in keys:
                    series = np.array(self.series[key])
                    series = series[:-7]
                    yield self.series_to_instance(series)

        else:
            keys = sorted(self.series.keys())
            for key in keys:
                series = np.array(self.series[key])
                yield self.series_to_instance(series, key)

    def series_to_instance(self, series, key) -> Instance:
        fields = {
            'series': ArrayField(series, dtype=self.dtype),
            'keys': MetadataField(key),
        }

        return Instance(fields)
