import json
import logging
import os
import pickle
import random
from collections import Counter
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
                 use_edge: bool = False,
                 eval_edge: bool = False,
                 batch_size: int = 64,
                 lazy: bool = True) -> None:
        super().__init__(lazy)
        self.data_dir = data_dir
        self.dtype = np.float16 if fp16 else np.float32

        random.seed(1234)
        self.rs = np.random.RandomState(1234)

        in_degrees_path = f'{data_dir}/{seed_word}/in_degrees.pkl'
        logger.info(f'Loading {in_degrees_path} into dataset reader')
        with open(in_degrees_path, 'rb') as f:
            in_degrees = pickle.load(f)

        series_path = f'{data_dir}/{seed_word}/series.pkl'
        logger.info(f'Loading {series_path} into dataset reader')
        with open(series_path, 'rb') as f:
            series = pickle.load(f)

        self.connected_keys = sorted(set(in_degrees.keys()))
        self.all_keys = sorted(set(series.keys()))
        self.keys = self.connected_keys if use_edge else self.all_keys
        self.rs.shuffle(self.keys)
        self.batch_size = batch_size
        self.in_degrees = in_degrees
        self.eval_edge = eval_edge

    @overrides
    def _read(self, split: str):
        if split not in ['train', 'valid', 'test']:
            raise ValueError(f'Unknown split: {split}')

        if split == 'train':
            keys = sorted(self.keys)
            while True:
                self.rs.shuffle(keys)
                for key in keys:
                    yield self.series_to_instance(key, split)

        elif split in ['valid', 'test']:
            keys = sorted(
                self.connected_keys) if self.eval_edge else sorted(self.keys)
            for key in keys:
                yield self.series_to_instance(key, split)

    def series_to_instance(self, key, split) -> Instance:
        fields = {
            'keys': MetadataField(key),
            'splits': MetadataField(split),
        }

        return Instance(fields)
