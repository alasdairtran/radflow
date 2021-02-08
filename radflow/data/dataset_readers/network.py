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
from pymongo import MongoClient

from radflow.utils import keystoint

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register('network')
class NetworkReader(DatasetReader):
    def __init__(self,
                 n_nodes: int = None,
                 train_path: str = 'data/vevo_all_nodes.pkl',
                 test_path: str = 'data/vevo_connected_nodes.pkl',
                 lazy: bool = True) -> None:
        super().__init__(lazy)
        random.seed(1234)
        self.rs = np.random.RandomState(1234)

        if n_nodes is not None:
            self.train_ids = list(range(n_nodes))
            self.test_ids = list(range(n_nodes))

        else:
            with open(train_path, 'rb') as f:
                self.train_ids = sorted(pickle.load(f))

            with open(test_path, 'rb') as f:
                self.test_ids = sorted(pickle.load(f))

    @overrides
    def _read(self, split: str):
        if split not in ['train', 'valid', 'test']:
            raise ValueError(f'Unknown split: {split}')

        if split == 'train':
            keys = sorted(self.train_ids)
            while True:
                self.rs.shuffle(keys)
                for key in keys:
                    yield self.series_to_instance(key, split)

        elif split in ['valid', 'test']:
            for key in sorted(self.test_ids):
                yield self.series_to_instance(key, split)

    def series_to_instance(self, key, split) -> Instance:
        fields = {
            'keys': MetadataField(key),
            'splits': MetadataField(split),
        }

        return Instance(fields)
