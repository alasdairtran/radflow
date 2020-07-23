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

from nos.utils import keystoint

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register('subwiki_network')
class SubWikivNetworkReader(DatasetReader):
    def __init__(self,
                 database: str = 'vevo',
                 collection: str = 'graph',
                 train_all: bool = False,
                 lazy: bool = True) -> None:
        super().__init__(lazy)
        random.seed(1234)
        self.rs = np.random.RandomState(1234)

        client = MongoClient(host='localhost', port=27017)
        db = client[database]
        all_cursor = db[collection].find({}, projection=['_id'])
        self.all_ids = sorted([s['_id'] for s in all_cursor])

        node_cursor = db[collection].find({}, projection=['_id'])
        self.node_ids = sorted([s['_id'] for s in node_cursor])

        self.train_all = train_all

    @overrides
    def _read(self, split: str):
        if split not in ['train', 'valid', 'test']:
            raise ValueError(f'Unknown split: {split}')

        if split == 'train':
            keys = self.all_ids if self.train_all else self.node_ids
            keys = sorted(keys)
            while True:
                self.rs.shuffle(keys)
                for key in keys:
                    yield self.series_to_instance(key, split)

        elif split in ['valid', 'test']:
            for key in self.node_ids:
                yield self.series_to_instance(key, split)

    def series_to_instance(self, key, split) -> Instance:
        fields = {
            'keys': MetadataField(key),
            'splits': MetadataField(split),
        }

        return Instance(fields)
