import json
import logging
import os
import random
from typing import Dict, List

import numpy as np
import pandas as pd
import pymongo
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import MetadataField
from allennlp.data.instance import Instance
from overrides import overrides
from pymongo import MongoClient

from nos.utils import keystoint

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register('wiki_network')
class WikiNetworkReader(DatasetReader):
    def __init__(self,
                 data_dir: str,
                 db_filter: Dict,
                 mongo_host: str = 'localhost',
                 mongo_port: int = 27017,
                 fp16: bool = True,
                 lazy: bool = True) -> None:
        super().__init__(lazy)
        self.data_dir = data_dir
        self.dtype = np.float16 if fp16 else np.float32
        random.seed(1234)
        self.rs = np.random.RandomState(1234)

        client = MongoClient(host=mongo_host, port=mongo_port)
        self.db = client.wiki
        self.db_filter = db_filter

    @overrides
    def _read(self, split: str):
        if split not in ['train', 'valid', 'test']:
            raise ValueError(f'Unknown split: {split}')

        sample_cursor = self.db.series.find(self.db_filter, projection=['_id'])
        sample_cursor = sample_cursor.sort('_id', pymongo.ASCENDING)
        ids = np.array([article['_id'] for article in sample_cursor])
        sample_cursor.close()

        if split == 'train':
            while True:
                self.rs.shuffle(ids)
                for key in ids:
                    yield self.series_to_instance(key, split)
        else:
            for key in ids:
                yield self.series_to_instance(key, split)

    def series_to_instance(self, key, split) -> Instance:
        fields = {
            'keys': MetadataField(key),
            'splits': MetadataField(split),
        }

        return Instance(fields)
