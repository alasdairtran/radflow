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
from allennlp.data.fields import ArrayField, MetadataField
from allennlp.data.instance import Instance
from overrides import overrides
from pymongo import MongoClient

from radflow.utils import keystoint

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register('taxi_2')
class Taxi2Reader(DatasetReader):
    def __init__(self,
                 series_path: str = 'data/taxi/sz_speed.csv',
                 seq_len: int = 72,
                 pre_len: int = 12,
                 scale: bool = True,
                 lazy: bool = True) -> None:
        super().__init__(lazy)
        random.seed(1234)
        self.rs = np.random.RandomState(1234)
        series = pd.read_csv(series_path).to_numpy()
        # self.series.shape == [total_steps, n_roads]

        # We can the same normalization as in the original T-GCN paper
        if scale:
            self.scale = series.max()
            series = series / self.scale
        else:
            self.scale = 1

        sample_len = seq_len + pre_len
        train_end = -(pre_len * 2)
        train_series = series[:train_end]

        valid_start = -(seq_len + pre_len * 2)
        valid_end = -pre_len
        valid_series = series[valid_start:valid_end]

        test_start = -(seq_len + pre_len)
        test_series = series[test_start:]

        X_train_list, y_train_list = [], []
        X_valid_list, y_valid_list = [], []
        X_test_list, y_test_list = [], []

        for i in range(len(train_series) - sample_len):
            sample = train_series[i:i+sample_len]
            X_train_list.append(sample[:seq_len])
            y_train_list.append(sample[seq_len:seq_len + pre_len])

        X_valid_list.append(valid_series[:seq_len])
        y_valid_list.append(valid_series[seq_len:])

        X_test_list.append(test_series[:seq_len])
        y_test_list.append(test_series[seq_len:])

        self.x_train = np.array(X_train_list)
        self.y_train = np.array(y_train_list)
        self.x_valid = np.array(X_valid_list)
        self.y_valid = np.array(y_valid_list)
        self.x_test = np.array(X_test_list)
        self.y_test = np.array(y_test_list)

    @overrides
    def _read(self, split: str):
        if split not in ['train', 'valid', 'test']:
            raise ValueError(f'Unknown split: {split}')

        if split == 'train':
            idx = self.rs.permutation(len(self.x_train))
            x_train = self.x_train[idx]
            y_train = self.y_train[idx]
            for i, key in enumerate(idx):
                yield self.series_to_instance(x_train[i], y_train[i], key, split)

        elif split == 'valid':
            for i in range(len(self.x_valid)):
                yield self.series_to_instance(self.x_valid[i], self.y_valid[i], i, split)

        elif split == 'test':
            for i in range(len(self.x_test)):
                yield self.series_to_instance(self.x_test[i], self.y_test[i], i, split)

    def series_to_instance(self, x, y, key, split) -> Instance:
        fields = {
            'x': ArrayField(x),
            'y': ArrayField(y),
            'scale': MetadataField(self.scale),
            'keys': MetadataField(key),
            'splits': MetadataField(split),
        }

        return Instance(fields)
