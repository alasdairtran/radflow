import json
import logging
import random

import numpy as np
import pandas as pd
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, MetadataField
from allennlp.data.instance import Instance
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register('taxi')
class TaxiReader(DatasetReader):
    def __init__(self,
                 series_path: str = 'data/taxi/sz_speed.csv',
                 train_p: float = 0.8,
                 seq_len: int = 12,
                 pre_len: int = 3,
                 lazy: bool = True) -> None:
        super().__init__(lazy)
        random.seed(1234)
        self.rs = np.random.RandomState(1234)
        series = pd.read_csv(series_path).to_numpy()
        # self.series.shape == [total_steps, n_roads]

        # We'll use the same normalization as in the original T-GCN paper
        self.scale = series.max()
        series = series / self.scale

        # The original paper did not have a validation test. Thus our
        # validation and test sets will be the same.
        total_steps = series.shape[0]
        train_steps = int(total_steps * train_p)
        train_series = series[:train_steps]
        test_series = series[train_steps:]

        sample_len = seq_len + pre_len
        X_train_list, y_train_list, X_test_list, y_test_list = [], [], [], []

        for i in range(train_steps - sample_len):
            sample = train_series[i:i+sample_len]
            X_train_list.append(sample[:seq_len])
            y_train_list.append(sample[seq_len:seq_len + pre_len])

        for i in range(len(test_series) - sample_len):
            sample = test_series[i:i+sample_len]
            X_test_list.append(sample[:seq_len])
            y_test_list.append(sample[seq_len:seq_len + pre_len])

        self.x_train = np.array(X_train_list)
        self.y_train = np.array(y_train_list)
        self.x_test = np.array(X_test_list)
        self.y_test = np.array(y_test_list)

    @overrides
    def _read(self, split: str):
        if split not in ['train', 'valid', 'test']:
            raise ValueError(f'Unknown split: {split}')

        if split == 'train':
            ids = list(range(len(self.x_train)))
            self.rs.shuffle(ids)
            for i in ids:
                yield self.series_to_instance(self.x_train[i], self.y_train[i], i, split)

        elif split in ['valid', 'test']:
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
