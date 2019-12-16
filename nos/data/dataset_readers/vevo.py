import logging
import os
import random
from typing import Dict, List

import numpy as np
import pandas as pd
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField
from allennlp.data.instance import Instance
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register('vevo')
class VevovReader(DatasetReader):
    def __init__(self,
                 data_dir: str,
                 fp16: bool = True,
                 lazy: bool = True) -> None:
        super().__init__(lazy)
        self.data_dir = data_dir
        self.dtype = np.float16 if fp16 else np.float32

        # Load persistent network
        network_path = os.path.join(data_dir, 'persistent_network.csv')
        network_df = pd.read_csv(network_path)
        target_ids = set(network_df['Target'])

        self.series: Dict[int, List[int]] = {}
        path = os.path.join(data_dir, 'vevo_forecast_data_60k.tsv')
        with open(path) as f:
            for line in f:
                embed, _, ts_view, _ = line.rstrip().split('\t')
                if int(embed) not in target_ids:
                    continue
                self.series[int(embed)] = [int(x) for x in ts_view.split(',')]
        assert len(self.series) == 13710

        random.seed(1234)
        self.rs = np.random.RandomState(1234)

    @overrides
    def _read(self, split: str):
        if split not in ['train', 'valid', 'test']:
            raise ValueError(f'Unknown split: {split}')

        while True:
            keys = list(self.series.keys())
            self.rs.shuffle(keys)
            for key in keys:
                series = np.array(self.series[key])

                if split == 'train':
                    series = series[:-7]

                yield self.series_to_instance(series)

            if split != 'train':
                break

    def series_to_instance(self, series) -> Instance:
        fields = {
            'series': ArrayField(series, dtype=self.dtype),
        }

        return Instance(fields)
