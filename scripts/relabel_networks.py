import json
import logging
import os
from collections import defaultdict
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def relabel_networks():
    data_dir = 'data'

    # Load persistent network
    logger.info('Loading persistent network.')
    network_path = os.path.join(data_dir, 'persistent_network.csv')
    network_df = pd.read_csv(network_path)
    target_ids = set(network_df['Target'])
    source_ids = set(network_df['Source'])
    node_ids = sorted(target_ids | source_ids)

    # Map original ID to new ID
    node_map = {int(k): int(i) for i, k in enumerate(node_ids)}

    # Get the edges
    network_df['source'] = network_df['Source'].replace(node_map)
    network_df['target'] = network_df['Target'].replace(node_map)

    logger.info('Remapping node IDs.')
    sources: Dict[int, List[int]] = defaultdict(list)
    for _, row in network_df.iterrows():
        target = int(row['target'])
        source = int(row['source'])
        sources[target].append(source)

    out_network_path = os.path.join(data_dir, 'persistent_network_2.csv')
    network_df[['source', 'target']].to_csv(out_network_path, index=False)

    out_adj_list_path = os.path.join(data_dir, 'adjacency_list.json')
    with open(out_adj_list_path, 'w') as f:
        json.dump(sources, f)

    logger.info('Loading time series.')
    full_series: Dict[int, List[int]] = {}
    series: Dict[int, List[int]] = {}
    path = os.path.join(data_dir, 'vevo_forecast_data_60k.tsv')
    with open(path) as f:
        for line in f:
            embed, _, ts_view, _ = line.rstrip().split('\t')
            if int(embed) in node_ids:
                key = node_map[int(embed)]
                full_series[key] = [int(x) for x in ts_view.split(',')]
            if int(embed) in target_ids:
                key = node_map[int(embed)]
                series[key] = [int(x) for x in ts_view.split(',')]
    assert len(series) == 13710

    out_series_path = os.path.join(data_dir, 'vevo_series.json')
    with open(out_series_path, 'w') as f:
        json.dump(series, f)

    out_full_series_path = os.path.join(data_dir, 'vevo_full_series.json')
    with open(out_full_series_path, 'w') as f:
        json.dump(full_series, f)


def main():
    relabel_networks()


if __name__ == '__main__':
    main()
