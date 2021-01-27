import json
import os
import pickle
from collections import defaultdict

import h5py
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from nos.utils import setup_logger

logger = setup_logger()


def main():
    # Two versions:
    #   - streams_neighbours.graphml: edges are only whether or not you
    #     can get to a stream gauge without going through another stream gauge
    #   - streams-2.graphml original graph that also includes edge weights (length_)

    path = './data/streams/raw/streams-2.graphml'
    logger.info(f'Reading graph info from {path}')
    G = nx.read_graphml(path)

    with open('./data/streams/raw/stream_series_discharge.json') as f:
        json_obj = json.load(f)

    streams_f_path = './data/streams/streams.h5'
    logger.info(f'Creating new h5 file at {streams_f_path}')
    streams_f = h5py.File(streams_f_path, 'a')

    with open('./data/streams/raw/node_stations.json') as f:
        gkey2okey = json.load(f)

    csv_path = './data/streams/streams.csv'
    if not os.path.exists(csv_path):
        logger.info('Converting time series from JSON to DataFrame.')
        stream_names = list(json_obj.keys())
        df = pd.DataFrame()
        for k in tqdm(stream_names):
            # Ignore streams with no measurements
            if len(json_obj[k]['Value']) == 0:
                continue
            s = pd.Series(json_obj[k]['Value'], name=k)
            s.index = pd.to_datetime(s.index, unit='ms').floor('d')
            df = df.join(s, how='outer')
        df.to_csv(csv_path)

    # Load in the dataset
    logger.info(f'Loading in stream csv from {csv_path}')
    df = pd.read_csv(csv_path, index_col=0)
    df.index = pd.to_datetime(df.index)

    # Monthly aggregation. Treat NaN as no obs.
    df = df.resample('M').mean()
    df = df.apply(np.cbrt)

    # Either ignore streams with missing observation or replace them with -1
    df = df.dropna(axis=1, how='any')
    # df = df.fillna(-1)

    okey2idx = {key: i for i, key in enumerate(list(df.columns))}

    # Store the series
    series_array = df.to_numpy().transpose()
    series_array = np.ascontiguousarray(series_array)
    assert series_array.data.c_contiguous
    n_nodes, n_steps = series_array.shape
    streams_f.create_dataset('views', data=series_array)
    # series_array.shape == [475, 493]

    int32_dt = h5py.vlen_dtype(np.dtype('int32'))
    edges = streams_f.create_dataset('edges', (n_nodes, n_steps), int32_dt)

    bool_dt = h5py.vlen_dtype(np.dtype('bool'))
    masks = streams_f.create_dataset('masks', (n_nodes, n_steps), bool_dt)

    outdegrees = np.ones((n_nodes, n_steps), dtype=np.int32)  # add self-loops
    key2pos = [{} for _ in range(n_nodes)]

    logger.info(f'Constructing adjacency lists.')
    neigh_dict = defaultdict(list)
    for u, v, a in tqdm(G.edges(data=True)):
        if u not in gkey2okey or v not in gkey2okey:
            continue
        u, v = gkey2okey[u], gkey2okey[v]
        # a['length_'] is the edge weight
        # Ignore edges where one node has missing measurements
        if u not in okey2idx or v not in okey2idx:
            continue

        u_idx = okey2idx[u]
        v_idx = okey2idx[v]

        # The graph is undirected. We need to include both directions.
        neigh_dict[u_idx].append(v_idx)
        neigh_dict[v_idx].append(u_idx)

    logger.info(f'Constructing masks and outdegrees.')
    for k, neighs in neigh_dict.items():
        n_neighs = len(neighs)
        masks[k] = np.ones((n_steps, n_neighs), dtype=np.bool_)

        for i, n in enumerate(neighs):
            key2pos[k][n] = i

        outdegrees[k] += n_neighs

    assert outdegrees.data.c_contiguous
    streams_f.create_dataset('outdegrees', dtype=np.int32, data=outdegrees)

    with open(f'./data/streams/key2pos.pkl', 'wb') as f:
        pickle.dump(key2pos, f)

    logger.info(f'Sorting edges.')
    normalised_series = series_array / outdegrees
    for k, neighs in neigh_dict.items():
        edges_array = np.full((n_steps, len(neighs)), -1, dtype=np.int32)
        for day in range(n_steps):
            sorted_edges = sorted(neighs, key=lambda n: normalised_series[n, day],
                                  reverse=True)
            sorted_edges = np.array(sorted_edges, dtype=np.int32)
            edges_array[day, :len(sorted_edges)] = sorted_edges
        edges[k] = edges_array


if __name__ == '__main__':
    main()
