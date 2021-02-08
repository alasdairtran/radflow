import json
import logging
import os
import pickle
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List

import h5py
import numpy as np
import pandas as pd
import pymongo
from pymongo import MongoClient
from tqdm import tqdm

from radflow.utils import setup_logger

logger = setup_logger()


def relabel_networks():
    data_dir = 'data/vevo/raw'

    # Load persistent network
    logger.info('Loading persistent network.')
    network_path = os.path.join(data_dir, 'persistent_network.csv')
    network_df = pd.read_csv(network_path)

    # Get the edges
    network_df['source'] = network_df['Source']
    network_df['target'] = network_df['Target']

    logger.info('Constructing in-degree dict of persistent network.')
    persistent_indegrees: Dict[int, List[int]] = defaultdict(list)
    for _, row in tqdm(network_df.iterrows()):
        target = int(row['target'])
        source = int(row['source'])
        persistent_indegrees[target].append(source)

    logger.info('Loading time series.')
    full_series: Dict[int, List[int]] = {}
    path = os.path.join(data_dir, 'vevo_forecast_data_60k.tsv')
    embed_dict = {}
    with open(path) as f:
        for line in tqdm(f):
            embed, embed_name, ts_view, _ = line.rstrip().split('\t')
            key = int(embed)
            embed_dict[embed_name] = key
            full_series[key] = [int(x) for x in ts_view.split(',')]

    snapshots = defaultdict(dict)

    # Daily snapshots
    logger.info('Computing snapshots.')
    start = datetime(2018, 9, 1)
    for i in tqdm(range(63)):
        d = start + timedelta(days=i)
        filename = f'network_{d.year}-{d.month:02}-{d.day:02}.p'
        path = os.path.join(data_dir, 'network_pickle', filename)
        with open(path, 'rb') as f:
            obj = pickle.load(f)

        for key, values in obj.items():
            s = [v[0] for v in values]
            snapshots[i][key] = s

    # We store neighbours, each with a time mask
    masked_neighs = {}
    n_days = len(snapshots.keys())

    logger.info('Computing masked neigh dict')
    for day, snapshot in tqdm(snapshots.items()):
        for key, neighs in snapshot.items():
            if key not in masked_neighs:
                masked_neighs[key] = {}
            for neigh in neighs:
                if neigh not in masked_neighs[key]:
                    masked_neighs[key][neigh] = {
                        'id': neigh,
                        'mask': [True] * n_days,
                    }
                masked_neighs[key][neigh]['mask'][day] = False

    for k, v in masked_neighs.items():
        masked_neighs[k] = list(v.values())

    # Reshape so that we now store neighbours separately for each day
    logger.info('Reshaping masked neigh dict')
    neighs = {k: {} for k in masked_neighs}
    for t in tqdm(range(n_days)):
        for k, v in masked_neighs.items():
            k_neighs = [n['id'] for n in v if n['mask'][t] == 0]
            neighs[k][t] = k_neighs

    output_dir = 'data/vevo/processed/dynamic'
    os.makedirs(output_dir, exist_ok=True)
    logger.info('Saving processed dynamic network data')
    with open(os.path.join(output_dir, 'in_degrees.pkl'), 'wb') as f:
        pickle.dump(masked_neighs, f)
    with open(os.path.join(output_dir, 'series.pkl'), 'wb') as f:
        pickle.dump(full_series, f)
    with open(os.path.join(output_dir, 'neighs.pkl'), 'wb') as f:
        pickle.dump(neighs, f)

    # We also save static graph
    static_in_degrees = {k: [] for k in persistent_indegrees.keys()}
    for k in persistent_indegrees:
        for n in persistent_indegrees[k]:
            mask = [0] * n_days
            static_in_degrees[k].append({'id': n, 'mask': mask})

    neighs = {k: {} for k in persistent_indegrees.keys()}
    for t in tqdm(range(n_days)):
        for k, v in persistent_indegrees.items():
            neighs[k][t] = v

    output_dir = 'data/vevo/processed/static'
    os.makedirs(output_dir, exist_ok=True)
    logger.info('Saving processed static network data')
    with open(os.path.join(output_dir, 'in_degrees.pkl'), 'wb') as f:
        pickle.dump(static_in_degrees, f)
    with open(os.path.join(output_dir, 'series.pkl'), 'wb') as f:
        pickle.dump(full_series, f)
    with open(os.path.join(output_dir, 'neighs.pkl'), 'wb') as f:
        pickle.dump(neighs, f)


def populate_database(seed_word, collection):
    data_dir = 'data/vevo/processed'
    series_path = f'{data_dir}/{seed_word}/series.pkl'
    with open(series_path, 'rb') as f:
        series = pickle.load(f)

    in_degrees_path = f'{data_dir}/{seed_word}/in_degrees.pkl'
    with open(in_degrees_path, 'rb') as f:
        in_degrees = pickle.load(f)

    neighs_path = f'{data_dir}/{seed_word}/neighs.pkl'
    with open(neighs_path, 'rb') as f:
        neighs = pickle.load(f)

    docs = []

    logger.info('Constructing database entries')
    for k, s in tqdm(series.items()):
        n_days = len(s)
        neighs_list = []
        if k in neighs:
            for d in range(n_days):
                ns = [] if d not in neighs[k] else neighs[k][d]
                neighs_list.append(ns)

        masks = {}
        if k in in_degrees:
            for n in in_degrees[k]:
                masks[str(n['id'])] = list(map(bool, n['mask']))

        doc = {
            '_id': k,
            's': s,
            'e': neighs_list,
            'm': masks,
            'n': len(masks),
        }

        docs.append(doc)

    client = MongoClient(host='localhost', port=27017)
    db = client.vevo
    db[collection].create_index([
        ('n', pymongo.DESCENDING),
    ])

    logger.info('Inserting graph into database.')
    _ = db[collection].insert_many(docs)

    # Add metadata
    with open('data/vevo/raw/vevo_en_embeds_60k.txt') as f:
        for line in tqdm(f):
            parts = line.strip().split(',')
            i = int(parts[0])
            video_id = parts[1]
            title = parts[2]

            query = {'_id': i}
            update = {'$set': {'title': title, 'video_id': video_id}}
            db.graph.update_one(query, update)


def populate_hdf5(collection, name):
    data_path = f'data/vevo/{name}.hdf5'
    data_f = h5py.File(data_path, 'a')
    views = np.full((60740, 63), -1, dtype=np.int32)

    int32_dt = h5py.vlen_dtype(np.dtype('int32'))
    edges = data_f.create_dataset('edges', (60740, 63), int32_dt)

    bool_dt = h5py.vlen_dtype(np.dtype('bool'))
    masks = data_f.create_dataset('masks', (60740, 63), bool_dt)

    logger.info('Populating series in memory')
    client = MongoClient(host='localhost', port=27017)

    outdegrees = np.ones((60740, 63), dtype=np.int32)  # add self-loops
    key2pos = [{} for _ in range(60740)]
    for p in tqdm(client.vevo[collection].find({})):
        s = np.array(p['s'])
        views[p['_id']] = s

        mask = np.ones((len(p['m']), 63), dtype=np.bool_)
        for i, (k, v) in enumerate(p['m'].items()):
            m = np.array(v, dtype=np.bool_)
            mask[i] = m
            key2pos[p['_id']][int(k)] = i
            outdegrees[int(k)] += (~m).astype(np.int32)

        masks[p['_id']] = np.ascontiguousarray(mask.transpose())

    assert outdegrees.data.c_contiguous
    assert views.data.c_contiguous
    data_f.create_dataset('outdegrees', dtype=np.int32, data=outdegrees)
    data_f.create_dataset('views', dtype=np.int32, data=views)

    with open(f'data/vevo/{name}.key2pos.pkl', 'wb') as f:
        pickle.dump(key2pos, f)

    views[views == -1] = 0
    normalised_views = views / outdegrees
    for p in tqdm(client.vevo[collection].find({})):
        if not p['e']:
            continue

        max_count = max([len(es) for es in p['e']])
        edges_array = np.full((63, max_count), -1, dtype=np.int32)
        for day in range(63):
            day_edges = p['e'][day]
            sorted_edges = sorted(day_edges, key=lambda n: normalised_views[n, day],
                                  reverse=True)
            sorted_edges = np.array(sorted_edges, dtype=np.int32)
            edges_array[day, :len(sorted_edges)] = sorted_edges

        edges[p['_id']] = edges_array

    all_cursor = client.vevo[collection].find({}, projection=['_id'])
    all_ids = set(s['_id'] for s in all_cursor)

    node_cursor = client.vevo[collection].find(
        {'n': {'$gt': 0}}, projection=['_id'])
    connected_ids = set(s['_id'] for s in node_cursor)

    with open(f'data/vevo/{name}_all_nodes.pkl', 'wb') as f:
        pickle.dump(all_ids, f)

    with open(f'data/vevo/{name}_connected_nodes.pkl', 'wb') as f:
        pickle.dump(connected_ids, f)

    float16_dt = h5py.vlen_dtype(np.dtype('float16'))
    probs = data_f.create_dataset('probs', (60740, 63), float16_dt)
    edges = data_f['edges'][...]

    for k, edge in tqdm(enumerate(edges)):
        if len(edge[0]) == 0:
            continue

        key_probs = np.ones((63, len(edge[0])), dtype=np.float16)
        for d, ns in enumerate(edge):
            if len(ns) == 0 or ns[0] == -1:
                continue
            counts = np.array([normalised_views[n, d] for n in ns[ns != -1]])

            total = counts.sum()
            if total < 1e-6:
                continue

            prob = counts / total
            key_probs[d, :len(prob)] = np.array(prob.cumsum(), np.float16)
        probs[k] = np.ascontiguousarray(key_probs)


def main():
    logger.info('Relabeling networks')
    relabel_networks()

    logger.info('Populating dynamic database')
    populate_database('dynamic', 'graph')

    logger.info('Populating static database')
    populate_database('static', 'static')

    # Reading series from hdf5 is 20% faster than from mongo.
    logger.info('Populating dynamic HDF5')
    populate_hdf5('graph', 'vevo')

    logger.info('Populating static HDF5')
    populate_hdf5('static', 'vevo_static')


if __name__ == '__main__':
    main()
