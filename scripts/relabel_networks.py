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
import redis
import tiledb
from pymongo import MongoClient
from tqdm import tqdm

from nos.utils import keystoint

logger = logging.getLogger(__name__)


def relabel_networks():
    data_dir = 'data/vevo/raw'

    # Load persistent network
    logger.info('Loading persistent network.')
    network_path = os.path.join(data_dir, 'persistent_network.csv')
    network_df = pd.read_csv(network_path)
    target_ids = set(network_df['Target'])
    source_ids = set(network_df['Source'])
    # node_ids = sorted(target_ids | source_ids)

    # Map original ID to new ID
    # node_map = {int(k): int(i) for i, k in enumerate(node_ids)}

    # Get the edges
    network_df['source'] = network_df['Source']  # .replace(node_map)
    network_df['target'] = network_df['Target']  # .replace(node_map)

    logger.info('Remapping node IDs.')
    sources: Dict[int, List[int]] = defaultdict(list)
    for _, row in tqdm(network_df.iterrows()):
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
    embed_dict = {}
    with open(path) as f:
        for line in tqdm(f):
            embed, embed_name, ts_view, _ = line.rstrip().split('\t')
            key = int(embed)
            embed_dict[embed_name] = key
            full_series[key] = [int(x) for x in ts_view.split(',')]
            if int(embed) in target_ids:
                key = int(embed)
                series[key] = [int(x) for x in ts_view.split(',')]
    assert len(series) == 13710

    out_series_path = os.path.join(data_dir, 'vevo_series.json')
    with open(out_series_path, 'w') as f:
        json.dump(series, f)

    out_full_series_path = os.path.join(data_dir, 'vevo_full_series.json')
    with open(out_full_series_path, 'w') as f:
        json.dump(full_series, f)

    snapshots = defaultdict(dict)

    # TODO: COMMENTED OUT FOR FASTER LOADING. UNCOMMENT THIS WHEN CODE IS PUBLISHED
    # Daily snapshots
    # start = datetime(2018, 9, 1)
    # for i in tqdm(range(63)):
    #     d = start + timedelta(days=i)
    #     filename = f'network_{d.year}-{d.month:02}-{d.day:02}.p'
    #     path = os.path.join(data_dir, 'network_pickle', filename)
    #     with open(path, 'rb') as f:
    #         obj = pickle.load(f)

    #     for key, values in obj.items():
    #         values = sorted(values, key=lambda x: x[2], reverse=True)
    #         s = [v[0] for v in values]
    #         snapshots[i][key] = s

    # snapshot_path = os.path.join(data_dir, 'snapshots.json')
    # with open(snapshot_path, 'w') as f:
    #     json.dump(snapshots, f)

    new_in_degrees = {}
    for key in sources.keys():
        new_in_degrees[key] = {}

    n_days = len(snapshots.keys())

    for day, snapshot in tqdm(snapshots.items()):
        for key, neighs in snapshot.items():
            if key not in new_in_degrees:
                continue
            for neigh in neighs:
                if neigh not in new_in_degrees[key]:
                    new_in_degrees[key][neigh] = {
                        'id': neigh,
                        'mask': [True] * n_days,
                    }
                new_in_degrees[key][neigh]['mask'][day] = False

    for k, v in new_in_degrees.items():
        new_in_degrees[k] = sorted(v.values(), key=lambda x: sum(
            series[x['id']][:7]), reverse=True)

    neighs = {k: {} for k in new_in_degrees.keys()}
    max_days = len(next(iter(series.values())))
    for t in tqdm(range(max_days)):
        for k, v in new_in_degrees.items():
            k_neighs = [n['id'] for n in v if n['mask'][t] == 0]
            k_views = [series[n['id']][t]
                       for n in v if n['mask'][t] == 0]
            k_neighs = [x for _, x in sorted(
                zip(k_views, k_neighs), reverse=True)]
            neighs[k][t] = k_neighs

    output_dir = 'data/wiki/subgraphs/vevo'
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'in_degrees.pkl'), 'wb') as f:
        pickle.dump(new_in_degrees, f)
    with open(os.path.join(output_dir, 'out_degrees.pkl'), 'wb') as f:
        pickle.dump({}, f)
    with open(os.path.join(output_dir, 'series.pkl'), 'wb') as f:
        pickle.dump(series, f)
    with open(os.path.join(output_dir, 'neighs.pkl'), 'wb') as f:
        pickle.dump(neighs, f)

    tags = ['acoustic', 'alternative', 'audio', 'blue', 'classical', 'country',
            'cover', 'dance', 'electronic', 'gospel', 'guitar', 'hd', 'hip hop',
            'holiday', 'indie', 'instrumental', 'jazz', 'karaoke', 'live',
            'lyrics', 'metal', 'musical', 'official', 'piano', 'pop', 'r&b',
            'rap', 'remix', 'rock', 'single']

    path = 'data/vevo_en_videos_60k.json'
    vevo_tags = {}
    with open(path) as f:
        for line in tqdm(f):
            o = json.loads(line)
            key = embed_dict[o['id']]
            cleaned_tags = set()
            if 'tags' in o['snippet']:
                v_tags = o['snippet']['tags']
                for t in v_tags:
                    t = t.lower().replace('-', ' ')
                    for tag in tags:
                        if tag in t:
                            cleaned_tags.add(tag)
            cleaned_tags = sorted(cleaned_tags)
            vevo_tags[key] = cleaned_tags

    vevo_tag_path = os.path.join(data_dir, 'vevo_tags.json')
    with open(vevo_tag_path, 'w') as f:
        json.dump(vevo_tags, f)

    # We also save static graph
    static_in_degrees = {k: [] for k in sources.keys()}
    for k in static_in_degrees:
        for n in sources[k]:
            mask = [0] * n_days
            static_in_degrees[k].append({'id': n, 'mask': mask})

    neighs = {k: {} for k in sources.keys()}
    for t in tqdm(range(n_days)):
        for k, v in sources.items():
            k_views = [series[n][t] for n in v]
            v = [x for _, x in sorted(zip(k_views, v), reverse=True)]
            neighs[k][t] = v

    output_dir = 'data/wiki/subgraphs/vevo_static'
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'in_degrees.pkl'), 'wb') as f:
        pickle.dump(static_in_degrees, f)
    with open(os.path.join(output_dir, 'out_degrees.pkl'), 'wb') as f:
        pickle.dump({}, f)
    with open(os.path.join(output_dir, 'series.pkl'), 'wb') as f:
        pickle.dump(series, f)
    with open(os.path.join(output_dir, 'neighs.pkl'), 'wb') as f:
        pickle.dump(neighs, f)


def create_traffic_tile(traffic_path, n_nodes, train_length, start, end):
    if os.path.exists(traffic_path):
        return

    os.makedirs(traffic_path, exist_ok=True)

    logger.info(f'Creating traffic tile at {traffic_path}')
    n_steps = (end - start).days + 1
    logger.info(f'No of days: {n_steps}')

    dom = tiledb.Domain(tiledb.Dim(name='i',
                                   domain=(0, n_nodes - 1),
                                   tile=1,
                                   dtype=np.uint32),
                        tiledb.Dim(name="t",
                                   domain=(0, n_steps - 1),
                                   tile=train_length,
                                   dtype=np.uint32))

    # The array will be dense with a single attribute "v" so each (i,j) cell can store an integer.
    schema = tiledb.ArraySchema(domain=dom, sparse=False,
                                attrs=[tiledb.Attr(name="v", dtype=np.uint32)])

    # Create the (empty) array on disk.
    # Empty cells are represented by 4294967295 for uint32 (largest number  2^32 − 1)
    tiledb.DenseArray.create(traffic_path, schema)


def populate_database(seed_word, collection):
    data_dir = 'data/wiki/subgraphs'
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
    result = db[collection].insert_many(docs)


def populate_tiledb():
    traffic_path = 'data/series/vevo'
    start = datetime(2018, 9, 1)
    end = datetime(2018, 11, 2)  # includes endpoint
    create_traffic_tile(traffic_path, 60740, 63, start, end)

    views = np.full((60740, 63), np.iinfo(np.uint32).max, dtype=np.uint32)

    logger.info('Populating series in memory')
    client = MongoClient(host='localhost', port=27017)
    for p in tqdm(client.vevo.graph.find({}, projection=['s'])):
        s = np.array(p['s'])
        s[s == -1] = np.iinfo(np.uint32).max
        views[p['_id']] = s

    logger.info(f'Writing series matrix to {traffic_path}')
    with tiledb.DenseArray(traffic_path, mode='w') as A:
        A[:] = views


def populate_hdf5(collection, name):
    data_path = f'data/vevo/{name}.hdf5'
    f = h5py.File(data_path, 'a')
    views = np.full((60740, 63), -1, dtype=np.int32)
    edges = f.create_dataset('edges', (60740, 63, 10), np.int32, fillvalue=-1)

    dt = h5py.vlen_dtype(np.dtype('bool'))
    masks = f.create_dataset('masks', (60740,), dt)

    logger.info('Populating series in memory')
    client = MongoClient(host='localhost', port=27017)
    key2pos = [{} for _ in range(60740)]
    for p in tqdm(client.vevo[collection].find({})):
        s = np.array(p['s'])
        views[p['_id']] = s

        for d, day_ns in enumerate(p['e']):
            day_ns = day_ns[:10]
            edges[p['_id'], d, :len(day_ns)] = day_ns

        mask = np.ones((len(p['m']), 63), dtype=np.bool_)
        for i, (k, v) in enumerate(p['m'].items()):
            mask[i] = np.array(v, dtype=np.bool_)
            key2pos[p['_id']][int(k)] = i
        masks[p['_id']] = mask.reshape(-1)

    hdf5_views = f.create_dataset("views", (60740, 63), dtype='int32')
    hdf5_views[:] = views

    with open(f'data/vevo/{name}.key2pos.pkl', 'wb') as f:
        pickle.dump(key2pos, f)

    all_cursor = client.vevo[collection].find({}, projection=['_id'])
    all_ids = set(s['_id'] for s in all_cursor)

    node_cursor = client.vevo[collection].find(
        {'n': {'$gt': 0}}, projection=['_id'])
    connected_ids = set(s['_id'] for s in node_cursor)

    with open(f'data/vevo/vevo_all_nodes.pkl', 'wb') as f:
        pickle.dump(all_ids, f)

    with open(f'data/vevo/vevo_connected_nodes.pkl', 'wb') as f:
        pickle.dump(connected_ids, f)


def populate_redis():
    r = redis.Redis(host='localhost', port=6379, db=0)
    client = MongoClient(host='localhost', port=27017)

    # r.flushall()

    logger.info('Populating dynamic graph to redis')
    cursor = client.vevo.graph.find({}, {'s': False})
    for p in tqdm(cursor):
        i = p['_id']
        load = pickle.dumps({
            'e': [np.array(ns, dtype=np.uint32) for ns in p['e']],
            'm': {int(k): np.array(v) for k, v in p['m'].items()},
        })
        r.set(f'vevo:{i}', load)

    logger.info('Populating static graph to redis')
    cursor = client.vevo.static.find({}, {'s': False})
    for p in tqdm(cursor):
        i = p['_id']
        load = pickle.dumps({
            'e': [np.array(ns, dtype=np.uint32) for ns in p['e']],
            'm': {int(k): np.array(v) for k, v in p['m'].items()},
        })
        r.set(f'vevo.static:{i}', load)


def pickle_graph():
    client = MongoClient(host='localhost', port=27017)
    os.makedirs('data/graphs', exist_ok=True)

    cursor = client.vevo.graph.find({}, {'s': False})
    graph = {}
    for p in tqdm(cursor):
        graph[p['_id']] = {
            'e': [np.array(ns, dtype=np.uint32) for ns in p['e']],
            'm': {int(k): np.array(v) for k, v in p['m'].items()},
        }

    with open('data/graphs/vevo.pkl', 'wb') as f:
        pickle.dump(graph, f)

    cursor = client.vevo.static.find({}, {'s': False})
    graph = {}
    for p in tqdm(cursor):
        graph[p['_id']] = {
            'e': [np.array(ns, dtype=np.uint32) for ns in p['e']],
            'm': {int(k): np.array(v) for k, v in p['m'].items()},
        }

    with open('data/graphs/vevo_static.pkl', 'wb') as f:
        pickle.dump(graph, f)


def main():
    relabel_networks()
    populate_database('vevo', 'graph')
    populate_database('vevo_static', 'static')

    # We leave this here, but TileDB random read speed is still slower
    # than mongo.
    populate_tiledb()

    # Reading series from hdf5 is 20% than from mongo.
    populate_hdf5('graph', 'vevo')
    populate_hdf5('static', 'vevo_static')

    # Reading graph from redis is much faster than mongo. Pickle is twice
    # as fast as JSON.
    populate_redis()

    # In-memory solution is still twice as fast as than redis + pickle.
    pickle_graph()


if __name__ == '__main__':
    main()
