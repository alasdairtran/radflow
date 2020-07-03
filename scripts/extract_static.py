"""Annotate Good News with parts of speech.

Usage:
    get_traffic.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -m --mongo HOST     MongoDB host [default: localhost].
    -r --redis HOST     Redis host [default: localhost].
"""

import bz2
import fileinput
import html
import os
import pickle
import random
import re
import sqlite3
import time
from calendar import monthrange
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from glob import glob

import numpy as np
import pandas as pd
import ptvsd
import pymongo
import pytz
import requests
import scipy.sparse as ss
from bz2file import BZ2File
from docopt import docopt
from joblib import Parallel, delayed
from pymongo import MongoClient
from schema import And, Or, Schema, Use
from scipy import sparse as ss
from tqdm import tqdm

import redis
import tiledb
from nos.utils import setup_logger

logger = setup_logger()


def read_data_file_as_coo_matrix(filename='edges.txt'):
    # https://stackoverflow.com/a/38734771
    "Read data file and return sparse matrix in coordinate format."
    data = pd.read_csv(filename, sep=' ', header=None, dtype=np.uint32)
    rows = data[0]  # Not a copy, just a reference.
    cols = data[1]
    ones = np.ones(len(rows), np.uint32)
    matrix = ss.coo_matrix((ones, (rows, cols)))
    return matrix


def extract_static_graph(mongo_host, redis_host, out_path):
    client = MongoClient(host=mongo_host, port=27017)
    db = client.wiki

    r = redis.Redis(host=redis_host, port=6379, db=0)

    # Takes 10.5h to extract 17,035,758 nodes and 185,809,379 edges on 1 Jan 2019
    # Takes 7.5h to extract edges on 1 Dec 2011.
    origin = datetime(2015, 7, 1)
    with open(out_path, 'a') as f:
        for p in tqdm(db.pages.find({})):
            source = p['i']
            for link in p['links']:
                for t in link['t']:
                    if t['s'] < origin and ('e' not in t or t['e'] > origin):
                        target = r.get(link['n'])
                        if target:
                            f.write(f"{source} {int(target)}\n")
                            break


def validate(args):
    """Validate command line arguments."""
    args = {k.lstrip('-').lower().replace('-', '_'): v
            for k, v in args.items()}
    schema = Schema({
        'ptvsd': Or(None, And(Use(int), lambda port: 1 <= port <= 65535)),
        'mongo': Or(None, str),
        'redis': Or(None, str),
    })
    args = schema.validate(args)
    return args


def get_seeds_from_title(title, db, csr_matric):
    seeds = set()
    p = db.pages.find_one({'title': title})
    seeds.add(p['i'])
    seeds = seeds | set(list(csr_matric.getrow(p['i']).nonzero()[1]))
    return sorted(seeds)


def get_seeds_from_cat(seed_word, db):
    seeds = []
    for p in db.pages.find({'cats': seed_word}, projection=['i']):
        seeds.append(p['i'])
    return sorted(seeds)


def grow_from_seeds(key, seeds, mongo_host, matrix_path, i):
    client = MongoClient(host='localhost', port=27017)
    db = client.wiki

    matrix = ss.load_npz(matrix_path)
    csr_matric = matrix.tocsr()  # 1.3GB!
    csc_matric = matrix.tocsc()

    output_path = f'data/wiki/subgraphs/{key}.pkl'
    if os.path.exists(output_path):
        return

    inlinks = {}
    outlinks = {}
    series = {}
    counter = Counter()

    logger.info(f'Building graph for {key}')

    for p in seeds:
        page = db.pages.find_one({'i': int(p)}, projection=['title'])
        if page['title'].startswith('List of'):
            continue

        s = get_traffic_for_page(page['title'], int(p), db)
        if s is None or -1 in s:
            continue

        outlinks[p] = list(csr_matric.getrow(p).nonzero()[1])
        inlinks[p] = list(csc_matric.getcol(p).nonzero()[0])
        series[p] = s

    for page in inlinks:
        # for link in outlinks[page]:
        #     if link not in outlinks:
        #         counter[link] += 1
        for link in inlinks[page]:
            if link not in inlinks:
                counter[link] += 1

    # This between 5-30 minutes
    pbar = tqdm(total=10000, desc=key, position=i)
    pbar.update(len(seeds))
    while len(inlinks) < 10000:
        p = counter.most_common(1)[0][0]
        del counter[p]
        assert p not in inlinks

        page = db.pages.find_one({'i': int(p)}, projection=['title'])
        if page['title'].startswith('List of'):
            continue

        s = get_traffic_for_page(page['title'], int(p), db)
        if s is None or -1 in s:
            continue

        pbar.update(1)

        outlinks[p] = list(csr_matric.getrow(p).nonzero()[1])
        inlinks[p] = list(csc_matric.getcol(p).nonzero()[0])
        series[p] = s

        # for link in outlinks[p]:
        #     if link not in outlinks:
        #         counter[link] += 1
        for link in inlinks[p]:
            if link not in inlinks:
                counter[link] += 1
    pbar.close()

    for link in inlinks:
        inlinks[link] = list(filter(lambda n: n in inlinks, inlinks[link]))
    for link in outlinks:
        outlinks[link] = list(filter(lambda n: n in outlinks, outlinks[link]))

    n_days = len(next(iter(series.values())))
    inlinks = get_dynamic_info(inlinks, n_days, db)

    os.makedirs('data/wiki/subgraphs', exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump([inlinks, outlinks, series], f)


def get_dynamic_info(in_degrees, n_days, db):
    origin = datetime(2015, 7, 1)
    endpoint = datetime(2020, 6, 9)  # includes endpoint

    new_in_degrees = {k: [] for k in in_degrees.keys()}
    for k, neighs in tqdm(in_degrees.items()):
        title = db.pages.find_one({'i': int(k)}, projection=['title'])['title']
        for n in neighs:
            new_in_degrees[k].append({
                'id': n,
                'mask': [True] * n_days,
            })

            p = db.pages.find_one({'i': int(n)})
            periods = None
            for link in p['links']:
                if link['n'] == title:
                    periods = link['t']
                    break
            for period in periods:
                start = (period['s'] - origin).days
                start = max(0, start)

                if 'e' not in period:
                    end = n_days
                else:
                    end = (period['e'] - origin).days
                    if end < 0:
                        continue
                    end += 1  # since we need to exclude endpoint

                new_in_degrees[k][-1]['mask'][start:end] = [False] * \
                    (end - start)

    return new_in_degrees


def get_subgraph_traffic_from_dump(topic):
    series_path = f'data/wiki/subgraphs/{topic}.2011.series.pkl'
    if os.path.exists(series_path):
        return

    with open(f'data/wiki/subgraphs/{topic}.pkl', 'rb') as f:
        inlinks, outlinks = pickle.load(f)

    paths = glob('data/wiki/serialized_traffic/*_*.npy')
    paths.sort()

    series = defaultdict(list)
    missing = set()

    # Extract time series
    for path in tqdm(paths):
        m = np.load(path)

        for p in inlinks:
            s = m[p]
            if -1 in s:
                missing.add(p)
            else:
                series[p] += s.tolist()

    # Remove nodes with missing time series
    for p in inlinks:
        inlinks[p] = list(filter(lambda n: n not in missing, inlinks[p]))
        outlinks[p] = list(filter(lambda n: n not in missing, outlinks[p]))
    for p in missing:
        del inlinks[p]
        del outlinks[p]
        if p in series:
            del series[p]

    # Count the number of days in 2017
    total = {}
    for k, v in series.items():
        total[k] = sum(v[:(7 * 52 * 7)])

    # Sort neighbours by traffic in 2017 (most view counts to least)
    for k, v in inlinks.items():
        inlinks[k] = sorted(v, reverse=True, key=lambda x: total[x])

    with open(f'data/wiki/subgraphs/{topic}.2011.cleaned.pkl', 'wb') as f:
        pickle.dump([inlinks, outlinks], f)

    with open(series_path, 'wb') as f:
        pickle.dump(series, f)


def get_traffic_for_page(o_title, i, db):
    s = db.series.find_one({'_id': i})
    if s is not None:
        return s['s']

    # Reproduce the data collection process
    start = '2015070100'
    end = '2020060900'
    title = o_title.replace('%', r'%25').replace(
        '/', r'%2F').replace('?', r'%3F')
    domain = 'en.wikipedia.org'
    source = 'all-access'
    agent = 'user'
    title = title.replace(' ', '_')
    url = f'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{domain}/{source}/{agent}/{title}/daily/{start}/{end}'

    # Rate limit 100 requests/second
    while True:
        response = requests.get(url).json()
        if 'items' in response:
            break
        elif 'type' in response and 'request_rate_exceeded' in response['type']:
            time.sleep(random.uniform(5, 10))
            continue
        else:
            print(o_title, response)
            return None

    response = response['items']

    if len(response) < 1:
        return {}
    first = response[0]['timestamp']
    last = response[-1]['timestamp']

    first = datetime.strptime(first, '%Y%m%d00')
    last = datetime.strptime(last, '%Y%m%d00')

    start_dt = datetime.strptime(start, '%Y%m%d00')
    end_dt = datetime.strptime(end, '%Y%m%d00')

    left_pad_size = (first - start_dt).days
    series = [-1] * left_pad_size

    current_ts = first - timedelta(days=1)

    for o in response:
        ts = datetime.strptime(o['timestamp'], '%Y%m%d00')
        diff = (ts - current_ts).days
        if diff > 1:
            n_empty_days = diff - 1
            for _ in range(n_empty_days):
                series.append(-1)
        else:
            assert diff == 1

        series.append(o['views'])
        current_ts = ts

    if end_dt != last:
        n_empty_days = (end_dt - last).days
        for _ in range(n_empty_days):
            series.append(-1)
    assert len(series) == 1806

    return series


def extract_all(mongo_host, redis_host):
    out_path = 'data/wiki/static_edge_list_2015.txt'
    if not os.path.exists(out_path):
        extract_static_graph(mongo_host, redis_host, out_path)

    matrix_path = "data/wiki/static_edge_list_2015.npz"
    if not os.path.exists(matrix_path):
        matrix = read_data_file_as_coo_matrix(out_path)  # 1.5GB!
        ss.save_npz(matrix_path, matrix)
    matrix = ss.load_npz(matrix_path)
    csr_matric = matrix.tocsr()  # 1.3GB!

    client = MongoClient(host='localhost', port=27017)
    db = client.wiki

    seeds = {}
    seeds['programming'] = get_seeds_from_cat('Programming languages', db)
    seeds['star_wars'] = get_seeds_from_title('Star Wars', db, csr_matric)
    seeds['stats'] = get_seeds_from_title('Statistics', db, csr_matric)
    seeds['flu'] = get_seeds_from_title('2009 swine flu pandemic',
                                        db, csr_matric)

    with Parallel(n_jobs=8, backend='loky') as parallel:
        parallel(delayed(grow_from_seeds)(key, seeds, mongo_host, matrix_path, i)
                 for i, (key, seeds) in enumerate(seeds.items()))

    client.close()


def main():
    args = docopt(__doc__, version='0.0.1')
    args = validate(args)

    if args['ptvsd']:
        address = ('0.0.0.0', args['ptvsd'])
        ptvsd.enable_attach(address)
        ptvsd.wait_for_attach()

    extract_all(args['mongo'], args['redis'])


if __name__ == '__main__':
    main()
