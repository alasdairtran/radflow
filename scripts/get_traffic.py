"""Annotate Good News with parts of speech.

Usage:
    get_traffic.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -h --host HOST      MongoDB host.
    -n --n-jobs INT     Number of jobs [default: 36].
    -t --traffic PATH   Path to traffic [default: data/wiki/traffic].

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
from datetime import datetime, timedelta
from glob import glob

import numpy as np
import ptvsd
import pymongo
import pytz
import requests
from bz2file import BZ2File
from docopt import docopt
from joblib import Parallel, delayed
from pymongo import MongoClient
from schema import And, Or, Schema, Use
from tqdm import tqdm

import tiledb
from nos.utils import setup_logger

logger = setup_logger()

PATTERN = re.compile(r'[0-9]+')


def extract_line(fin, db, traffic_path, path, n_days, is_bz2):
    page_views = np.full((17035758, n_days), -1, dtype=np.int32)

    for line in fin:
        # All English wikipedia articles start with en.z
        prefix = b'en.z' if is_bz2 else 'en.z'
        if not line.startswith(prefix):
            continue

        # Everything from now is needed. Let's convert to str
        if is_bz2:
            try:
                line = line.decode('utf-8')
            except UnicodeDecodeError:
                continue  # let's just ignore error for now

        _, o_title, _, hourly_counts = line.strip().split(' ')
        title = html.unescape(o_title)
        title = title.replace('_', ' ')
        p = db.pages.find_one({'title': title}, projection=['i'])
        if not p:
            continue

        # query = {'p': p['_id'], 'y': year, 'm': month}
        # if db.monthlyTraffic.find_one(query, projection=['_id']):
        #     continue

        # monthly_total = int(monthly_total)

        # Hourly counts are comma separated
        count_list = hourly_counts.split(',')

        # We're only interested in daily counts, so let's aggregate
        views = [0] * n_days
        for count in count_list:
            if not count:
                continue
            day = ord(count[0]) - ord('A')
            numbers = PATTERN.findall(count)
            views[day] = sum([int(n) for n in numbers])

        # views = np.array(views, dtype=np.uint32)
        page_views[p['i']] = views
        # db.monthlyTraffic.insert_one({
        #     'p': p['_id'],
        #     'y': year,
        #     'm': month,
        #     'v': views,
        # })

    return page_views


# def get_id_maps(db, index_path):
#     id2title = {}
#     title2id = {}
#     logger.info('Building title-id index map')

#     for p in tqdm(db.pages.find({}, projection=['i', 'title'])):
#         title = p['title']
#         title2id[title] = p['_id']
#         id2title[p['_id']] = title

#     # Statistics: 1,7035,758 unique titles/IDs.
#     logger.info(f"Number of IDs: {len(id2title)}")
#     logger.info(f"Number of Titles: {len(title2id)}")
#     with open(index_path, 'wb') as f:
#         pickle.dump([title2id, id2title], f)


def get_traffic(path, host, traffic_path, i):
    time.sleep(random.uniform(1, 5))

    client = MongoClient(host=host, port=27017)
    db = client.wiki

    filename = os.path.basename(path)
    parts = filename.split('.')[0].split('-')
    year = int(parts[1])
    month = int(parts[2])
    n_days = monthrange(year, month)[1]

    origin = datetime(2011, 12, 1)
    first_date = datetime(year, month, 1)
    start = (first_date - origin).days
    assert start >= 0
    end = start + n_days

    if path.endswith('.bz2'):
        with BZ2File(path) as fin:
            page_views = extract_line(
                fin, db, traffic_path, path, n_days, True)
    else:
        with open(path) as fin:
            page_views = extract_line(
                fin, db, traffic_path, path, n_days, False)

    with tiledb.DenseArray(traffic_path, mode='w') as A:
        A[:, start:end] = page_views

    client.close()


def create_traffic_tile(traffic_path):
    if os.path.exists(traffic_path):
        return

    start = datetime(2011, 12, 1)
    end = datetime(2020, 1, 31)
    logger.info('Creating traffic tile')
    # There are 2984 days. Day 0 is 1 Dec 2011
    logger.info(f'No of days: {(end - start).days + 1}')  # 2984

    dom = tiledb.Domain(tiledb.Dim(name='i', domain=(0, 17035757), tile=1, dtype=np.uint32),
                        tiledb.Dim(name="t", domain=(0, 2983), tile=2984, dtype=np.uint32))

    # The array will be dense with a single attribute "v" so each (i,j) cell can store an integer.
    schema = tiledb.ArraySchema(domain=dom, sparse=False,
                                attrs=[tiledb.Attr(name="v", dtype=np.uint32)])

    # Create the (empty) array on disk.
    # Empty cells are represented by 4294967295 for uint32 (largest number  2^32 − 1)
    tiledb.DenseArray.create(traffic_path, schema)


def get_all_traffic(host, n_jobs, traffic_path):
    paths = glob('/data4/u4921817/nos/data/pagecounts/pagecounts-2019*')
    paths.sort()

    # client = MongoClient(host=host, port=27017)
    # db = client.wiki
    # db.monthlyTraffic.create_index([
    #     ('p', pymongo.ASCENDING),
    #     ('y', pymongo.ASCENDING),
    #     ('m', pymongo.ASCENDING),
    # ], unique=True)

    # index_path = os.path.join('data/wiki', 'graph_ids.pkl')
    # if not os.path.exists(index_path):
    #     get_id_maps(db, index_path)

    # client.close()

    create_traffic_tile(traffic_path)

    # This takes 1.5 hours
    logger.info('Extracting traffic counts')
    with Parallel(n_jobs=n_jobs, backend='loky') as parallel:
        parallel(delayed(get_traffic)(path, host, traffic_path, i)
                 for i, path in tqdm(enumerate(paths)))


def validate(args):
    """Validate command line arguments."""
    args = {k.lstrip('-').lower().replace('-', '_'): v
            for k, v in args.items()}
    schema = Schema({
        'ptvsd': Or(None, And(Use(int), lambda port: 1 <= port <= 65535)),
        'n_jobs': Use(int),
        'host': Or(None, str),
        'traffic': str,
    })
    args = schema.validate(args)
    return args


def main():
    args = docopt(__doc__, version='0.0.1')
    args = validate(args)

    if args['ptvsd']:
        address = ('0.0.0.0', args['ptvsd'])
        ptvsd.enable_attach(address)
        ptvsd.wait_for_attach()

    get_all_traffic(args['host'], args['n_jobs'], args['traffic'])


if __name__ == '__main__':
    main()
