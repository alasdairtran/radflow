"""Extract intro
Usage:
    get_traffic_all.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -m --mongo HOST     MongoDB host [default: localhost].
    -n --n-jobs INT     Number of jobs [default: 50].

"""

import math
import pickle
import random
import time
from datetime import datetime, timedelta

import h5py
import numpy as np
import ptvsd
import requests
from docopt import docopt
from joblib import Parallel, delayed
from pymongo import MongoClient
from requests.exceptions import ChunkedEncodingError
from schema import And, Or, Schema, Use
from tqdm import tqdm

from nos.utils import setup_logger

logger = setup_logger()


def get_traffic_for_page(o_title, source):
    # Reproduce the data collection process
    start = '2015070100'
    end = '2020063000'
    title = o_title.replace('%', r'%25').replace(
        '/', r'%2F').replace('?', r'%3F')
    domain = 'en.wikipedia.org'
    agent = 'user'
    title = title.replace(' ', '_')
    url = f'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{domain}/{source}/{agent}/{title}/daily/{start}/{end}'

    # Rate limit 100 requests/second
    while True:
        count = 0
        try:
            response = requests.get(url).json()
        except (ConnectionError, ChunkedEncodingError):
            if count < 10:
                count += 1
                time.sleep(random.uniform(5, 10))
                continue
            else:
                print(o_title, response)
                return None
        if 'items' in response:
            break
        elif 'type' in response and ('request_rate_exceeded' in response['type'] or 'internal_http_error' in response['type']):
            if count < 10:
                count += 1
                time.sleep(random.uniform(5, 10))
                continue
            else:
                print(o_title, response)
                return None
        else:
            print(o_title, response)
            return None

    response = response['items']

    if len(response) < 1:
        return {
            'series': [],
            'first_date': None,
            'complete': False,  # no missing views last 140 days
            'avg_views': 0,  # in the last 140 days
        }
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
    assert len(series) == 1827

    output = {
        'series': series,
        'first_date': first,
        'complete': -1 not in series[-140:],  # no missing views last 140 days
        'avg_views': sum(series[-140:]) / 140,  # in the last 140 days
    }
    return output


def get_traffic_for_batch(mongo_host, i, to_do_titles):
    time.sleep(random.uniform(1, 5))

    client = MongoClient(host=mongo_host, port=27017)
    db = client.wiki2

    for title, node_id in tqdm(to_do_titles, position=i, desc=str(i)):
        desktop = get_traffic_for_page(title, 'desktop')
        app = get_traffic_for_page(title, 'mobile-app')
        mobile = get_traffic_for_page(title, 'mobile-web')

        db.series.insert_one({
            '_id': node_id,
            'd': desktop,
            'a': app,
            'm': mobile,
        })


def get_traffic_all(mongo_host, n_jobs):
    with open('data/wiki/title2graphid.pkl', 'rb') as f:
        title2graphid = pickle.load(f)

    all_ids = set(title2graphid.values())

    client = MongoClient(host=mongo_host, port=27017)
    db = client.wiki2

    pages = db.series.find({}, [], batch_size=1000000)
    done_ids = set(page['_id'] for page in pages)

    to_do_ids = all_ids - done_ids
    to_do_titles = [(k, v) for k, v in title2graphid.items() if v in to_do_ids]
    to_do_titles = sorted(to_do_titles)

    batch_size = int(math.ceil(len(to_do_ids) / n_jobs))

    client.close()

    logger.info('Extracting intros')
    with Parallel(n_jobs=n_jobs, backend='loky') as parallel:
        parallel(delayed(get_traffic_for_batch)(mongo_host, i, sorted(to_do_titles[s: s + batch_size]))
                 for i, s in enumerate(range(0, len(to_do_titles), batch_size)))


def generate_hdf5(mongo_host):
    client = MongoClient(host=mongo_host, port=27017)
    db = client.wiki2
    pages = db.series.find({})

    views = np.full((366802, 1827, 3), -1, dtype=np.int32)
    for page in tqdm(pages):
        series = np.array([page['d']['series'], page['m']
                           ['series'], page['a']['series']], dtype=np.int32)
        series = series.transpose()
        views[page['_id']] = series

    f = h5py.File('data/wiki/views_all.hdf5', 'a')
    f.create_dataset('views', dtype=np.int32, data=views)


def validate(args):
    """Validate command line arguments."""
    args = {k.lstrip('-').lower().replace('-', '_'): v
            for k, v in args.items()}
    schema = Schema({
        'ptvsd': Or(None, And(Use(int), lambda port: 1 <= port <= 65535)),
        'n_jobs': Use(int),
        'mongo': Or(None, str),
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

    get_traffic_all(args['mongo'], args['n_jobs'])
    generate_hdf5(args['mongo'])


if __name__ == '__main__':
    main()
