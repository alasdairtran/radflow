"""Extract wiki subgraphs

Usage:
    extract_wiki_subgraphs.py [options] SEED

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -m --mongo HOST     MongoDB host [default: localhost].
    -d --depth INT      Depth of the category [default: 5]
    -e --end DATE       End date [default: 2020013100]
    SEED                Seed word.
"""

import os
import pickle
import random
import time
from collections import deque
from datetime import datetime, timedelta

import ptvsd
import requests
from bs4 import BeautifulSoup
from docopt import docopt
from pymongo import MongoClient
from schema import And, Or, Schema, Use
from tqdm import tqdm

from nos.utils import setup_logger

logger = setup_logger()


def get_titles_from_cat(seed_word, db):
    titles = set()
    title2id = {}
    for p in db.pages.find({'cats': seed_word}, projection=['i', 'title']):
        if p['title'].startswith('List of'):
            continue
        titles.add(p['title'])
        title2id[p['title']] = int(p['i'])
    return titles, title2id


def get_series(seed, series_path, cat_path, title2id_path, db, max_depth, end):
    series = {}
    title2id = {}
    id2title = {}
    explored_titles = set()
    cats = set([seed])
    unexplored_queue = deque([seed])
    cat_depths = {seed: 0}

    logger.info(f'Getting series for {seed}')
    pbar = tqdm(position=0)
    with open(cat_path, 'w') as f:
        f.write(f'{seed}\n')
        while unexplored_queue:
            query = unexplored_queue.popleft()
            cat_titles, cat_title2id = get_titles_from_cat(query, db)
            unexplored_titles = cat_titles - explored_titles
            for title in unexplored_titles:
                i = cat_title2id[title]
                s = get_traffic_for_page(title, i, db, end)
                if s is None:
                    continue
                series[i] = s
                title2id[title] = i
                id2title[i] = title
                pbar.update(1)
                if len(series) >= 10000:
                    break
            if len(series) >= 10000:
                break
            explored_titles |= unexplored_titles

            main_url = f'https://en.wikipedia.org/wiki/Category:{query}'
            page = requests.get(main_url)
            soup = BeautifulSoup(page.content, 'html.parser')
            body = soup.find(id='mw-subcategories')
            if body:
                links = body.find_all('a')
                for link in links:
                    title = link.get('title')
                    if title.startswith('Category:'):
                        cat = title[9:]
                        d = cat_depths[query] + 1
                        if cat not in cats and d < max_depth:
                            cats.add(cat)
                            f.write(f'{cat}\n')
                            unexplored_queue.append(cat)
                            cat_depths[cat] = d

    with open(series_path, 'wb') as f:
        pickle.dump(series, f)
    with open(title2id_path, 'wb') as f:
        pickle.dump(title2id, f)


def grow_from_cats(key, seed, mongo_host, depth, end):
    client = MongoClient(host=mongo_host, port=27017)
    db = client.wiki

    output_dir = os.path.join('data', 'wiki', 'subgraphs', key)
    os.makedirs(output_dir, exist_ok=True)

    cat_path = os.path.join(output_dir, 'cats.txt')
    series_path = os.path.join(output_dir, 'series.pkl')
    in_degrees_path = os.path.join(output_dir, 'in_degrees.pkl')
    title2id_path = os.path.join(output_dir, 'title2id.pkl')
    neighs_path = os.path.join(output_dir, 'neighs.pkl')

    if not os.path.exists(series_path):
        get_series(seed, series_path, cat_path, title2id_path, db, depth, end)
    logger.info(f'Loading series from {series_path}')
    with open(series_path, 'rb') as f:
        series = pickle.load(f)
    logger.info(f'Loading title2id from {series_path}')
    with open(title2id_path, 'rb') as f:
        title2id = pickle.load(f)

    n_days = len(next(iter(series.values())))
    if not os.path.exists(in_degrees_path):
        in_degrees = get_dynamic_edges(series.keys(), title2id, n_days, db)
        with open(in_degrees_path, 'wb') as f:
            pickle.dump(in_degrees, f)
    logger.info(f'Loading dynamic edges from {in_degrees_path}')
    with open(in_degrees_path, 'rb') as f:
        in_degrees = pickle.load(f)

    if not os.path.exists(neighs_path):
        neighs = {k: {} for k in in_degrees.keys()}
        logger.info(f'Caching neighbours')
        for t in tqdm(range(n_days)):
            for k, v in in_degrees.items():
                k_neighs = [n['id'] for n in v if n['mask'][t] == 0]
                k_views = [series[n['id']][t]
                           for n in v if n['mask'][t] == 0]
                k_neighs = [x for _, x in sorted(
                    zip(k_views, k_neighs), reverse=True)]
                neighs[k][t] = k_neighs
        with open(neighs_path, 'wb') as f:
            pickle.dump(neighs, f)


def get_dynamic_edges(keys, title2id, n_days, db):
    origin = datetime(2015, 7, 1)

    in_degrees = {k: [] for k in keys}
    logger.info(f'Generating dynamic edges')
    for u in tqdm(in_degrees):
        page = db.pages.find_one({'i': u})
        for link in page['links']:
            # If there is edge from u to v
            if link['n'] in title2id:
                v = title2id[link['n']]
                in_degrees[v].append({
                    'id': u,
                    'mask': [True] * n_days,
                })
                for period in link['t']:
                    start = (period['s'] - origin).days
                    start = max(0, start)
                    if 'e' not in period:
                        end = n_days
                    else:
                        end = (period['e'] - origin).days
                        if end < 0:
                            continue
                        end += 1  # since we need to exclude endpoint
                    duration = end - start
                    in_degrees[v][-1]['mask'][start:end] = [False] * duration

    return in_degrees


def get_traffic_for_page(o_title, i, db, end):
    s = db.series.find_one({'_id': i})
    if s is not None:
        return s['s']

    # Reproduce the data collection process
    start = '2015070100'
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
    assert len(series) == 1676

    # The last 140 days must not contain missing data
    if -1 in series[-140:]:
        return None

    # Ignore low traffic pages
    non_missing_views = list(filter(lambda x: x != 1, series))
    avg_views = sum(non_missing_views) / len(non_missing_views)
    if avg_views < 100:
        return None

    return series


def extract_all(mongo_host, seed, depth, end):
    key = seed.lower().replace(' ', '_')
    grow_from_cats(key, seed, mongo_host, depth, end)


def validate(args):
    """Validate command line arguments."""
    args = {k.lstrip('-').lower().replace('-', '_'): v
            for k, v in args.items()}
    schema = Schema({
        'ptvsd': Or(None, And(Use(int), lambda port: 1 <= port <= 65535)),
        'mongo': Or(None, str),
        'seed': str,
        'depth': Use(int),
        'end': str,
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

    extract_all(args['mongo'], args['seed'], args['depth'], args['end'])


if __name__ == '__main__':
    main()
