"""Annotate Good News with parts of speech.

Usage:
    get_traffic.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -h --host HOST      MongoDB host.
    -n --n-jobs INT     Number of jobs [default: 40].

"""

import bz2
import fileinput
import html
import os
import pickle
import random
import re
import time
from calendar import monthrange
from datetime import datetime, timedelta
from glob import glob

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

from nos.utils import setup_logger

logger = setup_logger()

PATTERN = re.compile(r'[0-9]+')


def extract_line(fin, db, path, is_bz2):
    filename = os.path.basename(path)
    parts = filename.split('.')[0].split('-')
    year = int(parts[1])
    month = int(parts[2])
    n_days = monthrange(year, month)[1]

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

        _, o_title, monthly_total, hourly_counts = line.strip().split(' ')
        title = html.unescape(o_title)
        title = title.replace('_', ' ')
        p = db.pages.find_one({'title': title}, projection=['_id', 'title'])
        if not p:
            continue

        query = {'p': p['_id'], 'y': year, 'm': month}
        if db.monthlyTraffic.find_one(query, projection=['_id']):
            continue

        monthly_total = int(monthly_total)

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

        db.monthlyTraffic.insert_one({
            'p': p['_id'],
            'y': year,
            'm': month,
            'v': views,
        })


def resolve_ambiguity(db, title, id1, id2, title2id, id2title):
    url1 = f'https://en.wikipedia.org/w/api.php?action=query&prop=info&pageids={id1}&inprop=url&format=json'
    url2 = f'https://en.wikipedia.org/w/api.php?action=query&prop=info&pageids={id2}&inprop=url&format=json'
    res1 = requests.get(url1).json()['query']['pages'][str(id1)]
    res2 = requests.get(url2).json()['query']['pages'][str(id2)]

    if 'missing' in res1:
        title2 = res2['title']
        logger.info(f"Title: {title} -> {title2}, remove {id1}, keep {id2}")
        db.pages.delete_one({'_id': id1})
        db.pages.update_one({'_id': id2}, {'$set': {'title': title2}})
        del id2title[id1]
        del title2id[title]
        title2id[title2] = id2
        id2title[id2] = title2

    elif 'missing' in res2:
        title1 = res1['title']
        logger.info(f"Title: {title} -> {title1}, remove {id2}, keep {id1}")
        db.pages.delete_one({'_id': id2})
        db.pages.update_one({'_id': id1}, {'$set': {'title': title1}})
        del title2id[title]
        title2id[title1] = id1
        id2title[id1] = title1

    else:
        title1 = res1['title']
        title2 = res2['title']

        logger.info(f"Title: {title} -> {title1}, keep {id1}")
        logger.info(f"Title: {title} -> {title2}, keep {id2}")

        assert title1 != title2 and title in [title1, title2]

        db.pages.update_one({'_id': id1}, {'$set': {'title': title1}})
        db.pages.update_one({'_id': id2}, {'$set': {'title': title2}})

        title2id[title1] = id1
        title2id[title2] = id2

        id2title[id1] = title1
        id2title[id2] = title2


def get_id_maps(db, index_path):
    id2title = {}
    title2id = {}

    # Note that these articles have the same title but different page IDs
    # Title: Anjan Chowdhury -> Anjan Chowdhury, remove 21606610, keep 62911656
    # Title: You Si-kun -> Yu Shyi-kun, keep 349072
    # Title: You Si-kun -> You Si-kun, keep 62998113
    # Title: James Coyne -> James Elliott Coyne, keep 520170
    # Title: James Coyne -> James Coyne, keep 62999621
    # Title: Amanieu VI -> Amanieu V d'Albret, keep 10037159
    # Title: Amanieu VI -> Amanieu VI, keep 63000573
    # Title: Amanieu VIII -> Amanieu VII d'Albret, keep 10037418
    # Title: Amanieu VIII -> Amanieu VIII, keep 63000585
    # Title: Bernard Ezi IV -> Bernard Ezi II d'Albret, keep 10038254
    # Title: Bernard Ezi IV -> Bernard Ezi IV, keep 63000589
    # Title: Arnaud Amanieu, Lord of Albret -> Arnaud Amanieu d'Albret, keep 10038002
    # Title: Arnaud Amanieu, Lord of Albret -> Arnaud Amanieu, Lord of Albret, keep 63000655
    # Title: James Elliott Coyne -> James Elliott Coyne, remove 1217332, keep 520170
    # Title: Air New Zealand Flight 901 -> Air New Zealand Flight 901, keep 62998459
    # Title: Air New Zealand Flight 901 -> Mount Erebus disaster, keep 1158000
    # Title: FIVB Senior World and Continental Rankings -> FIVB Senior World and Continental Rankings, keep 63000600
    # Title: FIVB Senior World and Continental Rankings -> FIVB Senior World Rankings, keep 1463363
    # Title: Mount Erebus disaster -> Mount Erebus disaster, remove 2224953, keep 1158000
    # Title: Daily Mashriq -> Daily Mashriq, remove 63000119, keep 4737576
    # Title: Zeewijk -> Zeewijk, keep 63000552
    # Zeewijk -> Zeewijk (1725), keep 5931503
    # Title: XMultiply -> XMultiply, keep 62998806
    # Title: XMultiply -> X Multiply, keep 5103829

    logger.info('Building title-id index map')
    for p in tqdm(db.pages.find({}, projection=['_id', 'title'])):
        title = p['title']
        if title in title2id and title2id[title] != p['_id']:
            resolve_ambiguity(db, title, title2id[title], p['_id'],
                              title2id, id2title)
        else:
            title2id[title] = p['_id']
            assert p['_id'] not in title2id
            id2title[p['_id']] = title

    # Statistics: 1,7035,758 unique titles/IDs.
    logger.info(f"Number of IDs: {len(id2title)}")
    logger.info(f"Number of Titles: {len(title2id)}")
    with open(index_path, 'wb') as f:
        pickle.dump([title2id, id2title], f)


def get_traffic(path, index_path, host, i):
    time.sleep(random.uniform(1, 5))

    client = MongoClient(host=host, port=27017)
    db = client.wiki

    if path.endswith('.bz2'):
        with BZ2File(path) as fin:
            extract_line(fin, db, path, True)
    else:
        with open(path) as fin:
            extract_line(fin, db, path, False)


def get_all_traffic(host, n_jobs):
    paths = glob('/data4/u4921817/nos/data/pagecounts/pagecounts*')
    paths.sort()

    client = MongoClient(host=host, port=27017)
    db = client.wiki
    db.monthlyTraffic.create_index([
        ('p', pymongo.ASCENDING),
        ('y', pymongo.ASCENDING),
        ('m', pymongo.ASCENDING),
    ], unique=True)

    index_path = os.path.join('data/wiki', 'title_index.pkl')
    if not os.path.exists(index_path):
        get_id_maps(db, index_path)

    client.close()

    # This takes 1.5 hours
    logger.info('Extracting traffic counts')
    with Parallel(n_jobs=n_jobs, backend='loky') as parallel:
        parallel(delayed(get_traffic)(path, index_path, host, i)
                 for i, path in tqdm(enumerate(paths)))


def validate(args):
    """Validate command line arguments."""
    args = {k.lstrip('-').lower().replace('-', '_'): v
            for k, v in args.items()}
    schema = Schema({
        'ptvsd': Or(None, And(Use(int), lambda port: 1 <= port <= 65535)),
        'n_jobs': Use(int),
        'host': Or(None, str),
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

    get_all_traffic(args['host'], args['n_jobs'])


if __name__ == '__main__':
    main()
