"""Annotate Good News with parts of speech.

Usage:
    get_traffic.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -n --n-jobs INT     Number of jobs [default: 36].

"""

import bz2
import fileinput
import html
import os
import pickle
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


def extract_bytes(fin, fout, ferr, title2id, n_days):
    for line in fin:
        # All English wikipedia articles start with en.z
        if not line.startswith(b'en.z'):
            continue

        # Everything from now is needed. Let's convert to str
        try:
            line = line.decode('utf-8')
        except UnicodeDecodeError:
            ferr.write(f'Unicode error\n')
            continue  # let's just ignore error for now

        try:
            _, o_title, monthly_total, hourly_counts = line.split()
        except ValueError:
            ferr.write(f'Unexpected format error\n')
            continue  # let's just ignore error for now
        title = html.unescape(o_title)
        title = title.replace('_', ' ')
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

        if title in title2id:
            id_ = title2id[title]
            views_str = ' '.join([str(v) for v in views])
            fout.write(f'{id_} {views_str}\n')

        else:
            ferr.write(f'{o_title} {title}\n')


def extract_text(fin, fout, ferr, title2id, n_days):
    for line in fin:
        # All English wikipedia articles start with en.z
        if not line.startswith('en.z'):
            continue

        try:
            _, o_title, monthly_total, hourly_counts = line.split()
        except ValueError:
            ferr.write(f'Unexpected format error\n')
            continue  # let's just ignore error for now
        title = html.unescape(o_title)
        title = title.replace('_', ' ')
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

        if title in title2id:
            id_ = title2id[title]
            views_str = ' '.join([str(v) for v in views])
            fout.write(f'{id_} {views_str}\n')

        else:
            ferr.write(f'{o_title} {title}\n')


def get_traffic(path, result_dir, i):
    filename = os.path.basename(path)
    out_path = os.path.join(result_dir, f'{filename}.txt')
    err_path = os.path.join(result_dir, f'{filename}.err')
    if os.path.exists(out_path):
        return

    index_path = os.path.join('data/wiki', 'title_index.pkl')
    with open(index_path, 'rb') as f:
        title2id, _ = pickle.load(f)

    # First figure out which month and year it is
    parts = filename.split('.')[0].split('-')
    year = int(parts[1])
    month = int(parts[2])

    n_days = monthrange(year, month)[1]

    with open(out_path, 'a') as fout:
        with open(err_path, 'a') as ferr:
            if path.endswith('.bz2'):
                with BZ2File(path) as fin:
                    extract_bytes(fin, fout, ferr, title2id, n_days)
            else:
                with open(path) as fin:
                    extract_text(fin, fout, ferr, title2id, n_days)


def get_all_traffic(n_jobs):
    # This takes 1.5 hours
    result_dir = 'data/wiki/traffic'
    os.makedirs(result_dir, exist_ok=True)

    paths = glob('/data4/u4921817/nos/data/pagecounts/pagecounts*')
    paths.sort()

    with Parallel(n_jobs=n_jobs, backend='loky') as parallel:
        parallel(delayed(get_traffic)(path, result_dir, i)
                 for i, path in tqdm(enumerate(paths)))


def validate(args):
    """Validate command line arguments."""
    args = {k.lstrip('-').lower().replace('-', '_'): v
            for k, v in args.items()}
    schema = Schema({
        'ptvsd': Or(None, And(Use(int), lambda port: 1 <= port <= 65535)),
        'n_jobs': Use(int),
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

    get_all_traffic(args['n_jobs'])


if __name__ == '__main__':
    main()
