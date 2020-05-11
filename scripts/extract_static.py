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

import redis
import tiledb
from nos.utils import setup_logger

logger = setup_logger()


def extract_static_graph(mongo_host, redis_host, out_path):
    client = MongoClient(host=mongo_host, port=27017)
    db = client.wiki

    r = redis.Redis(host=redis_host, port=6379, db=0)

    origin = datetime(2019, 1, 1)
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


def main():
    args = docopt(__doc__, version='0.0.1')
    args = validate(args)

    if args['ptvsd']:
        address = ('0.0.0.0', args['ptvsd'])
        ptvsd.enable_attach(address)
        ptvsd.wait_for_attach()

    out_path = 'data/wiki/static_edge_list_2019.txt'
    if not os.path.exists(out_path):
        extract_static_graph(args['mongo'], args['redis'], out_path)


if __name__ == '__main__':
    main()
