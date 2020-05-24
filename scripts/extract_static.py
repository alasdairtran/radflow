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
from collections import Counter
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

    # Takes 10.5h to extract 17,035,758 nodes and 185,809,379 edges
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


def grow_from_seeds(seed_word, db, csr_matric, csc_matric):
    seeds = []
    for p in db.pages.find({'cats': seed_word}, projection=['i']):
        seeds.append(p['i'])

    inlinks = {}
    outlinks = {}
    counter = Counter()

    for p in seeds:
        outlinks[p] = list(csr_matric.getrow(p).nonzero()[1])
        inlinks[p] = list(csc_matric.getcol(p).nonzero()[0])

    for page in outlinks:
        for link in outlinks[page]:
            if link not in outlinks:
                counter[link] += 1
        for link in inlinks[page]:
            if link not in inlinks:
                counter[link] += 1

    # This takes 30m
    for _ in tqdm(range(3000)):
        p = counter.most_common()[0][0]
        del counter[p]
        assert p not in outlinks

        outlinks[p] = list(csr_matric.getrow(p).nonzero()[1])
        inlinks[p] = list(csc_matric.getcol(p).nonzero()[0])

        for link in outlinks[p]:
            if link not in outlinks:
                counter[link] += 1
        for link in inlinks[p]:
            if link not in inlinks:
                counter[link] += 1

    for link in inlinks:
        inlinks[link] = list(filter(lambda n: n in inlinks, inlinks[link]))
    for link in outlinks:
        outlinks[link] = list(filter(lambda n: n in outlinks, outlinks[link]))

    os.makedirs('data/wiki/subgraphs', exist_ok=True)
    with open(f'data/wiki/subgraphs/{seed_word}.pkl', 'wb') as f:
        pickle.dump([inlinks, outlinks])


def extract_all(mongo_host, redis_host):
    out_path = 'data/wiki/static_edge_list_2019.txt'
    if not os.path.exists(out_path):
        extract_static_graph(mongo_host, redis_host, out_path)

    matrix_path = "data/wiki/static_edge_list_2019.npz"
    if not os.path.exists(matrix_path):
        matrix = read_data_file_as_coo_matrix(out_path)  # 1.5GB!
        ss.save_npz(matrix_path, matrix)
    matrix = ss.load_npz(matrix_path)
    csr_matric = matrix.tocsr()  # 1.3GB!
    csc_matric = matrix.tocsc()

    client = MongoClient(host='localhost', port=27017)
    db = client.wiki
    grow_from_seeds('Programming languages', db, csr_matric, csc_matric)


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
