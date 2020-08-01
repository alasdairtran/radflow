"""Extract intro
Usage:
    get_traffic.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -m --mongo HOST     MongoDB host [default: localhost].
    -n --n-jobs INT     Number of jobs [default: 20].

"""

import math
import pickle
import random
import time

import h5py
import numpy as np
import ptvsd
import requests
import torch
from docopt import docopt
from joblib import Parallel, delayed
from pymongo import MongoClient
from requests.exceptions import ChunkedEncodingError, ConnectionError
from schema import And, Or, Schema, Use
from tqdm import tqdm

from nos.utils import setup_logger

logger = setup_logger()


def get_intro_for_batch(mongo_host, i, to_do_titles):
    time.sleep(random.uniform(1, 5))

    client = MongoClient(host=mongo_host, port=27017)
    db = client.wiki2

    pbar = tqdm(total=len(to_do_titles), desc=str(i), position=i)
    for start in range(0, len(to_do_titles), 10):
        pairs = to_do_titles[start: start + 10]
        tile2id = {p[0]: p[1] for p in pairs}
        titles = '|'.join([p[0] for p in pairs])

        url = 'https://en.wikipedia.org/w/api.php'
        params = {
            'action': 'query',
            'format': 'json',
            'titles': titles,
            'prop': 'extracts',
            'exintro': True,
            'explaintext': True,
        }
        response = requests.get(url, params).json()
        for v in response['query']['pages'].values():
            title = v['title']
            extract = v['extract'] if 'extract' in v else ''

            db.intros.insert_one({
                '_id': tile2id[title],
                't': title,
                'e': extract,
            })

            del tile2id[title]

        for title in tile2id:
            print('Not found: ', title)

        pbar.update(10)

    pbar.close()


def get_all_intros(mongo_host, n_jobs):
    client = MongoClient(host=mongo_host, port=27017)
    db = client.wiki2

    with open('data/wiki/title2graphid.pkl', 'rb') as f:
        title2graphid = pickle.load(f)

    all_ids = set(title2graphid.values())

    pages = db.intros.find({}, [], batch_size=1000000)
    done_ids = set(page['_id'] for page in pages)

    to_do_ids = all_ids - done_ids
    to_do_titles = [(k, v) for k, v in title2graphid.items() if v in to_do_ids]
    to_do_titles = sorted(to_do_titles)

    batch_size = int(math.ceil(len(to_do_ids) / n_jobs))

    client.close()

    logger.info('Extracting intros')
    with Parallel(n_jobs=n_jobs, backend='loky') as parallel:
        parallel(delayed(get_intro_for_batch)(mongo_host, i, sorted(to_do_titles[s: s + batch_size]))
                 for i, s in enumerate(range(0, len(to_do_titles), batch_size)))


def get_intro_embeds(mongo_host):
    roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
    roberta = roberta.eval().cuda()

    client = MongoClient(host=mongo_host, port=27017)
    db = client.wiki2

    data_f = h5py.File('data/wiki/embeds.hdf5', 'a')
    embeds = np.zeros((366802, 1024), dtype=np.float64)

    with torch.no_grad():
        for p in tqdm(db.intros.find({})):
            if not p['e']:
                continue

            i = p['_id']

            text = p['e']
            tokens = roberta.encode(text)[:512]
            last_layer_features = roberta.extract_features(tokens[:512])
            embeds[i] = last_layer_features[0][0].cpu().numpy()

    data_f.create_dataset('probs', dtype=np.float64, data=embeds)


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

    get_all_intros(args['mongo'], args['n_jobs'])
    get_intro_embeds(args['mongo'])


if __name__ == '__main__':
    main()
