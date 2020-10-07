"""Build Neo4j inputs.

Usage:
    build_neo4j_inputs.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.

"""
import csv
import os
import pickle
from datetime import datetime

import h5py
import ptvsd
from docopt import docopt
from pymongo import MongoClient
from schema import And, Or, Schema, Use
from tqdm import tqdm


def build_inputs():
    h5f = h5py.File('data/wiki/wiki.hdf5', 'r')
    views = h5f['views'][...]

    os.makedirs('data/wiki/dgraph', exist_ok=True)
    rdf_f = open('data/wiki/dgraph/wiki.rdf', 'w')

    with open('data/wiki/neo4j/pages.csv') as p_f:
        for i, line in tqdm(enumerate(p_f)):
            if i == 0:
                continue

            # csv automatically replaces "" with " and remove outer quotes
            parts = list(csv.reader([line.strip()],
                                    delimiter=',', quotechar='"'))[0]
            graph_id = parts[0]
            title = parts[3]
            first_date = parts[5]

            title = title.replace('"', r'\"')
            title = f'"{title}"'

            v = views[int(graph_id)]

            first_dt = datetime.strptime(first_date, '%Y-%m-%d')
            n_empty = (first_dt - datetime(2015, 7, 1)).days
            if n_empty > 0:
                assert sum(v[:n_empty]) == -1 * n_empty
                v = v[n_empty:]
                assert v[0] != -1

            # when loading into dgraph, uid must be greater than 0
            uid = int(graph_id) + 1

            rdf_f.write(f'<{uid}> <title>  {title} .\n')
            rdf_f.write(f'<{uid}> <graph_id>  "{graph_id}" .\n')
            rdf_f.write(f'<{uid}> <first_date> "{first_date}" .\n')
            for o in v:
                rdf_f.write(f'<{uid}> <views> "{o}" .\n')
            rdf_f.write(f'<{uid}> <dgraph.type> "Article" .\n')

    attn_path = 'expt/network_aggregation/wiki_univariate/reports/test/serialization/wiki_attention_test.csv'
    with open(attn_path, 'r') as f:
        for i, line in tqdm(enumerate(f)):
            if i == 0:
                continue
            parts = line.strip().split(',')
            source = int(parts[1]) + 1
            target = int(parts[0]) + 1
            rdf_f.write(f'<{source}> <links_to> <{target}> .\n')

    attn_path = 'expt/network_aggregation/wiki_univariate/reports/train/serialization/wiki_attention_train.csv'
    with open(attn_path, 'r') as f:
        for i, line in tqdm(enumerate(f)):
            if i == 0:
                continue
            parts = line.strip().split(',')
            source = int(parts[1]) + 1
            target = int(parts[0]) + 1
            rdf_f.write(f'<{source}> <links_to> <{target}> .\n')

    rdf_f.close()


def validate(args):
    """Validate command line arguments."""
    args = {k.lstrip('-').lower().replace('-', '_'): v
            for k, v in args.items()}
    schema = Schema({
        'ptvsd': Or(None, And(Use(int), lambda port: 1 <= port <= 65535)),
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

    build_inputs()


if __name__ == '__main__':
    main()
