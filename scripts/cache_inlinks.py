"""Cache inlinks

Usage:
    cache_inlinks.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -m --mongo HOST     MongoDB host [default: localhost].

"""


import ptvsd
from docopt import docopt
from pymongo import MongoClient
from schema import And, Or, Schema, Use
from tqdm import tqdm

from radflow.utils import setup_logger

logger = setup_logger()


def cache_all_inlinks(mongo_host):
    client = MongoClient(host=mongo_host, port=27017)
    db = client.wiki

    for source_page in tqdm(db.pages.find({})):
        for link in source_page['links']:
            dest_title = link['n']
            inlink = {
                'i': source_page['i'],
                't': link['t'],
            }
            db.pages.update_one({'title': dest_title}, {
                                '$push': {'inlinks': inlink}})


def validate(args):
    """Validate command line arguments."""
    args = {k.lstrip('-').lower().replace('-', '_'): v
            for k, v in args.items()}
    schema = Schema({
        'ptvsd': Or(None, And(Use(int), lambda port: 1 <= port <= 65535)),
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

    cache_all_inlinks(args['mongo'])


if __name__ == '__main__':
    main()
