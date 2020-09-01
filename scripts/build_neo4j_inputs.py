"""Build Neo4j inputs.

Usage:
    build_neo4j_inputs.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -m --mongo HOST     MongoDB host [default: localhost].

"""
import os
import pickle
from datetime import datetime

import ptvsd
from docopt import docopt
from pymongo import MongoClient
from schema import And, Or, Schema, Use
from tqdm import tqdm


def build_inputs(mongo_host):
    client = MongoClient(host=mongo_host, port=27017)
    db = client.wiki2

    path = './data/wiki/node_ids/test_ids.pkl'
    with open(path, 'rb') as f:
        test_ids = pickle.load(f)

    path = './data/wiki/trafficid2graphid.pkl'
    with open(path, 'rb') as f:
        trafficid2graphid = pickle.load(f)

    path = './data/wiki/title2graphid.pkl'
    with open(path, 'rb') as f:
        title2graphid = pickle.load(f)

    category_map = {}

    page_header = 'graphID,trafficID,pageID,title,test,startDate,desktop,mobile,app'
    cat_header = 'catID,title'
    page_link_header = 'startID,startDates,endDates,endID'
    cat_link_header = 'startID,startDates,endDates,endID'

    os.makedirs('data/wiki/neo4j', exist_ok=True)
    page_f = open('data/wiki/neo4j/pages.csv', 'a')
    cat_f = open('data/wiki/neo4j/cats.csv', 'a')
    page_link_f = open('data/wiki/neo4j/page_links.csv', 'a')
    cat_link_f = open('data/wiki/neo4j/cat_links.csv', 'a')

    page_f.write(page_header + '\n')
    cat_f.write(cat_header + '\n')
    page_link_f.write(page_link_header + '\n')
    cat_link_f.write(cat_link_header + '\n')

    for trafficID, graphID in tqdm(trafficid2graphid.items()):
        page = db.pages.find_one({'i': trafficID})
        pageID = page['_id']
        title = page['title']
        if '"' in title:
            title = title.replace('"', '""')
            title = f'"{title}"'
        elif ',' in title:
            title = f'"{title}"'

        series = db.series.find_one({'_id': graphID})
        desktop = ';'.join([str(o) for o in series['d']['series']])
        app = ';'.join([str(o) for o in series['a']['series']])
        mobile = ';'.join([str(o) for o in series['m']['series']])
        assert series['d']['first_date'] == series['a']['first_date'] == series['m']['first_date']
        first_date = series['d']['first_date'].strftime("%Y-%m-%d")

        test = 'true' if graphID in test_ids else 'false'

        line_list = [str(graphID), str(trafficID), str(
            pageID), title, test, first_date, desktop, mobile, app]
        page_f.write(','.join(line_list) + '\n')

        for c in page['categories']:
            if 'e' in c['t'][-1] and c['t'][-1]['e'] < datetime(2015, 7, 1):
                continue
            if c['cat'] not in category_map:
                category_map[c['cat']] = len(category_map)

            cat_id = category_map[c['cat']]
            start_dates = []
            end_dates = []
            for t in c['t']:
                if 'e' in t and t['e'] < datetime(2015, 7, 1):
                    continue

                start_date = t['s']
                start_date = max(t['s'], datetime(2015, 7, 1))
                start_dates.append(start_date.strftime("%Y-%m-%d"))

                if 'e' in t:
                    end_dates.append(t['e'].strftime("%Y-%m-%d"))
                else:
                    end_dates.append('2020-06-30')

            if not start_dates:
                continue

            start_dates = ';'.join(start_dates)
            end_dates = ';'.join(end_dates)
            cat_link_list = [str(graphID), start_dates,
                             end_dates, str(cat_id)]
            cat_link_f.write(','.join(cat_link_list) + '\n')

        for link in page['links']:
            if link['n'] not in title2graphid:
                continue

            from_id = title2graphid[link['n']]
            start_dates = []
            end_dates = []
            for t in link['t']:
                if 'e' in t and t['e'] < datetime(2015, 7, 1):
                    continue

                start_date = t['s']
                start_date = max(t['s'], datetime(2015, 7, 1))
                start_dates.append(start_date.strftime("%Y-%m-%d"))

                if 'e' in t:
                    end_dates.append(t['e'].strftime("%Y-%m-%d"))
                else:
                    end_dates.append('2020-06-30')

            if not start_dates:
                continue

            start_dates = ';'.join(start_dates)
            end_dates = ';'.join(end_dates)
            page_link_list = [str(from_id), start_dates,
                              end_dates, str(graphID)]
            page_link_f.write(','.join(page_link_list) + '\n')

    for title, i in category_map.items():
        if '"' in title:
            title = title.replace('"', '""')
            title = f'"{title}"'
        elif ',' in title:
            title = f'"{title}"'
        cat_list = [str(i), title]
        cat_f.write(','.join(cat_list) + '\n')


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

    build_inputs(args['mongo'])


if __name__ == '__main__':
    main()
