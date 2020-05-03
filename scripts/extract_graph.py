"""Annotate Good News with parts of speech.

Usage:
    extract_graph.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -h --host HOST      MongoDB host [default: localhost].
    -i --order ORDER    Order.
    -d --dump DUMP      Dump dir.
    -o --out OUT        Output dir.

"""
import bz2
import fileinput
import json
import os
import re
import time
from datetime import datetime
from glob import glob
from multiprocessing import Manager

import ptvsd
import pymongo
import pytz
from bs4 import BeautifulSoup
from bz2file import BZ2File
from docopt import docopt
from joblib import Parallel, delayed
from pymongo import MongoClient
from schema import And, Or, Schema, Use
from tqdm import tqdm

from nos.utils import setup_logger

logger = setup_logger()

tagRE = re.compile(r'(.*?)<(/?\w+)[^>]*?>(?:([^<]*)(<.*?>)?)?')
#                    1     2               3      4
keyRE = re.compile(r'key="(\d*)"')
# capture the category name [[Category:Category name|Sortkey]]"
catRE = re.compile(r'\[\[Category:([^\|]+).*\]\].*')


def get_new_page():
    return {
        '_id': None,
        'title': None,
        'ns': None,
        'redirect': None,
        'links': {},
        'categories': {},
    }


def get_new_revision_page():
    return {
        '_id': None,
        'parent_id': None,
        'timestamp': None,
        'text': [],
        'categories': set(),
    }


def process_revision(revision, page):
    # We store only the links to save on memory
    # Each line automatically ends with \n so so need to separate out the lines
    content = ''.join(revision['text'])
    matches = re.findall(r'\[\[(.+?)\]\]', content)
    links = page['links']
    current_links = set()
    for i, match in enumerate(matches):
        link = match.split('|')[0]

        # First time seeing the link
        if link not in links:
            links[link] = [{
                'link': link,
                'start': revision['timestamp'],
                'order': i}
            ]

        # If the link already exists and but has been cut off, then start a
        # new connection
        elif 'end' in links[link][-1]:
            # Avoid noise such as vandalism
            now = datetime.strptime(
                revision['timestamp'], '%Y-%m-%dT%H:%M:%S%z')
            prev = datetime.strptime(
                links[link][-1]['end'], '%Y-%m-%dT%H:%M:%S%z')

            if (now - prev).days == 0:
                del links[link][-1]['end']
            else:
                links[link].append({
                    'link': link,
                    'start': revision['timestamp'],
                    'order': i,
                })

        # Else the link already exists but no end date yet. No need to do
        # anything.
        current_links.add(link)

    # Now check if any edges have been removed
    for k, v in links.items():
        if k not in current_links and 'end' not in v[-1]:
            v[-1]['end'] = revision['timestamp']

    # We now do the same thing with categories
    categories = page['categories']
    for cat in revision['categories']:
        # First time seeing the category
        if cat not in categories:
            categories[cat] = [{'cat': cat, 'start': revision['timestamp']}]

        # If the category already exists and but has been cut off, then start a
        # new connection
        elif 'end' in categories[cat][-1]:
            # Avoid noise such as vandalism
            now = datetime.strptime(
                revision['timestamp'], '%Y-%m-%dT%H:%M:%S%z')
            prev = datetime.strptime(
                categories[cat][-1]['end'], '%Y-%m-%dT%H:%M:%S%z')

            if (now - prev).days == 0:
                del categories[cat][-1]['end']
            else:
                categories[cat].append({
                    'cat': cat, 'start': revision['timestamp']})

        # Else the category already exists but no end date yet. No need to do
        # anything.

    # Now check if any edges have been removed
    for k, v in categories.items():
        if k not in revision['categories'] and 'end' not in v[-1]:
            v[-1]['end'] = revision['timestamp']


def pages_from(f):
    """
    Scans input extracting pages.
    :return: (id, revid, title, namespace key, page), page is a list of lines.
    """
    # we collect individual lines, since str.join() is significantly faster
    # than concatenation
    in_page = False
    in_rev = False
    in_text = False
    page = get_new_page()
    rev_page = get_new_revision_page()
    ignore_page = False

    for line in f:
        # Convert bytes to str if neccessary
        if not isinstance(line, str):
            line = line.decode('utf-8')

        # If a tag is definitely not in this line
        # faster than doing re.search()
        if '<' not in line and not ignore_page:
            if in_text:
                rev_page['text'].append(line)
                # extract categories
                if line.lstrip().startswith('[[Category:'):
                    mCat = catRE.search(line)
                    if mCat:
                        rev_page['categories'].add(mCat.group(1))
            continue

        # If we're here, the line potentially contains a tag
        m = tagRE.search(line)

        # Hmm, this should never happen?
        if not m:
            continue

        # Extract the tag
        # The whole dump is in <mediawiki> tag.
        # On the next level we have <siteinfo>, which contains metadata.
        # Let's ignore this.
        # After <siteinfo>, we have the <page> tag. Everything about
        # a page (including revisions) is in here.
        tag = m.group(2)

        # Inside <page> there are 5 children:
        #   <title>
        #   <ns>
        #   <id>
        #   <redirect>
        #   <revision>
        if tag == 'page':
            page = get_new_page()
            in_page = True
            in_rev = False
            in_text = False
            ignore_page = False

        elif tag == '/page':
            if page['_id'] is not None and not ignore_page:
                yield page

            page = get_new_page()
            in_page = False
            in_rev = False
            in_text = False
            ignore_page = False

        elif ignore_page:
            continue

        # Start of a revision
        elif tag == 'revision':
            rev_page = get_new_revision_page()
            in_rev = True

        # End of a revision
        elif tag == '/revision':
            process_revision(rev_page, page)

            rev_page = get_new_revision_page()
            in_rev = False

        elif in_page and not in_rev:
            if tag == 'id' and not page['_id']:
                page['_id'] = int(m.group(3))

                # Check if we've already processed this page
                # result = db.pages.find_one(
                #     {'_id': page['_id']}, projection=['_id'])
                # if result is not None:
                #     ignore_page = True

            elif tag == 'title' and not page['title']:
                page['title'] = m.group(3)

            elif tag == 'ns' and not page['ns']:
                page['ns'] = m.group(3)
                if page['ns'] != '0':
                    ignore_page = True

            elif tag == 'redirect':
                soup = BeautifulSoup(line, 'html.parser')
                page['redirect'] = soup.redirect['title']

        elif in_page and in_rev:
            # There could also be id inside contributor tag
            if tag == 'id' and not rev_page['_id']:
                rev_page['_id'] = int(m.group(3))

            elif tag == 'parentid' and not rev_page['parent_id']:
                rev_page['parent_id'] = int(m.group(3))

            elif tag == 'timestamp' and not rev_page['timestamp']:
                rev_page['timestamp'] = m.group(3)
                # rev_page['timestamp'] = datetime.strptime(
                #     m.group(3), '%Y-%m-%dT%H:%M:%S%z')

            elif tag == 'text':
                # self closing
                if m.lastindex == 3 and line[m.start(3)-2] == '/':
                    # <text xml:space="preserve" />
                    continue
                in_text = True
                line = line[m.start(3):m.end(3)]
                rev_page['text'].append(line)
                if m.lastindex == 4:  # open-close
                    in_text = False

            elif tag == '/text':
                if m.group(1):
                    rev_page['text'].append(m.group(1))
                in_text = False

            elif in_text:
                rev_page['text'].append(line)

            # Ignore all other tags, e.g. <contributor>, <comment>, <format>, <sha1>


def process_path(path, result_dir):
    # client = MongoClient(host=host, port=27017)
    # db = client.wiki
    filename = os.path.basename(path)
    print(f'Reading {filename}')

    # file = fileinput.FileInput(path, openhook=fileinput.hook_compressed)
    # result = db.archives.find_one({'_id': filename}, projection=['_id'])
    # if result is not None:
    #     return

    # with open(path, 'rb') as f:
    #     data = f.read()

    # logger.info('Decompressing...', end=' ')
    # s = time.time()
    # data = bz2.decompress(data)
    # e = time.time() - s
    # logger.info(f'{e/60:.1} minutes')
    start_time = time.time()
    result_path = f'{result_dir}/{filename}.jsonl'
    if os.path.exists(result_path):
        os.remove(result_path)
    assert not os.path.exists(result_path)

    with BZ2File(path) as fin:
        with open(result_path, 'a') as fout:
            for page in pages_from(fin):
                # Reshape data because keys in mongo can't contain special chars

                page['links'] = [v for k, v in page['links'].items()]
                page['categories'] = [v for k, v in page['categories'].items()]

                # Insert page into database. We've already checked that it's not
                # in the database before.
                # db.pages.insert_one(page)

                fout.write(f'{json.dumps(page)}\n')

    # db.archives.insert_one({
    #     '_id': filename,
    #     'elapsed': time.time() - start_time,
    #     'finished': datetime.now(pytz.utc),
    # })

    with open(f'{result_dir}/archives.jsonl', 'a') as f:
        out = {
            '_id': filename,
            'elapsed': time.time() - start_time,
            'finished': str(datetime.now(pytz.utc)),
        }
        f.write(f'{json.dumps(out)}\n')

    # client.close()


def extract_wiki_graph(host, order, dump_dir, result_dir):
    # client = MongoClient(host=host, port=27017)
    # db = client.wiki

    # Only need to do this once. It's safe to run this multiple times. Mongo
    # will simply ignore the command if the index already exists. Index name is
    # ....
    # db.pages.create_index([
    #     ('cats', pymongo.ASCENDING),
    #     ('title', pymongo.ASCENDING),
    # ])

    # db.revisions.create_index([
    #     ('timestamp', pymongo.DESCENDING),
    # ])
    # client.close()

    paths = glob(f'{dump_dir}/*.bz2')
    paths.sort(key=os.path.getmtime)

    start = order * 2
    end = start + 2

    paths = paths[start:end]

    with Parallel(n_jobs=2, backend='loky') as parallel:
        parallel(delayed(process_path)(path, result_dir)
                 for i, path in enumerate(paths))


def validate(args):
    """Validate command line arguments."""
    args = {k.lstrip('-').lower().replace('-', '_'): v
            for k, v in args.items()}
    schema = Schema({
        'ptvsd': Or(None, And(Use(int), lambda port: 1 <= port <= 65535)),
        'host': str,
        'order': Use(int),
        'dump': os.path.exists,
        'out': os.path.exists,
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

    extract_wiki_graph(args['host'], args['order'],
                       args['dump'], args['out'])


if __name__ == '__main__':
    main()
