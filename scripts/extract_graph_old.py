"""Annotate Good News with parts of speech.

Usage:
    extract_graph.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -h --host HOST      MongoDB host [default: localhost].
    -s --start INT      Start index [default: 0].
    -e --end INT        End index.
    -n --n_jobs INT     Number of jobs [default: 36].

"""
import bz2
import fileinput
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
        'redirect': False,
        'redirect_title': None,
        'revisions': [],
        'revision_links': [],
        'cats': set(),
    }


def get_new_revision_page():
    return {
        '_id': None,
        'parent_id': None,
        'timestamp': None,
        'text': [],
        'links': None,
        'cats': set(),
    }


def process_revision(rev_page, revisions):
    # We store only the links to save on memory
    # Each line automatically ends with \n so so need to separate out the lines
    content = ''.join(rev_page['text'])
    matches = re.findall(r'\[\[(.+?)\]\]', content)
    links = set()
    for match in matches:
        link = match.split('|')[0]
        links.add(link)
    sorted_links = sorted(links)

    rev_page['links'] = sorted_links
    del rev_page['text']

    # Merge links with previous revision if on the same day
    if revisions:
        ts_1 = revisions[-1]['timestamp']
        ts_2 = rev_page['timestamp']
        if ts_1.year == ts_2.year and ts_1.month == ts_2.month and ts_1.day == ts_2.day:
            merged_links = links | set(revisions[-1]['links'])
            revisions[-1]['links'] = sorted(merged_links)
            revisions[-1]['cats'] |= rev_page['cats']
            return False

    return True


def pages_from(f, db):
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
    last_id = None
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
                        rev_page['cats'].add(mCat.group(1))
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
            should_add = process_revision(rev_page, page['revision_links'])
            if should_add:
                page['revision_links'].append(rev_page)
                page['revisions'].append({
                    'timestamp': rev_page['timestamp'],
                    'revision_id': rev_page['_id'],
                })
            page['cats'] |= rev_page['cats']

            rev_page = get_new_revision_page()
            in_rev = False

        elif in_page and not in_rev:
            if tag == 'id' and not page['_id']:
                page['_id'] = int(m.group(3))

                # Check if we've already processed this page
                result = db.pages.find_one(
                    {'_id': page['_id']}, projection=['_id'])
                if result is not None:
                    ignore_page = True

            elif tag == 'title' and not page['title']:
                page['title'] = m.group(3)

            elif tag == 'ns' and not page['ns']:
                page['ns'] = m.group(3)
                if page['ns'] != '0':
                    ignore_page = True

            elif tag == 'redirect':
                page['redirect'] = True
                soup = BeautifulSoup(line, 'html.parser')
                page['redirect_title'] = soup.redirect['title']

        elif in_page and in_rev:
            # There could also be id inside contributor tag
            if tag == 'id' and not rev_page['_id']:
                rev_page['_id'] = int(m.group(3))

            elif tag == 'parentid' and not rev_page['parent_id']:
                rev_page['parent_id'] = int(m.group(3))

            elif tag == 'timestamp' and not rev_page['timestamp']:
                rev_page['timestamp'] = datetime.strptime(
                    m.group(3), '%Y-%m-%dT%H:%M:%S%z')

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


def process_path(path, host, queue):
    client = MongoClient(host=host, port=27017)
    db = client.wiki
    filename = os.path.basename(path)
    # logger.info(f'Reading {filename}')

    # file = fileinput.FileInput(path, openhook=fileinput.hook_compressed)
    result = db.archives.find_one({'_id': filename}, projection=['_id'])
    if result is not None:
        return

    # with open(path, 'rb') as f:
    #     data = f.read()

    # logger.info('Decompressing...', end=' ')
    # s = time.time()
    # data = bz2.decompress(data)
    # e = time.time() - s
    # logger.info(f'{e/60:.1} minutes')
    start_time = time.time()
    position = queue.pop()
    with BZ2File(path) as f:
        for page in tqdm(pages_from(f, db), position=position, desc=filename[34:]):
            page['cats'] = sorted(page['cats'])

            # Insert revision links into database
            for rev in page['revision_links']:
                del rev['cats']
                result = db.revisions.find_one(
                    {'_id': rev['_id']}, projection=['_id'])
                if result is None:
                    db.revisions.insert_one(rev)
            del page['revision_links']

            # Insert page into database. We've already checked that it's not
            # in the database before.
            db.pages.insert_one(page)

    db.archives.insert_one({
        '_id': filename,
        'elapsed': time.time() - start_time,
        'finished': datetime.now(pytz.utc),
    })

    client.close()
    queue.append(position)


def extract_wiki_graph(host, start, end, n_jobs):
    client = MongoClient(host=host, port=27017)
    db = client.wiki

    # Only need to do this once. It's safe to run this multiple times. Mongo
    # will simply ignore the command if the index already exists. Index name is
    # ....
    db.pages.create_index([
        ('cats', pymongo.ASCENDING),
        ('title', pymongo.ASCENDING),
    ])

    db.revisions.create_index([
        ('timestamp', pymongo.DESCENDING),
    ])
    client.close()

    paths = glob('/data4/u4921817/nos/data/wikidump/*.bz2')

    # Assume we downloaded the files from oldest to newest
    paths.sort(key=os.path.getmtime)
    paths = paths[start:end]

    logger.info('Scraping from wiki dump.')
    # for path in paths:
    #     logger.info(f'Scraping from {os.path.basename(path)}')
    #     process_path(path, db)
    manager = Manager()
    queue = manager.list(range(n_jobs, 0, -1))
    with Parallel(n_jobs=n_jobs, backend='loky') as parallel:
        parallel(delayed(process_path)(path, host, queue)
                 for i, path in enumerate(paths))


def validate(args):
    """Validate command line arguments."""
    args = {k.lstrip('-').lower().replace('-', '_'): v
            for k, v in args.items()}
    schema = Schema({
        'ptvsd': Or(None, And(Use(int), lambda port: 1 <= port <= 65535)),
        'host': str,
        'start': Use(int),
        'end': Or(None, Use(int)),
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

    extract_wiki_graph(args['host'], args['start'],
                       args['end'], args['n_jobs'])


if __name__ == '__main__':
    main()
