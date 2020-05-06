"""Annotate Good News with parts of speech.

Usage:
    extract_graph.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -h --host HOST      MongoDB host.
    -s --split SPLIT    Order.
    -d --dump DUMP      Dump dir.
    -n --n-jobs INT     Number of jobs [default: 39].
    -t --total INT      Total number of jobs.

"""
import bz2
import fileinput
import html
import json
import os
import random
import re
import time
from datetime import datetime
from glob import glob

import bs4
import ptvsd
import pymongo
import pytz
import requests
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
        'cats': [],
    }


def get_new_revision_page():
    return {
        '_id': None,
        'parent_id': None,
        'timestamp': None,
        'text': [],
        'categories': set(),
    }


def get_prefixes():
    res = requests.get(
        'https://meta.wikimedia.org/wiki/Template:List_of_language_names_ordered_by_code')
    soup = bs4.BeautifulSoup(res.content, 'html.parser')
    table = soup.find('table', class_='wikitable').find('tbody')
    codes = set()
    for tr in table.find_all('tr')[1:]:
        if tr.find('td'):
            code = tr.find('td').text.strip()
            codes.add(code)

    namespaces = set(['User', 'User talk', 'File', 'File talk',
                      'MediaWiki', 'MediaWiki talk', 'Template', 'Template talk',
                      'Help', 'Help talk', 'Draft', 'Draft talk',
                      'TimedText', 'TimedText talk', 'Module', 'Module talk',
                      'Image', 'Image talk', 'Talk', 'Media', 'Special',
                      'Wiktionary', 'Wiktionary talk', 'Category talk',
                      'Wikipedia', 'WP', 'Wikipedia talk', 'Portal talk'])

    lower_namespaces = set([n.lower() for n in namespaces])

    prefixes = codes | namespaces | lower_namespaces
    prefixes = set([p + ':' for p in prefixes])

    return prefixes


def clean_up_page(page):
    # Remove any links that are present for less than 12 hours.
    delete_keys = set()
    for k, v in page['links'].items():
        for i in reversed(range(len(v['t']))):
            ts = v['t'][i]
            if 'e' not in ts:
                continue
            start = ts['s']
            end = ts['e']
            diff = end - start if end > start else start - end
            if diff.days == 0 and diff.seconds < 43200:  # 12 hours
                del v['t'][i]

        if not v['t']:
            delete_keys.add(k)

    for k in delete_keys:
        del page['links'][k]

    # Remove any categories that are present for less than 12 hours.
    delete_cats = set()
    for k, v in page['categories'].items():
        for i in reversed(range(len(v['t']))):
            ts = v['t'][i]
            if 'e' not in ts:
                page['cats'].append(k)
                continue
            start = ts['s']
            end = ts['e']
            diff = end - start if end > start else start - end
            if diff.days == 0 and diff.seconds < 43200:  # 12 hours
                del v['t'][i]

        if not v['t']:
            delete_cats.add(k)

    for k in delete_cats:
        del page['categories'][k]

    page['cats'] = sorted(page['cats'])


def process_revision(revision, page, prefixes):
    # We store only the links to save on memory
    # Each line automatically ends with \n so so need to separate out the lines
    content = ''.join(revision['text'])
    matches = re.findall(r'\[\[(.+?)\]\]', content)
    current_links = set()
    for i, match in enumerate(matches):
        link = html.unescape(match.split('|')[0])

        # See https://en.wikipedia.org/wiki/Help:Colon_trick
        if link.startswith(':'):
            link = link[1:]

        has_prefix = False
        for prefix in prefixes:
            if link.startswith(prefix):
                has_prefix = True
                break
        if has_prefix:
            continue

        # First time seeing the link
        if link not in page['links']:
            page['links'][link] = {
                'n': link,
                't': [{'i': i, 's': revision['timestamp']}],
            }

        # If the link already exists and but has been cut off, then start a
        # new connection
        elif 'e' in page['links'][link]['t'][-1]:
            # Avoid noise such as vandalism
            now = revision['timestamp']
            prev = page['links'][link]['t'][-1]['e']

            # Very rarely, the next start time might be a few milliseconds
            # behind the previous end time!
            if (now - prev).days == 0 or (prev - now).days == 0:
                del page['links'][link]['t'][-1]['e']
            else:
                page['links'][link]['t'].append({
                    'i': i,
                    's': revision['timestamp'],
                })

        # Else the link already exists but no end date yet. No need to do
        # anything.
        current_links.add(link)

    # Now check if any edges have been removed
    for k, v in page['links'].items():
        if k not in current_links and 'e' not in v['t'][-1]:
            v['t'][-1]['e'] = revision['timestamp']

    # We now do the same thing with categories
    categories = page['categories']
    for cat in revision['categories']:
        # First time seeing the category
        if cat not in categories:
            categories[cat] = {
                'cat': cat,
                't': [{'s': revision['timestamp']}],
            }

        # If the category already exists and but has been cut off, then start a
        # new connection
        elif 'e' in categories[cat]['t'][-1]:
            # Avoid noise such as vandalism
            now = revision['timestamp']
            prev = categories[cat]['t'][-1]['e']

            if (now - prev).days == 0 or (prev - now).days == 0:
                del categories[cat]['t'][-1]['e']
            else:
                categories[cat]['t'].append({'s': revision['timestamp']})

        # Else the category already exists but no end date yet. No need to do
        # anything.

    # Now check if any edges have been removed
    for k, v in categories.items():
        if k not in revision['categories'] and 'e' not in v['t'][-1]:
            v['t'][-1]['e'] = revision['timestamp']


def pages_from(f, db, prefixes):
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
        # Quick short circuit. I had a bottleneck before, which could come
        # either from the tag regex search, or the conversion from bytes to str
        # I wasted so many days of compute time on this. Should've done some
        # profiling right from the beginning :-/
        # Each dump file should only take 1 hour on average to go through on one CPU.
        # Even the largest dump file only takes 5h, the second largest 2h.
        if not isinstance(line, str) and b'<' not in line and ignore_page:
            continue

        # Convert bytes to str if neccessary
        line = line.decode('utf-8')

        # If a tag is definitely not in this line
        # faster than doing re.search()
        if '<' not in line:
            if in_text:
                rev_page['text'].append(line)
                # extract categories
                if line.lstrip().startswith('[[Category:'):
                    mCat = catRE.search(line)
                    if mCat:
                        rev_page['categories'].add(
                            html.unescape(mCat.group(1)))
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
                clean_up_page(page)
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
            process_revision(rev_page, page, prefixes)

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
                page['title'] = html.unescape(m.group(3))

            elif tag == 'ns' and not page['ns']:
                page['ns'] = m.group(3)
                # 0 is Main, 14 is Category, 100 is Portal
                if page['ns'] not in ['0', '14', '100']:
                    ignore_page = True

            elif tag == 'redirect':
                soup = BeautifulSoup(line, 'html.parser')
                page['redirect'] = html.unescape(soup.redirect['title'])

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


def convert_to_datetime_str(page):
    for key in ['links', 'categories']:
        for obj in page[key]:
            for ts in obj['t']:
                ts['s'] = ts['s'].strftime('%Y-%m-%dT%H:%M:%S%z')
                if 'e' in ts:
                    ts['e'] = ts['e'].strftime('%Y-%m-%dT%H:%M:%S%z')


def process_path(path, host, prefixes):
    time.sleep(random.uniform(1, 5))
    client = MongoClient(host=host, port=27017)
    db = client.wiki

    filename = os.path.basename(path)

    result = db.archives.find_one({'_id': filename}, projection=['_id'])
    if result is not None:
        return

    start_time = time.time()
    with BZ2File(path) as fin:
        for i, page in enumerate(pages_from(fin, db, prefixes)):
            # Reshape data because keys in mongo can't contain special chars
            page['links'] = [v for k, v in page['links'].items()]
            page['categories'] = [v for k, v in page['categories'].items()]

            # Insert page into database. We've already checked that it's not
            # in the database before.
            try:
                db.pages.insert_one(page)
            except pymongo.errors.DocumentTooLarge:
                convert_to_datetime_str(page)
                with open('extract_graph.err', 'a') as fout:
                    fout.write(f'{json.dumps(page)}\n')

    db.archives.insert_one({
        '_id': filename,
        'elapsed': (time.time() - start_time) / 3600,
        'finished': datetime.now(pytz.utc),
        'size': os.stat(path).st_size / 1000000000,
        'n_articles': i,
    })
    client.close()


def extract_wiki_graph(host, split, n_jobs, total, dump_dir):
    client = MongoClient(host=host, port=27017)
    db = client.wiki

    # Only need to do this once. It's safe to run this multiple times.
    # Mongo will simply ignore the command if the index already exists.
    # Index name is ....
    db.pages.create_index([
        ('ns', pymongo.ASCENDING),
    ])

    db.pages.create_index([
        ('cats', pymongo.ASCENDING),
    ])

    db.pages.create_index([
        ('title', pymongo.ASCENDING),
    ])

    client.close()

    paths = glob(f'{dump_dir}/*.bz2')
    paths.sort()
    # Shuffling ensures equal distribution among machines. In turns out that
    # if we sort the dumps, the first half takes 37h, while the second half
    # takes 22h using 39 cores.
    random.Random(2318).shuffle(paths)

    if split is not None:
        start = split * total
        end = start + total
        paths = paths[start:end]

    prefixes = get_prefixes()

    with Parallel(n_jobs=n_jobs, backend='loky') as parallel:
        parallel(delayed(process_path)(path, host, prefixes.copy())
                 for i, path in tqdm(enumerate(paths)))


def validate(args):
    """Validate command line arguments."""
    args = {k.lstrip('-').lower().replace('-', '_'): v
            for k, v in args.items()}
    schema = Schema({
        'ptvsd': Or(None, And(Use(int), lambda port: 1 <= port <= 65535)),
        'dump': os.path.exists,
        'host': Or(None, str),
        'split': Or(None, Use(int)),
        'n_jobs': Use(int),
        'total': Or(None, Use(int)),
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

    extract_wiki_graph(args['host'], args['split'],
                       args['n_jobs'], args['total'], args['dump'])


if __name__ == '__main__':
    main()
