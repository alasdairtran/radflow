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
    --reindex           Reindex the page ID.

"""
import bz2
import fileinput
import html
import json
import os
import pickle
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
    db = client.wiki2

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


def resolve_ambiguity(db, title, id1, id2, title2id, id2title):
    url1 = f'https://en.wikipedia.org/w/api.php?action=query&prop=info&pageids={id1}&inprop=url&format=json'
    url2 = f'https://en.wikipedia.org/w/api.php?action=query&prop=info&pageids={id2}&inprop=url&format=json'
    res1 = requests.get(url1).json()['query']['pages'][str(id1)]
    res2 = requests.get(url2).json()['query']['pages'][str(id2)]

    if 'missing' in res1:
        title2 = res2['title']
        logger.info(f"Title: {title} -> {title2}, remove {id1}, keep {id2}")
        db.pages.delete_one({'_id': id1})
        db.pages.update_one({'_id': id2}, {'$set': {'title': title2}})
        del id2title[id1]
        del title2id[title]
        title2id[title2] = id2
        id2title[id2] = title2

    elif 'missing' in res2:
        title1 = res1['title']
        logger.info(f"Title: {title} -> {title1}, remove {id2}, keep {id1}")
        db.pages.delete_one({'_id': id2})
        db.pages.update_one({'_id': id1}, {'$set': {'title': title1}})
        del title2id[title]
        title2id[title1] = id1
        id2title[id1] = title1

    else:
        title1 = res1['title']
        title2 = res2['title']

        logger.info(f"Title: {title} -> {title1}, keep {id1}")
        logger.info(f"Title: {title} -> {title2}, keep {id2}")

        assert title1 != title2 and title in [title1, title2]

        db.pages.update_one({'_id': id1}, {'$set': {'title': title1}})
        db.pages.update_one({'_id': id2}, {'$set': {'title': title2}})

        title2id[title1] = id1
        title2id[title2] = id2

        id2title[id1] = title1
        id2title[id2] = title2


def fix_duplicates(db, index_path):
    id2title = {}
    title2id = {}

    # Note that these articles have the same title but different page IDs
    # Title: Children of the Moon -> Children of the Moon, remove 5443592, keep 55611314
    # Title: Simultaneous interpretation -> Simultaneous interpretation, remove 6707331, keep 60027184
    # Title: Max Hunter -> Max Hunter, remove 45359352, keep 64057525
    # Title: Essent N.V. -> Essent, keep 7324758
    # Title: Essent N.V. -> Essent N.V., keep 64168198
    # Title: Chibi-Robo! Zip Lash -> Chibi-Robo! Zip Lash (video game), keep 46988924
    # Title: Chibi-Robo! Zip Lash -> Chibi-Robo! Zip Lash, keep 64356807
    # Title: No Pressure -> No Pressure (Erick Sermon album), keep 7354786
    # Title: No Pressure -> No Pressure, keep 64364969
    # Title: Kimberly Jones -> Kimberly Jones, remove 2182650, keep 64386907
    # Title: Michael Hecht -> Michael Hecht, remove 64435049, keep 991517
    # Title: List of patricides -> Patricide, keep 578580
    # Title: List of patricides -> List of patricides, keep 64435981
    # Title: Otok, Vukovar-Srijem County -> Otok, Vukovar-Syrmia County, keep 1526279
    # Title: Otok, Vukovar-Srijem County -> Otok, Vukovar-Srijem County, keep 64436191
    # Title: Ka Leo O Hawaii -> Ka Leo O Hawaiʻi, keep 2063589
    # Title: Ka Leo O Hawaii -> Ka Leo O Hawaii, keep 64436348
    # Title: University of Hawaii Maui College -> University of Hawaiʻi Maui College, keep 770165
    # Title: University of Hawaii Maui College -> University of Hawaii Maui College, keep 64436368
    # Title: University of Hawaii–West Oahu -> University of Hawaiʻi – West Oʻahu, keep 767176
    # Title: University of Hawaii–West Oahu -> University of Hawaii–West Oahu, keep 64436430
    # Title: Kapiolani Community College -> Kapiʻolani Community College, keep 769380
    # Title: Kapiolani Community College -> Kapiolani Community College, keep 64436458
    # Title: Kauai Community College -> Kauaʻi Community College, keep 770154
    # Title: Kauai Community College -> Kauai Community College, keep 64436461
    # Title: Waikiki Aquarium -> Waikīkī Aquarium, keep 764819
    # Title: Waikiki Aquarium -> Waikiki Aquarium, keep 64436480
    # Title: Haleakala Observatory -> Haleakalā Observatory, keep 1526755
    # Title: Haleakala Observatory -> Haleakala Observatory, keep 64436484
    # Title: Limestone College -> Limestone University, keep 2165207
    # Title: Limestone College -> Limestone College, keep 64437109
    # Title: Quadrics -> Quadrics (company), keep 259109
    # Title: Quadrics -> Quadrics, keep 64437467
    # Title: Flossenbürg -> Flossenbürg, Bavaria, keep 360881
    # Title: Flossenbürg -> Flossenbürg, keep 64437896
    # Title: Todd Martinez -> Todd Martínez, keep 6774439
    # Title: Todd Martinez -> Todd Martinez, keep 64437916
    # Title: Geminiano T. de Ocampo -> Geminiano de Ocampo, keep 5304445
    # Title: Geminiano T. de Ocampo -> Geminiano T. de Ocampo, keep 64438043
    # Title: Additional Protocol II -> Protocol II, keep 1953425
    # Title: Additional Protocol II -> Additional Protocol II, keep 64438069
    # Title: Ophelia Alcantara Dimalanta -> Ophelia Dimalanta, keep 12550037
    # Title: Ophelia Alcantara Dimalanta -> Ophelia Alcantara Dimalanta, keep 64438237
    # Title: Red Line (Cleveland) -> Red Line (RTA Rapid Transit), keep 3254444
    # Title: Red Line (Cleveland) -> Red Line (Cleveland), keep 64438361
    # Title: Blue Line (Cleveland) -> Blue Line (RTA Rapid Transit), keep 3253785
    # Title: Blue Line (Cleveland) -> Blue Line (Cleveland), keep 64438364
    # Title: Green Line (Cleveland) -> Green Line (RTA Rapid Transit), keep 3253787
    # Title: Green Line (Cleveland) -> Green Line (Cleveland), keep 64438367
    # Title: Dachau (disambiguation) -> Dachau (disambiguation), remove 64438510, keep 8518155
    # Title: Toyama (city) -> Toyama, Toyama, keep 6792777
    # Title: Toyama (city) -> Toyama (city), keep 64438616
    # Title: 1985 (Anthony Burgess novel) -> 1985 (Burgess novel), keep 2235536
    # Title: 1985 (Anthony Burgess novel) -> 1985 (Anthony Burgess novel), keep 64438752
    # Title: Heavy Metal: a Tank Company's Battle to Baghdad -> Heavy Metal: A Tank Company's Battle to Baghdad, keep 1880633
    # Title: Heavy Metal: a Tank Company's Battle to Baghdad -> Heavy Metal: a Tank Company's Battle to Baghdad, keep 64438775
    # Title: Never Call Retreat: Lee and Grant: The Final Victory -> Never Call Retreat, keep 10003526
    # Title: Never Call Retreat: Lee and Grant: The Final Victory -> Never Call Retreat: Lee and Grant: The Final Victory, keep 64438785
    # Title: Hubris: The Inside Story of Spin, Scandal, and the Selling of the Iraq War -> Hubris (book), keep 7367152
    # Title: Hubris: The Inside Story of Spin, Scandal, and the Selling of the Iraq War -> Hubris: The Inside Story of Spin, Scandal, and the Selling of the Iraq War, keep 64438823
    # Title: They Marched Into Sunlight -> They Marched into Sunlight, keep 9755591
    # Title: They Marched Into Sunlight -> They Marched Into Sunlight, keep 64438840
    # Title: For God and Country (James Yee) -> For God and Country (Yee book), keep 10011965
    # Title: For God and Country (James Yee) -> For God and Country (James Yee), keep 64438845
    # Title: The General (C. S. Forester novel) -> The General (Forester novel), keep 198506
    # Title: The General (C. S. Forester novel) -> The General (C. S. Forester novel), keep 64438870
    # Title: To Ruhleben – And Back -> To Ruhleben – and Back, keep 4602991
    # Title: To Ruhleben – And Back -> To Ruhleben – And Back, keep 64438877
    # Title: The Left was Never Right -> The Left Was Never Right, keep 7536287
    # Title: The Left was Never Right -> The Left was Never Right, keep 64438881
    # Title: Peacemakers: The Paris Peace Conference of 1919 and Its Attempt to End War -> Peacemakers (book), keep 245520
    # Title: Peacemakers: The Paris Peace Conference of 1919 and Its Attempt to End War -> Peacemakers: The Paris Peace Conference of 1919 and Its Attempt to End War, keep 64438887
    # Title: For Us, the Living: A Comedy of Customs -> For Us, the Living, keep 406609
    # Title: For Us, the Living: A Comedy of Customs -> For Us, the Living: A Comedy of Customs, keep 64438912
    # Title: The Snow Goose: A Story of Dunkirk -> The Snow Goose (novella), keep 7206031
    # Title: The Snow Goose: A Story of Dunkirk -> The Snow Goose: A Story of Dunkirk, keep 64438919
    # Title: The Sunflower: On the Possibilities and Limits of Forgiveness -> The Sunflower (book), keep 14577792
    # Title: The Sunflower: On the Possibilities and Limits of Forgiveness -> The Sunflower: On the Possibilities and Limits of Forgiveness, keep 64438926
    # Title: Once (Morris Gleitzman novel) -> Once (Gleitzman novel), keep 12256255
    # Title: Once (Morris Gleitzman novel) -> Once (Morris Gleitzman novel), keep 64438930
    # Title: Super Buddies -> Formerly Known as the Justice League, keep 2221988
    # Title: Super Buddies -> Super Buddies, keep 64439024
    # Title: Air Guitar -> Air guitar (disambiguation), keep 3896605
    # Title: Air Guitar -> Air Guitar, keep 64439054
    # Title: Death of Benno Ohnesorg -> Shooting of Benno Ohnesorg, keep 493649
    # Title: Death of Benno Ohnesorg -> Death of Benno Ohnesorg, keep 64439074
    # Title: University of Hawaii at Manoa -> University of Hawaii at Manoa, keep 64436236
    # Title: University of Hawaii at Manoa -> University of Hawaiʻi at Mānoa, keep 646743
    # Title: Middle Child -> Middle Child, remove 20244502, keep 59734056
    # Title: Waikiki -> Waikiki, keep 64436502
    # Title: Waikiki -> Waikīkī, keep 59649
    # Title: Mahatma Jyotiba Phule Mandai -> Mahatma Jyotiba Phule Mandai, keep 64434775
    # Title: Mahatma Jyotiba Phule Mandai -> Crawford Market, keep 907894
    # Title: Israel's Border Wars 1949–1956 -> Israel's Border Wars 1949–1956, keep 64438835
    # Title: Israel's Border Wars 1949–1956 -> Israel's Border Wars, 1949–1956, keep 470454

    logger.info('Building title-id index map')
    for p in tqdm(db.pages.find({}, projection=['_id', 'title'])):
        title = p['title']
        if title in title2id and title2id[title] != p['_id']:
            resolve_ambiguity(db, title, title2id[title], p['_id'],
                              title2id, id2title)
        else:
            title2id[title] = p['_id']
            assert p['_id'] not in title2id
            id2title[p['_id']] = title

    # Another pass. We manually remove IDs pointing to these missing pages:
    # 406610 For Us, the Living
    # 10003539 Never Call Retreat
    # 565953 Formerly Known as the Justice League
    # 15918058 The Sunflower (book)
    ids = set(title2id.values())
    delete_ids = []
    for i, title in id2title.items():
        if i not in ids:
            delete_ids.append(i)

    for i in delete_ids:
        del id2title[i]

    # Statistics: 17,380,550 unique titles/IDs.
    logger.info(f"Number of IDs: {len(id2title)}")
    logger.info(f"Number of Titles: {len(title2id)}")
    logger.info(f"Saving cached maps to {index_path}")
    with open(index_path, 'wb') as f:
        pickle.dump([title2id, id2title], f)


def reindex(host):

    client = MongoClient(host=host, port=27017)
    db = client.wiki2
    db.pages.create_index([
        ('i', pymongo.ASCENDING),
    ])

    index_path = os.path.join('data/wiki', 'title2pageid.pkl')
    if not os.path.exists(index_path):
        fix_duplicates(db, index_path)

    count = 0
    logger.info('Reindexing page IDs')
    for p in tqdm(db.pages.find({}, projection=['_id']).sort('_id', pymongo.ASCENDING)):
        db.pages.update_one({'_id': p['_id']}, {'$set': {'i': count}})
        count += 1


def extract_wiki_graph(host, split, n_jobs, total, dump_dir):
    client = MongoClient(host=host, port=27017)
    db = client.wiki2

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
        'dump': Or(None, os.path.exists),
        'host': Or(None, str),
        'split': Or(None, Use(int)),
        'n_jobs': Use(int),
        'total': Or(None, Use(int)),
        'reindex': bool,
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

    if args['dump']:
        extract_wiki_graph(args['host'], args['split'],
                           args['n_jobs'], args['total'], args['dump'])

    if args['reindex']:
        reindex(args['host'])


if __name__ == '__main__':
    main()
