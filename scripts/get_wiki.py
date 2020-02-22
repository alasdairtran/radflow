import re
from datetime import datetime, timedelta

import pandas as pd
import pymongo
import requests
from joblib import Parallel, delayed
from pymongo import MongoClient
from tqdm import tqdm

from nos.data.dataset_readers.wiki import extract_attributes
from nos.utils import setup_logger

logger = setup_logger()


def process_page(page, db):
    a = extract_attributes(page)
    scrape_page(db, a['agent'], a['source'], a['endpoint'],
                a['title'], a['lang'], a['domain'])


def scrape_page(db, agent, source, endpoint, title, lang, domain):
    params = {
        'title': title,
        'domain': domain,
        'lang': lang,
        'agent': agent,
        'source': source,
    }
    result = db.series.find_one(params)
    if result is not None:
        return

    series = get_traffic(domain, source, agent, title)

    rvstart = '2015-07-01T00:00:00Z'
    rvend = '2017-11-13T23:59:99Z'
    revisions = []
    first_rev = get_revision_just_before(params, endpoint, rvstart, db)
    if first_rev:
        revisions.append(first_rev)
    revisions += get_revisions_between(params, endpoint, rvstart, rvend, db)

    db.series.insert_one({
        **params,
        'series': series,
        'revisions': revisions,
    })


def get_revisions_between(params, endpoint, rvstart, rvend, db):
    params = {
        "action": "query",
        "prop": "revisions",
        "titles": params['title'],
        "rvprop": "timestamp|content",
        "rvdir": "newer",
        "rvstart": rvstart,
        "rvend": rvend,
        "formatversion": "2",
        "format": "json",
        "rvlimit": "max",
        "rvslots": "*",
    }

    revisions = []
    prev_ts = None
    while True:
        response = requests.get(url=endpoint, params=params).json()
        page = response["query"]["pages"][0]

        # No new revision during this time period
        if 'revisions' not in page:
            break

        for rev in page['revisions']:
            if 'content' not in rev['slots']['main']:
                continue
            content = rev['slots']['main']['content']
            timestamp = rev['timestamp']

            matches = re.findall(r'\[\[(.+?)\]\]', content)
            links = set()
            for match in matches:
                link = match.split('|')[0]
                links.add(link)
            links = sorted(links)

            r_params = {
                **params,
                'timestamp': datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S%z'),
            }

            # Keep only the latest revision for the day
            ts = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S%z')
            if prev_ts is not None and ts.day == prev_ts.day and ts.month == prev_ts.month and ts.year == prev_ts.year:
                db.revisions.delete_one({'_id': revision_id})
                revisions.pop()
            prev_ts = ts

            result = db.revisions.find_one(r_params)
            if result is not None:
                revision_id = result['_id']
            else:
                r_params['content'] = content
                r_params['outlinks'] = links
                result = db.revision.insert_one(r_params)
                revision_id = result.inserted_id

            revisions.append({
                'timestamp': timestamp,
                'revision_id': revision_id,
            })

        if 'continue' in response:
            params['continue'] = response['continue']['continue']
            params['rvcontinue'] = response['continue']['rvcontinue']
        else:
            break

    return revisions


def get_revision_just_before(params, endpoint, rvstart, db):
    params = {
        "action": "query",
        "prop": "revisions",
        "titles": params['title'],
        "rvprop": "timestamp|content",
        "rvdir": "older",
        "rvstart": rvstart,
        "formatversion": "2",
        "format": "json",
        "rvlimit": "1",
        "rvslots": "*",
    }

    response = requests.get(url=endpoint, params=params).json()
    page = response["query"]["pages"][0]

    if 'revisions' not in page:
        return None
    if 'content' not in page['revisions'][0]['slots']['main']:
        return None

    content = page['revisions'][0]['slots']['main']['content']
    timestamp = page['revisions'][0]['timestamp']

    matches = re.findall(r'\[\[(.+?)\]\]', content)
    links = set()
    for match in matches:
        link = match.split('|')[0]
        links.add(link)
    links = sorted(links)

    r_params = {
        **params,
        'timestamp': datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S%z'),
    }

    result = db.revisions.find_one(r_params)
    if result is not None:
        revision_id = result['_id']
    else:
        r_params['content'] = content
        r_params['outlinks'] = links
        result = db.revision.insert_one(r_params)
        revision_id = result.inserted_id

    revision = {
        'timestamp': timestamp,
        'revision_id': revision_id,
    }

    return revision


def get_traffic(domain, source, agent, title):
    # Reproduce the data collection process
    start = '2015070100'
    end = '2017111300'
    title = title.replace('/', r'%2F')
    url = f'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{domain}/{source}/{agent}/{title}/daily/{start}/{end}'

    response = requests.get(url).json()

    if 'items' not in response:
        print(response)
        return []

    response = response['items']

    if len(response) < 1:
        return [None] * 867
    first = response[0]['timestamp']
    last = response[-1]['timestamp']

    first = datetime.strptime(first, '%Y%m%d00')
    last = datetime.strptime(last, '%Y%m%d00')

    start_dt = datetime.strptime(start, '%Y%m%d00')
    end_dt = datetime.strptime(end, '%Y%m%d00')

    left_pad_size = (first - start_dt).days
    series = [None] * left_pad_size

    current_ts = first - timedelta(days=1)

    for o in response:
        ts = datetime.strptime(o['timestamp'], '%Y%m%d00')
        diff = (ts - current_ts).days
        if diff > 1:
            n_empty_days = diff - 1
            for _ in range(n_empty_days):
                series.append(None)
        else:
            assert diff == 1

        series.append(o['views'])
        current_ts = ts

    if end_dt != last:
        n_empty_days = (end_dt - last).days
        for _ in range(n_empty_days):
            series.append(None)
    assert len(series) == 867

    return series


def scrape_wiki():
    path = 'data/wiki/train_1.csv'
    logger.info(f'Reading data from {path}')
    df = pd.read_csv(path)

    client = MongoClient(host='localhost', port=27017)
    db = client.wiki

    # Only need to do this once. It's safe to run this multiple times. Mongo
    # will simply ignore the command if the index already exists. Index name is
    # lang_1_domain_1_agent_1_source_1_title_1
    db.series.create_index([
        ('lang', pymongo.ASCENDING),
        ('domain', pymongo.ASCENDING),
        ('agent', pymongo.ASCENDING),
        ('source', pymongo.ASCENDING),
        ('title', pymongo.ASCENDING),
    ], unique=True)

    db.revisions.create_index([
        ('lang', pymongo.ASCENDING),
        ('domain', pymongo.ASCENDING),
        ('agent', pymongo.ASCENDING),
        ('source', pymongo.ASCENDING),
        ('title', pymongo.ASCENDING),
        ('timestamp', pymongo.DESCENDING),
    ], unique=True)

    logger.info('Scraping from wiki.')
    with Parallel(n_jobs=24, backend='threading') as parallel:
        parallel(delayed(process_page)(page, db)
                 for page in tqdm(df['Page']))


def main():
    scrape_wiki()


if __name__ == '__main__':
    main()
