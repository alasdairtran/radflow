"""Download Wiki dump.

Usage:
    download_wikidump.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -d --dump DUMP      Dump dir.

"""

import calendar
import os
import urllib
from datetime import datetime

import requests
import wget
from bs4 import BeautifulSoup
from docopt import docopt
from schema import And, Or, Schema, Use
from tqdm import tqdm


def add_months(sourcedate, months):
    # From https://stackoverflow.com/a/4131114
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year, month)[1])
    return datetime(year, month, day)


def download_wiki(data_dir):
    # This takes a few days
    main_url = 'https://dumps.wikimedia.org/enwiki/20200701/'
    page = requests.get(main_url)
    assert page.status_code == 200

    soup = BeautifulSoup(page.content, 'html.parser')

    file_urls = []

    title_span = soup.find('span',
                           text='All pages with complete page edit history (.bz2)')
    for tag in title_span.parent():
        if tag.name == 'li':
            suffix = tag.find('a')['href']
            file_url = f'https://dumps.wikimedia.org/{suffix}'
            file_urls.append(file_url)

    os.makedirs(data_dir, exist_ok=True)

    for url in tqdm(file_urls):
        filename = url.split('/')[-1]
        out_path = os.path.join(data_dir, filename)

        if not os.path.exists(out_path):
            wget.download(url, out_path)


def download_pagecounts():
    # This takes half a day
    t = datetime(2011, 12, 1)
    end = datetime(2020, 2, 1)
    data_dir = '/data4/u4921817/radflow/data/pagecounts'
    os.makedirs(data_dir, exist_ok=True)

    while t < end:
        if t.year == 2016 and t.month in [7, 8, 9]:
            url = f'https://dumps.wikimedia.org/other/pagecounts-ez/merged/pagecounts-{t.year}-{t.month:02}.bz2'
        else:
            url = f'https://dumps.wikimedia.org/other/pagecounts-ez/merged/pagecounts-{t.year}-{t.month:02}-views-ge-5.bz2'

        filename = url.split('/')[-1]
        out_path = os.path.join(data_dir, filename)
        if not os.path.exists(out_path):
            try:
                wget.download(url, out_path)
            except urllib.error.HTTPError:
                # Let's try the non-compress version
                url = url[:-4]
                filename = url.split('/')[-1]
                out_path = os.path.join(data_dir, filename)
                if not os.path.exists(out_path):
                    try:
                        wget.download(url, out_path)
                    except urllib.error.HTTPError:
                        continue

        t = add_months(t, 1)


def validate(args):
    """Validate command line arguments."""
    args = {k.lstrip('-').lower().replace('-', '_'): v
            for k, v in args.items()}
    schema = Schema({
        'ptvsd': Or(None, And(Use(int), lambda port: 1 <= port <= 65535)),
        'dump': str,
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

    download_wiki(args['dump'])
    # download_pagecounts()


if __name__ == '__main__':
    main()
