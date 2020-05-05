import ast
import html
import json
import os
import pickle
from datetime import datetime
from glob import glob

import bs4
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import requests
from bz2file import BZ2File
from pymongo import MongoClient
from tqdm import tqdm


def resolve_ambiguity(title, id1, id2, title2id, id2title):
    url1 = f'https://en.wikipedia.org/w/api.php?action=query&prop=info&pageids={id1}&inprop=url&format=json'
    url2 = f'https://en.wikipedia.org/w/api.php?action=query&prop=info&pageids={id2}&inprop=url&format=json'
    res1 = requests.get(url1).json()['query']['pages'][str(id1)]
    res2 = requests.get(url2).json()['query']['pages'][str(id2)]

    if 'missing' in res1:
        title2 = res2['title']
        title2id[title2] = id2
        id2title[id2] = title2
    elif 'missing' in res2:
        title1 = res1['title']
        title2id[title1] = id1
        id2title[id1] = title1
    else:
        title1 = res1['title']
        title2 = res2['title']

        assert title1 != title2 and title in [title1, title2]

        title2id[title1] = id1
        title2id[title2] = id2

        id2title[id1] = title1
        id2title[id2] = title2


def extract_edges(index_path):

    # Next step: compress the graph into numbers
    title2id = {}
    id2title = {}

    # We loop through the whole thing once to get build the page id index
    paths = glob(
        '/localdata/u4921817/projects/nos/data/wiki/results/*.bz2.jsonl')

    # Note that these articles have the same title but different page IDs
    # Anjan Chowdhury 21606610 62911656
    # XMultiply 5103829 62998806
    # Daily Mashriq 4737576 63000119
    # Zeewijk 5931503 63000552
    # Amanieu VI 10037159 63000573
    # Amanieu VIII 10037418 63000585
    # Bernard Ezi IV 10038254 63000589
    # Arnaud Amanieu, Lord of Albret 10038002 63000655
    # You Si-kun 62998113 349072
    # James Coyne 62999621 520170
    # Air New Zealand Flight 901 62998459 1158000
    # FIVB Senior World and Continental Rankings 63000600 1463363
    paths.sort()
    # This takes 12 minutes
    for path in tqdm(paths):
        with open(path) as f:
            for line in f:
                o = json.loads(line)
                title = html.unescape(o['title'])

                if title in title2id and title2id[title] != o['_id']:
                    resolve_ambiguity(
                        title, title2id[title], o['_id'], title2id, id2title)
                title2id[title] = o['_id']

                assert o['_id'] not in title2id
                id2title[o['_id']] = title

    # Statistics: 15,026,614 unique titles and 15,026,618 unique IDs

    with open(index_path, 'wb') as f:
        pickle.dump([title2id, id2title], f)


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

    namespaces = set(['User', 'Wikipedia', 'File', 'MediaWiki', 'Template', 'Help',
                      'Category', 'Portal', 'Draft', 'TimedText', 'Module', 'Image',
                      'Talk', 'Media', 'WP', 'Special', 'Wiktionary'])

    lower_namespaces = set([n.lower() for n in namespaces])

    prefixes = codes | namespaces | lower_namespaces

    return prefixes


def construct_graph(id2title, title2id, f_edges):
    # G = nx.DiGraph()
    # for id_, title in tqdm(id2title.items()):
    #     G.add_node(id_, title=title)

    prefixes = get_prefixes()

    # Next we reindex the processed dumps
    paths = glob(
        '/localdata/u4921817/projects/nos/data/wiki/results/*.bz2.jsonl')
    paths.sort()

    for path in tqdm(paths):
        with open(path) as f:
            for line in f:
                o = json.loads(line)
                from_id = o['_id']
                redirect_id = None
                # If there is a redirection, we assume it's redirected
                # throughout the whole period
                if o['redirect']:
                    redirect_title = html.unescape(o['redirect'])
                    # It is possible that the redirect title is not
                    # a valid redirect, e.g. Orla Brady redirecting Portal:Ukraine
                    # If that's the case, we shall
                    # just ignore it and see if there are any
                    # valid links inside the page
                    if redirect_title in title2id:
                        redirect_id = title2id[redirect_title]
                        timestamp = ['2000-01-01T00:00:00Z',
                                     '2030-01-01T00:00:00Z']
                        attr = {'ts': timestamp}
                        # timestamps = [
                        #     datetime(2000, 1, 1), datetime(2030, 1, 1)]
                        # G.add_edge(from_id, redirect_id, ts=timestamps)
                        f_edges.write(f'{from_id} {redirect_id} {str(attr)}\n')
                        continue

                for link in o['links']:
                    # First let's check if the link actually exists
                    p = link[0]
                    to_title = html.unescape(p['link'])

                    # Remove excess spaces
                    to_title = to_title.strip()

                    # Ignore empty titles
                    if not to_title:
                        continue

                    parts = to_title.split(':')
                    if len(parts) >= 2 and parts[0] in prefixes:
                        continue

                    # Capitalize first letter of title
                    # Remove trailing hashes (subsections)
                    if to_title not in title2id:
                        to_title = to_title[0].upper() + to_title[1:]
                        to_title = to_title.split('#')[0]

                    # Attempt 2: Convert underscores to spaces
                    if to_title not in title2id:
                        to_title = to_title.replace('_', ' ')

                    if to_title not in title2id:
                        continue

                    # If we reach here, the title exists!
                    to_id = title2id[to_title]

                    # Extract the timestamps
                    timestamps = []
                    for p in link:
                        start = p['start']
                        # start = datetime.strptime(
                        #     p['start'], '%Y-%m-%dT%H:%M:%S%z')
                        if 'end' in p:
                            end = p['end']
                            # end = datetime.strptime(
                            #     p['end'], '%Y-%m-%dT%H:%M:%S%z')
                        else:
                            end = '2030-01-01T00:00:00Z'
                            # end = datetime(2030, 1, 1)
                        timestamps.append((start, end))
                    # G.add_edge(from_id, to_id, ts=timestamps)
                    attr = {'ts': timestamps}
                    f_edges.write(f'{from_id} {to_id} {str(attr)}\n')

    # nx.write_gpickle(G, graph_path, pickle.HIGHEST_PROTOCOL)


START = datetime(2011, 12, 1)


def compress_dates(pairs):
    new_pairs = []
    if isinstance(pairs[0], str):
        pairs = [pairs]

    for pair in pairs:
        # Number of days since 2011-12-01
        start, end = pair
        start = datetime.strptime(start[:10], '%Y-%m-%d')
        end = datetime.strptime(end[:10], '%Y-%m-%d')

        start = (start - START).days
        end = (end - START).days

        if end < 0:
            continue
        elif start < 0:
            start = 0

        end = min(end, 3400)

        new_pairs.append((start, end))

    return new_pairs


def compress_edges(fin, fout):
    for line in tqdm(fin):
        parts = line.split()
        from_id = parts[0]
        to_id = parts[1]
        ts = ast.literal_eval(' '.join(parts[2:]))['ts']

        attr = {
            'ts': compress_dates(ts)
        }

        if not attr['ts']:
            continue

        fout.write(f'{from_id} {to_id} {str(attr)}\n')


def main():
    os.makedirs('data/wiki', exist_ok=True)
    index_path = os.path.join('data/wiki', 'title_index.pkl')
    if not os.path.exists(index_path):
        extract_edges(index_path)

    edge_path = os.path.join('data/wiki', 'edges.txt')
    if not os.path.exists(edge_path):
        with open(index_path, 'rb') as f:
            title2id, id2title = pickle.load(f)
        with open(edge_path, 'a') as f_edges:
            construct_graph(id2title, title2id, f_edges)

    compressed_edge_path = os.path.join('data/wiki', 'edges_int.txt')
    if not os.path.exists(compressed_edge_path):
        with open(edge_path) as fin:
            with open(compressed_edge_path, 'a') as fout:
                compress_edges(fin, fout)


if __name__ == '__main__':
    main()
