"""Visualize subgraphs.

Usage:
    visualize_subgraphs.py [options] SEED

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -d --data_dir DIR   Data directory [default: data/wiki/subgraphs].
    -f --fig_dir DIR    Figure directory [default: figures].
    SEED                Seed word.
"""

import json
import os
import pickle
from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import ptvsd
from docopt import docopt
from matplotlib.pyplot import setp
from schema import And, Or, Schema, Use
from tqdm import tqdm

from wordcloud import STOPWORDS, WordCloud


def generate_word_cloud(titles, plot_title, word_cloud_path):
    titles = [t.lower() for t in titles]
    wordcloud = WordCloud(
        background_color='white',
        stopwords=STOPWORDS,
        max_words=200,
        max_font_size=40,
        scale=3,
        random_state=1,
    ).generate(' '.join(titles))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if plot_title:
        fig.suptitle(plot_title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    fig.tight_layout()
    fig.savefig(word_cloud_path)


def plot_avg_daily_traffic(series, topic, series_path):
    n_days = len(next(iter(series.values())))
    views = np.zeros(n_days)
    for v in series.values():
        views += np.array(v)
    views /= len(series)

    fig = plt.figure(figsize=(20, 6))
    ax = plt.subplot(1, 1, 1)

    start = datetime(2015, 7, 1)
    end = datetime(2020, 2, 1)  # exclude endpoint
    days = mdates.drange(start, end, timedelta(days=1))

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=90))

    ax.plot(days, views, label=topic)
    fig.autofmt_xdate()

    ax.set_title(f"{topic}: Average Daily Traffic in a Year")
    ax.set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(series_path)


def get_subgraph_stats(in_degrees):
    G = nx.DiGraph()
    for node, neighours in in_degrees.items():
        for neigh in neighours:
            G.add_edge(neigh['id'], node)

    get_stats_from_graph(G)


def get_stats_from_graph(G):
    print(f'Number of nodes with edges:', len(G))

    in_degrees = []
    for n, deg in G.in_degree(None, weight=None):
        in_degrees.append(deg)
    avg_in_degree = sum(in_degrees) / len(in_degrees)
    print(f'Average in-degree of nodes with edges:', avg_in_degree)

    out_degrees = []
    for n, deg in G.out_degree(None, weight=None):
        out_degrees.append(deg)
    avg_out_degree = sum(out_degrees) / len(out_degrees)
    print(f'Average out-degree of nodes with edges:', avg_out_degree)

    in_degrees = []
    for n, deg in G.in_degree(None, weight=None):
        if deg > 0:
            in_degrees.append(deg)
    avg_in_degree = sum(in_degrees) / len(in_degrees)
    print(f'Number of nodes with incoming edges', len(in_degrees))
    print(f'Average in-degree of nodes with incoming edges:', avg_in_degree)

    in_degrees = []
    for n, deg in G.in_degree(None, weight=None):
        in_neighbours = [pair[0] for pair in G.in_edges(n)]
        neighbour_degrees = G.in_degree(in_neighbours)
        in_degrees.append(sum(dict(neighbour_degrees).values()))
    avg_in_degree = sum(in_degrees) / len(in_degrees)
    print(f'Average 2-hop in-degree of nodes with edges:', avg_in_degree)

    in_degrees = []
    for n, deg in G.in_degree(None, weight=None):
        in_neighbours = [pair[0] for pair in G.in_edges(n)]
        neighbour_degrees = G.in_degree(in_neighbours)
        d = sum(dict(neighbour_degrees).values())
        if d > 0:
            in_degrees.append(d)
    avg_in_degree = sum(in_degrees) / len(in_degrees)
    print(f'Average 2-hop in-degree of nodes with incoming edges:', avg_in_degree)


def visualize(seed_word, data_dir, fig_dir):
    graph_dir = os.path.join(data_dir, seed_word)
    fig_dir = os.path.join(fig_dir, seed_word)
    os.makedirs(fig_dir, exist_ok=True)

    title2id_path = os.path.join(graph_dir, 'title2id.pkl')
    with open(title2id_path, 'rb') as f:
        title2id = pickle.load(f)
    print(f'Number of central nodes: {len(title2id)}')

    neightitle2id_path = os.path.join(graph_dir, 'neightitle2id.pkl')
    with open(neightitle2id_path, 'rb') as f:
        neightitle2id = pickle.load(f)
    print(f'Number of neighbours: {len(neightitle2id)}')

    word_cloud_path = os.path.join(fig_dir, 'word_cloud_main.png')
    title = f'{seed_word} ({len(title2id)} seed nodes)'
    generate_word_cloud(title2id, title, word_cloud_path)

    word_cloud_path = os.path.join(fig_dir, 'word_cloud_neighs.png')
    title = f'{seed_word} ({len(neightitle2id)} neighbours)'
    generate_word_cloud(neightitle2id, title, word_cloud_path)

    series_path = os.path.join(graph_dir, 'series.pkl')
    with open(series_path, 'rb') as f:
        series = pickle.load(f)

    series_plot_path = os.path.join(fig_dir, 'daily_traffic.png')
    plot_avg_daily_traffic(series, seed_word, series_plot_path)

    in_degrees_path = os.path.join(graph_dir, 'in_degrees.pkl')
    with open(in_degrees_path, 'rb') as f:
        in_degrees = pickle.load(f)

    get_subgraph_stats(in_degrees)


def validate(args):
    """Validate command line arguments."""
    args = {k.lstrip('-').lower().replace('-', '_'): v
            for k, v in args.items()}
    schema = Schema({
        'ptvsd': Or(None, And(Use(int), lambda port: 1 <= port <= 65535)),
        'seed': str,
        'data_dir': os.path.exists,
        'fig_dir': str,
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

    visualize(args['seed'], args['data_dir'], args['fig_dir'])


if __name__ == '__main__':
    main()
