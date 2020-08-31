import json
import os
import pickle

import h5py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm


def eccentricity(G, v=None, sp=None):
    """Return the eccentricity of nodes in G.

    The eccentricity of a node v is the maximum distance from v to
    all other nodes in G.

    Parameters
    ----------
    G : NetworkX graph
       A graph

    v : node, optional
       Return value of specified node

    sp : dict of dicts, optional
       All pairs shortest path lengths as a dictionary of dictionaries

    Returns
    -------
    ecc : dictionary
       A dictionary of eccentricity values keyed by node.
    """
    e = {}
    for n in G.nbunch_iter(v):
        length = nx.single_source_shortest_path_length(G, n)
        e[n] = max(length.values())

    if v in G:
        return e[v]  # return single value
    else:
        return e


def get_stats(G):
    print(f'Number of nodes with edges:', len(G))
    print(f'Number of edges:', G.number_of_edges())

    in_degrees = []
    for n, deg in G.in_degree(None, weight=None):
        if deg > 0:
            in_degrees.append(deg)
    in_degrees = np.array(in_degrees)
    avg_in_degree = in_degrees.mean()
    med_in_degree = np.median(in_degrees)
    print(f'Number of nodes with incoming edges', len(in_degrees))
    print(f'    Average in-degree:', avg_in_degree)
    print(f'    Median in-degree:', med_in_degree)

    out_degrees = []
    for n, deg in G.out_degree(None, weight=None):
        if deg > 0:
            out_degrees.append(deg)
    out_degrees = np.array(out_degrees)
    avg_out_degree = out_degrees.mean()
    med_out_degree = np.median(out_degrees)
    print(f'Number of nodes with outgoing edges', len(out_degrees))
    print(f'    Average in-degree:', avg_out_degree)
    print(f'    Median in-degree:', med_out_degree)

    # e = eccentricity(G)
    # print(f'Diameter:', max(e.values()))

    print()


def plot_degree_dist(G, topic):
    outs = sorted([d for n, d in G.out_degree()], reverse=True)
    ins = sorted([d for n, d in G.in_degree()], reverse=True)

    fig = plt.figure(figsize=(20, 6))

    ax = plt.subplot(1, 2, 1)
    ax.hist(outs, bins=100)
    ax.set_title(f"{topic} Out-degree Histogram")
    ax.set_ylabel("Count")
    ax.set_xlabel("Degree")

    ax = plt.subplot(1, 2, 2)
    ax.hist(ins, bins=100)
    ax.set_title(f"{topic} In-degree Histogram")
    ax.set_ylabel("Count")
    ax.set_xlabel("Degree")

    fig.savefig(f'{topic}_dist.png')


def get_vevo_stats():
    path = "data/vevo/static_graph.gpickle"

    if not os.path.exists(path):
        f = h5py.File('data/vevo/vevo.hdf5', 'r')
        edges = f['edges']
        G = nx.DiGraph()
        for target_id, neighs in tqdm(enumerate(edges)):
            for n in np.unique(np.vstack(neighs)):
                if n != -1:
                    G.add_edge(n, target_id)
        nx.write_gpickle(G, path)

    G1 = nx.read_gpickle(path)

    print('Stats for Vevo graph')
    get_stats(G1)


def get_wiki_stats():
    path = "data/wiki/static_graph.gpickle"
    if not os.path.exists(path):
        f2 = h5py.File('data/wiki/wiki.hdf5', 'r')
        edges2 = f2['edges']
        G2 = nx.DiGraph()
        for target_id, neighs in tqdm(enumerate(edges2)):
            for n in np.unique(np.vstack(neighs)):
                if n != -1:
                    G2.add_edge(n, target_id)

        nx.write_gpickle(G2, path)

    G2 = nx.read_gpickle(path)

    print('Stats for Wiki graph')
    get_stats(G2)


def get_wiki_bivariate_nbeats_stats():
    path = 'expt/pure_time_series/wiki_bivariate/07_nbeats_desktop/serialization/evaluate-metrics.json'
    with open(path) as f:
        o3 = json.load(f)

    path = 'expt/pure_time_series/wiki_bivariate/07_nbeats_mobile/serialization/evaluate-metrics.json'
    with open(path) as f:
        o8 = json.load(f)

    out_dir = 'expt/pure_time_series/wiki_bivariate/07_nbeats/serialization/'
    os.makedirs(out_dir, exist_ok=True)
    path = f'{out_dir}/evaluate-metrics.json'

    s1 = np.array(o3['smapes'])
    s2 = np.array(o8['smapes'])
    s = np.stack([s1, s2], axis=-1)

    o = {'smapes': s.tolist()}

    with open(path, 'w') as f:
        json.dump(o, f)

    print('Wiki Bivariate NBEATS SMAPE-7 Mean:', s.mean())


def main():
    get_wiki_bivariate_nbeats_stats()
    get_vevo_stats()
    get_wiki_stats()


if __name__ == '__main__':
    main()
