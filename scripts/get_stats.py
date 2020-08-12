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
    order = G.order()

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

    # e = eccentricity(G)
    # print(f'Diameter:', max(e.values()))


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

    get_stats(G2)


def main():
    get_vevo_stats()
    get_wiki_stats()


if __name__ == '__main__':
    main()
