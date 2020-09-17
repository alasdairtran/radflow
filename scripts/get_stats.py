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


def get_bowtie_stats(G, views):
    lscc = None
    lscc_size = 0
    for comp in nx.strongly_connected_components(G):
        if len(comp) > lscc_size:
            lscc_size = len(comp)
            lscc = comp

    in_set = set()
    for source, dest in G.in_edges(lscc):
        if source not in lscc:
            in_set.add(source)

    out_set = set()
    for source, dest in G.out_edges(lscc):
        if dest not in lscc:
            out_set.add(dest)

    tendrils = set()
    for source, dest in G.out_edges(in_set):
        if dest not in in_set and dest not in lscc and dest not in out_set:
            tendrils.add(dest)

    for source, dest in G.in_edges(out_set):
        if source not in in_set and source not in lscc and source not in out_set:
            tendrils.add(dest)

    dis_set = set()
    connected_set = in_set | out_set | tendrils | lscc
    for n in G.nodes:
        if n not in connected_set:
            dis_set.add(n)

    total_views = views.sum()

    def print_stats(nodes, G, views):
        count_pct = len(nodes) / len(G)
        node_views = views[np.array(sorted(nodes))].sum()
        views_pct = node_views / total_views

        return f'{count_pct:.1%} & {views_pct:.1%}'.replace('%', r'\%')

    print(f"LSCC: {print_stats(lscc, G, views)}")
    print(f"IN: {print_stats(in_set, G, views)}")
    print(f"OUT: {print_stats(out_set, G, views)}")
    print(f"Tendrils: {print_stats(tendrils, G, views)}")
    print(f"Disconnected: {print_stats(dis_set, G, views)}")


def get_vevo_bowtie_stats():
    path = "data/vevo/static_graph_2018_10_01.gpickle"
    f = h5py.File('data/vevo/vevo.hdf5', 'r')

    # Day 30 is 1 oct 2018
    day = 30
    if not os.path.exists(path):
        edges = f['edges']
        G = nx.DiGraph()
        for target_id, neighs in tqdm(enumerate(edges)):
            if len(neighs[day]) == 0:
                G.add_node(target_id)
            for n in neighs[day]:
                if n != -1:
                    G.add_edge(n, target_id)
        nx.write_gpickle(G, path)

    G = nx.read_gpickle(path)
    views = f['views'][:, day]

    print('Stats for Vevo graph')
    get_bowtie_stats(G, views)


def get_wiki_bowtie_stats():
    path = "data/wiki/static_graph_2018_10_01.gpickle"
    f = h5py.File('data/wiki/wiki.hdf5.delete', 'r')

    # Day 1,188 is 1 oct 2018
    day = 1188

    t2i_path = "data/wiki/title2graphid.pkl"
    with open(t2i_path, 'rb') as f_t2i:
        t2i = pickle.load(f_t2i)

    cat_ids = set([i for t, i in t2i.items() if t.startswith('Category:')])
    list_ids = set([i for t, i in t2i.items() if t.startswith('List of')])
    blacklist = cat_ids | list_ids

    # This takes 45 minutes
    if not os.path.exists(path):
        edges = f['edges']
        G = nx.DiGraph()
        for target_id, neighs in tqdm(enumerate(edges)):
            if target_id in blacklist:
                continue
            ns = neighs[day][neighs[day] != -1]
            if len(ns) == 0:
                G.add_node(target_id)
            for n in ns:
                if n != -1 and n not in blacklist:
                    G.add_edge(n, target_id)
        nx.write_gpickle(G, path)

    G = nx.read_gpickle(path)
    views = f['views'][:, day]

    print('Stats for Wiki graph')
    get_bowtie_stats(G, views)


def main():
    # get_wiki_bivariate_nbeats_stats()
    # get_vevo_stats()
    # get_wiki_stats()

    # get_vevo_bowtie_stats()
    get_wiki_bowtie_stats()


if __name__ == '__main__':
    main()
