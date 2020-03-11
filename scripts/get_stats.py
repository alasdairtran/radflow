import networkx as nx
import pandas as pd


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

    e = eccentricity(G)
    print(f'Diameter:', max(e.values()))


def get_vevo_stats():
    df = pd.read_csv('data/persistent_network.csv')
    G = nx.convert_matrix.from_pandas_edgelist(
        df, 'Source', 'Target', create_using=nx.DiGraph)

    get_stats(G)


def main():
    get_vevo_stats()


if __name__ == '__main__':
    main()
