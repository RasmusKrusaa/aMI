import networkx as net
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_graph():
    G = net.DiGraph()

    G.add_edge('Univ', 'ProfA')
    G.add_edge('Univ', 'ProfB')
    G.add_edge('ProfA', 'StudA')
    G.add_edge('StudA', 'Univ')
    G.add_edge('ProfB', 'StudB')
    G.add_edge('StudB', 'ProfB')

    return G

def create_bipartite_graph():
    G = net.DiGraph()

    G.add_edge('A', 'sugar')
    G.add_edge('A', 'frosting')
    G.add_edge('A', 'eggs')
    G.add_edge('B', 'frosting')
    G.add_edge('B', 'eggs')
    G.add_edge('B', 'flour')

    return G

def in_neighbors(G : net.DiGraph, node):
    res = []
    nodes = G.nodes

    for n in nodes:
        if G.has_edge(n, node):
            res.append(n)
    return res

def out_neighbors(G : net.DiGraph, node):
    res = []
    nodes = G.nodes

    for n in nodes:
        if G.has_edge(node, n):
            res.append(n)
    return res

def iterate(G : net.Graph, c : float = 0.8, steps : int = 10):
    nodes = G.nodes
    n_nodes = len(nodes)
    array = np.zeros((n_nodes, n_nodes))
    R = pd.DataFrame(data=array, index=G.nodes)
    R.columns = G.nodes

    # initialize before iterations start, i.e. R0
    for i in nodes:
        R.at[i, i] = 1

    # start of iterations
    for step in range(1, steps + 1):
        R_old = R.copy()

        for a in G.nodes:
            for b in G.nodes:
                if a == b: continue

                left_part = c / (len(in_neighbors(G, a)) * len(in_neighbors(G, b)))

                sum = 0
                for in_a in in_neighbors(G, a):
                    for in_b in in_neighbors(G, b):
                        sum += R_old.at[in_a, in_b]
                R.at[a, b] = left_part * sum

    return R

if __name__ == '__main__':
    G = create_graph()
    bipartite_G = create_bipartite_graph()

    R = iterate(G, steps=10)
    print(R)
    bipartite_R = iterate(bipartite_G, steps=5)
    print(bipartite_R)

    net.draw(G, pos = net.circular_layout(G), with_labels=True)
    plt.show()