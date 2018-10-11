import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from pprint import pprint
from edge_fetch import edge_terrier

# Find the 1-hop and 2-hop neighbors
def khop_locality(G):
    data = []

    for i in G.nodes:

        one_hops = []
        two_hops = []

        for key, value in nx.single_source_shortest_path_length(G, i, cutoff=2).items():

            if value == 1:
                one_hops.append(int(key))
            elif value == 2:
                two_hops.append(int(key))

        data.append([i, len(one_hops), len(two_hops)+len(one_hops)])

    data = np.array(data)
    data = data[data[:,0].argsort()]
    return data



f = edge_terrier()
edgelists = f.convert_edgelist_all()



for G in edgelists:
    data = khop_locality(G)
    
    plt.scatter([line[1] for line in data],
            [line[2] for line in data],
            alpha=0.75
           )
    plt.show()
