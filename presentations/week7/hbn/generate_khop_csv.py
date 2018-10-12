import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Personal function for loading edgelists off the S3 bucket
from edge_fetch import edge_terrier


def khop_locality(G, filename):

    subject_id = filename.split('_')[0]
    embed = [subject_id]

    for node in G.nodes:

        one_hop = list(nx.single_source_shortest_path_length(G, node, cutoff=1).keys())
        two_hop = list(nx.single_source_shortest_path_length(G, node, cutoff=2).keys())

        embed += len(G.subgraph(one_hop).edges()), len(G.subgraph(two_hop).edges())

    if len(embed) == 97:
        return embed

# Main
if __name__ == '__main__':

    f = edge_terrier()

    # Place embeddings
    M = []
    for file in f.filelist:
        G, filename = f.convert_edgelist(file)

        if G is not None:
            embed = khop_locality(G, filename)
        if embed is not None:
            M.append(embed)

    # Save to csv
    df = pd.DataFrame.from_records(M)
    df.columns = ['id'] + list(range(96))
    df.to_csv('khop.csv', index=False)
