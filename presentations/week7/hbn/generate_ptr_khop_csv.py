import pandas as pd
import networkx as nx
from graspy.utils.ptr import pass_to_ranks
from edge_fetch import edge_terrier


def weighted_khop_locality(G, filename):

    # Make ID
    subject_id = filename.split('_')[0].split('-')[1]
    print(subject_id)
    embed = [subject_id]

    # PTR graph
    ptr_G = pass_to_ranks(G)
    G = nx.from_numpy_matrix(ptr_G)

    for node in G.nodes:

        for k in [1, 2]:

            k_hop = list(nx.single_source_shortest_path_length(
                G, node, cutoff=k).keys())
            induced = nx.get_edge_attributes(G.subgraph(k_hop), 'weight')
            embed += [sum(induced.values())]

    if len(embed) == 97:
        return embed


if __name__=='__main__':

    f = edge_terrier()

    M = []
    for file in f.filelist:

        G, filename = f.convert_edgelist(file)

        if G is not None:
            embed = weighted_khop_locality(G, filename)
        if embed is not None:
            M.append(embed)

    # Convert to df
    df = pd.DataFrame.from_records(M)
    df.columns = ['EID'] + list(range(96))

    # Get phenotypic data
    phenotypic = pd.read_csv('phenotypic.csv').loc[:,['EID', 'Sex', 'Age']]

    # Make single df
    df = pd.merge(phenotypic, df, how='inner', on=['EID'])
    df.columns = ['id', 'sex', 'age'] + list(range(96))
    df['sex'] = df['sex'].map({'F': 1, 'M': 0})

    # Save to csv
    df.to_csv('weighted_klocality_with_age-sex.csv', index=False)
