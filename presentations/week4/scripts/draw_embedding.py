from load_embedding import emb

import matplotlib
import matplotlib.pyplot as plt
import requests
import numpy as np
import networkx as nx


def convert_edgelist(filename='hbn/derivatives/graphs/JHU/sub-NDARAA536PTU_acq-64dir_dwi_JHU.edgelist'):

    # Fetch edgelist
    link = 'http://neurodatadesign.s3.amazonaws.com/' + filename
    edges = requests.get(link).text.split()
    edges = np.array([int(x) for x in edges])
    edges = [tuple(edges[x:x + 3]) for x in range(0, len(edges), 3)]

    if edges == []:
        print(filename + ' is empty.')
        return

    # Convert edgelist to networkx object
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    adj = nx.to_numpy_matrix(G)
    return np.diagonal(adj).tolist()


def main():
    filename = 'sub-NDARAA536PTU_acq-64dir_dwi_JHU.emb'
    filepath = '/Users/vivek/Documents/GitHub/NDD/test/struc2vec/emb/'

    # Load embedding
    emb1 = emb(filepath + filename)
    weights = convert_edgelist()

    cmap = matplotlib.cm.get_cmap('viridis')
    normalize = matplotlib.colors.Normalize(vmin=min(weights), vmax=max(weights))
    colors = [cmap(normalize(value)) for value in weights]

    print(colors)

    # Make plot
    fig, ax = plt.subplots()
    ax.set_title(filename.split('.')[0])
    ax.scatter(x=emb1.data[:, 1], y=emb1.data[:, 2], label=emb1.filename, color=colors, alpha=0.75)

    cax, _ = matplotlib.colorbar.make_axes(ax)
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)


    plt.savefig('./figures/{}-embedding.png'.format(filename.split('.')[0]), dpi=500)
    plt.show()

if __name__ == '__main__':
    main()
