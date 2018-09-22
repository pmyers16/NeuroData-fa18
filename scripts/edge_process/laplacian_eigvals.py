### IMPORTS ###

# 3rd party
from networkx import normalized_laplacian_matrix
import numpy as np

# Local
import sys
path = '/Users/vivek/Documents/GitHub/ndd-2018/scripts'
if sys.path[0] != path:
    sys.path.insert(0, path)
print(path)
from edge_process.edge_fetch import edge_terrier


### CODE ###
def _get_edgelists(filepath):
    # Initialize an object with all processed edgelists
    graph_collector = edge_terrier(filepath)
    return graph_collector


def _calculate_eigenvalues(G):
    # Return eigenvalues of the Laplacian
    L = normalized_laplacian_matrix(G)
    e = np.linalg.eigvals(L.A)
    return e


def _feature_select(graph_collector):

    # Get generator
    edgelists = graph_collector.convert_edgelist_all()

    # Calculate eigenvalues for each graph
    eigvals = [_calculate_eigenvalues(G[0]) for G in edgelists]
    filelist = [_calculate_eigenvalues(G[1]) for G in edgelists]
    len_eig = [len(row) for row in eigvals]
    cutoff = len_eig[np.argmax(np.bincount(len_eig))]

    # Preserve only the necessary features
    X = []
    for row in eigvals:
        if len(row) >= cutoff:
            X.append(row[-cutoff:])

    return X


def calc_eigval_feature_matrix(filepath='hbn/derivatives/graphs/JHU/'):
    graph_collector = _get_edgelists(filepath)
    X = _feature_select(graph_collector)
    return X
