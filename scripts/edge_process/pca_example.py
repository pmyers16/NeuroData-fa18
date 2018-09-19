### IMPORTS ###
# 3rd parts
import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Local
from get_eigenvalues import calc_eigval_feature_matrix


### CODE ###
def main():

    # Load data matrix
    X = calc_eigval_feature_matrix()

    # Create PCA object and perform operation
    pca = PCA(n_components=2)
    projected = pca.fit_transform(X)
    plt.scatter(projected[:,0], projected[:,1])
    plt.show()


if __name__=='__main__':
    main()
