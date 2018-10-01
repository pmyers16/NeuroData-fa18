### IMPORTS ###
# 3rd party inputs
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Local
from laplacian_eigvals import calc_eigval_feature_matrix


### CODE ###
def main():

    # Load data matrix
    X = calc_eigval_feature_matrix()

    # Create PCA object and perform operation
    print('starting pca...')
    pca = PCA(n_components=2)
    projected = pca.fit_transform(X)
    plt.scatter(projected[:,0], projected[:,1])
    plt.show()


if __name__=='__main__':
    main()
