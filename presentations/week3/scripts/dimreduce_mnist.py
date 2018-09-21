### IMPORTS ###

# Data import
from scipy.io import loadmat

# Numerical processing
import numpy as np

# Data compressions
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Plotting
import matplotlib.pyplot as plt


### CODE ###
def load_digits():
    f_x = loadmat('./data/digits-train.mat')
    xtrain = f_x['images_train'].transpose()
    xlabel = f_x['labels_train'].reshape(5000,)

    return xtrain, xlabel


def tsne_dim_reduce(data):
    # Project into PCA space and then embed in 2D using t-SNE
    pca = PCA(20)
    projected = pca.fit_transform(data)
    x_embedded = TSNE(n_components=2, learning_rate=150, perplexity=30,
                      angle=0.2, n_iter=500, verbose=2).fit_transform(projected)

    # Scale t-SNE_x and t-SNE_y
    tx, ty = x_embedded[:, 0], x_embedded[:, 1]
    tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

    return tx, ty


def pca_dim_reduce(data):
    pca = PCA(20)
    projected = pca.fit_transform(data)
    np.savetxt("mnist_pca_projection.csv", projected, delimiter=",")
    return projected


def plot_projection(x, y, label):
    plt.scatter(x, y, c=label, edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('nipy_spectral', 10))
    plt.show()


if __name__ == '__main__':
    xtrain, xlabel = load_digits()
    projected = pca_dim_reduce(xtrain)
    xlabels = np.savetxt("xlabel.csv", xlabel, delimiter=",")
    print(projected[:, 0:2])
    plot_projection(projected[:, 0], projected[:, 1], xlabel)
