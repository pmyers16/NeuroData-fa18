import numpy as np
import matplotlib.pyplot as plt

from dimreduce_mnist import load_digits, tsne_dim_reduce



def input_data():
    mnist_projection = np.genfromtxt('mnist_pca_projection.csv', delimiter=',')
    xlabel = np.genfromtxt('xlabel.csv', delimiter=',')
    return mnist_projection, xlabel


def plotting(x, y, label):

    data, labels = load_digits()
    tx, ty = tsne_dim_reduce(data)

    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False)
    fig.set_size_inches(18.5, 10.5)

    ax[0].scatter(tx, ty, c=label, edgecolor='none', alpha=0.5,
                  cmap=plt.cm.get_cmap('nipy_spectral', 10))
    ax[1].scatter(x, y, c=label, edgecolor='none', alpha=0.5,
                  cmap=plt.cm.get_cmap('nipy_spectral', 10))

    plt.savefig('../groundtruth.png')
    plt.show()


if __name__ == '__main__':
    mnist_projection, xlabel = input_data()
    x, y = mnist_projection[:, 0], mnist_projection[:, 1]
    plotting(x, y, xlabel)
