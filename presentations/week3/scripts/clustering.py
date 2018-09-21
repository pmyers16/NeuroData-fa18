### IMPORTS ###
# Timing
import time

# Data structures and input
import numpy as np

# Clustering packages
from sklearn.cluster import KMeans, MiniBatchKMeans, Birch

# Evaluation metrics
from sklearn.metrics.cluster import supervised
from scipy.optimize import linear_sum_assignment

# Plotting
import matplotlib.pyplot as plt

### CODE ###
def input_data():
    mnist_projection = np.genfromtxt('mnist_pca_projection.csv', delimiter=',')
    print(mnist_projection.shape)
    xlabel = np.genfromtxt('xlabel.csv', delimiter=',')
    urerf_labels = np.genfromtxt('urerf_labels.csv', delimiter=',')
    return mnist_projection, xlabel, urerf_labels


def cluster(projected):
    # KMeans
    start_time = time.time()
    kmeans = KMeans(n_clusters=5, random_state=10).fit(projected)
    kmeans_time = time.time() - start_time
    print('KMeans: {}'.format(kmeans_time))

    # SpectralClustering
    start_time = time.time()
    mbatch = MiniBatchKMeans(n_clusters=5).fit(projected)
    mbatch_time = time.time() - start_time
    print('MiniBatchKMeans: {}'.format(mbatch_time))

    # DBSCAN
    start_time = time.time()
    birch = Birch(n_clusters=5).fit(projected)
    birch_time = time.time() - start_time
    print('Birch: {}'.format(birch_time))

    return [(kmeans, kmeans_time), (mbatch, mbatch_time), (birch, birch_time)]


def evaluate(xlabel, clustering_object):
    # Input is a sklearn clustering object (KMeans, SpectralClustering)
    labels_true, labels_pred = supervised.check_clusterings(
        xlabel, clustering_object)
    value = supervised.contingency_matrix(labels_true, labels_pred)
    [r, c] = linear_sum_assignment(-value)
    accr = value[r, c].sum() / len(labels_true)
    return accr


if __name__ == '__main__':
    mnist_projection, xlabel, urerf_labels = input_data()

    # U-Rerf
    plt.scatter(mnist_projection[:,0], mnist_projection[:,1],
                c=urerf_labels, edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('nipy_spectral', 10))
    plt.savefig('../urerf_labels.png', dpi=500)
    plt.show()
    accr = evaluate(xlabel=xlabel, clustering_object=urerf_labels)
    print(accr)

    # Other clustering
    objects = cluster(mnist_projection)
    print('got objects')
    i=0
    for clustering_object in objects:
        accr = evaluate(xlabel=xlabel, clustering_object=clustering_object[0].labels_)

        plt.scatter(mnist_projection[:,0], mnist_projection[:,1],
                    c=clustering_object[0].labels_, edgecolor='none', alpha=0.5,
                    cmap=plt.cm.get_cmap('nipy_spectral', 10))
        plt.savefig(f'../{str(i)}.png', dpi=500)
        plt.show()

        print(accr)
        i += 1
