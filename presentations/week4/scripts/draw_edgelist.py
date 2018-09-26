### IMPORTS ###
# Fetch data
import subprocess
import requests

# Data structure
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


### CODE ###
class edge_terrier():
    # edge_terrier class returns all edgelists in a specified directory as networkx objects

    def __init__(self, filepath='hbn/derivatives/graphs/JHU/'):

        # Establish filepath
        self.filepath = filepath

        # Get filelist for specific filepath
        self.filelist = self.s3_ls()

    def s3_ls(self):

        # Parse the output of the subprocess query
        filelist = subprocess.check_output(
            ['aws', 's3', 'ls', 'neurodatadesign/' + self.filepath, '--no-sign-request']).split()
        filelist = [file.decode("utf-8") for file in filelist]
        filelist = [file for file in filelist if (
            '/' in file) or ('.' in file)]

        return filelist

    def convert_edgelist(self, filename, draw_graph=False):

        # Fetch edgelist
        link = 'http://neurodatadesign.s3.amazonaws.com/' + self.filepath + filename
        edges = requests.get(link).text.split()
        edges = np.array([int(x) for x in edges])
        edges = [tuple(edges[x:x + 3]) for x in range(0, len(edges), 3)]

        if edges == []:
            print(filename + ' is empty.')
            return

        # Convert edgelist to networkx object
        G = nx.Graph()
        G.add_weighted_edges_from(edges)

        if draw_graph:
            nx.draw(G)
            plt.show()

        return G, filename

    def convert_edgelist_all(self):
        # returns a generator of all filelists
        # change indexing to get more edgelists
        for filename in self.filelist[0:1]:
            G, filename = self.convert_edgelist(filename)
            if G is not None:
                yield G, filename


if __name__ == '__main__':
    fetch = edge_terrier()
    edgelists = fetch.convert_edgelist_all()

    for graph in edgelists:
        G, filename = graph
        filename = filename.split('.')[0]
        plt.title(filename.split('.')[0])

        adj = nx.to_numpy_matrix(G)
        print(adj)
        plt.imshow(adj)
        plt.colorbar()
        # plt.show()

        plt.savefig('./figures/{}-edglist-image.png'.format(filename), dpi=500)
