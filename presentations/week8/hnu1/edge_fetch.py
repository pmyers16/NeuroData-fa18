### IMPORTS ###

"""
edge_fetch for the HNU1 data
need to use BytesIO because HNU1 data is encoded using gpickle
might require networkx1.9?
"""

# Fetch data
import subprocess
import requests
import pickle
from io import BytesIO

# Data structure
import numpy as np
import networkx as nx

# edge_fetch.py written by @v715
### CODE ###
class edge_terrier():
    # edge_terrier class returns all edgelists in a specified directory as networkx objects

    def __init__(self, aws_path='aws', neuro=False, filepath='hbn/derivatives/graphs/JHU/'):

        # Establish filepath
        self.filepath = filepath
        self.aws_path = aws_path
        self.neuro = neuro
        # Get filelist for specific filepath
        self.filelist = self.s3_ls()

    def s3_ls(self):

        # Parse the output of the subprocess query
        if self.neuro:
            filelist = subprocess.check_output(
                [self.aws_path, 's3', 'ls', 'neurodatadesign/' + self.filepath, '--no-sign-request']).split()
        else:
            filelist = subprocess.check_output(
            [self.aws_path, 's3', 'ls', 'mrneurodata/' + self.filepath, '--no-sign-request']).split()
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
        for filename in self.filelist:
            G = self.convert_edgelist(filename)
            if G is not None:
                yield G

    def convert_gpickle(self, filename, draw_graph=False):

        # Fetch edgelist
        link = 'http://mrneurodata.s3.amazonaws.com/' + self.filepath + filename
        edges = requests.get(link).content
        G = pickle.loads(edges)

        if draw_graph:
            nx.draw(G)
            plt.show()

        return G, filename

    def convert_gpickle_all(self):
        # returns a generator of all filelists
        for filename in self.filelist:
            G = self.convert_gpickle(filename, True)
            if G is not None:
                yield G

    def get_graphs(self, _list):
        graph_list = []
        for item in _list:
            graph_list.append(item[0])
        return graph_list
