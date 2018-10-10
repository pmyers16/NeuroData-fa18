import numpy as np


# Embedding class for loading data
class emb():

    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = filepath.split('/')[-1].split('.')[0]
        self.data = self.process()

    def process(self):

        # Initialize empty array
        data = []

        # Read data
        with open(self.filepath) as file:

            for index, line in enumerate(file):
                line = line.split()

                # Skip the 0th line
                if index == 0:
                    emb.number_of_nodes = int(line[0])
                    emb.dim = int(line[1])
                else:
                    line = [float(x) for x in line]
                    data.append(line)

        data = np.array(data)

        # Sort array by 0th column
        data = data[data[:, 0].argsort()]

        return data
