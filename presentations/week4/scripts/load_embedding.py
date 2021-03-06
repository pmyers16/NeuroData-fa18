import numpy as np
import matplotlib.pyplot as plt


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

        # Sort array by 0th column
        data = np.array(data)
        data = data[data[:, 0].argsort()]

        return data

    def plot(self):
        plt.scatter(x=self.data[:, 1], y=self.data[:, 2])
        plt.show()


def test_function():
    path = '/Users/vivek/Documents/GitHub/NDD/test/struc2vec/emb/sub-NDARBA521RA8_acq-64dir_dwi_JHU.emb'
    embbed = emb(filename=path)
    embbed.plot()


if __name__ == '__main__':
    test_function()
