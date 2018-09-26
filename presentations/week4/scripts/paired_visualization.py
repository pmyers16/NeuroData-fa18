from load_embeddings import emb

import os
import matplotlib.pyplot as plt


def main():

    # Get files
    filepath = '/Users/vivek/Documents/GitHub/NDD/test/struc2vec/emb/'
    filelist = os.listdir(filepath)

    # load 2 data files
    emb1 = emb(filepath + filelist[0])
    emb2 = emb(filepath + filelist[1])
    emb3 = emb(filepath + filelist[2])

    # Let's see some plots
    plt.scatter(x=emb1.data[:, 1], y=emb1.data[:, 2], label=emb1.filename)
    plt.scatter(x=emb2.data[:, 1], y=emb2.data[:, 2], label=emb2.filename)
    plt.scatter(x=emb3.data[:, 1], y=emb3.data[:, 2], label=emb3.filename)

    for i in range(emb1.number_of_nodes):
        x_ = [emb1.data[i, 1], emb2.data[i, 1]]
        y_ = [emb1.data[i, 2], emb2.data[i, 2]]
        plt.plot(x_, y_, 'k--', alpha=0.25)

        x_ = [emb1.data[i, 1], emb3.data[i, 1]]
        y_ = [emb1.data[i, 2], emb3.data[i, 2]]
        plt.plot(x_, y_, 'k--', alpha=0.25)

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
