from load_embedding import emb

import matplotlib.pyplot as plt



def main():
    filename = 'sub-NDARAA536PTU_acq-64dir_dwi_JHU.emb'
    filepath = '/Users/vivek/Documents/GitHub/NDD/test/struc2vec/emb/'

    # Load embedding
    emb1 = emb(filepath + filename)

    # Make plot
    plt.title(filename.split('.')[0])
    plt.scatter(x=emb1.data[:, 1], y=emb1.data[:, 2], label=emb1.filename)
    plt.savefig('./figures/{}-embedding.png'.format(filename.split('.')[0]), dpi=500)


if __name__ == '__main__':
    main()
