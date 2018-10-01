''''
Define each layer of the tensor:
    - k: Atlas region
    - t_{i,j,k} = d_k(i,v)
'''

import os
import numpy as np
from load_emb import emb


def collect_all_embs():
    path = '/Users/vivek/Documents/GitHub/NDD/test/struc2vec/emb/'
    filelist = [file for file in os.listdir(path) if '.emb' in file]
    return [emb(path+file) for file in filelist]

def construct_tensor(embs):

    # Make empty tensor
    n = len(embs)
    k = min([emb.number_of_nodes for emb in embs]) - 1
    dist_tensor = np.zeros([k, n, n])
    print(dist_tensor.shape)

    # Populate tensor entries
    for node in range(k):
        for i_0, emb_0 in enumerate(embs):
            for i_1, emb_1 in enumerate(embs):

                print(node, i_0, i_1)

                a = embs[i_0].data[node][1:]
                b = embs[i_1].data[node][1:]

                dist = np.linalg.norm(a - b)
                dist_tensor[node, i_0, i_1] = dist

    print('Finished calculating tensor')
    print(dist_tensor.shape)

    return dist_tensor




def main():
    embs = collect_all_embs()
    dist_tensor = construct_tensor(embs)



if __name__ == '__main__':
    main()
