### IMPORTS ###
# 3rd party inputs
import numpy as np

# Import local modules
import sys
path = '/Users/vivek/Documents/GitHub/ndd-2018/scripts'
if sys.path[0] != path:
    sys.path.insert(0, path)
from edge_process import get_eigenvalues


### CODE ###





if __name__=='__main__':
    X = get_eigenvalues.calc_eigval_feature_matrix()
