import numpy as np
import h5py

with h5py.File('distance_tensor.h5', 'r') as hf:
    data = hf['distance_tensor'][:]

print(data)
