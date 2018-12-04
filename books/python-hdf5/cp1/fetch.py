import h5py
import numpy as np

f = h5py.File('weather.hdf5')

dataset = f['/15/temperature']

# slicing
print(dataset[1:10])
print(dataset[1:10:2])


# big storage
big_dataset = f.create_dataset('comp', shape=(1024,), dtype='int32', compression='gzip')
big_dataset[:] = np.arange(1024)
print(big_dataset[:])