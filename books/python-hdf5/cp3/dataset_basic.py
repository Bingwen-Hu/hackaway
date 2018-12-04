import h5py
import numpy as np


def basic():
    f = h5py.File('testfile.h5')
    array = np.ones((5, 2))
    f['mory'] = array
    dset = f['mory']
    # proxy object lets you read and write
    return dset


dset = basic()
print(dset[:])
print(dset[...])


# empty datasets
f = h5py.File('test1.h5')
dset = f.create_dataset('test1', (10, 10))


# boolean operation
data = np.random.random(10) * 2 - 1
dset[data<0] = 0

# read_directly
def read_direct():
    "for big data"
    f = h5py.File('test1.h5')
    dataset = f['test1']
    out = np.empty((10, 10), dtype=np.float64)
    dset.read_direct(out)
    return out