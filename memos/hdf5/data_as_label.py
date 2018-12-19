import h5py
import os



class FaceHdf5(object):

    def __init__(self, hdf5Path, mode='r'):
        if mode == 'w' and os.path.exists(hdf5Path):
            raise ValueError("File exists!")
        elif mode == 'r' and not os.path.exists(hdf5Path):
            raise ValueError("File does not exist!")
        self.db = h5py.File(hdf5Path, mode)
        self.mode = mode

    def write(self, data, labels) -> None:
        if self.mode != 'w':
            raise ValueError('the `write` method is only valid in `w` mode')

        dim = data.shape
        maxshape = None, *dim[1:]
        dset_data = self.db.create_dataset('data', dim, maxshape=maxshape, dtype='float')
        dset_data = data
        dset_labels = self.db.create_dataset('labels', (dim[0],), maxshape=(None,), dtype=int)
        dset_labels = labels
    
    def append(self, data, labels):
        if self.mode not in ('w', 'a'):
            raise ValueError('the `append` method is only valid in `w` or `a` mode')
        
        dset_data = self.db['data']
        dset_labels = self.db['labels']
        index = dset_data.shape[0]
        dset_data.resize((index + len(labels), *data.shape[1:]))
        dset_data[index:, ...] = data
        dset_labels.resize((index + len(labels), ))
        dset_labels[index:] = labels
        

    def read(self, rows=None):
        if self.mode not in ['r', 'a']:
            raise ValueError('the `read` method is only valid in `r` or `a` mode')

        dset_data = self.db['data']
        dset_labels = self.db['labels']
        if rows:
            return dset_data[:rows], dset_labels[:rows]
        return dset_data[:], dset_labels[:]

    def close(self):
        self.db.close()
    


import unittest
import numpy as np
class TestFaceHdf5(unittest.TestCase):

    def setUp(self):
        db = FaceHdf5('test.h5', 'w')
        data = np.ones((7, 128), dtype=int)
        labels = np.arange(7, dtype=int)
        db.write(data, labels)
        db.close()
        
    def test_read_append(self):
        # test reader
        db = FaceHdf5('test.h5', 'r')
        data, labels = db.read()
        self.assertTupleEqual(data.shape, (7, 128))
        self.assertTupleEqual(labels.shape, (7,))
        db.close()

        # test append
        db = FaceHdf5('test.h5', 'a')
        data = np.ones((14, 128), dtype=float)
        data[2, 4] = 42
        labels = np.arange(14, dtype=int)
        # first write into
        db.append(data, labels)
        db.close()
        # then read back
        db = FaceHdf5('test.h5', 'r')
        data_, _ = db.read()
        self.assertTupleEqual(data_.shape, (21, 128))
        # test the new data is exactly what we append
        new_data = data_[7:, ...]
        print(new_data.shape, data.shape, type(new_data), type(data))
        self.assertTrue((new_data == data).all())
        db.close()

    def tearDown(self):
        os.remove('test.h5')
