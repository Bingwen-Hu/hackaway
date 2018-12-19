import h5py
import os



class FaceHdf5Writer(object):

    def __init__(self, hdf5Path):
        if os.path.exists(hdf5Path):
            raise ValueError("The supplied `hdf5path` already "
                    "exists and cannot be overwritten. Manually "
                    "delete before continuing.", hdf5Path)
        self.db = h5py.File(hdf5Path, "w")
        
    def write(self, label, data, **attrs):
        dim = data.shape
        maxshape = None, *dim[1:]
        print(f"write into label: {label} as maxshape: {maxshape}...")
        dset = self.db.create_dataset(label, dim, maxshape=maxshape, dtype='float')
        dset[...] = data
        for k, v in attrs.items():
            dset.attrs[k] = v
        return dset
    
    def close(self):
        self.db.close()
    
class FaceHdf5Reader(object):

    def __init__(self, hdf5Path):
        if not os.path.exists(hdf5Path):
            raise ValueError("The supplied `hdf5path` does not "
                    "exist", hdf5Path)
        self.db = h5py.File(hdf5Path)
    
    def read(self, label):
        dset = self.db[label]
        return dset[:]

    def close(self):
        self.db.close()


import unittest
import numpy as np
class TestFaceHdf5Writer(unittest.TestCase):

    def setUp(self):
        self.writer = FaceHdf5Writer('test.h5')
    
    def test_writer(self):
        data = np.ones((7, 128), dtype=int)
        dset = self.writer.write('test', data, **{'level': 1, 'active': True})
        data_ = dset[...]
        self.assertTupleEqual(data_.shape, data.shape)

        # test attributes
        self.assertEqual(dset.attrs['active'], True)
        self.assertEqual(dset.attrs['level'], 1)

        # another test
        data = np.ones((100, 64, 64), dtype=float)
        dset = self.writer.write('image', data)
        data_ = dset[...]
        self.assertTupleEqual(data_.shape, data.shape)

        # close here and test reader
        self.writer.close()
        reader = FaceHdf5Reader('test.h5')
        image = reader.read('image')
        self.assertTupleEqual(image.shape, data.shape)
        reader.close()

    def tearDown(self):
        self.writer.close()
        os.remove('test.h5')
