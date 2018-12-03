from hdf5io import HDF5DatasetWriter
from imutils import paths
from sklearn.model_selection import train_test_split
import os



class Config:
    # build dataset into HDF5 format
    IMAGES_PATH = "/home/jenny/hackaway/books/dl-for-cv-with-python/network/train"

    # dataset attributes
    NUM_CLASSES = 2
    NUM_VAL_IMAGES = 100 * NUM_CLASSES
    NUM_TEST_IMAGES = 100 * NUM_CLASSES

    # output path
    TRAIN_HDF5 = 'output/train.hdf5'
    VAL_HDF5 = 'output/val.hdf5'
    TEST_HDF5 = 'output/test.hdf5'

    # use for mean subtraction
    DATASET_MEAN = 'output/dataset_mean.json'


config = Config

trainPaths = list(paths.list_images(config.IMAGES_PATH))
# NOTE: dataset format: path/to/data/{label}/xx.png
trainLabels = [p.split(os.path.sep)[-2] for p in trainPaths]

split = train_test_split(trainPaths, trainLabels, 
    test_size=config.NUM_TEST_IMAGES, stratify=trainLabels,
    random_state=42)

trainPaths, testPaths, trainLabels, testLabels = split

split = train_test_split(trainPaths, trainLabels, 
    test_size=config.NUM_VAL_IMAGES, stratify=trainLabels, 
    random_state=42)

trainPaths, valPaths, trainLabels, valLabels = split