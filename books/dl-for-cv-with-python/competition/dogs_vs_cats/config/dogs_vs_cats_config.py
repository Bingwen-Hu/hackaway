IMAGES_PATH = "../datasets/kaggle_dogs_vs_cats"

NUM_CLASSES = 2
NUM_VAL_IMAGES = 250 * NUM_CLASSES
NUM_TEST_IMAGES = 250 * NUM_CLASSES

TRAIN_HDF5 = "../datasets/kaggle_dogs_vs_cats/hdf5/train.hdf5"
VAL_HDF5 = "../datasets/kaggle_dogs_vs_cats/hdf5/val.hdf5"
TEST_HDF5 = "../datasets/kaggle_dogs_vs_cats/hdf5/test.hdf5"

MODEL_PATH = "output/alexnet_dogs_vs_cats.model"
DATASET_MEAN = 'output/dogs_vs_cats_mean.json'

OUTPUT_PATH = 'output'

"""
The DATASET_MEAN file will be used to store the average red, green, and blue pixel intensity
values across the entire (training) dataset. When we train our network, we’ll subtract the mean
RGB values from every pixel in the image (the same goes for testing and evaluation as well). This
method, called mean subtraction, is a type of data normalization technique and is more often used
than scaling pixel intensities to the range [0, 1] as it’s shown to be more effective on large datasets
and deeper neural networks.
"""
