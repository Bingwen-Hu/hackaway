from hdf5io import HDF5DatasetWriter
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import os
import cv2
import json
import progressbar
import numpy as np

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
trainLabels = LabelBinarizer().fit_transform(trainLabels)

split = train_test_split(trainPaths, trainLabels, 
    test_size=config.NUM_TEST_IMAGES, stratify=trainLabels,
    random_state=42)

trainPaths, testPaths, trainLabels, testLabels = split

split = train_test_split(trainPaths, trainLabels, 
    test_size=config.NUM_VAL_IMAGES, stratify=trainLabels, 
    random_state=42)

trainPaths, valPaths, trainLabels, valLabels = split

datasets = [
    ('train', trainPaths, trainLabels, config.TRAIN_HDF5),
    ('val', valPaths, valLabels, config.VAL_HDF5),
    ('test', testPaths, testLabels, config.TEST_HDF5),
]

# if there is any transforms, set up here
transforms = []
# placeholder for RGB pixel
R, G, B = [], [], []

for (dType, paths, labels, outputPath) in datasets:
    print('[INFO] building {}...'.format(outputPath))
    writer = HDF5DatasetWriter((len(paths), 256, 256, 3), outputPath)


    widgets = ['Building Dataset: ', progressbar.Percentage(), ' ',
        progressbar.Bar(), ' ', progressbar.ETA()]
    
    pbar = progressbar.ProgressBar(maxval=len(paths), 
        widgets=widgets).start()

    for (i, (path, label)) in enumerate(zip(paths, labels)):
        image = cv2.imread(path)
        image = cv2.resize(image, (256, 256)) # match writer 
        
        if dType == 'train':
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)
        
        writer.add([image], [label[0]])
        pbar.update(i)
    
    pbar.finish()
    writer.close()

print('[INFO] serializing means...')
D = {'R': np.mean(R), 'G': np.mean(G), 'B': np.mean(B)}
with open(config.DATASET_MEAN, 'w') as f:
    f.write(json.dumps(D))