# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image

from config import args

def text2int(text, wordset):
    return wordset.index(text)

def text2vec(text, wordset):
    vec = np.zeros(len(wordset))
    index = wordset.index(text)
    vec[index] = 1
    return vec

def int2text(index, wordset):
    return wordset[index]


def get_X(path, size):
    """resize, convert to gray, flatten and normalizes
    random_red: random change the image to red.
    """
    img = Image.open(path)
    img = img.resize(size).convert('L')
    img = np.array(img)
    img = img[np.newaxis, :, :] / 255
    return img


def get_Y(path, wordset):
    """assume captche text is at the beginning of path
    """
    basename = os.path.basename(path)
    text = basename[0]
    vec = text2int(text, wordset)
    return vec


def data_iterator(data_dir, batch_size, num_epochs):
    """iterate around data
    data_dir: data directory, allow sub directory exists
    """
    data = [os.path.join(dir, f) for dir, _, files in
            os.walk(data_dir) for f in files]
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for _ in range(num_epochs):
        indices = np.random.permutation(len(data))
        shuffled_data = data[indices]
        for batch_idx in range(num_batches_per_epoch):
            start_index = batch_idx * batch_size
            end_index = min((batch_idx + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

# most ugly function
def train_data_iterator():
    size = (args.image_size, args.image_size)
    data_iter = data_iterator(args.train_data_dir, args.batch_size, args.epochs)
    for data in data_iter:
        X = [get_X(datum, size) for datum in data]
        y = [get_Y(datum, args.wordset) for datum in data]
        yield X, y

# most ugly function
def test_data_helper(batch_size=None):
    size = (args.image_size, args.image_size)
    data = [os.path.join(dir, f) for dir, _, files in
            os.walk(args.test_data_dir) for f in files]

    if batch_size is not None:
        np.random.shuffle(data)
        data = data[:batch_size]
    X = [get_X(datum, size) for datum in data]
    y = [get_Y(datum, args.wordset) for datum in data]
    return X, y