# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image

from config import FLAGS

def text2int(text, wordset):
    return wordset.index(text)


def int2text(index, wordset):
    return wordset[index]

def black2red(img):
    data = np.array(img)
    n_rows, n_columns, _ = data.shape
    for i in range(n_rows):
        for j in range(n_columns):
            R, G, B = data[i][j]
            if R < 255 or G < 255 or B < 255:
                data[i][j] = [255, 0, 0]
    img = Image.fromarray(data)
    return img

def get_X(path, size, random_red=False):
    """resize, convert to gray, flatten and normalizes
    random_red: random change the image to red.
    """
    img = Image.open(path)

    if random_red:
        img = black2red(img)

    img = img.resize(size).convert('L')
    img = np.array(img).flatten() / 255
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
    size = (FLAGS.image_size, FLAGS.image_size)
    data_iter = data_iterator(FLAGS.train_data_dir, FLAGS.batch_size, FLAGS.num_epochs)
    for data in data_iter:
        X = [get_X(datum, size, random_red=True) for datum in data]
        y = [get_Y(datum, FLAGS.wordset) for datum in data]
        yield X, y

# most ugly function
def test_data_helper(batch_size=None):
    size = (FLAGS.image_size, FLAGS.image_size)
    data = [os.path.join(dir, f) for dir, _, files in
            os.walk(FLAGS.test_data_dir) for f in files]

    if batch_size is not None:
        np.random.shuffle(data)
        data = data[:batch_size]
    X = [get_X(datum, size) for datum in data]
    y = [get_Y(datum, FLAGS.wordset) for datum in data]
    return X, y