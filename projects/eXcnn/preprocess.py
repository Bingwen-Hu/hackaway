# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image

from config import FLAGS

def text2vec(text, charset):
    """Function used to transform text string to a numpy array vector.
    :param text: namely the captcha code.
    :param charset: charset used by the specific problem.

    Usage::

      >>> from utils import text2vec
      >>> from constants import ALL_ALPHABET
      >>> text = "AbdTd"
      >>> text2vec(text, ALL_ALPHABET)
      array([ 0.,  0.,  0., ...,  0.,  0.,  0.])
    """
    def char2vec(c):
        y = np.zeros((len(charset),))
        y[charset.index(c)] = 1.0
        return y
    vec = np.vstack([char2vec(c) for c in text])
    vec = vec.flatten()
    return vec


def index2text(index, charset):
    """Transform index of CHARSET to text
    """
    text = ''.join([charset[i] for i in index])
    return text


def get_X(path, size):
    """resize, convert to gray, flatten and normalizes
    """
    img = Image.open(path)
    img = img.resize(size).convert('L')
    img = np.array(img).flatten() / 255
    return img


def get_Y(path, charset, captcha_size):
    """assume captche text is at the beginning of path
    """
    basename = os.path.basename(path)
    text = basename[:captcha_size]
    vec = text2vec(text, charset)
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
    size = (FLAGS.image_height, FLAGS.image_width)
    data_iter = data_iterator(FLAGS.train_data_dir, FLAGS.batch_size, FLAGS.num_epochs)
    for data in data_iter:
        X = [get_X(datum, size) for datum in data]
        y = [get_Y(datum, FLAGS.charset, FLAGS.captcha_size) for datum in data]
        yield X, y

# most ugly function
def test_data_helper(batch_size=None):
    size = (FLAGS.image_height, FLAGS.image_width)
    data = [os.path.join(dir, f) for dir, _, files in
            os.walk(FLAGS.test_data_dir) for f in files]

    if batch_size is not None:
        data = np.random.shuffle(data)
        data = data[:batch_size]
    X = [get_X(datum, size) for datum in data]
    y = [get_Y(datum, FLAGS.charset, FLAGS.captcha_size) for datum in data]
    return X, y