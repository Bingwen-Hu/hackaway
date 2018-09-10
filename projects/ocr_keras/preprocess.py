# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image

from config import FLAGS

def text2int(text, wordset):
    try:
        index = wordset.index(text) 
    except:
        print(text)
        
    return index

def text2vec(text, charset):
    """Function used to transform text string to a numpy array vector.
    :param text: namely the captcha code.
    :param charset: charset used by the specific problem.
    """
    def char2vec(c):
        y = np.zeros((len(charset),))
        y[charset.index(c)] = 1.0
        return y
    vec = np.vstack([char2vec(c) for c in text])
    vec = vec.flatten()
    return vec

def int2text(index, wordset):
    return wordset[index]


def get_X(img, size):
    """resize, convert to gray, flatten and normalizes
    random_red: random change the image to red.
    """
    img = Image.open(img)
    img = img.convert('L')
    img = img.resize(size, Image.BICUBIC)
    img = np.array(img) / 255
    img = img[:, :, np.newaxis]
    return img


def get_Y(path, wordset):
    """assume captche text is at the beginning of path
    """
    basename = os.path.basename(path)
    text = basename[0]
    vec = text2vec(text, wordset)
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
        X = [get_X(datum, size) for datum in data]
        y = [get_Y(datum, FLAGS.wordset) for datum in data]
        yield X, y

# from mimic import merge_char, generate_char
# import glob
# import random
# size_t = (FLAGS.image_size, FLAGS.image_size)
# fonts = ['msyhbd', 'msyh']
# background_glob_dir = "org/*.jpg"
# sizes = [40, 45, 50, 55, 60]
# background_imgs = glob.glob(background_glob_dir)
# angles = [45, 90, 180, 270, -45]
# chars = FLAGS.wordset

# def train_data_iterator():
#     for i in range(FLAGS.num_epochs):
#         yield train_data_helper()

# def train_data_helper():    
#     data = []
#     for i in range(FLAGS.batch_size):
#         angle = random.choice(angles)
#         char = random.choice(chars)
#         font = random.choice(fonts)
#         size = random.choice(sizes)
#         bg = random.choice(background_imgs)
#         bg_img = Image.open(bg)
#         charimg = generate_char(char, size, font)
#         charimg = charimg.rotate(angle, resample=Image.BICUBIC, expand=True)
#         newimg, (x, y), (height, width) = merge_char(charimg, bg_img)
#         crop = newimg.crop((y, x, y+width, x+height))
#         data.append([crop, char])
#     X = [get_X(crop, size_t) for (crop, _) in data]
#     y = [text2int(char, FLAGS.wordset) for (_, char) in data]
#     return X, y

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