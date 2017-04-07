import re

from PIL import Image
import numpy as np
import pandas as pd

import common


def get_gray_pixel(filepath):
    """Open the image by filepath and convert it to gray image,
    then return as numpy array"""
    image = Image.open(filepath)
    return np.array(image.convert("L"))

# using flatten will be better
def convert2str(np_array):
    m = map(str, np_array)
    s = ' '.join(m)
    p = re.compile(r'[\[\]\n]')
    return p.sub("", s)

def comp(filepath):
    arr = get_gray_pixel(filepath)
    s = convert2str(arr)
    return s

def text_to_vec(text):
    def char_to_vec(c):
        y = np.zeros((len(common.CHARS),))
        y[common.CHARS.index(c)] = 1.0
        return y
    vec = np.vstack([char_to_vec(c) for c in text])
    return vec.flatten()


def data_prepare(df):
    df['image'] = df['image'].apply(lambda img: np.fromstring(img, sep=' ') / 255.0)
    df['text'] = df['text'].apply(lambda text: text_to_vec(text))
    X = np.vstack(df['image'])
    X = X.reshape((-1, 60, 250, 1))
    y = np.vstack(df['text'])
    return X, y



def get_dataset(filepath, nrows=3000, usecols=[2, 3]):
    df =  pd.read_csv(filepath, nrows=nrows, usecols=usecols)
    X, y = data_prepare(df)
    return df, X, y
