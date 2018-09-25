import numpy as np
import os
import cv2



def padding(img, size):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = size
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_CUBIC)
    canvas = np.full((h, w, 3), 128)
    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w, :] = resized_image
    return canvas



def text2vec(text, charset):
    vec = np.zeros(len(charset))
    index = charset.index(text)
    vec[index] = 1
    return vec


def vec2text(vec, charset):
    text = ''.join([charset[i] for i in index])
    return text


def get_X(path, size):
    """resize, convert to gray, flatten and normalizes
    random_red: random change the image to red.
    """
    img = cv2.imread(path)
    img = padding(img, size)
    img = img.transpose((2, 0, 1)) / 255
    return img


def get_Y(path, charset, charlen):
    """assume captche text is at the beginning of path
    """
    basename = os.path.basename(path)
    text = basename[0:charlen]
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
    size = (config.width, config.height)
    data_iter = data_iterator(config.train_dir, config.batch_size, config.epochs)
    for data in data_iter:
        X = [get_X(datum, size) for datum in data]
        y = [get_Y(datum, config.charset) for datum in data]
        yield X, y

# most ugly function
def test_data_helper(batch_size=None):
    size = (config.width, config.height)
    data = [os.path.join(dir, f) for dir, _, files in
            os.walk(config.test_data_dir) for f in files]

    if batch_size is not None:
        np.random.shuffle(data)
        data = data[:batch_size]
    X = [get_X(datum, size) for datum in data]
    y = [get_Y(datum, config.charset) for datum in data]
    return X, y