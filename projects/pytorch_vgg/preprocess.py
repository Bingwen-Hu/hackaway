import os
import numpy as np
from PIL import Image


import utils.paste as paste
import utils.process as process


from config import FLAGS




def get_X(path, size):
    """resize, convert to gray, flatten and normalizes
    """
    img = Image.open(path).convert("L")
    img = process.pixel_replace(img, 140, 255, True)
    img = img.resize(size, Image.BICUBIC)
    img = np.array(img).flatten() / 255.
    return img


def get_Y(path, charset, captcha_size):
    """assume captche text is at the beginning of path
    """
    basename = os.path.basename(path)
    text = basename[:captcha_size]
    vec = text2vec(text, charset)
    return vec




def paste_image(piecespaths):
    selectpaths = paste.random_select_pieces(piecespaths, 4)
    codes = [os.path.basename(path)[0] for path in selectpaths]
    codes = ''.join(codes)
    image = paste.paste("L", selectpaths, (104, 30))
    image = process.add_spot(image, 20, 0)
    return image, codes


def train_data_generator():
    """iterate around data
    data_dir: data directory, allow sub directory exists
    """

    data_dir = FLAGS.train_data_dir
    num_epochs = FLAGS.num_epochs
    batch_size = FLAGS.batch_size
    piecespaths = [os.path.join(dir, f) for dir, _, files in
                   os.walk(data_dir) for f in files]


    def get_images(piecespaths, batch_size):
        for i in range(batch_size):
            img, text = paste_image(piecespaths)
            img = img.resize(size, Image.BICUBIC)
            X = np.array(img).flatten() / 255.
            y = text2vec(text, FLAGS.charset)
            yield X, y
    size = FLAGS.image_width, FLAGS.image_height
    for _ in range(num_epochs):
        X_y = list(get_images(piecespaths, batch_size))
        X = [x_ for (x_, y_) in X_y]
        y = [y_ for (x_, y_) in X_y]
        yield X, y


# most ugly function
def test_data_helper(batch_size=None):
    size = (FLAGS.image_width, FLAGS.image_height)
    data = [os.path.join(dir, f) for dir, _, files in
            os.walk(FLAGS.test_data_dir) for f in files]

    if batch_size is not None:
        np.random.shuffle(data)
        data = data[:batch_size]
    X = [get_X(datum, size) for datum in data]
    y = [get_Y(datum, FLAGS.charset, FLAGS.captcha_size) for datum in data]
    return X, y