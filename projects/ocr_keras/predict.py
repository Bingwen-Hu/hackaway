from keras.models import load_model
import numpy as np
from preprocess import get_X, int2text
from config import FLAGS
from PIL import Image

global_model = None


def predict(image_path):
    size = FLAGS.image_size, FLAGS.image_size
    img = get_X(image_path, size)
    ret = model.predict(np.array([img]))
    ret = np.squeeze(ret)
    index = np.argmax(ret, axis=0)
    text = int2text(index, FLAGS.wordset)
    return text

def predict_top_5(image_path):
    size = FLAGS.image_size, FLAGS.image_size
    img = get_X(image_path, size)
    ret = model.predict(np.array([img]))
    ret = np.squeeze(ret)
    ret = np.argsort(ret)
    ret = ret[-5:][::-1]
    text = [int2text(index, FLAGS.wordset) for index in ret]
    return text

def predict_interface(img):
    global global_model
    if not global_model:
        global_model = load_model('model.h5')
        model = global_model
    else:
        model = global_model
    size = FLAGS.image_size, FLAGS.image_size
    def get_X_(img):
        img = img.convert('L')
        img = img.resize(size, Image.BICUBIC)
        img = np.array(img) / 255
        img = img[:, :, np.newaxis]
        return img
    img = get_X_(img)
    ret = model.predict(np.array([img]))
    ret = np.squeeze(ret)
    ret = np.argsort(ret)
    ret = ret[-5:][::-1]
    text = [int2text(index, FLAGS.wordset) for index in ret]
    return text

def test_accuracy_top_5(paths):
    import os.path
    results = 0
    for p in paths:
        char = os.path.basename(p)[0]
        text = predict_top_5(p)
        if char in text:
            results += 1
    print('accuracy : %.3f' % (results / len(paths)))

