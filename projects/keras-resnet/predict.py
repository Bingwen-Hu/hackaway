from keras.models import load_model
import numpy as np
from preprocess import get_X, index2text
from vgg16 import sigmoid_loss, captcha_accuracy
from config import FLAGS



custom_object = {"sigmoid_loss": sigmoid_loss,
                 'captcha_accuracy': captcha_accuracy}
model = load_model('sina.h5', custom_object)


def predict(image_path):
    size = FLAGS.image_width, FLAGS.image_height
    img = get_X(image_path, size)
    ret = model.predict(np.array([img]))
    ret = ret.reshape(FLAGS.captcha_size, FLAGS.charset_size)
    index = np.argmax(ret, axis=1)
    text = index2text(index, FLAGS.charset)
    return text


