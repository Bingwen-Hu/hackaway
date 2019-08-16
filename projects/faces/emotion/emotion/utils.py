import os
import keras
from .models import mini_XCEPTION


def preprocess_input(x):
    x = x.astype('float32')
    x = x / 255.0
    x = x - 0.5
    x = x * 2.0
    return x

def load_model():
    cwd = os.path.dirname(__file__) 
    net = keras.models.load_model(os.path.join(cwd, 'emotion.hdf5'), compile=False)
    return net
