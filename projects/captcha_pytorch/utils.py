import numpy as np


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


def index2text(index, charset):
    """Transform index of CHARSET to text
    """
    text = ''.join([charset[i] for i in index])
    return text