import os
import os.path as osp
import random
import string

import cv2
import numpy as np



class Parameter:
    # Charset size and captcha size
    charset = string.digits + string.ascii_uppercase
    charset_size = len(charset)
    captcha_size = 4

    # Input image shape
    im_h = 60
    im_w = 140

    # batch size and training epoch
    batch = 128
    epoch = 300

    # directory to save model and resume
    checkpoint_dir = 'checkpoints'
    resume = False

    # dataset path
    trainset = 'path/to/train/data'
    testset = 'path/to/test/data'


class CaptchaMixin:

    @staticmethod
    def preprocess(path):
        im = cv2.imread(path)
        im = im / 255.0
        return im


class InvalidLabel(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class CaptchaTrain(CaptchaMixin):

    def __init__(self, params: Parameter, check_label=True):
        self.params = params
        self.train_images = self.make_path(params.trainset) 
        self.test_images = self.make_path(params.testset)

        if check_label:
            self.check(self.train_images)
            self.check(self.test_images)

    def check(self, paths):
        """check out invalid data labels"""
        invalid = set()
        for path in paths:
            basename = osp.basename(path)
            chars = basename[:self.params.captcha_size]
            for char in chars:
                if char not in self.params.charset:
                    invalid.add(path)
        if len(invalid) > 0:
            with open('invalid.txt', 'w') as f:
                content = '\n'.join(invalid)
                f.write(content)
            raise InvalidLabel("Invalid labels detected!")

    def make_path(self, dir):
        images = os.listdir(dir)
        paths = [osp.join(dir, im) for im in images]
        return paths

    def data_generator(self, images, shuffle=True):
        nb_sample = len(images)
        nb_batch = nb_sample / self.params.batch

        if shuffle:
            random.shuffle(images)

        for i in range(nb_batch):
            start = i * self.params.batch
            end = min(start + self.params.batch, nb_sample)
            batch = images[start : end]
            xs = list(map(self.get_X, batch))
            ys = list(map(self.get_Y, batch))
            yield xs, ys

    def chars2vec(self, chars):
        """Convert captcha chars to vector using one-hot notation"""
        shape = self.params.captcha_size, self.params.charset_size
        matrix = np.zeros(shape, dtype=np.float32)

        for vec, char in zip(matrix, chars):
            index = self.params.charset.index(char)
            vec[index] = 1.0

        vec = matrix.flatten()
        return vec

    def get_X(self, path):
        return self.preprocess(path)

    def get_Y(self, path):
        filename = osp.basename(path)    
        chars = filename[:self.params.captcha_size]
        vec = self.chars2vec(chars)
        return vec
        

class CaptchaTest(CaptchaMixin):

    def __init__(self, params: Parameter):
        self.params = Parameter

    def index2chars(self, index):
        chars = ''.join([self.params.charset[i] for i in index])
        return chars




if __name__ == '__main__':

    # test chars2vec
    chars = 'AB21'
    train = CaptchaTrain(Parameter)
    vec = train.chars2vec(chars)

    test = CaptchaTest(Parameter)
    chars = [1,2,3,4]