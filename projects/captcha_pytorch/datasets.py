import os
import torch.utils.data as data
import numpy as np
from PIL import Image


class Captcha(data.Dataset):

    def __init__(self, args, train=True):
        self.batch_size = args.batch_size
        self.width = args.image_width
        self.height = args.image_height
        self.charset = args.charset
        self.textlen = args.captcha_size
        
        data_dir = args.train_data_dir if train else args.test_data_dir
        self.path = [os.path.join(dir, f) for dir, _, files in 
                     os.walk(data_dir) for f in files]
    
        
    def __getitem__(self, index):
        return self.get_X(index), self.get_y(index)

    def __len__(self):
        return len(self.path)
    
    def get_y(self, index):
        basename = os.path.basename(self.path[index])
        text = basename[:self.textlen]
        vec = self.text2vec(text)
        return vec

    def get_X(self, index):
        """对path指向的文件处理"""
        img = Image.open(self.path[index])
        img = img.resize([self.width, self.height])
        img = img.convert('L')
        img = np.array(img).flatten() / 255
        return img
    
    def text2vec(self, text):
        def char2vec(c):
            y = np.zeros(self.charset)
            y[self.charset.index(c)] = 1.0
            return y
        vec = np.vstack([char2vec(c) for c in text])
        vec = vec.flatten()
        return vec


