import os
import torch
import torch.utils.data as data
from torchvision import transforms
import numpy as np

from PIL import Image


import utils.paste as paste
import utils.process as process



def paste_image(piecespaths):
    selectpaths = paste.random_select_pieces(piecespaths, 4)
    codes = [os.path.basename(path)[0] for path in selectpaths]
    codes = ''.join(codes)
    image = paste.paste("L", selectpaths, (104, 30))
    image = process.add_spot(image, 20, 0)
    return image.convert("RGB"), codes

# Just for training
class Captcha(data.Dataset):

    def __init__(self, args):
        self.batch_size = args.batch_size
        self.width = args.image_width
        self.height = args.image_height
        self.charset = args.charset
        self.textlen = args.captcha_size
        
        data_dir = args.train_data_dir
        self.path = [os.path.join(dir, f) for dir, _, files in 
                     os.walk(data_dir) for f in files]
        
    def __getitem__(self, index):
        img, text = paste_image(self.path)
        transform = transforms.Compose([
            transforms.Resize((self.width, self.height), interpolation=Image.BICUBIC),
            transforms.Pad(padding=(82, 12)),
            transforms.ToTensor(),
        ])
        
        X = transform(img)
        y = torch.FloatTensor((text2vec(text, self.charset)))
        return X, y

    def __len__(self):
        # just a joke
        return 10000


def text2vec(text, charset):
    def char2vec(c):
        y = np.zeros(len(charset))
        y[charset.index(c)] = 1.0
        return y
    vec = np.vstack([char2vec(c) for c in text])
    vec = vec.flatten()
    return vec


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


def test_data_helper(args, batch_size=None):
    size = (args.image_width, args.image_height)
    data = [os.path.join(dir, f) for dir, _, files in
            os.walk(args.test_data_dir) for f in files]

    if batch_size is not None:
        np.random.shuffle(data)
        data = data[:batch_size]
    X = [get_X(datum, size) for datum in data]
    y = [get_Y(datum, args.charset, args.captcha_size) for datum in data]
    X = torch.FloatTensor(X).view(len(data), args.image_height, args.image_width)
    y = torch.FloatTensor(y).view(len(data), -1)
    return X, y