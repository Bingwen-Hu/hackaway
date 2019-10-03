from models import GMM
from easydict import EasyDict
import torch
from torchvision import transforms
import numpy as np
import os.path as osp
from PIL import Image

opt = EasyDict()
opt.fine_height = 256
opt.fine_width = 192
opt.grid_size = 5
opt.data_path = './data/test/'
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
        

model = GMM(opt.fine_height, opt.fine_width, opt.grid_size)

model.load_state_dict(torch.load('checkpoints/gmm_last.pth', map_location='cpu'))

def preprocess(opt, c_name, im_name):
    # cloth image
    data_path = opt.data_path
    c = Image.open(osp.join(data_path, 'cloth', c_name))
    c = transform(c)  # [-1,1]

    # person image 
    im = Image.open(osp.join(data_path, 'image', im_name))
    im = transform(im) # [-1,1]

    # cloth on person as groundtruth
    parse_name = im_name.replace('.jpg', '.png')
    im_parse = Image.open(osp.join(data_path, 'image-parse', parse_name))
    parse_array = np.array(im_parse)
    parse_cloth = (parse_array == 5).astype(np.float32) + \
            (parse_array == 6).astype(np.float32) + \
            (parse_array == 7).astype(np.float32)

    parse_cloth = torch.from_numpy(parse_cloth)
    c_gt = im * parse_cloth + (1 - parse_cloth)

    # masked person image
    mask_name = parse_name
    im_mask = Image.open(osp.join(data_path, 'image-mask', mask_name))
    im_mask = transform(im_mask)
    
    inputs = {
        'c': c,
        'im_mask': im_mask,
        'c_gt': c_gt,
        'im': im,
    }

    return inputs

tests = [
    '000057_0.jpg', '012578_1.jpg',
    '000066_0.jpg', '009595_1.jpg'
]

4