# First stage
# GMM is skipped
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

import argparse
import os
import os.path as osp
import json
import time
from networks import GMM, UnetGenerator, load_checkpoint

from PIL import Image, ImageDraw

from easydict import EasyDict

opt = EasyDict()
opt.fine_width = 192
opt.fine_height = 256
opt.data_path = 'data/test/'
opt.radius = 5
opt.grid_size = 5
opt.checkpoint = 'checkpoints/gmm_final.pth'
opt.checkpoint = 'checkpoints/tom_final.pth'

    
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def preprocess(opt: EasyDict, c_name, im_name, stage='GMM'):
    """
    Args:
        c_name: image (cloth) name
        im_name: image (person) name
        stage: GMM or TOM
    """
        
    # cloth image & cloth mask
    if stage == 'GMM':
        c = Image.open(osp.join(opt.data_path, 'cloth', c_name))
        cm = Image.open(osp.join(opt.data_path, 'cloth-mask', c_name))
    elif stage == "TOM":
        c = Image.open(osp.join(opt.data_path, 'warp-cloth', c_name))
        cm = Image.open(osp.join(opt.data_path, 'warp-mask', c_name))
    
    c = transform(c)  # [-1,1]
    cm_array = np.array(cm)
    cm_array = (cm_array >= 128).astype(np.float32)
    cm = torch.from_numpy(cm_array) # [0,1]
    cm.unsqueeze_(0) # inplace, expand the channel

    # person image 
    im = Image.open(osp.join(opt.data_path, 'image', im_name))
    im = transform(im) # [-1,1]

    # load parsing image
    parse_name = im_name.replace('.jpg', '.png')
    # 这是一个二值图片
    im_parse = Image.open(osp.join(opt.data_path, 'image-parse', parse_name))
    parse_array = np.array(im_parse)
    # 所有大于0的即是身体
    parse_shape = (parse_array > 0).astype(np.float32)
    # 头部由多种组成，可能是头发，帽子和脸等
    parse_head = (parse_array == 1).astype(np.float32) + \
            (parse_array == 2).astype(np.float32) + \
            (parse_array == 4).astype(np.float32) + \
            (parse_array == 13).astype(np.float32)
    # 衣服可能也是
    parse_cloth = (parse_array == 5).astype(np.float32) + \
            (parse_array == 6).astype(np.float32) + \
            (parse_array == 7).astype(np.float32)
    
    # shape downsample
    # 先放缩到一个低精度的图片，再放回原来的大小，使其模糊化
    parse_shape = Image.fromarray((parse_shape*255).astype(np.uint8))
    parse_shape = parse_shape.resize((opt.fine_width//16, opt.fine_height//16), Image.BILINEAR)
    parse_shape = parse_shape.resize((opt.fine_width, opt.fine_height), Image.BILINEAR)
    shape = transform(parse_shape) # 各种规范化
    phead = torch.from_numpy(parse_head) # [0,1]
    pcm = torch.from_numpy(parse_cloth) # [0,1]

    # upper cloth
    im_c = im * pcm + (1 - pcm) # [-1,1], fill 1 for other parts，除了pcm，其他都变成1
    im_h = im * phead - (1 - phead) # [-1,1], fill 0 for other parts

    # load pose points
    pose_name = im_name.replace('.jpg', '_keypoints.json')
    with open(osp.join(opt.data_path, 'pose', pose_name), 'r') as f:
        pose_label = json.load(f)
        pose_data = pose_label['people'][0]['pose_keypoints']
        pose_data = np.array(pose_data)
        pose_data = pose_data.reshape((-1,3))

    point_num = pose_data.shape[0]
    pose_map = torch.zeros(point_num, opt.fine_height, opt.fine_width)
    r = opt.radius
    im_pose = Image.new('L', (opt.fine_width, opt.fine_height))
    pose_draw = ImageDraw.Draw(im_pose)
    for i in range(point_num):
        one_map = Image.new('L', (opt.fine_width, opt.fine_height))
        draw = ImageDraw.Draw(one_map)
        pointx = pose_data[i,0]
        pointy = pose_data[i,1]
        if pointx > 1 and pointy > 1:
            draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
            pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
        one_map = transform(one_map)
        pose_map[i] = one_map[0]

    # just for visualization
    im_pose = transform(im_pose)
    
    # cloth-agnostic representation
    # shape: 指的是身体形状，且是模糊的
    # im_h：指的是头部的图像
    # pose_map：指的是姿态的各个通道
    agnostic = torch.cat([shape, im_h, pose_map], 0) 

    if stage == 'GMM':
        im_g = Image.open('grid.png')
        im_g = transform(im_g)
    else:
        im_g = ''

    result = {
        'c_name':   c_name,     # for visualization
        'im_name':  im_name,    # for visualization or ground truth
        'cloth':    c,          # for input
        'cloth_mask':     cm,   # for input
        'image':    im,         # for visualization
        'agnostic': agnostic,   # for input
        'parse_cloth': im_c,    # for ground truth
        'shape': shape,         # for visualization
        'head': im_h,           # for visualization
        'pose_image': im_pose,  # for visualization
        'grid_image': im_g,     # for visualization
        }

    return result

def gmm(opt, inputs, model):
    model.eval()

    agnostic = inputs['agnostic']
    c = inputs['cloth']
    cm = inputs['cloth_mask']
    im_g = inputs['grid_image']

    agnostic = agnostic[None, ...]
    c = c[None, ...]
    cm = cm[None, ...]
    im_g = im_g[None, ...]
    print(agnostic.shape, c.shape)
    grid, theta = model(agnostic, c)
    warped_cloth = F.grid_sample(c, grid, padding_mode='border')
    warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
    warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')

    return warped_cloth, warped_mask


def save_image(img_tensor, img_name, save_dir):
    tensor = (img_tensor.clone()+1)*0.5 * 255
    tensor = tensor.cpu().clamp(0,255)

    array = tensor.numpy().astype('uint8')
    if array.shape[0] == 1:
        array = array.squeeze(0)
    elif array.shape[0] == 3:
        array = array.swapaxes(0, 1).swapaxes(1, 2)
        
    img = Image.fromarray(array)
    if save_dir and img_name:
        img.save(os.path.join(save_dir, img_name))
    return img

# how to save images?
# save_image(warped_cloth.squeeze_(), 'warped_cloth.png', os.getcwd())
# save_image(warped_mask.squeeze_()*2-1, 'warped_mask.png', os.getcwd())

def tom(opt, inputs, model):
    model.eval()
        
    im_names = inputs['im_name']
    im = inputs['image']
    im_pose = inputs['pose_image']
    im_h = inputs['head']
    shape = inputs['shape']

    agnostic = inputs['agnostic']
    c = inputs['cloth']
    cm = inputs['cloth_mask']
    
    agnostic = agnostic[None, ...]
    c = c[None, ...]

    outputs = model(torch.cat([agnostic, c], 1))
    p_rendered, m_composite = torch.split(outputs, 3, 1)
    p_rendered = torch.tanh(p_rendered)
    m_composite = torch.sigmoid(m_composite)
    p_tryon = c * m_composite + p_rendered * (1 - m_composite)
    return p_tryon



names_for_test = [
    ('000048_0.jpg', '010608_1.jpg'),
    ('000048_0.jpg', "012578_1.jpg"),
    ('000048_0.jpg', "010816_1.jpg"),
    ('000048_0.jpg', "010454_1.jpg"),
]

names_for_test = [
    ('000174_0.jpg', '010608_1.jpg'),
    ('000174_0.jpg', "012578_1.jpg"),
    ('000174_0.jpg', "010816_1.jpg"),
    ('000174_0.jpg', "010454_1.jpg"),
]

# create model & train
gmm_model = GMM(opt)
load_checkpoint(gmm_model, 'checkpoints/gmm_final.pth')
tom_model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
load_checkpoint(tom_model, 'checkpoints/tom_final.pth')

for im_name, c_name in names_for_test:
    inputs = preprocess(opt, c_name, im_name, 'GMM')
    with torch.no_grad():
        warped_cloth, warped_mask = gmm(opt, inputs, gmm_model)
        cloth = save_image(warped_cloth.squeeze_(), None, None)
        cloth_mask = save_image(warped_mask.squeeze_()*2-1, None, None)

        c = transform(cloth)  # [-1,1]
        cm_array = np.array(cloth_mask)
        cm_array = (cm_array >= 128).astype(np.float32)
        cm = torch.from_numpy(cm_array) # [0,1]
        cm.unsqueeze_(0) # inplace, expand the channel

        inputs['cloth'] = c
        inputs['cloth_mask'] = cm

        tryon = tom(opt, inputs, tom_model)
        save_image(tryon.squeeze_(), 'tryon' + im_name[:-4] + c_name[:-4] + '.png', os.getcwd())