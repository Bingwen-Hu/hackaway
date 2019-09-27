# Second stage
import os
import os.path as osp
import uuid

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw
from easydict import EasyDict

import psp
import rtpose
from networks import GMM, UnetGenerator, load_checkpoint

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


def preprocess(cpath, impath):
    """Preprocess to generate person representation

    Args:
        cpath: cloth image path 
        impath: person image path
    """
    c = Image.open(cpath)
    cm = c.convert('L')
    cm_array = np.array(cm)
    cm_array = np.where(cm_array > 220, 0, 255).astype(np.float32)
       
    c = transform(c)  # [-1,1]
    cm = torch.from_numpy(cm_array) # [0,1]
    cm = cm[None, ...]

    im = Image.open(impath)
    im = transform(im) # [-1,1]

    parse_array = psp.parse(impath)
    # NOTE: Test whether human parsing is failed
    # filename = osp.basename(impath).replace('jpg', 'png')
    # image_parse = osp.join(opt.data_path, 'image-parse', filename)
    # parse_array = np.array(Image.open(image_parse))
    # 所有大于0的即是身体
    parse_shape = (parse_array > 0).astype(np.float32)
    # 头部由多种组成，可能是头发，帽子和脸等
    parse_head = (parse_array == 1).astype(np.float32) + \
            (parse_array == 2).astype(np.float32) + \
            (parse_array == 4).astype(np.float32) + \
            (parse_array == 13).astype(np.float32)

    # shape downsample
    # 先放缩到一个低精度的图片，再放回原来的大小，使其模糊化
    parse_shape = Image.fromarray((parse_shape*255).astype(np.uint8))
    parse_shape = parse_shape.resize((opt.fine_width//16, opt.fine_height//16), Image.BILINEAR)
    parse_shape = parse_shape.resize((opt.fine_width, opt.fine_height), Image.BILINEAR)
    shape = transform(parse_shape) # 各种规范化

    phead = torch.from_numpy(parse_head) # [0,1]
    # pcm = torch.from_numpy(parse_cloth) # [0,1]

    # upper cloth
    # im_c = im * pcm + (1 - pcm) # [-1,1], fill 1 for other parts，除了pcm，其他都变成1
    im_h = im * phead - (1 - phead) # [-1,1], fill 0 for other parts

    _, pose_label = rtpose.estimation(impath)
    pose_data = pose_label['people'][0]
    pose_data = np.array(pose_data)
    pose_data = pose_data.reshape((-1,3))

    point_num = pose_data.shape[0]
    pose_map = torch.zeros(point_num, opt.fine_height, opt.fine_width)
    r = opt.radius
    for i in range(point_num):
        one_map = Image.new('L', (opt.fine_width, opt.fine_height))
        draw = ImageDraw.Draw(one_map)
        pointx = pose_data[i,0]
        pointy = pose_data[i,1]
        if pointx > 1 and pointy > 1:
            draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
        one_map = transform(one_map)
        pose_map[i] = one_map[0]

    # cloth-agnostic representation
    # shape: 指的是身体形状，且是模糊的
    # im_h：指的是头部的图像
    # pose_map：指的是姿态的各个通道
    agnostic = torch.cat([shape, im_h, pose_map], 0) 


    result = {
        'cloth': c,          # for input
        'cloth_mask': cm,   # for input
        'agnostic': agnostic,   # for input
    }

    return result


def gmm(opt, inputs, model):
    model.eval()

    agnostic = inputs['agnostic']
    c = inputs['cloth']
    cm = inputs['cloth_mask']

    agnostic = agnostic[None, ...]
    c = c[None, ...]
    cm = cm[None, ...]
 
    grid, theta = model(agnostic, c)
    warped_cloth = F.grid_sample(c, grid, padding_mode='border')
    warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')

    return warped_cloth, warped_mask


def save_image(img_tensor, img_path=None):
    tensor = (img_tensor.clone()+1)*0.5 * 255
    tensor = tensor.cpu().clamp(0,255)

    array = tensor.numpy().astype('uint8')
    if array.shape[0] == 1:
        array = array.squeeze(0)
    elif array.shape[0] == 3:
        array = array.swapaxes(0, 1).swapaxes(1, 2)
        
    img = Image.fromarray(array)
    if img_path:
        img.save(img_path)
    return img


def tom(opt, inputs, model):
    model.eval()
        
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



# create model & train
gmm_model = GMM(opt)
load_checkpoint(gmm_model, 'checkpoints/gmm_final.pth')
tom_model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
load_checkpoint(tom_model, 'checkpoints/tom_final.pth')



def tryon(cloth, person, target_dir='./'):
    """试衣镜功能函数 v0.1

    Args:
        cloth: 字符串或者内存文件
        person: 字符串或者内存文件
    Returns:
        tryon_path: 生成图片的路径
    """
    inputs = preprocess(cloth, person)
    with torch.no_grad():
        warped_cloth, warped_mask = gmm(opt, inputs, gmm_model)
        cloth = save_image(warped_cloth.squeeze_(), None)
        cloth_mask = save_image(warped_mask.squeeze_()*2-1, None)

        c = transform(cloth)  # [-1,1]
        cm_array = np.array(cloth_mask)
        cm_array = (cm_array >= 128).astype(np.float32)
        cm = torch.from_numpy(cm_array) # [0,1]
        cm.unsqueeze_(0) # inplace, expand the channel

        inputs['cloth'] = c
        inputs['cloth_mask'] = cm

        output = tom(opt, inputs, tom_model)
        tryon_path = osp.join(target_dir, f"{uuid.uuid1()}.png")
        tryon_image = save_image(output.squeeze_(), tryon_path)
        return tryon_path


if __name__ == '__main__':
    tests = [
        ('000048_0.jpg', '010608_1.jpg'),
        ('000048_0.jpg', "012578_1.jpg"),
        ('000048_0.jpg', "010816_1.jpg"),
        ('000048_0.jpg', "010454_1.jpg"),
    ]
    for person, cloth in tests:
        person = osp.join(opt.data_path, 'image', person)
        cloth = osp.join(opt.data_path, 'cloth', cloth)
        save_path = tryon(cloth, person)
        print(save_path)