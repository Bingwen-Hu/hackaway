import argparse
import os
import cv2
import numpy as np
import torch
import torch.optim as optim

from .model.dataset.factory import get_imdb
from .model.utils.config import cfg
from .model.roi_data_layer.layer import RoIDataLayer
from .model.SSH import SSH
from .model.network import save_check_point, load_check_point
from .model.nms.nms_wrapper import nms
from .model.utils.test_utils import _get_image_blob, _compute_scaling_factor, visusalize_detections



thresh = 0.6
gpu = None
cwd = os.path.dirname(__file__)
saved_model_path = os.path.join(cwd, 'check_point/check_point.zip')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

def load_model():
    global device
    net = SSH(vgg16_image_net=False)
    check_point = load_check_point(saved_model_path)
    net.load_state_dict(check_point['model_state_dict'])
    net.to(device)
    net.eval()
    return net

net = load_model()


scales = {
    'fastest': [50, 66],
    'faster': [100, 133],
    'fast': [300, 400],
    'normal': [600, 800],
    'large': [1200, 1600],
}


def detect(im, scale_mode='faster', margin=0):
    """detect face on image
    Args:
        im: path of image or numpy-format image 
        scale_mode: scale of input image, larger the mode, 
            more accurate the detection, but slower
    Returns:
        numpy-array detection results, for example:
            [
                [x1, y1, x2, y2, confidence],
                [x1, y1, x2, y2, confidence],
            ]
    """
    global device
    if type(im) == str:
        im = cv2.imread(im)

    assert scale_mode in ('fastest', 'faster', 'fast', 'normal', 'large'), ("only `fastest`, " 
            "`faster`, 'fast', `normal`, `large` support")
    scale, max_size = scales[scale_mode]
    with torch.no_grad():
        # im_scale = _compute_scaling_factor(im.shape, cfg.TEST.SCALES[0], cfg.TEST.MAX_SIZE)
        im_scale = _compute_scaling_factor(im.shape, scale, max_size)
        im_blob = _get_image_blob(im, [im_scale])[0]
        im_info = np.array([[im_blob['data'].shape[2], im_blob['data'].shape[3], im_scale]])
        im_data = im_blob['data']
        im_info = torch.from_numpy(im_info).to(device)
        im_data = torch.from_numpy(im_data).to(device)
        batch_size = im_data.size()[0]
        ssh_rois = net(im_data, im_info)
        inds = (ssh_rois[:, :, 4] > thresh)
        ssh_roi_keep = ssh_rois[:, inds[0], :]
        ssh_roi_keep[:, :, 0:4] /= im_scale

        for i in range(batch_size):
            ssh_roi_single = ssh_roi_keep[i].cpu().numpy()
            nms_keep = nms(ssh_roi_single, cfg.TEST.RPN_NMS_THRESH, force_cpu=True)
            cls_dets_single = ssh_roi_single[nms_keep, :]
    detections = cls_dets_single.tolist()
    return detections


def draw(im, detections):
    h, w = im.shape[:2]
    def draw(bbox):
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(w, int(bbox[3]))
        cv2.rectangle(im, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
    list(map(lambda x: draw(x[:4]), detections))
    return im
 

def show(im):
    if type(im) == str:
        im = cv2.imread(im)
    detections = detect(im)
    im = draw(im, detections)
    cv2.imshow("face SSH", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
