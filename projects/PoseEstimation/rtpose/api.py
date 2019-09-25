import os.path as osp
import cv2
import torch

from .models import PoseNet
from .rtpose import KeyPointTest
from .rtpose import KeyPointParams

# load model
net = PoseNet()
weights = osp.join(osp.dirname(__file__), 'weights/rtpose_sd.pth')
net.load_state_dict(torch.load(weights))
net.eval()

# inference setup
params = KeyPointParams()
KP = KeyPointTest(params)


# pytorch specific input preprocess
def input_prepare(im):
    im_prep = KP.preprocess(im)
    im_prep = im_prep.transpose(2, 0, 1)
    im_tensor = torch.Tensor(im_prep[None, ...])
    return im_tensor


def pose_estimation(im):
    """
    Args:
        im: string or image object return by cv2.imread 
    Returns:
        canvas with keypoint and keypoint in openpose format
    """
    if type(im) == str:
        im = cv2.imread(im)

    im_tensor = input_prepare(im)
    with torch.no_grad():
        PAFs, CFMs = net(im_tensor)
    pafs, heatmaps = PAFs[-1], CFMs[-1]
    pafs = pafs.numpy().squeeze().transpose(1, 2, 0)
    heatmaps = heatmaps.numpy().squeeze().transpose(1, 2, 0)
    #scale to inference size
    parts_list, persons = KP.postprocess(im, pafs, heatmaps)
    canvas = KP.plot_pose(im, parts_list, persons)
    # save as openpose format
    keypoints = KP.openpose(im, parts_list, persons)
    return canvas, keypoints


if __name__ == '__main__':
    import json

    im = 'imgs/dinner.png'
    canvas, keypoints = pose_estimation(im)
    cv2.imwrite("result.png", canvas)
    with open("keypoint.json", 'w') as f:
        json.dump(keypoints, f)