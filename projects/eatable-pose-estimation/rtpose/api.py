import cv2
import torch

from rtpose import PoseNet
from epc import KeyPointTest
from epc import KeyPointParams

# load model
net = PoseNet()
net.load_state_dict(torch.load('weights/rtpose_sd.pth'))
net.eval()

# inference setup
params = KeyPointParams()
infer = KeyPointTest(params)


# pytorch specific input preprocess
def input_prepare(im):
    im_prep = infer.preprocess(im)
    im_prep = im_prep.transpose(2, 0, 1)
    im_tensor = torch.Tensor(im_prep[None, ...])
    return im_tensor


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        impath = sys.argv[1]
    else:
        impath = "imgs/dinner.png"

    im = cv2.imread(impath)
    im_tensor = input_prepare(im)
    with torch.no_grad():
        PAFs, CFMs = net(im_tensor)
    pafs, heatmaps = PAFs[-1], CFMs[-1]
    pafs = pafs.numpy().squeeze().transpose(1, 2, 0)
    heatmaps = heatmaps.numpy().squeeze().transpose(1, 2, 0)
    #scale to inference size
    parts_list, persons = infer.postprocess(im, pafs, heatmaps)
    canvas = infer.plot_pose(im, parts_list, persons)
    cv2.imwrite('test.png', canvas)