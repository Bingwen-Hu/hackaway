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

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        impath = sys.argv[1]
    else:
        impath = "imgs/dinner.png"

    im = cv2.imread(impath)
    im_resize = infer.im_letterbox(im, 
        infer.params.infer_insize, infer.params.stride)
    im_prep = infer.im_preprocess(im_resize)
    im_prep = im_prep.transpose(2, 0, 1)
    im_tensor = torch.Tensor(im_prep[None, ...])
    with torch.no_grad():
        PAFs, CFMs = net(im_tensor)
    pafs, heatmaps = PAFs[-1], CFMs[-1]
    pafs = pafs.numpy().squeeze().transpose(1, 2, 0)
    heatmaps = heatmaps.numpy().squeeze().transpose(1, 2, 0)
    #scale to inference size
    heatmaps_scale = infer.im_letterbox(heatmaps, 
        infer.params.heatmap_size, infer.params.stride)
    pafs_scale = infer.im_letterbox(pafs, 
        infer.params.heatmap_size, infer.params.stride)
  
    parts_list, persons = infer.pose_decode(im, heatmaps_scale, pafs_scale)
    canvas = infer.plot_pose(im, parts_list, persons)
    cv2.imwrite('test.png', canvas)