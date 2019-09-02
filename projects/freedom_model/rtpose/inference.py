import cv2
import torch


from rtpose import PoseNet, Pose
from freedom.data.coco import KeyPointTest
from freedom.data.coco.params import KeyPointParams

if __name__ == "__main__":
    net = PoseNet(Pose)
    net.load_state_dict(torch.load('weights/rtpose_sd.pth'))
    net.eval()
    params = KeyPointParams()
    infer = KeyPointTest(params)

    impath = "imgs/dinner.png"
    im = cv2.imread(impath)

    im_resize = infer.im_letterbox(im, 
        infer.params.infer_insize, infer.params.stride)
    im_prep = infer.im_preprocess(im_resize, True)
    im_tensor = torch.Tensor(im_prep[None, ...])
    with torch.no_grad():
        PAFs, CFMs = net(im_tensor)
    pafs, heatmaps = PAFs[-1], CFMs[-1]
    pafs = pafs.numpy().squeeze().transpose(1, 2, 0)
    heatmaps = heatmaps.numpy().squeeze().transpose(1, 2, 0)
    #scale to inference size
    heatmaps = infer.im_letterbox(heatmaps, 
        infer.params.heatmap_size, infer.params.stride)
    pafs = infer.im_letterbox(pafs, 
        infer.params.heatmap_size, infer.params.stride)
  
    parts_list, persons = infer.pose_decode(im, heatmaps, pafs)
    for part in parts_list:
        print(part)
    canvas = infer.plot_pose(im, parts_list, persons)
    cv2.imwrite('test.png', canvas)