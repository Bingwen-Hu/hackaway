import cv2
import torch


from rtpose import PoseNet, Pose
from freedom.data.coco import KeyPointTest
from freedom.data.coco.params import KeyPointParams

if __name__ == "__main__":
    net = PoseNet(Pose)
    net.load_state_dict(torch.load('weights/rtpose_sd.pth'))

    params = KeyPointParams()
    params.limb_threshold
    infer = KeyPointTest(params)

    impath = "imgs/dinner.png"
    im = cv2.imread(impath)

    im_resize = infer.im_letterbox(im, 
        infer.params.infer_insize, infer.params.stride)
    im_prep = infer.im_preprocess(im_resize, True)
    im_tensor = torch.from_numpy(im_prep[None, ...])
    with torch.no_grad():
        PAFs, CFMs = net(im_tensor)
    pafs, heatmaps = PAFs[-1], CFMs[-1]
    pafs = pafs.numpy().squeeze().transpose(1, 2, 0)
    heatmaps = heatmaps.numpy().squeeze().transpose(1, 2, 0)
    # scale to inference size
    # heatmaps = infer.im_letterbox(heatmaps, 
    #     infer.params.heatmap_size, infer.params.stride)
    # pafs = infer.im_letterbox(pafs, 
    #     infer.params.heatmap_size, infer.params.stride)
    #heatmaps = cv2.resize(heatmaps, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    #pafs = cv2.resize(pafs, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    #heatmaps = cv2.resize(heatmaps, im.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)

    # parts_list, persons = infer.pose_decode(im, heatmaps, pafs)
    # for part in parts_list:
    #     print(part)
    # canvas = infer.plot_pose(im, parts_list, persons)
    # cv2.imwrite('test.png', canvas)

    # test
    # person_list = list(persons.values())
    # num = len(person_list[0])
    # for p in person_list: 
    #     for i in range(num-2): 
    #         top = str(int(p[i])) if p[i] != -1 else " "  
    #         print(top,  end='\t|') 
    #     print() 