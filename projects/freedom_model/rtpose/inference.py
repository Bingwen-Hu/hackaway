import torch

from rtpose import PoseNet, Pose
from freedom.data.coco import KeyPointTest

if __name__ == "__main__":
    net = PoseNet(Pose)
    net.load_state_dict(torch.load('weights/rtpose_sd.pth'))

    