import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from fireu.arch import PoseEstimation as Pose


def build_block(block):
    """block is a common data structure in fireu. It is a Orderdict
    which key is name of layer and value is a list of parameters
    """
    layers = []
    for name, params in block.items():
        if 'conv' in name:
            # *params works because both parameter orders are same
            layers += [nn.Conv2d(*params)]
        elif 'pool' in name:
            layers += [nn.MaxPool2d(*params)]
            layers += [nn.ReLU(inplace=True)]
        else:
            print(f"layer {name} is skipped")
    return nn.Sequential(*layers)


class PoseNet(nn.Module):
    """pytorch implementation for original codebase
    https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation
    """

    def __init__(self, arch: Pose):
        super().__init__()
        self.arch = arch
        # bulid the network
        self.build_backbone()
        self.build_network() 
    
    def build_network(self):
        stages = [f'stage{i}' for i in range(1, 7)]
        for stage in stages:
            block = getattr(self.arch, stage)
            PAF, CFM = block.keys()
            PAF = build_block(block[PAF])
            CFM = build_block(block[CFM])
            setattr(self, f"{stage}_PAF", PAF)
            setattr(self, f"{stage}_CFM", CFM)

    def build_backbone(self):
        backbone = self.arch.backbone
        self.backbone = build_block(backbone)