import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from freedom.arch import PoseEstimation as Pose


def build_block(block):
    """block is a common data structure in fireu. It is a Orderdict
    which key is name of layer and value is a list of parameters
    """
    layers = []
    for name, params in block.items():
        if 'conv' in name:
            # *params works because both parameter orders are same
            layers += [nn.Conv2d(*params)]
            layers += [nn.ReLU(inplace=True)]
        elif 'pool' in name:
            layers += [nn.MaxPool2d(*params)]
        else:
            print(f"layer {name} is skipped")
    return nn.Sequential(*layers)


class PoseNet(nn.Module):
    """pytorch implementation for original codebase
    https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation
    """

    def __init__(self, arch: Pose):
        """Build the network architecture
        Args:
            Pose: Type of architecture 
        """
        super().__init__()
        self.arch = arch()
        # bulid the network, if you are confused,
        # refer to the forward method, which show 
        # the whole network architecture
        self.build_backbone()
        self.build_head() 
    
    def build_backbone(self):
        """build the backbone of the whole network
        network = backbone + head 
        """
        backbone = self.arch.backbone
        self.backbone = build_block(backbone)

    def build_head(self):
        """build the head of the whole network
        network = backbone + head
        """
        stages = [f'stage{i}' for i in range(1, 7)]
        for stage in stages:
            block = getattr(self.arch, stage)
            PAF, CFM = block.keys()
            PAF = build_block(block[PAF])
            CFM = build_block(block[CFM])
            setattr(self, f"{stage}_PAF", PAF)
            setattr(self, f"{stage}_CFM", CFM)

    def forward(self, x):
        filter_map = self.backbone(x)
        # stage1
        PAF_1 = self.stage1_PAF(filter_map)
        CFM_1 = self.stage1_CFM(filter_map)
        out_1 = torch.cat([PAF_1, CFM_1, filter_map], dim=1)
        # stage2
        PAF_2 = self.stage2_PAF(out_1)
        CFM_2 = self.stage2_CFM(out_1)
        out_2 = torch.cat([PAF_2, CFM_2, filter_map], dim=1)
        # stage3
        PAF_3 = self.stage3_PAF(out_2)
        CFM_3 = self.stage3_CFM(out_2)
        out_3 = torch.cat([PAF_3, CFM_3, filter_map], dim=1)
        # stage4
        PAF_4 = self.stage4_PAF(out_3)
        CFM_4 = self.stage4_CFM(out_3)
        out_4 = torch.cat([PAF_4, CFM_4, filter_map], dim=1)
        # stage5
        PAF_5 = self.stage3_PAF(out_4)
        CFM_5 = self.stage3_CFM(out_4)
        out_5 = torch.cat([PAF_5, CFM_5, filter_map], dim=1)
        # stage6
        PAF_6 = self.stage3_PAF(out_5)
        CFM_6 = self.stage3_CFM(out_5)
        
        # because loss is computed in every stage, 
        # so we need to return all of them
        PAFs = [PAF_1, PAF_2, PAF_3, PAF_4, PAF_5, PAF_6]
        CFMs = [CFM_1, CFM_2, CFM_3, CFM_4, CFM_5, CFM_6]

        return PAFs, CFMs

if __name__ == '__main__':
    net = PoseNet(Pose)
    net.load_state_dict(torch.load('weights/rtpose_sd.pth'))