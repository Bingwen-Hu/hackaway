import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class PoseArch(object):
    """Paper: https://arxiv.org/abs/1611.08050"""
    def __init__(self):
        super().__init__()
        self.description = ("Pose estimation model architecture for Realtime "
            "Multi-Person 2D Pose Estimation using Part Affinity Fields")
        self.convolution = ['channels_in', 'channels_out', 'ksize', 'stride', 'pad']
        self.pool = ['ksize', 'stride']
        self.backbone = OrderedDict(
            conv1_1 = [ 3, 64, 3, 1, 1],
            conv1_2 = [64, 64, 3, 1, 1],
            pool1 = [2, 2],
            conv2_1 = [ 64, 128, 3, 1, 1],
            conv2_2 = [128, 128, 3, 1, 1],
            pool2 = [2, 2],
            conv3_1 = [128, 256, 3, 1, 1],
            conv3_2 = [256, 256, 3, 1, 1],
            conv3_3 = [256, 256, 3, 1, 1],
            conv3_4 = [256, 256, 3, 1, 1],
            pool3 = [2, 2],
            conv4_1 = [256, 512, 3, 1, 1],
            conv4_2 = [512, 512, 3, 1, 1],
            # PE donates Pose Estimation, this two layers 
            # do not belong to VGG.
            PE0_conv4_3 = [512, 256, 3, 1, 1],
            PE0_conv4_4 = [256, 128, 3, 1, 1],
        )

        # for each stage of rtpose, there are two branchs, one output
        # PAF, the other output confidence map (CFM)
        self.stage1 = OrderedDict(
            PAF = OrderedDict(
                PE1_conv5_1_L1 = [128, 128, 3, 1, 1],
                PE1_conv5_2_L1 = [128, 128, 3, 1, 1],
                PE1_conv5_3_L1 = [128, 128, 3, 1, 1],
                PE1_conv5_4_L1 = [128, 512, 1, 1, 0],
                PE1_conv5_5_L1 = [512, 38, 1, 1, 0]),
            CFM = OrderedDict(
                PE1_conv5_1_L2 = [128, 128, 3, 1, 1],
                PE1_conv5_2_L2 = [128, 128, 3, 1, 1],
                PE1_conv5_3_L2 = [128, 128, 3, 1, 1],
                PE1_conv5_4_L2 = [128, 512, 1, 1, 0],
                PE1_conv5_5_L2 = [512, 19, 1, 1, 0]),
        )
        # define the structure and generate stage[2-6] dynamically
        self.stage2_6 = OrderedDict(
            PAF = OrderedDict(
                PEi_conv1_L1 = [185, 128, 7, 1, 3],
                PEi_conv2_L1 = [128, 128, 7, 1, 3],
                PEi_conv3_L1 = [128, 128, 7, 1, 3],
                PEi_conv4_L1 = [128, 128, 7, 1, 3],
                PEi_conv5_L1 = [128, 128, 7, 1, 3],
                PEi_conv6_L1 = [128, 128, 1, 1, 0],
                PEi_conv7_L1 = [128, 38, 1, 1, 0]),
            CFM = OrderedDict(
                PEi_conv1_L2 = [185, 128, 7, 1, 3],
                PEi_conv2_L2 = [128, 128, 7, 1, 3],
                PEi_conv3_L2 = [128, 128, 7, 1, 3],
                PEi_conv4_L2 = [128, 128, 7, 1, 3],
                PEi_conv5_L2 = [128, 128, 7, 1, 3],
                PEi_conv6_L2 = [128, 128, 1, 1, 0],
                PEi_conv7_L2 = [128, 19, 1, 1, 0]
            )
        )
        self.build_stage2_6()
        
    def build_stage2_6(self):
        """Create attribute for self, makes self.stage[2-6] available """
        paf, cfm = self.stage2_6.values()
        for i in range(2, 7):
            paf_ = OrderedDict([(k.replace('i', str(i)),paf[k]) for k in paf])
            cfm_ = OrderedDict([(k.replace('i', str(i)),cfm[k]) for k in cfm])
            stage_ = OrderedDict(PAF=paf_, CFM=cfm_)
            setattr(self, f'stage{i}', stage_)



def build_blocks(blocks, part):
    """
    Args: 
        blocks: Orderdict whose key is name of layer and 
            value is a list of parameters.
        part: either `backbone` or `head`
    Returns:
        nn.Sequential contains layers defined in blocks
    """
    assert part in ["backbone", "head"]

    layers = []
    for name, params in blocks.items():
        if 'conv' in name:
            # *params works because both parameter orders are same
            layers += [nn.Conv2d(*params)]
            layers += [nn.ReLU(inplace=True)]
        elif 'pool' in name:
            layers += [nn.MaxPool2d(*params)]
        else:
            print(f"layer {name} is skipped")
    # if we are building head of network, we don't need the last ReLU
    # layer. Why? Because the original author have not include it! :-)
    if part == 'head':
        layers = layers[:-1]
    return nn.Sequential(*layers)


class PoseNet(nn.Module):
    """pytorch implementation for original codebase
    https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation
    """

    def __init__(self):
        """Build the network architecture """
        super().__init__()
        self.arch = PoseArch()
        self.build_backbone()
        self.build_head() 
    
    def build_backbone(self):
        """build the backbone of the whole network
        network = backbone + head 
        """
        backbone = self.arch.backbone
        self.backbone = build_blocks(backbone, 'backbone')

    def build_head(self):
        """build the head of the whole network
        network = backbone + head
        """
        stages = [f'stage{i}' for i in range(1, 7)]
        for stage in stages:
            block = getattr(self.arch, stage)
            PAF, CFM = block.keys()
            PAF = build_blocks(block[PAF], 'head')
            CFM = build_blocks(block[CFM], 'head')
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
        PAF_5 = self.stage5_PAF(out_4)
        CFM_5 = self.stage5_CFM(out_4)
        out_5 = torch.cat([PAF_5, CFM_5, filter_map], dim=1)
        # stage6
        PAF_6 = self.stage6_PAF(out_5)
        CFM_6 = self.stage6_CFM(out_5)
        
        # because loss is computed in every stage, 
        # so we need to return all of them
        PAFs = [PAF_1, PAF_2, PAF_3, PAF_4, PAF_5, PAF_6]
        CFMs = [CFM_1, CFM_2, CFM_3, CFM_4, CFM_5, CFM_6]

        return PAFs, CFMs

if __name__ == '__main__':
    net = PoseNet()
    net.load_state_dict(torch.load('weights/rtpose_sd.pth'))
    net.eval()
    print(f"Test file: {__file__} ... ok")