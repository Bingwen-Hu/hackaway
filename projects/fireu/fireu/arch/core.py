# Yes, for different deep learning framework, I think 
# only the network architecture is portable. That's 
# why this module exists
from collections import OrderedDict


class Arch(object):

    def __init__(self):
        self.description = "This is base class of network architecture"
        self.parameter_order = {
            'convolution' : [
                'in', # input channels, aka in_filters
                'out', # output channels, aka out_filters
                'ksize', # kernel size
                'stride', # stride
                'pad', # padding
                'dilation', # spacing between kernel points
                'group', # channels groups
                'bias', # True of False, whether to contains bias
            ],
            'pooling': ['ksize', 'stride', 'pad'],
        }
        # for shortcut
        self.conv_order = self.parameter_order['convolution']
        self.pool_order = self.parameter_order['pooling']

        # activation
        # TODO: Fix this for consistency
        self.activation = 'relu'

class PoseEstimation(Arch):
    # 在论文中，作者采用了vgg的前10个卷积层输出的特征图作为输入，并且这10个卷积层是
    # 可以finetune的。随后，采用了6个不同的stage来逐渐改进识别的效果，这种思想借鉴
    # 至Pose machine。我个人还觉得，这种做法很像resnet的残差思想。
    def __init__(self):
        super().__init__()
        self.description = ("Pose estimation model architecture for Realtime "
            "Multi-Person 2D Pose Estimation using Part Affinity Fields")
        self.conv_order = self.conv_order[:5] # only need first 5 parameters
        self.pool_order = self.pool_order[:2] # only need first 2 parameters
        self.backbone = OrderedDict(
            conv1_1 = [3, 64, 3, 1, 1],
            conv1_2 = [64, 64, 3, 1, 1],
            pool1 = [2, 2],
            conv2_1 = [64, 128, 3, 1, 1],
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

        # 在rtpose的每个stage中，都有两个分支，一个输出PAF，一个输出confidence map
        # 这里，我们叫它作CFM
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
        # stage2至stage6的网络结构完全一致，这里我们采用动态生成的方式
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
        stage2_6 = self.stage2_6
        paf, cfm = stage2_6.keys()
        paf = stage2_6[paf]
        cfm = stage2_6[cfm]
        for i in range(2, 7):
            paf_ = OrderedDict([(k.replace('i', str(i)),paf[k]) for k in paf])
            cfm_ = OrderedDict([(k.replace('i', str(i)),cfm[k]) for k in cfm])
            stage_ = OrderedDict(PAF=paf_, CFM=cfm_)
            setattr(self, f'stage{i}', stage_)
            
    def visual_arch(self):
        pass
