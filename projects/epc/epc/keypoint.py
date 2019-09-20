import os
import os.path as osp
import enum
from collections import OrderedDict

# third party import
import cv2
import numpy as np
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import gaussian_filter
from scipy.ndimage import maximum_filter

# local import
from .base import Arch, Parameter, COCO


# 在论文中，作者采用了vgg的前10个卷积层输出的特征图作为输入，并且这10个卷积层是
# 可以finetune的。随后，采用了6个不同的stage来逐渐改进识别的效果，这种思想借鉴
# 至Pose machine。我个人还觉得，这种做法很像resnet的残差思想。
class PoseEstimation(Arch):
    """Paper: https://arxiv.org/abs/1611.08050"""
    def __init__(self):
        super().__init__()
        self.description = ("Pose estimation model architecture for Realtime "
            "Multi-Person 2D Pose Estimation using Part Affinity Fields")
        self.convolution = self.convolution[:5] # only need first 5 parameters
        self.pool = self.pool[:2] # only need first 2 parameters
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

        # for each stage of rtpose, there are two branchs, one output
        # PAF, the other output confidence map (CFM)
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
        # define the structure and generate stage[2-6] dynamically
        # stage2至stage6的网络结构完全一致，这里我们采用动态生成的方式
        # 这里的i代码第i个stage
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
        stage2_6 = self.stage2_6
        paf, cfm = stage2_6.keys()
        paf = stage2_6[paf]
        cfm = stage2_6[cfm]
        for i in range(2, 7):
            paf_ = OrderedDict([(k.replace('i', str(i)),paf[k]) for k in paf])
            cfm_ = OrderedDict([(k.replace('i', str(i)),cfm[k]) for k in cfm])
            stage_ = OrderedDict(PAF=paf_, CFM=cfm_)
            setattr(self, f'stage{i}', stage_)


class Joint(enum.IntEnum):
    """Joint type for pose estimation task, we define our 
    own keypoints order and joint type, which is different
    from COCO's. 

    MS COCO annotation order:
    0: nose         1: l eye        2: r eye    3: l ear    4: r ear
    5: l shoulder   6: r shoulder   7: l elbow  8: r elbow
    9: l wrist      10: r wrist     11: l hip   12: r hip   
    13: l knee      14: r knee      15: l ankle 16: r ankle

    NOTE: R means right, L means left. There are 17 parts 
    in MS COCO dataset. Following authors' implementation, 
    we compute neck position according to shoulders.
    """
    Nose = 0 
    Neck = 1 
    RShoulder = 2; RElbow = 3; RWrist = 4
    LShoulder = 5; LElbow = 6; LWrist = 7
    RHip = 8;   RKnee = 9;  RAnkle = 10
    LHip = 11;  LKnee = 12; LAnkle = 13
    REye = 14;  LEye = 15 
    REar = 16;  LEar = 17


class KeyPointParams(Parameter):

    # Training Parameters
    insize = 368
    min_area = 32 * 32
    min_keypoints = 5
    heatmap_sigma = 7
    paf_sigma = 8 # aka `limb width`

    # Inference Parameters
    infer_insize = 368
    infer_scales = [0.5, 1.0, 1.5, 2.0]
    heatmap_size = 320
    stride = 8

    # for gaussian filter smoothing
    smooth = False
    gaussian_sigma = 2.5
    # for peak selection
    peak_threshold = 0.1
    # for peak refinement
    peak_refine = True
    winsize = 2
    # for PAFs evaluate
    nb_sample = 10
    nb_sample_threshold = 8
    limb_threshold = 0.05
    # for person parsing
    mean_score_threshold = 0.2
    nb_part_threshold = 3
   
    # specific params
    joint = Joint
    # In MS COCO, this is called skeleton
    # What we define here is different.
    limbs = [
        [Joint.Neck, Joint.RHip],
        [Joint.RHip, Joint.RKnee],
        [Joint.RKnee, Joint.RAnkle],

        [Joint.Neck, Joint.LHip],
        [Joint.LHip, Joint.LKnee],
        [Joint.LKnee, Joint.LAnkle],

        [Joint.Neck, Joint.RShoulder],
        [Joint.RShoulder, Joint.RElbow],
        [Joint.RElbow, Joint.RWrist],
        [Joint.RShoulder, Joint.REar],

        [Joint.Neck, Joint.LShoulder],
        [Joint.LShoulder, Joint.LElbow],
        [Joint.LElbow, Joint.LWrist],
        [Joint.LShoulder, Joint.LEar],

        [Joint.Neck, Joint.Nose],
        [Joint.Nose, Joint.REye],
        [Joint.Nose, Joint.LEye],
        [Joint.REye, Joint.REar],
        [Joint.LEye, Joint.LEar],
    ]

    # original coco keypoint order, here we use 
    # Joint to represent it.
    coco_keypoints = [
        Joint.Nose,
        Joint.LEye,
        Joint.REye,
        Joint.LEar,
        Joint.REar,
        Joint.LShoulder,
        Joint.RShoulder,
        Joint.LElbow,
        Joint.RElbow,
        Joint.LWrist,
        Joint.RWrist,
        Joint.LHip,
        Joint.RHip,
        Joint.LKnee,
        Joint.RKnee,
        Joint.LAnkle,
        Joint.RAnkle,
    ]

    # limb color for visualization
    colors = [
        [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
        [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
        [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
        [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 0]
    ]


class KeyPointMixin(object):
    """This mixin class provide several common functions for Training and 
    Testing. They are image preprocessing and visualization utilities.
    """
    @staticmethod
    def im_preprocess(im):
        """image pre-processing, scale image by divide 255,
        centre image by substract 0.5
        
        Args:
            im: image object return by cv2.imread
            channel_first: if True, put channels on axis 0
        Returns:
            the preprocessed image
        """
        im = im / 255.0
        im -= 0.5
        return im

    @staticmethod
    def im_letterbox(im, tsize, stride):
        """compute the suitable shape of for resize

        Args:
            im: image object return by cv2.imread
            tsize: short for target size
            stride: decided by network, total downscale of input and output
        Returns:
            resized image 
        """
        im_h, im_w = im.shape[:2]

        if im_h < im_w:
            im_w = round(tsize / im_h * im_w)
            # 因为不想用if，所以用个trick，以下的注释解释了下面代码的行为 
            # NOTE: the following code do the same things with `if`
            # surplus = im_w % stride
            # if surplus != 0:
            #     im_w += stride - surplus
            surplus = (stride - (im_w % stride)) % stride
            im_w += surplus
            im_h = tsize
        else:
            im_h = round(tsize / im_w * im_h)
            surplus = (stride - (im_h % stride)) % stride
            im_h += surplus
            im_w = tsize

        im_ = cv2.resize(im, (im_w, im_h), interpolation=cv2.INTER_CUBIC)
        return im_

    @staticmethod 
    def plot_ignore_mask(im, mask, color=(0, 0, 1)):
        """visualize mask genrated by Class Keypoint
        
        Args:
            im: numpy.array return by cv2.imread
            mask: generated ignore_mask
            color: for BGR channels, 0 means ignore, 1 means keep 
        Returns:
            im: image with highlighted mask region
        """
        # expand to 3 channels
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        # turn to 0s and 1s
        mask = mask.astype(np.int32)
        # only segmentation area is kept
        im_masked = im * mask
        # keep certain channels according to color
        # change 1s to 255s
        for i in range(3):
            mask[:, :, i] = mask[:, :, i] * color[i] * 255
        # 对于非mask区域，原图片是没有影响的。对于mask区域，保留它原来
        # 颜色的30%，然后加上70%的color指定的颜色
        # for masked region, keep its original color 30%, then 
        # added 70% mask color
        im = im - im_masked * 0.7 + mask * 0.7

        return im.astype(np.uint8)

    @staticmethod
    def overlay_PAF(im, paf):
        """Pad PAF layer on image"""
        # 首先，创建HSV层
        x, y = paf
        # 这一句是什么意思呢？看不懂
        hue = np.arctan2(y, x) / np.pi / -2 + 0.5
        # 饱和度的取值为0-1
        saturation = np.sqrt(x ** 2 + y ** 2)
        saturation[saturation > 1.0] = 1.0
        value = saturation.copy()
        # None的作用等同np.newaxis
        hsv_paf = np.vstack([hue[None], saturation[None], value[None]]).transpose(1, 2, 0)
        hsv_paf = (hsv_paf * 255).astype(np.uint8)
        rgb_paf = cv2.cvtColor(hsv_paf, cv2.COLOR_HSV2BGR)
        # alpha = 0.6, beta = 0.4, gamma = 0
        im = cv2.addWeighted(im, 0.6, rgb_paf, 0.4, 0)
        return im
    
    @staticmethod
    def overlay_PAFs(im, pafs): 
        # 这里的思路是先将PAFs合成一个，然后再与im合并
        mix_paf = np.zeros((2,) + im.shape[:2])
        paf_overlay = np.zeros_like(mix_paf)

        channels, height, width = pafs.shape
        new_shape = channels // 2, 2, height, width
        pafs = pafs.reshape(new_shape)
        for paf in pafs:
            mix_paf += paf
            paf_overlay = paf != 0
            paf_overlay += np.broadcast_to(paf_overlay[0] | paf_overlay[1], paf.shape)
        mix_paf[paf_overlay > 0] /= paf_overlay[paf_overlay > 0]
        im = KeyPointMixin.overlay_PAF(im, mix_paf)
        return im

    @staticmethod
    def overlay_heatmap(im, heatmap):
        heatmap = (heatmap * 255).astype(np.uint8)
        rgb_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        im = cv2.addWeighted(im, 0.6, rgb_heatmap, 0.4, 0)
        return im
    
    @staticmethod
    def overlay_ignore_mask(im, mask):
        """set ignore region dark"""
        mask = (mask == 0).astype(np.uint8)
        im = im * np.repeat(mask[:, :, None], 3, axis=2)
        return im

class KeyPointTrain(COCO, KeyPointMixin):
    """A class for pose estimation, especially for this paper
    link: https://arxiv.org/abs/1611.08050.

    NOTE: In this class, I use `heatmap` or `confidence map` to represent the
    concept of confidence map in the Paper. Heatmap emphasizes concrete details
    while confidence map emphasizes the meaning of concept.
    """
    def __init__(self, images_directory, annotation_file, params: KeyPointParams):
        """
        Args:
            images_directory: same as COCO
            annotations_file: same as COCO
            params: KeyPointParams instance 
        """
        super().__init__(images_directory, annotation_file)
        self.params = params
        self.mask_dir = osp.join(self.root, 'mask')
        # create mask directory because we need mask to train
        os.makedirs(self.mask_dir, exist_ok=True)
    
    # -------- Preprocessing --------
    def convert_joint_order(self, ann_metas):
        """convert the joint order from COCO to ours

        Args:
            ann_metas: annotations of COCO
        Params:
            joint: Joint type.
        Returns:
            a numpy array (N, 18, 3) encodes keypoints for single image
            where N is the number of persons.
        """ 
        joint = self.params.joint
        joint_len = len(joint)
        poses = np.zeros((0, joint_len, 3))

        for ann in ann_metas:
            ann_pose = np.array(ann['keypoints']).reshape(-1, 3)
            # pose for single person
            # 这里创建一个存放单个人关键点的array (J, 3)，J是关键点的数目
            pose = np.zeros((joint_len, 3), dtype=np.int32)

            # 将COCO的顺序映射为我们自己的顺序
            for i, joint_i in enumerate(self.params.coco_keypoints):
                pose[joint_i] = ann_pose[i]
            # 通过取左右双肩的平均，计算出脖子的位置
            if pose[joint.LShoulder][2] > 0 and pose[joint.RShoulder][2] > 0:
                pose[joint.Neck][0] = int((pose[joint.LShoulder][0] + pose[joint.RShoulder][0])/2)
                pose[joint.Neck][1] = int((pose[joint.LShoulder][1] + pose[joint.RShoulder][1])/2)
                pose[joint.Neck][2] = 2
            # 为了能叠加到poses里面，需要reshape保持维度一致
            pose = pose.reshape(-1, joint_len, 3)
            poses = np.vstack([poses, pose])
        return poses

    def make_confidence_map(self, shape, peak):
        """create a confidence map (a.k.a. heatmap).

        Args:
            shape: shape of confidence map 
            peak: a.k.a keypoint, (x, y)
        Params:
            sigma: the hyperparameter control the spread of peak
        Returns:
            Confidence map with shape
        """
        x, y = peak
        # grid_x是所有点的x坐标，grid_y是所有点的y坐标，这里的x、y是从图片的
        # 像素坐标的角度来理解
        grid_x = np.tile(np.arange(shape[1]), (shape[0], 1))
        grid_y = np.tile(np.arange(shape[0]), (shape[1], 1)).transpose()
        grid_dist = (grid_x - x) ** 2 + (grid_y - y) ** 2
        # NOTE: 这里并没有对heatmap的最大值进行约束，而原作caffe中有
        heatmap = np.exp(-0.5 * grid_dist / self.params.heatmap_sigma ** 2)
        return heatmap

    def make_confidence_maps(self, shape, poses):
        """generate confidence map for single image
        
        Args:
            shape: shape of confidence map 
            poses: poses for im, return by convert_joint_order
        Returns:
            comfidence map with shape (len(joint), h, w)
        """
        # init heatmaps as (0, h, w) 
        heatmaps = np.zeros([0, *shape])
        # 为了计算背景，需要把所有heatmap进行累加
        heatmap_sum = np.zeros(shape)
        # 这里的思路是，将所有相同关键点放到一个heatmap中
        # 所以是先遍历关键点，再遍历每个人的pose
        # 因为Joint是整型枚举型数据，所以可以直接当整数用
        for joint_i in self.params.joint:
            heatmap = np.zeros(shape)
            for pose in poses:
                # 查看每一个关节点的v值，大于0说明该点有标签
                if pose[joint_i, 2] > 0:
                    jointmap = self.make_confidence_map(shape, pose[joint_i][:2])
                    # 这里是论文中的对不同人的关节点的max运算，以保留峰值
                    heatmap[jointmap > heatmap] = jointmap[jointmap > heatmap]
                    heatmap_sum[jointmap > heatmap_sum] = jointmap[jointmap > heatmap_sum]
            # 将这个关键点的热力图添加进返回值中
            heatmaps = np.vstack([heatmaps, heatmap.reshape([1, *shape])])
        heatmap_bg = (1 - heatmap_sum).reshape([1, *shape]) # 背景
        heatmaps = np.vstack([heatmaps, heatmap_bg])
        return heatmaps.astype(np.float32)

    def make_PAF(self, shape, joint_from, joint_to):
        """generate part_affinity_field following the paper

        Args:
            shape: (image_h, image_w) tuple or list
            joint_from: the first joint in limbs pairs
            joint_to: the second joint in limbs pairs

        Returns:
            PAF for one joint type
        """
        joint_dist = np.linalg.norm(joint_to - joint_from)
        unit_vector = (joint_to - joint_from) / joint_dist
        # 以下部分 计算unit_vector的正交单位向量
        radius = np.pi / 2
        rot_matrix = np.array([
            [np.cos(radius), np.sin(radius)],
            [-np.sin(radius), np.cos(radius)],
        ])
        vertical_unit_vector = np.dot(rot_matrix, unit_vector)
        # 现在开始计算PAF
        grid_x = np.tile(np.arange(shape[1]), (shape[0], 1))
        grid_y = np.tile(np.arange(shape[0]), (shape[1], 1)).transpose()
        # 论文中有两个约束，一个是基于节点距离的，一个是基于连结宽度(limb width)的
        # 先计算节点距离的约束
        joint_distances = (unit_vector[0] * (grid_x - joint_from[0]) + 
                           unit_vector[1] * (grid_y - joint_from[1]))
        joint_distances_flag = (0 <= joint_distances) & (joint_distances <= joint_dist)
        limb_widths = (vertical_unit_vector[0] * (grid_x - joint_from[0]) + 
                       vertical_unit_vector[1] * (grid_y - joint_from[1]))
        limb_widths_flag = np.abs(limb_widths) <= self.params.paf_sigma
        # 将同时满足两个约束条件的点集合起来
        paf_flag = joint_distances_flag & limb_widths_flag
        # 因为有x, y两个坐标，所以需要重复两次
        paf_flag = np.stack([paf_flag, paf_flag])
        # 创建一个所有点都是单位向量的平面
        paf = np.broadcast_to(unit_vector, (*shape, 2)).transpose(2, 0, 1)
        return paf * paf_flag # 仅返回那些需要的点

    def make_PAFs(self, shape, poses):
        """generate PAFs for input image

        Args:
            shape: (im_h, im_w) tuple or list
            poses: poses for im, return by convert_joint_order
        Returns:
            PAFs shape as (2 * len(limbs), im_h, im_w)
        """
        pafs = np.zeros([0, *shape])

        # 这里的思路同heatmap，先遍历同一种关节的联结，再遍历每个pose
        for limb in self.params.limbs:
            paf = np.zeros([2, *shape])
            # NOTE: 记录每个点重合的次数，目的是让最终的paf的点保持一样的水平
            # TODO: 以一种更直观的方式实现PAF
            paf_overlay = np.zeros_like(paf, np.int32)

            for pose in poses:
                joint_from, joint_to = pose[limb]
                # 如果这两个关键点都有标出来
                if joint_from[2] > 0 and joint_to[2] > 0:
                    limb_paf = self.make_PAF(shape, joint_from[:2], joint_to[:2])
                    paf += limb_paf
                    
                    limb_paf_flags = limb_paf != 0
                    # limb_paf_flags[0]代表from节点，[1]代表to节点，将两个节点合在一起
                    paf_overlay += np.broadcast_to(limb_paf_flags[0] | limb_paf_flags[1], limb_paf.shape)
            # 将倍数除去，对应论文公式9
            paf[paf_overlay > 0] /= paf_overlay[paf_overlay > 0]
            pafs = np.vstack([pafs, paf])
        return pafs.astype(float)
    
    def make_ignore_mask(self, shape, ann_metas):
        """Generate mask to ignore some unsuitable data to improve training

        Args:
            shape: shape of mask, same as image
            ann_metas: COCO annotation
        Returns:
            mask_all: mask contains all the sample, only used to visualization
            mask_ignore: mask contains the sample that should be ignored, used 
            to train the network
        """
        mask_all = np.zeros(shape, 'bool')
        mask_ignore = np.zeros(shape, 'bool')

        for ann in ann_metas:
            mask = self.coco.annToMask(ann).astype('bool')
            # if iscrowd, in order to avoid accidently mask out other people's
            # mask, compute the interaction area firstly then only mask out 
            # remain area.
            # 如果是密集人群，先求出这个人与所有人的mask的交集，然后减掉不是交集的部分
            # 这样可以避免将其他在参加训练的标签给屏蔽了。
            # mask^overlay会将mask与mask_all之间的非重叠部分算出来
            if ann['iscrowd'] == 1:
                overlay = mask_all & mask
                mask_ignore = mask_ignore | (mask ^ overlay)
                mask_all = mask_all | mask
            # if number of keypoints or mask area is too small, mask it out
            # 过滤掉那些关键点和面积太小的样本
            elif (ann['num_keypoints'] < self.params.min_keypoints 
                        or ann['area'] <= self.params.min_area):
                mask_all = mask_all | mask
                mask_ignore = mask_ignore | mask 
            # 样本符合要求，则不生成对应的mask，只是做可视化
            else:
                mask_all = mask_all | mask
        return mask_all, mask_ignore

    def save_ignore_mask(self, im_id, mask):
        # 只保留那些有mask区域的mask以节约空间
        if np.any(mask):
            mask = mask.astype(np.uint8) * 255
            # save beside to images directory, same name of image
            # but use PNG for its lossless compression
            # 这里将图片保存在原来images目录的同一级上，但使用无损的png格式保存
            file_path = self.get_im_path(im_id=im_id).replace('.jpg', '.png')
            mask_path = osp.join(self.mask_dir, osp.basename(file_path))
            cv2.imwrite(mask_path, mask)
            
    def get_ignore_mask(self, im_id):
        """return ignore mask according to image id"""
        path = self.get_im_path(im_id=im_id)
        name = osp.basename(path).replace('.jpg', '.png')
        mask_path = osp.join(self.mask_dir, name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            return mask == 255
        # 如果没有mask，则根据原图片尺寸创建一个
        im = cv2.imread(path)
        return np.zeros(im.shape[:2], 'bool')

    def generate_ignore_masks(self, visual=True):
        """generate ignore masks whether for visualization or save into
        disk for training.

        Args:
            visual: bool, if True, just visualize image, or else save 
                ignore mask into disk.
        """
        for i in range(self.size):
            im_id, ann_metas, im = self.get_im_with_anns(index=i)
            _, mask = self.make_ignore_mask(im.shape[:2], ann_metas)

            # visual
            if visual:
                im_origin = self.plot_masks_and_keypoints(im, ann_metas)
                im_train = self.plot_ignore_mask(im, mask)
                cv2.imshow("image", np.hstack([im_origin, im_train]))
                k = cv2.waitKey()
                if k == ord('q'):
                    break
            else:
                self.save_ignore_mask(im_id, mask)

    # -------- Data Augmentation --------
    def resize_data(self, im, mask, poses, shape):
        """resize all data as shape

        Args:
            im: image object return by cv2.imread
            mask: ignore mask for image
            poses: poses for im, return by convert_joint_order
            shape: target shape, two value tuple.
        Returns:
            resized im, mask, poses
        """
        im_h, im_w, _ = im.shape
        im = cv2.resize(im, shape)
        mask = cv2.resize(mask.astype(np.uint8), shape).astype('bool')
        # 对于pose，针对它的x跟y，按宽度比例进行缩放
        poses[:, :, :2] = poses[:, :, :2] * np.array(shape) / np.array([im_w, im_h])
        return im, mask, poses

    def augment_data(self, im, mask, poses):
        """Augment strategy for Pose estimation"""
        # TODO: implement it
        return im, mask, poses

    # -------- Data Generator --------
    def generate_labels(self, im, mask, poses):
        """generate labels for training purpose

        Args:
            im: image object return by cv2.imread
            mask: generated ignore_mask
            poses: poses for im, return by convert_joint_order
        Returns: 
            im, mask, heatmaps, pafs
        """
        im, mask, poses = self.augment_data(im, mask, poses)
        im, mask, poses = self.resize_data(im, mask, poses, shape=(self.params.insize, ) * 2)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_DILATE, np.ones([16, 16]))
        mask = mask.astype('bool')
        heatmaps = self.make_confidence_maps(im.shape[:2], poses) 
        pafs = self.make_PAFs(im.shape[:2], poses)
        return im, mask, heatmaps, pafs

    # -------- Visualization --------
    def plot_masks_and_keypoints(self, im, ann_metas):
        """visualize mask region and keypoints in COCO dataset
        
        Args:
            im: numpy.array return by cv2.imread
            ann_metas: annotations list of im
        Returns:
            im: image with highlighted mask region and keypoint.
        """
        for ann in ann_metas:
            # plot the mask
            mask = self.coco.annToMask(ann).astype(np.uint8)
            # paint crowd people red
            # paint no keypoint people green
            # paint training people blue
            if ann['iscrowd'] == 1:
                color = (0, 0, 1)
            elif ann['num_keypoints'] == 0:
                color = (0, 1, 0)
            else:
                color = (1, 0, 0)
            # mask the images, same as plot_ignore_mask
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            mask = mask.astype(np.int32)
            im_masked = im * mask
            for i in range(3):
                mask[:, :, i] = mask[:, :, i] * color[i] * 255
            im = im - 0.7 * im_masked + 0.7 * mask
            
            # plot the keypoints
            for x, y, v in np.array(ann['keypoints']).reshape(-1, 3):
                if v == 0: continue
                # invisible is Blue+Green, visible is Blue+Red
                color = (255, 255, 0) if v == 1 else (255, 0, 255)
                cv2.circle(im, (x, y), radius=3, color=color, thickness=-1) 

        return im.astype(np.uint8)

    def plot_labels(self):
        """visualize labels for checking purpose"""
        for i in range(self.size):
            im_id, ann_metas, im = self.get_im_with_anns(index=i)
            mask = self.get_ignore_mask(im_id)
            poses = self.convert_joint_order(ann_metas)
            im, mask, heatmaps, pafs = self.generate_labels(im, mask, poses)
            # resize to view
            shape = (self.params.insize, self.params.insize)
            pafs = cv2.resize(pafs.transpose(1, 2, 0), shape).transpose(2, 0, 1)
            heatmaps = cv2.resize(heatmaps.transpose(1, 2, 0), shape).transpose(2, 0, 1)
            mask = cv2.resize(mask.astype(np.uint8) * 255, shape) == 255
            # overlay labels
            # 以下三个分别是PAF，heatmap和mask的效果
            # 你可以注释掉其中的语句来看效果
            img_to_show = im.copy()
            img_to_show = self.overlay_PAFs(img_to_show, pafs)
            img_to_show = self.overlay_heatmap(img_to_show, heatmaps[:-1].max(axis=0))
            img_to_show = self.overlay_ignore_mask(img_to_show, mask)
            # cv2.imwrite("resized_img.png", resized_img)
            # cv2.imwrite('img_to_show.png', img_to_show)
            cv2.imshow('w', np.hstack((im, img_to_show)))
            k = cv2.waitKey(0)
            if k == ord('q'):
                break
     

class KeyPointTest(KeyPointMixin):

    def __init__(self, params: KeyPointParams):
        self.params = params
    
    def preprocess(self, im):
        """Combine preprocess methods for convenience

        Args:
            im: image object return by cv2.imread
        Params:
            infer_size: image size on inference time
            stride: decided by network, total downscale of input and output
        Returns:
            image preprocessed
        """
        im = self.im_letterbox(im, self.params.infer_insize, self.params.stride)
        im = self.im_preprocess(im)
        return im

    def postprocess(self, im, pafs, heatmaps):
        """Combine post preprocess methods for convenience

        Args:
            im: image object return by cv2.imread
            pafs: original PAF of network
            heatmaps: original CFM of network
        Returns:
            same as `pose_decode`
        """
        tsize = self.params.heatmap_size
        heatmaps = self.im_letterbox(heatmaps, tsize, self.params.stride)
        pafs = self.im_letterbox(pafs, tsize, self.params.stride) 
        parts_list, persons = self.pose_decode(im, heatmaps, pafs)
        return parts_list, persons


    def find_peaks(self, heatmap):
        """find peaks from heatmap. Here, we decide a peak by 2 conditions:
        1. the peak must surpass the threshold
        2. the peak must surpass its four direction points around
           ----    Top     -----
           Left    Peak    Right  s.t. Peak is maximum
           ----    Bottom  -----
           this struct is referred as `footprint` in this function 

        Args:
            heatmap: heatmap of joint, usually is one channel of network output
        Returns:
            Coordinates of peaks in image-form, with shape (N, 2). Note that for 
            a point(x, y) in an image, you should fetch it by (y, x) in numpy. 
            Here we use the image-form instead of numpy.array format.
        """
        footprint = generate_binary_structure(2, 1)
        peaks = maximum_filter(heatmap, footprint=footprint)
        # now we compute the coordinates
        binary = (peaks == heatmap) * (heatmap > self.params.peak_threshold)
        # we reverse the return value to get the coordinates in image-form
        coords = np.nonzero(binary)[::-1]
        coords = np.array(coords).T
        return coords

    def coordinates_resize(self, coords, factor):
        """helper function used for finetuning peaks. resize coordinates
        according to given factor. Given cell [1,2] return [2.5, 4.5]
        Get from:
        https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation

        Args:
            coords: coordinates of peaks in image-form, with shape (N, 2)
            factor: indicate the scale to perform
        Returns:
            New coordinates with the same shape. Note that new coordinates 
            is float type.
        """
        coords = coords.astype('f')
        return (coords + 0.5) * factor - 0.5

    def peak_refine(self, heatmap, peak, factor):
        """helper function to refine coordinates of peak
        
        Args:
            heatmap: heatmap of joint, usually is one channel of network output
            peak: peak coordinate (x, y) in image-form
            factor: size ratio between CFM heatmap and insize
        Params: 
            winsize: control patch size. E.g. winsize=1, patch => 3x3
                winsize=2, patch => 5x5
            smooth: if True, apply Gaussian Filter
            gaussian_sigma: gaussian filter parameter
        Returns:
            A tuple( Refined version of peak, score of peak )
        """
        winsize = self.params.winsize
        smooth = self.params.smooth
        gaussian_sigma = self.params.gaussian_sigma

        # heatmap.T.shape => (width, height)
        shape = np.array(heatmap.T.shape)
        # Get patch border, still in image format
        x_min, y_min = np.maximum(0, peak - winsize)
        x_max, y_max = np.minimum(shape - 1, peak + winsize)
        # take a small patch around peak and upsample it
        # 将peak周围的区域进行上采样
        patch = heatmap[y_min : y_max+1, x_min : x_max+1]
        patch = cv2.resize(patch, None, None, factor, factor, cv2.INTER_CUBIC)
        
        # apply gaussian filter
        if smooth:
            patch = gaussian_filter(patch, sigma=gaussian_sigma)

        # 从这里我们开始修正peak的坐标。peak现在不一定是patch中的最大值（当然，一般是）。
        # 所以我们的做法是，先计算patch中最大值的坐标，然后计算这个坐标跟peak的偏置。如果
        # 两者相同，那么偏置为0。如果不同，这用这个偏置去修正peak的坐标。
        # 
        # 1. 计算当前patch的最大值的坐标，注意这是numpy format
        patch_max = np.unravel_index(patch.argmax(), patch.shape)
        # 2. 计算peak在patch上的坐标，也即是patch的中心坐标
        patch_center = self.coordinates_resize(peak[::-1] - [y_min, x_min], factor)
        # 3. 计算最大值坐标与中心坐标的偏置，用来对peak的坐标进行修正
        offset = patch_max - patch_center
        peak_refined = self.coordinates_resize(peak, factor) + offset[::-1]
        peak_score = patch[patch_max]
        return peak_refined, peak_score

    def NMS(self, heatmaps, factor):
        """Follow the paper section 2.2,  NMS obtains body part candidates

        Args:
            heatmaps: (h, w, len(joints)) numpy.array
            factor: size ratio between CFM heatmap and insize
        Returns:
            List, each element represents one part, each part contains several
            peaks. Each peak contains (x, y, score, global id). Note that the
            global id is unique for every joint regardless its type. we refer
            return value as `parts_list` in `part_associate` and `person_parse`.
        Note: 
            The global id for each joint is used to decide which person owns
            this joint. See `person_parse` for more details.
        """
        parts_list = []
        global_id = 0
        
        for joint_i in self.params.joint:
            heatmap = heatmaps[..., joint_i]
            peaks = self.find_peaks(heatmap)
            # each part contains (x, y, score, global_id)
            parts = np.zeros([len(peaks), 4])
            
            for peak_i, peak in enumerate(peaks):
                # 这里，如果对peak进行修正，那么修正时会返回分数。如果不修正，
                # 则直接获取分类。但peak的位置仍需要进行放缩
                if self.params.peak_refine:
                    peak, score = self.peak_refine(heatmap, peak, factor)
                else:
                    score = heatmap[peak[::-1]]
                    peak = self.coordinates_resize(peak, factor)
                # 现在peak已经变成浮点型，需要转成整型
                peak = [int(x) for x in peak]
                # 将新的peak坐标，分数，以及这个peak的id放入对应的part
                part = [*peak, score, global_id]
                parts[peak_i, :] = part
                # 更新全局id
                global_id += 1
            
            parts_list.append(parts)
            
        return parts_list

    def limb_score(self, pafs, fparts, tparts, channels):
        """helper function for part association

        Args:
            pafs: PAFs has already been resized to original input image shape
            fparts: from parts, element of parts_list 
            tparts: to parts, element of parts_list
            channels: index array to fetch PAFs from `pafs`
        Params:
            nb_sample: number of samples. In order to evaluate connections, we
                sample some points lied on the limb (connection).
            nb_sample_threshold: number of samples that a limb should surpass
            limb_threshold: keep limb which surpass this value
        Returns: 
            All possible limbs.
            Each limb contains (fGid, tGid, score of limb, fpart_i, tpart_i)
        """
        possible_limbs = []
        nb_sample = self.params.nb_sample
        nb_sample_threshold = self.params.nb_sample_threshold
        limb_threshold = self.params.limb_threshold

        for fpart_i, fpart in enumerate(fparts):
            for tpart_i, tpart in enumerate(tparts):
                # part affinity unit vector
                vector = tpart[:2] - fpart[:2]
                norm = np.linalg.norm(vector)
                # two points may overlay
                if norm == 0:
                    continue
                vector = vector / norm # shape: (2, )

                # approximate integral by sampling and summing uniformly-spaced
                # values of u. Note that xs, ys are # in image form. 
                # So we access sample in order [ys, xs]
                xs = np.round(np.linspace(fpart[0], tpart[0], nb_sample)).astype(int)
                ys = np.round(np.linspace(fpart[1], tpart[1], nb_sample)).astype(int)
                samples = pafs[ys, xs, channels] # shape: (2, nb_sample)
                # compute the limb score
                score = samples.T.dot(vector) # shape: (nb_sample, )
                # penalty for long limb (long than half of height of PAF)
                penalty = min(0, 0.5 * pafs.shape[0] / norm - 1)
                score_penalty = score.mean() + penalty
                # now we evaluate the score of limb
                # 1. 80% of points surpass threshold 
                # 2. score_penalty must be positive
                nb_surpass = np.count_nonzero(score > limb_threshold)
                criterion1 = nb_surpass > nb_sample_threshold
                criterion2 = score_penalty > 0
                if criterion1 and criterion2:
                    fGid, tGid = fparts[fpart_i, 3], tparts[tpart_i, 3]
                    limb = [fGid, tGid, score_penalty, fpart_i, tpart_i]
                    possible_limbs.append(limb)
    
        return possible_limbs

    def limb_select(self, fparts, tparts, possible_limbs):
        """Perform greedy algorithms to find good limbs
        
        Args:
            fparts: from parts, element of parts_list 
            tparts: to parts, element of parts_list
            possible_limbs: refer to `limb_score`
        Returns:
            a subset of possible_limbs, represent good limbs
        """
        # now, we have all possible limbs for each limb type, we need to
        # evaluate them. This part matches formula 13, 14 in paper
        # 1. sort limbs according to its score
        possible_limbs.sort(key=lambda x: x[2], reverse=True)
        # 2. use greedy algorithms to decide which limb is good, note that 
        # maximum number of limb is the minimum number of two parts
        max_limb = min(len(fparts), len(tparts))
        # recall that each limb in possible_limbs contains
        # (fGid, tGid, score, fpart_i, tpart_i) 
        limbs = []
        fpart_used = []
        tpart_used = []
        for limb in possible_limbs:
            fpart_i, tpart_i = limb[3], limb[4]
            if fpart_i in fpart_used or tpart_i in tpart_used:
                # if either each part is used, then the limb is invalid
                continue
            else:
                limbs.append(limb)
                fpart_used.append(fpart_i)
                tpart_used.append(tpart_i)
                # if maximum is reached, stop parsing
                if len(limbs) == max_limb:
                    break
        return limbs

    def part_associate(self, pafs, parts_list):
        """Follow the paper section 2.3, this function leverages PAFs to find 
        connections between peaks.
    
        Args:
            pafs: PAFs has already been resized to original input image shape
            parts_list: return value of `NMS`
        Returns:
            Nested list, length equals to limbs type. For each limb type, may 
            have none or more limbs. We refer this return value as `limbs_list` 
            in `person_parse`.
            Each limb contains (fGid, tGid, score of limb, fpart_i, tpart_i)
        """
        limbs_list = []

        for limb_i, (fparts_i, tparts_i) in enumerate(self.params.limbs):
            fparts = parts_list[fparts_i]
            tparts = parts_list[tparts_i]

            if len(fparts) == 0 or len(tparts) == 0:
                # no limb between these two parts is found
                limbs = []
            else:
                # compute the index of PAFs of fpart and tpart
                fpaf_i = 2 * limb_i
                tpaf_i = 2 * limb_i + 1
                # this strange syntax is for broadcast in numpy
                channels = [[fpaf_i], [tpaf_i]]
                # find possible limbs
                possible_limbs = self.limb_score(pafs, fparts, tparts, channels)
                # select good limbs
                limbs = self.limb_select(fparts, tparts, possible_limbs)

            limbs_list.append(limbs)

        return limbs_list

    def person_parse(self, parts_list, limbs_list):
        """Follow the paper section 2.4, this function parse multi-person
        using PAFs
        
        Args:
            parts_list: return value of `NMS`
            limbs_list: return value of `part_associate`
        Returns:
            persons: A dict with Key: person id, Value: a list. let's say 
            we have 10 joints, then the first 10 elements of list is either
            -1 or Gid for the joint. The 2nd-to-last element is score. 
            The last one is the number of joints this person owns.
        """
        # 我们的目标是输出每个人的关键点(part)，而每个关键点都有一个全局唯一的ID，所以
        # 我们只需要找出每个人的关节点的ID就可以了。如果没有这个节点，就用-1来表示
        # 对于每个人，我们还额外记录两个值，一个是这个人拥有的节点数，一个是他的分数
        # 所以对于字典的每个元素，长度为 len(self.params.joint) + 2
        persons = {}
        person_size = len(self.params.joint) + 2
        # 我们建立一个字典来索引part跟person的关系，如果这个part已经被关联了，那么
        # 字典里将保存这个人的ID. Key: part Gid, Value: person ID
        mapping = {}

        for limb_i, (fpart, tpart) in enumerate(self.params.limbs):
            for limb in limbs_list[limb_i]:
                fGid, tGid, limb_score, fpart_i, tpart_i = limb
                # 看看两个关节已经关联到的人的ID
                fPid = mapping.get(fGid)
                tPid = mapping.get(tGid)

                # 对于每个联结(limb)，有好几种情况：
                # 1. 两个节点(part)都没有关联到人，则创建新人来关联 
                # 2. 两个节点，其中一个已关联，另一个没有，则将没有关联的那个节点
                #    加入到已关联的那个人里去
                # 3. 两个节点都已关联
                # 3.1 但关联的是同一个人，则只需要将limb的分数加给那个人就好
                # 3.2 两个节点关联到不同的人，由于一个节点只能关联到一个人，所以
                #    这两个人必定没有交叉的节点，把这两个人合成一个人
                if fPid is None and tPid is None:
                    person = [-1] * person_size
                    person[fpart] = fGid
                    person[tpart] = tGid
                    # -1 位置放节点数
                    person[-1] = 2
                    # -2 位置放分数，包括两个peak_score和他们的limb_score
                    person[-2] = sum([
                        parts_list[fpart][fpart_i, 2],
                        parts_list[tpart][tpart_i, 2],
                        limb_score,
                    ])
                    # 计算当前(current)这个人的id。如果还没有人，就以0作为id。如果有人
                    # 将取其中最大的id值+1作为新人的id
                    cPid = 0 if len(persons) == 0 else max(persons.keys()) + 1
                    # 将节点与这个人的id进行关联
                    mapping[fGid] = cPid
                    mapping[tGid] = cPid
                    # 增加新人
                    persons.update({cPid: person})
                    
                elif fPid is None:
                    person = persons[tPid]
                    person[fpart] = fGid
                    # 增加了fpart，所以节点数和分数相应增加
                    person[-1] += 1
                    person[-2] += sum([
                        parts_list[fpart][fpart_i, 2],
                        limb_score,
                    ])
                    # 将fGid与这个人关联起来
                    mapping[fGid] = tPid
                elif tPid is None:
                    person = persons[fPid]
                    person[tpart] = tGid
                    person[-1] += 1
                    person[-2] += sum([
                        parts_list[tpart][tpart_i, 2],
                        limb_score,
                    ])
                    mapping[tGid] = fPid
                elif fPid == tPid:
                    person = persons[fPid]
                    person[-2] += limb_score
                else:
                    # 分配到两个不同的人，将他们合并
                    fperson = persons[fPid]
                    tperson = persons[tPid]
                    # 这里我们想把tperson的所有关节点都保存在fperson
                    # 同时把tperson移除，需修改persons及mapping
                    for part_i, part in enumerate(tperson[:-2]):
                        if part != -1:
                            fperson[part_i] = part
                            mapping[part] = fPid
                    # 合并分数
                    fperson[-1] += tperson[-1]
                    fperson[-2] += sum([
                        tperson[-2],
                        limb_score,
                    ])
                    # 删除tperson这个人
                    persons.pop(tPid)

        # 现在把那些节点比较少或者分数比较低的人给删除
        pid_del = []
        for pid, person in persons.items():
            nb_part = person[-1]
            mean_score = person[-2] / nb_part
            critierion1 = nb_part < self.params.nb_part_threshold
            critierion2 = mean_score < self.params.mean_score_threshold
            if critierion1 or critierion2:
                pid_del.append(pid)
        for pid in  pid_del:
            persons.pop(pid)
        
        return persons

    def pose_decode(self, im, heatmaps, pafs):
        """Given an image, return poses information
        
        Args:
            im: input image
            heatmaps: network output
            pafs: network output
        Returns:
            A dict contains person id and pose information
        """
        # 放缩是一个很微妙的问题，是为记
        # 对于part来说，因为还在进行其他处理，如微调以及平滑，如果在一个小的map上做，最后
        # 再放大为原来的样子，是比较高效的。
        # 对于paf来说，因为不需要做其他处理，只要让它跟原图一样就好了
        parts_list = self.NMS(heatmaps, factor=im.shape[0] / heatmaps.shape[0])
        pafs = cv2.resize(pafs, im.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
        limbs_list = self.part_associate(pafs, parts_list)
        persons = self.person_parse(parts_list, limbs_list)
        return parts_list, persons

    def plot_pose(self, im, parts_list, persons):
        """draw pose on input image, including joint and limbs
        
        Args:
            im: input image
            parts_list: refer to `NMS`
            persons: refer to `person_parse`
        Params:
            colors: colors for joint type
        Returns: 
            drawed image
        """
        canvas = im * 0.3
        canvas = canvas.astype(np.uint8)

        # 为了方便画出关键点，我们将parts_list合成一个array，那么一个part的Gid
        # 就刚好与它的行数相对应
        parts = np.vstack(parts_list)
        
        joint = self.params.joint
        for limb_i, (fpart, tpart) in enumerate(self.params.limbs):
            if fpart == joint.LShoulder and tpart == joint.LEar:
                continue
            elif fpart == joint.RShoulder and tpart == joint.REar:
                continue
            for person in persons.values():
                fGid, tGid = int(person[fpart]), int(person[tpart])
                if fGid == -1 or tGid == -1:
                    # 并没有这个limb
                    continue
                # 如果有，则取出这两个part的坐标
                fpeak = tuple(parts[fGid, :2].astype(int))
                tpeak = tuple(parts[tGid, :2].astype(int))
                cv2.circle(canvas, fpeak, 3, (255, 255, 255), thickness=-1)
                cv2.circle(canvas, tpeak, 3, (255, 255, 255), thickness=-1)
                # 画limb，从fpeak到tpeak，从params中选颜色
                cv2.line(canvas, fpeak, tpeak, self.params.colors[limb_i])
        return canvas
    
    def openpose(self, im, parts_list, persons):
        """A helpful function to output same format as OpenPose project

        Args:
            im: input image
            parts_list: refer to `NMS`
            persons: refer to `person_parse`
        Params:
            colors: colors for joint type
            joint: Joint type 
        Returns: 
            Dictionary as OpenPose Json format
        """
        output = {"version": 1}
        people = []
        # 为了方便画出关键点，我们将parts_list合成一个array，那么一个part的Gid
        # 就刚好与它的行数相对应
        parts = np.vstack(parts_list)
        for person in persons.values():
            pose_keypoints = []
            for Gid in person[:18]:
                # x, y, score
                x, y, score = parts[int(Gid), :3]
                pose_keypoints += [int(x), int(y), round(score, 5)]
            people.append(pose_keypoints)
        output['people'] = people
        return output

__all__ = [
    'PoseEstimation',
    'KeyPointParams',
    'KeyPointTrain',
    'KeyPointTest',
]