import os
import os.path as osp

import cv2
import numpy as np
import pycocotools.coco as coco

from .entity import KeyPointParams # for KeyPoint

# NOTE: for debug, should be removed
import pysnooper

class COCO(object):

    def __init__(self, images_directory, annotation_file):
        """initialize datasets

        Args:
            images_directory: string, usually as 'COCO/images/'
            annotations_file: string, usually a json file
        """
        self.coco = coco.COCO(annotation_file)
        self.root = osp.dirname(images_directory.rstrip(osp.sep))
        self.im_dir = images_directory
        self.im_ids = sorted(self.coco.imgs.keys())
        self.size = len(self.im_ids)
    
    def __len__(self):
        return self.size

    def get_im_path(self, index=None, im_id=None):
        if index is None and im_id is None:
            raise Exception("Either index or im_id should be provided")
        if index is not None:
            im_id = self.im_ids[index]
        im_meta = self.coco.loadImgs([im_id])[0]
        return osp.join(self.im_dir, im_meta['file_name'])

    def get_im_with_anns(self, index=None, im_id=None):
        """
        Args:
            index: index of image ids list 
            im_id: image id in COCO dataset
        Returns:
            a triple of (im_id, ann_metas, im) where im is image 
            object return by cv2.imread
        """
        if index is None and im_id is None:
            raise Exception("Either index or im_id should be provided")
        if index is not None:
            im_id = self.im_ids[index]
        ann_ids = self.coco.getAnnIds(im_id)
        ann_metas = self.coco.loadAnns(ann_ids)
        
        im_path = self.get_im_path(im_id=im_id)
        im = cv2.imread(im_path)
        return im_id, ann_metas, im


class KeyPoint(COCO):
    """A class for pose estimation, especially for this paper
    link: https://arxiv.org/abs/1611.08050.

    """
    def __init__(self, images_directory, annotation_file, params: KeyPointParams):
        """
        Args:
            params: KeyPointParams instance 
        """
        super().__init__(images_directory, annotation_file)
        self.params = params
        self.mask_dir = osp.join(self.root, 'mask')
        # create mask directory
        os.makedirs(self.mask_dir, exist_ok=True)
    
    def convert_joint_order(self, ann_metas):
        """convert the joint order from COCO to ours

        Args:
            ann_metas: annotations of COCO
        
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

    def make_confidence_map(self, shape, joint):
        """create a confidence map (heatmap, jointmap).

        Args:
            shape: shape of confidence map 
            joint: a.k.a keypoint, (x, y)
            sigma: the hyperparameter control the spread of peak
        Returns:
            Confidence map with shape
        """
        x, y = joint
        # grid_x是所有点的x坐标，grid_y是所有点的y坐标，这里的x、y是从图片的
        # 像素坐标的角度来理解
        grid_x = np.tile(np.arange(shape[1]), (shape[0], 1))
        grid_y = np.tile(np.arange(shape[0]), (shape[1], 1)).transpose()
        grid_dist = (grid_x - x) ** 2 + (grid_y - y) ** 2
        # NOTE: 这里并没有对heatmap的最大值进行约束，而原作caffe中有
        heatmap = np.exp(-0.5 * grid_dist / self.params.heatmap_sigma ** 2)
        return heatmap

    def make_confidence_maps(self, im, poses):
        """generate confidence map for single image
        
        Args:
            im: image object return by cv2.imread
            poses: poses for im, return by convert_joint_order
        Returns:
            comfidence map with shape (len(joint), h, w)
        """
        # init heatmaps as (0, h, w) 
        im_h, im_w = im.shape[:2]
        heatmaps = np.zeros([0, im_h, im_w])
        # 为了计算背景，需要把所有heatmap进行累加
        heatmap_sum = np.zeros([im_h, im_w])
        # 这里的思路是，将所有相同关键点放到一个heatmap中
        # 所以是先遍历关键点，再遍历每个人的pose
        for joint_i in range(len(self.params.joint)):
            heatmap = np.zeros([im_h, im_w])
            for pose in poses:
                # 查看每一个关节点的v值，大于0说明该点有标签
                if pose[joint_i, 2] > 0:
                    jointmap = self.make_confidence_map([im_h, im_w], pose[joint_i][:2])
                    # 这里是论文中的对不同人的关节点的max运算，以保留峰值
                    heatmap[jointmap > heatmap] = jointmap[jointmap > heatmap]
                    heatmap_sum[jointmap > heatmap_sum] = jointmap[jointmap > heatmap_sum]
            # 将这个关键点的热力图添加进返回值中
            heatmaps = np.vstack([heatmaps, heatmap.reshape([1, im_h, im_w])])
        heatmap_bg = (1 - heatmap_sum).reshape([1, im_h, im_w]) # 背景
        heatmaps = np.vstack([heatmaps, heatmap_bg])
        return heatmaps.astype(np.float32)


    def make_PAF(self, shape, joint_from, joint_to):
        """generate part_affinity_field following the paper

        Args:
            shape: (image_h, image_w) tuple or list
            joint_from: the first joint in limbs pairs
            joint_to: the second joint in limbs pairs
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
        paf = np.broadcast_to(unit_vector, shape + [2]).transpose(2, 0, 1)
        return paf * paf_flag # 仅返回那些需要的点

    def make_PAFs(self, im, poses):
        """generate PAFs for input image

        Args:
            im: image object return by cv2.imread
            poses: poses for im, return by convert_joint_order
        Returns:
            PAFs shape as (2 * len(limbs), im_h, im_w)
        """
        im_h, im_w = im.shape[:2]
        pafs = np.zeros([0, im_h, im_w])

        # 这里的思路同heatmap，先遍历同一种关节的联结，再遍历每个pose
        for limb in self.params.limbs:
            paf = np.zeros([2, im_h, im_w])
            # NOTE: 记录每个点重合的次数，目的是让最终的paf的点保持一样的水平
            # TODO: 以一种更直观的方式实现PAF
            paf_overlay = np.zeros_like(paf, np.int32)

            for pose in poses:
                joint_from, joint_to = pose[limb]
                # 如果这两个关键点都有标出来
                if joint_from[2] > 0 and joint_to[2] > 0:
                    limb_paf = self.make_PAF([im_h, im_w], joint_from[:2], joint_to[:2])
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
        heatmaps = self.make_confidence_maps(im, poses) 
        pafs = self.make_PAFs(im, poses)
        return im, mask, heatmaps, pafs


    # 这里开始是可视化相关的函数
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
        im = KeyPoint.overlay_PAF(im, mix_paf)
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