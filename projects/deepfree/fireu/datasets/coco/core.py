import os
import os.path as osp

import cv2
import numpy as np
import pycocotools.coco as coco


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


from entity import KeyPointParams
class KeyPoint(COCO):
    """A class for pose estimation, especially for this paper
    link: https://arxiv.org/abs/1611.08050.

    """
    def __init__(self, images_directory, annotation_file, params):
        """
        Args:
            params: KeyPointParams instance 
        """
        super().__init__(images_directory, annotation_file)
        self.params = params
        self.mask_dir = osp.join(self.root, 'mask')
        # create mask directory
        os.makedirs(self.mask_dir, exist_ok=True)

    def make_confidence_map(self, shape, joint, sigma):
        """create a confidence map (heatmap).

        Args:
            shape: shape of confidence map 
            joint: a.k.a keypoint, (x, y)
            sigma: the hyperparameter control the spread of peak
        Returns:
            Confidence map with shape
        """
        x, y = joint
        grid_x = np.tile(np.arange(shape[1]), (shape[0], 1))
        grid_y = np.tile(np.arange(shape[0]), (shape[1], 1)).transpose()
        grid_distance = (grid_x - x) ** 2 + (grid_y - y) ** 2
        gaussian_heatmap = np.exp(-0.5 * grid_distance / sigma**2)
        return gaussian_heatmap
    
    def make_confidence_maps(self):
        pass

    def make_part_affinity_field(self):
        pass

    def make_part_affinity_fields(self):
        pass
    
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
                        or ann['area'] < self.params.min_area):
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
            file_path = self.get_im_path(im_id=im_id)
            save_path = osp.join(self.mask_dir, osp.basename(file_path))
            cv2.imwrite(save_path, mask)
            print("Save mask for image: {}".format(im_id))
        else:
            print("Empty mask for image: {}".format(im_id))
        # Empty Mask. When training, create one by run:
        # mask = np.zeros(im.shape[:2], 'bool')
        # for those have a mask, turn it to bool mask
        # mask = mask == 255
            
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
        im = im - 0.7 * im_masked + 0.7 * mask
        return im
            
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

if __name__ == "__main__":
    images_directory = '/data/minicoco/images'
    annotation_file = "/data/minicoco/annotations/mini_person_keypoints_val2014.json"
    dataset = KeyPoint(images_directory, annotation_file, KeyPointParams)

    for i in range(len(dataset)):
        im_id, ann_metas, im = dataset.get_im_with_anns(index=i)
        _, mask = dataset.make_ignore_mask(im.shape[:2], ann_metas)

        # visual
        im_origin = dataset.plot_masks_and_keypoints(im, ann_metas)
        im_train = dataset.plot_ignore_mask(im, mask)
        # cv2.imshow("image", np.hstack([im_origin, im_train]))
        # k = cv2.waitKey()
        # if k == ord('q'):
        #     break

        dataset.save_ignore_mask(im_id, mask)