"""
Mask R-CNN
Configurations and data loading code for the synthetic Shapes dataset.
This is a duplicate of the code in the noteobook train_shapes.ipynb for easy
import into other notebooks, such as inspect_model.ipynb.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""
import os
import glob
import random
import numpy as np

from PIL import Image

from config import Config
import utils


class CaptchasConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "captchas"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 62  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


class CaptchasDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        self.class_info = []
        self.source_class_ids = {}
        self.charset = ("abcdefghijklmnopqrstuvwxyz" +
                        "ABCDEFGHIJKLMNOPQRSTUVWXYZ" +
                        "0123456789")

    def load_image(self, image_id):
        """返回对应的图片数据"""
        info = self.image_info[image_id]
        return info['image']

    def image_reference(self, image_id):
        """返回这张图片对应的验证码"""
        info = self.image_info[image_id]
        return info["captchas"]


    def load_mask(self, image_id):
        """返回mask层和对应的类
        """
        info = self.image_info[image_id]
        mask = info['mask']
        class_ids = info['class_ids']
        return mask, class_ids


    def generate_mask(self, image, piece, bbox):
        width, height = image.size
        mask = np.zeros((height, width))
        data = np.array(piece)
        minimask = np.where(data <= 140, 1, 0)
        w_start, h_start, w_end, h_end = bbox
        mask[h_start:h_end, w_start:w_end] = minimask
        return mask


    def align_mask(self, masks):
        count = len(masks)
        rows, columns = masks[0].shape
        aligend_mask = np.zeros((rows, columns, count))
        for i in range(count):
            aligend_mask[:, :, i] = masks[i]
        return aligend_mask


    def random_select_pieces(self, piecepaths, num):
        random.shuffle(piecepaths)
        return piecepaths[:num]


    def generate_data(self, mode, piecepaths, size, bgcolor=255, savepath=None):
        """将几个片段拼成一个文件
        Args:
            mode: 图片的模式, RGB, L等
            piecepaths: 片段的路径
            size: 生成图片的大小
            savepath: 保存图片的路径，当提供时，保存图片
        """
        image = Image.new(mode, size, color=bgcolor)
        codes = [os.path.basename(path)[0] for path in piecepaths]
        pieces = [Image.open(path) for path in piecepaths]

        h_offset = random.randint(15, 20)
        v_offset = random.randint(5, 10)
        masks = []
        for c, p in zip(codes, pieces):
            w, h = p.size
            bbox = [h_offset, v_offset, w + h_offset, h + v_offset]
            image.paste(p, bbox)
            mask = self.generate_mask(image, p, bbox)
            masks.append(mask)
            h_offset = h_offset + w + random.randint(2, 8)

        aligned_masks = self.align_mask(masks)
        class_ids = np.array([self.charset.index(c) for c in codes], dtype=np.int32)
        return image.convert("RGB"), aligned_masks, class_ids


    def create_data(self, count, width, height):
        for i in range(count):
            piecepaths = glob.glob("crops/*")
            piecepaths = self.random_select_pieces(piecepaths, 3)
            image, mask, class_ids = self.generate_data("L", piecepaths, (width, height))
            self.add_image(source='captchas', image_id=i, image=image, mask=mask, class_ids=class_ids)
