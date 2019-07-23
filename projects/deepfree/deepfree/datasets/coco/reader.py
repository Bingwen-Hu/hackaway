import os.path as osp

import numpy as np
import pycocotools.coco as coco


class COCO(object):

    def __init__(self, images_directory, annotation_file):
        self.coco = coco.COCO(annotation_file)
        self.im_dir = images_directory
        self.im_ids = np.array((self.coco.imgs.keys()))
        self.size = len(self.im_ids)
    
    def fetch(self, batch_size, random=False):
        if random:
            self.im_ids = np.random.shuffle(self.im_ids)
        
        iters = self.size / batch_size + 1 # number of iteration
        for i in range(iters):
            start, end = i*batch_size, min((i+1)*batch_size, self.size)
            im_ids = self.im_ids[start:end]
            ann_ids = [self.coco.getAnnIds(im_id) for im_id in im_ids]
            im_metas = [self.coco.loadImgs(im_id) for im_id in im_ids]
            ann_metas = [self.coco.loadAnns(ann_id) for ann_id in ann_ids]
            yield (im_metas, ann_metas)

    def __len__(self):
        return self.size

        