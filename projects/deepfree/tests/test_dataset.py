import sys
import os
import os.path as osp
# add into PYTHONPATH
sys.path.insert(0, os.path.join(os.path.abspath('..'), 'source_test'))


from deepfree.datasets.coco import reader

import unittest


class TestDatasets(unittest.TestCase):
    def setUp(self):
        COCO_DIR = "/media/data/urun_tandong_video/data/COCO/"
        images_directory = osp.join(COCO_DIR, 'images/train2014/')
        annotations_file = osp.join(COCO_DIR, "annotations/person_keypoints_train2014.json")
        self.reader = reader.COCO(images_directory, annotations_file)
    
    def test_fetch(self):
        data_generator = self.reader.fetch(batch_size=10)
        im_metas, ann_metas = next(data_generator)
        self.assertEquals(len(im_metas), 10)
        self.assertEquals(len(ann_metas), 10)
        self.assertIsNotNone(im_metas[0])
        