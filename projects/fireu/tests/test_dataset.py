import sys
import os
import os.path as osp
import unittest

# enable import deepfree
sys.path.insert(0, os.path.abspath('..'))
from deepfree.datasets.coco import reader



class TestDatasets(unittest.TestCase):
    def setUp(self):
        COCO_DIR = "/data/minicoco"
        images_directory = osp.join(COCO_DIR, 'images')
        annotations_file = osp.join(COCO_DIR, "annotations/mini_person_keypoints_val2014.json")
        self.reader = reader.COCO(images_directory, annotations_file)
    
    def test_make_generator(self):
        data_generator = self.reader.make_generator(batch_size=10)
        im_metas, ann_metas = next(data_generator)
        self.assertEqual(len(im_metas), 10)
        self.assertEqual(len(ann_metas), 10)
        self.assertIsNotNone(im_metas[0])
        

if __name__ == '__main__':
    unittest.main()