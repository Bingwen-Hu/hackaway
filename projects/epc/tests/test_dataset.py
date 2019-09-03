import sys
import os
import os.path as osp
import unittest

from fireu.data import coco



class TestDatasets(unittest.TestCase):
    def setUp(self):
        COCO_DIR = "/data/minicoco"
        images_directory = osp.join(COCO_DIR, 'images')
        annotations_file = osp.join(COCO_DIR, "annotations/mini_person_keypoints_val2014.json")
        self.coco = coco.COCO(images_directory, annotations_file)
    
    def test_get_im_path(self):
        path = self.coco.get_im_path(index=1)
        self.assertIn("val2014", path)
        

if __name__ == '__main__':
    unittest.main()