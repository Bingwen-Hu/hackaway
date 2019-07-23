import sys
import os.path as osp

sys.path.insert(0, os.path.join(os.path.abspath('..'), 'source_test'))



from deepfree.datasets.coco import reader


COCO_DIR = "/media/data/urun_tandong_video/data/COCO/"
images_directory = osp.join(COCO_DIR, 'images/train2014/')
annotations_file = osp.join(COCO_DIR, "annotations/person_keypoints_train2014.json")


coco = reader.COCO(images_directory, annotations_file)
data_generator = coco.fetch(batch_size=10)
im_metas, ann_metas = next(data_generator)
assert len(im_metas) == 10, "batch_size not match!"
assert im_metas[0] is not None, "return None!"