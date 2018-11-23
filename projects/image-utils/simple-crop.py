import os
import cv2
from uuid import uuid1
from imutils import paths

crops_xy = [
    [0, 37], [37, 74], [74, 111], [111, 150],
    [0, 74], [74, 150], [0, 111],  [37, 150],
]


def crop(imgpath):
    """separate a image (150x60) into 
    four pieces"""
    img = cv2.imread(imgpath)
    widths = [0, 37, 150]
    for (s, e) in crops_xy:
        crop = img[:, s:e]
        cv2.imwrite("sub/{}.png".format(uuid1()), crop)



if __name__ == "__main__":
    images = paths.list_images('unuse')
    for i, f in enumerate(images):
        crop(f)
        os.rename(f, "finish/{}".format(os.path.basename(f)))
