import os
import cv2
from pcn import models
from pcn import pcn
from pcn import utils

if __name__ == '__main__':
    # usage settings
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 pcn.py path/to/img")
        sys.exit()
    else:
        imgpath = sys.argv[1]
    # network detection
    nets = models.load_model()
    img = cv2.imread(imgpath)
    faces = pcn.pcn_detect(img, nets)
    # draw image
    for face in faces:
        face_ = utils.crop_face(img, face)
    # show image
    cv2.imshow("pytorch-PCN", face_)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # save image
    name = os.path.basename(imgpath)
    cv2.imwrite('result/ret_{}'.format(name), img)