import sys
import os.path as osp
import cv2


path = sys.argv[1]
basename = osp.basename(path)
size = int(sys.argv[2])
im = cv2.imread(path)
im = cv2.resize(im, None, fx=size, fy=size, interpolation=cv2.INTER_CUBIC)
newname = "r" + basename.replace('jpg', 'png')
cv2.imwrite(newname, im)
