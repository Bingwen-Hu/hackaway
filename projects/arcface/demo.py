import cv2
import pcn
import arcface




img = cv2.imread(path)
winlist = pcn.detect(img)
crops = pcn.crop(img, winlist, 128)
c = crops[0]


f = arcface.featurize(c)
print(f.shape)