import pcn
import os
import landmark
import cv2

from uuid import uuid1


def cropface(path):
    os.makedirs("crops", exist_ok=True)
    os.makedirs('errors', exist_ok=True)

    img = cv2.imread(path)
    if img is None:
        os.rename(path, f"errors/{os.path.basename(path)}")
        return 
    height, width = img.shape[:2]
    

    margin = 5
    res_lm = landmark.detect(path)
    for det in res_lm:
        x1, y1, x2, y2 = det['bbox']
        x1 = max(0, x1-margin)
        y1 = max(0, y1-margin)
        x2 = min(width, x2+margin)
        y2 = min(height, y2+margin)
        face = img[y1:y2, x1:x2, :]
        cv2.imwrite(f"crops/{uuid1()}.jpg", face)

    winlist = pcn.detect(img)
    crops = pcn.crop(img, winlist)
    for face in crops:
        cv2.imwrite(f"crops/{uuid1()}.jpg", face)

if __name__ == '__main__':
    # root = '/home/mory/hackaway/projects/aio/dataset/'
    # dirs = ['anger', 'disgust', 'fear', 'neutral', 'sadness', 'smile', 'surprise']
    # for dir in dirs:
    #     source = os.path.join(root, dir)
    #     os.chdir(source)
    #     files = os.listdir()
    #     for file in files:
    #         path = os.path.join(source, file)
    #         cropface(path)
    cropface('t.jpg')