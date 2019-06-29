from . import face_model
import cv2
import numpy as np

# general
class Args():
    image_size = '112,112'
    model = 'model/model,0'
    det = 0
args = Args()

model = face_model.FaceModel(args)

def estimate(img):
    """
    Args:
        img: path of image
    Returns:
        empty list or [(gender, age), bbox]
    """
    if type(img) == str:
        img = cv2.imread(img)
    faces = model.get_input(img)
    if not faces:
        return []
    return [(model.get_ga(face), bbox.tolist()) for (face, bbox) in faces]


def show(img):
    if type(img) == str:
        img = cv2.imread(img)
    retlist = estimate(img)
    for ret in retlist:
        bbox = ret[1] 
        x1 = int(max(0, bbox[0]))
        y1 = int(max(0, bbox[1]))
        x2 = int(min(img.shape[1], bbox[2]))
        y2 = int(min(img.shape[0], bbox[3]))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("gender and age", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img