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

    """
    img = cv2.imread(img)
    faces = model.get_input(img)
    return [(model.get_ga(face), bbox.tolist()) for (face, bbox) in faces]

if __name__ == '__main__':
    import sys
    rets = estimate(sys.argv[1])
    print(rets)