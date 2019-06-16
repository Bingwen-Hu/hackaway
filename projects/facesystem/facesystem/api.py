"""api.py provide face detection and face recognition utilities"""
import cv2
import pcn
import arcface
from .db import Facedb

def detect(image_path):
    """detect a face from an image

    Args:
        image_path: path to the image
        
    Returns:
        A numpy-array-format face. Shape is (128x128x3) for RGB image
        and (128x128x1) for signal channels image.

    Raises:
        IOError: when more than one face is detected
    """
    image = cv2.imread(image_path)
    winlist = pcn.detect(image)
    crops = pcn.crop(image, winlist, size=128) # 128 is the input size of arcface
    if len(crops) != 1:
        raise IOError("two many face detected!")
    face = crops[0]
    return face


def featurize(face):
    """transform a face to embedding"""
    emb = arcface.featurize(face)
    return emb