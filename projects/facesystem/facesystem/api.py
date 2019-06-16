"""api.py provide face detection and face recognition utilities"""
import os
import cv2
import pcn
import arcface
from facedb import Facedb


__all__ = ['face_detect', 'face_recognize', 'load_database_from_directory']


def face_detect(image_path):
    """detect a face from an image, 

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


def face_recognize(image_path, facedb:Facedb):
    """Recogniton face in the input image"""
    face = face_detect(image_path)
    emb = arcface.featurize(face)
    info = facedb.search(emb)
    return info

    
def load_database_from_directory(directory, facedb:Facedb, mode='append'):
    """load already face and informations from certain directory

    Args:
        directory: path of directory contains face images and information
        mode: `overwrite` mode to refresh the database with new directory content
            or `append` mode to append new data to existed ones
    """
    assert mode in ('overwrite', 'append'), "only 'overwrite' and 'append' mode support"
    info = os.path.join(directory, 'info.txt')
    images = os.path.join(directory, 'images')