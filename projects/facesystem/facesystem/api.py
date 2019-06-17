"""api.py provide face detection and face recognition utilities"""
import os
import json
import cv2
import pcn
import arcface
from uuid import uuid1
from .facedb import Facedb







# create directory to save face and information
data = 'facesystem_data'
data_face = f"{data}/face"
data_info = f"{data}/info"
os.makedirs(data, exist_ok=True)
os.makedirs(data_face, exist_ok=True)
os.makedirs(data_info, exist_ok=True)


def generate_id():
    return uuid1().hex[:8]

def face_detect(image_path):
    """detect a face from an image, 

    Args:
        image_path: path to the image
        
    Returns:
        If exactly one face is found, return a numpy-array-format face. Shape is (128x128x3) 
        for RGB image and (128x128x1) for signal channels image. Else return None or the first
        found face.
    """
    image = cv2.imread(image_path)
    winlist = pcn.detect(image)
    crops = pcn.crop(image, winlist, size=128) # 128 is the input size of arcface
    return crops[0] if crops else None


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
    pass


def load_database_from_backup(backup, facedb:Facedb, mode='overwrite'):
    """load already face and informations from certain backup

    Args:
        backup: path of backup directory
        mode: `overwrite` mode to refresh the database with new directory content
            or `append` mode to append new data to existed ones
    """
    assert mode in ('overwrite', 'append'), "only 'overwrite' and 'append' mode support"
    pass
 

def face_register(image_path, jsoninfo, facedb:Facedb):
    """register a face into face system, namely facedb

    Args:
        image_path: path of input image 
        jsoninfo: inforamtion in JSON key-value format or python dict, for example:
            {'name': 'xxx', 'QQ': 23241432}
        facedb: instance of Facedb
    Returns:
        a dictionary contain operation state and message, for example:
            {'state': 10000, 'message': 'succeed'}
            {'state': 10010, 'message': 'already exists'}
    """
    face = face_detect(image_path)
    if face:
        emb = arcface.featurize(face)
        info = facedb.search(emb)
        if info: # already exist
            return {"state": 10010, "message": "already exists"}
        # insert into facedb and save it
        if type(jsoninfo) == str:
            jsoninfo = json.loads(jsoninfo)
        faceid = generate_id()
        jsoninfo['id'] = faceid
        facedb.insert(emb, jsoninfo)
        cv2.imwrite(f"{data_face}/{faceid}.jpg", face)
        return {'state': 10000, "message": "succeed"}
    # register failed
    return {"state": 10011, "message": "no face detected"}


__all__ = ['face_detect', 'face_recognize', 'face_register']