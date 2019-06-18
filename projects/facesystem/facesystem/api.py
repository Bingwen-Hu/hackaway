"""api.py provide face detection and face recognition utilities"""
import os
import json
import cv2
import pcn
import arcface
import pickle
from uuid import uuid1
from imutils import paths
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


def face_recognize(image_path):
    """Recogniton face in the input image"""
    global facedb
    face = face_detect(image_path)
    if face is None:
        return {}
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


def load_database_from_backup(backup, facedb:Facedb):
    """load already face and informations from certain backup

    Args:
        backup: path of backup directory
        facedb: Instance of Facedb
    """
    faces = list(paths.list_images(f'{backup}/face'))
    if len(faces) == 0:
        return facedb
    info = pickle.load(open(f'{backup}/info/backup.pkl', 'rb')) # NOTE: weak design
    facedb.info = info
    for record in info:
        filename = record['id']
        facedb.emb.append(arcface.featurize(f"{backup}/face/{filename}.jpg"))
    return facedb


def facedb_backup(backup):
    global facedb
    with open(f"facesystem_data/info/{backup}.pkl", 'wb') as f:
        pickle.dump(facedb.info, f)


def face_register(image_path, jsoninfo, duplicate=True):
    """register a face into face system, namely facedb

    Args:
        image_path: path of input image 
        jsoninfo: inforamtion in JSON key-value format or python dict, for example:
            {'name': 'xxx', 'QQ': 23241432}
        duplicate: if face already exists, stop register
    Returns:
        a dictionary contain operation state and message, for example:
            {'state': 10000, 'message': 'succeed'}
            {'state': 10010, 'message': 'already exists'}
    """
    global facedb
    face = face_detect(image_path)
    if face is not None:
        emb = arcface.featurize(face)
        if duplicate:
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
        facedb_backup("backup") # TODO: fix overwrite problem
        return {'state': 10000, "message": "succeed"}
    # register failed
    return {"state": 10011, "message": "no face detected"}


# restore facedb
facedb = Facedb()
load_database_from_backup(data, facedb)


__all__ = ['face_detect', 'face_recognize', 'face_register']