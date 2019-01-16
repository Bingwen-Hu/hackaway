import face_recognition
from imutils import paths
import os
from os.path import basename
import numpy as np


DATA_DIR = './data/images'

infos = {
    'jiang': ['江纬','AILab','研究员','quick quick learn day day up'],
    'zhou': ['陈周','AILab','研究员','简单'],
    'kejian': ['李科健','AILab','研究员','哈哈哈'],
    'changshu': ['陈昌澍','AILab','研究员','和你一样'],
    'fengjiao': ['王凤娇','AILab','研究员','笑死我了'],
    'zhihao': ['曹志豪','AILab','研究员','别闹'],
    'zhan': ['陈站','AILab','研究员','不学习会死'],
    'jianlong': ['邓建龙','AILab','研究员','怎么又错了'],
    'junguang': ['冼俊光','AILab','研究员','好像没有'],
    'guangyi': ['袁广益','数据平台','Java开发','快睡觉'],
}

def build_facedb(dirpath:str):
    """ build a face database as following:
    Args:
        dirpath: known face database, assume subdir name is the name of person     

    Returns:
        key-value face database
    """
    db_embed = []
    db_infos = []

    image_paths = paths.list_images(dirpath)
    for imgpath in image_paths:
        img = face_recognition.load_image_file(imgpath)
        locations = face_recognition.face_locations(img)
        encodings = face_recognition.face_encodings(img, locations)
        try:
            db_embed.append(encodings[0])
        except:
            print('Could not find face!')
        else:
            db_infos.append(basename(imgpath).split('.')[0])
    return db_embed, db_infos

DB_EMBED, DB_LABELS = build_facedb(DATA_DIR)



def recognize(imgpath:str):
    img = face_recognition.load_image_file(imgpath)
    locations = face_recognition.face_locations(img)
    encodings = face_recognition.face_encodings(img, locations)
    def distance_helper(compared_encoding, bbox):
        # NOTE: location just use as return value
        distances = face_recognition.face_distance(DB_EMBED, compared_encoding)
        min_index = np.argmin(distances)
        min_distance = distances[min_index]
        # threshold
        label = DB_LABELS[min_index] if min_distance < 0.4 else 'unknown'
        return {'bbox': [bbox[3], bbox[0], bbox[1], bbox[2]], 'label': label}
    results = list(map(distance_helper, encodings, locations))
    return results

if __name__ == "__main__":
    pass    