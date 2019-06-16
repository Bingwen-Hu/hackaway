"""database.py implement face database function
"""
import json
import numpy as np
from uuid import uuid1


def generate_id():
    return uuid1().hex[:8]


def cosin_metric(f1, f2):
    return np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))


class Facedb(object):
    """this class only handle vector represented faces, supply function like
    :insert :update :delete :search on a vector level"""

    def __init__(self):
        self.emb = []  # store face embedding
        self.info = [] # store face information
    
    def __len__(self):
        return len(self.info)

    def update(self, id):
        pass

    def insert(self, im_emb, jsoninfo):
        self.emb.append(im_emb)
        jsoninfo = json.loads(jsoninfo)
        jsoninfo["id"] = generate_id()
        self.info.append(jsoninfo)
        assert len(self.emb) == len(self.info), "nonInsistent happen!"
        return True
    
    def search(self, im_emb):
        distances = list(map(lambda emb: cosin_metric(emb, im_emb), self.emb))
        min_i = np.argmin(distances)
        distance = distances[min_i] 
        if distance > 0.2557: # threshold for arcface
            return ""
        else:
            return self.info[min_i]
    
    def delete(self, id):
        pass

    def backup(self):
        pass