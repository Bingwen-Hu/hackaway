"""database.py implement face database function
"""
import json
import numpy as np


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

    def insert(self, im_emb, jsoninfo):
        self.emb.append(im_emb)
        self.info.append(jsoninfo)
        assert len(self.emb) == len(self.info), "inconsistent happen!"
        return True
    
    def search(self, im_emb):
        """
        Returns:
            dict
        """
        if len(self.emb) == 0:
            return {}
        distances = list(map(lambda emb: cosin_metric(emb, im_emb), self.emb))
        max_i = np.argmax(distances)
        distance = distances[max_i] 
        if distance > 0.4: # threshold for arcface
            return self.info[max_i]
        else:
            return {}