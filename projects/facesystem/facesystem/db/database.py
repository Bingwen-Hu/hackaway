"""db.py implement face database function
"""

import numpy as np


class Facedb(object):
    """this class only handle vector represented faces, supply function like
    :insert :update :delete :search on a vector level"""

    def __init__(self):
        self.db = None  # store face embedding
        self.info = None # store face information
    
    def load_directory(self, directory):
        pass

    def update(self, id):
        pass

    def insert(self, im_emb, **kwargs):
        return id
    
    def search(self, im_emb):
        pass