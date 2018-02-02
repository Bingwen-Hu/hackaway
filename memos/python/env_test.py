# -*- coding: utf-8 -*-
""" os.environ test

@author: Mory
"""

import os

HOST = os.environ.get("HOST")
USER = os.environ.get('USER')
PASSWD = os.environ.get('PASSWD')
PORT = os.environ.get('PORT')

print(HOST, USER, PASSWD, PORT)
print("type of port is ", type(PORT))