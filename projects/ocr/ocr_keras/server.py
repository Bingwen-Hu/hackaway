# coding: utf-8
"""
python:  3.6
flask:   1.0.2

"""
from flask import Flask
from flask import request, jsonify, render_template, redirect
from PIL import Image
from predict import predict_interface


import os
from uuid import uuid1
import requests

app = Flask(__name__)



def process_output(coords, name):
    img = Image.open(name)
    chars = [img.crop(xy) for xy in coords]
    
    texts = [predict_interface(char) for char in chars]
    results = []
    for text, coord in zip(texts, coords):
        key = ''.join(text)
        center = (coord[2] + coord[0]) / 2, (coord[3] + coord[1]) / 2
        results.append([key, center])
    return results

@app.route('/netease', methods=["POST", "GET"])
def locate():
    if request.method == "POST":
        img = request.files['img']
        ustr = uuid1()
        name = f'/home/barfoo/mory/netease/yolo/temp/{ustr}.jpg'
        img.save(name)
        coords = requests.get('http://localhost:38016/local/{}'.format(ustr))
        outputs = process_output(coords.json(), name)
        os.remove(name)
        return jsonify(outputs)
    return render_template('netease.html')


if __name__ == '__main__':
    from werkzeug.contrib.fixers import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app)
    app.run(host='localhost', port=8016, debug=False)
