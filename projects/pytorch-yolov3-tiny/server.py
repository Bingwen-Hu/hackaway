# coding: utf-8
"""
python:  3.6
flask:   1.0.2

"""
from flask import Flask
from flask import request, jsonify, render_template, redirect

from locater import predict


import os
from uuid import uuid1

app = Flask(__name__)

@app.route('/locate', methods=["POST", "GET"])
def locate():
    if request.method == "POST":
        img = request.files['img']
        name = f'{uuid1()}.jpg'
        img.save(name)
        coords = predict(name)
        os.remove(name)
        return jsonify(coords)
    return render_template('locate.html')



if __name__ == '__main__':
    app.run(host='localhost', port=38016, debug=True)
