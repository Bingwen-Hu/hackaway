# coding: utf-8
"""
python:  3.6
flask:   1.0.2

"""
from flask import Flask
from flask import request, jsonify, render_template, redirect


app = Flask(__name__)

@app.route('/slider', methods=["POST", "GET"])
def distance():
    if request.method == "POST":
        org = request.files['org']
        new = request.files['new']
        print(org.filename, new.filename)
        org.save("org.png")
        new.save("new.png")
        dist = 1
        return jsonify({'distance': dist})
    return render_template('slider.html')



if __name__ == '__main__':
    app.run(host='localhost', port=38016, debug=False)
