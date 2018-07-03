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
        org.save("org.png")
        new.save("new.png")

        dist = detect()
        return jsonify({'distance': dist})
    return render_template('slider.html')

def detect():
    from PIL import Image
    import numpy as np
    from skimage.morphology import erosion
    org = Image.open('org.png').convert('L')
    new = Image.open('new.png').convert('L')
    org_np = np.array(org)
    new_np = np.array(new)
    diff = np.abs(new_np - org_np)
    ero = erosion(diff)
    ero = np.where(ero > 210, 0, ero)
    img = Image.fromarray(ero)
    img.save('diff.png')

    data = ero
    x, y = data.shape
    print("%d rows, %d columns" % (x, y))
    vsums = np.sum(data, axis=0)
    result = []
    for i, s in enumerate(vsums):
        if s > 500:
            if i+1 == y:
                result.append(i)
            elif vsums[i+1] < 500:
                result.append(i)

    if len(result) == 2:
        return result[1] - result[0] + 5
    else:
        return result[0]/2


if __name__ == '__main__':
    app.run(host='localhost', port=38016, debug=False)
