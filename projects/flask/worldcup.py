# coding: utf-8
"""
python:  2.7.6
pymongo: 2.7.1
flask:   0.10.x

"""
from flask import Flask
from flask import request
import pymysql
import json

app = Flask(__name__)

SERVER_PARAMS = {
    'host': "localhost",
    'port': 27702,
    'user': 'like_jian',
    'password': 'worldcup2018',
    'database': 'worldcup',
    'charset': 'utf8'
}

server = pymysql.Connection(**SERVER_PARAMS)

@app.route('/predict/<gpid>')
def predict(gpid):
    if not server.open:
        server.connect()
    with server.cursor() as cursor:
        sqls = "select * from Predict where GPid=%s" % gpid
        cursor.execute(sqls)
        result = cursor.fetchone()
        sqldict = predict_parser(result)
    jsondata = json.dumps(sqldict)
    return jsondata


def predict_parser(result):
    return {
        'code': 0,
        'GPid': result[0],
        'host': result[1],
        'guest': result[2],
        'points': result[3] if result[3] else result[6],
        'rate': result[4] if result[4] else result[5],
    }

@app.route("/reply/<gpid>", methods=['POST', 'GET'])
def reply(gpid):
    if request.method == 'POST':
        parameters = request.form
        print(parameters)
        return json.dumps(parameters)
    elif request.method == 'GET':
        return json.dumps({"good": "Mory"})


if __name__ == '__main__':
    from werkzeug.contrib.fixers import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app)
    app.run(host='localhost', port=38016, debug=True)
